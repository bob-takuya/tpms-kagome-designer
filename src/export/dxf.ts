/**
 * dxf.ts – DXF export optimized for laser-cutting + OpenNest nesting workflow
 *
 * Layer structure:
 *   CUT_A / CUT_B / CUT_C  → closed strip outlines per family  (laser: through-cut)
 *   HOLE                    → junction hole circles              (laser: through-cut / drill)
 *   SCORE                   → fold / centerlines                 (laser: score / engrave)
 *   LABEL                   → text annotations                   (laser: skip or light mark)
 *
 * Each strip is emitted as a DXF BLOCK so that OpenNest (Rhino / Grasshopper)
 * can treat outline + holes + fold lines as a single nestable unit.
 * After nesting, "Explode" the blocks to get flat geometry per layer for the
 * laser-cutter driver.
 *
 * Format: AutoCAD R2000 (AC1015) with entity handles and INSUNITS = mm.
 */

import * as THREE from 'three';
import type { UnfoldedStrip } from '../core/unfold';

// ─────────────────────────────────────────────────────────────────────────────
// Layer definitions
// ─────────────────────────────────────────────────────────────────────────────

export interface DXFLayer {
  name: string;
  color: number; // AutoCAD Color Index (ACI)
}

export const DXF_LAYERS: Record<string, DXFLayer> = {
  CUT_A: { name: 'CUT_A', color: 1 }, // Red    – Family 0 outlines
  CUT_B: { name: 'CUT_B', color: 2 }, // Yellow – Family 1 outlines
  CUT_C: { name: 'CUT_C', color: 3 }, // Green  – Family 2 outlines
  HOLE:  { name: 'HOLE',  color: 4 }, // Cyan   – Junction holes
  SCORE: { name: 'SCORE', color: 6 }, // Magenta – Fold / centerlines
  LABEL: { name: 'LABEL', color: 7 }, // White  – Strip IDs & hole IDs
};

const CUT_LAYERS = ['CUT_A', 'CUT_B', 'CUT_C'];

// ─────────────────────────────────────────────────────────────────────────────
// Handle counter (reset per export call)
// ─────────────────────────────────────────────────────────────────────────────

let _handle = 0x100;
function nextHandle(): string {
  return (_handle++).toString(16).toUpperCase();
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

export function generateDXF(
  strips: UnfoldedStrip[],
  includeHoleIds: boolean,
  includeFoldLines: boolean,
): string {
  _handle = 0x100;
  const L: string[] = [];

  // ── HEADER ────────────────────────────────────────────────────────────────
  dxf(L, 0, 'SECTION');
  dxf(L, 2, 'HEADER');
  dxf(L, 9, '$ACADVER');  dxf(L, 1, 'AC1015');   // AutoCAD R2000
  dxf(L, 9, '$INSUNITS'); dxf(L, 70, '4');        // 4 = millimeters
  dxf(L, 9, '$HANDSEED'); dxf(L, 5, 'FFFF');
  dxf(L, 0, 'ENDSEC');

  // ── TABLES ────────────────────────────────────────────────────────────────
  dxf(L, 0, 'SECTION');
  dxf(L, 2, 'TABLES');
  writeLtypeTable(L);
  writeLayerTable(L);
  dxf(L, 0, 'ENDSEC');

  // ── BLOCKS ────────────────────────────────────────────────────────────────
  dxf(L, 0, 'SECTION');
  dxf(L, 2, 'BLOCKS');

  // Required built-in blocks for R2000
  writeBuiltinBlock(L, '*MODEL_SPACE');
  writeBuiltinBlock(L, '*PAPER_SPACE');

  // One block per strip (outline + holes + fold + label grouped together)
  for (const strip of strips) {
    writeStripBlock(L, strip, includeHoleIds, includeFoldLines);
  }

  dxf(L, 0, 'ENDSEC');

  // ── ENTITIES (one INSERT per strip block) ─────────────────────────────────
  dxf(L, 0, 'SECTION');
  dxf(L, 2, 'ENTITIES');

  for (const strip of strips) {
    dxf(L, 0, 'INSERT');
    dxf(L, 5, nextHandle());
    dxf(L, 8, '0');
    dxf(L, 2, blockNameFor(strip));
    dxf(L, 10, fmt(strip.boundingBox.minX));
    dxf(L, 20, fmt(strip.boundingBox.minY));
    dxf(L, 30, '0');
  }

  dxf(L, 0, 'ENDSEC');
  dxf(L, 0, 'EOF');

  return L.join('\n');
}

// ─────────────────────────────────────────────────────────────────────────────
// Junction CSV (unchanged)
// ─────────────────────────────────────────────────────────────────────────────

export function generateJunctionCSV(strips: UnfoldedStrip[]): string {
  const rows: string[] = ['junction_id,strip_id,x,y,radius'];
  for (const strip of strips) {
    for (const seg of strip.segments) {
      for (const hole of seg.holes) {
        rows.push(
          `${hole.junctionId},${strip.stripId},${hole.center.x.toFixed(4)},${hole.center.y.toFixed(4)},${hole.radius.toFixed(4)}`,
        );
      }
    }
  }
  return rows.join('\n');
}

// ─────────────────────────────────────────────────────────────────────────────
// Block writers
// ─────────────────────────────────────────────────────────────────────────────

function blockNameFor(strip: UnfoldedStrip): string {
  return `STRIP_${strip.stripId.replace(/[^A-Za-z0-9_]/g, '_').toUpperCase()}`;
}

function writeBuiltinBlock(L: string[], name: string): void {
  dxf(L, 0, 'BLOCK');
  dxf(L, 5, nextHandle());
  dxf(L, 8, '0');
  dxf(L, 2, name);
  dxf(L, 70, '0');
  dxf(L, 10, '0'); dxf(L, 20, '0'); dxf(L, 30, '0');
  dxf(L, 0, 'ENDBLK');
  dxf(L, 5, nextHandle());
  dxf(L, 8, '0');
}

/**
 * Write one BLOCK containing all geometry for a single strip.
 *
 * Block base point = (0, 0).  All coordinates are relative to the strip's
 * bounding-box origin (minX, minY) so that the INSERT position places the
 * block at its correct layout location.
 */
function writeStripBlock(
  L: string[],
  strip: UnfoldedStrip,
  includeHoleIds: boolean,
  includeFoldLines: boolean,
): void {
  const name     = blockNameFor(strip);
  const cutLayer = CUT_LAYERS[strip.family] ?? 'CUT_A';
  const ox       = strip.boundingBox.minX;
  const oy       = strip.boundingBox.minY;

  // Block header
  dxf(L, 0, 'BLOCK');
  dxf(L, 5, nextHandle());
  dxf(L, 8, '0');
  dxf(L, 2, name);
  dxf(L, 70, '0');
  dxf(L, 10, '0'); dxf(L, 20, '0'); dxf(L, 30, '0');

  for (const seg of strip.segments) {
    // ── Closed outline (single LWPOLYLINE for OpenNest) ───────────────────
    // Path: left[0] → left[N-1] → right[N-1] → right[0] → close
    // The closed flag auto-connects right[0] back to left[0] (start cap).
    if (seg.leftBoundary.length > 0 && seg.rightBoundary.length > 0) {
      const outline = [
        ...seg.leftBoundary,
        ...seg.rightBoundary.slice().reverse(),
      ].map(p => new THREE.Vector2(p.x - ox, p.y - oy));
      addClosedPolyline(L, outline, cutLayer);
    }

    // ── Fold / score line (centerline) ────────────────────────────────────
    if (includeFoldLines && seg.centerline.length > 1) {
      const cl = seg.centerline.map(p => new THREE.Vector2(p.x - ox, p.y - oy));
      addPolyline(L, cl, 'SCORE');
    }

    // ── Junction holes ────────────────────────────────────────────────────
    for (const hole of seg.holes) {
      const c = new THREE.Vector2(hole.center.x - ox, hole.center.y - oy);
      addCircle(L, c, hole.radius, 'HOLE');

      if (includeHoleIds) {
        addText(L, c, String(hole.junctionId), 'LABEL', hole.radius * 0.8);
      }
    }
  }

  // ── Strip label ─────────────────────────────────────────────────────────
  if (strip.segments.length > 0 && strip.segments[0].centerline.length > 0) {
    const pt = strip.segments[0].centerline[0];
    const pos = new THREE.Vector2(
      pt.x - ox,
      pt.y - oy + strip.segments[0].width * 0.5,
    );
    addText(L, pos, strip.stripId, 'LABEL', strip.segments[0].width * 0.3);
  }

  // Block footer
  dxf(L, 0, 'ENDBLK');
  dxf(L, 5, nextHandle());
  dxf(L, 8, '0');
}

// ─────────────────────────────────────────────────────────────────────────────
// Table writers
// ─────────────────────────────────────────────────────────────────────────────

function writeLtypeTable(L: string[]): void {
  dxf(L, 0, 'TABLE');
  dxf(L, 2, 'LTYPE');
  dxf(L, 5, nextHandle());
  dxf(L, 70, '2');

  // CONTINUOUS
  dxf(L, 0, 'LTYPE'); dxf(L, 5, nextHandle());
  dxf(L, 2, 'CONTINUOUS'); dxf(L, 70, '0');
  dxf(L, 3, 'Solid line');
  dxf(L, 72, '65'); dxf(L, 73, '0'); dxf(L, 40, '0.0');

  // CENTER (dash-dot for fold / score lines)
  dxf(L, 0, 'LTYPE'); dxf(L, 5, nextHandle());
  dxf(L, 2, 'CENTER'); dxf(L, 70, '0');
  dxf(L, 3, 'Center ____ _ ____');
  dxf(L, 72, '65'); dxf(L, 73, '4'); dxf(L, 40, '2.0');
  dxf(L, 49, '1.25'); dxf(L, 49, '-0.25');
  dxf(L, 49, '0.25'); dxf(L, 49, '-0.25');

  dxf(L, 0, 'ENDTAB');
}

function writeLayerTable(L: string[]): void {
  dxf(L, 0, 'TABLE');
  dxf(L, 2, 'LAYER');
  dxf(L, 5, nextHandle());
  dxf(L, 70, String(Object.keys(DXF_LAYERS).length));

  for (const layer of Object.values(DXF_LAYERS)) {
    dxf(L, 0, 'LAYER'); dxf(L, 5, nextHandle());
    dxf(L, 2, layer.name);
    dxf(L, 70, '0');
    dxf(L, 62, String(layer.color));
    // SCORE layer uses CENTER linetype (dash-dot), others use solid
    dxf(L, 6, layer.name === 'SCORE' ? 'CENTER' : 'CONTINUOUS');
  }

  dxf(L, 0, 'ENDTAB');
}

// ─────────────────────────────────────────────────────────────────────────────
// Entity writers
// ─────────────────────────────────────────────────────────────────────────────

/** Closed LWPOLYLINE – the nestable outline for OpenNest */
function addClosedPolyline(L: string[], pts: THREE.Vector2[], layer: string): void {
  dxf(L, 0, 'LWPOLYLINE'); dxf(L, 5, nextHandle());
  dxf(L, 8, layer);
  dxf(L, 90, String(pts.length));
  dxf(L, 70, '1'); // 1 = closed polyline
  for (const p of pts) {
    dxf(L, 10, fmt(p.x));
    dxf(L, 20, fmt(p.y));
  }
}

/** Open LWPOLYLINE – for fold / score lines */
function addPolyline(L: string[], pts: THREE.Vector2[], layer: string): void {
  dxf(L, 0, 'LWPOLYLINE'); dxf(L, 5, nextHandle());
  dxf(L, 8, layer);
  dxf(L, 90, String(pts.length));
  dxf(L, 70, '0'); // 0 = open polyline
  for (const p of pts) {
    dxf(L, 10, fmt(p.x));
    dxf(L, 20, fmt(p.y));
  }
}

function addCircle(L: string[], center: THREE.Vector2, radius: number, layer: string): void {
  dxf(L, 0, 'CIRCLE'); dxf(L, 5, nextHandle());
  dxf(L, 8, layer);
  dxf(L, 10, fmt(center.x));
  dxf(L, 20, fmt(center.y));
  dxf(L, 40, fmt(radius));
}

function addText(
  L: string[],
  pos: THREE.Vector2,
  text: string,
  layer: string,
  height: number,
): void {
  dxf(L, 0, 'TEXT'); dxf(L, 5, nextHandle());
  dxf(L, 8, layer);
  dxf(L, 10, fmt(pos.x));
  dxf(L, 20, fmt(pos.y));
  dxf(L, 40, fmt(height));
  dxf(L, 1, text);
  dxf(L, 72, '1'); // center horizontal alignment
  dxf(L, 11, fmt(pos.x));
  dxf(L, 21, fmt(pos.y));
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Push one DXF group-code / value pair */
function dxf(L: string[], code: number, value: string): void {
  L.push(String(code), value);
}

/** Format a number to 4 decimal places (0.1 µm precision – plenty for laser) */
function fmt(n: number): string {
  return n.toFixed(4);
}

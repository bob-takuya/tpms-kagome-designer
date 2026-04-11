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
 * Format: AutoCAD R12 (AC1009) – maximum compatibility with all CAD readers.
 * Uses POLYLINE+VERTEX+SEQEND (R12 closed polylines) instead of LWPOLYLINE.
 * All coordinates are in millimeters.
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
// Public API
// ─────────────────────────────────────────────────────────────────────────────

export function generateDXF(
  strips: UnfoldedStrip[],
  includeHoleIds: boolean,
  includeFoldLines: boolean,
): string {
  const L: string[] = [];

  // ── HEADER ────────────────────────────────────────────────────────────────
  w(L, 0, 'SECTION');
  w(L, 2, 'HEADER');
  w(L, 9, '$ACADVER');
  w(L, 1, 'AC1009');
  w(L, 9, '$INSBASE');
  w(L, 10, '0'); w(L, 20, '0'); w(L, 30, '0');
  w(L, 9, '$EXTMIN');
  w(L, 10, '0'); w(L, 20, '0'); w(L, 30, '0');
  w(L, 9, '$EXTMAX');
  w(L, 10, '1000'); w(L, 20, '1000'); w(L, 30, '0');
  w(L, 0, 'ENDSEC');

  // ── TABLES ────────────────────────────────────────────────────────────────
  w(L, 0, 'SECTION');
  w(L, 2, 'TABLES');
  writeLtypeTable(L);
  writeLayerTable(L);
  w(L, 0, 'ENDSEC');

  // ── BLOCKS ────────────────────────────────────────────────────────────────
  w(L, 0, 'SECTION');
  w(L, 2, 'BLOCKS');

  for (const strip of strips) {
    writeStripBlock(L, strip, includeHoleIds, includeFoldLines);
  }

  w(L, 0, 'ENDSEC');

  // ── ENTITIES (one INSERT per strip block) ─────────────────────────────────
  w(L, 0, 'SECTION');
  w(L, 2, 'ENTITIES');

  for (const strip of strips) {
    w(L, 0, 'INSERT');
    w(L, 8, '0');
    w(L, 2, blockNameFor(strip));
    w(L, 10, fmt(strip.boundingBox.minX));
    w(L, 20, fmt(strip.boundingBox.minY));
    w(L, 30, '0');
  }

  w(L, 0, 'ENDSEC');
  w(L, 0, 'EOF');

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
  w(L, 0, 'BLOCK');
  w(L, 8, '0');
  w(L, 2, name);
  w(L, 70, '0');
  w(L, 10, '0'); w(L, 20, '0'); w(L, 30, '0');

  for (const seg of strip.segments) {
    // ── Closed outline (single closed POLYLINE for OpenNest) ──────────────
    // Path: left[0] → left[N-1] → right[N-1] → right[0] → close
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
      addOpenPolyline(L, cl, 'SCORE');
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
  w(L, 0, 'ENDBLK');
  w(L, 8, '0');
}

// ─────────────────────────────────────────────────────────────────────────────
// Table writers (R12 – only LTYPE and LAYER required)
// ─────────────────────────────────────────────────────────────────────────────

function writeLtypeTable(L: string[]): void {
  w(L, 0, 'TABLE');
  w(L, 2, 'LTYPE');
  w(L, 70, '2');

  // CONTINUOUS
  w(L, 0, 'LTYPE');
  w(L, 2, 'CONTINUOUS');
  w(L, 70, '0');
  w(L, 3, 'Solid line');
  w(L, 72, '65'); w(L, 73, '0'); w(L, 40, '0.0');

  // CENTER (dash-dot for fold / score lines)
  w(L, 0, 'LTYPE');
  w(L, 2, 'CENTER');
  w(L, 70, '0');
  w(L, 3, 'Center ____ _ ____');
  w(L, 72, '65'); w(L, 73, '4'); w(L, 40, '2.0');
  w(L, 49, '1.25'); w(L, 49, '-0.25');
  w(L, 49, '0.25'); w(L, 49, '-0.25');

  w(L, 0, 'ENDTAB');
}

function writeLayerTable(L: string[]): void {
  const allLayers = [
    { name: '0', color: 7, ltype: 'CONTINUOUS' },
    ...Object.values(DXF_LAYERS).map(l => ({
      name: l.name,
      color: l.color,
      ltype: l.name === 'SCORE' ? 'CENTER' : 'CONTINUOUS',
    })),
  ];

  w(L, 0, 'TABLE');
  w(L, 2, 'LAYER');
  w(L, 70, String(allLayers.length));

  for (const layer of allLayers) {
    w(L, 0, 'LAYER');
    w(L, 2, layer.name);
    w(L, 70, '0');
    w(L, 62, String(layer.color));
    w(L, 6, layer.ltype);
  }

  w(L, 0, 'ENDTAB');
}

// ─────────────────────────────────────────────────────────────────────────────
// Entity writers (R12 POLYLINE + VERTEX + SEQEND)
// ─────────────────────────────────────────────────────────────────────────────

/** Closed polyline (R12 POLYLINE with flag 1) – the nestable outline for OpenNest */
function addClosedPolyline(L: string[], pts: THREE.Vector2[], layer: string): void {
  w(L, 0, 'POLYLINE');
  w(L, 8, layer);
  w(L, 66, '1');   // vertices follow
  w(L, 70, '1');   // 1 = closed polyline
  for (const p of pts) {
    w(L, 0, 'VERTEX');
    w(L, 8, layer);
    w(L, 10, fmt(p.x));
    w(L, 20, fmt(p.y));
    w(L, 30, '0');
  }
  w(L, 0, 'SEQEND');
  w(L, 8, layer);
}

/** Open polyline (R12 POLYLINE with flag 0) – for fold / score lines */
function addOpenPolyline(L: string[], pts: THREE.Vector2[], layer: string): void {
  w(L, 0, 'POLYLINE');
  w(L, 8, layer);
  w(L, 66, '1');   // vertices follow
  w(L, 70, '0');   // 0 = open polyline
  for (const p of pts) {
    w(L, 0, 'VERTEX');
    w(L, 8, layer);
    w(L, 10, fmt(p.x));
    w(L, 20, fmt(p.y));
    w(L, 30, '0');
  }
  w(L, 0, 'SEQEND');
  w(L, 8, layer);
}

function addCircle(L: string[], center: THREE.Vector2, radius: number, layer: string): void {
  w(L, 0, 'CIRCLE');
  w(L, 8, layer);
  w(L, 10, fmt(center.x));
  w(L, 20, fmt(center.y));
  w(L, 30, '0');
  w(L, 40, fmt(radius));
}

function addText(
  L: string[],
  pos: THREE.Vector2,
  text: string,
  layer: string,
  height: number,
): void {
  w(L, 0, 'TEXT');
  w(L, 8, layer);
  w(L, 10, fmt(pos.x));
  w(L, 20, fmt(pos.y));
  w(L, 30, '0');
  w(L, 40, fmt(height));
  w(L, 1, text);
  w(L, 72, '1'); // center horizontal alignment
  w(L, 11, fmt(pos.x));
  w(L, 21, fmt(pos.y));
  w(L, 31, '0');
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Push one DXF group-code / value pair */
function w(L: string[], code: number, value: string): void {
  L.push(String(code), value);
}

/** Format a number to 4 decimal places (0.1 µm precision – plenty for laser) */
function fmt(n: number): string {
  return n.toFixed(4);
}

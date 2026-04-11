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
 * Format: AutoCAD R14 (AC1014) with all 9 required symbol tables.
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
// Handle counter (reset per export call)
// ─────────────────────────────────────────────────────────────────────────────

let _h = 0x20;
function H(): string { return (_h++).toString(16).toUpperCase(); }

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

export function generateDXF(
  strips: UnfoldedStrip[],
  includeHoleIds: boolean,
  includeFoldLines: boolean,
): string {
  _h = 0x20;
  const L: string[] = [];

  // Pre-compute block names for BLOCK_RECORD table
  const blockNames = strips.map(s => blockNameFor(s));

  // ── HEADER ────────────────────────────────────────────────────────────────
  w(L, 0, 'SECTION');  w(L, 2, 'HEADER');
  w(L, 9, '$ACADVER'); w(L, 1, 'AC1014');
  w(L, 9, '$INSBASE'); w(L, 10, '0.0'); w(L, 20, '0.0'); w(L, 30, '0.0');
  w(L, 9, '$INSUNITS'); w(L, 70, '4');        // 4 = mm
  w(L, 9, '$MEASUREMENT'); w(L, 70, '1');     // 1 = metric
  w(L, 0, 'ENDSEC');

  // ── TABLES (all 9 symbol tables required by R14) ──────────────────────────
  w(L, 0, 'SECTION');  w(L, 2, 'TABLES');
  writeVportTable(L);
  writeLtypeTable(L);
  writeLayerTable(L);
  writeStyleTable(L);
  writeViewTable(L);
  writeUcsTable(L);
  writeAppidTable(L);
  writeDimstyleTable(L);
  writeBlockRecordTable(L, blockNames);
  w(L, 0, 'ENDSEC');

  // ── BLOCKS ────────────────────────────────────────────────────────────────
  w(L, 0, 'SECTION');  w(L, 2, 'BLOCKS');

  // Built-in blocks
  writeEmptyBlock(L, '*MODEL_SPACE');
  writeEmptyBlock(L, '*PAPER_SPACE');

  // One block per strip
  for (const strip of strips) {
    writeStripBlock(L, strip, includeHoleIds, includeFoldLines);
  }

  w(L, 0, 'ENDSEC');

  // ── ENTITIES ──────────────────────────────────────────────────────────────
  w(L, 0, 'SECTION');  w(L, 2, 'ENTITIES');

  for (const strip of strips) {
    w(L, 0, 'INSERT');
    w(L, 5, H());
    w(L, 8, '0');
    w(L, 2, blockNameFor(strip));
    w(L, 10, fmt(strip.boundingBox.minX));
    w(L, 20, fmt(strip.boundingBox.minY));
    w(L, 30, '0.0');
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
// Symbol-table writers (all 9 required tables)
// ─────────────────────────────────────────────────────────────────────────────

function writeVportTable(L: string[]): void {
  w(L, 0, 'TABLE');  w(L, 2, 'VPORT');  w(L, 5, H());  w(L, 70, '1');

  w(L, 0, 'VPORT');  w(L, 5, H());
  w(L, 2, '*ACTIVE');
  w(L, 70, '0');
  w(L, 10, '0.0');  w(L, 20, '0.0');   // lower-left
  w(L, 11, '1.0');  w(L, 21, '1.0');   // upper-right
  w(L, 12, '0.0');  w(L, 22, '0.0');   // view center
  w(L, 13, '0.0');  w(L, 23, '0.0');   // snap base
  w(L, 14, '1.0');  w(L, 24, '1.0');   // snap spacing
  w(L, 15, '0.0');  w(L, 25, '0.0');   // grid spacing
  w(L, 16, '0.0');  w(L, 26, '0.0');  w(L, 36, '1.0');  // view dir
  w(L, 17, '0.0');  w(L, 27, '0.0');  w(L, 37, '0.0');  // view target
  w(L, 40, '1.0');   // view height
  w(L, 41, '1.0');   // aspect ratio
  w(L, 42, '50.0');  // lens length
  w(L, 43, '0.0');   // front clip
  w(L, 44, '0.0');   // back clip
  w(L, 50, '0.0');   // snap rotation
  w(L, 51, '0.0');   // twist angle

  w(L, 0, 'ENDTAB');
}

function writeLtypeTable(L: string[]): void {
  w(L, 0, 'TABLE');  w(L, 2, 'LTYPE');  w(L, 5, H());  w(L, 70, '3');

  // ByBlock
  w(L, 0, 'LTYPE');  w(L, 5, H());
  w(L, 2, 'BYBLOCK');  w(L, 70, '0');
  w(L, 3, '');  w(L, 72, '65');  w(L, 73, '0');  w(L, 40, '0.0');

  // CONTINUOUS
  w(L, 0, 'LTYPE');  w(L, 5, H());
  w(L, 2, 'CONTINUOUS');  w(L, 70, '0');
  w(L, 3, 'Solid line');  w(L, 72, '65');  w(L, 73, '0');  w(L, 40, '0.0');

  // CENTER
  w(L, 0, 'LTYPE');  w(L, 5, H());
  w(L, 2, 'CENTER');  w(L, 70, '0');
  w(L, 3, 'Center ____ _ ____');
  w(L, 72, '65');  w(L, 73, '4');  w(L, 40, '2.0');
  w(L, 49, '1.25');  w(L, 49, '-0.25');
  w(L, 49, '0.25');  w(L, 49, '-0.25');

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

  w(L, 0, 'TABLE');  w(L, 2, 'LAYER');  w(L, 5, H());
  w(L, 70, String(allLayers.length));

  for (const ly of allLayers) {
    w(L, 0, 'LAYER');  w(L, 5, H());
    w(L, 2, ly.name);
    w(L, 70, '0');
    w(L, 62, String(ly.color));
    w(L, 6, ly.ltype);
  }

  w(L, 0, 'ENDTAB');
}

function writeStyleTable(L: string[]): void {
  w(L, 0, 'TABLE');  w(L, 2, 'STYLE');  w(L, 5, H());  w(L, 70, '1');

  w(L, 0, 'STYLE');  w(L, 5, H());
  w(L, 2, 'STANDARD');
  w(L, 70, '0');
  w(L, 40, '0.0');   // height (0=variable)
  w(L, 41, '1.0');   // width factor
  w(L, 50, '0.0');   // oblique angle
  w(L, 71, '0');     // generation flags
  w(L, 42, '2.5');   // last height
  w(L, 3, 'txt');    // font file
  w(L, 4, '');       // big font

  w(L, 0, 'ENDTAB');
}

function writeViewTable(L: string[]): void {
  w(L, 0, 'TABLE');  w(L, 2, 'VIEW');  w(L, 5, H());  w(L, 70, '0');
  w(L, 0, 'ENDTAB');
}

function writeUcsTable(L: string[]): void {
  w(L, 0, 'TABLE');  w(L, 2, 'UCS');  w(L, 5, H());  w(L, 70, '0');
  w(L, 0, 'ENDTAB');
}

function writeAppidTable(L: string[]): void {
  w(L, 0, 'TABLE');  w(L, 2, 'APPID');  w(L, 5, H());  w(L, 70, '1');

  w(L, 0, 'APPID');  w(L, 5, H());
  w(L, 2, 'ACAD');
  w(L, 70, '0');

  w(L, 0, 'ENDTAB');
}

function writeDimstyleTable(L: string[]): void {
  w(L, 0, 'TABLE');  w(L, 2, 'DIMSTYLE');  w(L, 5, H());  w(L, 70, '1');

  w(L, 0, 'DIMSTYLE');  w(L, 5, H());
  w(L, 2, 'STANDARD');
  w(L, 70, '0');
  w(L, 3, '');       // DIMPOST
  w(L, 4, '');       // DIMAPOST
  w(L, 40, '1.0');   // DIMSCALE
  w(L, 41, '2.5');   // DIMASZ
  w(L, 42, '0.625'); // DIMEXO
  w(L, 43, '3.75');  // DIMDLI
  w(L, 44, '1.25');  // DIMEXE
  w(L, 140, '2.5');  // DIMTXT
  w(L, 77, '1');     // DIMTAD
  w(L, 271, '2');    // DIMDEC

  w(L, 0, 'ENDTAB');
}

function writeBlockRecordTable(L: string[], userBlockNames: string[]): void {
  const count = 2 + userBlockNames.length;
  w(L, 0, 'TABLE');  w(L, 2, 'BLOCK_RECORD');  w(L, 5, H());
  w(L, 70, String(count));

  w(L, 0, 'BLOCK_RECORD');  w(L, 5, H());
  w(L, 2, '*MODEL_SPACE');

  w(L, 0, 'BLOCK_RECORD');  w(L, 5, H());
  w(L, 2, '*PAPER_SPACE');

  for (const name of userBlockNames) {
    w(L, 0, 'BLOCK_RECORD');  w(L, 5, H());
    w(L, 2, name);
  }

  w(L, 0, 'ENDTAB');
}

// ─────────────────────────────────────────────────────────────────────────────
// Block writers
// ─────────────────────────────────────────────────────────────────────────────

function blockNameFor(strip: UnfoldedStrip): string {
  return `STRIP_${strip.stripId.replace(/[^A-Za-z0-9_]/g, '_').toUpperCase()}`;
}

function writeEmptyBlock(L: string[], name: string): void {
  w(L, 0, 'BLOCK');  w(L, 5, H());
  w(L, 8, '0');
  w(L, 2, name);
  w(L, 70, '0');
  w(L, 10, '0.0');  w(L, 20, '0.0');  w(L, 30, '0.0');
  w(L, 0, 'ENDBLK');  w(L, 5, H());
  w(L, 8, '0');
}

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

  w(L, 0, 'BLOCK');  w(L, 5, H());
  w(L, 8, '0');
  w(L, 2, name);
  w(L, 70, '0');
  w(L, 10, '0.0');  w(L, 20, '0.0');  w(L, 30, '0.0');

  for (const seg of strip.segments) {
    // Closed outline
    if (seg.leftBoundary.length > 0 && seg.rightBoundary.length > 0) {
      const outline = [
        ...seg.leftBoundary,
        ...seg.rightBoundary.slice().reverse(),
      ].map(p => new THREE.Vector2(p.x - ox, p.y - oy));
      addClosedPolyline(L, outline, cutLayer);
    }

    // Fold / score line
    if (includeFoldLines && seg.centerline.length > 1) {
      const cl = seg.centerline.map(p => new THREE.Vector2(p.x - ox, p.y - oy));
      addPolyline(L, cl, 'SCORE');
    }

    // Holes
    for (const hole of seg.holes) {
      const c = new THREE.Vector2(hole.center.x - ox, hole.center.y - oy);
      addCircle(L, c, hole.radius, 'HOLE');
      if (includeHoleIds) {
        addText(L, c, String(hole.junctionId), 'LABEL', hole.radius * 0.8);
      }
    }
  }

  // Strip label
  if (strip.segments.length > 0 && strip.segments[0].centerline.length > 0) {
    const pt = strip.segments[0].centerline[0];
    const pos = new THREE.Vector2(
      pt.x - ox,
      pt.y - oy + strip.segments[0].width * 0.5,
    );
    addText(L, pos, strip.stripId, 'LABEL', strip.segments[0].width * 0.3);
  }

  w(L, 0, 'ENDBLK');  w(L, 5, H());
  w(L, 8, '0');
}

// ─────────────────────────────────────────────────────────────────────────────
// Entity writers
// ─────────────────────────────────────────────────────────────────────────────

function addClosedPolyline(L: string[], pts: THREE.Vector2[], layer: string): void {
  w(L, 0, 'LWPOLYLINE');  w(L, 5, H());
  w(L, 8, layer);
  w(L, 90, String(pts.length));
  w(L, 70, '1'); // closed
  for (const p of pts) {
    w(L, 10, fmt(p.x));
    w(L, 20, fmt(p.y));
  }
}

function addPolyline(L: string[], pts: THREE.Vector2[], layer: string): void {
  w(L, 0, 'LWPOLYLINE');  w(L, 5, H());
  w(L, 8, layer);
  w(L, 90, String(pts.length));
  w(L, 70, '0'); // open
  for (const p of pts) {
    w(L, 10, fmt(p.x));
    w(L, 20, fmt(p.y));
  }
}

function addCircle(L: string[], center: THREE.Vector2, radius: number, layer: string): void {
  w(L, 0, 'CIRCLE');  w(L, 5, H());
  w(L, 8, layer);
  w(L, 10, fmt(center.x));
  w(L, 20, fmt(center.y));
  w(L, 30, '0.0');
  w(L, 40, fmt(radius));
}

function addText(
  L: string[],
  pos: THREE.Vector2,
  text: string,
  layer: string,
  height: number,
): void {
  w(L, 0, 'TEXT');  w(L, 5, H());
  w(L, 8, layer);
  w(L, 10, fmt(pos.x));
  w(L, 20, fmt(pos.y));
  w(L, 30, '0.0');
  w(L, 40, fmt(height));
  w(L, 7, 'STANDARD');
  w(L, 1, text);
  w(L, 72, '1'); // center horizontal
  w(L, 11, fmt(pos.x));
  w(L, 21, fmt(pos.y));
  w(L, 31, '0.0');
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function w(L: string[], code: number, value: string): void {
  L.push(String(code), value);
}

function fmt(n: number): string {
  return n.toFixed(4);
}

/**
 * dxf.ts – DXF export optimized for laser-cutting + OpenNest nesting workflow
 *
 * Layer structure:
 *   CUT_A / CUT_B / CUT_C  → closed strip outlines per family  (laser: through-cut)
 *   HOLE                    → junction hole circles              (laser: through-cut / drill)
 *   SCORE                   → fold / centerlines                 (laser: score / engrave)
 *   LABEL                   → text annotations                   (laser: skip or light mark)
 *
 * Each strip outline is a single closed LWPOLYLINE on its CUT_* layer.
 * OpenNest can select all closed curves from CUT layers for nesting.
 *
 * Format: R12-compatible core (POLYLINE+VERTEX+SEQEND) with AC1012 header
 * for maximum Rhino / ODA compatibility.
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
  w(L, 9, '$ACADVER');     w(L, 1, 'AC1009');
  w(L, 9, '$INSBASE');     w(L, 10, '0.0'); w(L, 20, '0.0'); w(L, 30, '0.0');
  w(L, 9, '$MEASUREMENT'); w(L, 70, '1');   // metric
  w(L, 0, 'ENDSEC');

  // ── TABLES ────────────────────────────────────────────────────────────────
  w(L, 0, 'SECTION');
  w(L, 2, 'TABLES');
  writeLtypeTable(L);
  writeLayerTable(L);
  writeStyleTable(L);
  w(L, 0, 'ENDSEC');

  // ── ENTITIES (flat – no BLOCKs) ───────────────────────────────────────────
  w(L, 0, 'SECTION');
  w(L, 2, 'ENTITIES');

  for (const strip of strips) {
    const cutLayer = CUT_LAYERS[strip.family] ?? 'CUT_A';

    for (const seg of strip.segments) {
      // ── Closed outline (single closed polyline for OpenNest) ─────────────
      if (seg.leftBoundary.length > 0 && seg.rightBoundary.length > 0) {
        const outline = [
          ...seg.leftBoundary,
          ...seg.rightBoundary.slice().reverse(),
        ];
        addClosedPolyline(L, outline, cutLayer);
      }

      // ── Fold / score line ───────────────────────────────────────────────
      if (includeFoldLines && seg.centerline.length > 1) {
        addOpenPolyline(L, seg.centerline, 'SCORE');
      }

      // ── Junction holes ──────────────────────────────────────────────────
      for (const hole of seg.holes) {
        addCircle(L, hole.center, hole.radius, 'HOLE');

        if (includeHoleIds) {
          addText(L, hole.center, String(hole.junctionId), 'LABEL', hole.radius * 0.8);
        }
      }
    }

    // ── Strip label ─────────────────────────────────────────────────────────
    if (strip.segments.length > 0 && strip.segments[0].centerline.length > 0) {
      const labelPos = strip.segments[0].centerline[0].clone();
      labelPos.y += strip.segments[0].width * 0.5;
      addText(L, labelPos, strip.stripId, 'LABEL', strip.segments[0].width * 0.3);
    }
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
// Table writers
// ─────────────────────────────────────────────────────────────────────────────

function writeLtypeTable(L: string[]): void {
  w(L, 0, 'TABLE');
  w(L, 2, 'LTYPE');
  w(L, 70, '2');

  w(L, 0, 'LTYPE');
  w(L, 2, 'CONTINUOUS');
  w(L, 70, '0');
  w(L, 3, 'Solid line');
  w(L, 72, '65');
  w(L, 73, '0');
  w(L, 40, '0.0');

  w(L, 0, 'LTYPE');
  w(L, 2, 'CENTER');
  w(L, 70, '0');
  w(L, 3, 'Center ____ _ ____');
  w(L, 72, '65');
  w(L, 73, '4');
  w(L, 40, '2.0');
  w(L, 49, '1.25');
  w(L, 49, '-0.25');
  w(L, 49, '0.25');
  w(L, 49, '-0.25');

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

  for (const ly of allLayers) {
    w(L, 0, 'LAYER');
    w(L, 2, ly.name);
    w(L, 70, '0');
    w(L, 62, String(ly.color));
    w(L, 6, ly.ltype);
  }

  w(L, 0, 'ENDTAB');
}

function writeStyleTable(L: string[]): void {
  w(L, 0, 'TABLE');
  w(L, 2, 'STYLE');
  w(L, 70, '1');

  w(L, 0, 'STYLE');
  w(L, 2, 'STANDARD');
  w(L, 70, '0');
  w(L, 40, '0.0');
  w(L, 41, '1.0');
  w(L, 50, '0.0');
  w(L, 71, '0');
  w(L, 42, '2.5');
  w(L, 3, 'txt');
  w(L, 4, '');

  w(L, 0, 'ENDTAB');
}

// ─────────────────────────────────────────────────────────────────────────────
// Entity writers (R12 POLYLINE + VERTEX + SEQEND for max compatibility)
// ─────────────────────────────────────────────────────────────────────────────

function addClosedPolyline(L: string[], pts: THREE.Vector2[], layer: string): void {
  w(L, 0, 'POLYLINE');
  w(L, 8, layer);
  w(L, 66, '1');
  w(L, 70, '1');   // closed
  for (const p of pts) {
    w(L, 0, 'VERTEX');
    w(L, 8, layer);
    w(L, 10, fmt(p.x));
    w(L, 20, fmt(p.y));
    w(L, 30, '0.0');
  }
  w(L, 0, 'SEQEND');
  w(L, 8, layer);
}

function addOpenPolyline(L: string[], pts: THREE.Vector2[], layer: string): void {
  w(L, 0, 'POLYLINE');
  w(L, 8, layer);
  w(L, 6, 'CENTER');
  w(L, 66, '1');
  w(L, 70, '0');   // open
  for (const p of pts) {
    w(L, 0, 'VERTEX');
    w(L, 8, layer);
    w(L, 10, fmt(p.x));
    w(L, 20, fmt(p.y));
    w(L, 30, '0.0');
  }
  w(L, 0, 'SEQEND');
  w(L, 8, layer);
}

function addCircle(L: string[], center: THREE.Vector2, radius: number, layer: string): void {
  w(L, 0, 'CIRCLE');
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
  w(L, 0, 'TEXT');
  w(L, 8, layer);
  w(L, 10, fmt(pos.x));
  w(L, 20, fmt(pos.y));
  w(L, 30, '0.0');
  w(L, 40, fmt(height));
  w(L, 1, text);
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

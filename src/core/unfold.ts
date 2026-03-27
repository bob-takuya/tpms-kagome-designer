/**
 * unfold.ts – Phase 4: 2D flat-pattern unfolding of Kagome strips
 *
 * Strategy: Arc-length-preserving STRAIGHT unrolling.
 *
 * The centerline is laid out as a straight horizontal line (+x axis).
 * Each 3D segment length is preserved exactly (in mm).
 * Left/right edges are at ±halfWidth (uniform).
 * Junction holes are placed at their arc-length position on the centerline.
 *
 * Why NOT geodesic curvature integration:
 *   TPMS isolines have high geodesic curvature that accumulates into U/spiral
 *   shapes when integrated step-by-step, making the pattern unusable for
 *   fabrication.  For flat-cut Kagome strips, straight patterns are correct:
 *   the material is bent/curved during weaving assembly.
 */

import * as THREE from 'three';
import type { Strip, Junction } from './kagome';
import type { HalfEdgeMesh } from './halfEdge';

// ─────────────────────────────────────────────────────────────────────────────
// Public interfaces (backward-compatible with dxf.ts / viewport2d.ts)
// ─────────────────────────────────────────────────────────────────────────────

export interface UnfoldedStrip {
  stripId: string;
  family: number;
  layer: number;
  segments: UnfoldedSegment[];
  boundingBox: { minX: number; maxX: number; minY: number; maxY: number };
}

export interface UnfoldedSegment {
  startJunctionId: number | null;
  endJunctionId:   number | null;
  leftBoundary:    THREE.Vector2[];
  rightBoundary:   THREE.Vector2[];
  centerline:      THREE.Vector2[];
  holes:           UnfoldedHole[];
  width:           number;
}

export interface UnfoldedHole {
  junctionId: number;
  center:     THREE.Vector2;
  radius:     number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Core: straight arc-length unrolling
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Returns a straight horizontal 2D centerline with the same arc-length profile
 * as the 3D centerline.  Points go left→right along the x-axis; y = 0.
 */
function unfoldCenterline(
  cl:    THREE.Vector3[],
  scale: number,   // mm per world unit
): THREE.Vector2[] {
  const n = cl.length;
  if (n === 0) return [];

  const pts2: THREE.Vector2[] = [];
  let x = 0;

  for (let i = 0; i < n; i++) {
    pts2.push(new THREE.Vector2(x, 0));
    if (i < n - 1) {
      x += cl[i].distanceTo(cl[i + 1]) * scale;
    }
  }

  return pts2;
}

// ─────────────────────────────────────────────────────────────────────────────
// Left / right edges: simply ±halfW in the y-direction (strip is straight)
// ─────────────────────────────────────────────────────────────────────────────

function buildBoundaries(
  cl2:   THREE.Vector2[],
  halfW: number,
): { left: THREE.Vector2[]; right: THREE.Vector2[] } {
  const left  = cl2.map(p => new THREE.Vector2(p.x,  halfW));
  const right = cl2.map(p => new THREE.Vector2(p.x, -halfW));
  return { left, right };
}

// ─────────────────────────────────────────────────────────────────────────────
// Map each junction to its arc-length position on the 2D centerline
// ─────────────────────────────────────────────────────────────────────────────

function mapJunctions(
  strip:  Strip,
  cl3:    THREE.Vector3[],
  cl2:    THREE.Vector2[],
  scale:  number,
): UnfoldedHole[] {
  const holes: UnfoldedHole[] = [];

  for (const junc of strip.junctions) {
    // Find closest centerline point in 3D
    let minD2 = Infinity, idx = 0;
    for (let i = 0; i < cl3.length; i++) {
      const d2 = cl3[i].distanceToSquared(junc.position);
      if (d2 < minD2) { minD2 = d2; idx = i; }
    }

    // Interpolate between idx and idx+1 for sub-segment accuracy
    let xMm = cl2[idx].x;
    if (idx < cl3.length - 1) {
      const segLen3 = cl3[idx].distanceTo(cl3[idx + 1]);
      if (segLen3 > 1e-9) {
        const dA = cl3[idx    ].distanceTo(junc.position);
        const dB = cl3[idx + 1].distanceTo(junc.position);
        const t  = Math.max(0, Math.min(1, dA / (dA + dB)));
        xMm = cl2[idx].x + t * (cl2[idx + 1].x - cl2[idx].x);
      }
    }

    holes.push({
      junctionId: junc.id,
      center:     new THREE.Vector2(xMm, 0),
      radius:     junc.holeRadius * scale,
    });
  }

  // De-duplicate near-coincident holes
  const seen = new Set<string>();
  return holes.filter(h => {
    const k = `${Math.round(h.center.x * 10)}`;
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

export function unfoldStrip(
  strip:      Strip,
  _mesh:      HalfEdgeMesh,   // kept for API compatibility
  _junctions: Junction[],
  scale:      number,          // mm per world unit
): UnfoldedStrip {
  const cl3 = strip.centerline;
  const cl2 = unfoldCenterline(cl3, scale);

  // Strip half-width in mm; ensure at least 1 mm visible
  const halfW = Math.max(strip.width * scale * 0.5, 1.0);

  const { left, right } = buildBoundaries(cl2, halfW);
  const holes = mapJunctions(strip, cl3, cl2, scale);

  const totalLen = cl2.length > 0 ? cl2[cl2.length - 1].x : 0;

  const segment: UnfoldedSegment = {
    startJunctionId: null,
    endJunctionId:   null,
    leftBoundary:    left,
    rightBoundary:   right,
    centerline:      cl2,
    holes,
    width: strip.width * scale,
  };

  return {
    stripId: strip.id,
    family:  strip.family,
    layer:   strip.layer,
    segments: [segment],
    boundingBox: {
      minX: 0,
      maxX: totalLen,
      minY: -halfW,
      maxY:  halfW,
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Layout: row-based packing
// ─────────────────────────────────────────────────────────────────────────────

export interface StripLayout {
  strips:      UnfoldedStrip[];
  totalWidth:  number;
  totalHeight: number;
  positions:   THREE.Vector2[];
}

export function layoutStrips(
  strips:      UnfoldedStrip[],
  margin:      number,
  maxRowWidth: number,
): StripLayout {
  const positions: THREE.Vector2[] = [];

  let rowX = margin, rowY = margin, rowH = 0;

  for (const strip of strips) {
    const w = strip.boundingBox.maxX - strip.boundingBox.minX;
    const h = strip.boundingBox.maxY - strip.boundingBox.minY;

    if (rowX + w > maxRowWidth && rowX > margin) {
      rowY += rowH + margin;
      rowX  = margin;
      rowH  = 0;
    }

    positions.push(new THREE.Vector2(
      rowX - strip.boundingBox.minX,
      rowY - strip.boundingBox.minY,
    ));

    rowX += w + margin;
    rowH  = Math.max(rowH, h);
  }

  return {
    strips,
    totalWidth:  maxRowWidth + margin,
    totalHeight: rowY + rowH + margin,
    positions,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Apply layout offsets
// ─────────────────────────────────────────────────────────────────────────────

export function applyLayout(layout: StripLayout): UnfoldedStrip[] {
  return layout.strips.map((strip, idx) => {
    const off = layout.positions[idx];

    return {
      ...strip,
      segments: strip.segments.map(seg => ({
        ...seg,
        leftBoundary:  seg.leftBoundary.map(p  => p.clone().add(off)),
        rightBoundary: seg.rightBoundary.map(p  => p.clone().add(off)),
        centerline:    seg.centerline.map(p    => p.clone().add(off)),
        holes:         seg.holes.map(h => ({
          ...h,
          center: h.center.clone().add(off),
        })),
      })),
      boundingBox: {
        minX: strip.boundingBox.minX + off.x,
        maxX: strip.boundingBox.maxX + off.x,
        minY: strip.boundingBox.minY + off.y,
        maxY: strip.boundingBox.maxY + off.y,
      },
    };
  });
}

/**
 * unfold.ts – Phase 4: 2D flat-pattern unfolding of Kagome strips
 *
 * Geodesic unfolding via parallel transport:
 *   At each interior point p_i, the "in-surface normal" vector n_s is
 *   parallel-transported along the curve (no mesh-normal lookups needed).
 *   The signed geodesic turning angle uses this self-consistent n_s:
 *     θ_i = atan2( (t_in × t_out) · n_s,  t_in · t_out )
 *   Clamped to ±MAX_STEP per link, and total accumulated turn ≤ 270°.
 *   Strips that accumulate > 270° are re-rendered as straight lines (fallback).
 *
 * Outputs coordinates in mm (scale = mm per TPMS world unit).
 */

import * as THREE from 'three';
import type { Strip, Junction } from './kagome';
import type { HalfEdgeMesh } from './halfEdge';

// ─────────────────────────────────────────────────────────────────────────────
// Public interfaces  (backward-compatible with dxf.ts / viewport2d.ts)
// ─────────────────────────────────────────────────────────────────────────────

export interface UnfoldedStrip {
  stripId: string;
  family:  number;
  layer:   number;
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
// Straight arc-length unrolling (used as fallback when curvature blows up)
// ─────────────────────────────────────────────────────────────────────────────

function unfoldStraight(cl: THREE.Vector3[], scale: number): THREE.Vector2[] {
  const pts: THREE.Vector2[] = [];
  let x = 0;
  for (let i = 0; i < cl.length; i++) {
    pts.push(new THREE.Vector2(x, 0));
    if (i < cl.length - 1) x += cl[i].distanceTo(cl[i + 1]) * scale;
  }
  return pts;
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallel-transport geodesic unfolding
// ─────────────────────────────────────────────────────────────────────────────

const MAX_STEP  = Math.PI / 8;   // 22.5° per link max
const MAX_TOTAL = Math.PI * 1.5; // 270° total – beyond this assume spiral → fallback

function unfoldGeodesic(cl: THREE.Vector3[], scale: number): THREE.Vector2[] {
  const n = cl.length;
  if (n === 0) return [];

  const pts: THREE.Vector2[] = [new THREE.Vector2(0, 0)];
  if (n === 1) return pts;

  // ── Initial tangent ───────────────────────────────────────────────────────
  let t = new THREE.Vector3().subVectors(cl[1], cl[0]).normalize();

  // ── Initial in-surface perpendicular (n_s) ────────────────────────────────
  // Choose a world axis that's not collinear with t, then orthogonalise.
  const up = Math.abs(t.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
  let n_s = new THREE.Vector3().crossVectors(t, up).normalize();

  let dir       = 0;  // accumulated 2D direction (radians)
  let totalTurn = 0;

  for (let i = 1; i < n; i++) {
    const len  = cl[i - 1].distanceTo(cl[i]) * scale;
    const prev = pts[pts.length - 1];
    pts.push(new THREE.Vector2(
      prev.x + Math.cos(dir) * len,
      prev.y + Math.sin(dir) * len,
    ));

    if (i < n - 1) {
      const t_next = new THREE.Vector3().subVectors(cl[i + 1], cl[i]).normalize();

      // Signed geodesic turning angle using parallel-transported n_s
      const cross = new THREE.Vector3().crossVectors(t, t_next);
      const sinθ  = cross.dot(n_s);
      const cosθ  = Math.max(-1, Math.min(1, t.dot(t_next)));
      const θ     = Math.atan2(sinθ, cosθ);
      const step  = Math.max(-MAX_STEP, Math.min(MAX_STEP, θ));

      totalTurn += Math.abs(step);
      if (totalTurn > MAX_TOTAL) return null!; // signal fallback

      dir += step;

      // Parallel-transport n_s: remove component along t_next
      n_s.addScaledVector(t_next, -n_s.dot(t_next));
      if (n_s.lengthSq() < 1e-14) {
        // Degenerate: reinitialise perpendicular to t_next
        const alt = Math.abs(t_next.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
        n_s.crossVectors(t_next, alt);
      }
      n_s.normalize();
      t = t_next;
    }
  }

  return pts;
}

// ─────────────────────────────────────────────────────────────────────────────
// Choose geodesic or straight unrolling per strip
// ─────────────────────────────────────────────────────────────────────────────

function unfoldCenterline(cl: THREE.Vector3[], scale: number): THREE.Vector2[] {
  if (cl.length < 2) return unfoldStraight(cl, scale);
  const geo = unfoldGeodesic(cl, scale);
  if (geo) return geo;
  // Fallback: straight unrolling
  console.debug('[Unfold] geodesic overflowed → straight fallback');
  return unfoldStraight(cl, scale);
}

// ─────────────────────────────────────────────────────────────────────────────
// Boundaries: ±halfW perpendicular to the local 2D tangent at each point
// ─────────────────────────────────────────────────────────────────────────────

function buildBoundaries(
  cl2:   THREE.Vector2[],
  halfW: number,
): { left: THREE.Vector2[]; right: THREE.Vector2[] } {
  const n    = cl2.length;
  const left:  THREE.Vector2[] = [];
  const right: THREE.Vector2[] = [];

  for (let i = 0; i < n; i++) {
    // Central-difference tangent (clamped at ends)
    const a = cl2[Math.max(0, i - 1)];
    const b = cl2[Math.min(n - 1, i + 1)];
    const dx = b.x - a.x, dy = b.y - a.y;
    const len = Math.sqrt(dx * dx + dy * dy);
    // Perpendicular (rotated 90° CCW)
    const nx = len > 1e-12 ? -dy / len : 0;
    const ny = len > 1e-12 ?  dx / len : 1;

    const p = cl2[i];
    left.push(new THREE.Vector2(p.x + nx * halfW, p.y + ny * halfW));
    right.push(new THREE.Vector2(p.x - nx * halfW, p.y - ny * halfW));
  }

  return { left, right };
}

// ─────────────────────────────────────────────────────────────────────────────
// Map junction positions to 2D via arc-length interpolation
// ─────────────────────────────────────────────────────────────────────────────

function mapJunctions(
  strip: Strip,
  cl3:   THREE.Vector3[],
  cl2:   THREE.Vector2[],
  scale: number,
): UnfoldedHole[] {
  const holes: UnfoldedHole[] = [];

  for (const junc of strip.junctions) {
    let minD2 = Infinity, idx = 0;
    for (let i = 0; i < cl3.length; i++) {
      const d2 = cl3[i].distanceToSquared(junc.position);
      if (d2 < minD2) { minD2 = d2; idx = i; }
    }

    // Sub-segment interpolation
    let pt2 = cl2[idx].clone();
    if (idx < cl3.length - 1) {
      const dA  = cl3[idx    ].distanceTo(junc.position);
      const dB  = cl3[idx + 1].distanceTo(junc.position);
      const t   = dA + dB > 1e-9 ? dA / (dA + dB) : 0;
      pt2 = cl2[idx].clone().lerp(cl2[Math.min(idx + 1, cl2.length - 1)], t);
    }

    holes.push({
      junctionId: junc.id,
      center:     pt2,
      radius:     junc.holeRadius * scale,
    });
  }

  // De-duplicate by arc-length position
  const seen = new Set<string>();
  return holes.filter(h => {
    const k = `${Math.round(h.center.x)},${Math.round(h.center.y)}`;
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Bounding box from boundary points
// ─────────────────────────────────────────────────────────────────────────────

function bbox(left: THREE.Vector2[], right: THREE.Vector2[]) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const p of [...left, ...right]) {
    if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
  }
  return { minX, maxX, minY, maxY };
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

export function unfoldStrip(
  strip:      Strip,
  _mesh:      HalfEdgeMesh,   // API compatibility
  _junctions: Junction[],
  scale:      number,          // mm per world unit
): UnfoldedStrip {
  const cl3  = strip.centerline;
  const cl2  = unfoldCenterline(cl3, scale);
  const halfW = Math.max(strip.width * scale * 0.5, 0.5);

  const { left, right } = buildBoundaries(cl2, halfW);
  const holes = mapJunctions(strip, cl3, cl2, scale);

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
    stripId:  strip.id,
    family:   strip.family,
    layer:    strip.layer,
    segments: [segment],
    boundingBox: bbox(left, right),
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

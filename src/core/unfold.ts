/**
 * unfold.ts – Phase 4: Developable-strip approximation + isometric triangle unfolding
 *
 * TPMS surfaces have negative Gaussian curvature everywhere → strips are NOT
 * developable. We approximate each strip as a RULED surface using surface normals:
 *
 *   L3[i] = cl3[i] + halfW · perp_i
 *   R3[i] = cl3[i] − halfW · perp_i
 *   perp_i = normalize( tangent_i × surfNormal_i )  ← in-surface perpendicular
 *
 * Then unfold the ruled strip to 2D using sequential TRIANGLE UNFOLDING:
 *
 *   Each quad [L_i, R_i, L_{i+1}, R_{i+1}] is split into two triangles.
 *   Edge lengths are preserved exactly → isometric for developable surfaces,
 *   minimal-distortion for non-developable ones.
 *
 * Coordinates are output in mm (scale = mm per TPMS world unit).
 * Strip width per isoline is non-uniform (reflects local isoline density).
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
// Spatial hash for surface normal lookup
// ─────────────────────────────────────────────────────────────────────────────

let _normCache: { mesh: HalfEdgeMesh; fn: (p: THREE.Vector3) => THREE.Vector3 } | null = null;

function buildNormalLookup(mesh: HalfEdgeMesh): (p: THREE.Vector3) => THREE.Vector3 {
  if (_normCache?.mesh === mesh) return _normCache.fn;

  const CELL = 0.35;
  const grid = new Map<string, number[]>();
  for (let i = 0; i < mesh.vertices.length; i++) {
    const v = mesh.vertices[i];
    const k = `${Math.floor(v.x / CELL)},${Math.floor(v.y / CELL)},${Math.floor(v.z / CELL)}`;
    if (!grid.has(k)) grid.set(k, []);
    grid.get(k)!.push(i);
  }

  const fn = (p: THREE.Vector3): THREE.Vector3 => {
    const cx = Math.floor(p.x / CELL);
    const cy = Math.floor(p.y / CELL);
    const cz = Math.floor(p.z / CELL);
    let minD2 = Infinity, best = 0;
    for (let dx = -1; dx <= 1; dx++)
    for (let dy = -1; dy <= 1; dy++)
    for (let dz = -1; dz <= 1; dz++) {
      for (const idx of grid.get(`${cx + dx},${cy + dy},${cz + dz}`) ?? []) {
        const d2 = mesh.vertices[idx].distanceToSquared(p);
        if (d2 < minD2) { minD2 = d2; best = idx; }
      }
    }
    return mesh.normals[best].clone().normalize();
  };

  _normCache = { mesh, fn };
  return fn;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 1: Compute 3D left/right edges using surface normals
//
//   perp_i = normalize( tangent_i × surfNormal_i )
//   L3[i]  = cl3[i] + halfW_world · perp_i
//   R3[i]  = cl3[i] − halfW_world · perp_i
// ─────────────────────────────────────────────────────────────────────────────

function computeEdges3D(
  cl3:       THREE.Vector3[],
  halfW:     number,           // half-width in world units
  mesh:      HalfEdgeMesh,
): { L3: THREE.Vector3[]; R3: THREE.Vector3[] } {
  const getNorm = buildNormalLookup(mesh);
  const n = cl3.length;
  const L3: THREE.Vector3[] = [];
  const R3: THREE.Vector3[] = [];

  for (let i = 0; i < n; i++) {
    // Central-difference tangent
    const pa = cl3[Math.max(0, i - 1)];
    const pb = cl3[Math.min(n - 1, i + 1)];
    const t  = new THREE.Vector3().subVectors(pb, pa).normalize();

    // Surface normal at this point
    const nSurf = getNorm(cl3[i]);

    // In-surface perpendicular: tangent × surface_normal
    let perp = new THREE.Vector3().crossVectors(t, nSurf);
    if (perp.lengthSq() < 1e-10) {
      // Degenerate: t ≈ nSurf → use world-up fallback
      perp.crossVectors(t, new THREE.Vector3(0, 1, 0));
    }
    perp.normalize();

    L3.push(cl3[i].clone().addScaledVector(perp,  halfW));
    R3.push(cl3[i].clone().addScaledVector(perp, -halfW));
  }

  return { L3, R3 };
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 2: Triangle unfolding
//
// Each quad [L_i, R_i, L_{i+1}, R_{i+1}] is split by diagonal [R_i, L_{i+1}]:
//   Tri-1: (L_i, R_i, L_{i+1})  → place L_{i+1}_2d
//   Tri-2: (R_i, L_{i+1}, R_{i+1}) → place R_{i+1}_2d
//
// triPoint: given two known 2D points a, b and distances da, db to unknown p,
//   returns p on the LEFT (onLeft=true) or RIGHT (onLeft=false) of directed a→b.
// ─────────────────────────────────────────────────────────────────────────────

function triPoint(
  a: THREE.Vector2, b: THREE.Vector2,
  da: number, db: number,
  onLeft: boolean,
): THREE.Vector2 {
  const ab  = new THREE.Vector2().subVectors(b, a);
  const d   = ab.length();

  if (d < 1e-12) {
    // Degenerate edge: just extend in the last direction
    return a.clone().addScaledVector(ab.set(1, 0), da);
  }

  // Law of cosines: t = projection along a→b, s = perpendicular distance
  const t   = (da * da - db * db + d * d) / (2 * d);
  const s2  = Math.max(0, da * da - t * t);  // clamp for floating-point safety
  const s   = Math.sqrt(s2) * (onLeft ? 1 : -1);

  const dir  = ab.clone().divideScalar(d);
  const perp = new THREE.Vector2(-dir.y, dir.x);  // CCW perpendicular

  return a.clone()
    .addScaledVector(dir,  t)
    .addScaledVector(perp, s);
}

function unfoldTriangles(
  L3: THREE.Vector3[],
  R3: THREE.Vector3[],
): { L2: THREE.Vector2[]; R2: THREE.Vector2[] } {
  const n = L3.length;
  if (n === 0) return { L2: [], R2: [] };

  // Initial edge centred on origin
  const w0 = Math.max(L3[0].distanceTo(R3[0]), 1e-6);
  const L2: THREE.Vector2[] = [new THREE.Vector2(0,  w0 / 2)];
  const R2: THREE.Vector2[] = [new THREE.Vector2(0, -w0 / 2)];

  for (let i = 0; i < n - 1; i++) {
    const dLL = L3[i].distanceTo(L3[i + 1]);       // left  top edge
    const dRL = R3[i].distanceTo(L3[i + 1]);       // diagonal R_i → L_{i+1}
    const dRR = R3[i].distanceTo(R3[i + 1]);       // right top edge
    const dLR = L3[i + 1].distanceTo(R3[i + 1]);  // new cross edge

    // Tri-1: (L_i, R_i, L_{i+1})
    //   L_i → R_i points "downward"; LEFT = forward (+x side) → onLeft = true
    const L_next = triPoint(L2[i], R2[i], dLL, dRL, true);

    // Tri-2: (R_i, L_{i+1}, R_{i+1})
    //   R_{i+1} should be on the RIGHT side (the "under" side of the strip)
    //   R_i → L_{i+1} has mixed direction; RIGHT = toward previous right edge → onLeft = false
    const R_next = triPoint(R2[i], L_next, dRR, dLR, false);

    L2.push(L_next);
    R2.push(R_next);
  }

  return { L2, R2 };
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 3: Map junction 3D positions to 2D via centerline lookup
// ─────────────────────────────────────────────────────────────────────────────

function mapJunctions(
  strip: Strip,
  cl3:   THREE.Vector3[],
  cl2:   THREE.Vector2[],
  scale: number,
): UnfoldedHole[] {
  const holes: UnfoldedHole[] = [];

  for (const junc of strip.junctions) {
    // Closest centerline point in 3D
    let minD2 = Infinity, idx = 0;
    for (let i = 0; i < cl3.length; i++) {
      const d2 = cl3[i].distanceToSquared(junc.position);
      if (d2 < minD2) { minD2 = d2; idx = i; }
    }

    // Sub-segment interpolation for accuracy
    let pt2 = cl2[idx].clone();
    if (idx < cl3.length - 1 && idx < cl2.length - 1) {
      const dA = cl3[idx    ].distanceTo(junc.position);
      const dB = cl3[idx + 1].distanceTo(junc.position);
      const t  = dA + dB > 1e-9 ? dA / (dA + dB) : 0;
      pt2 = cl2[idx].clone().lerp(cl2[idx + 1], t);
    }

    holes.push({
      junctionId: junc.id,
      center:     pt2,
      radius:     junc.holeRadius * scale,
    });
  }

  // De-duplicate near-coincident holes
  const seen = new Set<string>();
  return holes.filter(h => {
    const k = `${Math.round(h.center.x)},${Math.round(h.center.y)}`;
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Bounding box
// ─────────────────────────────────────────────────────────────────────────────

function computeBBox(left: THREE.Vector2[], right: THREE.Vector2[]) {
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
  mesh:       HalfEdgeMesh,
  _junctions: Junction[],
  scale:      number,           // mm per world unit
): UnfoldedStrip {
  const cl3   = strip.centerline;
  const halfW = Math.max(strip.width / 2, 1e-6);  // world units

  // 1. Compute 3D ruled-surface edges
  const { L3, R3 } = computeEdges3D(cl3, halfW, mesh);

  // 2. Triangle unfolding → world-unit 2D coordinates → convert to mm
  const { L2: L2w, R2: R2w } = unfoldTriangles(L3, R3);
  const L2  = L2w.map(p => p.clone().multiplyScalar(scale));
  const R2  = R2w.map(p => p.clone().multiplyScalar(scale));
  const cl2 = L2.map((l, i) => l.clone().add(R2[i]).multiplyScalar(0.5));

  // 3. Junction holes (in mm)
  const holes = mapJunctions(strip, cl3, cl2, scale);

  const segment: UnfoldedSegment = {
    startJunctionId: null,
    endJunctionId:   null,
    leftBoundary:    L2,
    rightBoundary:   R2,
    centerline:      cl2,
    holes,
    width: strip.width * scale,
  };

  return {
    stripId:  strip.id,
    family:   strip.family,
    layer:    strip.layer,
    segments: [segment],
    boundingBox: computeBBox(L2, R2),
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Layout: row-based packing with auto maxRowWidth
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

/**
 * unfold.ts – Phase 4: Developable strip approximation via arc-length geodesic unrolling
 *
 * For strips on TPMS surfaces (which have K < 0 everywhere), we approximate
 * each strip as a developable ruled surface using the following approach:
 *
 * 1. Compute arc-length along the 3D centerline
 * 2. Compute geodesic curvature κ_g at each point (turning rate in the tangent plane)
 * 3. Unroll to 2D by integrating the turning angle: θ(s) = ∫ κ_g ds
 * 4. Place 2D centerline points: x(s) = ∫ cos(θ) ds,  y(s) = ∫ sin(θ) ds
 * 5. Add left/right boundaries at ±width/2 perpendicular to the 2D tangent
 *
 * This produces a developable strip that:
 * - Preserves arc length along the centerline exactly
 * - Preserves geodesic curvature (the "bending" in the surface plane)
 * - Has constant width along its length (uses strip.width or per-point strip.widths)
 * - Can be physically bent to conform to the TPMS surface
 *
 * Output coordinates are in mm (scale = mm per TPMS world unit).
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
// Spatial-hash surface normal lookup
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
// Compute geodesic curvature along the centerline
//
// Discrete geodesic unrolling via parallel transport
//
// At each step we project the next 3D tangent onto the previous tangent plane,
// then measure the signed rotation angle (dθ) using the surface normal as
// the reference axis.  No numerical differentiation → no κ_g noise.
//
// This is equivalent to computing ∫ κ_g ds sequentially along arc length s:
//   • Normal curvature κ_n  → absorbed by the projection (flattened out)
//   • Geodesic curvature κ_g → preserved as the 2D bending angle dθ
// ─────────────────────────────────────────────────────────────────────────────

interface GeodesicData {
  arcLengths: number[];   // cumulative arc length at each point
  kappaG:     number[];   // (kept for downstream compat; set to dθ/ds approximation)
  tangents2D: THREE.Vector2[];  // 2D tangent directions after unrolling
}

function computeGeodesicData(
  cl3:  THREE.Vector3[],
  mesh: HalfEdgeMesh,
): GeodesicData {
  const n = cl3.length;
  const getNorm = buildNormalLookup(mesh);

  // ── Unit tangents ──────────────────────────────────────────────────────────
  const tangents3D: THREE.Vector3[] = [];
  for (let i = 0; i < n; i++) {
    const pa = cl3[Math.max(0, i - 1)];
    const pb = cl3[Math.min(n - 1, i + 1)];
    const t = new THREE.Vector3().subVectors(pb, pa);
    if (t.lengthSq() < 1e-14) t.set(1, 0, 0);
    tangents3D.push(t.normalize());
  }

  // ── Surface normals ────────────────────────────────────────────────────────
  const surfNormals: THREE.Vector3[] = cl3.map(p => getNorm(p));

  // ── Arc lengths (cumulative) ───────────────────────────────────────────────
  const arcLengths: number[] = [0];
  for (let i = 1; i < n; i++) {
    arcLengths.push(arcLengths[i - 1] + cl3[i - 1].distanceTo(cl3[i]));
  }

  // ── Parallel-transport unrolling ───────────────────────────────────────────
  // At each step i → i+1:
  //   1. Project tangents3D[i+1] onto the tangent plane at point i
  //      (this removes the normal-curvature component – "stretching it out")
  //   2. Measure the signed angle dθ between t_prev and the projected vector
  //      (this IS the discrete geodesic curvature contribution – "kept as-is")
  //   3. Accumulate θ and emit the 2D tangent (cos θ, sin θ)
  const tangents2D: THREE.Vector2[] = [];
  const kappaG: number[] = new Array(n).fill(0);
  let theta = 0;
  tangents2D.push(new THREE.Vector2(1, 0));

  for (let i = 1; i < n; i++) {
    const n_i  = surfNormals[i - 1];          // surface normal at previous point
    const tPrv = tangents3D[i - 1];           // tangent at previous point
    const tNxt = tangents3D[i].clone();       // tangent at current point

    // Project tNxt onto the tangent plane at i-1 (removes κ_n contribution)
    const proj = tNxt.addScaledVector(n_i, -tNxt.dot(n_i));
    if (proj.lengthSq() < 1e-20) {
      tangents2D.push(new THREE.Vector2(Math.cos(theta), Math.sin(theta)));
      continue;
    }
    proj.normalize();

    // Signed angle in the tangent plane (n_i as the rotation axis)
    const cross = new THREE.Vector3().crossVectors(tPrv, proj);
    const dTheta = Math.atan2(cross.dot(n_i), tPrv.dot(proj));

    theta += dTheta;
    tangents2D.push(new THREE.Vector2(Math.cos(theta), Math.sin(theta)));

    // Store approximate κ_g for downstream compat (ds > 0 guaranteed by arcLengths check)
    const ds = arcLengths[i] - arcLengths[i - 1];
    kappaG[i] = ds > 1e-12 ? dTheta / ds : 0;
  }
  kappaG[0] = kappaG[1] ?? 0;

  return { arcLengths, kappaG, tangents2D };
}

// ─────────────────────────────────────────────────────────────────────────────
// Arc-length geodesic unrolling to 2D
//
// Place centerline points by integrating 2D tangent directions:
//   x(s) = ∫ cos(θ(s)) ds
//   y(s) = ∫ sin(θ(s)) ds
// Then add left/right boundaries at ±halfWidth perpendicular to tangent
// ─────────────────────────────────────────────────────────────────────────────

function unrollGeodesic(
  cl3:       THREE.Vector3[],
  geodesic:  GeodesicData,
  widths:    number[],      // per-point width in world units
  scale:     number,        // mm per world unit
): { L2: THREE.Vector2[]; R2: THREE.Vector2[]; cl2: THREE.Vector2[] } {
  const n = cl3.length;
  if (n === 0) return { L2: [], R2: [], cl2: [] };

  const cl2: THREE.Vector2[] = [];
  const L2:  THREE.Vector2[] = [];
  const R2:  THREE.Vector2[] = [];

  // Start at origin
  let x = 0, y = 0;
  cl2.push(new THREE.Vector2(x * scale, y * scale));

  // Integrate along the curve
  for (let i = 1; i < n; i++) {
    const ds = geodesic.arcLengths[i] - geodesic.arcLengths[i - 1];
    // Average of consecutive 2D tangents for smoother centerline.
    // Guard: if tangents are nearly anti-parallel (dθ ≈ π), their sum ≈ 0 and
    // normalize() returns (0,0), causing zero advancement → "abnormally short" strip.
    // Fall back to the previous tangent direction in that case.
    const tSum = geodesic.tangents2D[i - 1].clone().add(geodesic.tangents2D[i]);
    const tAvg = tSum.lengthSq() > 1e-10
      ? tSum.normalize()
      : geodesic.tangents2D[i - 1].clone();

    x += tAvg.x * ds;
    y += tAvg.y * ds;
    cl2.push(new THREE.Vector2(x * scale, y * scale));
  }

  // Add left/right boundaries using per-point widths
  for (let i = 0; i < n; i++) {
    const t2 = geodesic.tangents2D[i];
    // Perpendicular direction (90° CCW rotation)
    const perp = new THREE.Vector2(-t2.y, t2.x);

    const halfW = (widths[i] / 2) * scale;
    L2.push(cl2[i].clone().addScaledVector(perp,  halfW));
    R2.push(cl2[i].clone().addScaledVector(perp, -halfW));
  }

  return { L2, R2, cl2 };
}

// ─────────────────────────────────────────────────────────────────────────────
// Junction mapping: 3D junction position → 2D centerline point
// ─────────────────────────────────────────────────────────────────────────────

function mapJunctions(
  strip:       Strip,
  cl3:         THREE.Vector3[],
  cl2:         THREE.Vector2[],
  _arcLengths: number[],  // kept for potential future use (arc-length interpolation)
  scale:       number,
): UnfoldedHole[] {
  const holes: UnfoldedHole[] = [];

  for (const junc of strip.junctions) {
    // Find closest centerline point in 3D
    let minD2 = Infinity, idx = 0;
    for (let i = 0; i < cl3.length; i++) {
      const d2 = cl3[i].distanceToSquared(junc.position);
      if (d2 < minD2) { minD2 = d2; idx = i; }
    }

    // Interpolate along arc length for better accuracy
    let pt2 = cl2[idx].clone();
    if (idx < cl3.length - 1 && idx < cl2.length - 1) {
      const d3to = cl3[idx].distanceTo(junc.position);
      const segLen = cl3[idx].distanceTo(cl3[idx + 1]);
      if (segLen > 1e-9) {
        const t = Math.min(d3to / segLen, 1);
        pt2 = cl2[idx].clone().lerp(cl2[idx + 1], t);
      }
    }

    holes.push({
      junctionId: junc.id,
      center:     pt2,
      radius:     junc.holeRadius * scale,
    });
  }

  // Deduplicate near-coincident holes
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
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
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
  const cl3 = strip.centerline;
  if (cl3.length < 2) {
    return {
      stripId:     strip.id,
      family:      strip.family,
      layer:       strip.layer,
      segments:    [],
      boundingBox: { minX: 0, maxX: 0, minY: 0, maxY: 0 },
    };
  }

  // Use per-point widths if available, otherwise fall back to uniform width
  const widths: number[] = strip.widths && strip.widths.length === cl3.length
    ? strip.widths
    : new Array(cl3.length).fill(Math.max(strip.width, 1e-6));

  // Compute geodesic data (arc lengths, curvatures, 2D tangents)
  const geodesic = computeGeodesicData(cl3, mesh);

  // Unroll to 2D using arc-length parameterization
  const { L2, R2, cl2 } = unrollGeodesic(cl3, geodesic, widths, scale);

  // Map junction positions to 2D
  const holes = mapJunctions(strip, cl3, cl2, geodesic.arcLengths, scale);

  const segment: UnfoldedSegment = {
    startJunctionId: null,
    endJunctionId:   null,
    leftBoundary:    L2,
    rightBoundary:   R2,
    centerline:      cl2,
    holes,
    width:           strip.width * scale,
  };

  return {
    stripId:     strip.id,
    family:      strip.family,
    layer:       strip.layer,
    segments:    [segment],
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

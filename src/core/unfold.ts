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
// κ_g = (dt/ds) · n_s,  where:
//   t   = tangent vector (along the curve)
//   n_s = in-surface perpendicular = normalize(surfNormal × t)
// ─────────────────────────────────────────────────────────────────────────────

interface GeodesicData {
  arcLengths:  number[];   // cumulative arc length at each point
  kappaG:      number[];   // geodesic curvature at each point
  tangents2D:  THREE.Vector2[];  // 2D tangent directions after unrolling
}

function computeGeodesicData(
  cl3:  THREE.Vector3[],
  mesh: HalfEdgeMesh,
): GeodesicData {
  const n = cl3.length;
  const getNorm = buildNormalLookup(mesh);

  // ── Tangents (forward difference for stability) ────────────────────────────
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

  // ── In-surface perpendicular: n_s = normalize(surfNormal × t) ──────────────
  const nPerps: THREE.Vector3[] = tangents3D.map((t, i) => {
    let ns = new THREE.Vector3().crossVectors(surfNormals[i], t);
    if (ns.lengthSq() < 1e-10) ns.set(0, 0, 1).cross(t);
    return ns.normalize();
  });

  // ── Arc lengths (cumulative) ───────────────────────────────────────────────
  const arcLengths: number[] = [0];
  for (let i = 1; i < n; i++) {
    arcLengths.push(arcLengths[i - 1] + cl3[i - 1].distanceTo(cl3[i]));
  }

  // ── Geodesic curvature: κ_g = (dt/ds) · n_s  (central difference) ──────────
  const kappaG: number[] = new Array(n).fill(0);
  for (let i = 1; i < n - 1; i++) {
    const ds = cl3[i - 1].distanceTo(cl3[i + 1]);
    if (ds < 1e-12) continue;
    const dt = new THREE.Vector3().subVectors(tangents3D[i + 1], tangents3D[i - 1]);
    kappaG[i] = dt.dot(nPerps[i]) / ds;
  }
  // Extrapolate endpoints
  if (n >= 2) {
    kappaG[0] = kappaG[1];
    kappaG[n - 1] = kappaG[n - 2];
  }

  // ── Clamp extreme curvatures to avoid self-intersecting strips ─────────────
  const maxKappa = 0.8;  // physical limit: radius of curvature ≥ 1.25 world units
  for (let i = 0; i < n; i++) {
    kappaG[i] = Math.max(-maxKappa, Math.min(maxKappa, kappaG[i]));
  }

  // ── Compute 2D tangents by integrating geodesic curvature ──────────────────
  // θ(s) = ∫ κ_g ds  →  tangent2D = (cos θ, sin θ)
  // Guard: if total |θ| exceeds 120° the strip spirals back on itself.
  // Beyond that threshold we freeze θ so the remainder unfolds as a straight line.
  const MAX_TOTAL_THETA = (2 * Math.PI) / 3;  // 120° – physical bending limit
  const tangents2D: THREE.Vector2[] = [];
  let theta = 0;  // initial angle (strip starts pointing in +x direction)
  tangents2D.push(new THREE.Vector2(Math.cos(theta), Math.sin(theta)));

  for (let i = 1; i < n; i++) {
    const ds = arcLengths[i] - arcLengths[i - 1];
    const kAvg = (kappaG[i - 1] + kappaG[i]) / 2;
    const dTheta = kAvg * ds;
    const nextTheta = theta + dTheta;
    // Freeze accumulation once the strip would exceed the bending limit
    theta = Math.abs(nextTheta) <= MAX_TOTAL_THETA
      ? nextTheta
      : Math.sign(nextTheta) * MAX_TOTAL_THETA;
    tangents2D.push(new THREE.Vector2(Math.cos(theta), Math.sin(theta)));
  }

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
    // Use average tangent for better accuracy
    const tAvg = geodesic.tangents2D[i - 1].clone().add(geodesic.tangents2D[i]).multiplyScalar(0.5);
    tAvg.normalize();

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

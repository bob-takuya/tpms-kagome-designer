/**
 * unfold.ts – Phase 4: Darboux-frame geodesic unfolding with normal curvature
 *
 * For each strip on the TPMS surface we compute the Darboux frame at every
 * centerline point:
 *
 *   t   – tangent (along the strip)
 *   n_s – in-surface normal (perp to t inside the tangent plane)
 *   n   – surface normal
 *
 * Differential-geometry quantities:
 *   κ_g = geodesic curvature  = (dt/ds) · n_s
 *   τ_g = geodesic torsion    = -(dn_s/ds) · n
 *
 * LEFT / RIGHT edge arc length per centerline arc-length ds:
 *   dL/ds = √( (1 − halfW·κ_g)² + (halfW·τ_g)² )
 *   dR/ds = √( (1 + halfW·κ_g)² + (halfW·τ_g)² )
 *
 * The cross-section width (arc length in the n_s direction) = 2·halfW (constant).
 *
 * 2-D placement uses sequential triangle unfolding (law of cosines), which is
 * exact for developable surfaces and minimal-distortion for non-developable ones.
 *
 * All output coordinates are in mm (scale = mm per TPMS world unit).
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
    const cx = Math.floor(p.x / CELL), cy = Math.floor(p.y / CELL), cz = Math.floor(p.z / CELL);
    let minD2 = Infinity, best = 0;
    for (let dx = -1; dx <= 1; dx++)
    for (let dy = -1; dy <= 1; dy++)
    for (let dz = -1; dz <= 1; dz++) {
      for (const idx of grid.get(`${cx+dx},${cy+dy},${cz+dz}`) ?? []) {
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
// Darboux frame at each centerline point
// ─────────────────────────────────────────────────────────────────────────────

interface DarbouxData {
  tangents: THREE.Vector3[];
  nPerps:   THREE.Vector3[];  // in-surface perpendicular = n_s
  nSurfs:   THREE.Vector3[];  // surface normal = n
  kappasG:  number[];         // geodesic curvature κ_g
  tausG:    number[];         // geodesic torsion τ_g
}

function computeDarbouxFrame(
  cl3:  THREE.Vector3[],
  mesh: HalfEdgeMesh,
): DarbouxData {
  const n = cl3.length;
  const getNorm = buildNormalLookup(mesh);

  // ── Tangents (central difference) ─────────────────────────────────────────
  const tangents: THREE.Vector3[] = cl3.map((_, i) => {
    const a = cl3[Math.max(0, i - 1)];
    const b = cl3[Math.min(n - 1, i + 1)];
    return new THREE.Vector3().subVectors(b, a).normalize();
  });

  // ── Surface normals ────────────────────────────────────────────────────────
  const nSurfs: THREE.Vector3[] = cl3.map(p => getNorm(p));

  // ── In-surface perpendicular: n_s = normalize(n × t) ─────────────────────
  const nPerps: THREE.Vector3[] = tangents.map((t, i) => {
    let ns = new THREE.Vector3().crossVectors(nSurfs[i], t);
    if (ns.lengthSq() < 1e-10) ns.crossVectors(new THREE.Vector3(0, 0, 1), t);
    return ns.normalize();
  });

  // ── Geodesic curvature: κ_g = (dt/ds) · n_s  (central difference) ────────
  const kappasG: number[] = new Array(n).fill(0);
  for (let i = 1; i < n - 1; i++) {
    const ds = cl3[i - 1].distanceTo(cl3[i + 1]);
    if (ds < 1e-12) continue;
    const dt = new THREE.Vector3().subVectors(tangents[i + 1], tangents[i - 1]);
    kappasG[i] = dt.dot(nPerps[i]) / ds;
  }
  if (n >= 2) { kappasG[0] = kappasG[1]; kappasG[n - 1] = kappasG[n - 2]; }

  // ── Geodesic torsion: τ_g = −(dn_s/ds) · n  (central difference) ─────────
  const tausG: number[] = new Array(n).fill(0);
  for (let i = 1; i < n - 1; i++) {
    const ds = cl3[i - 1].distanceTo(cl3[i + 1]);
    if (ds < 1e-12) continue;
    const dns = new THREE.Vector3().subVectors(nPerps[i + 1], nPerps[i - 1]);
    tausG[i] = -dns.dot(nSurfs[i]) / ds;
  }
  if (n >= 2) { tausG[0] = tausG[1]; tausG[n - 1] = tausG[n - 2]; }

  return { tangents, nPerps, nSurfs, kappasG, tausG };
}

// ─────────────────────────────────────────────────────────────────────────────
// Triangle unfolding: place a 2-D point from two known base points + two distances
// "onLeft" = place on the left side (CCW) of the directed base edge a → b
// ─────────────────────────────────────────────────────────────────────────────

function triPoint(
  a: THREE.Vector2, b: THREE.Vector2,
  da: number, db: number,
  onLeft: boolean,
): THREE.Vector2 {
  const ab  = new THREE.Vector2().subVectors(b, a);
  const d   = ab.length();
  if (d < 1e-12) return a.clone();
  const t   = (da * da - db * db + d * d) / (2 * d);
  const s2  = Math.max(0, da * da - t * t);
  const s   = Math.sqrt(s2) * (onLeft ? 1 : -1);
  const dir = ab.clone().divideScalar(d);
  const perp = new THREE.Vector2(-dir.y, dir.x);
  return a.clone().addScaledVector(dir, t).addScaledVector(perp, s);
}

// ─────────────────────────────────────────────────────────────────────────────
// Main unfold: Darboux edge lengths + triangle unfolding
// ─────────────────────────────────────────────────────────────────────────────

function unfoldWithDarboux(
  cl3:    THREE.Vector3[],
  darboux: DarbouxData,
  halfW:  number,   // world units
  scale:  number,   // mm per world unit
): { L2: THREE.Vector2[]; R2: THREE.Vector2[] } {
  const n = cl3.length;
  if (n === 0) return { L2: [], R2: [] };

  // 3D edge positions (for diagonal computation)
  const L3 = cl3.map((p, i) => p.clone().addScaledVector(darboux.nPerps[i],  halfW));
  const R3 = cl3.map((p, i) => p.clone().addScaledVector(darboux.nPerps[i], -halfW));

  const w0mm = halfW * 2 * scale;
  const L2: THREE.Vector2[] = [new THREE.Vector2(0,  w0mm / 2)];
  const R2: THREE.Vector2[] = [new THREE.Vector2(0, -w0mm / 2)];

  for (let i = 0; i < n - 1; i++) {
    // Centerline arc-length for this link
    const ds = cl3[i].distanceTo(cl3[i + 1]);

    // Average Darboux scalars over the link
    const kg = (darboux.kappasG[i] + darboux.kappasG[i + 1]) / 2;
    const tg = (darboux.tausG[i]   + darboux.tausG[i + 1])   / 2;

    // Clamp curvature to avoid degenerate strips
    // (halfW · |κ_g| < 1 is required for the strip to be non-self-intersecting)
    const kg_clamped = Math.max(-0.9 / Math.max(halfW, 1e-9),
                               Math.min( 0.9 / Math.max(halfW, 1e-9), kg));

    // Analytical edge arc lengths (Darboux formula)
    const dL_mm = ds * Math.sqrt((1 - halfW * kg_clamped) ** 2 + (halfW * tg) ** 2) * scale;
    const dR_mm = ds * Math.sqrt((1 + halfW * kg_clamped) ** 2 + (halfW * tg) ** 2) * scale;

    // Diagonal from R3[i] to L3[i+1] in 3D (preserves the quad's diagonal in 2D)
    const dDiag_mm = R3[i].distanceTo(L3[i + 1]) * scale;

    // Cross-section at i+1 = 2*halfW (constant in Darboux approximation)
    const dCross_mm = 2 * halfW * scale;

    // Tri-1: (L_i, R_i, L_{i+1}) → place L_{i+1} to the LEFT of L_i→R_i (= forward)
    const L_next = triPoint(L2[i], R2[i], dL_mm, dDiag_mm, true);

    // Tri-2: (R_i, L_{i+1}, R_{i+1}) → place R_{i+1} to the RIGHT of R_i→L_{i+1}
    const R_next = triPoint(R2[i], L_next, dR_mm, dCross_mm, false);

    L2.push(L_next);
    R2.push(R_next);
  }

  return { L2, R2 };
}

// ─────────────────────────────────────────────────────────────────────────────
// Junction mapping: 3D junction position → 2D centerline point
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
    if (idx < cl3.length - 1 && idx < cl2.length - 1) {
      const dA = cl3[idx    ].distanceTo(junc.position);
      const dB = cl3[idx + 1].distanceTo(junc.position);
      pt2 = cl2[idx].clone().lerp(cl2[idx + 1], dA / Math.max(dA + dB, 1e-12));
    }
    holes.push({ junctionId: junc.id, center: pt2, radius: junc.holeRadius * scale });
  }

  // Deduplicate
  const seen = new Set<string>();
  return holes.filter(h => {
    const k = `${Math.round(h.center.x)},${Math.round(h.center.y)}`;
    return seen.has(k) ? false : (seen.add(k), true);
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
  scale:      number,
): UnfoldedStrip {
  const cl3   = strip.centerline;
  const halfW = Math.max(strip.width / 2, 1e-6);

  // Darboux frame
  const darboux = computeDarbouxFrame(cl3, mesh);

  // Triangle unfold (Darboux edge lengths)
  const { L2, R2 } = unfoldWithDarboux(cl3, darboux, halfW, scale);
  const cl2 = L2.map((l, i) => l.clone().add(R2[i]).multiplyScalar(0.5));

  const holes   = mapJunctions(strip, cl3, cl2, scale);
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
// Layout
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
    if (rowX + w > maxRowWidth && rowX > margin) { rowY += rowH + margin; rowX = margin; rowH = 0; }
    positions.push(new THREE.Vector2(rowX - strip.boundingBox.minX, rowY - strip.boundingBox.minY));
    rowX += w + margin;
    rowH = Math.max(rowH, h);
  }

  return { strips, totalWidth: maxRowWidth + margin, totalHeight: rowY + rowH + margin, positions };
}

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
        holes:         seg.holes.map(h => ({ ...h, center: h.center.clone().add(off) })),
      })),
      boundingBox: {
        minX: strip.boundingBox.minX + off.x, maxX: strip.boundingBox.maxX + off.x,
        minY: strip.boundingBox.minY + off.y, maxY: strip.boundingBox.maxY + off.y,
      },
    };
  });
}

/**
 * unfold.ts – Phase 4: Geodesic unfolding of Kagome strips to 2D
 *
 * Algorithm (per strip):
 *   1. Walk the 3D centerline [p0, p1, ..., pN]
 *   2. At each interior point p_i, compute the signed geodesic turning angle:
 *        θ_i = atan2( (t_in × t_out) · n_i,  t_in · t_out )
 *      where t_in = normalize(p_i − p_{i−1}),
 *            t_out= normalize(p_{i+1} − p_i),
 *            n_i  = surface normal at p_i (nearest vertex)
 *   3. Accumulate arc lengths → 2D centerline
 *   4. Build left/right edges at ±halfWidth perpendicular to the 2D tangent
 *   5. Map junction positions to 2D via closest-centerline-point lookup
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
// Spatial hash for nearest-vertex normal lookup (same approach as stripMesh.ts)
// ─────────────────────────────────────────────────────────────────────────────

interface NormalLookup { fn: (p: THREE.Vector3) => THREE.Vector3; mesh: HalfEdgeMesh }
let _nlCache: NormalLookup | null = null;

function getNormalLookup(mesh: HalfEdgeMesh): (p: THREE.Vector3) => THREE.Vector3 {
  if (_nlCache?.mesh === mesh) return _nlCache.fn;

  const CELL = 0.35;
  const grid = new Map<string, number[]>();
  for (let i = 0; i < mesh.vertices.length; i++) {
    const v = mesh.vertices[i];
    const k = `${Math.floor(v.x/CELL)},${Math.floor(v.y/CELL)},${Math.floor(v.z/CELL)}`;
    if (!grid.has(k)) grid.set(k, []);
    grid.get(k)!.push(i);
  }

  const fn = (p: THREE.Vector3): THREE.Vector3 => {
    const cx = Math.floor(p.x/CELL), cy = Math.floor(p.y/CELL), cz = Math.floor(p.z/CELL);
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

  _nlCache = { fn, mesh };
  return fn;
}

// ─────────────────────────────────────────────────────────────────────────────
// Core: unfold a 3D centerline to 2D via signed geodesic turning angles
// ─────────────────────────────────────────────────────────────────────────────

function unfoldCenterline(
  cl:    THREE.Vector3[],
  mesh:  HalfEdgeMesh,
  scale: number,
): THREE.Vector2[] {
  const n = cl.length;
  if (n === 0) return [];
  if (n === 1) return [new THREE.Vector2(0, 0)];

  const getNorm = getNormalLookup(mesh);
  const pts2: THREE.Vector2[] = [new THREE.Vector2(0, 0)];

  let dir = 0; // current 2D direction (radians, 0 = +x)

  for (let i = 1; i < n; i++) {
    const segLen = cl[i - 1].distanceTo(cl[i]) * scale;

    // Place next point along current direction
    const prev = pts2[pts2.length - 1];
    pts2.push(new THREE.Vector2(
      prev.x + Math.cos(dir) * segLen,
      prev.y + Math.sin(dir) * segLen,
    ));

    // Update direction at point i (for the NEXT segment)
    if (i < n - 1) {
      const surfN = getNorm(cl[i]);
      const tIn   = new THREE.Vector3().subVectors(cl[i],   cl[i - 1]).normalize();
      const tOut  = new THREE.Vector3().subVectors(cl[i + 1], cl[i]).normalize();

      // Signed angle in the surface tangent plane:
      //   sinθ = (tIn × tOut) · surfN
      //   cosθ = tIn · tOut
      const cross = new THREE.Vector3().crossVectors(tIn, tOut);
      const sinθ  = cross.dot(surfN);
      const cosθ  = tIn.dot(tOut);
      const θ     = Math.atan2(sinθ, cosθ);

      dir += θ; // accumulate geodesic curvature
    }
  }

  return pts2;
}

// ─────────────────────────────────────────────────────────────────────────────
// Build left/right boundary from unfolded centerline + halfWidth
// ─────────────────────────────────────────────────────────────────────────────

function buildBoundaries(
  cl2: THREE.Vector2[],
  halfW: number,
): { left: THREE.Vector2[]; right: THREE.Vector2[] } {
  const n = cl2.length;
  const left:  THREE.Vector2[] = [];
  const right: THREE.Vector2[] = [];

  for (let i = 0; i < n; i++) {
    // Tangent in 2D (central difference, clamped at ends)
    const a = cl2[Math.max(0, i - 1)];
    const b = cl2[Math.min(n - 1, i + 1)];
    const tx = b.x - a.x, ty = b.y - a.y;
    const len = Math.sqrt(tx * tx + ty * ty);
    const nx  = len > 1e-12 ? -ty / len : 0;   // normal (rotated 90° CCW)
    const ny  = len > 1e-12 ?  tx / len : 1;

    const p = cl2[i];
    left.push(new THREE.Vector2(p.x + nx * halfW,  p.y + ny * halfW));
    right.push(new THREE.Vector2(p.x - nx * halfW, p.y - ny * halfW));
  }

  return { left, right };
}

// ─────────────────────────────────────────────────────────────────────────────
// Map each junction to the closest 2D centerline position
// ─────────────────────────────────────────────────────────────────────────────

function mapJunctions(
  strip:  Strip,
  cl3:    THREE.Vector3[],
  cl2:    THREE.Vector2[],
  _junctionsParam: Junction[],
  scale:  number,
): UnfoldedHole[] {
  const holes: UnfoldedHole[] = [];

  for (const junc of strip.junctions) {
    let minD2 = Infinity, idx = 0;
    for (let i = 0; i < cl3.length; i++) {
      const d2 = cl3[i].distanceToSquared(junc.position);
      if (d2 < minD2) { minD2 = d2; idx = i; }
    }

    if (idx < cl2.length) {
      holes.push({
        junctionId: junc.id,
        center:     cl2[idx].clone(),
        radius:     junc.holeRadius * scale,
      });
    }
  }

  // De-duplicate: if two junctions map to the same 2D point, keep only one
  const seen = new Set<string>();
  return holes.filter(h => {
    const k = `${Math.round(h.center.x * 100)},${Math.round(h.center.y * 100)}`;
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
  mesh:       HalfEdgeMesh,
  _junctions: Junction[],
  scale:      number,
): UnfoldedStrip {
  const cl3 = strip.centerline;

  // Geodesic unfolding
  const cl2  = unfoldCenterline(cl3, mesh, scale);
  const halfW = Math.max(strip.width * scale * 0.5, 0.5); // min 0.5 units visible
  const { left, right } = buildBoundaries(cl2, halfW);

  // Junction holes (use strip.junctions, not the passed-in array)
  const holes = mapJunctions(strip, cl3, cl2, strip.junctions, scale);

  // One UnfoldedSegment covers the whole strip
  const segment: UnfoldedSegment = {
    startJunctionId: null,
    endJunctionId:   null,
    leftBoundary:    left,
    rightBoundary:   right,
    centerline:      cl2,
    holes,
    width: strip.width * scale,
  };

  // Bounding box
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const p of [...left, ...right]) {
    if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
  }

  return {
    stripId: strip.id,
    family:  strip.family,
    layer:   strip.layer,
    segments: [segment],
    boundingBox: { minX, maxX, minY, maxY },
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Layout: row-based packing with max row width
// ─────────────────────────────────────────────────────────────────────────────

export interface StripLayout {
  strips:      UnfoldedStrip[];
  totalWidth:  number;
  totalHeight: number;
  positions:   THREE.Vector2[];
}

export function layoutStrips(
  strips:    UnfoldedStrip[],
  margin:    number,
  maxRowWidth = 400,   // world units per row before wrapping
): StripLayout {
  const positions: THREE.Vector2[] = [];

  let rowX     = margin;
  let rowY     = margin;
  let rowH     = 0;

  for (const strip of strips) {
    const w = strip.boundingBox.maxX - strip.boundingBox.minX;
    const h = strip.boundingBox.maxY - strip.boundingBox.minY;

    // Wrap to next row when this strip doesn't fit
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

  const totalWidth  = maxRowWidth + margin;
  const totalHeight = rowY + rowH + margin;

  return { strips, totalWidth, totalHeight, positions };
}

// ─────────────────────────────────────────────────────────────────────────────
// Apply layout offsets to all segments/holes/boundaries
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

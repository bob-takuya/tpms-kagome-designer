/**
 * geodesic.ts – Discrete geodesic tracing on a triangle mesh.
 *
 * A geodesic is the straightest possible path on a curved surface.  On a
 * triangle mesh, a geodesic is traced by:
 *   1. Starting at a point P on face F with direction d
 *   2. Finding the edge where the ray from P in direction d exits F
 *   3. Unfolding the adjacent face F' into the plane of F (preserving angles)
 *   4. Continuing the straight line into F' and re-folding to 3D
 *
 * The component of d along the shared edge axis is preserved across the
 * crossing; the perpendicular component rotates by the dihedral angle.
 */

import * as THREE from 'three';
import type { HalfEdgeMesh } from './halfEdge';

// ─────────────────────────────────────────────────────────────────────────────
// Face adjacency (precomputed once per mesh)
// ─────────────────────────────────────────────────────────────────────────────

export interface FaceAdjacency {
  /** neighbors[3*fi + e] = index of the face adjacent to fi across edge e (or -1 at boundary) */
  neighbors: Int32Array;
}

function edgeKey(a: number, b: number): string {
  return a < b ? `${a},${b}` : `${b},${a}`;
}

export function buildFaceAdjacency(mesh: HalfEdgeMesh): FaceAdjacency {
  const nFaces = mesh.faces.length;
  const neighbors = new Int32Array(3 * nFaces).fill(-1);

  // Map edge → list of (face, local edge index)
  const edgeMap = new Map<string, Array<{ face: number; edge: number }>>();

  for (let fi = 0; fi < nFaces; fi++) {
    const face = mesh.faces[fi];
    for (let e = 0; e < 3; e++) {
      const v0 = face[e];
      const v1 = face[(e + 1) % 3];
      const key = edgeKey(v0, v1);
      if (!edgeMap.has(key)) edgeMap.set(key, []);
      edgeMap.get(key)!.push({ face: fi, edge: e });
    }
  }

  // Fill adjacency from matched pairs
  for (const entries of edgeMap.values()) {
    if (entries.length === 2) {
      neighbors[3 * entries[0].face + entries[0].edge] = entries[1].face;
      neighbors[3 * entries[1].face + entries[1].edge] = entries[0].face;
    }
  }

  return { neighbors };
}

// ─────────────────────────────────────────────────────────────────────────────
// Ray-edge intersection (in the plane of a triangle face)
//
// Solves:  origin + t * dir = edgeStart + s * (edgeEnd - edgeStart)
// Returns t if s ∈ [0, 1] and t > 0.
// ─────────────────────────────────────────────────────────────────────────────

function rayEdgeIntersect(
  origin: THREE.Vector3,
  dir: THREE.Vector3,
  edgeStart: THREE.Vector3,
  edgeEnd: THREE.Vector3,
): number | null {
  const edge = new THREE.Vector3().subVectors(edgeEnd, edgeStart);
  const rhs  = new THREE.Vector3().subVectors(edgeStart, origin);

  // n = dir × edge.  If n ≈ 0, ray is parallel to edge.
  const n = new THREE.Vector3().crossVectors(dir, edge);
  const n2 = n.lengthSq();
  if (n2 < 1e-20) return null;

  // t = (rhs × edge · n) / n2
  const t = new THREE.Vector3().crossVectors(rhs, edge).dot(n) / n2;
  // s = (rhs × dir · n) / n2
  const s = new THREE.Vector3().crossVectors(rhs, dir).dot(n) / n2;

  if (s < -1e-9 || s > 1 + 1e-9) return null;
  return t;
}

// ─────────────────────────────────────────────────────────────────────────────
// Unfold direction across an edge
//
// Decomposes dir into (along edge) + (perpendicular to edge) components.
// The "along" component is preserved.  The perpendicular component is
// rotated by the dihedral angle to lie in the target face's tangent plane,
// pointing "forward" into the target face's interior.
// ─────────────────────────────────────────────────────────────────────────────

function unfoldDirection(
  mesh: HalfEdgeMesh,
  dir: THREE.Vector3,
  toFace: number,
  edgeV0: THREE.Vector3,
  edgeV1: THREE.Vector3,
): THREE.Vector3 {
  const edgeAxis = new THREE.Vector3().subVectors(edgeV1, edgeV0);
  const edgeLen  = edgeAxis.length();
  if (edgeLen < 1e-12) return dir.clone();
  edgeAxis.divideScalar(edgeLen);

  // Decompose dir
  const along   = dir.dot(edgeAxis);
  const perpVec = dir.clone().addScaledVector(edgeAxis, -along);
  const perpLen = perpVec.length();
  if (perpLen < 1e-10) return dir.clone();

  // "Forward" perpendicular in toFace's tangent plane:
  //   from edge midpoint toward toFace's centroid, projected perpendicular to edge.
  const f = mesh.faces[toFace];
  const p0 = mesh.vertices[f[0]];
  const p1 = mesh.vertices[f[1]];
  const p2 = mesh.vertices[f[2]];
  const toCentroid = new THREE.Vector3().add(p0).add(p1).add(p2).multiplyScalar(1 / 3);
  const edgeMid = new THREE.Vector3().addVectors(edgeV0, edgeV1).multiplyScalar(0.5);

  const forward = new THREE.Vector3().subVectors(toCentroid, edgeMid);
  forward.addScaledVector(edgeAxis, -forward.dot(edgeAxis));
  const fLen = forward.length();
  if (fLen < 1e-10) return dir.clone();
  forward.divideScalar(fLen);

  // Reconstruct: along component preserved, perpendicular replaced by forward · perpLen
  return new THREE.Vector3()
    .addScaledVector(edgeAxis, along)
    .addScaledVector(forward,  perpLen)
    .normalize();
}

// ─────────────────────────────────────────────────────────────────────────────
// Project a vector onto a face's tangent plane
// ─────────────────────────────────────────────────────────────────────────────

function projectToFace(mesh: HalfEdgeMesh, faceIdx: number, v: THREE.Vector3): THREE.Vector3 {
  const f = mesh.faces[faceIdx];
  const p0 = mesh.vertices[f[0]];
  const p1 = mesh.vertices[f[1]];
  const p2 = mesh.vertices[f[2]];
  const n = new THREE.Vector3().crossVectors(
    new THREE.Vector3().subVectors(p1, p0),
    new THREE.Vector3().subVectors(p2, p0),
  ).normalize();
  return v.clone().addScaledVector(n, -v.dot(n));
}

// ─────────────────────────────────────────────────────────────────────────────
// Trace a geodesic on the mesh
// ─────────────────────────────────────────────────────────────────────────────

export function traceGeodesic(
  mesh: HalfEdgeMesh,
  adj: FaceAdjacency,
  startFace: number,
  startPoint: THREE.Vector3,
  startDir: THREE.Vector3,
  maxLength: number,
  maxSteps = 2000,
): THREE.Vector3[] {
  const points: THREE.Vector3[] = [startPoint.clone()];
  let curFace  = startFace;
  let curPoint = startPoint.clone();
  let curDir   = projectToFace(mesh, curFace, startDir).normalize();

  if (curDir.lengthSq() < 1e-10) return points;

  let totalLength = 0;

  for (let step = 0; step < maxSteps && totalLength < maxLength; step++) {
    const face = mesh.faces[curFace];
    const vs = [mesh.vertices[face[0]], mesh.vertices[face[1]], mesh.vertices[face[2]]];

    // Find exit edge (smallest positive t)
    let bestT = Infinity;
    let bestEdge = -1;

    for (let e = 0; e < 3; e++) {
      const va = vs[e];
      const vb = vs[(e + 1) % 3];
      const t = rayEdgeIntersect(curPoint, curDir, va, vb);
      if (t !== null && t > 1e-9 && t < bestT) {
        bestT = t;
        bestEdge = e;
      }
    }

    if (bestEdge < 0) break;

    // Don't overshoot maxLength
    const remaining = maxLength - totalLength;
    const advance = Math.min(bestT, remaining);
    const exitPoint = curPoint.clone().addScaledVector(curDir, advance);
    points.push(exitPoint.clone());
    totalLength += advance;

    if (advance < bestT) break; // reached maxLength inside this triangle

    // Move to adjacent face
    const nextFace = adj.neighbors[3 * curFace + bestEdge];
    if (nextFace < 0) break; // boundary

    const edgeV0 = vs[bestEdge];
    const edgeV1 = vs[(bestEdge + 1) % 3];

    curDir   = unfoldDirection(mesh, curDir, nextFace, edgeV0, edgeV1);
    curPoint = exitPoint;
    curFace  = nextFace;
  }

  return points;
}

// ─────────────────────────────────────────────────────────────────────────────
// Find the face containing a point  (brute-force nearest-centroid lookup)
//
// For geodesic restart, we need to know which face a given 3D point lives on.
// We use the nearest face by centroid distance; sufficient when the point is
// known to lie on the mesh surface.
// ─────────────────────────────────────────────────────────────────────────────

export function findContainingFace(mesh: HalfEdgeMesh, pt: THREE.Vector3): number {
  let best = 0;
  let bestD2 = Infinity;
  for (let fi = 0; fi < mesh.faces.length; fi++) {
    const f = mesh.faces[fi];
    const cx = (mesh.vertices[f[0]].x + mesh.vertices[f[1]].x + mesh.vertices[f[2]].x) / 3;
    const cy = (mesh.vertices[f[0]].y + mesh.vertices[f[1]].y + mesh.vertices[f[2]].y) / 3;
    const cz = (mesh.vertices[f[0]].z + mesh.vertices[f[1]].z + mesh.vertices[f[2]].z) / 3;
    const dx = pt.x - cx, dy = pt.y - cy, dz = pt.z - cz;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 < bestD2) { bestD2 = d2; best = fi; }
  }
  return best;
}

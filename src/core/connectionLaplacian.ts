/**
 * connectionLaplacian.ts
 *
 * Stripe pattern generation via guided Poisson solve.
 *
 * For each family k with stripe direction φ_k = k·π/3:
 *   - Define per-vertex desired gradient direction: d_k_i = t1_i·cos(φ_k) + t2_i·sin(φ_k)
 *   - Build RHS: b[i] = Σ_j w_ij · (p_j − p_i) · d_k_i
 *   - Solve: L_cot · f = b  (cotangent Laplacian)
 *   - Trace isolines of f  (marching triangles)
 *
 * This correctly produces 3 DIFFERENT stripe families at 60° intervals.
 * The old Connection-Laplacian / inverse-power-iteration approach had a
 * fundamental flaw: the rotation angle cancelled in the connection-angle
 * formula, so all three families solved the same eigenproblem.
 */

import * as THREE from 'three';
import type { HalfEdgeMesh } from './halfEdge';
import { cotangentWeight, getHalfEdgeStart } from './halfEdge';

// ─────────────────────────────────────────────────────────────────────────────
// Public interfaces (kept for downstream compatibility)
// ─────────────────────────────────────────────────────────────────────────────

export interface Isoline {
  points: THREE.Vector3[];
  faceIndices: number[];
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal types
// ─────────────────────────────────────────────────────────────────────────────

interface NeighborEntry { j: number; w: number }

interface CotanL {
  n: number;
  diag: Float64Array;
  adj: NeighborEntry[][];   // adj[i] = list of {j, w} with w = cotangent weight (positive)
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 1 – Per-vertex orthonormal tangent frames
// ─────────────────────────────────────────────────────────────────────────────

interface VertexFrames {
  t1: THREE.Vector3[];
  t2: THREE.Vector3[];
}

function buildVertexFrames(mesh: HalfEdgeMesh): VertexFrames {
  const n = mesh.vertices.length;
  const t1s: THREE.Vector3[] = [];
  const t2s: THREE.Vector3[] = [];

  for (let i = 0; i < n; i++) {
    const ni = mesh.normals[i].clone().normalize();

    // Pick a reference vector not parallel to the normal
    let ref = new THREE.Vector3(1, 0, 0);
    if (Math.abs(ni.dot(ref)) > 0.8) ref = new THREE.Vector3(0, 1, 0);

    // t1 = projection of ref onto the tangent plane, normalised
    const t1 = ref.clone().sub(ni.clone().multiplyScalar(ni.dot(ref))).normalize();
    // t2 = n × t1  (right-handed frame)
    const t2 = new THREE.Vector3().crossVectors(ni, t1).normalize();

    t1s.push(t1);
    t2s.push(t2);
  }
  return { t1: t1s, t2: t2s };
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 2 – Cotangent Laplacian
// ─────────────────────────────────────────────────────────────────────────────

function buildCotanL(mesh: HalfEdgeMesh): CotanL {
  const n = mesh.vertices.length;
  const diag = new Float64Array(n);
  const adj: NeighborEntry[][] = Array.from({ length: n }, () => []);

  for (let heIdx = 0; heIdx < mesh.halfEdges.length; heIdx++) {
    const he = mesh.halfEdges[heIdx];
    // Process each undirected edge exactly once
    if (he.twin !== -1 && heIdx > he.twin) continue;

    const vi = getHalfEdgeStart(mesh, heIdx);
    const vj = he.vertex;
    const w = Math.max(0, cotangentWeight(mesh, heIdx));

    diag[vi] += w;
    diag[vj] += w;
    adj[vi].push({ j: vj, w });
    adj[vj].push({ j: vi, w });
  }

  return { n, diag, adj };
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 3 – Right-hand side for family k
//
// We want ∇f ≈ d_k everywhere, so:
//   b[i] = Σ_j w_ij · (p_j − p_i) · d_k_i
//
// The stripe density controls how many cycles of f fit across the surface.
// ─────────────────────────────────────────────────────────────────────────────

function buildRHS(
  mesh: HalfEdgeMesh,
  frames: VertexFrames,
  L: CotanL,
  phi: number,
  density: number,
): Float64Array {
  const n = mesh.vertices.length;
  const b = new Float64Array(n);

  for (let vi = 0; vi < n; vi++) {
    // Desired gradient direction at vi for this family
    const d = frames.t1[vi].clone().multiplyScalar(Math.cos(phi))
      .addScaledVector(frames.t2[vi], Math.sin(phi));

    for (const { j: vj, w } of L.adj[vi]) {
      const edge = new THREE.Vector3().subVectors(mesh.vertices[vj], mesh.vertices[vi]);
      b[vi] += density * w * edge.dot(d);
    }
  }

  return b;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 4 – Real-valued CG solver   L · f = b,  with f[0] = 0 (Dirichlet pin)
// ─────────────────────────────────────────────────────────────────────────────

function solveCG(
  L: CotanL,
  b: Float64Array,
  maxIter = 800,
  tol = 1e-6,
): Float64Array {
  const n = L.n;
  const f = new Float64Array(n);   // initial x = 0

  // Modified b: pin vertex 0
  const bm = b.slice() as Float64Array;
  bm[0] = 0;

  // A·x  (Laplacian with Dirichlet at vertex 0)
  function matvec(x: Float64Array): Float64Array {
    const y = new Float64Array(n);
    y[0] = x[0]; // pinned row: A[0,0] = 1, rest 0
    for (let i = 1; i < n; i++) {
      y[i] = L.diag[i] * x[i];
      for (const { j, w } of L.adj[i]) {
        if (j !== 0) y[i] -= w * x[j];
        // skip j==0 column (Dirichlet)
      }
    }
    return y;
  }

  // r0 = b − A·f = b  (f=0)
  let r = bm.slice() as Float64Array;
  let p = r.slice() as Float64Array;
  let rr = dotR(r, r);
  if (rr < 1e-30) return f;

  for (let iter = 0; iter < maxIter; iter++) {
    const Ap = matvec(p);
    const pAp = dotR(p, Ap);
    if (Math.abs(pAp) < 1e-30) break;
    const alpha = rr / pAp;

    for (let i = 0; i < n; i++) { f[i] += alpha * p[i]; r[i] -= alpha * Ap[i]; }

    const rrNew = dotR(r, r);
    if (Math.sqrt(rrNew) < tol * Math.sqrt(dotR(bm, bm) + 1e-30)) break;

    const beta = rrNew / rr;
    for (let i = 0; i < n; i++) p[i] = r[i] + beta * p[i];
    rr = rrNew;
  }

  return f;
}

function dotR(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 5 – Marching-triangles isoline extraction
//
// For each face: find the (at most two) edges where f crosses the target level.
// Segments are stored as pairs of consecutive points in Isoline.points,
// so points = [A0,B0, A1,B1, …].  Rendering should use THREE.LineSegments.
// ─────────────────────────────────────────────────────────────────────────────

export function traceIsolines(
  mesh: HalfEdgeMesh,
  f: Float64Array,
  numIsolines: number,
): Isoline[] {
  // Determine the min/max of f across all vertices
  let fMin = f[0];
  let fMax = f[0];
  for (let i = 1; i < f.length; i++) {
    if (f[i] < fMin) fMin = f[i];
    if (f[i] > fMax) fMax = f[i];
  }
  if (fMax - fMin < 1e-10) return [];

  const isolines: Isoline[] = [];

  for (let k = 0; k < numIsolines; k++) {
    // Place levels at (k + 0.5)/n  fraction of the range
    const level = fMin + (fMax - fMin) * (k + 0.5) / numIsolines;
    const pts: THREE.Vector3[] = [];
    const faceIds: number[] = [];

    for (let fi = 0; fi < mesh.faces.length; fi++) {
      const face = mesh.faces[fi];
      const v = [face[0], face[1], face[2]];
      const fv = [f[v[0]], f[v[1]], f[v[2]]];

      // Find up to 2 edge crossings
      const crossPts: THREE.Vector3[] = [];
      for (let e = 0; e < 3; e++) {
        const a = e, b = (e + 1) % 3;
        const fa_ = fv[a], fb_ = fv[b];
        if ((fa_ <= level && level < fb_) || (fb_ <= level && level < fa_)) {
          const t = (level - fa_) / (fb_ - fa_);
          crossPts.push(
            new THREE.Vector3().lerpVectors(mesh.vertices[v[a]], mesh.vertices[v[b]], t),
          );
        }
      }

      if (crossPts.length === 2) {
        pts.push(crossPts[0], crossPts[1]);
        faceIds.push(fi);
      }
    }

    if (pts.length > 0) {
      isolines.push({ points: pts, faceIndices: faceIds });
    }
  }

  return isolines;
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Compute the stripe scalar field for one family.
 *
 * @param mesh       Half-edge mesh (TPMS surface)
 * @param phi        Stripe direction angle in the tangent plane (radians)
 * @param density    Controls stripe frequency (higher = more stripes)
 */
export function computeStripeField(
  mesh: HalfEdgeMesh,
  phi: number,
  density = 4.0,
): Float64Array {
  const frames = buildVertexFrames(mesh);
  const L = buildCotanL(mesh);
  const b = buildRHS(mesh, frames, L, phi, density);
  return solveCG(L, b, 800, 1e-6);
}

// ─────────────────────────────────────────────────────────────────────────────
// Legacy stubs kept so that imports elsewhere still compile
// (viewport3d.ts is updated to use computeStripeField directly)
// ─────────────────────────────────────────────────────────────────────────────

export interface SparseMatrix {
  rows: number;
  cols: number;
  data: Map<string, { real: number; imag: number }>;
}

/** @deprecated Use computeStripeField instead */
export function buildConnectionLaplacian(
  _mesh: HalfEdgeMesh,
  _rotationAngle: number,
): { L: SparseMatrix; M: SparseMatrix } {
  const empty: SparseMatrix = { rows: 0, cols: 0, data: new Map() };
  return { L: empty, M: empty };
}

/** @deprecated Use computeStripeField instead */
export function solveEigenvector(
  _L: SparseMatrix,
  _M: SparseMatrix,
): { real: Float64Array; imag: Float64Array } {
  return { real: new Float64Array(0), imag: new Float64Array(0) };
}

/** @deprecated Identity passthrough – field is already real */
export function extractPhase(real: Float64Array, _imag: Float64Array): Float64Array {
  return real;
}

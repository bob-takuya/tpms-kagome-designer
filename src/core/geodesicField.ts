/**
 * geodesicField.ts
 *
 * Vekhter, Zhuo, Gil Fandino, Huang, Vouga (SIGGRAPH 2019)
 * "Weaving Geodesic Foliations" — Part 1 (face-based geodesic vector field).
 *
 * Variational formulation (eq. 4/5 in the paper):
 *
 *     min_{ŵ, δ}   (1/2) ‖δ‖²      s.t.  C(ŵ + δ) = 0,   ‖ŵ_f‖ = 1
 *
 * where C is the discrete curl operator acting on a face-based tangent
 * vector field ŵ : F → ℝ². The constraint Cv = 0 is equivalent to the
 * edge-wise condition v_i · e_ij = v_j · e_ij for every interior edge,
 * which is the Levi-Civita / holonomy test in discrete form.
 *
 * A unit-norm face field that satisfies C(ŵ)=0 exactly cannot exist on a
 * surface with non-zero angle defect anywhere (in particular on any TPMS,
 * which has K < 0 pointwise). The slack variable δ absorbs the residual.
 *
 * We run Algorithm 1 from the paper — alternating minimization:
 *
 *   repeat
 *     (A)  δ  ← argmin (1/2)‖δ‖²  s.t.  Cδ = −C ŵ         (linear projection)
 *     (B)  for each face f:
 *              ŵ_f ← (ŵ_f + δ_f) / ‖ŵ_f + δ_f‖            (unit normalize)
 *
 * Step (A) is solved via the normal-equation Schur complement: the Lagrange
 * multiplier μ ∈ ℝ^|E_int| satisfies
 *
 *     (C Cᵀ + ε I) μ = C ŵ     →     δ = −Cᵀ μ
 *
 * ε is a small Tikhonov term that handles the rank deficiency of Cᵀ on
 * topologically non-trivial surfaces (in particular TPMS, genus 3). We
 * solve the symmetric system with CG.
 *
 * This file intentionally implements only Part 1 of the paper. The
 * 6-fold branched cover / paired-cover rounding / MIP stage (Part 2)
 * is not implemented yet — the geodesic field computed here is fed to
 * the existing per-vertex Knöppel 2015 stripe eigensolve for scalar
 * integration, which handles each kagome family independently.
 */

import * as THREE from 'three';
import type { HalfEdgeMesh } from './halfEdge';
import { getHalfEdgeStart } from './halfEdge';

// ─────────────────────────────────────────────────────────────────────────────
// Face frames
// ─────────────────────────────────────────────────────────────────────────────

export interface FaceFrame {
  e1: THREE.Vector3;  // first in-plane basis
  e2: THREE.Vector3;  // second in-plane basis (n × e1)
  n:  THREE.Vector3;  // face normal
  area: number;
}

export function buildFaceFrames(mesh: HalfEdgeMesh): FaceFrame[] {
  const frames: FaceFrame[] = [];
  for (const face of mesh.faces) {
    const a = mesh.vertices[face[0]];
    const b = mesh.vertices[face[1]];
    const c = mesh.vertices[face[2]];
    const ab = new THREE.Vector3().subVectors(b, a);
    const ac = new THREE.Vector3().subVectors(c, a);
    const cross = new THREE.Vector3().crossVectors(ab, ac);
    const area = 0.5 * cross.length();
    const n = cross.clone().normalize();
    const e1 = ab.clone().normalize();
    const e2 = new THREE.Vector3().crossVectors(n, e1).normalize();
    frames.push({ e1, e2, n, area });
  }
  return frames;
}

// ─────────────────────────────────────────────────────────────────────────────
// Discrete curl operator C : ℝ^{2|F|} → ℝ^{|E_int|}
//
// For an interior edge with adjacent faces fi, fj and 3D edge vector e_ij,
//     (Cw)_e = w_i · e_ij − w_j · e_ij
//
// In the 2D face-local coordinates of each adjacent face, this becomes
//     (Cw)_e = (w_fi_x, w_fi_y) · (e_ij_fi)  −  (w_fj_x, w_fj_y) · (e_ij_fj)
//
// Each row has exactly 4 nonzero columns, two per adjacent face.
// ─────────────────────────────────────────────────────────────────────────────

export interface CurlRow {
  col: [number, number, number, number];  // 2*fi, 2*fi+1, 2*fj, 2*fj+1
  val: [number, number, number, number];
}

export interface CurlOperator {
  numRows: number;  // |E_int|
  numCols: number;  // 2|F|
  rows: CurlRow[];
}

export function buildCurlOperator(mesh: HalfEdgeMesh, frames: FaceFrame[]): CurlOperator {
  const rows: CurlRow[] = [];

  // Walk every interior edge exactly once using half-edge convention.
  for (let heIdx = 0; heIdx < mesh.halfEdges.length; heIdx++) {
    const he = mesh.halfEdges[heIdx];
    if (he.twin === -1) continue;
    if (heIdx > he.twin) continue;  // canonical orientation

    const fi = he.face;
    const fj = mesh.halfEdges[he.twin].face;
    const va = getHalfEdgeStart(mesh, heIdx);
    const vb = he.vertex;

    const e = new THREE.Vector3().subVectors(mesh.vertices[vb], mesh.vertices[va]);
    const Fi = frames[fi];
    const Fj = frames[fj];
    const eix = e.dot(Fi.e1), eiy = e.dot(Fi.e2);
    const ejx = e.dot(Fj.e1), ejy = e.dot(Fj.e2);

    rows.push({
      col: [2 * fi, 2 * fi + 1, 2 * fj, 2 * fj + 1],
      val: [eix, eiy, -ejx, -ejy],
    });
  }

  return { numRows: rows.length, numCols: 2 * mesh.faces.length, rows };
}

// ─────────────────────────────────────────────────────────────────────────────
// Sparse matvec helpers for C and Cᵀ
// ─────────────────────────────────────────────────────────────────────────────

function applyC(C: CurlOperator, w: Float64Array, out?: Float64Array): Float64Array {
  const y = out ?? new Float64Array(C.numRows);
  if (out) y.fill(0);
  for (let r = 0; r < C.numRows; r++) {
    const { col, val } = C.rows[r];
    y[r] = val[0] * w[col[0]] + val[1] * w[col[1]]
         + val[2] * w[col[2]] + val[3] * w[col[3]];
  }
  return y;
}

function applyCt(C: CurlOperator, r: Float64Array, out?: Float64Array): Float64Array {
  const y = out ?? new Float64Array(C.numCols);
  y.fill(0);
  for (let i = 0; i < C.numRows; i++) {
    const { col, val } = C.rows[i];
    const ri = r[i];
    y[col[0]] += val[0] * ri;
    y[col[1]] += val[1] * ri;
    y[col[2]] += val[2] * ri;
    y[col[3]] += val[3] * ri;
  }
  return y;
}

// ─────────────────────────────────────────────────────────────────────────────
// Solve (C Cᵀ + εI) μ = b via CG.
//
// The Tikhonov ε regularizes the rank deficiency of C on closed / high-genus
// surfaces. For TPMS (genus 3) CCᵀ has a non-trivial kernel of dimension ≥ 2g
// so regularization is essential.
// ─────────────────────────────────────────────────────────────────────────────

function solveCCt(
  C: CurlOperator,
  b: Float64Array,
  epsilon: number,
  maxIter: number,
  tol: number,
): Float64Array {
  const n = C.numRows;
  const mu = new Float64Array(n);
  const r  = Float64Array.from(b);
  const p  = Float64Array.from(r);
  const Ap = new Float64Array(n);
  const Ctp = new Float64Array(C.numCols);

  let rr = dot(r, r);
  if (rr < 1e-30) return mu;
  const bn2 = rr;

  for (let iter = 0; iter < maxIter; iter++) {
    // A p = C (Cᵀ p) + ε p
    applyCt(C, p, Ctp);
    applyC(C, Ctp, Ap);
    for (let i = 0; i < n; i++) Ap[i] += epsilon * p[i];

    const pAp = dot(p, Ap);
    if (Math.abs(pAp) < 1e-30) break;
    const alpha = rr / pAp;

    for (let i = 0; i < n; i++) {
      mu[i] += alpha * p[i];
      r[i]  -= alpha * Ap[i];
    }

    const rrNew = dot(r, r);
    if (rrNew < tol * tol * bn2) break;
    const beta = rrNew / rr;
    for (let i = 0; i < n; i++) p[i] = r[i] + beta * p[i];
    rr = rrNew;
  }
  return mu;
}

function dot(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// Initial field — 6-RoSy lift of the per-vertex 3-RoSy eigenvector to faces.
//
// Given the Knöppel 2013 vertex field w_v = exp(3 i θ_v) (already computed in
// connectionLaplacian.ts by inversePowerIteration of the 3-RoSy connection
// Laplacian), we want an initial unit vector per face.
//
// The trick to avoid line-field / 60° sign ambiguity is to work with the
// 6-RoSy lift: take the 2D tangent direction at each vertex, express it in
// the face-local frame, compute z_v = exp(6 i atan2(·)), average over the 3
// face vertices, then recover ŵ_f = (cos(arg(z_f)/6), sin(arg(z_f)/6)).
// Averaging z gives a canonical representative per face.
// ─────────────────────────────────────────────────────────────────────────────

export function initFieldFromVertex3RoSy(
  mesh: HalfEdgeMesh,
  frames: FaceFrame[],
  vertexT1: THREE.Vector3[],
  vertexT2: THREE.Vector3[],
  re: Float64Array,
  im: Float64Array,
): Float64Array {
  const F = mesh.faces.length;
  const w = new Float64Array(2 * F);

  for (let fi = 0; fi < F; fi++) {
    const face = mesh.faces[fi];
    const Fi = frames[fi];

    let zre = 0, zim = 0;
    for (let k = 0; k < 3; k++) {
      const v = face[k];
      // θ_v in vertex local frame (t1_v, t2_v)
      const theta_v = Math.atan2(im[v], re[v]) / 3;
      // 3D tangent direction at vertex v
      const dx = vertexT1[v].x * Math.cos(theta_v) + vertexT2[v].x * Math.sin(theta_v);
      const dy = vertexT1[v].y * Math.cos(theta_v) + vertexT2[v].y * Math.sin(theta_v);
      const dz = vertexT1[v].z * Math.cos(theta_v) + vertexT2[v].z * Math.sin(theta_v);
      // Project onto face plane: remove face-normal component.
      const dn = dx * Fi.n.x + dy * Fi.n.y + dz * Fi.n.z;
      const px = dx - dn * Fi.n.x;
      const py = dy - dn * Fi.n.y;
      const pz = dz - dn * Fi.n.z;
      // Face-local 2D coords
      const u = px * Fi.e1.x + py * Fi.e1.y + pz * Fi.e1.z;
      const v2 = px * Fi.e2.x + py * Fi.e2.y + pz * Fi.e2.z;
      const ang = Math.atan2(v2, u);
      // 6-RoSy lift
      zre += Math.cos(6 * ang);
      zim += Math.sin(6 * ang);
    }
    const phi = Math.atan2(zim, zre) / 6;
    w[2 * fi]     = Math.cos(phi);
    w[2 * fi + 1] = Math.sin(phi);
  }

  return w;
}

/**
 * Same as initFieldFromVertex3RoSy but returns only the canonical angle
 * per face, φ_f ∈ (−π/6, π/6]. Used by the branched-cover construction
 * (branchedCover.ts) to initialize the 6 layer angles as φ_f + k·π/3.
 */
export function computeFaceBaseAngle(
  mesh: HalfEdgeMesh,
  frames: FaceFrame[],
  vertexT1: THREE.Vector3[],
  vertexT2: THREE.Vector3[],
  re: Float64Array,
  im: Float64Array,
): Float64Array {
  const w = initFieldFromVertex3RoSy(mesh, frames, vertexT1, vertexT2, re, im);
  const F = mesh.faces.length;
  const angles = new Float64Array(F);
  for (let f = 0; f < F; f++) {
    angles[f] = Math.atan2(w[2 * f + 1], w[2 * f]);
  }
  return angles;
}

// ─────────────────────────────────────────────────────────────────────────────
// Algorithm 1 — alternating minimization (δ-projection + face normalization)
// ─────────────────────────────────────────────────────────────────────────────

export interface GeodesicFieldResult {
  w: Float64Array;          // 2|F| per face, unit norm
  initialCurl: number;      // ‖Cŵ₀‖² before optimization
  finalCurl: number;        // ‖Cŵ‖² after optimization (post-normalization residual)
  iterations: number;
}

export function optimizeGeodesicField(
  C: CurlOperator,
  w0: Float64Array,
  opts: {
    maxIter?: number;
    tol?: number;
    epsilon?: number;       // Tikhonov for CCᵀ solve
    cgMaxIter?: number;
    cgTol?: number;
  } = {},
): GeodesicFieldResult {
  const maxIter  = opts.maxIter  ?? 30;
  const tol      = opts.tol      ?? 1e-5;
  const epsilon  = opts.epsilon  ?? 1e-6;
  const cgMax    = opts.cgMaxIter ?? 300;
  const cgTol    = opts.cgTol    ?? 1e-7;

  const F = w0.length / 2;
  const w = Float64Array.from(w0);

  // initial curl norm (sanity / reporting)
  const Cw0 = applyC(C, w);
  const initialCurl = Math.sqrt(dot(Cw0, Cw0));

  let iter = 0;
  let finalCurl = initialCurl;

  for (iter = 0; iter < maxIter; iter++) {
    // Step A: δ-projection.
    // Solve (CCᵀ + εI) μ = C w   →   δ = −Cᵀ μ
    const Cw  = applyC(C, w);
    const mu  = solveCCt(C, Cw, epsilon, cgMax, cgTol);
    const Ctm = applyCt(C, mu);

    // Step B: per-face normalization
    let maxDelta = 0;
    for (let f = 0; f < F; f++) {
      const nx = w[2 * f]     - Ctm[2 * f];     // ŵ + δ = ŵ − Cᵀμ
      const ny = w[2 * f + 1] - Ctm[2 * f + 1];
      const len = Math.sqrt(nx * nx + ny * ny);
      const invLen = len > 1e-12 ? 1 / len : 0;
      const wxOld = w[2 * f],     wyOld = w[2 * f + 1];
      const wxNew = nx * invLen,  wyNew = ny * invLen;
      w[2 * f]     = wxNew;
      w[2 * f + 1] = wyNew;
      const dx = wxNew - wxOld, dy = wyNew - wyOld;
      const d2 = dx * dx + dy * dy;
      if (d2 > maxDelta) maxDelta = d2;
    }

    // Convergence check
    const Cw1 = applyC(C, w);
    finalCurl = Math.sqrt(dot(Cw1, Cw1));

    if (Math.sqrt(maxDelta) < tol) {
      iter++;
      break;
    }
  }

  return { w, initialCurl, finalCurl, iterations: iter };
}

// ─────────────────────────────────────────────────────────────────────────────
// Convert face-based ŵ back to per-vertex direction angles (in the vertex
// tangent frames expected by computeKnoppelStripeFields).
//
// For each vertex v:
//   1. For every incident face f, lift ŵ_f to 3D:  d_f = ŵ_f.x·e1_f + ŵ_f.y·e2_f
//   2. Take the 2-RoSy average (project signs via exp(2iφ) so that ±d cancel
//      to the same representative) in the vertex tangent frame (t1_v, t2_v).
//   3. Recover θ_v = arg(z_v)/2.
//
// 2-RoSy (line field) is sufficient here because the downstream Knöppel
// stripe eigensolve is invariant under direction negation, and Algorithm 1
// may still have incompatible signs across edges.
// ─────────────────────────────────────────────────────────────────────────────

export function faceFieldToVertexAngles(
  mesh: HalfEdgeMesh,
  frames: FaceFrame[],
  vertexT1: THREE.Vector3[],
  vertexT2: THREE.Vector3[],
  w: Float64Array,
): Float64Array {
  const nv = mesh.vertices.length;
  const zre = new Float64Array(nv);
  const zim = new Float64Array(nv);
  const deg = new Uint32Array(nv);

  for (let fi = 0; fi < mesh.faces.length; fi++) {
    const face = mesh.faces[fi];
    const Fi = frames[fi];
    const wx = w[2 * fi], wy = w[2 * fi + 1];
    // Face-field in 3D
    const dx = wx * Fi.e1.x + wy * Fi.e2.x;
    const dy = wx * Fi.e1.y + wy * Fi.e2.y;
    const dz = wx * Fi.e1.z + wy * Fi.e2.z;

    for (let k = 0; k < 3; k++) {
      const v = face[k];
      // Project onto vertex tangent plane using its own t1/t2.
      const u = dx * vertexT1[v].x + dy * vertexT1[v].y + dz * vertexT1[v].z;
      const s = dx * vertexT2[v].x + dy * vertexT2[v].y + dz * vertexT2[v].z;
      const ang = Math.atan2(s, u);
      zre[v] += Math.cos(2 * ang);
      zim[v] += Math.sin(2 * ang);
      deg[v] += 1;
    }
  }

  const theta = new Float64Array(nv);
  for (let v = 0; v < nv; v++) {
    if (deg[v] === 0) { theta[v] = 0; continue; }
    theta[v] = Math.atan2(zim[v], zre[v]) / 2;
  }
  return theta;
}

// ─────────────────────────────────────────────────────────────────────────────
// Build per-vertex orthonormal tangent frames from smoothed vertex normals.
// Duplicated here to avoid a circular import; matches connectionLaplacian.ts.
// ─────────────────────────────────────────────────────────────────────────────

export function buildVertexTangentFrames(
  mesh: HalfEdgeMesh,
): { t1: THREE.Vector3[]; t2: THREE.Vector3[] } {
  const n = mesh.vertices.length;
  const t1: THREE.Vector3[] = [];
  const t2: THREE.Vector3[] = [];
  for (let i = 0; i < n; i++) {
    const ni = mesh.normals[i].clone().normalize();
    let ref = new THREE.Vector3(1, 0, 0);
    if (Math.abs(ni.dot(ref)) > 0.8) ref = new THREE.Vector3(0, 1, 0);
    const a = ref.clone().sub(ni.clone().multiplyScalar(ni.dot(ref))).normalize();
    const b = new THREE.Vector3().crossVectors(ni, a).normalize();
    t1.push(a);
    t2.push(b);
  }
  return { t1, t2 };
}

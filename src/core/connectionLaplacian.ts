/**
 * connectionLaplacian.ts
 *
 * Stripe pattern generation via guided Poisson solve + optional Connection
 * Laplacian singularity detection.
 *
 * Stripe generation (3 families at 60° intervals):
 *   For each family k with stripe direction φ_k = k·π/3:
 *     - Define per-vertex desired gradient direction: d_k_i = t1_i·cos(φ_k) + t2_i·sin(φ_k)
 *     - Build RHS: b[i] = Σ_j w_ij · (p_j − p_i) · d_k_i
 *     - Solve: L_cot · f = b  (cotangent Laplacian)
 *     - Trace isolines of f  (marching triangles)
 *
 * Singularity detection (for future branching/short-strip support):
 *   - Build Connection Laplacian with discrete Levi-Civita connection angles
 *   - Find smallest eigenvector ψ ∈ ℂ^n via inverse power iteration
 *   - Singularities at |ψ| ≈ 0 indicate where stripes naturally branch
 */

import * as THREE from 'three';
import type { HalfEdgeMesh } from './halfEdge';
import { cotangentWeight, getHalfEdgeStart } from './halfEdge';

// ─────────────────────────────────────────────────────────────────────────────
// Public interfaces
// ─────────────────────────────────────────────────────────────────────────────

export interface Isoline {
  points: THREE.Vector3[];
  faceIndices: number[];
  /** Family index this chain belongs to (0..2). Set by importers that
   *  split per-family buckets into per-chain entries. -1 if unknown. */
  family?: number;
  /** Iso-level index within the family (v3+ wgf-output). -1 if unknown. */
  isoLevel?: number;
  /** Native chainId from the WGF tracer (v2+ wgf-output). -1 if unknown. */
  chainId?: number;
  /** Per-chain classification flag (v4+):
   *  0=normal, 1=cut chain, 2=fast-path chain, -1 if unknown. */
  flag?: number;
}

export interface StripeFieldResult {
  phase:         Float64Array;
  amplitude:     Float64Array;
  singularities: number[];
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal types
// ─────────────────────────────────────────────────────────────────────────────

interface VertexFrames {
  t1: THREE.Vector3[];
  t2: THREE.Vector3[];
}

interface NeighborEntry { j: number; w: number }

interface CotanL {
  n: number;
  diag: Float64Array;
  adj: NeighborEntry[][];
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-vertex orthonormal tangent frames
// ─────────────────────────────────────────────────────────────────────────────

function buildVertexFrames(mesh: HalfEdgeMesh): VertexFrames {
  const n = mesh.vertices.length;
  const t1s: THREE.Vector3[] = [];
  const t2s: THREE.Vector3[] = [];

  for (let i = 0; i < n; i++) {
    const ni = mesh.normals[i].clone().normalize();
    let ref = new THREE.Vector3(1, 0, 0);
    if (Math.abs(ni.dot(ref)) > 0.8) ref = new THREE.Vector3(0, 1, 0);
    const t1 = ref.clone().sub(ni.clone().multiplyScalar(ni.dot(ref))).normalize();
    const t2 = new THREE.Vector3().crossVectors(ni, t1).normalize();
    t1s.push(t1);
    t2s.push(t2);
  }
  return { t1: t1s, t2: t2s };
}

// ─────────────────────────────────────────────────────────────────────────────
// Cotangent Laplacian
// ─────────────────────────────────────────────────────────────────────────────

function buildCotanL(mesh: HalfEdgeMesh): CotanL {
  const n = mesh.vertices.length;
  const diag = new Float64Array(n);
  const adj: NeighborEntry[][] = Array.from({ length: n }, () => []);

  for (let heIdx = 0; heIdx < mesh.halfEdges.length; heIdx++) {
    const he = mesh.halfEdges[heIdx];
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
// Right-hand side for family k  (guided Poisson solve)
//
// Supports two modes:
//   (a) Fixed angle φ in every local frame (original comb-frame approach)
//   (b) Per-vertex direction angles from the Connection Laplacian eigenvector
//       (globally smooth direction field)
// ─────────────────────────────────────────────────────────────────────────────

function buildRHS(
  mesh: HalfEdgeMesh,
  frames: VertexFrames,
  L: CotanL,
  directionAngles: Float64Array | number, // per-vertex angle OR uniform angle
  density: number,
): Float64Array {
  const n = mesh.vertices.length;
  const b = new Float64Array(n);
  const uniform = typeof directionAngles === 'number';

  for (let vi = 0; vi < n; vi++) {
    const angle = uniform ? directionAngles : directionAngles[vi];
    const d = frames.t1[vi].clone().multiplyScalar(Math.cos(angle))
      .addScaledVector(frames.t2[vi], Math.sin(angle));

    for (const { j: vj, w } of L.adj[vi]) {
      const edge = new THREE.Vector3().subVectors(mesh.vertices[vj], mesh.vertices[vi]);
      b[vi] += density * w * edge.dot(d);
    }
  }

  return b;
}

// ─────────────────────────────────────────────────────────────────────────────
// Connected component detection (Dirichlet pins for CG)
// ─────────────────────────────────────────────────────────────────────────────

function findComponentPins(L: CotanL): number[] {
  const visited = new Uint8Array(L.n);
  const pins: number[] = [];
  for (let start = 0; start < L.n; start++) {
    if (visited[start]) continue;
    pins.push(start);
    const stack = [start];
    while (stack.length) {
      const v = stack.pop()!;
      if (visited[v]) continue;
      visited[v] = 1;
      for (const { j } of L.adj[v]) if (!visited[j]) stack.push(j);
    }
  }
  return pins;
}

// ─────────────────────────────────────────────────────────────────────────────
// Real CG solver  L·f = b  with Dirichlet pins
// ─────────────────────────────────────────────────────────────────────────────

function solveCG(
  L: CotanL,
  b: Float64Array,
  maxIter = 800,
  tol = 1e-6,
): Float64Array {
  const n = L.n;
  const f = new Float64Array(n);
  const pins = new Set(findComponentPins(L));

  const bm = Float64Array.from(b);
  for (const p of pins) bm[p] = 0;

  function matvec(x: Float64Array): Float64Array {
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      if (pins.has(i)) { y[i] = x[i]; continue; }
      y[i] = L.diag[i] * x[i];
      for (const { j, w } of L.adj[i]) {
        if (!pins.has(j)) y[i] -= w * x[j];
      }
    }
    return y;
  }

  let r = Float64Array.from(bm);
  let p = Float64Array.from(r);
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
// Marching-triangles isoline extraction (standard scalar field)
// ─────────────────────────────────────────────────────────────────────────────

export function traceIsolines(
  mesh: HalfEdgeMesh,
  f: Float64Array,
  numIsolines: number,
): Isoline[] {
  const sorted = Array.from(f).sort((a, b) => a - b);
  const n = sorted.length;
  const fMin = sorted[Math.max(0,     Math.floor(0.02 * n))];
  const fMax = sorted[Math.min(n - 1, Math.floor(0.98 * n))];
  if (fMax - fMin < 1e-10) return [];

  const isolines: Isoline[] = [];

  for (let k = 0; k < numIsolines; k++) {
    const level = fMin + (fMax - fMin) * (k + 0.5) / numIsolines;
    const pts: THREE.Vector3[] = [];
    const faceIds: number[] = [];

    for (let fi = 0; fi < mesh.faces.length; fi++) {
      const face = mesh.faces[fi];
      const v = [face[0], face[1], face[2]];
      const fv = [f[v[0]], f[v[1]], f[v[2]]];

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
// Public entry point – stripe field via guided Poisson solve
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Compute a stripe field for one family (backward-compatible API).
 * Uses the comb-frame approach with a fixed angle φ in every local frame.
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
// Connection Laplacian eigenvector-guided stripe generation
//
// 1. Solve the Connection Laplacian eigenproblem to find the smoothest
//    1-direction field on the surface (globally optimal, handles topology).
// 2. Use the eigenvector's phase θ_i as the base direction at each vertex.
// 3. For each kagome family k = 0,1,2: rotate by k·60° → Poisson RHS.
// 4. Solve 3 Poisson problems with these improved per-vertex directions.
//
// Advantage over the comb-frame approach: the direction field is globally
// smooth (no frame discontinuities), and singularities are detected via |ψ|.
// ─────────────────────────────────────────────────────────────────────────────

export interface GuidedStripeResult {
  fields:        [Float64Array, Float64Array, Float64Array];
  amplitude:     Float64Array;
  singularities: number[];
}

/**
 * Compute all 3 kagome stripe fields using the Connection Laplacian
 * eigenvector for globally smooth direction guidance.
 */
export function computeGuidedStripeFields(
  mesh: HalfEdgeMesh,
  density = 4.0,
): GuidedStripeResult {
  const frames = buildVertexFrames(mesh);
  const areas  = computeVertexAreas(mesh);
  const realL  = buildCotanL(mesh);

  // Step 1: Connection Laplacian eigenvector → optimal direction field
  console.time('[Stripe] connection eigenvector');
  const cplxL = buildComplexCotanL(mesh, frames);
  const { re, im } = inversePowerIteration(cplxL, areas);
  console.timeEnd('[Stripe] connection eigenvector');

  // Extract per-vertex 3-RoSy direction θ = arg(w) / 3  and amplitude
  // θ ∈ [−π/3, π/3] (= [−60°, +60°]).  Since the field has 3-fold symmetry,
  // this uniquely determines the stripe direction modulo 60°.
  const n = mesh.vertices.length;
  const eigenDir  = new Float64Array(n);
  const amplitude = new Float64Array(n);
  let maxAmp = 0;
  for (let i = 0; i < n; i++) {
    eigenDir[i]  = Math.atan2(im[i], re[i]) / 3;
    amplitude[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
    if (amplitude[i] > maxAmp) maxAmp = amplitude[i];
  }

  // Detect singularities (vertices where |ψ| ≈ 0)
  const ampThresh = 0.01 * maxAmp;
  const singularities: number[] = [];
  for (let i = 0; i < n; i++) {
    if (amplitude[i] < ampThresh) singularities.push(i);
  }

  // Phase 5 (spec): detect per-face singularities via 3-RoSy holonomy.
  // For face (i,j,k), sum the wrapped phase jumps around the loop:
  //   ϕ_ij = arg(w[j] · e^{−3iα_ij} · conj(w[i])) / 3
  // holonomy Σϕ rounded to multiples of 2π/3 gives the index ∈ {0, ±1/3}.
  const faceSingIndex = new Int8Array(mesh.faces.length);
  let nSingFaces = 0;
  const ALPHA_IJ = (vi: number, vj: number): number => {
    // Look up precomputed 3-α between vi and vj (use neighbor table)
    for (const nb of cplxL.adj[vi]) {
      if (nb.j === vj) return Math.atan2(nb.sinR, nb.cosR); // 3α_ij
    }
    return 0;
  };
  for (let fi = 0; fi < mesh.faces.length; fi++) {
    const f = mesh.faces[fi];
    let total = 0;
    for (let e = 0; e < 3; e++) {
      const a = f[e], b = f[(e + 1) % 3];
      const alpha3 = ALPHA_IJ(a, b);
      // arg(w[b] / (e^{3iα} · w[a])) in one step:
      //   w[b] · e^{−3iα} · conj(w[a])
      const wbRe = re[b], wbIm = im[b];
      const waRe = re[a], waIm = im[a];
      // e^{−3iα}
      const cR = Math.cos(-alpha3), cI = Math.sin(-alpha3);
      // e^{−3iα} · conj(w[a]) = (cR + i·cI)(waRe − i·waIm)
      const cRc = cR * waRe + cI * waIm;
      const cIc = cI * waRe - cR * waIm;
      // w[b] · that = (wbRe + i·wbIm)(cRc + i·cIc)
      const pR = wbRe * cRc - wbIm * cIc;
      const pI = wbRe * cIc + wbIm * cRc;
      // Wrapped phase jump
      total += Math.atan2(pI, pR);
    }
    // Wrap to (−π, π]
    total = total - 2 * Math.PI * Math.round(total / (2 * Math.PI));
    const idx = Math.round(total / (2 * Math.PI / 3));
    if (idx !== 0) {
      faceSingIndex[fi] = idx as -1 | 0 | 1;
      nSingFaces++;
    }
  }

  console.log(
    `[Stripe] 3-RoSy eigenvector: ` +
    `${singularities.length} vertex zeros (${(100 * singularities.length / n).toFixed(1)}%), ` +
    `${nSingFaces} face singularities`,
  );

  // Step 2: For each family k, build Poisson RHS using eigenvector direction + k·60°
  const fields: [Float64Array, Float64Array, Float64Array] = [
    new Float64Array(0), new Float64Array(0), new Float64Array(0),
  ];

  for (let k = 0; k < 3; k++) {
    const dirAngles = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      dirAngles[i] = eigenDir[i] + k * Math.PI / 3;
    }

    console.time(`[Stripe] Poisson family ${k}`);
    const b = buildRHS(mesh, frames, realL, dirAngles, density);
    fields[k] = solveCG(realL, b, 800, 1e-6);
    console.timeEnd(`[Stripe] Poisson family ${k}`);
  }

  return { fields, amplitude, singularities };
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-mesh cache for frames + vertex areas (avoids rebuilding)
// ─────────────────────────────────────────────────────────────────────────────

let _cache: { mesh: HalfEdgeMesh; frames: VertexFrames; areas: Float64Array } | null = null;

function ensureCache(mesh: HalfEdgeMesh): { frames: VertexFrames; areas: Float64Array } {
  if (_cache?.mesh === mesh) return _cache;
  const frames = buildVertexFrames(mesh);
  const areas  = computeVertexAreas(mesh);
  _cache = { mesh, frames, areas };
  return { frames, areas };
}

// ─────────────────────────────────────────────────────────────────────────────
// Knöppel et al. 2015 "Stripe Patterns on Surfaces" — proper eigenvalue formulation
//
// For each kagome family k, compute a complex scalar ψ_k minimizing
//     E(ψ) = ∫|∇ψ − iω·X_k·ψ|² dA
// where X_k is the direction field for family k and ω is the stripe frequency.
//
// The discrete form is a Hermitian generalized eigenproblem:
//     L_k ψ = λ M ψ
// with edge-wise twisted Laplacian:
//     L[i,i] += w_ij
//     L[j,j] += w_ij
//     L[i,j] -= w_ij · e^{−i·ω·⟨e_ij, X⟩_avg}
//     L[j,i] -= w_ij · e^{+i·ω·⟨e_ij, X⟩_avg}
//
// Stripes are extracted as zero-crossings of Re(ψ_k) per face (see traceZeroCrossings).
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Knöppel 2015 stripe pattern computation.
 *
 * Returns THREE complex fields (Re, Im, amplitude per vertex) — one per
 * kagome family.  Stripes are extracted from Re(ψ) zero-crossings by
 * traceZeroCrossings; low-amplitude faces (|ψ| ≈ 0) are singularities
 * and should be skipped.
 *
 * @param omega  Stripe frequency (radians per world unit along X).
 *               Higher = more stripes.  Typical: 3-6 for TPMS at grid=40.
 */
export function computeKnoppelStripeFields(
  mesh: HalfEdgeMesh,
  omega: number = 4.0,
): { re: Float64Array; im: Float64Array; amplitude: Float64Array }[] {
  const { frames, areas } = ensureCache(mesh);

  // Step 1: 3-RoSy eigenvector to get direction field
  const cplxL = buildComplexCotanL(mesh, frames);
  console.time('[Knoppel] 3-RoSy direction field');
  const eig = inversePowerIteration(cplxL, areas);
  console.timeEnd('[Knoppel] 3-RoSy direction field');

  const n = mesh.vertices.length;
  const eigenDir = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    eigenDir[i] = Math.atan2(eig.im[i], eig.re[i]) / 3;
  }

  return computeKnoppelStripeFieldsFromDirection(mesh, omega, eigenDir);
}

/**
 * Knöppel 2015 stripe pattern solve, but with an externally supplied
 * per-vertex base direction angle (measured in the same vertex tangent
 * frames that `buildVertexFrames` produces).
 *
 * This is the hook used by the Vekhter 2019 "Weaving Geodesic Foliations"
 * pipeline: the direction field comes from the Algorithm-1 curl-minimizing
 * face field, lifted back to vertex angles, instead of the raw 3-RoSy
 * eigenvector. The three kagome families are then θ, θ+60°, θ+120°.
 */
export function computeKnoppelStripeFieldsFromDirection(
  mesh: HalfEdgeMesh,
  omega: number,
  baseDir: Float64Array,    // per-vertex direction angle in vertex frame
): { re: Float64Array; im: Float64Array; amplitude: Float64Array }[] {
  const { frames, areas } = ensureCache(mesh);

  const n = mesh.vertices.length;
  if (baseDir.length !== n) {
    throw new Error(`baseDir length ${baseDir.length} ≠ vertex count ${n}`);
  }

  const results: { re: Float64Array; im: Float64Array; amplitude: Float64Array }[] = [];
  for (let k = 0; k < 3; k++) {
    const dirAngles = new Float64Array(n);
    for (let i = 0; i < n; i++) dirAngles[i] = baseDir[i] + k * Math.PI / 3;

    console.time(`[Knoppel/geodesic] family ${k} twisted eigensolve`);
    const L_k = buildKnoppelMatrix(mesh, frames, dirAngles, omega);
    const psi = inversePowerIteration(L_k, areas);
    console.timeEnd(`[Knoppel/geodesic] family ${k} twisted eigensolve`);

    const amp = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      amp[i] = Math.sqrt(psi.re[i] * psi.re[i] + psi.im[i] * psi.im[i]);
    }

    results.push({ re: psi.re, im: psi.im, amplitude: amp });
  }

  return results;
}

/**
 * Expose the 3-RoSy connection-Laplacian eigenvector used internally by
 * computeKnoppelStripeFields, so that other modules (the geodesic-field
 * optimizer) can use it as an initial guess without re-solving.
 *
 * Returns the raw complex entries w_v = exp(3 i θ_v) per vertex.
 */
export function computeVertex3RoSyEigenvector(
  mesh: HalfEdgeMesh,
): { re: Float64Array; im: Float64Array } {
  const { frames, areas } = ensureCache(mesh);
  const cplxL = buildComplexCotanL(mesh, frames);
  return inversePowerIteration(cplxL, areas);
}

/**
 * Knöppel 2015 stripe eigensolve for a SINGLE family, given a direction
 * angle per vertex. Used by the branched-cover pipeline: each layer of
 * the 6-fold cover produces its own per-vertex direction, fed here to
 * obtain that layer's scalar stripe field ψ.
 */
export function computeKnoppelStripeFieldForFamily(
  mesh: HalfEdgeMesh,
  omega: number,
  dirAngles: Float64Array,
): { re: Float64Array; im: Float64Array; amplitude: Float64Array } {
  const { frames, areas } = ensureCache(mesh);
  const L_k = buildKnoppelMatrix(mesh, frames, dirAngles, omega);
  const psi = inversePowerIteration(L_k, areas);
  const n = mesh.vertices.length;
  const amp = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    amp[i] = Math.sqrt(psi.re[i] * psi.re[i] + psi.im[i] * psi.im[i]);
  }
  return { re: psi.re, im: psi.im, amplitude: amp };
}

/**
 * Build the twisted Laplacian for Knöppel 2015 stripe energy.
 *
 * dirAngles[v] = angle of the desired stripe direction in the local frame at vertex v.
 */
function buildKnoppelMatrix(
  mesh: HalfEdgeMesh,
  frames: VertexFrames,
  dirAngles: Float64Array,
  omega: number,
): ComplexCotanL {
  const n = mesh.vertices.length;
  const diag = new Float64Array(n);
  const adj: ComplexNeighbor[][] = Array.from({ length: n }, () => []);

  for (let heIdx = 0; heIdx < mesh.halfEdges.length; heIdx++) {
    const he = mesh.halfEdges[heIdx];
    if (he.twin !== -1 && heIdx > he.twin) continue;

    const vi = getHalfEdgeStart(mesh, heIdx);
    const vj = he.vertex;
    const w = Math.max(0, cotangentWeight(mesh, heIdx));

    // Direction X_i and X_j in 3D via local tangent frames
    const Xi_x = Math.cos(dirAngles[vi]);
    const Xi_y = Math.sin(dirAngles[vi]);
    const Xj_x = Math.cos(dirAngles[vj]);
    const Xj_y = Math.sin(dirAngles[vj]);

    const t1i = frames.t1[vi], t2i = frames.t2[vi];
    const t1j = frames.t1[vj], t2j = frames.t2[vj];
    const Xi = new THREE.Vector3(
      t1i.x * Xi_x + t2i.x * Xi_y,
      t1i.y * Xi_x + t2i.y * Xi_y,
      t1i.z * Xi_x + t2i.z * Xi_y,
    );
    const Xj = new THREE.Vector3(
      t1j.x * Xj_x + t2j.x * Xj_y,
      t1j.y * Xj_x + t2j.y * Xj_y,
      t1j.z * Xj_x + t2j.z * Xj_y,
    );

    // Edge vector
    const e = new THREE.Vector3().subVectors(mesh.vertices[vj], mesh.vertices[vi]);

    // Projected lengths onto X at each endpoint
    const proj_i = e.dot(Xi);
    const proj_j = e.dot(Xj);

    // Line field sign handling: if the two projections disagree in sign,
    // flip one (the field is undirected; pick the side that gives consistent sign).
    let avgProj: number;
    if (proj_i * proj_j < 0) {
      avgProj = (Math.abs(proj_i) + Math.abs(proj_j)) / 2
                * (proj_i + proj_j >= 0 ? 1 : -1);
    } else {
      avgProj = (proj_i + proj_j) / 2;
    }

    const targetPhase = omega * avgProj;

    diag[vi] += w;
    diag[vj] += w;

    // L[i,j] = −w · e^{−i·targetPhase}
    // L[j,i] = −w · e^{+i·targetPhase}   (makes the matrix Hermitian)
    adj[vi].push({ j: vj, w, cosR: Math.cos(-targetPhase), sinR: Math.sin(-targetPhase) });
    adj[vj].push({ j: vi, w, cosR: Math.cos( targetPhase), sinR: Math.sin( targetPhase) });
  }

  return { n, diag, adj };
}

/**
 * Trace zero-crossings of a real scalar field on a triangle mesh.
 *
 * For each face where sign(f) changes across edges, emit a line segment
 * connecting the two zero-crossing points.  This is marching triangles
 * at level = 0, used for extracting stripes from Re(ψ) in the Knöppel pipeline.
 *
 * If `amplitude` is provided, faces where min(|ψ|) < ampRatio * max(|ψ|)
 * are skipped — these are singularities where the stripe direction is
 * undefined and zero-crossings produce spurious U-turns.
 */
export function traceZeroCrossings(
  mesh: HalfEdgeMesh,
  f: Float64Array,
  amplitude?: Float64Array,
  ampRatio: number = 0.10,
): Isoline[] {
  const pts: THREE.Vector3[] = [];
  const faceIds: number[] = [];

  let ampThresh = -Infinity;
  if (amplitude) {
    let maxAmp = 0;
    for (let i = 0; i < amplitude.length; i++) {
      if (amplitude[i] > maxAmp) maxAmp = amplitude[i];
    }
    ampThresh = ampRatio * maxAmp;
  }

  for (let fi = 0; fi < mesh.faces.length; fi++) {
    const face = mesh.faces[fi];
    const v = [face[0], face[1], face[2]];

    // Skip singular faces (|ψ| ≈ 0 at any vertex)
    if (amplitude) {
      const minAmp = Math.min(amplitude[v[0]], amplitude[v[1]], amplitude[v[2]]);
      if (minAmp < ampThresh) continue;
    }

    const fv = [f[v[0]], f[v[1]], f[v[2]]];
    const crossPts: THREE.Vector3[] = [];
    for (let e = 0; e < 3; e++) {
      const a = e, b = (e + 1) % 3;
      const fa_ = fv[a], fb_ = fv[b];
      if ((fa_ <= 0 && 0 < fb_) || (fb_ <= 0 && 0 < fa_)) {
        const t = (0 - fa_) / (fb_ - fa_);
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

  return pts.length > 0 ? [{ points: pts, faceIndices: faceIds }] : [];
}

// ─────────────────────────────────────────────────────────────────────────────
// Connection Laplacian internals
// ─────────────────────────────────────────────────────────────────────────────

interface ComplexNeighbor {
  j: number; w: number; cosR: number; sinR: number;
}

interface ComplexCotanL {
  n: number; diag: Float64Array; adj: ComplexNeighbor[][];
}

function connectionAngle(
  mesh: HalfEdgeMesh, frames: VertexFrames, vi: number, vj: number,
): number {
  const edge = new THREE.Vector3().subVectors(mesh.vertices[vj], mesh.vertices[vi]);
  const ni = mesh.normals[vi];
  const ei = edge.clone().addScaledVector(ni, -edge.dot(ni));
  const alphaI = Math.atan2(ei.dot(frames.t2[vi]), ei.dot(frames.t1[vi]));
  const nj = mesh.normals[vj];
  const ej = edge.clone().addScaledVector(nj, -edge.dot(nj));
  const alphaJ = Math.atan2(ej.dot(frames.t2[vj]), ej.dot(frames.t1[vj]));
  return alphaJ - alphaI;
}

/**
 * Build the 3-RoSy connection Laplacian.
 *
 * For the complex variable w[v] = exp(3i·θ[v]), the Dirichlet energy
 *   E = Σ_{(i,j)∈E} w_ij · |w[j] − e^{3i·α_ij} · w[i]|²
 * yields the Hermitian matrix:
 *   L[i,i] = Σ_j w_ij
 *   L[i,j] = −w_ij · e^{+3i·α_ij}   (stored in adj[vi] as the vi→vj entry)
 *   L[j,i] = −w_ij · e^{−3i·α_ij}   (stored in adj[vj] as the vj→vi entry)
 *
 * The factor of 3 encodes the 3-RoSy symmetry: rotating θ by 2π/3 leaves
 * the field unchanged.  The 3 kagome families are then at θ, θ+π/3, θ+2π/3.
 */
function buildComplexCotanL(mesh: HalfEdgeMesh, frames: VertexFrames): ComplexCotanL {
  const n = mesh.vertices.length;
  const diag = new Float64Array(n);
  const adj: ComplexNeighbor[][] = Array.from({ length: n }, () => []);

  for (let heIdx = 0; heIdx < mesh.halfEdges.length; heIdx++) {
    const he = mesh.halfEdges[heIdx];
    if (he.twin !== -1 && heIdx > he.twin) continue;
    const vi = getHalfEdgeStart(mesh, heIdx);
    const vj = he.vertex;
    const w = Math.max(0, cotangentWeight(mesh, heIdx));
    diag[vi] += w;
    diag[vj] += w;

    // 3-RoSy: multiply the connection angle by 3
    const omega3 = 3 * connectionAngle(mesh, frames, vi, vj);
    adj[vi].push({ j: vj, w, cosR: Math.cos( omega3), sinR: Math.sin( omega3) });
    adj[vj].push({ j: vi, w, cosR: Math.cos(-omega3), sinR: Math.sin(-omega3) });
  }

  return { n, diag, adj };
}

function computeVertexAreas(mesh: HalfEdgeMesh): Float64Array {
  const areas = new Float64Array(mesh.vertices.length);
  for (const face of mesh.faces) {
    const area = new THREE.Vector3().crossVectors(
      new THREE.Vector3().subVectors(mesh.vertices[face[1]], mesh.vertices[face[0]]),
      new THREE.Vector3().subVectors(mesh.vertices[face[2]], mesh.vertices[face[0]]),
    ).length() * 0.5 / 3;
    areas[face[0]] += area;
    areas[face[1]] += area;
    areas[face[2]] += area;
  }
  return areas;
}

// ─────────────────────────────────────────────────────────────────────────────
// Inverse power iteration for the smallest eigenvector of the
// complex connection Laplacian  (L + σM)ψ = Mψ_old
// ─────────────────────────────────────────────────────────────────────────────

function inversePowerIteration(
  L: ComplexCotanL,
  M: Float64Array,
  outerIter = 8,
  cgIter = 300,
): { re: Float64Array; im: Float64Array } {
  const n = L.n;
  const SIGMA = 1e-4;

  let psiRe: Float64Array = new Float64Array(n);
  let psiIm: Float64Array = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const a = (i * 2654435761) % (2 * Math.PI);
    psiRe[i] = Math.cos(a);
    psiIm[i] = Math.sin(a);
  }
  normC(psiRe, psiIm, M);

  for (let iter = 0; iter < outerIter; iter++) {
    const bRe = new Float64Array(n);
    const bIm = new Float64Array(n);
    for (let i = 0; i < n; i++) { bRe[i] = M[i] * psiRe[i]; bIm[i] = M[i] * psiIm[i]; }

    const xRe = new Float64Array(n);
    const xIm = new Float64Array(n);
    let rRe = Float64Array.from(bRe);
    let rIm = Float64Array.from(bIm);
    let pRe = Float64Array.from(rRe);
    let pIm = Float64Array.from(rIm);
    let rr = d2(rRe, rIm, rRe, rIm);

    for (let cg = 0; cg < cgIter; cg++) {
      const ApRe = new Float64Array(n);
      const ApIm = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const dd = L.diag[i] + SIGMA * M[i];
        ApRe[i] = dd * pRe[i];
        ApIm[i] = dd * pIm[i];
        for (const { j, w, cosR, sinR } of L.adj[i]) {
          ApRe[i] -= w * (cosR * pRe[j] - sinR * pIm[j]);
          ApIm[i] -= w * (sinR * pRe[j] + cosR * pIm[j]);
        }
      }
      const pAp = d2(pRe, pIm, ApRe, ApIm);
      if (Math.abs(pAp) < 1e-30) break;
      const alpha = rr / pAp;
      for (let i = 0; i < n; i++) {
        xRe[i] += alpha * pRe[i]; xIm[i] += alpha * pIm[i];
        rRe[i] -= alpha * ApRe[i]; rIm[i] -= alpha * ApIm[i];
      }
      const rrN = d2(rRe, rIm, rRe, rIm);
      if (Math.sqrt(rrN) < 1e-6 * Math.sqrt(d2(bRe, bIm, bRe, bIm) + 1e-30)) break;
      const beta = rrN / rr;
      for (let i = 0; i < n; i++) { pRe[i] = rRe[i] + beta * pRe[i]; pIm[i] = rIm[i] + beta * pIm[i]; }
      rr = rrN;
    }
    normC(xRe, xIm, M);
    psiRe = xRe; psiIm = xIm;
  }

  return { re: psiRe, im: psiIm };
}

/** Detect singularities via Connection Laplacian smallest eigenvector */
export function detectSingularities(mesh: HalfEdgeMesh): StripeFieldResult {
  const frames = buildVertexFrames(mesh);
  const L = buildComplexCotanL(mesh, frames);
  const M = computeVertexAreas(mesh);
  const { re, im } = inversePowerIteration(L, M);

  const n = L.n;
  const phase = new Float64Array(n);
  const amplitude = new Float64Array(n);
  let maxAmp = 0;
  for (let i = 0; i < n; i++) {
    phase[i] = Math.atan2(im[i], re[i]);
    amplitude[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
    if (amplitude[i] > maxAmp) maxAmp = amplitude[i];
  }
  const thresh = 0.01 * maxAmp;
  const singularities: number[] = [];
  for (let i = 0; i < n; i++) if (amplitude[i] < thresh) singularities.push(i);

  console.log(`[Singularity] ${singularities.length} detected (${(100 * singularities.length / n).toFixed(1)}%)`);
  return { phase, amplitude, singularities };
}

function d2(aR: Float64Array, aI: Float64Array, bR: Float64Array, bI: Float64Array): number {
  let s = 0;
  for (let i = 0; i < aR.length; i++) s += aR[i] * bR[i] + aI[i] * bI[i];
  return s;
}

function normC(re: Float64Array, im: Float64Array, M: Float64Array): void {
  let n2 = 0;
  for (let i = 0; i < re.length; i++) n2 += M[i] * (re[i] * re[i] + im[i] * im[i]);
  const inv = 1 / Math.sqrt(n2 + 1e-30);
  for (let i = 0; i < re.length; i++) { re[i] *= inv; im[i] *= inv; }
}

// ─────────────────────────────────────────────────────────────────────────────
// Legacy stubs (backward compat)
// ─────────────────────────────────────────────────────────────────────────────

export interface SparseMatrix {
  rows: number; cols: number;
  data: Map<string, { real: number; imag: number }>;
}

/** @deprecated */
export function buildConnectionLaplacian(
  _mesh: HalfEdgeMesh, _rotationAngle: number,
): { L: SparseMatrix; M: SparseMatrix } {
  const empty: SparseMatrix = { rows: 0, cols: 0, data: new Map() };
  return { L: empty, M: empty };
}

/** @deprecated */
export function solveEigenvector(
  _L: SparseMatrix, _M: SparseMatrix,
): { real: Float64Array; imag: Float64Array } {
  return { real: new Float64Array(0), imag: new Float64Array(0) };
}

/** @deprecated */
export function extractPhase(real: Float64Array, _imag: Float64Array): Float64Array {
  return real;
}

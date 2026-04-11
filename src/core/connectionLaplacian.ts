/**
 * connectionLaplacian.ts
 *
 * Stripe pattern generation via the Connection Laplacian eigenvector method,
 * based on Knöppel, Crane, Pinkall, Schröder (SIGGRAPH 2015).
 *
 * For each family k with stripe direction φ_k = k·π/3:
 *   1. Build per-vertex tangent frames
 *   2. Compute connection angles ω_ij (parallel transport rotation per edge)
 *   3. Assemble complex Connection Laplacian:
 *        L_ii = Σ_j w_ij,  L_ij = −w_ij · e^{i(ω_ij + φ_k)}
 *   4. Find smallest eigenvector ψ ∈ ℂ^n via inverse power iteration
 *   5. Extract phase θ = arg(ψ) and amplitude |ψ|
 *   6. Trace isolines of the phase field θ (with 2π branch-cut handling)
 *
 * Singularities (|ψ| ≈ 0) are where stripes naturally emerge or terminate,
 * enabling branching / short ribbons that start mid-surface.
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
}

export interface StripeFieldResult {
  phase:         Float64Array;
  amplitude:     Float64Array;
  singularities: number[];     // vertex indices where |ψ| < threshold
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal types
// ─────────────────────────────────────────────────────────────────────────────

interface VertexFrames {
  t1: THREE.Vector3[];
  t2: THREE.Vector3[];
}

interface ComplexNeighbor {
  j:   number;
  w:   number;   // cotangent weight (≥ 0)
  cosR: number;  // cos(ω_ij + φ)
  sinR: number;  // sin(ω_ij + φ)
}

interface ComplexCotanL {
  n:    number;
  diag: Float64Array;
  adj:  ComplexNeighbor[][];
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 1 – Per-vertex orthonormal tangent frames
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
// Step 2 – Connection angle ω_ij
//
// Measures how much the tangent frame rotates when parallel-transporting
// along edge (i → j).  Uses the Levi-Civita discrete connection.
// ─────────────────────────────────────────────────────────────────────────────

function connectionAngle(
  mesh: HalfEdgeMesh,
  frames: VertexFrames,
  vi: number,
  vj: number,
): number {
  const edge = new THREE.Vector3().subVectors(mesh.vertices[vj], mesh.vertices[vi]);

  // Project edge into tangent plane at i
  const ni = mesh.normals[vi];
  const ei = edge.clone().addScaledVector(ni, -edge.dot(ni));
  const alphaI = Math.atan2(ei.dot(frames.t2[vi]), ei.dot(frames.t1[vi]));

  // Project −edge into tangent plane at j
  const nj = mesh.normals[vj];
  const ej = edge.clone().negate().addScaledVector(nj, edge.dot(nj)); // -edge projected
  const alphaJ = Math.atan2(ej.dot(frames.t2[vj]), ej.dot(frames.t1[vj]));

  return alphaJ - alphaI + Math.PI;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 3 – Complex Connection Laplacian
//
// L_ii = Σ_j w_ij,   L_ij = −w_ij · e^{i(ω_ij + φ)}
//
// The rotation offset φ selects the stripe direction family.
// ─────────────────────────────────────────────────────────────────────────────

function buildComplexCotanL(
  mesh: HalfEdgeMesh,
  frames: VertexFrames,
  phi: number,
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

    diag[vi] += w;
    diag[vj] += w;

    // ω_ij for i→j
    const omega_ij = connectionAngle(mesh, frames, vi, vj);
    const angle_ij = omega_ij + phi;
    const angle_ji = -omega_ij - phi; // antisymmetric connection + same φ sign convention

    adj[vi].push({ j: vj, w, cosR: Math.cos(angle_ij), sinR: Math.sin(angle_ij) });
    adj[vj].push({ j: vi, w, cosR: Math.cos(angle_ji), sinR: Math.sin(angle_ji) });
  }

  return { n, diag, adj };
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 4 – Vertex areas (barycentric lumped mass matrix)
// ─────────────────────────────────────────────────────────────────────────────

function computeVertexAreas(mesh: HalfEdgeMesh): Float64Array {
  const areas = new Float64Array(mesh.vertices.length);
  for (const face of mesh.faces) {
    const p0 = mesh.vertices[face[0]];
    const p1 = mesh.vertices[face[1]];
    const p2 = mesh.vertices[face[2]];
    const area = new THREE.Vector3().crossVectors(
      new THREE.Vector3().subVectors(p1, p0),
      new THREE.Vector3().subVectors(p2, p0),
    ).length() * 0.5;
    const third = area / 3;
    areas[face[0]] += third;
    areas[face[1]] += third;
    areas[face[2]] += third;
  }
  return areas;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 5 – Complex CG solver for (L + σM)ψ = Mψ_old
//
// Operates on interleaved real/imaginary parts.
// Shifted system: (L + σM) ensures positive-definiteness.
// ─────────────────────────────────────────────────────────────────────────────

function complexMatvec(
  L: ComplexCotanL,
  M: Float64Array,
  sigma: number,
  xRe: Float64Array,
  xIm: Float64Array,
): { yRe: Float64Array; yIm: Float64Array } {
  const n = L.n;
  const yRe = new Float64Array(n);
  const yIm = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    // Diagonal: (L_ii + σ·M_i)
    const d = L.diag[i] + sigma * M[i];
    yRe[i] = d * xRe[i];
    yIm[i] = d * xIm[i];

    // Off-diagonal: −w_ij · e^{iR} · x_j
    // = −w · [(cosR · xRe_j − sinR · xIm_j) + i(sinR · xRe_j + cosR · xIm_j)]
    for (const { j, w, cosR, sinR } of L.adj[i]) {
      yRe[i] -= w * (cosR * xRe[j] - sinR * xIm[j]);
      yIm[i] -= w * (sinR * xRe[j] + cosR * xIm[j]);
    }
  }

  return { yRe, yIm };
}

function solveComplexCG(
  L: ComplexCotanL,
  M: Float64Array,
  sigma: number,
  bRe: Float64Array,
  bIm: Float64Array,
  maxIter = 800,
  tol = 1e-6,
): { re: Float64Array; im: Float64Array } {
  const n = L.n;
  const xRe = new Float64Array(n);
  const xIm = new Float64Array(n);

  // r = b − A·x = b (x=0)
  let rRe = Float64Array.from(bRe);
  let rIm = Float64Array.from(bIm);
  let pRe = Float64Array.from(rRe);
  let pIm = Float64Array.from(rIm);
  let rr = dot2(rRe, rIm, rRe, rIm);
  const bb = dot2(bRe, bIm, bRe, bIm);
  if (rr < 1e-30) return { re: xRe, im: xIm };

  for (let iter = 0; iter < maxIter; iter++) {
    const { yRe: ApRe, yIm: ApIm } = complexMatvec(L, M, sigma, pRe, pIm);
    const pAp = dot2(pRe, pIm, ApRe, ApIm);
    if (Math.abs(pAp) < 1e-30) break;
    const alpha = rr / pAp;

    for (let i = 0; i < n; i++) {
      xRe[i] += alpha * pRe[i];
      xIm[i] += alpha * pIm[i];
      rRe[i] -= alpha * ApRe[i];
      rIm[i] -= alpha * ApIm[i];
    }

    const rrNew = dot2(rRe, rIm, rRe, rIm);
    if (Math.sqrt(rrNew) < tol * Math.sqrt(bb + 1e-30)) break;

    const beta = rrNew / rr;
    for (let i = 0; i < n; i++) {
      pRe[i] = rRe[i] + beta * pRe[i];
      pIm[i] = rIm[i] + beta * pIm[i];
    }
    rr = rrNew;
  }

  return { re: xRe, im: xIm };
}

/** Real dot product on ℝ^{2n}: Σ(aRe·bRe + aIm·bIm) */
function dot2(
  aRe: Float64Array, aIm: Float64Array,
  bRe: Float64Array, bIm: Float64Array,
): number {
  let s = 0;
  for (let i = 0; i < aRe.length; i++) s += aRe[i] * bRe[i] + aIm[i] * bIm[i];
  return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 6 – Inverse power iteration for smallest eigenvector
//
// Solves (L + σM)ψ = Mψ_old repeatedly, normalising each time.
// The shift σ makes L+σM positive-definite.
// ─────────────────────────────────────────────────────────────────────────────

function inversePowerIteration(
  L: ComplexCotanL,
  M: Float64Array,
  outerIter = 25,
  cgIter = 800,
  cgTol = 1e-6,
): { re: Float64Array; im: Float64Array } {
  const n = L.n;
  const SIGMA = 1e-4;

  // Deterministic initial guess: ψ_i = e^{i · hash(i)}
  let psiRe: Float64Array = new Float64Array(n);
  let psiIm: Float64Array = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const angle = (i * 2654435761) % (2 * Math.PI); // Knuth hash
    psiRe[i] = Math.cos(angle);
    psiIm[i] = Math.sin(angle);
  }
  normalizeComplex(psiRe, psiIm, M);

  for (let iter = 0; iter < outerIter; iter++) {
    // RHS = M · ψ_old
    const bRe = new Float64Array(n);
    const bIm = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      bRe[i] = M[i] * psiRe[i];
      bIm[i] = M[i] * psiIm[i];
    }

    // Solve (L + σM) · ψ_new = b
    const { re, im } = solveComplexCG(L, M, SIGMA, bRe, bIm, cgIter, cgTol);

    // Normalise ψ_new in the M-norm
    normalizeComplex(re, im, M);
    psiRe = re;
    psiIm = im;
  }

  return { re: psiRe, im: psiIm };
}

function normalizeComplex(re: Float64Array, im: Float64Array, M: Float64Array): void {
  let norm2 = 0;
  for (let i = 0; i < re.length; i++) {
    norm2 += M[i] * (re[i] * re[i] + im[i] * im[i]);
  }
  const invNorm = 1 / Math.sqrt(norm2 + 1e-30);
  for (let i = 0; i < re.length; i++) {
    re[i] *= invNorm;
    im[i] *= invNorm;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 7 – Phase & amplitude extraction
// ─────────────────────────────────────────────────────────────────────────────

function extractPhaseAmplitude(
  re: Float64Array, im: Float64Array,
): { phase: Float64Array; amplitude: Float64Array } {
  const n = re.length;
  const phase     = new Float64Array(n);
  const amplitude = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    phase[i]     = Math.atan2(im[i], re[i]);
    amplitude[i] = Math.sqrt(re[i] * re[i] + im[i] * im[i]);
  }
  return { phase, amplitude };
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 8 – Phase-aware isoline tracing (handles 2π branch cuts)
//
// Isolines of the phase field θ ∈ (−π, π].
// At each triangle edge, the "true" phase difference is the wrapped
// difference δ = θ_j − θ_i  wrapped to (−π, π].
// Singularity detection: if the holonomy around a triangle ≠ 0 → skip.
// ─────────────────────────────────────────────────────────────────────────────

function wrapAngle(x: number): number {
  return x - 2 * Math.PI * Math.round(x / (2 * Math.PI));
}

export function traceIsolines(
  mesh: HalfEdgeMesh,
  phase: Float64Array,
  numIsolines: number,
  amplitude?: Float64Array,
): Isoline[] {
  const AMP_THRESHOLD = 0.01;  // skip singular faces
  const isolines: Isoline[] = [];

  for (let k = 0; k < numIsolines; k++) {
    // Levels evenly spaced in [−π, π)
    const level = -Math.PI + (2 * Math.PI) * (k + 0.5) / numIsolines;
    const pts: THREE.Vector3[] = [];
    const faceIds: number[] = [];

    for (let fi = 0; fi < mesh.faces.length; fi++) {
      const face = mesh.faces[fi];
      const v = [face[0], face[1], face[2]];

      // Skip low-amplitude (singular) faces
      if (amplitude) {
        const minAmp = Math.min(amplitude[v[0]], amplitude[v[1]], amplitude[v[2]]);
        if (minAmp < AMP_THRESHOLD) continue;
      }

      const theta = [phase[v[0]], phase[v[1]], phase[v[2]]];

      // Wrapped differences around the triangle
      const d01 = wrapAngle(theta[1] - theta[0]);
      const d12 = wrapAngle(theta[2] - theta[1]);
      const d20 = wrapAngle(theta[0] - theta[2]);

      // Holonomy check: non-zero holonomy = singularity inside face
      const holonomy = d01 + d12 + d20;
      if (Math.abs(holonomy) > Math.PI * 0.5) continue;

      // Check each edge for crossing
      const deltas = [d01, d12, d20];
      const edges: [number, number][] = [[0, 1], [1, 2], [2, 0]];
      const crossPts: THREE.Vector3[] = [];

      for (let e = 0; e < 3; e++) {
        const a = edges[e][0], b = edges[e][1];
        const delta = deltas[e];
        if (Math.abs(delta) < 1e-10) continue;

        // Offset of level from theta_a, wrapped to (−π, π]
        const offset = wrapAngle(level - theta[a]);

        // Check if offset lies within the traversal interval [0, delta] or [delta, 0]
        let t = -1;
        if (delta > 0 && offset >= 0 && offset < delta) {
          t = offset / delta;
        } else if (delta < 0 && offset <= 0 && offset > delta) {
          t = offset / delta;
        }

        if (t >= 0 && t < 1) {
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
// Public entry points
// ─────────────────────────────────────────────────────────────────────────────

// Cache: avoid rebuilding frames + areas for the same mesh
let _cache: {
  mesh: HalfEdgeMesh;
  frames: VertexFrames;
  areas: Float64Array;
} | null = null;

function ensureCache(mesh: HalfEdgeMesh): { frames: VertexFrames; areas: Float64Array } {
  if (_cache?.mesh === mesh) return _cache;
  const frames = buildVertexFrames(mesh);
  const areas  = computeVertexAreas(mesh);
  _cache = { mesh, frames, areas };
  return { frames, areas };
}

/**
 * Compute the stripe field for one family using the connection Laplacian.
 *
 * Returns the phase field θ ∈ (−π, π] (same Float64Array type as before
 * for backward compatibility).
 */
export function computeStripeField(
  mesh: HalfEdgeMesh,
  phi: number,
  _density = 4.0,
): Float64Array {
  return computeStripeFieldFull(mesh, phi).phase;
}

/**
 * Full stripe field result including amplitude and singularity data.
 */
export function computeStripeFieldFull(
  mesh: HalfEdgeMesh,
  phi: number,
): StripeFieldResult {
  const { frames, areas } = ensureCache(mesh);
  const L = buildComplexCotanL(mesh, frames, phi);

  console.time(`[Stripe] eigenvector φ=${(phi * 180 / Math.PI).toFixed(0)}°`);
  const { re, im } = inversePowerIteration(L, areas);
  console.timeEnd(`[Stripe] eigenvector φ=${(phi * 180 / Math.PI).toFixed(0)}°`);

  const { phase, amplitude } = extractPhaseAmplitude(re, im);

  // Detect singularities
  const AMP_THRESHOLD = 0.01;
  const maxAmp = Math.max(...Array.from(amplitude));
  const threshold = AMP_THRESHOLD * maxAmp;
  const singularities: number[] = [];
  for (let i = 0; i < amplitude.length; i++) {
    if (amplitude[i] < threshold) singularities.push(i);
  }

  console.log(
    `[Stripe] φ=${(phi * 180 / Math.PI).toFixed(0)}° → ` +
    `${singularities.length} singularities (${(100 * singularities.length / mesh.vertices.length).toFixed(1)}% of vertices)`,
  );

  return { phase, amplitude, singularities };
}

// ─────────────────────────────────────────────────────────────────────────────
// Legacy stubs kept so that imports elsewhere still compile
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

/** @deprecated Use computeStripeFieldFull instead */
export function extractPhase(real: Float64Array, _imag: Float64Array): Float64Array {
  return real;
}

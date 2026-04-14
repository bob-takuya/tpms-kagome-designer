/**
 * foliationPipeline.ts
 *
 * End-to-end wrapper for the Vekhter et al. 2019 "Weaving Geodesic
 * Foliations" direction-field stage, plumbed into the existing Knöppel
 * 2015 stripe eigensolve for per-family scalar integration.
 *
 * Pipeline:
 *   1. Solve the Knöppel 2013 3-RoSy connection-Laplacian eigenproblem
 *      on vertices to get an initial smooth direction field.
 *   2. Lift that field to face-based unit vectors (6-RoSy canonical
 *      representative per face).
 *   3. Run Algorithm 1 from the paper — alternating δ-projection onto
 *      the kernel of the discrete curl operator + per-face unit
 *      normalization — to make the field as geodesic as possible.
 *   4. Push the resulting ŵ back to per-vertex direction angles.
 *   5. Feed those angles into the Knöppel 2015 stripe eigensolve
 *      (three twisted-Laplacian eigenproblems, one per family).
 *
 * The output is API-compatible with computeKnoppelStripeFields: three
 * complex scalar fields ψ_k whose Re-zero-crossings are the kagome
 * isolines traced downstream.
 */

import type { HalfEdgeMesh } from './halfEdge';
import {
  computeKnoppelStripeFieldsFromDirection,
  computeVertex3RoSyEigenvector,
} from './connectionLaplacian';
import {
  buildFaceFrames,
  buildCurlOperator,
  buildVertexTangentFrames,
  initFieldFromVertex3RoSy,
  optimizeGeodesicField,
  faceFieldToVertexAngles,
  type GeodesicFieldResult,
} from './geodesicField';

export interface GeodesicFoliationOptions {
  maxIter?: number;      // Algorithm 1 outer iterations
  tol?: number;          // convergence tol on per-face update magnitude
  epsilon?: number;      // Tikhonov for CCᵀ solve
  cgMaxIter?: number;    // CG inner iters
  cgTol?: number;        // CG residual tolerance
}

export interface GeodesicFoliationResult {
  fields: { re: Float64Array; im: Float64Array; amplitude: Float64Array }[];
  field: GeodesicFieldResult;     // diagnostics for the geodesic step
}

export function computeGeodesicFoliationStripeFields(
  mesh: HalfEdgeMesh,
  omega: number,
  opts: GeodesicFoliationOptions = {},
): GeodesicFoliationResult {
  // ── 1. Initial 3-RoSy direction field ─────────────────────────────────────
  console.time('[Geodesic] 3-RoSy eigenvector');
  const { re, im } = computeVertex3RoSyEigenvector(mesh);
  console.timeEnd('[Geodesic] 3-RoSy eigenvector');

  // ── 2. Lift to face-based unit field ──────────────────────────────────────
  const frames = buildFaceFrames(mesh);
  const { t1, t2 } = buildVertexTangentFrames(mesh);
  const w0 = initFieldFromVertex3RoSy(mesh, frames, t1, t2, re, im);

  // ── 3. Algorithm 1 — curl-minimizing δ-projection ─────────────────────────
  console.time('[Geodesic] curl operator build');
  const C = buildCurlOperator(mesh, frames);
  console.timeEnd('[Geodesic] curl operator build');

  console.time('[Geodesic] Algorithm 1');
  const field = optimizeGeodesicField(C, w0, opts);
  console.timeEnd('[Geodesic] Algorithm 1');

  console.log(
    `[Geodesic] |Cŵ|: ${field.initialCurl.toExponential(3)} → ` +
    `${field.finalCurl.toExponential(3)} ` +
    `(${((1 - field.finalCurl / Math.max(field.initialCurl, 1e-30)) * 100).toFixed(1)}% ↓) ` +
    `in ${field.iterations} iters`,
  );

  // ── 4. Lift face field back to per-vertex direction angles ────────────────
  const baseDir = faceFieldToVertexAngles(mesh, frames, t1, t2, field.w);

  // ── 5. Knöppel 2015 stripe eigensolve per family ──────────────────────────
  const fields = computeKnoppelStripeFieldsFromDirection(mesh, omega, baseDir);

  return { fields, field };
}

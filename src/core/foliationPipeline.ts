/**
 * foliationPipeline.ts
 *
 * Thin TypeScript façade over the Vekhter 2019 "Weaving Geodesic
 * Foliations" WASM module (built from native/src/wgf_main.cpp via
 * Emscripten). The WASM module implements the full paper pipeline:
 *
 *   1. Knöppel et al. 2013 globally-optimal 6-RoSy direction field
 *      (connection-Laplacian smallest eigenvector).
 *   2. Algorithm 1 (paper §3.3, eq. 5): λ-sharpened alternating
 *      minimization with the full KKT system solved via Eigen's
 *      SparseLU.
 *   3. 6-fold branched cover construction (paper §5.1) with per-edge
 *      cyclic-shift permutation σ ∈ ℤ/6ℤ, vertex-holonomy detection
 *      and puncturing of branch-point vertices, and connected-component
 *      decomposition.
 *   4. Per-component Algorithm 2 (paper §4, eq. 7): initial scaling s
 *      via generalized-eigenproblem inverse power iteration,
 *      anti-aliasing rescale, and joint (s, θ) optimization (eq. 8).
 *   5. θ zero-crossing extraction on the cover, projection back to the
 *      base mesh with family labels derived from the layer index.
 *
 * This file packs the half-edge mesh vertices/faces into the WASM
 * heap, invokes wgf_run, and repacks the returned segments into the
 * Isoline[] form consumed downstream by kagome.ts.
 */

import * as THREE from 'three';
import type { HalfEdgeMesh } from './halfEdge';
import type { Isoline } from './connectionLaplacian';
import { runWgfPipeline, type WgfResult, type WgfOptions } from './wgfWasm';

export type { WgfOptions };

export interface GeodesicFoliationResult {
  isolinesByFamily: [Isoline[], Isoline[], Isoline[]];
  initialCurl: number;
  finalCurl:   number;
  iterations:  number;
  numSingular: number;
}

/**
 * Run the Vekhter 2019 WGF pipeline on a half-edge mesh and return
 * per-family isolines in the same shape the downstream kagome builder
 * expects.
 */
export async function computeGeodesicFoliationIsolines(
  mesh: HalfEdgeMesh,
  opts: WgfOptions = {},
): Promise<GeodesicFoliationResult> {

  const t0 = performance.now();
  const res: WgfResult = await runWgfPipeline(mesh, opts);
  const t1 = performance.now();

  console.log(
    `[WGF] ${res.segments.length} segments in ${(t1 - t0).toFixed(0)} ms | ` +
    `curl ${res.initialCurl.toExponential(3)} → ${res.finalCurl.toExponential(3)} ` +
    `in ${res.iterations} iters | ${res.numSingular} singular base vertices`,
  );

  // Partition segments by family and repack into the Isoline[] layout.
  // kagome.ts interprets iso.points as a flat list of segment endpoints
  // (iso.points[2k], iso.points[2k + 1]) with iso.faceIndices[k] being
  // the face index of the k-th segment. One Isoline per family is all
  // we need — the downstream stitcher chains them into polylines.
  const perFamPoints: [THREE.Vector3[], THREE.Vector3[], THREE.Vector3[]] = [[], [], []];
  const perFamFaces:  [number[],        number[],        number[]]        = [[], [], []];

  for (const s of res.segments) {
    const fam = s.family % 3;
    perFamPoints[fam].push(new THREE.Vector3(s.ax, s.ay, s.az));
    perFamPoints[fam].push(new THREE.Vector3(s.bx, s.by, s.bz));
    perFamFaces[fam].push(s.faceIdx);
  }

  const isolinesByFamily: [Isoline[], Isoline[], Isoline[]] = [[], [], []];
  for (let k = 0; k < 3; k++) {
    if (perFamPoints[k].length === 0) continue;
    isolinesByFamily[k].push({
      points: perFamPoints[k],
      faceIndices: perFamFaces[k],
    });
  }

  return {
    isolinesByFamily,
    initialCurl: res.initialCurl,
    finalCurl:   res.finalCurl,
    iterations:  res.iterations,
    numSingular: res.numSingular,
  };
}

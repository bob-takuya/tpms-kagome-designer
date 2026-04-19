/**
 * foliationPipeline.ts
 *
 * Thin façade that runs the Vekhter et al. 2019 WGF WASM pipeline
 * inside a dedicated Web Worker (see `wgfClient.ts`) and repacks its
 * output segments into the Isoline[] layout consumed by kagome.ts.
 *
 * Pipeline overview (all phases emit `wgf-progress` callbacks to the
 * UI so the user can see where time is going):
 *
 *   stage 1  Knöppel 2013 6-RoSy direction field eigensolve
 *   stage 2  Algorithm 1 (λ-sharpened) on the base mesh
 *   stage 3  6-fold branched cover construction + σ permutations
 *   stage 4  Connected component labelling of the cover
 *   stage 5  Per-component Algorithm 1 + Algorithm 2 + Eq. 8
 *   stage 6  Segment assembly
 *   stage 7  Done
 */

import * as THREE from 'three';
import type { HalfEdgeMesh } from './halfEdge';
import type { Isoline } from './connectionLaplacian';
import {
  runWgfPipeline,
  type WgfOptions,
  type WgfProgressEvent,
  type WgfResult,
} from './wgfClient';
import type { ParsedWgfResult } from './wgfIO';

export type { WgfOptions, WgfProgressEvent };

export interface GeodesicFoliationResult {
  isolinesByFamily: [Isoline[], Isoline[], Isoline[]];
  initialCurl: number;
  finalCurl:   number;
  iterations:  number;
  numSingular: number;
  /** Producer capability hints — populated when an imported wgf-output
   *  exposes them via META. The in-browser pipeline reports v=2 (chains
   *  only); v3/v4 features are surfaced only by Colab-CLI imports. */
  meta?: {
    version: number;
    numIsoLevels: number;
    hasResample: boolean;
    hasCut:      boolean;
    hasPrune:    boolean;
  };
}

export async function computeGeodesicFoliationIsolines(
  mesh: HalfEdgeMesh,
  opts: WgfOptions = {},
  onProgress?: (p: WgfProgressEvent) => void,
): Promise<GeodesicFoliationResult> {

  const t0 = performance.now();
  const res: WgfResult = await runWgfPipeline(mesh, opts, onProgress);
  const t1 = performance.now();

  console.log(
    `[WGF] ${res.segments.length} segments in ${(t1 - t0).toFixed(0)} ms | ` +
    `curl ${res.initialCurl.toExponential(3)} → ${res.finalCurl.toExponential(3)} ` +
    `in ${res.iterations} iters | ${res.numSingular} singular base vertices`,
  );

  // Partition segments by family and repack into Isoline[] (one entry
  // per family, with flat points[] and parallel faceIndices[]).
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
      family: k,
    });
  }

  return {
    isolinesByFamily,
    initialCurl: res.initialCurl,
    finalCurl:   res.finalCurl,
    iterations:  res.iterations,
    numSingular: res.numSingular,
    meta: {
      version: 2,            // in-browser WASM pipeline emits v2-equivalent data
      numIsoLevels: 0,
      hasResample: false,
      hasCut:      false,
      hasPrune:    false,
    },
  };
}

/**
 * Convert a parsed Colab-CLI result into the same per-family Isoline[]
 * layout produced by the in-browser pipeline. This is what lets the
 * rest of the app (kagome.ts, stripMesh.ts, unfold.ts) stay completely
 * unaware of where the segments came from.
 */
export function applyParsedWgfResult(
  parsed: ParsedWgfResult,
): GeodesicFoliationResult {
  // Two buckets per family:
  //   - "unknown" (chainId < 0, i.e. v1 file): collapse into a single
  //     Isoline per family, matching the pre-chainId behaviour so old
  //     wgf-output files keep working.
  //   - chainId >= 0 (v2 file): one Isoline per (family, chainId) so the
  //     kagome stitcher sees each cover chain as its own polyline.
  const unknownPoints: [THREE.Vector3[], THREE.Vector3[], THREE.Vector3[]] = [[], [], []];
  const unknownFaces:  [number[],        number[],        number[]]        = [[], [], []];
  const chainBuckets:  [Map<number, Isoline>, Map<number, Isoline>, Map<number, Isoline>] =
    [new Map(), new Map(), new Map()];

  for (const s of parsed.segments) {
    const fam = s.family % 3;
    const a = new THREE.Vector3(s.ax, s.ay, s.az);
    const b = new THREE.Vector3(s.bx, s.by, s.bz);
    if (s.chainId < 0) {
      unknownPoints[fam].push(a, b);
      unknownFaces[fam].push(s.faceIdx);
    } else {
      let iso = chainBuckets[fam].get(s.chainId);
      if (!iso) {
        iso = {
          points: [],
          faceIndices: [],
          family:   fam,
          chainId:  s.chainId,
          // First segment seen wins for the per-chain attributes — by
          // construction the v3+ exporter writes the same isoLevel/flag
          // for every segment of one chain.
          isoLevel: s.isoLevel,
          flag:     s.flag,
        };
        chainBuckets[fam].set(s.chainId, iso);
      }
      iso.points.push(a, b);
      iso.faceIndices.push(s.faceIdx);
    }
  }

  const isolinesByFamily: [Isoline[], Isoline[], Isoline[]] = [[], [], []];
  for (let k = 0; k < 3; k++) {
    if (unknownPoints[k].length > 0) {
      isolinesByFamily[k].push({
        points: unknownPoints[k],
        faceIndices: unknownFaces[k],
        family: k,
        isoLevel: -1,
        chainId: -1,
        flag: -1,
      });
    }
    // Stable order by chainId so repeated imports produce identical output.
    const ids = [...chainBuckets[k].keys()].sort((a, b) => a - b);
    for (const id of ids) {
      isolinesByFamily[k].push(chainBuckets[k].get(id)!);
    }
  }

  return {
    isolinesByFamily,
    initialCurl: parsed.initialCurl,
    finalCurl:   parsed.finalCurl,
    iterations:  parsed.iterations,
    numSingular: parsed.numSingular,
    meta: {
      version:      parsed.version,
      numIsoLevels: parsed.numIsoLevels,
      hasResample:  parsed.hasResample,
      hasCut:       parsed.hasCut,
      hasPrune:     parsed.hasPrune,
    },
  };
}

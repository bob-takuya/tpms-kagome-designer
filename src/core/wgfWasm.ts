/**
 * wgfWasm.ts
 *
 * Thin TypeScript wrapper around the Weaving Geodesic Foliations WASM
 * module compiled from native/src/wgf_main.cpp via Emscripten.
 *
 * The C side exposes a small C API:
 *
 *   void* wgf_run(double* V, int nV, int* F, int nF,
 *                 double lambdaInit, double lambdaMin, int alg1MaxIter,
 *                 double mu, int jointIters, double userScale,
 *                 int useCover);
 *   int        wgf_result_num_segments(void* h);
 *   const double* wgf_result_segment_points(void* h);
 *   const int*    wgf_result_segment_family(void* h);
 *   double     wgf_alg1_curl_initial(void* h);
 *   double     wgf_alg1_curl_final(void* h);
 *   int        wgf_alg1_iters(void* h);
 *   int        wgf_num_singularities(void* h);
 *   void       wgf_result_free(void* h);
 *
 * This file handles module loading, input packing into WASM heap, and
 * result unpacking into plain TS arrays.
 */

import type { HalfEdgeMesh } from './halfEdge';

// Static import of the Emscripten-generated glue. With SINGLE_FILE=1
// the .wasm is base64-embedded into this ES module, so there is NO
// sibling .wasm file to fetch and no runtime URL resolution to worry
// about — vite bundles the glue together with the rest of the app.
// This is the fix for the "iOS Safari blank viewport" symptom: the
// earlier `/* @vite-ignore */ dynamic-import-by-absolute-URL` path
// was silently failing under the /tpms-kagome-designer/ GH Pages
// sub-path on iOS.
//
// The ?url suffix is intentionally NOT used — we want the module
// itself, not its URL.
// @ts-expect-error — generated file, no .d.ts
import createWgfModule from '../wasm/wgf.js';

// The Emscripten module factory signature — hand-typed since we don't
// import the generated declaration file.
type EmModule = {
  HEAPF64: Float64Array;
  HEAP32:  Int32Array;
  HEAPU8:  Uint8Array;
  _malloc: (n: number) => number;
  _free:   (p: number) => void;
  ccall:   (name: string, ret: string | null, argTypes: string[], args: unknown[]) => unknown;
  cwrap:   (name: string, ret: string | null, argTypes: string[]) => (...args: unknown[]) => unknown;
};

let modulePromise: Promise<EmModule> | null = null;

export function loadWgfModule(): Promise<EmModule> {
  if (!modulePromise) {
    modulePromise = (createWgfModule as (opts?: unknown) => Promise<EmModule>)({});
  }
  return modulePromise;
}

export interface WgfSegment {
  ax: number; ay: number; az: number;
  bx: number; by: number; bz: number;
  family: number;   // 0 | 1 | 2
  faceIdx: number;  // base mesh face index
}

export interface WgfResult {
  segments: WgfSegment[];
  initialCurl: number;
  finalCurl:   number;
  iterations:  number;
  numSingular: number;
}

export interface WgfOptions {
  lambdaInit?:  number;   // paper: 1000
  lambdaMin?:   number;   // paper: → 0, we use 1e-3
  alg1MaxIter?: number;   // paper: ~50
  mu?:          number;   // paper: 1e-4
  jointIters?:  number;   // paper: 10
  userScale?:   number;   // stripe density multiplier (1 = paper default)
  useCover?:    boolean;  // 6-fold branched cover pipeline (paper §5.2)
}

/**
 * Run the full Weaving Geodesic Foliations pipeline on a triangle mesh.
 *
 * Allocates V/F buffers in WASM heap, calls the native pipeline, copies
 * out the segments, and frees all temporaries. Safe to call many times.
 */
export async function runWgfPipeline(
  mesh: HalfEdgeMesh,
  opts: WgfOptions = {},
): Promise<WgfResult> {
  const mod = await loadWgfModule();

  const nV = mesh.vertices.length;
  const nF = mesh.faces.length;

  // Allocate V (double, 3*nV) + F (int, 3*nF) in the WASM heap.
  const vBytes = 8 * 3 * nV;
  const fBytes = 4 * 3 * nF;
  const vPtr = mod._malloc(vBytes);
  const fPtr = mod._malloc(fBytes);

  try {
    const vView = new Float64Array(mod.HEAPF64.buffer, vPtr, 3 * nV);
    for (let i = 0; i < nV; i++) {
      const p = mesh.vertices[i];
      vView[3*i]     = p.x;
      vView[3*i + 1] = p.y;
      vView[3*i + 2] = p.z;
    }
    const fView = new Int32Array(mod.HEAP32.buffer, fPtr, 3 * nF);
    for (let i = 0; i < nF; i++) {
      const face = mesh.faces[i];
      fView[3*i]     = face[0];
      fView[3*i + 1] = face[1];
      fView[3*i + 2] = face[2];
    }

    // Pack options with paper defaults.
    const lambdaInit  = opts.lambdaInit  ?? 1000;
    const lambdaMin   = opts.lambdaMin   ?? 1e-3;
    const alg1MaxIter = opts.alg1MaxIter ?? 50;
    const mu          = opts.mu          ?? 1e-4;
    const jointIters  = opts.jointIters  ?? 10;
    const userScale   = opts.userScale   ?? 1.0;
    const useCover    = (opts.useCover ?? true) ? 1 : 0;

    const handle = mod.ccall(
      'wgf_run', 'number',
      ['number','number','number','number','number','number','number','number','number','number','number'],
      [vPtr, nV, fPtr, nF, lambdaInit, lambdaMin, alg1MaxIter, mu, jointIters, userScale, useCover],
    ) as number;

    if (handle === 0) {
      throw new Error('wgf_run failed (returned null handle)');
    }

    const nSeg = mod.ccall('wgf_result_num_segments', 'number', ['number'], [handle]) as number;
    const ptsPtr = mod.ccall('wgf_result_segment_points', 'number', ['number'], [handle]) as number;
    const famPtr = mod.ccall('wgf_result_segment_family', 'number', ['number'], [handle]) as number;
    const fcPtr  = mod.ccall('wgf_result_segment_face',   'number', ['number'], [handle]) as number;

    const segs: WgfSegment[] = new Array(nSeg);
    const ptsView = new Float64Array(mod.HEAPF64.buffer, ptsPtr, 6 * nSeg);
    const famView = new Int32Array(mod.HEAP32.buffer, famPtr, nSeg);
    const fcView  = new Int32Array(mod.HEAP32.buffer, fcPtr,  nSeg);
    for (let i = 0; i < nSeg; i++) {
      segs[i] = {
        ax: ptsView[6*i],     ay: ptsView[6*i + 1], az: ptsView[6*i + 2],
        bx: ptsView[6*i + 3], by: ptsView[6*i + 4], bz: ptsView[6*i + 5],
        family: famView[i],
        faceIdx: fcView[i],
      };
    }

    const initialCurl = mod.ccall('wgf_alg1_curl_initial', 'number', ['number'], [handle]) as number;
    const finalCurl   = mod.ccall('wgf_alg1_curl_final',   'number', ['number'], [handle]) as number;
    const iterations  = mod.ccall('wgf_alg1_iters',        'number', ['number'], [handle]) as number;
    const numSingular = mod.ccall('wgf_num_singularities', 'number', ['number'], [handle]) as number;

    mod.ccall('wgf_result_free', null, ['number'], [handle]);

    return { segments: segs, initialCurl, finalCurl, iterations, numSingular };
  } finally {
    mod._free(vPtr);
    mod._free(fPtr);
  }
}

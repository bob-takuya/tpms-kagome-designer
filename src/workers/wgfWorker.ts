/**
 * wgfWorker.ts
 *
 * Dedicated Web Worker that hosts the Weaving Geodesic Foliations
 * WASM module. Running the pipeline here has two purposes:
 *
 *   1. The main thread stays responsive while the C++ pipeline runs
 *      (no more frozen viewport during Knöppel eigensolves, Algorithm 1
 *      sharpening, cover build, and per-component Alg 2 / Eq 8).
 *
 *   2. C++ phase-boundary `reportProgress` calls use
 *      `self.postMessage({ type: 'wgf-progress', ... })`; because this
 *      worker lives in a `DedicatedWorkerGlobalScope`, those messages
 *      are delivered directly to the parent (main) thread without any
 *      additional plumbing on our side. The main thread listens for
 *      them and updates an on-screen progress overlay.
 *
 * Message protocol (main ↔ worker):
 *
 *   main → worker   { type: 'run', V: Float64Array, F: Int32Array,
 *                     opts: WgfOptions, reqId: number }
 *   worker → main   { type: 'wgf-progress', stage, cur, total, label }
 *   worker → main   { type: 'wgf-done', reqId, segments,
 *                     initialCurl, finalCurl, iterations, numSingular }
 *   worker → main   { type: 'wgf-error', reqId, msg }
 */

// @ts-expect-error — generated Emscripten glue
import createWgfModule from '../wasm/wgf.js';

interface EmModule {
  HEAPF64: Float64Array;
  HEAP32:  Int32Array;
  HEAPU8:  Uint8Array;
  _malloc: (n: number) => number;
  _free:   (p: number) => void;
  ccall:   (name: string, ret: string | null, argTypes: string[], args: unknown[]) => unknown;
  UTF8ToString: (p: number, maxBytes?: number) => string;
}

interface WgfOptions {
  lambdaInit?:  number;
  lambdaMin?:   number;
  alg1MaxIter?: number;
  mu?:          number;
  jointIters?:  number;
  userScale?:   number;
  useCover?:    boolean;
}

interface RunMsg {
  type: 'run';
  reqId: number;
  V: Float64Array;
  F: Int32Array;
  opts: WgfOptions;
}

let modulePromise: Promise<EmModule> | null = null;
function getModule(): Promise<EmModule> {
  if (!modulePromise) {
    modulePromise = (createWgfModule as (opts?: unknown) => Promise<EmModule>)({});
  }
  return modulePromise;
}

function post(msg: unknown): void {
  (self as unknown as { postMessage: (m: unknown) => void }).postMessage(msg);
}

self.addEventListener('message', async (e: MessageEvent<RunMsg>) => {
  const msg = e.data;
  if (msg?.type !== 'run') return;

  const reqId = msg.reqId;
  let mod: EmModule;
  try {
    post({ type: 'wgf-progress', stage: -1, cur: 0, total: 1, label: 'Loading WASM module' });
    mod = await getModule();
  } catch (err) {
    post({ type: 'wgf-error', reqId, msg: 'Failed to load WASM: ' + (err as Error).message });
    return;
  }

  // Pack V and F into WASM heap.
  const nV = msg.V.length / 3;
  const nF = msg.F.length / 3;
  const vBytes = 8 * msg.V.length;
  const fBytes = 4 * msg.F.length;

  let vPtr = 0, fPtr = 0, handle = 0;
  try {
    vPtr = mod._malloc(vBytes);
    fPtr = mod._malloc(fBytes);
    new Float64Array(mod.HEAPF64.buffer, vPtr, msg.V.length).set(msg.V);
    new Int32Array(mod.HEAP32.buffer, fPtr, msg.F.length).set(msg.F);

    // These defaults are only used when the caller omits a value. The
    // real defaults live in PipelineOptions in pipeline.h; ours here
    // just mirror them so the worker's fallback path matches.
    const o = msg.opts;
    const lambdaInit  = o.lambdaInit  ?? 1000;
    const lambdaMin   = o.lambdaMin   ?? 1e-3;
    const alg1MaxIter = o.alg1MaxIter ?? 50;
    const mu          = o.mu          ?? 1e-4;
    const jointIters  = o.jointIters  ?? 10;
    const userScale   = o.userScale   ?? 1.0;
    const useCover    = (o.useCover ?? true) ? 1 : 0;

    handle = mod.ccall(
      'wgf_run', 'number',
      ['number','number','number','number','number','number','number','number','number','number','number'],
      [vPtr, nV, fPtr, nF, lambdaInit, lambdaMin, alg1MaxIter, mu, jointIters, userScale, useCover],
    ) as number;

    if (handle === 0) {
      const errPtr = mod.ccall('wgf_last_error', 'number', [], []) as number;
      const errMsg = errPtr ? mod.UTF8ToString(errPtr) : '(no message)';
      throw new Error(errMsg);
    }

    const nSeg = mod.ccall('wgf_result_num_segments', 'number', ['number'], [handle]) as number;
    const ptsPtr = mod.ccall('wgf_result_segment_points', 'number', ['number'], [handle]) as number;
    const famPtr = mod.ccall('wgf_result_segment_family', 'number', ['number'], [handle]) as number;
    const fcPtr  = mod.ccall('wgf_result_segment_face',   'number', ['number'], [handle]) as number;

    // Copy out of the WASM heap into transferable typed arrays, so we
    // can postMessage by reference instead of serializing large segment
    // arrays through the structured clone algorithm.
    const segPoints = new Float64Array(6 * nSeg);
    segPoints.set(new Float64Array(mod.HEAPF64.buffer, ptsPtr, 6 * nSeg));
    const segFamily = new Int32Array(nSeg);
    segFamily.set(new Int32Array(mod.HEAP32.buffer, famPtr, nSeg));
    const segFace = new Int32Array(nSeg);
    segFace.set(new Int32Array(mod.HEAP32.buffer, fcPtr, nSeg));

    const initialCurl = mod.ccall('wgf_alg1_curl_initial', 'number', ['number'], [handle]) as number;
    const finalCurl   = mod.ccall('wgf_alg1_curl_final',   'number', ['number'], [handle]) as number;
    const iterations  = mod.ccall('wgf_alg1_iters',        'number', ['number'], [handle]) as number;
    const numSingular = mod.ccall('wgf_num_singularities', 'number', ['number'], [handle]) as number;

    mod.ccall('wgf_result_free', null, ['number'], [handle]);
    handle = 0;

    post({
      type: 'wgf-done',
      reqId,
      nSeg,
      segPoints,
      segFamily,
      segFace,
      initialCurl,
      finalCurl,
      iterations,
      numSingular,
    });
  } catch (err) {
    if (handle) mod.ccall('wgf_result_free', null, ['number'], [handle]);
    post({ type: 'wgf-error', reqId, msg: (err as Error).message || String(err) });
  } finally {
    if (vPtr) mod._free(vPtr);
    if (fPtr) mod._free(fPtr);
  }
});

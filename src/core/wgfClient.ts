/**
 * wgfClient.ts
 *
 * Main-thread client for the Weaving Geodesic Foliations WASM
 * pipeline. Owns a single dedicated Web Worker that hosts the WASM
 * module; requests are issued via postMessage and progress/result
 * messages come back asynchronously.
 *
 * The worker sends two kinds of messages while a run is in progress:
 *
 *   'wgf-progress'  — emitted from C++ at each phase boundary
 *                     (stage / cur / total / label).  Streamed live
 *                     to the caller's onProgress callback.
 *
 *   'wgf-done' | 'wgf-error'
 *                   — emitted once when the run finishes. Resolves
 *                     or rejects the caller's Promise.
 */

import type { HalfEdgeMesh } from './halfEdge';

export interface WgfProgressEvent {
  stage: number;
  cur: number;
  total: number;
  label: string;
}

export interface WgfSegment {
  ax: number; ay: number; az: number;
  bx: number; by: number; bz: number;
  family: number;
  faceIdx: number;
}

export interface WgfResult {
  segments: WgfSegment[];
  initialCurl: number;
  finalCurl:   number;
  iterations:  number;
  numSingular: number;
}

export interface WgfOptions {
  lambdaInit?:  number;
  lambdaMin?:   number;
  alg1MaxIter?: number;
  mu?:          number;
  jointIters?:  number;
  userScale?:   number;
  useCover?:    boolean;
}

let worker: Worker | null = null;
let nextReqId = 1;

function getWorker(): Worker {
  if (!worker) {
    worker = new Worker(
      new URL('../workers/wgfWorker.ts', import.meta.url),
      { type: 'module' },
    );
  }
  return worker;
}

/**
 * Terminate the worker. Next call to runWgfPipeline will spawn a new
 * one. Used when the user aborts a long-running pipeline.
 */
export function cancelWgfRun(): void {
  if (worker) {
    worker.terminate();
    worker = null;
  }
}

export function runWgfPipeline(
  mesh: HalfEdgeMesh,
  opts: WgfOptions,
  onProgress?: (p: WgfProgressEvent) => void,
): Promise<WgfResult> {
  const w = getWorker();
  const reqId = nextReqId++;

  return new Promise((resolve, reject) => {
    const handleMsg = (e: MessageEvent) => {
      const d = e.data;
      if (!d || typeof d !== 'object') return;
      if (d.type === 'wgf-progress') {
        onProgress?.({ stage: d.stage, cur: d.cur, total: d.total, label: d.label });
        return;
      }
      // done / error are keyed on reqId so we don't pick up stale
      // messages from earlier runs.
      if (d.reqId !== reqId) return;
      w.removeEventListener('message', handleMsg);
      if (d.type === 'wgf-done') {
        const nSeg = d.nSeg as number;
        const pts = d.segPoints as Float64Array;
        const fam = d.segFamily as Int32Array;
        const face = d.segFace as Int32Array;
        const segments: WgfSegment[] = new Array(nSeg);
        for (let i = 0; i < nSeg; i++) {
          segments[i] = {
            ax: pts[6*i],     ay: pts[6*i + 1], az: pts[6*i + 2],
            bx: pts[6*i + 3], by: pts[6*i + 4], bz: pts[6*i + 5],
            family:  fam[i],
            faceIdx: face[i],
          };
        }
        resolve({
          segments,
          initialCurl: d.initialCurl,
          finalCurl:   d.finalCurl,
          iterations:  d.iterations,
          numSingular: d.numSingular,
        });
      } else if (d.type === 'wgf-error') {
        reject(new Error(d.msg || 'wgf worker error'));
      }
    };
    w.addEventListener('message', handleMsg);

    // Pack V and F as transferable typed arrays so the post is a
    // zero-copy handoff rather than a structured-clone of THREE.Vector3
    // objects.
    const V = new Float64Array(mesh.vertices.length * 3);
    for (let i = 0; i < mesh.vertices.length; i++) {
      const p = mesh.vertices[i];
      V[3*i]     = p.x;
      V[3*i + 1] = p.y;
      V[3*i + 2] = p.z;
    }
    const F = new Int32Array(mesh.faces.length * 3);
    for (let i = 0; i < mesh.faces.length; i++) {
      const f = mesh.faces[i];
      F[3*i]     = f[0];
      F[3*i + 1] = f[1];
      F[3*i + 2] = f[2];
    }
    w.postMessage({ type: 'run', reqId, V, F, opts }, [V.buffer, F.buffer]);
  });
}

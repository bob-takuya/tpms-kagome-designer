/**
 * Smoke test for the Vekhter 2019 "Weaving Geodesic Foliations" Algorithm 1
 * implementation. Generates a small Schwarz-P TPMS mesh, runs the geodesic
 * foliation pipeline, and prints diagnostics:
 *   - initial / final total curl
 *   - number of isolines produced per family
 *
 * Run with: npx tsx test/geodesic-smoke.ts
 */

import { createTPMSFunction } from '../src/core/tpms';
import { marchingCubes } from '../src/core/marchingCubes';
import { buildHalfEdgeMesh } from '../src/core/halfEdge';
import { computeGeodesicFoliationStripeFields } from '../src/core/foliationPipeline';
import { traceZeroCrossings } from '../src/core/connectionLaplacian';

const surfaceType = 'schwarzP' as const;
const period = 2 * Math.PI;
const box = { min: -Math.PI, max: Math.PI };
const gridRes = 36;

const f = createTPMSFunction(surfaceType, 0, false, 0, 0, 0);
const meshData = marchingCubes(f, box.min, box.max, gridRes, period);
const mesh = buildHalfEdgeMesh(meshData);

console.log(`[smoke] mesh: V=${mesh.vertices.length} F=${mesh.faces.length}`);

const omega = 4;

for (const useCover of [false, true]) {
  console.log(`\n=== useCover=${useCover} ===`);
  const { fields, field } = computeGeodesicFoliationStripeFields(mesh, omega, {
    maxIter: 30,
    tol:     1e-5,
    epsilon: 1e-6,
    useCover,
  });

  console.log(
    `[smoke] Algorithm 1: |Cŵ|: ${field.initialCurl.toExponential(3)} → ` +
    `${field.finalCurl.toExponential(3)} ` +
    `(${((1 - field.finalCurl / Math.max(field.initialCurl, 1e-30)) * 100).toFixed(1)}% reduction) ` +
    `in ${field.iterations} iters`,
  );

  for (let k = 0; k < 3; k++) {
    const iso = traceZeroCrossings(mesh, fields[k].re, fields[k].amplitude, 0.1);
    const totalSeg = iso.reduce((s, il) => s + il.faceIndices.length, 0);
    console.log(`[smoke] family ${k}: ${iso.length} isoline components, ${totalSeg} segments`);
  }
}

import * as THREE from 'three';
import { marchingCubes } from './src/core/marchingCubes.js';
import { buildHalfEdgeMesh } from './src/core/halfEdge.js';
import { computeStripeField } from './src/core/connectionLaplacian.js';
import { cotangentWeight, getHalfEdgeStart } from './src/core/halfEdge.js';

const fn = (x: number, y: number, z: number) =>
  Math.cos(x)*Math.cos(y)*Math.cos(z) - Math.sin(x)*Math.sin(y)*Math.sin(z);

const meshData = marchingCubes(fn, -Math.PI, Math.PI, 50, 2*Math.PI);
console.log('verts:', meshData.vertices.length/3, 'tris:', meshData.indices.length/3);
const hem = buildHalfEdgeMesh(meshData);

// Check cotangent weight distribution
const cotanWeights: number[] = [];
for (let heIdx = 0; heIdx < hem.halfEdges.length; heIdx++) {
  const he = hem.halfEdges[heIdx];
  if (he.twin !== -1 && heIdx > he.twin) continue;
  const raw = cotangentWeight(hem, heIdx);
  if (isFinite(raw)) cotanWeights.push(raw);
}
cotanWeights.sort((a,b) => a-b);
const nc = cotanWeights.length;
console.log(`Cotan weights: n=${nc}`);
console.log(`  min=${cotanWeights[0].toFixed(4)}  p01=${cotanWeights[Math.floor(0.01*nc)].toFixed(4)}  p50=${cotanWeights[Math.floor(0.50*nc)].toFixed(4)}  p99=${cotanWeights[Math.floor(0.99*nc)].toFixed(4)}  max=${cotanWeights[nc-1].toFixed(4)}`);
const hugeCnt = cotanWeights.filter(w => w > 10).length;
console.log(`  weights > 10: ${hugeCnt} (${(100*hugeCnt/nc).toFixed(1)}%)`);

// Check connected components via BFS
const n = hem.vertices.length;
const visited = new Uint8Array(n);
const adj: number[][] = Array.from({length: n}, () => []);
for (const he of hem.halfEdges) {
  const vi = he.vertex;
  // find start via halfEdge structure
}
// Simpler: build adjacency from faces
for (const face of hem.faces) {
  for (let i = 0; i < 3; i++) {
    adj[face[i]].push(face[(i+1)%3]);
    adj[face[(i+1)%3]].push(face[i]);
  }
}
let numComponents = 0;
const compSize: number[] = [];
for (let start = 0; start < n; start++) {
  if (visited[start]) continue;
  numComponents++;
  const queue = [start];
  let size = 0;
  while (queue.length) {
    const v = queue.pop()!;
    if (visited[v]) continue;
    visited[v] = 1;
    size++;
    for (const nb of adj[v]) if (!visited[nb]) queue.push(nb);
  }
  compSize.push(size);
}
console.log(`\nConnected components: ${numComponents}`);
console.log(`  sizes: ${compSize.join(', ')}`);


import { traceIsolines } from './src/core/connectionLaplacian.js';

for (let phi = 0; phi < 3; phi++) {
  const f = computeStripeField(hem, phi * Math.PI/3, 4.0);
  let fMin = f[0], fMax = f[0];
  let nanCount = 0, infCount = 0;
  for (const v of f) {
    if (isNaN(v)) nanCount++;
    else if (!isFinite(v)) infCount++;
    else { if (v < fMin) fMin = v; if (v > fMax) fMax = v; }
  }
  console.log(`phi=${(phi*60).toString().padStart(3)}°: fMin=${fMin.toFixed(4)}  fMax=${fMax.toFixed(4)}  range=${(fMax-fMin).toFixed(4)}  nan=${nanCount}  inf=${infCount}`);
  
  // Check distribution: how many vertices are in each quartile?
  const sorted = Array.from(f).filter(v => isFinite(v)).sort((a,b)=>a-b);
  const q = [0.01, 0.1, 0.5, 0.9, 0.99].map(p => sorted[Math.floor(p * sorted.length)]);
  console.log(`  p01=${q[0].toFixed(2)} p10=${q[1].toFixed(2)} p50=${q[2].toFixed(2)} p90=${q[3].toFixed(2)} p99=${q[4].toFixed(2)}`);

  const isos = traceIsolines(hem, f, 8);
  const segs = isos.reduce((s, iso) => s + iso.points.length/2, 0);
  console.log(`  → isolines=${isos.length}  segments=${segs}`);
}

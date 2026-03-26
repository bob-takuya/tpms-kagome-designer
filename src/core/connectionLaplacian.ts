import * as THREE from 'three';
import type { HalfEdgeMesh } from './halfEdge';
import { cotangentWeight, getHalfEdgeStart } from './halfEdge';

export interface SparseMatrix {
  rows: number;
  cols: number;
  data: Map<string, { real: number; imag: number }>;
}

function sparseKey(i: number, j: number): string {
  return `${i},${j}`;
}

function sparseGet(mat: SparseMatrix, i: number, j: number): { real: number; imag: number } {
  return mat.data.get(sparseKey(i, j)) || { real: 0, imag: 0 };
}

function sparseSet(mat: SparseMatrix, i: number, j: number, val: { real: number; imag: number }): void {
  if (Math.abs(val.real) < 1e-15 && Math.abs(val.imag) < 1e-15) {
    mat.data.delete(sparseKey(i, j));
  } else {
    mat.data.set(sparseKey(i, j), val);
  }
}

function sparseAdd(mat: SparseMatrix, i: number, j: number, val: { real: number; imag: number }): void {
  const existing = sparseGet(mat, i, j);
  sparseSet(mat, i, j, {
    real: existing.real + val.real,
    imag: existing.imag + val.imag
  });
}

// Build Connection Laplacian with rotation angles
export function buildConnectionLaplacian(
  mesh: HalfEdgeMesh,
  rotationAngle: number
): { L: SparseMatrix; M: SparseMatrix } {
  const n = mesh.vertices.length;

  const L: SparseMatrix = { rows: n, cols: n, data: new Map() };
  const M: SparseMatrix = { rows: n, cols: n, data: new Map() };

  // Build direction field angles per edge
  const edgeAngles = new Map<number, number>();

  for (let heIdx = 0; heIdx < mesh.halfEdges.length; heIdx++) {
    const he = mesh.halfEdges[heIdx];
    const vi = getHalfEdgeStart(mesh, heIdx);
    const vj = he.vertex;

    // Compute edge direction angle in tangent plane
    const pi = mesh.vertices[vi];
    const pj = mesh.vertices[vj];
    const ni = mesh.normals[vi];

    const edge = new THREE.Vector3().subVectors(pj, pi);

    // Project edge onto tangent plane
    const edgeTangent = edge.clone().sub(ni.clone().multiplyScalar(edge.dot(ni)));
    edgeTangent.normalize();

    // Create a local frame on the tangent plane
    const tangent1 = new THREE.Vector3(1, 0, 0);
    if (Math.abs(ni.dot(tangent1)) > 0.9) {
      tangent1.set(0, 1, 0);
    }
    const tangent2 = new THREE.Vector3().crossVectors(ni, tangent1).normalize();
    tangent1.crossVectors(tangent2, ni).normalize();

    // Compute angle in local frame
    const angle = Math.atan2(edgeTangent.dot(tangent2), edgeTangent.dot(tangent1));

    // Add rotation for direction field family
    edgeAngles.set(heIdx, angle + rotationAngle);
  }

  // Build mass matrix (diagonal with vertex areas)
  const vertexAreas = computeVertexAreas(mesh);
  for (let i = 0; i < n; i++) {
    sparseSet(M, i, i, { real: vertexAreas[i], imag: 0 });
  }

  // Build Connection Laplacian
  for (let heIdx = 0; heIdx < mesh.halfEdges.length; heIdx++) {
    const he = mesh.halfEdges[heIdx];
    if (he.twin !== -1 && heIdx > he.twin) continue; // Process each edge once

    const vi = getHalfEdgeStart(mesh, heIdx);
    const vj = he.vertex;

    // Cotangent weight
    const w = Math.max(0, cotangentWeight(mesh, heIdx));

    // Connection angle (difference in direction field between vertices)
    const alphaIJ = edgeAngles.get(heIdx) || 0;
    const alphaJI = edgeAngles.get(he.twin) || (alphaIJ + Math.PI);
    const connectionAngle = alphaIJ - alphaJI + Math.PI;

    // Rotation matrix element
    const cosA = Math.cos(connectionAngle);
    const sinA = Math.sin(connectionAngle);

    // Off-diagonal entries (with rotation)
    sparseAdd(L, vi, vj, { real: -w * cosA, imag: -w * sinA });
    sparseAdd(L, vj, vi, { real: -w * cosA, imag: w * sinA });

    // Diagonal entries
    sparseAdd(L, vi, vi, { real: w, imag: 0 });
    sparseAdd(L, vj, vj, { real: w, imag: 0 });
  }

  return { L, M };
}

function computeVertexAreas(mesh: HalfEdgeMesh): number[] {
  const areas = new Array(mesh.vertices.length).fill(0);

  for (let faceIdx = 0; faceIdx < mesh.faces.length; faceIdx++) {
    const face = mesh.faces[faceIdx];
    const v0 = mesh.vertices[face[0]];
    const v1 = mesh.vertices[face[1]];
    const v2 = mesh.vertices[face[2]];

    const edge1 = new THREE.Vector3().subVectors(v1, v0);
    const edge2 = new THREE.Vector3().subVectors(v2, v0);
    const faceArea = new THREE.Vector3().crossVectors(edge1, edge2).length() / 2;

    // Distribute area to vertices (barycentric)
    for (const vIdx of face) {
      areas[vIdx] += faceArea / 3;
    }
  }

  return areas;
}

// Power iteration for smallest non-trivial eigenvector
export function solveEigenvector(
  L: SparseMatrix,
  M: SparseMatrix,
  maxIterations: number = 100,
  tolerance: number = 1e-6
): { real: Float64Array<ArrayBuffer>; imag: Float64Array<ArrayBuffer> } {
  const n = L.rows;

  // Initialize with random complex vector
  let vReal: Float64Array<ArrayBuffer> = new Float64Array(n);
  let vImag: Float64Array<ArrayBuffer> = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    vReal[i] = Math.random() - 0.5;
    vImag[i] = Math.random() - 0.5;
  }

  // Normalize
  normalizeComplex(vReal, vImag, M);

  // Inverse power iteration (for smallest eigenvalue)
  // We'll use Lanczos-style iteration with shift
  for (let iter = 0; iter < maxIterations; iter++) {
    // Apply M^-1 * L
    const applied = applyMatrix(L, vReal, vImag);
    const newReal = applied.real;
    const newImag = applied.imag;

    // Solve M * x = Lv (since M is diagonal, this is easy)
    for (let i = 0; i < n; i++) {
      const mVal = sparseGet(M, i, i).real;
      if (mVal > 1e-10) {
        newReal[i] /= mVal;
        newImag[i] /= mVal;
      }
    }

    // Compute Rayleigh quotient for convergence check
    const prevReal = vReal;
    const prevImag = vImag;

    vReal = newReal;
    vImag = newImag;

    // Remove constant mode (project out)
    let sumReal = 0, sumImag = 0, sumWeight = 0;
    for (let i = 0; i < n; i++) {
      const w = sparseGet(M, i, i).real;
      sumReal += vReal[i] * w;
      sumImag += vImag[i] * w;
      sumWeight += w;
    }
    if (sumWeight > 1e-10) {
      sumReal /= sumWeight;
      sumImag /= sumWeight;
      for (let i = 0; i < n; i++) {
        vReal[i] -= sumReal;
        vImag[i] -= sumImag;
      }
    }

    // Normalize
    normalizeComplex(vReal, vImag, M);

    // Check convergence
    let diff = 0;
    for (let i = 0; i < n; i++) {
      diff += (vReal[i] - prevReal[i]) ** 2 + (vImag[i] - prevImag[i]) ** 2;
    }
    if (Math.sqrt(diff) < tolerance) break;
  }

  return { real: vReal, imag: vImag };
}

function applyMatrix(
  mat: SparseMatrix,
  vReal: Float64Array<ArrayBuffer>,
  vImag: Float64Array<ArrayBuffer>
): { real: Float64Array<ArrayBuffer>; imag: Float64Array<ArrayBuffer> } {
  const n = mat.rows;
  const resultReal: Float64Array<ArrayBuffer> = new Float64Array(n);
  const resultImag: Float64Array<ArrayBuffer> = new Float64Array(n);

  mat.data.forEach((val, key) => {
    const [i, j] = key.split(',').map(Number);
    // Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    resultReal[i] += val.real * vReal[j] - val.imag * vImag[j];
    resultImag[i] += val.real * vImag[j] + val.imag * vReal[j];
  });

  return { real: resultReal, imag: resultImag };
}

function normalizeComplex(
  vReal: Float64Array<ArrayBuffer>,
  vImag: Float64Array<ArrayBuffer>,
  M: SparseMatrix
): void {
  let norm = 0;
  for (let i = 0; i < vReal.length; i++) {
    const mVal = sparseGet(M, i, i).real;
    norm += mVal * (vReal[i] ** 2 + vImag[i] ** 2);
  }
  norm = Math.sqrt(norm);

  if (norm > 1e-10) {
    for (let i = 0; i < vReal.length; i++) {
      vReal[i] /= norm;
      vImag[i] /= norm;
    }
  }
}

// Extract phase from complex eigenvector
export function extractPhase(real: Float64Array, imag: Float64Array): Float64Array {
  const phase = new Float64Array(real.length);
  for (let i = 0; i < real.length; i++) {
    phase[i] = Math.atan2(imag[i], real[i]);
  }
  return phase;
}

// Trace isolines on the mesh
export interface Isoline {
  points: THREE.Vector3[];
  faceIndices: number[];
}

export function traceIsolines(
  mesh: HalfEdgeMesh,
  phase: Float64Array,
  numIsolines: number
): Isoline[] {
  const isolines: Isoline[] = [];

  for (let i = 0; i < numIsolines; i++) {
    const targetPhase = (2 * Math.PI * i) / numIsolines - Math.PI;
    const isoline = traceSingleIsoline(mesh, phase, targetPhase);
    if (isoline.points.length > 1) {
      isolines.push(isoline);
    }
  }

  return isolines;
}

function traceSingleIsoline(
  mesh: HalfEdgeMesh,
  phase: Float64Array,
  targetPhase: number
): Isoline {
  const points: THREE.Vector3[] = [];
  const faceIndices: number[] = [];
  const visitedFaces = new Set<number>();

  // Find starting face and edge
  for (let faceIdx = 0; faceIdx < mesh.faces.length; faceIdx++) {
    if (visitedFaces.has(faceIdx)) continue;

    const face = mesh.faces[faceIdx];
    const crossings = findPhaseCrossings(mesh, phase, face, targetPhase);

    if (crossings.length >= 2) {
      // Start tracing from this face
      traceFromFace(
        mesh, phase, targetPhase, faceIdx,
        points, faceIndices, visitedFaces
      );
    }
  }

  return { points, faceIndices };
}

function findPhaseCrossings(
  mesh: HalfEdgeMesh,
  phase: Float64Array,
  face: number[],
  targetPhase: number
): { edgeIdx: number; point: THREE.Vector3; t: number }[] {
  const crossings: { edgeIdx: number; point: THREE.Vector3; t: number }[] = [];

  for (let i = 0; i < 3; i++) {
    const v0 = face[i];
    const v1 = face[(i + 1) % 3];

    let p0 = phase[v0];
    let p1 = phase[v1];

    // Handle phase wrapping
    const diff = p1 - p0;
    if (diff > Math.PI) p1 -= 2 * Math.PI;
    else if (diff < -Math.PI) p1 += 2 * Math.PI;

    // Check if isoline crosses this edge
    if ((p0 <= targetPhase && targetPhase < p1) ||
        (p1 <= targetPhase && targetPhase < p0)) {
      const t = (targetPhase - p0) / (p1 - p0);
      const point = new THREE.Vector3()
        .lerpVectors(mesh.vertices[v0], mesh.vertices[v1], t);

      crossings.push({ edgeIdx: i, point, t });
    }
  }

  return crossings;
}

function traceFromFace(
  mesh: HalfEdgeMesh,
  phase: Float64Array,
  targetPhase: number,
  startFace: number,
  points: THREE.Vector3[],
  faceIndices: number[],
  visitedFaces: Set<number>
): void {
  let currentFace = startFace;
  let entryEdge = -1;
  let maxSteps = mesh.faces.length;

  while (maxSteps-- > 0) {
    if (visitedFaces.has(currentFace)) break;
    visitedFaces.add(currentFace);

    const face = mesh.faces[currentFace];
    const crossings = findPhaseCrossings(mesh, phase, face, targetPhase);

    if (crossings.length < 2) break;

    // Find exit crossing (not the entry edge)
    for (const crossing of crossings) {
      if (crossing.edgeIdx !== entryEdge) {
        points.push(crossing.point);
        faceIndices.push(currentFace);

        // Find next face through this edge
        const heStart = mesh.faceToHalfEdge[currentFace];
        let heIdx = heStart;
        for (let i = 0; i < crossing.edgeIdx; i++) {
          heIdx = mesh.halfEdges[heIdx].next;
        }

        const twinIdx = mesh.halfEdges[heIdx].twin;
        if (twinIdx === -1) break;

        const nextFace = mesh.halfEdges[twinIdx].face;
        const nextFaceVerts = mesh.faces[nextFace];

        // Find entry edge index in next face
        const exitV0 = face[crossing.edgeIdx];
        const exitV1 = face[(crossing.edgeIdx + 1) % 3];

        entryEdge = -1;
        for (let i = 0; i < 3; i++) {
          const nv0 = nextFaceVerts[i];
          const nv1 = nextFaceVerts[(i + 1) % 3];
          if ((nv0 === exitV0 && nv1 === exitV1) || (nv0 === exitV1 && nv1 === exitV0)) {
            entryEdge = i;
            break;
          }
        }

        currentFace = nextFace;
        break;
      }
    }
  }
}

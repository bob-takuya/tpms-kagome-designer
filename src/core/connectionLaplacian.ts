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

    // Off-diagonal: -w * e^(i * connectionAngle)
    const offDiagIJ = {
      real: -w * Math.cos(connectionAngle),
      imag: -w * Math.sin(connectionAngle)
    };
    const offDiagJI = {
      real: -w * Math.cos(-connectionAngle),
      imag: -w * Math.sin(-connectionAngle)
    };

    sparseAdd(L, vi, vj, offDiagIJ);
    sparseAdd(L, vj, vi, offDiagJI);

    // Diagonal: +w
    sparseAdd(L, vi, vi, { real: w, imag: 0 });
    sparseAdd(L, vj, vj, { real: w, imag: 0 });
  }

  return { L, M };
}

function computeVertexAreas(mesh: HalfEdgeMesh): Float64Array {
  const areas = new Float64Array(mesh.vertices.length);

  for (let faceIdx = 0; faceIdx < mesh.faces.length; faceIdx++) {
    const face = mesh.faces[faceIdx];
    const v0 = mesh.vertices[face[0]];
    const v1 = mesh.vertices[face[1]];
    const v2 = mesh.vertices[face[2]];

    const e1 = new THREE.Vector3().subVectors(v1, v0);
    const e2 = new THREE.Vector3().subVectors(v2, v0);
    const area = e1.cross(e2).length() / 2;
    const oneThird = area / 3;

    areas[face[0]] += oneThird;
    areas[face[1]] += oneThird;
    areas[face[2]] += oneThird;
  }

  return areas;
}

// ---------------------------------------------------------------------------
// Complex Conjugate Gradient solver: solves (L + shift*M) * x = M * v
// L is Hermitian positive semi-definite; shift regularises the zero eigenvalue
// ---------------------------------------------------------------------------
function complexCG(
  L: SparseMatrix,
  M: SparseMatrix,
  bReal: Float64Array,
  bImag: Float64Array,
  shift: number,
  maxIter: number,
  tol: number
): { real: Float64Array; imag: Float64Array } {
  const n = L.rows;

  // x_0 = 0
  const xReal = new Float64Array(n);
  const xImag = new Float64Array(n);

  // r_0 = b  (since A*x_0 = 0)
  const rReal = bReal.slice() as Float64Array;
  const rImag = bImag.slice() as Float64Array;

  const pReal = rReal.slice() as Float64Array;
  const pImag = rImag.slice() as Float64Array;

  // <r,r>_real  (for Hermitian A, CG residuals stay real-valued)
  let rr = realInnerProduct(rReal, rImag, rReal, rImag);
  if (rr < 1e-30) return { real: xReal, imag: xImag };

  for (let iter = 0; iter < maxIter; iter++) {
    // Ap = (L + shift*M) * p
    const ApR = new Float64Array(n);
    const ApI = new Float64Array(n);

    // L * p
    L.data.forEach((val, key) => {
      const [i, j] = key.split(',').map(Number);
      ApR[i] += val.real * pReal[j] - val.imag * pImag[j];
      ApI[i] += val.real * pImag[j] + val.imag * pReal[j];
    });

    // shift * M * p  (M is diagonal and real)
    for (let i = 0; i < n; i++) {
      const m = sparseGet(M, i, i).real;
      ApR[i] += shift * m * pReal[i];
      ApI[i] += shift * m * pImag[i];
    }

    // alpha = <r,r> / <p, Ap>  (denominator is real for Hermitian A)
    const pAp = realInnerProduct(pReal, pImag, ApR, ApI);
    if (Math.abs(pAp) < 1e-30) break;
    const alpha = rr / pAp;

    // x += alpha * p
    for (let i = 0; i < n; i++) {
      xReal[i] += alpha * pReal[i];
      xImag[i] += alpha * pImag[i];
    }

    // r -= alpha * Ap
    for (let i = 0; i < n; i++) {
      rReal[i] -= alpha * ApR[i];
      rImag[i] -= alpha * ApI[i];
    }

    const rrNew = realInnerProduct(rReal, rImag, rReal, rImag);
    if (Math.sqrt(rrNew) < tol) break;

    const beta = rrNew / rr;
    for (let i = 0; i < n; i++) {
      pReal[i] = rReal[i] + beta * pReal[i];
      pImag[i] = rImag[i] + beta * pImag[i];
    }
    rr = rrNew;
  }

  return { real: xReal, imag: xImag };
}

/** Real part of the Hermitian inner product <u, v> = sum( conj(u_i) * v_i ) */
function realInnerProduct(
  uR: Float64Array, uI: Float64Array,
  vR: Float64Array, vI: Float64Array
): number {
  let s = 0;
  for (let i = 0; i < uR.length; i++) {
    s += uR[i] * vR[i] + uI[i] * vI[i]; // Re( conj(u)*v )
  }
  return s;
}

// ---------------------------------------------------------------------------
// Inverse power iteration to find the smallest eigenvector of L*x = λ M*x
// Uses CG to solve (L + σM) x_new = M x_old at each step.
// ---------------------------------------------------------------------------
export function solveEigenvector(
  L: SparseMatrix,
  M: SparseMatrix,
  maxIterations: number = 60,
  tolerance: number = 1e-5
): { real: Float64Array; imag: Float64Array } {
  const n = L.rows;

  // Regularisation shift (avoids singularity of L and skips the zero eigenvalue)
  const shift = 1e-6;

  // Initialise with random complex vector
  let vReal = new Float64Array(n);
  let vImag = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    vReal[i] = Math.random() - 0.5;
    vImag[i] = Math.random() - 0.5;
  }
  mNormalize(vReal, vImag, M);

  for (let iter = 0; iter < maxIterations; iter++) {
    // rhs = M * v
    const rhsR = new Float64Array(n);
    const rhsI = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const m = sparseGet(M, i, i).real;
      rhsR[i] = m * vReal[i];
      rhsI[i] = m * vImag[i];
    }

    // Solve (L + shift*M) * w = M * v
    const { real: wReal, imag: wImag } = complexCG(
      L, M, rhsR, rhsI, shift,
      /*maxIter=*/ 200, /*tol=*/ 1e-8
    );

    // Check convergence (angle between v and w after normalisation)
    const prevR = new Float64Array(vReal);
    const prevI = new Float64Array(vImag);

    vReal = wReal as Float64Array<ArrayBuffer>;
    vImag = wImag as Float64Array<ArrayBuffer>;
    mNormalize(vReal, vImag, M);

    // Remove constant (zero-eigenvalue) component
    let sumR = 0, sumI = 0, sumW = 0;
    for (let i = 0; i < n; i++) {
      const w = sparseGet(M, i, i).real;
      sumR += w * vReal[i];
      sumI += w * vImag[i];
      sumW += w;
    }
    if (sumW > 1e-10) {
      const cr = sumR / sumW;
      const ci = sumI / sumW;
      for (let i = 0; i < n; i++) {
        vReal[i] -= cr;
        vImag[i] -= ci;
      }
    }
    mNormalize(vReal, vImag, M);

    // Convergence test: 1 - |<v_old, v_new>|
    let dot = 0;
    for (let i = 0; i < n; i++) {
      dot += prevR[i] * vReal[i] + prevI[i] * vImag[i];
    }
    if (1 - Math.abs(dot) < tolerance) break;
  }

  return { real: vReal, imag: vImag };
}

function mNormalize(
  vReal: Float64Array,
  vImag: Float64Array,
  M: SparseMatrix
): void {
  let norm = 0;
  for (let i = 0; i < vReal.length; i++) {
    const m = sparseGet(M, i, i).real;
    norm += m * (vReal[i] * vReal[i] + vImag[i] * vImag[i]);
  }
  norm = Math.sqrt(norm);
  if (norm > 1e-12) {
    for (let i = 0; i < vReal.length; i++) {
      vReal[i] /= norm;
      vImag[i] /= norm;
    }
  }
}

// ---------------------------------------------------------------------------
// Extract phase from complex eigenvector
// ---------------------------------------------------------------------------
export function extractPhase(real: Float64Array, imag: Float64Array): Float64Array {
  const phase = new Float64Array(real.length);
  for (let i = 0; i < real.length; i++) {
    phase[i] = Math.atan2(imag[i], real[i]);
  }
  return phase;
}

// ---------------------------------------------------------------------------
// Trace isolines on the mesh
// ---------------------------------------------------------------------------
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

  for (let faceIdx = 0; faceIdx < mesh.faces.length; faceIdx++) {
    if (visitedFaces.has(faceIdx)) continue;

    const face = mesh.faces[faceIdx];
    const crossings = findPhaseCrossings(mesh, phase, face, targetPhase);

    if (crossings.length >= 2) {
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

    for (const crossing of crossings) {
      if (crossing.edgeIdx !== entryEdge) {
        points.push(crossing.point);
        faceIndices.push(currentFace);

        const heStart = mesh.faceToHalfEdge[currentFace];
        let heIdx = heStart;
        for (let i = 0; i < crossing.edgeIdx; i++) {
          heIdx = mesh.halfEdges[heIdx].next;
        }

        const twinIdx = mesh.halfEdges[heIdx].twin;
        if (twinIdx === -1) break;

        const nextFace = mesh.halfEdges[twinIdx].face;
        const nextFaceVerts = mesh.faces[nextFace];

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

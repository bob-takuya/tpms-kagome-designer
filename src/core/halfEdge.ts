import * as THREE from 'three';
import type { MeshData } from './marchingCubes';

export interface HalfEdge {
  vertex: number;
  face: number;
  next: number;
  prev: number;
  twin: number;
}

export interface HalfEdgeMesh {
  vertices: THREE.Vector3[];
  faces: number[][];
  halfEdges: HalfEdge[];
  vertexToHalfEdge: number[];
  faceToHalfEdge: number[];
  normals: THREE.Vector3[];
}

export function buildHalfEdgeMesh(meshData: MeshData): HalfEdgeMesh {
  const vertices: THREE.Vector3[] = [];
  const faces: number[][] = [];
  const halfEdges: HalfEdge[] = [];
  const vertexToHalfEdge: number[] = [];
  const faceToHalfEdge: number[] = [];
  const normals: THREE.Vector3[] = [];

  // Build vertices
  for (let i = 0; i < meshData.vertices.length; i += 3) {
    vertices.push(new THREE.Vector3(
      meshData.vertices[i],
      meshData.vertices[i + 1],
      meshData.vertices[i + 2]
    ));
    normals.push(new THREE.Vector3(
      meshData.normals[i],
      meshData.normals[i + 1],
      meshData.normals[i + 2]
    ));
  }

  // Build faces
  for (let i = 0; i < meshData.indices.length; i += 3) {
    faces.push([
      meshData.indices[i],
      meshData.indices[i + 1],
      meshData.indices[i + 2]
    ]);
  }

  // Initialize vertex to half-edge mapping
  for (let i = 0; i < vertices.length; i++) {
    vertexToHalfEdge[i] = -1;
  }

  // Build half-edges
  const edgeMap = new Map<string, number>();

  for (let faceIdx = 0; faceIdx < faces.length; faceIdx++) {
    const face = faces[faceIdx];
    const startHE = halfEdges.length;
    faceToHalfEdge[faceIdx] = startHE;

    for (let i = 0; i < 3; i++) {
      const heIdx = halfEdges.length;
      const v0 = face[i];
      const v1 = face[(i + 1) % 3];

      halfEdges.push({
        vertex: v1,
        face: faceIdx,
        next: startHE + (i + 1) % 3,
        prev: startHE + (i + 2) % 3,
        twin: -1
      });

      if (vertexToHalfEdge[v0] === -1) {
        vertexToHalfEdge[v0] = heIdx;
      }

      const edgeKey = `${Math.min(v0, v1)},${Math.max(v0, v1)}`;
      const existingHE = edgeMap.get(edgeKey);
      if (existingHE !== undefined) {
        halfEdges[heIdx].twin = existingHE;
        halfEdges[existingHE].twin = heIdx;
      } else {
        edgeMap.set(edgeKey, heIdx);
      }
    }
  }

  return {
    vertices,
    faces,
    halfEdges,
    vertexToHalfEdge,
    faceToHalfEdge,
    normals
  };
}

// Calculate cotangent weight for an edge
export function cotangentWeight(mesh: HalfEdgeMesh, heIdx: number): number {
  const he = mesh.halfEdges[heIdx];
  const twin = mesh.halfEdges[he.twin];

  if (he.twin === -1) {
    // Boundary edge - use single cotangent
    const prevHE = mesh.halfEdges[he.prev];
    const v0 = mesh.vertices[prevHE.vertex];
    const v1 = mesh.vertices[he.vertex];
    const vOpp = mesh.vertices[mesh.halfEdges[he.next].vertex];

    return singleCotangent(v0, vOpp, v1);
  }

  // Interior edge - sum of two cotangents
  const heNext = mesh.halfEdges[he.next];
  const twinNext = mesh.halfEdges[twin.next];

  const v0 = mesh.vertices[mesh.halfEdges[he.prev].vertex];
  const v1 = mesh.vertices[he.vertex];
  const vOpp1 = mesh.vertices[heNext.vertex];
  const vOpp2 = mesh.vertices[twinNext.vertex];

  return singleCotangent(v0, vOpp1, v1) + singleCotangent(v0, vOpp2, v1);
}

function singleCotangent(a: THREE.Vector3, b: THREE.Vector3, c: THREE.Vector3): number {
  const ab = new THREE.Vector3().subVectors(a, b);
  const cb = new THREE.Vector3().subVectors(c, b);

  const dot = ab.dot(cb);
  const cross = new THREE.Vector3().crossVectors(ab, cb);
  const crossLen = cross.length();

  if (crossLen < 1e-10) return 0;
  return dot / crossLen;
}

// Get all half-edges emanating from a vertex
export function getVertexHalfEdges(mesh: HalfEdgeMesh, vertexIdx: number): number[] {
  const result: number[] = [];
  const startHE = mesh.vertexToHalfEdge[vertexIdx];

  if (startHE === -1) return result;

  // Find all outgoing half-edges by circulating around the vertex
  let currentHE = startHE;
  const visited = new Set<number>();

  do {
    if (visited.has(currentHE)) break;
    visited.add(currentHE);

    // The previous half-edge starts at our vertex
    const prevHE = mesh.halfEdges[currentHE].prev;
    result.push(prevHE);

    // Move to twin's next to get next outgoing edge
    const twin = mesh.halfEdges[prevHE].twin;
    if (twin === -1) break;
    currentHE = mesh.halfEdges[twin].next;
  } while (currentHE !== startHE);

  return result;
}

// Get the vertex at the start of a half-edge
export function getHalfEdgeStart(mesh: HalfEdgeMesh, heIdx: number): number {
  const prevHE = mesh.halfEdges[heIdx].prev;
  return mesh.halfEdges[prevHE].vertex;
}

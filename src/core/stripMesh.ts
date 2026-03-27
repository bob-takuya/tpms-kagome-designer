/**
 * stripMesh.ts – Phase 3: Convert Strip centerlines into 3D ribbon meshes
 *
 * For each Strip:
 *   1. Sample the nearest mesh vertex normal at each centerline point
 *   2. Compute tangent (t) and bitangent (s = n × t) vectors
 *   3. Build a quad strip: left/right edges ± width/2 along s
 *   4. Apply over/under offset along n based on strip.layer
 */

import * as THREE from 'three';
import type { HalfEdgeMesh } from './halfEdge';
import type { Strip } from './kagome';

// Layer offset in world units:
//   layer=2 (over)    → +LAYER_OFFSET  (float above surface)
//   layer=1 (neutral) → 0
//   layer=0 (under)   → -LAYER_OFFSET  (sink below surface)
const LAYER_OFFSET = 0.03;

export interface StripMeshResult {
  geometry: THREE.BufferGeometry;
  family: number;
  layer: number;
  stripId: string;
}

/**
 * Build Three.js ribbon mesh for a single Strip.
 */
export function buildStripMesh(
  strip: Strip,
  mesh: HalfEdgeMesh,
  widthOverride?: number,
): StripMeshResult | null {
  const cl = strip.centerline;
  if (cl.length < 2) return null;

  const width = widthOverride ?? strip.width;
  if (width <= 0) return null;

  const halfW = width / 2;
  const layerOffset = (strip.layer - 1) * LAYER_OFFSET;  // -1, 0, +1 → offset

  const n = cl.length;
  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];

  // Build vertex rows
  for (let i = 0; i < n; i++) {
    const pt = cl[i];

    // Tangent: forward difference at start, backward at end, central elsewhere
    let tang: THREE.Vector3;
    if (i === 0) {
      tang = new THREE.Vector3().subVectors(cl[1], cl[0]).normalize();
    } else if (i === n - 1) {
      tang = new THREE.Vector3().subVectors(cl[n - 1], cl[n - 2]).normalize();
    } else {
      tang = new THREE.Vector3().subVectors(cl[i + 1], cl[i - 1]).normalize();
    }

    // Surface normal at this centerline point (nearest mesh vertex)
    const surfNormal = nearestVertexNormal(pt, mesh);

    // Bitangent (width direction): perpendicular to both tangent and surface normal
    const bitan = new THREE.Vector3().crossVectors(surfNormal, tang).normalize();
    if (bitan.length() < 1e-6) {
      // Degenerate: use arbitrary perpendicular
      bitan.set(1, 0, 0).crossVectors(surfNormal, new THREE.Vector3(1, 0, 0)).normalize();
    }

    // Base point with layer offset
    const base = pt.clone().addScaledVector(surfNormal, layerOffset);

    // Left and right edge vertices
    const left  = base.clone().addScaledVector(bitan, -halfW);
    const right = base.clone().addScaledVector(bitan,  halfW);

    positions.push(left.x,  left.y,  left.z);
    positions.push(right.x, right.y, right.z);

    // Use surface normal for both edge vertices
    normals.push(surfNormal.x, surfNormal.y, surfNormal.z);
    normals.push(surfNormal.x, surfNormal.y, surfNormal.z);
  }

  // Build quad strip indices
  // Row i → vertices [2i, 2i+1], Row i+1 → vertices [2i+2, 2i+3]
  for (let i = 0; i < n - 1; i++) {
    const tl = 2 * i;       // top-left  (left  edge of row i)
    const tr = 2 * i + 1;   // top-right (right edge of row i)
    const bl = 2 * i + 2;   // bottom-left  (left  edge of row i+1)
    const br = 2 * i + 3;   // bottom-right (right edge of row i+1)

    // Two triangles per quad (both faces visible → DoubleSide material)
    indices.push(tl, tr, br);
    indices.push(tl, br, bl);
  }

  if (indices.length === 0) return null;

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('normal',   new THREE.Float32BufferAttribute(normals, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals(); // smooth shading

  return {
    geometry,
    family: strip.family,
    layer: strip.layer,
    stripId: strip.id,
  };
}

/**
 * Build all strip meshes for a KagomePattern.
 * Returns an array of Three.js Mesh objects ready to add to the scene.
 */
export function buildAllStripMeshes(
  strips: Strip[],
  mesh: HalfEdgeMesh,
  familyColors: [string, string, string],
  widthOverride?: number,
): THREE.Mesh[] {
  const result: THREE.Mesh[] = [];

  for (const strip of strips) {
    const sm = buildStripMesh(strip, mesh, widthOverride);
    if (!sm) continue;

    const color = new THREE.Color(familyColors[strip.family]);

    // Over strips: slightly brighter + no transparency
    // Under strips: slightly darker + more transparent
    const opacity = strip.layer === 2 ? 1.0 : strip.layer === 1 ? 0.85 : 0.65;
    const emissive = strip.layer === 2
      ? color.clone().multiplyScalar(0.2)
      : new THREE.Color(0x000000);

    const material = new THREE.MeshPhongMaterial({
      color,
      emissive,
      side: THREE.DoubleSide,
      transparent: opacity < 1.0,
      opacity,
      shininess: 60,
    });

    const threeMesh = new THREE.Mesh(sm.geometry, material);
    (threeMesh as any).userData = { family: sm.family, layer: sm.layer, stripId: sm.stripId };
    result.push(threeMesh);
  }

  return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Find the nearest vertex in `mesh` to point `pt` and return its normal.
 * O(n_vertices) – acceptable for typical resolution-50 meshes (~50k vertices).
 */
function nearestVertexNormal(pt: THREE.Vector3, mesh: HalfEdgeMesh): THREE.Vector3 {
  let minDist2 = Infinity;
  let bestIdx = 0;

  // Sub-sample for speed if mesh is large
  const step = mesh.vertices.length > 20000 ? 3 : 1;

  for (let i = 0; i < mesh.vertices.length; i += step) {
    const d2 = mesh.vertices[i].distanceToSquared(pt);
    if (d2 < minDist2) {
      minDist2 = d2;
      bestIdx = i;
    }
  }

  return mesh.normals[bestIdx].clone().normalize();
}

/**
 * stripMesh.ts – Phase 3: Convert Strip centerlines into 3D ribbon meshes
 *
 * For each Strip:
 *   1. Look up the surface normal at each centerline point via a spatial hash
 *      (exact nearest-vertex lookup – no sub-sampling artefacts)
 *   2. Compute tangent (t) and bitangent (b = n × t) vectors per row
 *   3. Build a quad strip: left/right edges ± halfWidth along b
 *   4. Apply over/under offset along n (layer=2→+offset, 1→small, 0→0)
 */

import * as THREE from 'three';
import type { HalfEdgeMesh } from './halfEdge';
import type { Strip } from './kagome';

// Layer offsets: keep ribbons ON the surface (offset = 0).
// Physical over/under separation for fabrication is handled in the 2D unfold step.
// Here we rely entirely on polygonOffset in the material to render the ribbon
// in front of the TPMS mesh regardless of which side of the surface we're on.
const LAYER_OFFSET_OVER  = 0;
const LAYER_OFFSET_MID   = 0;
const LAYER_OFFSET_UNDER = 0;

// ─────────────────────────────────────────────────────────────────────────────
// Spatial hash for O(1) normal lookup
// ─────────────────────────────────────────────────────────────────────────────

interface NormalLookup {
  fn: (pt: THREE.Vector3) => THREE.Vector3;
  mesh: HalfEdgeMesh;
}

let _cachedLookup: NormalLookup | null = null;

function buildNormalLookup(mesh: HalfEdgeMesh): (pt: THREE.Vector3) => THREE.Vector3 {
  // Reuse cache if same mesh object
  if (_cachedLookup?.mesh === mesh) return _cachedLookup.fn;

  // Estimate cell size from average edge length (TPMS typically spans [-π, π])
  const CELL = 0.35;

  const grid = new Map<string, number[]>();
  for (let i = 0; i < mesh.vertices.length; i++) {
    const v = mesh.vertices[i];
    const k = `${Math.floor(v.x / CELL)},${Math.floor(v.y / CELL)},${Math.floor(v.z / CELL)}`;
    if (!grid.has(k)) grid.set(k, []);
    grid.get(k)!.push(i);
  }

  const fn = (pt: THREE.Vector3): THREE.Vector3 => {
    const cx = Math.floor(pt.x / CELL);
    const cy = Math.floor(pt.y / CELL);
    const cz = Math.floor(pt.z / CELL);

    let minD2 = Infinity;
    let best  = 0;

    for (let dx = -1; dx <= 1; dx++)
    for (let dy = -1; dy <= 1; dy++)
    for (let dz = -1; dz <= 1; dz++) {
      const bucket = grid.get(`${cx + dx},${cy + dy},${cz + dz}`);
      if (!bucket) continue;
      for (const idx of bucket) {
        const d2 = mesh.vertices[idx].distanceToSquared(pt);
        if (d2 < minD2) { minD2 = d2; best = idx; }
      }
    }

    return mesh.normals[best].clone().normalize();
  };

  _cachedLookup = { fn, mesh };
  return fn;
}

// ─────────────────────────────────────────────────────────────────────────────
// Single strip → ribbon mesh
// ─────────────────────────────────────────────────────────────────────────────

export interface StripMeshResult {
  geometry: THREE.BufferGeometry;
  family: number;
  layer: number;
  stripId: string;
}

export function buildStripMesh(
  strip: Strip,
  mesh: HalfEdgeMesh,
  widthOverride?: number,
): StripMeshResult | null {
  const cl = strip.centerline;
  if (cl.length < 2) {
    console.warn(`[StripMesh] ${strip.id}: centerline too short (${cl.length} pts) — skipped`);
    return null;
  }

  const rawWidth = widthOverride ?? strip.width;
  if (rawWidth <= 0) {
    console.warn(`[StripMesh] ${strip.id}: width=0 (strip.width=${strip.width}, override=${widthOverride}) — using min`);
  }
  const halfW = Math.max(rawWidth * 0.5, 0.01); // ensure minimum visible width

  // Over/under: all strips get a small positive offset so they float above the
  // surface and are always visible (never z-fighting with the TPMS mesh).
  const layerOffset =
    strip.layer === 2 ? LAYER_OFFSET_OVER :
    strip.layer === 0 ? LAYER_OFFSET_UNDER :
    LAYER_OFFSET_MID;

  const getNormal = buildNormalLookup(mesh);
  const n = cl.length;

  const positions: number[] = [];
  const vertNormals: number[] = [];
  const indices: number[] = [];

  let prevBitan = new THREE.Vector3(); // for continuity of the width direction

  for (let i = 0; i < n; i++) {
    const pt   = cl[i];
    const norm = getNormal(pt);

    // Tangent (central difference, clamped at endpoints)
    const pa = cl[Math.max(0, i - 1)];
    const pb = cl[Math.min(n - 1, i + 1)];
    const tang = new THREE.Vector3().subVectors(pb, pa);
    if (tang.lengthSq() < 1e-14) tang.copy(i > 0 ? new THREE.Vector3().subVectors(pt, cl[i - 1]) : new THREE.Vector3(1, 0, 0));
    tang.normalize();

    // Bitangent: lies in the tangent plane, perpendicular to both tang and norm
    const bitan = new THREE.Vector3().crossVectors(tang, norm).normalize();

    // Keep sign consistent with previous row to avoid ribbon twisting
    if (i > 0 && bitan.dot(prevBitan) < 0) bitan.negate();
    prevBitan.copy(bitan);

    // Base position: on the surface, offset along normal for layering
    const base = pt.clone().addScaledVector(norm, layerOffset);

    const left  = base.clone().addScaledVector(bitan, -halfW);
    const right = base.clone().addScaledVector(bitan,  halfW);

    positions.push(left.x,  left.y,  left.z);
    positions.push(right.x, right.y, right.z);

    vertNormals.push(norm.x, norm.y, norm.z, norm.x, norm.y, norm.z);
  }

  // Quad strip indices  (DoubleSide material → winding order doesn't matter)
  for (let i = 0; i < n - 1; i++) {
    const tl = 2 * i,     tr = 2 * i + 1;
    const bl = 2 * i + 2, br = 2 * i + 3;
    indices.push(tl, tr, br,  tl, br, bl);
  }

  if (indices.length === 0) return null;

  // ── Diagnostics ────────────────────────────────────────────────────────────
  let nanCount     = 0;
  let zeroRowCount = 0;
  for (let i = 0; i < n; i++) {
    const base = i * 6;
    const lx = positions[base],   ly = positions[base+1], lz = positions[base+2];
    const rx = positions[base+3], ry = positions[base+4], rz = positions[base+5];
    if (!isFinite(lx)||!isFinite(ly)||!isFinite(lz)||
        !isFinite(rx)||!isFinite(ry)||!isFinite(rz)) nanCount++;
    const rowWidth = Math.hypot(rx - lx, ry - ly, rz - lz);
    if (rowWidth < 1e-5) zeroRowCount++;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions,    3));
  geometry.setAttribute('normal',   new THREE.Float32BufferAttribute(vertNormals,  3));
  geometry.setIndex(indices);
  geometry.computeBoundingSphere();

  const bs = geometry.boundingSphere;
  const bsR = bs?.radius  ?? -1;
  const bsC = bs?.center;

  const ok = nanCount === 0 && zeroRowCount < n * 0.5 && bsR > 1e-6;
  const tag = ok ? '[strip ✓]' : '[strip ✗]';
  console[ok ? 'log' : 'warn'](
    `${tag} ${strip.id.padEnd(3)} ` +
    `clPts=${String(n).padStart(3)} halfW=${halfW.toFixed(3)} ` +
    `nan=${nanCount} zeroRow=${zeroRowCount}/${n} ` +
    `bs.r=${bsR.toFixed(3)} ` +
    `bs.c=(${bsC?.x.toFixed(2)},${bsC?.y.toFixed(2)},${bsC?.z.toFixed(2)})`,
  );
  // ── End diagnostics ────────────────────────────────────────────────────────

  return { geometry, family: strip.family, layer: strip.layer, stripId: strip.id };
}

// ─────────────────────────────────────────────────────────────────────────────
// All strips → array of THREE.Mesh
// ─────────────────────────────────────────────────────────────────────────────

export function buildAllStripMeshes(
  strips: Strip[],
  mesh: HalfEdgeMesh,
  familyColors: [string, string, string],
  widthOverride?: number,
): THREE.Mesh[] {
  const result: THREE.Mesh[] = [];
  let nullCount = 0;

  for (const strip of strips) {
    const sm = buildStripMesh(strip, mesh, widthOverride);
    if (!sm) { nullCount++; continue; }

    const base = new THREE.Color(familyColors[strip.family]);

    // Vary brightness by over/under (pure colour, no transparency)
    // Transparency in the ribbon material pushes objects into Three.js's
    // "transparent pass" which renders back-to-front; when a ribbon sits
    // inside the (also-transparent) TPMS surface the surface wins and the
    // ribbon disappears.  Keep ribbons fully opaque to stay in the opaque pass.
    const col =
      strip.layer === 2 ? base.clone() :
      strip.layer === 1 ? base.clone().multiplyScalar(0.85) :
      base.clone().multiplyScalar(0.60);

    // depthTest:false — ribbons on the "back" side of the TPMS were being
    // occluded by ribbons on the "front" side (depth buffer conflict).
    // With depthTest off, all ribbons always render.
    // renderOrder:1 — ensures ribbons draw AFTER the transparent TPMS surface.
    const mat = new THREE.MeshBasicMaterial({
      color: col,
      side: THREE.DoubleSide,
      transparent: true,   // must be true when depthTest:false for correct Three.js pass
      opacity: 1.0,
      depthTest: false,
      depthWrite: false,
    });

    const m = new THREE.Mesh(sm.geometry, mat);
    m.renderOrder = 1;
    m.userData = { family: sm.family, layer: sm.layer, stripId: sm.stripId };
    result.push(m);
  }

  console.log(`[StripMesh] buildAllStripMeshes: ${result.length} meshes built, ${nullCount} skipped`);
  if (nullCount > 0) {
    const skipped = strips
      .filter(s => !result.find(m => m.userData.stripId === s.id))
      .map(s => `${s.id}(cl=${s.centerline.length},w=${s.width.toFixed(3)})`);
    console.warn(`[StripMesh] Skipped: ${skipped.join(', ')}`);
  }

  return result;
}

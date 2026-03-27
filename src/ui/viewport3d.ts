import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { store } from '../store';
import { createTPMSFunction } from '../core/tpms';
import { marchingCubes } from '../core/marchingCubes';
import type { MeshData } from '../core/marchingCubes';
import { buildHalfEdgeMesh } from '../core/halfEdge';
import type { HalfEdgeMesh } from '../core/halfEdge';
import { computeStripeField, traceIsolines } from '../core/connectionLaplacian';
import type { Isoline } from '../core/connectionLaplacian';
import { buildKagomePattern } from '../core/kagome';
import type { KagomePattern } from '../core/kagome';
import { buildAllStripMeshes } from '../core/stripMesh';

export interface Viewport3DContext {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: OrbitControls;
  surfaceMesh: THREE.Mesh | null;
  /** Group holding all dynamic overlays (isolines, strip meshes, junctions) */
  stripMeshes: THREE.Group;
  halfEdgeMesh: HalfEdgeMesh | null;
  isolinesByFamily: [Isoline[], Isoline[], Isoline[]] | null;
  stripeFields: [Float64Array, Float64Array, Float64Array] | null;
  kagomePattern: KagomePattern | null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Scene setup
// ─────────────────────────────────────────────────────────────────────────────

export function createViewport3D(container: HTMLElement): Viewport3DContext {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a1a);

  const camera = new THREE.PerspectiveCamera(
    60,
    container.clientWidth / container.clientHeight,
    0.1,
    1000,
  );
  camera.position.set(5, 5, 5);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;

  // Lighting – warm key + fill + cool back
  scene.add(new THREE.AmbientLight(0x404040, 0.6));
  const key = new THREE.DirectionalLight(0xfff5e0, 1.0);
  key.position.set(5, 10, 5);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xe0f0ff, 0.5);
  fill.position.set(-5, -5, -5);
  scene.add(fill);

  scene.add(new THREE.GridHelper(10, 10, 0x444444, 0x333333));
  scene.add(new THREE.AxesHelper(2));

  const stripMeshes = new THREE.Group();
  scene.add(stripMeshes);

  const ctx: Viewport3DContext = {
    scene,
    camera,
    renderer,
    controls,
    surfaceMesh: null,
    stripMeshes,
    halfEdgeMesh: null,
    isolinesByFamily: null,
    stripeFields: null,
    kagomePattern: null,
  };

  regenerateMesh(ctx);

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  window.addEventListener('resize', () => {
    const w = container.clientWidth;
    const h = container.clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });

  return ctx;
}

// ─────────────────────────────────────────────────────────────────────────────
// Mesh regeneration (Stage 1 → Stage 2 → Stage 3)
// ─────────────────────────────────────────────────────────────────────────────

export function regenerateMesh(ctx: Viewport3DContext): void {
  const state = store.getState();

  // Remove old surface mesh
  if (ctx.surfaceMesh) {
    ctx.scene.remove(ctx.surfaceMesh);
    ctx.surfaceMesh.geometry.dispose();
    (ctx.surfaceMesh.material as THREE.Material).dispose();
  }

  // Stage 1 – TPMS + Marching Cubes
  const implicitFunc = createTPMSFunction(
    state.tpms.surfaceType,
    state.tpms.baseT,
    state.noise.enabled,
    state.noise.amplitude,
    state.noise.frequency,
    state.noise.seed,
  );

  const meshData = marchingCubes(
    implicitFunc,
    state.tpms.boundingBox.min,
    state.tpms.boundingBox.max,
    state.tpms.gridResolution,
    state.tpms.period,
  );

  ctx.halfEdgeMesh = buildHalfEdgeMesh(meshData);

  const geometry = createGeometry(meshData);
  const material = new THREE.MeshPhongMaterial({
    color: 0x3a6faa,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.25,   // semi-transparent so strips are clearly visible
    shininess: 40,
  });

  ctx.surfaceMesh = new THREE.Mesh(geometry, material);
  ctx.scene.add(ctx.surfaceMesh);

  store.getState().setStats({
    vertices: meshData.vertices.length / 3,
    faces: meshData.indices.length / 3,
  });

  regeneratePattern(ctx);
}

export function regeneratePattern(ctx: Viewport3DContext): void {
  if (!ctx.halfEdgeMesh) return;

  const state = store.getState();
  const mesh = ctx.halfEdgeMesh;

  // Clear previous overlays
  clearGroup(ctx.stripMeshes);

  // ── Stage 2 – Stripe fields + isoline tracing ─────────────────────────────
  const STRIP_DENSITY = 4.0;
  const stripeFields: [Float64Array, Float64Array, Float64Array] = [
    computeStripeField(mesh, 0,                   STRIP_DENSITY),
    computeStripeField(mesh, Math.PI / 3,         STRIP_DENSITY),
    computeStripeField(mesh, (2 * Math.PI) / 3,   STRIP_DENSITY),
  ];
  ctx.stripeFields = stripeFields;

  const isolinesByFamily: [Isoline[], Isoline[], Isoline[]] = [[], [], []];
  for (let k = 0; k < 3; k++) {
    isolinesByFamily[k] = traceIsolines(mesh, stripeFields[k], state.strip.numIsolines);
  }
  ctx.isolinesByFamily = isolinesByFamily;

  // ── Stage 3 – Kagome strip extraction ─────────────────────────────────────
  ctx.kagomePattern = buildKagomePattern(
    mesh,
    stripeFields,
    isolinesByFamily,
    state.strip.numIsolines,
    state.kagome.holeRadius,
  );

  const { strips, junctions } = ctx.kagomePattern;

  store.getState().setStats({
    strips: strips.length,
    junctions: junctions.length,
  });

  // ── Stage 3 visualization ──────────────────────────────────────────────────

  // 3a. Ribbon meshes for each strip (with over/under layer offsets)
  //     Width fallback: use store's stripWidth if per-strip width is zero
  const widthFallback = state.strip.stripWidth;
  const stripThreeMeshes = buildAllStripMeshes(
    strips,
    mesh,
    state.kagome.layerColors,
    widthFallback,
  );
  for (const m of stripThreeMeshes) ctx.stripMeshes.add(m);

  // 3b. Junction visualisation: coloured rings + ID labels
  for (const junc of junctions) {
    renderJunction(ctx.stripMeshes, junc, state.kagome.layerColors);
  }

  // 3c. Light isoline underlay (thin, semi-transparent) for reference
  for (let k = 0; k < 3; k++) {
    const col = new THREE.Color(state.kagome.layerColors[k]).multiplyScalar(0.4);
    for (const iso of isolinesByFamily[k]) {
      if (iso.points.length < 2) continue;
      const geo = new THREE.BufferGeometry().setFromPoints(iso.points);
      const mat = new THREE.LineBasicMaterial({ color: col, linewidth: 1 });
      mat.transparent = true;
      mat.opacity = 0.35;
      ctx.stripMeshes.add(new THREE.LineSegments(geo, mat));
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Junction rendering
// ─────────────────────────────────────────────────────────────────────────────

function renderJunction(
  group: THREE.Group,
  junc: KagomePattern['junctions'][number],
  familyColors: [string, string, string],
): void {
  const overColor   = new THREE.Color(familyColors[junc.overFamily]);
  const underColor  = new THREE.Color(familyColors[junc.underFamily]);
  const r = junc.holeRadius;

  // Outer ring (over-family colour)
  addCircle(group, junc.position, junc.normal, r * 2.0, overColor, 2);
  // Inner ring (under-family colour)
  addCircle(group, junc.position, junc.normal, r * 1.2, underColor, 1);

  // Centre dot
  const dotGeo = new THREE.SphereGeometry(r * 0.4, 6, 6);
  const dotMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const dot = new THREE.Mesh(dotGeo, dotMat);
  dot.position.copy(junc.position).addScaledVector(junc.normal, r * 0.5);
  group.add(dot);
}

function addCircle(
  group: THREE.Group,
  center: THREE.Vector3,
  normal: THREE.Vector3,
  radius: number,
  color: THREE.Color,
  linewidth: number,
  segments = 32,
): void {
  // Build local 2D frame
  const ref = Math.abs(normal.x) < 0.9
    ? new THREE.Vector3(1, 0, 0)
    : new THREE.Vector3(0, 1, 0);
  const t1 = new THREE.Vector3().crossVectors(normal, ref).normalize();
  const t2 = new THREE.Vector3().crossVectors(normal, t1).normalize();

  const pts: THREE.Vector3[] = [];
  for (let i = 0; i <= segments; i++) {
    const θ = (2 * Math.PI * i) / segments;
    pts.push(
      center.clone()
        .addScaledVector(t1, Math.cos(θ) * radius)
        .addScaledVector(t2, Math.sin(θ) * radius),
    );
  }

  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  const mat = new THREE.LineBasicMaterial({ color, linewidth });
  group.add(new THREE.Line(geo, mat));
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function clearGroup(group: THREE.Group): void {
  while (group.children.length > 0) {
    const child = group.children[0];
    group.remove(child);
    if ((child as THREE.Mesh).geometry) (child as THREE.Mesh).geometry.dispose();
    const mat = (child as THREE.Mesh).material;
    if (mat) {
      if (Array.isArray(mat)) mat.forEach(m => m.dispose());
      else (mat as THREE.Material).dispose();
    }
  }
}

function createGeometry(meshData: MeshData): THREE.BufferGeometry {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(meshData.vertices, 3));
  geo.setAttribute('normal',   new THREE.BufferAttribute(meshData.normals, 3));
  geo.setIndex(new THREE.BufferAttribute(meshData.indices, 1));
  return geo;
}

/** Update colours without full recomputation */
export function updateColors(ctx: Viewport3DContext): void {
  // Full regeneration is the safest approach given the interleaved layers
  regeneratePattern(ctx);
}

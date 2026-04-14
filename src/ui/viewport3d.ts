import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { store } from '../store';
import { createTPMSFunction } from '../core/tpms';
import { marchingCubes } from '../core/marchingCubes';
import type { MeshData } from '../core/marchingCubes';
import { buildHalfEdgeMesh } from '../core/halfEdge';
import type { HalfEdgeMesh } from '../core/halfEdge';
import type { Isoline } from '../core/connectionLaplacian';
import { computeGeodesicFoliationIsolines } from '../core/foliationPipeline';
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

  // ── Click → raycast strip selection ─────────────────────────────────────
  setupStripClick(ctx, renderer.domElement, camera);

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
    opacity: 0.15,        // low opacity → strips always dominate visually
    depthWrite: false,    // transparent surface never wins depth test vs ribbons
    shininess: 40,
  });

  ctx.surfaceMesh = new THREE.Mesh(geometry, material);
  ctx.scene.add(ctx.surfaceMesh);

  store.getState().setStats({
    vertices: meshData.vertices.length / 3,
    faces: meshData.indices.length / 3,
  });

  // Clear any prior ribbon overlays; leave the semi-transparent TPMS
  // surface visible so the user can preview the shape before choosing
  // to run the (expensive) WGF pattern computation. The user triggers
  // that step explicitly via the "Generate Pattern" button.
  clearGroup(ctx.stripMeshes);
  ctx.kagomePattern = null;
  ctx.isolinesByFamily = null;
  ctx.stripeFields = null;
  store.getState().setStats({ strips: 0, junctions: 0 });
}

export async function regeneratePattern(
  ctx: Viewport3DContext,
  onProgress?: (p: { stage: number; cur: number; total: number; label: string }) => void,
): Promise<void> {
  if (!ctx.halfEdgeMesh) return;

  const state = store.getState();
  const mesh = ctx.halfEdgeMesh;

  // Clear previous overlays
  clearGroup(ctx.stripMeshes);

  // ── Stage 2 – Vekhter et al. 2019 "Weaving Geodesic Foliations" ──────
  //
  // Runs in a dedicated Web Worker (src/workers/wgfWorker.ts). The
  // worker hosts the WASM module and streams phase-by-phase progress
  // via `self.postMessage({ type: 'wgf-progress', ... })` callbacks
  // emitted from the C++ pipeline. The main thread stays responsive
  // for the entire run.
  const wgf = await computeGeodesicFoliationIsolines(
    mesh,
    {
      lambdaInit:  1000,
      lambdaMin:   1e-3,
      alg1MaxIter: 50,
      mu:          1e-4,
      jointIters:  10,
      userScale:   state.strip.numIsolines / 8.0, // 8 ≈ default density
      useCover:    true,
    },
    onProgress,
  );

  // Downstream code expects a (Re,Im,amplitude) triple per family and an
  // isoline-per-family list. We only populate the isolines — the stripe
  // fields go unused by kagome.ts when fed pre-traced isolines.
  const isolinesByFamily = wgf.isolinesByFamily;
  const stripeFields: [Float64Array, Float64Array, Float64Array] = [
    new Float64Array(mesh.vertices.length),
    new Float64Array(mesh.vertices.length),
    new Float64Array(mesh.vertices.length),
  ];
  ctx.stripeFields = stripeFields;
  ctx.isolinesByFamily = isolinesByFamily;

  // ── Stage 3 – Kagome strip extraction ─────────────────────────────────────
  // Convert mm values to world units using the physical scale
  const scale = state.develop.scale;   // mm per world unit
  const holeRadiusWorld  = state.kagome.holeRadiusMm  / scale;
  const stripWidthWorld  = state.strip.stripWidthMm   / scale;

  ctx.kagomePattern = buildKagomePattern(
    mesh,
    stripeFields,
    isolinesByFamily,
    state.strip.numIsolines,
    holeRadiusWorld,
    state.strip.method,
    state.strip.widthRatio,
    stripWidthWorld,
  );

  const { strips, junctions } = ctx.kagomePattern;

  store.getState().setStats({
    strips: strips.length,
    junctions: junctions.length,
  });

  // ── Stage 3 visualization ──────────────────────────────────────────────────

  // 3a. Ribbon meshes for each strip (with over/under layer offsets)
  const widthFallback = stripWidthWorld;
  const stripThreeMeshes = buildAllStripMeshes(
    strips,
    mesh,
    state.kagome.layerColors,
    widthFallback,
  );

  console.group(`[Viewport] adding ${stripThreeMeshes.length} strip meshes to scene`);
  for (const m of stripThreeMeshes) {
    const ud  = m.userData as { stripId: string; family: number; layer: number };
    const geo = m.geometry as THREE.BufferGeometry;
    geo.computeBoundingSphere();
    const bs  = geo.boundingSphere;
    const mat = m.material as THREE.MeshBasicMaterial;
    console.log(
      `  ${ud.stripId.padEnd(3)} visible=${m.visible} ` +
      `frustumCulled=${m.frustumCulled} ` +
      `color=#${mat.color.getHexString()} ` +
      `opacity=${mat.opacity} transparent=${mat.transparent} ` +
      `bs.r=${bs?.radius?.toFixed(3) ?? 'null'}`,
    );
    ctx.stripMeshes.add(m);
  }
  console.log(`  → stripMeshes group total children: ${ctx.stripMeshes.children.length}`);
  console.groupEnd();

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

/** Update colours without recomputing the heavy WGF pipeline.
 *  If no pattern exists yet this is a no-op — the user must press
 *  "Generate Pattern" to trigger the initial computation. */
export function updateColors(ctx: Viewport3DContext): void {
  if (!ctx.kagomePattern) return;
  // TODO: swap materials in-place instead of rebuilding; for now the
  // cheapest correct behaviour is simply to leave the existing
  // geometry in place. Ribbon colours are baked into MeshBasicMaterial
  // at build time so this won't reflect colour-picker changes until
  // the next Generate Pattern press, which is acceptable UX.
}

// ─────────────────────────────────────────────────────────────────────────────
// Strip click detection (raycast) + highlight
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Distinguish click from orbit-drag: record mousedown position, and only
 * treat mouseup as a click if the cursor moved less than 4 px.
 */
function setupStripClick(
  ctx: Viewport3DContext,
  domElement: HTMLCanvasElement,
  camera: THREE.PerspectiveCamera,
): void {
  const raycaster = new THREE.Raycaster();
  const mouse     = new THREE.Vector2();
  const downPos   = new THREE.Vector2();

  domElement.addEventListener('mousedown', (e) => {
    downPos.set(e.clientX, e.clientY);
  });

  domElement.addEventListener('mouseup', (e) => {
    const dx = e.clientX - downPos.x;
    const dy = e.clientY - downPos.y;
    if (dx * dx + dy * dy > 16) return; // drag → ignore

    const rect = domElement.getBoundingClientRect();
    mouse.x =  ((e.clientX - rect.left) / rect.width)  * 2 - 1;
    mouse.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);

    // Only test strip ribbon meshes (they carry userData.stripId)
    const targets = ctx.stripMeshes.children.filter(
      (c): c is THREE.Mesh => c instanceof THREE.Mesh && !!c.userData.stripId,
    );
    const hits = raycaster.intersectObjects(targets, false);

    if (hits.length > 0) {
      const stripId = hits[0].object.userData.stripId as string;
      highlightStrip(ctx, stripId);
      window.dispatchEvent(new CustomEvent('strip-selected', { detail: { stripId } }));
    } else {
      highlightStrip(ctx, null);
      window.dispatchEvent(new CustomEvent('strip-deselected'));
    }
  });
}

/**
 * Dim all strips except the selected one.  Pass `null` to reset.
 */
export function highlightStrip(ctx: Viewport3DContext, stripId: string | null): void {
  for (const child of ctx.stripMeshes.children) {
    if (!(child instanceof THREE.Mesh) || !child.userData.stripId) continue;
    const mat = child.material as THREE.MeshBasicMaterial;
    if (stripId === null) {
      // Reset all to full opacity
      mat.opacity = 1.0;
    } else {
      mat.opacity = child.userData.stripId === stripId ? 1.0 : 0.12;
    }
  }
}

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

export interface Viewport3DContext {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: OrbitControls;
  surfaceMesh: THREE.Mesh | null;
  stripMeshes: THREE.Group;
  halfEdgeMesh: HalfEdgeMesh | null;
  isolinesByFamily: [Isoline[], Isoline[], Isoline[]] | null;
  stripeFields: [Float64Array, Float64Array, Float64Array] | null;
  kagomePattern: KagomePattern | null;
}

export function createViewport3D(container: HTMLElement): Viewport3DContext {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a1a);

  const camera = new THREE.PerspectiveCamera(
    60,
    container.clientWidth / container.clientHeight,
    0.1,
    1000
  );
  camera.position.set(5, 5, 5);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;

  // Lighting
  const ambientLight = new THREE.AmbientLight(0x404040, 0.5);
  scene.add(ambientLight);

  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
  directionalLight.position.set(5, 10, 5);
  scene.add(directionalLight);

  const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
  directionalLight2.position.set(-5, -5, -5);
  scene.add(directionalLight2);

  // Grid helper
  const gridHelper = new THREE.GridHelper(10, 10, 0x444444, 0x333333);
  scene.add(gridHelper);

  // Axes helper
  const axesHelper = new THREE.AxesHelper(2);
  scene.add(axesHelper);

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

  // Initial mesh generation
  regenerateMesh(ctx);

  // Animation loop
  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  // Handle resize
  window.addEventListener('resize', () => {
    const width = container.clientWidth;
    const height = container.clientHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
  });

  return ctx;
}

export function regenerateMesh(ctx: Viewport3DContext): void {
  const state = store.getState();

  // Remove old mesh
  if (ctx.surfaceMesh) {
    ctx.scene.remove(ctx.surfaceMesh);
    ctx.surfaceMesh.geometry.dispose();
    (ctx.surfaceMesh.material as THREE.Material).dispose();
  }

  // Create TPMS function
  const implicitFunc = createTPMSFunction(
    state.tpms.surfaceType,
    state.tpms.baseT,
    state.noise.enabled,
    state.noise.amplitude,
    state.noise.frequency,
    state.noise.seed
  );

  // Generate mesh using marching cubes
  const meshData = marchingCubes(
    implicitFunc,
    state.tpms.boundingBox.min,
    state.tpms.boundingBox.max,
    state.tpms.gridResolution,
    state.tpms.period
  );

  // Build half-edge mesh for later processing
  ctx.halfEdgeMesh = buildHalfEdgeMesh(meshData);

  // Create Three.js mesh
  const geometry = createGeometry(meshData);
  const material = new THREE.MeshPhongMaterial({
    color: 0x4488ff,
    side: THREE.DoubleSide,
    flatShading: false,
    transparent: true,
    opacity: 0.8,
  });

  ctx.surfaceMesh = new THREE.Mesh(geometry, material);
  ctx.scene.add(ctx.surfaceMesh);

  // Update stats
  store.getState().setStats({
    vertices: meshData.vertices.length / 3,
    faces: meshData.indices.length / 3,
  });

  // Generate stripe pattern
  regeneratePattern(ctx);
}

export function regeneratePattern(ctx: Viewport3DContext): void {
  if (!ctx.halfEdgeMesh) return;

  const state = store.getState();

  // Clear old strip meshes
  while (ctx.stripMeshes.children.length > 0) {
    const child = ctx.stripMeshes.children[0];
    ctx.stripMeshes.remove(child);
    if (child instanceof THREE.Line) {
      child.geometry.dispose();
      (child.material as THREE.Material).dispose();
    }
  }

  // ── Phase 2: stripe fields + isoline tracing ───────────────────────────────
  const STRIP_DENSITY = 4.0;
  const stripeFields: [Float64Array, Float64Array, Float64Array] = [
    computeStripeField(ctx.halfEdgeMesh, 0,                 STRIP_DENSITY),
    computeStripeField(ctx.halfEdgeMesh, Math.PI / 3,       STRIP_DENSITY),
    computeStripeField(ctx.halfEdgeMesh, (2 * Math.PI) / 3, STRIP_DENSITY),
  ];
  ctx.stripeFields = stripeFields;

  const isolinesByFamily: [Isoline[], Isoline[], Isoline[]] = [[], [], []];

  for (let family = 0; family < 3; family++) {
    const isolines = traceIsolines(ctx.halfEdgeMesh, stripeFields[family], state.strip.numIsolines);
    isolinesByFamily[family] = isolines;

    // Render centerlines as LineSegments (segment pairs from marching triangles)
    const color = new THREE.Color(state.kagome.layerColors[family]);
    for (const iso of isolines) {
      if (iso.points.length < 2) continue;
      const geo = new THREE.BufferGeometry().setFromPoints(iso.points);
      const mat = new THREE.LineBasicMaterial({ color, linewidth: 2 });
      ctx.stripMeshes.add(new THREE.LineSegments(geo, mat));
    }
  }
  ctx.isolinesByFamily = isolinesByFamily;

  // ── Phase 3: Kagome strip extraction ───────────────────────────────────────
  ctx.kagomePattern = buildKagomePattern(
    ctx.halfEdgeMesh,
    stripeFields,
    isolinesByFamily,
    state.strip.numIsolines,
    state.kagome.holeRadius,
  );

  store.getState().setStats({
    strips: ctx.kagomePattern.strips.length,
    junctions: ctx.kagomePattern.junctions.length,
  });

  // ── Phase 3 visualization ──────────────────────────────────────────────────

  // 3a. Render stitched strip centerlines as thicker colored polylines
  //     (over strips rendered with slight normal offset; under strips darker)
  const OVER_OFFSET  = 0.015;
  const UNDER_ALPHA  = 0.45;

  for (const strip of ctx.kagomePattern.strips) {
    if (strip.centerline.length < 2) continue;

    const isOver  = strip.layer === 2;
    const baseColor = new THREE.Color(state.kagome.layerColors[strip.family]);
    const color = isOver ? baseColor : baseColor.clone().multiplyScalar(UNDER_ALPHA);

    const pts = strip.centerline.map(p => {
      if (!isOver) return p.clone();
      // Offset over-strips along the closest vertex normal
      return p.clone().addScaledVector(ctx.halfEdgeMesh!.normals[0], OVER_OFFSET);
    });

    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const mat = new THREE.LineBasicMaterial({ color, linewidth: 3 });
    ctx.stripMeshes.add(new THREE.Line(geo, mat));
  }

  // 3b. Render junction holes as flat rings on the surface
  for (const junc of ctx.kagomePattern.junctions) {
    const overColor  = new THREE.Color(state.kagome.layerColors[junc.overFamily]);
    const underColor = new THREE.Color(state.kagome.layerColors[junc.underFamily]);

    // Outer ring: over-family color
    addJunctionRing(ctx.stripMeshes, junc.position, junc.normal,
      junc.holeRadius * 1.6, junc.holeRadius * 1.2, overColor);

    // Inner disc: under-family color (the "hole")
    addJunctionDisc(ctx.stripMeshes, junc.position, junc.normal,
      junc.holeRadius, underColor);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Junction visualization helpers
// ─────────────────────────────────────────────────────────────────────────────

function buildCirclePoints(
  center: THREE.Vector3,
  normal: THREE.Vector3,
  radius: number,
  segments = 24,
): THREE.Vector3[] {
  // Build a local 2D frame in the plane perpendicular to `normal`
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
  return pts;
}

function addJunctionRing(
  group: THREE.Group,
  center: THREE.Vector3,
  normal: THREE.Vector3,
  outerR: number,
  innerR: number,
  color: THREE.Color,
): void {
  // Two concentric circle lines
  for (const r of [outerR, innerR]) {
    const pts = buildCirclePoints(center, normal, r);
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const mat = new THREE.LineBasicMaterial({ color, linewidth: 2 });
    group.add(new THREE.Line(geo, mat));
  }
}

function addJunctionDisc(
  group: THREE.Group,
  center: THREE.Vector3,
  normal: THREE.Vector3,
  radius: number,
  color: THREE.Color,
): void {
  // Filled disc using a circle outline (small sphere as center marker too)
  const pts = buildCirclePoints(center, normal, radius);
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  const mat = new THREE.LineBasicMaterial({ color, linewidth: 1 });
  group.add(new THREE.Line(geo, mat));

  // Small sphere at the center
  const sphereGeo = new THREE.SphereGeometry(radius * 0.35, 8, 8);
  const sphereMat = new THREE.MeshBasicMaterial({ color });
  const sphere = new THREE.Mesh(sphereGeo, sphereMat);
  sphere.position.copy(center);
  group.add(sphere);
}

function createGeometry(meshData: MeshData): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();

  geometry.setAttribute('position', new THREE.BufferAttribute(meshData.vertices, 3));
  geometry.setAttribute('normal', new THREE.BufferAttribute(meshData.normals, 3));
  geometry.setIndex(new THREE.BufferAttribute(meshData.indices, 1));

  return geometry;
}

export function updateColors(ctx: Viewport3DContext): void {
  if (!ctx.isolinesByFamily) return;

  const state = store.getState();

  // Update line colors
  let lineIndex = 0;
  for (let family = 0; family < 3; family++) {
    const color = new THREE.Color(state.kagome.layerColors[family]);

    for (const _ of ctx.isolinesByFamily[family]) {
      const line = ctx.stripMeshes.children[lineIndex];
      if (line instanceof THREE.Line) {
        (line.material as THREE.LineBasicMaterial).color = color;
      }
      lineIndex++;
    }
  }
}

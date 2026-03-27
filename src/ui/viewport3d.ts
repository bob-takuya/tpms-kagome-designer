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
import { extractKagomeStrips, assignLayers } from '../core/kagome';
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

  // Generate isolines for 3 families (0°, 60°, 120°)
  // stripe density: controls how many stripes appear across the surface
  const stripDensity = 4.0;
  const isolinesByFamily: [Isoline[], Isoline[], Isoline[]] = [[], [], []];

  for (let family = 0; family < 3; family++) {
    // Each family uses a DIFFERENT angle → different Poisson RHS → different stripes
    const phi = (family * Math.PI) / 3;

    // Compute the scalar stripe field f for this family
    const stripeField = computeStripeField(ctx.halfEdgeMesh, phi, stripDensity);

    // Trace isolines (marching triangles)
    const isolines = traceIsolines(ctx.halfEdgeMesh, stripeField, state.strip.numIsolines);
    isolinesByFamily[family] = isolines;

    // Render as LineSegments (pairs of points per face crossing)
    const color = new THREE.Color(state.kagome.layerColors[family]);

    for (const isoline of isolines) {
      if (isoline.points.length < 2) continue;

      // points are stored as segment pairs: [A0,B0, A1,B1, …]
      const lineGeometry = new THREE.BufferGeometry().setFromPoints(isoline.points);
      const lineMaterial = new THREE.LineBasicMaterial({ color, linewidth: 2 });
      // Use LineSegments so disconnected pairs are rendered correctly
      const line = new THREE.LineSegments(lineGeometry, lineMaterial);
      ctx.stripMeshes.add(line);
    }
  }

  ctx.isolinesByFamily = isolinesByFamily;

  // Extract Kagome strips
  ctx.kagomePattern = extractKagomeStrips(
    ctx.halfEdgeMesh,
    isolinesByFamily,
    state.strip.widthRatio,
    state.kagome.holeRadius
  );

  assignLayers(ctx.kagomePattern);

  // Update stats
  store.getState().setStats({
    strips: ctx.kagomePattern.strips.length,
    junctions: ctx.kagomePattern.junctions.length,
  });

  // Visualize junctions
  const junctionMaterial = new THREE.MeshBasicMaterial({ color: 0x00ffff });
  for (const junction of ctx.kagomePattern.junctions) {
    const sphereGeometry = new THREE.SphereGeometry(state.kagome.holeRadius, 8, 8);
    const sphere = new THREE.Mesh(sphereGeometry, junctionMaterial);
    sphere.position.copy(junction.position);
    ctx.stripMeshes.add(sphere);
  }
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

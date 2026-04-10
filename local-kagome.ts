/**
 * local-kagome.ts
 * ブラウザなしでカゴメパターンを生成し、3ファミリのリボン長さを診断する。
 *
 * 実行: npx tsx local-kagome.ts [surface] [numStripes] [resolution]
 * 例:  npx tsx local-kagome.ts gyroid 4 30
 */

import { marchingCubes } from './src/core/marchingCubes.js';
import { buildHalfEdgeMesh } from './src/core/halfEdge.js';
import { computeStripeField, traceIsolines } from './src/core/connectionLaplacian.js';
import { buildKagomePattern } from './src/core/kagome.js';

type SurfaceType = 'gyroid' | 'schwarzP' | 'schwarzD';

function createTPMSFunction(type: SurfaceType, t = 0.0) {
  const fns: Record<SurfaceType, (x: number, y: number, z: number) => number> = {
    gyroid:   (x,y,z) => Math.sin(x)*Math.cos(y) + Math.sin(y)*Math.cos(z) + Math.sin(z)*Math.cos(x),
    schwarzP: (x,y,z) => Math.cos(x) + Math.cos(y) + Math.cos(z),
    schwarzD: (x,y,z) => Math.cos(x)*Math.cos(y)*Math.cos(z) - Math.sin(x)*Math.sin(y)*Math.sin(z),
  };
  const base = fns[type];
  return (x: number, y: number, z: number) => base(x,y,z) - t;
}

// ─── Args (browser defaults) ──────────────────────────────────────────────────
const surface    = (process.argv[2] as SurfaceType) ?? 'gyroid';
const numStripes = parseInt(process.argv[3] ?? '8', 10);   // browser default: 8
const resolution = parseInt(process.argv[4] ?? '50', 10);  // browser default: 50
const tOffset    = parseFloat(process.argv[5] ?? '0.0');
const STRIP_DENSITY   = 4.0;
const holeRadiusMm    = 2;
const scaleWorld      = 50;  // mm per world unit
const holeRadiusWorld = holeRadiusMm / scaleWorld;  // 0.04

console.log(`\n${'='.repeat(54)}`);
console.log(`  TPMS Kagome — Local Diagnostic`);
console.log(`  Surface: ${surface}  t=${tOffset}  res=${resolution}  stripes=${numStripes}`);
console.log(`${'='.repeat(54)}\n`);

// ─── 1. TPMS mesh ─────────────────────────────────────────────────────────────
console.log('[1/5] Marching cubes...');
const fn = createTPMSFunction(surface, tOffset);
const meshData = marchingCubes(fn, -Math.PI, Math.PI, resolution, 2 * Math.PI);
console.log(`      vertices=${meshData.vertices.length/3}  triangles=${meshData.indices.length/3}`);

if (meshData.vertices.length === 0) {
  console.error('ERROR: Marching cubes produced no geometry. Adjust t offset.');
  process.exit(1);
}

// ─── 2. Half-edge mesh ────────────────────────────────────────────────────────
console.log('[2/5] Building half-edge mesh...');
const hem = buildHalfEdgeMesh(meshData);
console.log(`      vertices=${hem.vertices.length}  faces=${hem.faces.length}`);

// ─── 3. Stripe fields (3 families at 60° intervals) ──────────────────────────
console.log('[3/5] Computing stripe fields...');
const t0 = Date.now();
const fields: [Float64Array, Float64Array, Float64Array] = [
  computeStripeField(hem, 0,                   STRIP_DENSITY),
  computeStripeField(hem, Math.PI / 3,         STRIP_DENSITY),
  computeStripeField(hem, (2 * Math.PI) / 3,   STRIP_DENSITY),
];
console.log(`      done in ${Date.now() - t0}ms`);

// ─── 4. Trace isolines ────────────────────────────────────────────────────────
console.log('[4/5] Tracing isolines...');
const isolinesByFamily: [any[], any[], any[]] = [
  traceIsolines(hem, fields[0], numStripes),
  traceIsolines(hem, fields[1], numStripes),
  traceIsolines(hem, fields[2], numStripes),
];
for (let k = 0; k < 3; k++) {
  const segs = isolinesByFamily[k].reduce((s: number, iso: any) => s + iso.points.length / 2, 0);
  console.log(`      family ${k}: ${isolinesByFamily[k].length} isolines  ${segs} raw segments`);
}

// ─── 5. Kagome pattern ────────────────────────────────────────────────────────
console.log('[5/5] Building Kagome pattern...');
const pattern = buildKagomePattern(
  hem,
  fields,
  isolinesByFamily as any,
  numStripes,
  holeRadiusWorld,  // 0.04 world units (= 2mm)
  'A',              // widthMethod A: auto from isoline spacing
  0.75,             // widthRatio
);

// ─── 6. Ribbon length diagnostics ────────────────────────────────────────────
function arcLength(pts: { distanceTo: (other: any) => number }[]): number {
  let len = 0;
  for (let i = 1; i < pts.length; i++) len += pts[i].distanceTo(pts[i-1]);
  return len;
}

console.log(`\n${'='.repeat(54)}`);
console.log('  RIBBON LENGTH DIAGNOSTICS');
console.log(`${'='.repeat(54)}`);

// 判定基準:
//   ✓ OK      : arc >= 1.0 (十分な長さ)
//   △ BRIDGE  : arc >= 0.3 かつ junctions > 0 (短いが織りに必要なブリッジ)
//   ✗ BAD     : arc < 0.3 または (arc < 0.5 かつ junctions == 0)
let totalOk = 0, totalBad = 0, totalBridge = 0;

for (let k = 0; k < 3; k++) {
  const fam = pattern.families[k];
  const phi = ['0°','60°','120°'][k];
  console.log(`\nFamily ${k} (φ=${phi}) — ${fam.length} strips`);
  if (fam.length === 0) {
    console.log('  ⚠ NO STRIPS generated');
    totalBad++;
    continue;
  }
  console.log('  id        pts  junc   width    arcLen');
  console.log('  ──────────────────────────────────────');
  for (const strip of fam) {
    const len = arcLength(strip.centerline);
    const junc = strip.junctions.length;
    let flag: string;
    if (len >= 1.0) {
      flag = '✓';
      totalOk++;
    } else if (len >= 0.3 && junc > 0) {
      flag = '△ bridge';   // short but junction-connected → valid
      totalBridge++;
    } else {
      flag = '✗ BAD';
      totalBad++;
    }
    console.log(
      `  ${strip.id.padEnd(8)}  ` +
      `${String(strip.centerline.length).padStart(3)}  ` +
      `${String(junc).padStart(4)}  ` +
      `${strip.width.toFixed(3)}  ` +
      `${len.toFixed(3).padStart(8)}  ${flag}`
    );
  }
  const total = fam.reduce((s, st) => s + arcLength(st.centerline), 0);
  console.log(`  → total arc: ${total.toFixed(3)}`);
}

const famCounts = [0, 1, 2].map(k => pattern.families[k].length);

console.log(`\n${'='.repeat(54)}`);
console.log(`  strips OK (arc≥1.0):         ${totalOk}`);
console.log(`  strips bridge (0.3-1.0,junc): ${totalBridge}`);
console.log(`  strips BAD:                  ${totalBad}`);
console.log(`  junctions:                   ${pattern.junctions.length}`);
console.log(`  family sizes: A=${famCounts[0]}  B=${famCounts[1]}  C=${famCounts[2]}`);
console.log(`${'='.repeat(54)}\n`);

if (totalBad > 0 || famCounts.some(c => c === 0)) {
  console.log(`⚠  Fix needed: ${totalBad} BAD strips, family sizes=${famCounts}`);
  process.exit(1);
} else {
  console.log(`✅ 3 sets of ribbons generated successfully!`);
  if (totalBridge > 0) {
    console.log(`   (${totalBridge} short bridge strips are normal — connect junctions across short gaps)`);
  }
  process.exit(0);
}

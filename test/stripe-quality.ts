/**
 * stripe-quality.ts – Local quality test harness for stripe generation
 *
 * Run: npx tsx test/stripe-quality.ts [options]
 *
 * Computes metrics for the current stripe generation algorithm:
 *   - Spacing uniformity (isoline inter-distance variance)
 *   - Per-strip smoothness (turning angle distribution)
 *   - Crossing angles at junctions (vs expected 60°)
 *   - Connectivity (strip length distribution, short strips, orphans)
 *
 * Outputs a quality report with a global score.
 */

import * as THREE from 'three';
import { createTPMSFunction } from '../src/core/tpms';
import { marchingCubes } from '../src/core/marchingCubes';
import { buildHalfEdgeMesh } from '../src/core/halfEdge';
import type { HalfEdgeMesh } from '../src/core/halfEdge';
import {
  computeGuidedStripeFields, computeStripeField, computeKnoppelStripeFields,
  traceIsolines, traceZeroCrossings,
} from '../src/core/connectionLaplacian';
import { buildKagomePattern } from '../src/core/kagome';
import type { KagomePattern, Strip } from '../src/core/kagome';

// ─────────────────────────────────────────────────────────────────────────────
// Parameters
// ─────────────────────────────────────────────────────────────────────────────

interface TestParams {
  surface:        'gyroid' | 'schwarzP' | 'schwarzD';
  period:         number;
  baseT:          number;
  bboxMin:        number;
  bboxMax:        number;
  gridRes:        number;
  numIsolines:    number;
  density:        number;
  holeRadius:     number;
  stripWidth:     number;
}

const DEFAULT_PARAMS: TestParams = {
  surface:     'gyroid',
  period:      2 * Math.PI,
  baseT:       0,
  bboxMin:     -Math.PI,
  bboxMax:     Math.PI,
  gridRes:     50,
  numIsolines: 8,
  density:     4.0,
  holeRadius:  0.04,
  stripWidth:  0.2,
};

// ─────────────────────────────────────────────────────────────────────────────
// Metrics
// ─────────────────────────────────────────────────────────────────────────────

interface Metrics {
  numStrips:        number;
  numJunctions:     number;
  // Smoothness (turning angle per polyline vertex)
  turningDeg:       { max: number; mean: number; p50: number; p90: number; p95: number };
  turningHistogram: { bucket: string; count: number }[];
  sharpCount:       number;  // vertices with turning > 15°
  sharpRatio:       number;
  worstStrips:      { id: string; maxDeg: number; where: 'end' | 'mid' }[];
  // Junction crossing angles (should cluster near 60° or 120°)
  crossingDeg:      { mean: number; std: number; count: number };
  crossingDevFrom60: number; // mean |angle - 60°|
  // Strip length distribution
  lengthStats:      { min: number; max: number; mean: number; std: number };
  shortStrips:      number;     // strips shorter than 30% of mean
  shortRatio:       number;
  // Spacing uniformity (distance between adjacent same-family strips)
  spacingStats:     { mean: number; std: number; cov: number };
  // Derived quality score (0-100, higher = better)
  qualityScore:     number;
}

function computeMetrics(pattern: KagomePattern): Metrics {
  const strips    = pattern.strips;
  const junctions = pattern.junctions;

  // ── Turning angles per strip vertex ──
  const turningAngles: number[] = [];
  const stripMaxAngles: { id: string; maxDeg: number; where: 'end' | 'mid' }[] = [];
  for (const strip of strips) {
    let stripMax = 0;
    let stripMaxIdx = 0;
    const nPts = strip.centerline.length;
    for (let i = 1; i < nPts - 1; i++) {
      const p0 = strip.centerline[i - 1];
      const p1 = strip.centerline[i];
      const p2 = strip.centerline[i + 1];
      const d1 = new THREE.Vector3().subVectors(p1, p0);
      const d2 = new THREE.Vector3().subVectors(p2, p1);
      if (d1.lengthSq() < 1e-10 || d2.lengthSq() < 1e-10) continue;
      d1.normalize(); d2.normalize();
      const cos = Math.max(-1, Math.min(1, d1.dot(d2)));
      const deg = Math.acos(cos) * 180 / Math.PI;
      turningAngles.push(deg);
      if (deg > stripMax) { stripMax = deg; stripMaxIdx = i; }
    }
    if (stripMax > 0) {
      const frac = stripMaxIdx / Math.max(1, nPts - 1);
      const where: 'end' | 'mid' = (frac < 0.15 || frac > 0.85) ? 'end' : 'mid';
      stripMaxAngles.push({ id: strip.id, maxDeg: stripMax, where });
    }
  }

  const sharpCount = turningAngles.filter(a => a > 15).length;

  // Turning angle histogram
  const buckets = [0, 5, 10, 15, 20, 30, 45, 60, 90, 180];
  const turningHistogram = buckets.slice(0, -1).map((lo, i) => {
    const hi = buckets[i + 1];
    const count = turningAngles.filter(a => a >= lo && a < hi).length;
    return { bucket: `[${lo},${hi})`, count };
  });

  // Worst 10 strips
  const worstStrips = stripMaxAngles
    .sort((a, b) => b.maxDeg - a.maxDeg)
    .slice(0, 10);

  // ── Junction crossing angles ──
  const crossingAngles: number[] = [];
  for (const j of junctions) {
    const [idA, idB] = j.stripIds;
    const sa = strips.find(s => s.id === idA);
    const sb = strips.find(s => s.id === idB);
    if (!sa || !sb) continue;

    const dirA = tangentAt(sa, j.position);
    const dirB = tangentAt(sb, j.position);
    if (!dirA || !dirB) continue;

    // Unsigned angle between lines (stripes are undirected)
    const cos = Math.abs(dirA.dot(dirB));
    const angle = Math.acos(Math.min(1, cos)) * 180 / Math.PI;
    crossingAngles.push(angle);
  }

  const crossingMean  = avg(crossingAngles);
  const crossingStd   = std(crossingAngles, crossingMean);
  const crossingDev60 = avg(crossingAngles.map(a => Math.abs(a - 60)));

  // ── Strip length distribution ──
  const lengths = strips.map(s => {
    let len = 0;
    for (let i = 1; i < s.centerline.length; i++) {
      len += s.centerline[i].distanceTo(s.centerline[i - 1]);
    }
    return len;
  });
  const lMean = avg(lengths);
  const lStd  = std(lengths, lMean);
  const shortStrips = lengths.filter(l => l < 0.3 * lMean).length;

  // ── Spacing (pairwise nearest-same-family average centerline distance) ──
  const spacings: number[] = [];
  for (let k = 0; k < 3; k++) {
    const fam = strips.filter(s => s.family === k);
    for (let i = 0; i < fam.length; i++) {
      let minD = Infinity;
      for (let j = 0; j < fam.length; j++) {
        if (i === j) continue;
        const d = avgCenterlineDistance(fam[i], fam[j]);
        if (d < minD) minD = d;
      }
      if (minD < Infinity) spacings.push(minD);
    }
  }
  const sMean = avg(spacings);
  const sStd  = std(spacings, sMean);
  const sCov  = sMean > 0 ? sStd / sMean : 0;

  // ── Quality score (0-100) ──
  // Weight different components
  const scoreSmoothness = 100 * Math.max(0, 1 - sharpCount / Math.max(1, turningAngles.length * 0.05));
  const scoreCrossing   = 100 * Math.max(0, 1 - crossingDev60 / 30); // 30° dev = 0 score
  const scoreConnectivity = 100 * Math.max(0, 1 - shortStrips / Math.max(1, strips.length * 0.2));
  const scoreSpacing    = 100 * Math.max(0, 1 - sCov);
  const qualityScore    = (scoreSmoothness + scoreCrossing + scoreConnectivity + scoreSpacing) / 4;

  return {
    numStrips:     strips.length,
    numJunctions:  junctions.length,
    turningDeg:    {
      max:  Math.max(...turningAngles, 0),
      mean: avg(turningAngles),
      p50:  percentile(turningAngles, 0.5),
      p90:  percentile(turningAngles, 0.9),
      p95:  percentile(turningAngles, 0.95),
    },
    turningHistogram,
    sharpCount,
    sharpRatio:   turningAngles.length > 0 ? sharpCount / turningAngles.length : 0,
    worstStrips,
    crossingDeg:  { mean: crossingMean, std: crossingStd, count: crossingAngles.length },
    crossingDevFrom60: crossingDev60,
    lengthStats:  { min: Math.min(...lengths, 0), max: Math.max(...lengths, 0), mean: lMean, std: lStd },
    shortStrips,
    shortRatio:   strips.length > 0 ? shortStrips / strips.length : 0,
    spacingStats: { mean: sMean, std: sStd, cov: sCov },
    qualityScore,
  };
}

function tangentAt(strip: Strip, pos: THREE.Vector3): THREE.Vector3 | null {
  // Find closest centerline index
  let minD2 = Infinity, idx = 0;
  for (let i = 0; i < strip.centerline.length; i++) {
    const d2 = strip.centerline[i].distanceToSquared(pos);
    if (d2 < minD2) { minD2 = d2; idx = i; }
  }
  const a = Math.max(0, idx - 1);
  const b = Math.min(strip.centerline.length - 1, idx + 1);
  if (a === b) return null;
  return new THREE.Vector3().subVectors(strip.centerline[b], strip.centerline[a]).normalize();
}

function avgCenterlineDistance(a: Strip, b: Strip): number {
  // Sample a few points on a, find nearest on b, return average
  const SAMPLES = 5;
  let sum = 0, count = 0;
  for (let i = 0; i < SAMPLES; i++) {
    const t = i / (SAMPLES - 1);
    const idx = Math.floor(t * (a.centerline.length - 1));
    const pa = a.centerline[idx];
    let minD = Infinity;
    for (const pb of b.centerline) {
      const d = pa.distanceTo(pb);
      if (d < minD) minD = d;
    }
    if (minD < Infinity) { sum += minD; count++; }
  }
  return count > 0 ? sum / count : Infinity;
}

function avg(arr: number[]): number {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

function std(arr: number[], mean?: number): number {
  if (arr.length < 2) return 0;
  const m = mean ?? avg(arr);
  const v = arr.reduce((s, x) => s + (x - m) ** 2, 0) / arr.length;
  return Math.sqrt(v);
}

function percentile(arr: number[], p: number): number {
  if (!arr.length) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline runner
// ─────────────────────────────────────────────────────────────────────────────

type StripeMode = 'guided' | 'poisson' | 'knoppel';

function runPipeline(params: TestParams, mode: StripeMode = 'guided'): {
  mesh: HalfEdgeMesh;
  pattern: KagomePattern;
  timings: { mesh: number; stripe: number; kagome: number };
} {
  const t0 = Date.now();
  const implicit = createTPMSFunction(params.surface, params.baseT, false, 0, 0, 42);
  const meshData = marchingCubes(
    implicit, params.bboxMin, params.bboxMax, params.gridRes, params.period,
  );
  const mesh = buildHalfEdgeMesh(meshData);
  const t1 = Date.now();

  let stripeFields: [Float64Array, Float64Array, Float64Array];
  let isolines: [any, any, any] = [[], [], []];

  if (mode === 'knoppel') {
    // Knöppel 2015: twisted eigensolve per family → Re(ψ) zero crossings
    const omega = 4.0;   // stripe frequency (tuneable)
    const psi = computeKnoppelStripeFields(mesh, omega);
    stripeFields = [psi[0].re, psi[1].re, psi[2].re];
    for (let k = 0; k < 3; k++) {
      isolines[k] = traceZeroCrossings(mesh, psi[k].re);  // no amplitude filter
    }
  } else if (mode === 'guided') {
    const guided = computeGuidedStripeFields(mesh, params.density);
    stripeFields = guided.fields;
    for (let k = 0; k < 3; k++) {
      isolines[k] = traceIsolines(mesh, stripeFields[k], params.numIsolines);
    }
  } else {
    // Old comb-frame Poisson
    stripeFields = [
      computeStripeField(mesh, 0,                   params.density),
      computeStripeField(mesh, Math.PI / 3,         params.density),
      computeStripeField(mesh, (2 * Math.PI) / 3,   params.density),
    ];
    for (let k = 0; k < 3; k++) {
      isolines[k] = traceIsolines(mesh, stripeFields[k], params.numIsolines);
    }
  }
  const t2 = Date.now();

  const pattern = buildKagomePattern(
    mesh, stripeFields, isolines,
    params.numIsolines, params.holeRadius, 'B', 0.75, params.stripWidth,
  );
  const t3 = Date.now();

  return {
    mesh, pattern,
    timings: { mesh: t1 - t0, stripe: t2 - t1, kagome: t3 - t2 },
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Report
// ─────────────────────────────────────────────────────────────────────────────

function printReport(label: string, m: Metrics, timings: any): void {
  console.log(`\n══ ${label} ══`);
  console.log(`  Strips: ${m.numStrips}  Junctions: ${m.numJunctions}`);
  console.log(`  Timings (ms): mesh=${timings.mesh}  stripe=${timings.stripe}  kagome=${timings.kagome}`);
  console.log(`\n  Turning angles (deg):`);
  console.log(`    max=${m.turningDeg.max.toFixed(1)}  mean=${m.turningDeg.mean.toFixed(2)}  p90=${m.turningDeg.p90.toFixed(1)}  p95=${m.turningDeg.p95.toFixed(1)}`);
  console.log(`    sharp (>15°) = ${m.sharpCount} (${(100 * m.sharpRatio).toFixed(1)}%)`);
  console.log(`    histogram: ${m.turningHistogram.map(h => `${h.bucket}=${h.count}`).join(' ')}`);
  console.log(`    worst 5 strips:`);
  for (const w of m.worstStrips.slice(0, 5)) {
    console.log(`      ${w.id.padEnd(8)} max=${w.maxDeg.toFixed(1)}° (${w.where})`);
  }
  console.log(`\n  Crossing angles (deg) [target: 60°]:`);
  console.log(`    count=${m.crossingDeg.count}  mean=${m.crossingDeg.mean.toFixed(1)}  std=${m.crossingDeg.std.toFixed(1)}`);
  console.log(`    mean |dev from 60°| = ${m.crossingDevFrom60.toFixed(1)}°`);
  console.log(`\n  Strip length:`);
  console.log(`    min=${m.lengthStats.min.toFixed(2)}  max=${m.lengthStats.max.toFixed(2)}  mean=${m.lengthStats.mean.toFixed(2)}  std=${m.lengthStats.std.toFixed(2)}`);
  console.log(`    short (< 30% mean) = ${m.shortStrips} (${(100 * m.shortRatio).toFixed(1)}%)`);
  console.log(`\n  Spacing:`);
  console.log(`    mean=${m.spacingStats.mean.toFixed(3)}  std=${m.spacingStats.std.toFixed(3)}  cov=${m.spacingStats.cov.toFixed(2)}`);
  console.log(`\n  ▶ QUALITY SCORE: ${m.qualityScore.toFixed(1)} / 100\n`);
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────

function main() {
  console.log('TPMS Kagome Designer — Stripe Quality Benchmark');
  console.log('='.repeat(60));

  const modeArg = process.argv[2] as StripeMode | undefined;
  const modes: StripeMode[] = modeArg ? [modeArg] : ['guided', 'poisson', 'knoppel'];
  const surfaces: TestParams['surface'][] = ['gyroid', 'schwarzP', 'schwarzD'];
  const byMode: Record<StripeMode, number[]> = { guided: [], poisson: [], knoppel: [] };

  for (const mode of modes) {
    const modeLabels: Record<StripeMode, string> = {
      guided:  '3-RoSy guided Poisson',
      poisson: 'Old comb-frame Poisson',
      knoppel: 'Knöppel 2015 Stripe Patterns',
    };
    console.log('\n' + '█'.repeat(60));
    console.log(`█ MODE: ${modeLabels[mode]}`);
    console.log('█'.repeat(60));

    const results: Array<{ label: string; m: Metrics; timings: any }> = [];
    for (const surface of surfaces) {
      console.log(`\n▶ ${surface}...`);
      const params = { ...DEFAULT_PARAMS, surface };
      try {
        const { pattern, timings } = runPipeline(params, mode);
        const m = computeMetrics(pattern);
        results.push({ label: surface, m, timings });
      } catch (e) {
        console.error(`  FAILED: ${e}`);
      }
    }

    console.log('\n' + '='.repeat(60));
    console.log(`[${mode}] RESULTS`);
    console.log('='.repeat(60));
    for (const r of results) printReport(r.label, r.m, r.timings);

    const overall = avg(results.map(r => r.m.qualityScore));
    byMode[mode].push(overall);
    console.log('═'.repeat(60));
    console.log(`  [${mode}] OVERALL QUALITY: ${overall.toFixed(1)} / 100`);
    console.log('═'.repeat(60));
  }

  // Comparison
  if (modes.length > 1) {
    console.log('\n\n' + '▓'.repeat(60));
    console.log('▓ COMPARISON');
    console.log('▓'.repeat(60));
    for (const mode of modes) {
      console.log(`  ${mode.padEnd(10)} = ${byMode[mode][0].toFixed(1)}`);
    }
  }
}

main();

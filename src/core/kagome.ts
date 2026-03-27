/**
 * kagome.ts – Phase 3: Kagome strip extraction
 *
 * Takes 3 stripe scalar fields + their isolines (from Phase 2) and:
 *   1. Stitches marching-triangle segments into ordered polylines (strip centerlines)
 *   2. Detects junctions by finding faces where 2 different families cross simultaneously
 *   3. Computes the exact 3D intersection point of the two crossing segments in each junction face
 *   4. Applies the kagome over/under rule: family k goes OVER family (k+1)%3
 *   5. Segments each strip centerline at its junction positions
 *   6. Assigns world-space width from adjacent isoline spacing
 */

import * as THREE from 'three';
import type { HalfEdgeMesh } from './halfEdge';
import type { Isoline } from './connectionLaplacian';

// ─────────────────────────────────────────────────────────────────────────────
// Public interfaces  (kept backward-compatible with unfold.ts / dxf.ts)
// ─────────────────────────────────────────────────────────────────────────────

export interface Strip {
  id: string;
  family: number;            // 0 | 1 | 2
  layer: number;             // 0=under, 1=neutral, 2=over  (per-junction)
  isolines: [Isoline, Isoline];  // [left-boundary isoline, right-boundary isoline] (kept for compat)
  centerline: THREE.Vector3[];   // stitched polyline
  width: number;             // world-space half-width × 2
  junctions: Junction[];
  segments: StripSegment[];
}

export interface Junction {
  id: number;
  position: THREE.Vector3;
  normal: THREE.Vector3;     // surface normal at junction
  stripIds: string[];        // exactly 2 strip IDs
  familyPair: [number, number];
  overFamily: number;        // kagome rule: family k over (k+1)%3
  underFamily: number;
  holeRadius: number;
  faceIndex: number;
}

export interface StripSegment {
  startJunctionId: number | null;
  endJunctionId: number | null;
  points: THREE.Vector3[];   // ordered polyline from startJunction to endJunction
  width: number;
  faceIndices: number[];
}

export interface KagomePattern {
  strips: Strip[];
  junctions: Junction[];
  families: [Strip[], Strip[], Strip[]];
}

// ─────────────────────────────────────────────────────────────────────────────
// Main entry point
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build a complete Kagome pattern from the 3 stripe fields.
 *
 * @param mesh              Half-edge mesh (TPMS surface)
 * @param stripeFields      3 scalar fields from the Poisson solve
 * @param isolinesByFamily  Marching-triangle isolines for each family
 * @param numStripes        Number of stripes per family
 * @param holeRadius        Radius of junction holes
 */
export function buildKagomePattern(
  mesh: HalfEdgeMesh,
  _stripeFields: [Float64Array, Float64Array, Float64Array],
  isolinesByFamily: [Isoline[], Isoline[], Isoline[]],
  _numStripes: number,
  holeRadius: number,
): KagomePattern {

  // Build surface projector once (shared across all strips)
  const projectToSurface = buildSurfaceProjector(mesh);

  // ── 1. Stitch segment pairs → ordered polylines per strip ─────────────────
  const families: [Strip[], Strip[], Strip[]] = [[], [], []];
  const allStrips: Strip[] = [];

  // Diagnostic counters
  const diagRows: {
    id: string; segments: number; chains: number;
    longestChain: number; accepted: boolean; reason: string;
  }[] = [];

  for (let k = 0; k < 3; k++) {
    const isos = isolinesByFamily[k];
    for (let li = 0; li < isos.length; li++) {
      const iso = isos[li];
      const id = `${String.fromCharCode(65 + k)}${li + 1}`;
      const rawSegs = iso.points.length / 2;

      if (iso.points.length < 2) {
        diagRows.push({ id, segments: rawSegs, chains: 0, longestChain: 0, accepted: false, reason: 'no segments' });
        continue;
      }

      // Stitch marching-triangle segment pairs into connected polyline chains
      const chains = stitchSegments(iso.points);
      const longestChain = chains.length > 0
        ? chains.reduce((m, c) => Math.max(m, c.length), 0) : 0;

      // Pick the longest chain, smooth, then re-project onto the surface
      const raw      = chains.sort((a, b) => b.length - a.length)[0] ?? [];
      const smoothed = smoothPolyline(raw, 4);

      // Measure max drift before projection (diagnostic)
      let maxDrift = 0;
      const centerline = smoothed.map(p => {
        const projected = projectToSurface(p);
        maxDrift = Math.max(maxDrift, p.distanceTo(projected));
        return projected;
      });

      if (centerline.length < 2) {
        diagRows.push({ id, segments: rawSegs, chains: chains.length, longestChain, accepted: false, reason: 'centerline < 2 pts after smooth' });
        continue;
      }

      diagRows.push({
        id, segments: rawSegs, chains: chains.length, longestChain,
        accepted: true,
        reason: `cl=${centerline.length}pts drift=${maxDrift.toFixed(4)}`,
      });

      const strip: Strip = {
        id,
        family: k,
        layer: 1,
        isolines: [iso, iso],    // placeholder; boundaries added later
        centerline,
        width: 0,                // computed below
        junctions: [],
        segments: [],
      };

      families[k].push(strip);
      allStrips.push(strip);
    }
  }

  // ── Diagnostic summary ────────────────────────────────────────────────────
  const accepted = diagRows.filter(r => r.accepted);
  const rejected = diagRows.filter(r => !r.accepted);
  console.group(`[Kagome] Strip extraction — ${accepted.length}/${diagRows.length} accepted`);
  console.log(`Accepted ${accepted.length}, Rejected ${rejected.length}`);

  // Tabular view: one row per strip
  if (typeof console.table === 'function') {
    console.table(
      diagRows.map(r => ({
        id: r.id,
        segs: r.segments,
        chains: r.chains,
        longestChain: r.longestChain,
        accepted: r.accepted ? '✓' : '✗',
        reason: r.reason,
      })),
    );
  }

  // Group rejected by reason
  if (rejected.length > 0) {
    const byReason: Record<string, string[]> = {};
    for (const r of rejected) {
      byReason[r.reason] = [...(byReason[r.reason] ?? []), r.id];
    }
    console.warn('[Kagome] Rejected strips by reason:');
    for (const [reason, ids] of Object.entries(byReason)) {
      console.warn(`  ${reason}: ${ids.join(', ')}`);
    }

    // Extra: for each rejected strip, log segment/chain distribution
    for (const r of rejected) {
      if (r.chains > 1) {
        console.warn(`  ${r.id}: ${r.chains} chains (segs=${r.segments}, longestChain=${r.longestChain}) — chain fragmentation?`);
      }
    }
  }
  console.groupEnd();

  // ── 2. Estimate world-space strip widths from adjacent centerline spacing ──
  for (let k = 0; k < 3; k++) {
    const fam = families[k];
    // Collect all pairwise spacings then assign average so isolated strips still get a width
    const spacings: number[] = [];
    for (let i = 0; i < fam.length - 1; i++) {
      const d = averageCenterlineDistance(fam[i].centerline, fam[i + 1].centerline);
      if (d > 1e-6) spacings.push(d);
    }
    const avgSpacing = spacings.length > 0
      ? spacings.reduce((a, b) => a + b, 0) / spacings.length
      : 0.3; // fallback: 0.3 world units when only 1 strip or no valid spacing

    for (let i = 0; i < fam.length; i++) {
      let d = avgSpacing;
      if (i < fam.length - 1) {
        const dd = averageCenterlineDistance(fam[i].centerline, fam[i + 1].centerline);
        if (dd > 1e-6) d = dd;
      }
      fam[i].width = d * 0.75; // 75% of spacing → narrow gap between strips
    }
  }

  // ── 3. Build face → strip-segment map from isoline faceIndices ─────────────
  // faceMap[faceIdx] = list of {stripId, family, segA, segB}
  const faceMap = new Map<number, { stripId: string; family: number; segA: THREE.Vector3; segB: THREE.Vector3 }[]>();

  for (let k = 0; k < 3; k++) {
    const isos = isolinesByFamily[k];
    for (let li = 0; li < isos.length; li++) {
      const iso = isos[li];
      const stripId = `${String.fromCharCode(65 + k)}${li + 1}`;
      // Verify this strip was actually built
      if (!allStrips.find(s => s.id === stripId)) continue;

      for (let si = 0; si < iso.faceIndices.length; si++) {
        const fi = iso.faceIndices[si];
        const segA = iso.points[si * 2];
        const segB = iso.points[si * 2 + 1];
        if (!segA || !segB) continue;

        if (!faceMap.has(fi)) faceMap.set(fi, []);
        faceMap.get(fi)!.push({ stripId, family: k, segA, segB });
      }
    }
  }

  // ── 4. Detect junctions ────────────────────────────────────────────────────
  const junctions: Junction[] = [];
  let jId = 1;
  for (const [fi, segs] of faceMap) {
    // Group segments by family
    const byFamily = new Map<number, typeof segs>();
    for (const s of segs) {
      if (!byFamily.has(s.family)) byFamily.set(s.family, []);
      byFamily.get(s.family)!.push(s);
    }
    if (byFamily.size < 2) continue;

    const faceVerts = mesh.faces[fi];
    const n = computeFaceNormal(
      mesh.vertices[faceVerts[0]],
      mesh.vertices[faceVerts[1]],
      mesh.vertices[faceVerts[2]],
    );

    const famKeys = Array.from(byFamily.keys());
    for (let a = 0; a < famKeys.length; a++) {
      for (let b = a + 1; b < famKeys.length; b++) {
        const ka = famKeys[a], kb = famKeys[b];
        const segsA = byFamily.get(ka)!;
        const segsB = byFamily.get(kb)!;

        for (const sa of segsA) {
          for (const sb of segsB) {
            const pt = intersectSegmentsInFace(sa.segA, sa.segB, sb.segA, sb.segB, n);
            if (!pt) continue;

            // Deduplicate: skip if a junction from the SAME two strips is already nearby
            const dedupKey = [sa.stripId, sb.stripId].sort().join('|');
            let tooClose = false;
            for (const junc of junctions) {
              if (
                junc.stripIds.slice().sort().join('|') === dedupKey &&
                junc.position.distanceTo(pt) < holeRadius * 3
              ) {
                tooClose = true;
                break;
              }
            }
            if (tooClose) continue;

            // Kagome over/under rule: family k goes OVER family (k+1)%3
            const overFamily = (ka + 1) % 3 === kb ? ka : kb;
            const underFamily = overFamily === ka ? kb : ka;
            const overStripId = overFamily === ka ? sa.stripId : sb.stripId;
            const underStripId = overFamily === ka ? sb.stripId : sa.stripId;

            junctions.push({
              id: jId++,
              position: pt.clone(),
              normal: n.clone(),
              stripIds: [overStripId, underStripId],
              familyPair: [ka, kb],
              overFamily,
              underFamily,
              holeRadius,
              faceIndex: fi,
            });
          }
        }
      }
    }
  }

  // ── 5. Assign junctions to strips ─────────────────────────────────────────
  const stripById = new Map(allStrips.map(s => [s.id, s]));
  for (const junc of junctions) {
    for (const sid of junc.stripIds) {
      const strip = stripById.get(sid);
      if (strip && !strip.junctions.includes(junc)) strip.junctions.push(junc);
    }
  }

  // ── 6. Over/under layer assignment ────────────────────────────────────────
  for (const junc of junctions) {
    const overStrip = stripById.get(junc.stripIds[0]);  // over strip
    const underStrip = stripById.get(junc.stripIds[1]); // under strip
    // Promote "over" strip if it isn't already
    if (overStrip && overStrip.layer < 2) overStrip.layer = 2;
    if (underStrip && underStrip.layer > 0) underStrip.layer = 0;
  }

  // ── 7. Segment each strip at its junction points ───────────────────────────
  for (const strip of allStrips) {
    strip.segments = segmentAtJunctions(strip);
  }

  return { strips: allStrips, junctions, families };
}

// ─────────────────────────────────────────────────────────────────────────────
// Legacy entry points kept for backward-compatibility
// ─────────────────────────────────────────────────────────────────────────────

/** @deprecated Use buildKagomePattern instead */
export function extractKagomeStrips(
  mesh: HalfEdgeMesh,
  isolinesByFamily: [Isoline[], Isoline[], Isoline[]],
  _widthRatio: number,
  holeRadius: number,
): KagomePattern {
  // No stripe fields available in legacy path → build empty fields
  const empty = new Float64Array(mesh.vertices.length);
  return buildKagomePattern(
    mesh,
    [empty, empty, empty],
    isolinesByFamily,
    isolinesByFamily[0].length,
    holeRadius,
  );
}

/** @deprecated Layer assignment is now done inside buildKagomePattern */
export function assignLayers(_pattern: KagomePattern): void { /* no-op */ }

// ─────────────────────────────────────────────────────────────────────────────
// Laplacian smoothing – keeps endpoints fixed
// ─────────────────────────────────────────────────────────────────────────────

function smoothPolyline(pts: THREE.Vector3[], iterations = 4): THREE.Vector3[] {
  if (pts.length < 3) return pts.map(p => p.clone());
  let cur = pts.map(p => p.clone());
  for (let iter = 0; iter < iterations; iter++) {
    const next = cur.map(p => p.clone());
    for (let i = 1; i < cur.length - 1; i++) {
      next[i].set(
        0.25 * cur[i - 1].x + 0.5 * cur[i].x + 0.25 * cur[i + 1].x,
        0.25 * cur[i - 1].y + 0.5 * cur[i].y + 0.25 * cur[i + 1].y,
        0.25 * cur[i - 1].z + 0.5 * cur[i].z + 0.25 * cur[i + 1].z,
      );
    }
    cur = next;
  }
  return cur;
}

// ─────────────────────────────────────────────────────────────────────────────
// Surface projection – re-project smoothed points back onto the mesh surface.
//
// Laplacian smoothing pulls points toward the chord of the curve.  On a
// saddle-shaped TPMS (negative Gaussian curvature), this causes some strips
// to drift BELOW the surface and become hidden by the mesh.  Projecting each
// point onto the tangent plane at the nearest mesh vertex corrects this.
// ─────────────────────────────────────────────────────────────────────────────

function buildSurfaceProjector(
  mesh: HalfEdgeMesh,
): (pt: THREE.Vector3) => THREE.Vector3 {
  const CELL = 0.35;
  const grid = new Map<string, number[]>();

  for (let i = 0; i < mesh.vertices.length; i++) {
    const v = mesh.vertices[i];
    const k = `${Math.floor(v.x / CELL)},${Math.floor(v.y / CELL)},${Math.floor(v.z / CELL)}`;
    if (!grid.has(k)) grid.set(k, []);
    grid.get(k)!.push(i);
  }

  return (pt: THREE.Vector3): THREE.Vector3 => {
    const cx = Math.floor(pt.x / CELL);
    const cy = Math.floor(pt.y / CELL);
    const cz = Math.floor(pt.z / CELL);

    let minD2 = Infinity;
    let best  = 0;
    for (let dx = -1; dx <= 1; dx++)
    for (let dy = -1; dy <= 1; dy++)
    for (let dz = -1; dz <= 1; dz++) {
      for (const idx of grid.get(`${cx + dx},${cy + dy},${cz + dz}`) ?? []) {
        const d2 = mesh.vertices[idx].distanceToSquared(pt);
        if (d2 < minD2) { minD2 = d2; best = idx; }
      }
    }

    // Signed distance from pt to the tangent plane at the nearest vertex.
    // Subtracting it projects pt back onto (an approximation of) the surface.
    const vPos = mesh.vertices[best];
    const vNrm = mesh.normals[best];
    const d    = vNrm.dot(new THREE.Vector3().subVectors(pt, vPos));
    return pt.clone().sub(new THREE.Vector3().copy(vNrm).multiplyScalar(d));
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Stitch marching-triangle segment pairs into connected polyline chains
// ─────────────────────────────────────────────────────────────────────────────

function stitchSegments(points: THREE.Vector3[]): THREE.Vector3[][] {
  const n = points.length / 2;
  if (n === 0) return [];

  // Hash endpoint coordinates (rounded to 4 decimal places for stability)
  // 0.002 unit grid – tolerates floating-point jitter on shared mesh edges
  // while keeping distinct endpoints separate
  const PREC = 500;
  const hash = (p: THREE.Vector3) =>
    `${Math.round(p.x * PREC)},${Math.round(p.y * PREC)},${Math.round(p.z * PREC)}`;

  // endpointMap: hash → list of (segIdx, end)
  const endpointMap = new Map<string, { segIdx: number; end: 0 | 1 }[]>();
  for (let i = 0; i < n; i++) {
    for (const end of [0, 1] as const) {
      const h = hash(points[2 * i + end]);
      if (!endpointMap.has(h)) endpointMap.set(h, []);
      endpointMap.get(h)!.push({ segIdx: i, end });
    }
  }

  const used = new Uint8Array(n);
  const chains: THREE.Vector3[][] = [];

  // At a T-junction (3+ segments sharing one endpoint), we pick the segment
  // that makes the smoothest angle (closest to 180°) with the current direction.
  function pickNext(
    curPt: THREE.Vector3,
    inDir: THREE.Vector3 | null,
  ): { segIdx: number; end: 0 | 1 } | null {
    const h = hash(curPt);
    const candidates = (endpointMap.get(h) ?? []).filter(c => !used[c.segIdx]);
    if (candidates.length === 0) return null;
    if (candidates.length === 1 || inDir === null) return candidates[0];

    // Pick the segment whose direction is most aligned with inDir
    // (i.e. the one that continues the polyline most smoothly)
    let best = candidates[0];
    let bestDot = -Infinity;
    for (const c of candidates) {
      const otherEnd = points[2 * c.segIdx + (1 - c.end)];
      const segDir = new THREE.Vector3().subVectors(otherEnd, curPt).normalize();
      const d = inDir.dot(segDir);
      if (d > bestDot) { bestDot = d; best = c; }
    }
    return best;
  }

  for (let start = 0; start < n; start++) {
    if (used[start]) continue;

    const chain: THREE.Vector3[] = [points[2 * start].clone(), points[2 * start + 1].clone()];
    used[start] = 1;

    // Extend forward (with direction continuity)
    let extending = true;
    while (extending) {
      extending = false;
      const tail = chain[chain.length - 1];
      const inDir = chain.length >= 2
        ? new THREE.Vector3().subVectors(tail, chain[chain.length - 2]).normalize()
        : null;
      const c = pickNext(tail, inDir);
      if (c) {
        chain.push(points[2 * c.segIdx + (1 - c.end)].clone());
        used[c.segIdx] = 1;
        extending = true;
      }
    }

    // Extend backward
    extending = true;
    while (extending) {
      extending = false;
      const head = chain[0];
      const inDir = chain.length >= 2
        ? new THREE.Vector3().subVectors(head, chain[1]).normalize()
        : null;
      const c = pickNext(head, inDir);
      if (c) {
        chain.unshift(points[2 * c.segIdx + (1 - c.end)].clone());
        used[c.segIdx] = 1;
        extending = true;
      }
    }

    if (chain.length >= 2) chains.push(chain);
  }

  return chains;
}

// ─────────────────────────────────────────────────────────────────────────────
// Segment a strip's centerline at its junction positions
// ─────────────────────────────────────────────────────────────────────────────

function segmentAtJunctions(strip: Strip): StripSegment[] {
  const cl = strip.centerline;
  if (cl.length < 2) return [];

  if (strip.junctions.length === 0) {
    return [{
      startJunctionId: null,
      endJunctionId: null,
      points: [...cl],
      width: strip.width,
      faceIndices: [],
    }];
  }

  // Map each junction to the closest index on the centerline
  const juncPoints: { jId: number; clIdx: number }[] =
    strip.junctions
      .map(j => ({ jId: j.id, clIdx: closestPointIndex(cl, j.position) }))
      .sort((a, b) => a.clIdx - b.clIdx);

  const segments: StripSegment[] = [];
  let prevIdx = 0;
  let prevJId: number | null = null;

  for (const { jId, clIdx } of juncPoints) {
    if (clIdx > prevIdx) {
      segments.push({
        startJunctionId: prevJId,
        endJunctionId: jId,
        points: cl.slice(prevIdx, clIdx + 1),
        width: strip.width,
        faceIndices: [],
      });
    }
    prevIdx = clIdx;
    prevJId = jId;
  }

  // Final segment after the last junction
  if (prevIdx < cl.length - 1) {
    segments.push({
      startJunctionId: prevJId,
      endJunctionId: null,
      points: cl.slice(prevIdx),
      width: strip.width,
      faceIndices: [],
    });
  }

  return segments.filter(s => s.points.length >= 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// Intersection of two coplanar segments inside a triangular face
//
// Uses 2D coordinates in the face plane (first segment AB defines the x-axis).
// Returns a 3D point on segment AB (or null if they don't intersect in [0,1]×[0,1]).
// ─────────────────────────────────────────────────────────────────────────────

function intersectSegmentsInFace(
  A: THREE.Vector3, B: THREE.Vector3,
  C: THREE.Vector3, D: THREE.Vector3,
  faceNormal: THREE.Vector3,
): THREE.Vector3 | null {
  const AB = new THREE.Vector3().subVectors(B, A);
  const lenAB = AB.length();
  if (lenAB < 1e-12) return null;

  // Local 2D frame: u along AB, w perpendicular in face plane
  const u = AB.clone().divideScalar(lenAB);
  const w = new THREE.Vector3().crossVectors(faceNormal, u).normalize();

  const dot3 = (v: THREE.Vector3, d: THREE.Vector3) =>
    (v.x - A.x) * d.x + (v.y - A.y) * d.y + (v.z - A.z) * d.z;

  // 2D coords (A is origin)
  // B2 = (lenAB, 0)  [by construction]
  const C2x = dot3(C, u), C2y = dot3(C, w);
  const D2x = dot3(D, u), D2y = dot3(D, w);

  // Line 1: (0,0)→(lenAB,0), parametric s·(lenAB,0), s∈[0,1]
  // Line 2: (C2x,C2y)→(D2x,D2y), parametric C + t·(D-C), t∈[0,1]
  //
  // At intersection: y=0  →  C2y + t·(D2y-C2y)=0  →  t = -C2y/(D2y-C2y)
  // Then x = C2x + t·(D2x-C2x)  →  s = x/lenAB

  const ddy = D2y - C2y;

  if (Math.abs(ddy) < 1e-10) {
    // Segments parallel within face → use centroid of the 4 endpoints
    return A.clone().add(B).add(C).add(D).multiplyScalar(0.25);
  }

  const t = -C2y / ddy;
  if (t < -0.02 || t > 1.02) return null;

  const px = C2x + t * (D2x - C2x);
  const s = px / lenAB;
  if (s < -0.02 || s > 1.02) return null;

  const sClamped = Math.max(0, Math.min(1, s));
  return new THREE.Vector3().lerpVectors(A, B, sClamped);
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function computeFaceNormal(
  v0: THREE.Vector3,
  v1: THREE.Vector3,
  v2: THREE.Vector3,
): THREE.Vector3 {
  const e1 = new THREE.Vector3().subVectors(v1, v0);
  const e2 = new THREE.Vector3().subVectors(v2, v0);
  return new THREE.Vector3().crossVectors(e1, e2).normalize();
}

function closestPointIndex(points: THREE.Vector3[], target: THREE.Vector3): number {
  let minD = Infinity, idx = 0;
  for (let i = 0; i < points.length; i++) {
    const d = points[i].distanceToSquared(target);
    if (d < minD) { minD = d; idx = i; }
  }
  return idx;
}

function averageCenterlineDistance(cl1: THREE.Vector3[], cl2: THREE.Vector3[]): number {
  const len = Math.min(cl1.length, cl2.length, 10);
  if (len === 0) return 0.1;
  let sum = 0;
  for (let i = 0; i < len; i++) {
    const idx1 = Math.floor(i * cl1.length / len);
    const idx2 = Math.floor(i * cl2.length / len);
    sum += cl1[idx1].distanceTo(cl2[idx2]);
  }
  return sum / len;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kept for downstream compatibility (generateJunctionHoles in old code)
// ─────────────────────────────────────────────────────────────────────────────

export function generateJunctionHoles(
  junctions: Junction[],
  radius: number,
): { centers: THREE.Vector3[]; radii: number[] } {
  return {
    centers: junctions.map(j => j.position.clone()),
    radii: junctions.map(() => radius),
  };
}

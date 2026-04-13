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
  width: number;             // average world-space half-width × 2 (backward compat)
  widths: number[];          // per-point width (same length as centerline) for developable unfolding
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
  _stripeFields:    [Float64Array, Float64Array, Float64Array],
  isolinesByFamily: [Isoline[], Isoline[], Isoline[]],
  _numStripes:      number,
  holeRadius:       number,
  widthMethod:      'A' | 'B' = 'B',
  widthRatio:       number    = 0.75,
  stripWidthWorld:  number    = 0,    // world units; used when widthMethod='B'
): KagomePattern {

  // Build surface projector once (shared across all strips)
  const projectToSurface = buildSurfaceProjector(mesh);

  // ── 1. Stitch segment pairs → ordered polylines per strip ─────────────────
  const families: [Strip[], Strip[], Strip[]] = [[], [], []];
  const allStrips: Strip[] = [];

  // faceMap built inline while processing each chain (step 1)
  const faceMap = new Map<number, { stripId: string; family: number; segA: THREE.Vector3; segB: THREE.Vector3 }[]>();

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

      // Stitch into chains (each chain also carries segIndices for faceMap)
      const chains = stitchSegments(iso.points);
      chains.sort((a, b) => b.points.length - a.points.length); // longest first

      const longestChain = chains.length > 0 ? chains[0].points.length : 0;
      let acceptedChains = 0;

      // Minimum arc length threshold: ~10% of the longest chain's arc length,
      // but at least 0.1 world units.  Removes:
      //   (a) 2-point isolated segments that failed to stitch (zero arc)
      //   (b) tiny closed loops smaller than one junction hole
      //   (c) boundary fragments too short to weave physically
      let longestArc = 0;
      for (const ch of chains) {
        let a = 0;
        for (let i = 1; i < ch.points.length; i++) a += ch.points[i].distanceTo(ch.points[i - 1]);
        if (a > longestArc) longestArc = a;
      }
      const minArc = Math.max(0.1, longestArc * 0.10);

      for (let ci = 0; ci < chains.length; ci++) {
        const chain = chains[ci];
        // Always skip single-point or no-point chains
        if (chain.points.length < 4) continue;
        // Skip chains shorter than minimum arc threshold
        let chainArc = 0;
        for (let i = 1; i < chain.points.length; i++) {
          chainArc += chain.points[i].distanceTo(chain.points[i - 1]);
        }
        if (chainArc < minArc) continue;

        const smoothed   = smoothPolyline(chain.points, 4);
        const projected  = smoothed.map(p => projectToSurface(p));
        const centerline = adaptiveSubdivide(projected, projectToSurface);
        if (centerline.length < 2) continue;

        const chainSuffix = chains.length > 1 ? `_${ci + 1}` : '';
        const stripId     = `${id}${chainSuffix}`;

        const strip: Strip = {
          id: stripId,
          family: k,
          layer: 1,
          isolines: [iso, iso],
          centerline,
          width: 0,
          widths: new Array(centerline.length).fill(0),  // filled in step 2
          junctions: [],
          segments: [],
        };
        families[k].push(strip);
        allStrips.push(strip);
        acceptedChains++;

        // ── Build faceMap for this chain's segments ──────────────────────────
        // chain.segIndices[j] = original segment index for link j
        // iso.faceIndices[segIdx] = which mesh face the segment lives in
        for (const segIdx of chain.segIndices) {
          if (segIdx >= iso.faceIndices.length) continue;
          const fi   = iso.faceIndices[segIdx];
          const segA = iso.points[segIdx * 2];
          const segB = iso.points[segIdx * 2 + 1];
          if (!segA || !segB) continue;
          if (!faceMap.has(fi)) faceMap.set(fi, []);
          faceMap.get(fi)!.push({ stripId, family: k, segA, segB });
        }
      }

      diagRows.push({
        id, segments: rawSegs, chains: chains.length, longestChain,
        accepted: acceptedChains > 0,
        reason: acceptedChains > 0
          ? `${acceptedChains} chain(s) → ${acceptedChains} strips`
          : 'no valid chains',
      });
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

  // ── 2. Estimate world-space strip widths: per-isoline, inter-isoline spacing only ──
  //
  // Strip IDs: single-chain = "A3"; multi-chain = "A3_1", "A3_2", ...
  // Adjacent strips in families[] may be chains of the SAME isoline (distance ≈ 0),
  // so we group by base isoline ID and measure ONLY between different isolines.
  // Each isoline gets its OWN local width (= avg of spacings to neighbors).
  // → Non-uniform widths across isolines are preserved (Plan A behaviour).
  const isoKey = (id: string) => id.replace(/_\d+$/, ''); // "A3_2" → "A3"

  for (let k = 0; k < 3; k++) {
    const fam = families[k];

    // One representative per isoline (first chain inserted)
    const repMap = new Map<string, Strip>();
    for (const s of fam) {
      const key = isoKey(s.id);
      if (!repMap.has(key)) repMap.set(key, s);
    }
    const repList = [...repMap.values()];

    if (widthMethod === 'B') {
      // ── Method B: uniform mm-specified width ──────────────────────────────
      for (const s of fam) {
        s.width = stripWidthWorld;
        s.widths = new Array(s.centerline.length).fill(stripWidthWorld);
      }
    } else {
      // ── Method A: per-isoline local spacing × widthRatio (non-uniform) ───

      // Global average inter-isoline spacing (fallback)
      const allSpacings: number[] = [];
      for (let i = 0; i < repList.length - 1; i++) {
        const d = averageCenterlineDistance(repList[i].centerline, repList[i + 1].centerline);
        if (d > 1e-6) allSpacings.push(d);
      }
      const avgSpacing = allSpacings.length > 0
        ? allSpacings.reduce((a, b) => a + b, 0) / allSpacings.length
        : 0.3;

      // Assign LOCAL width to each isoline (avg of spacings to prev + next)
      for (let i = 0; i < repList.length; i++) {
        const localS: number[] = [];
        if (i > 0) {
          const d = averageCenterlineDistance(repList[i - 1].centerline, repList[i].centerline);
          if (d > 1e-6) localS.push(d);
        }
        if (i < repList.length - 1) {
          const d = averageCenterlineDistance(repList[i].centerline, repList[i + 1].centerline);
          if (d > 1e-6) localS.push(d);
        }
        const localSpacing = localS.length > 0
          ? localS.reduce((a, b) => a + b, 0) / localS.length
          : avgSpacing;
        const width = localSpacing * widthRatio;

        // All chains of this isoline get the same width
        const key = isoKey(repList[i].id);
        for (const s of fam) {
          if (isoKey(s.id) === key) {
            s.width = width;
            s.widths = new Array(s.centerline.length).fill(width);
          }
        }
      }
    }
  }

  // ── 3. faceMap was built inline in step 1 (each chain → per-segment entries)

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

  // ── 8. Prune isolated short strips (no junctions + too short to be physically meaningful)
  //
  // Isolated strips (0 junctions) contribute nothing to the Kagome weave pattern.
  // Short ones are either boundary fragments or tiny closed loops from the domain edge.
  // Keep isolated strips only if they are long enough to be potentially useful (e.g. unfold export).
  //
  // Threshold: 20% of the average non-isolated strip arc length (minimum 0.3 world units).
  // Junction-connected strips are ALWAYS kept regardless of length.
  const connectedArcs = allStrips
    .filter(s => s.junctions.length > 0)
    .map(s => {
      let arc = 0;
      for (let i = 1; i < s.centerline.length; i++) arc += s.centerline[i].distanceTo(s.centerline[i - 1]);
      return arc;
    });
  const avgConnectedArc = connectedArcs.length > 0
    ? connectedArcs.reduce((a, b) => a + b, 0) / connectedArcs.length
    : 1.0;
  // Isolated strips (no junctions) need a higher bar: at least 1.0 world units
  // (= 50 mm at the default 50 mm/unit scale), or 25% of the average connected strip.
  const minIsolatedArc = Math.max(1.0, avgConnectedArc * 0.25);

  const pruned: Strip[] = [];
  const finalStrips: Strip[] = [];
  for (const s of allStrips) {
    if (s.junctions.length > 0) {
      finalStrips.push(s);
      continue;
    }
    let arc = 0;
    for (let i = 1; i < s.centerline.length; i++) arc += s.centerline[i].distanceTo(s.centerline[i - 1]);
    if (arc >= minIsolatedArc) {
      finalStrips.push(s);
    } else {
      pruned.push(s);
    }
  }

  // Rebuild families from finalStrips
  for (let k = 0; k < 3; k++) families[k] = families[k].filter(s => finalStrips.includes(s));

  const withJunc = finalStrips.filter(s => s.junctions.length > 0).length;
  console.log(
    `[Kagome] junctions=${junctions.length}  ` +
    `strips: total=${finalStrips.length}  with-junctions=${withJunc}  ` +
    `isolated=${finalStrips.length - withJunc}  pruned=${pruned.length}` +
    (pruned.length > 0 ? ` (${pruned.map(s => s.id).join(',')})` : ''),
  );

  // ── Curvature continuity report ──────────────────────────────────────────
  logCurvatureReport(finalStrips);

  return { strips: finalStrips, junctions, families };
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
// Stitch marching-triangle segment pairs into connected polyline chains.
// Each chain also records segIndices: the original segment index (i where
// iso.points[2i]/[2i+1] is the segment) for every link in the chain.
// segIndices.length === points.length - 1 for each chain.
// ─────────────────────────────────────────────────────────────────────────────

interface StitchedChain {
  points:     THREE.Vector3[];
  segIndices: number[];   // segIndices[j] = original segment index for link j
}

function stitchSegments(rawPts: THREE.Vector3[]): StitchedChain[] {
  const n = rawPts.length / 2;
  if (n === 0) return [];

  const PREC = 500;
  const hash = (p: THREE.Vector3) =>
    `${Math.round(p.x * PREC)},${Math.round(p.y * PREC)},${Math.round(p.z * PREC)}`;

  const endpointMap = new Map<string, { segIdx: number; end: 0 | 1 }[]>();
  for (let i = 0; i < n; i++) {
    for (const end of [0, 1] as const) {
      const h = hash(rawPts[2 * i + end]);
      if (!endpointMap.has(h)) endpointMap.set(h, []);
      endpointMap.get(h)!.push({ segIdx: i, end });
    }
  }

  const used = new Uint8Array(n);
  const chains: StitchedChain[] = [];

  function pickNext(
    curPt: THREE.Vector3,
    inDir: THREE.Vector3 | null,
  ): { segIdx: number; end: 0 | 1 } | null {
    const candidates = (endpointMap.get(hash(curPt)) ?? []).filter(c => !used[c.segIdx]);
    if (candidates.length === 0) return null;
    if (candidates.length === 1 || inDir === null) return candidates[0];
    let best = candidates[0], bestDot = -Infinity;
    for (const c of candidates) {
      const dir = new THREE.Vector3()
        .subVectors(rawPts[2 * c.segIdx + (1 - c.end)], curPt).normalize();
      const d = inDir.dot(dir);
      if (d > bestDot) { bestDot = d; best = c; }
    }
    return best;
  }

  for (let start = 0; start < n; start++) {
    if (used[start]) continue;

    const pts:  THREE.Vector3[] = [rawPts[2 * start].clone(), rawPts[2 * start + 1].clone()];
    const segs: number[]        = [start];
    used[start] = 1;

    // Forward
    let ext = true;
    while (ext) {
      ext = false;
      const tail  = pts[pts.length - 1];
      const inDir = pts.length >= 2
        ? new THREE.Vector3().subVectors(tail, pts[pts.length - 2]).normalize() : null;
      const c = pickNext(tail, inDir);
      if (c) {
        pts.push(rawPts[2 * c.segIdx + (1 - c.end)].clone());
        segs.push(c.segIdx);
        used[c.segIdx] = 1;
        ext = true;
      }
    }

    // Backward
    ext = true;
    while (ext) {
      ext = false;
      const head  = pts[0];
      const inDir = pts.length >= 2
        ? new THREE.Vector3().subVectors(head, pts[1]).normalize() : null;
      const c = pickNext(head, inDir);
      if (c) {
        pts.unshift(rawPts[2 * c.segIdx + (1 - c.end)].clone());
        segs.unshift(c.segIdx);
        used[c.segIdx] = 1;
        ext = true;
      }
    }

    if (pts.length >= 2) chains.push({ points: pts, segIndices: segs });
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

// Nearest-neighbour average: sample K points from cl1, find closest point in cl2.
// Index-based comparison is wrong when two adjacent isolines start at geometrically
// distant positions (common on periodic TPMS surfaces).
function averageCenterlineDistance(cl1: THREE.Vector3[], cl2: THREE.Vector3[]): number {
  const K = Math.min(cl1.length, 8);
  if (K === 0 || cl2.length === 0) return 0.1;
  let sum = 0;
  for (let k = 0; k < K; k++) {
    const p = cl1[Math.floor(k * cl1.length / K)];
    let minD2 = Infinity;
    for (const q of cl2) {
      const d2 = p.distanceToSquared(q);
      if (d2 < minD2) minD2 = d2;
    }
    sum += Math.sqrt(minD2);
  }
  return sum / K;
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

// ─────────────────────────────────────────────────────────────────────────────
// Adaptive subdivision – insert midpoints where turning angle exceeds threshold
//
// On a TPMS with grid=50, edge length ≈ 2π/50 ≈ 0.126.
// Max geodesic curvature at saddle ≈ √2 → expected dθ/step ≈ 10°.
// Angles > 15° indicate discretization artifacts; we subdivide there.
// ─────────────────────────────────────────────────────────────────────────────

const MAX_ANGLE_DEG = 15;
const MAX_SUBDIV_ITER = 3;

function adaptiveSubdivide(
  pts: THREE.Vector3[],
  projectFn: (p: THREE.Vector3) => THREE.Vector3,
): THREE.Vector3[] {
  const maxAngleRad = MAX_ANGLE_DEG * Math.PI / 180;
  let current = pts;

  for (let iter = 0; iter < MAX_SUBDIV_ITER; iter++) {
    let needsSubdiv = false;
    const next: THREE.Vector3[] = [current[0]];

    for (let i = 1; i < current.length - 1; i++) {
      const v0 = current[i - 1], v1 = current[i], v2 = current[i + 1];
      const d1 = new THREE.Vector3().subVectors(v1, v0);
      const d2 = new THREE.Vector3().subVectors(v2, v1);
      const len1 = d1.length(), len2 = d2.length();

      if (len1 < 1e-10 || len2 < 1e-10) {
        next.push(v1);
        continue;
      }

      d1.divideScalar(len1);
      d2.divideScalar(len2);
      const cosA = Math.max(-1, Math.min(1, d1.dot(d2)));
      const angle = Math.acos(cosA);

      if (angle > maxAngleRad) {
        // Insert midpoint before v1 (between v0 and v1)
        const mid1 = new THREE.Vector3().addVectors(v0, v1).multiplyScalar(0.5);
        next.push(projectFn(mid1));
        next.push(v1);
        // Insert midpoint after v1 (between v1 and v2)
        const mid2 = new THREE.Vector3().addVectors(v1, v2).multiplyScalar(0.5);
        next.push(projectFn(mid2));
        needsSubdiv = true;
      } else {
        next.push(v1);
      }
    }

    next.push(current[current.length - 1]);
    current = next;

    if (!needsSubdiv) break;

    // Re-smooth the subdivided polyline (lightweight – 2 iterations)
    current = smoothPolyline(current, 2);
    current = current.map(p => projectFn(p));
  }

  return current;
}

// ─────────────────────────────────────────────────────────────────────────────
// Curvature continuity diagnostics – logs angle statistics per strip
// ─────────────────────────────────────────────────────────────────────────────

interface AngleStats {
  stripId: string;
  numPts: number;
  maxAngleDeg: number;
  meanAngleDeg: number;
  p95AngleDeg: number;
  sharpCount: number;   // points with angle > MAX_ANGLE_DEG
}

function computeAngleStats(stripId: string, pts: THREE.Vector3[]): AngleStats {
  const angles: number[] = [];
  for (let i = 1; i < pts.length - 1; i++) {
    const d1 = new THREE.Vector3().subVectors(pts[i], pts[i - 1]);
    const d2 = new THREE.Vector3().subVectors(pts[i + 1], pts[i]);
    const len1 = d1.length(), len2 = d2.length();
    if (len1 < 1e-10 || len2 < 1e-10) continue;
    d1.divideScalar(len1);
    d2.divideScalar(len2);
    const cosA = Math.max(-1, Math.min(1, d1.dot(d2)));
    angles.push(Math.acos(cosA) * 180 / Math.PI);
  }

  if (angles.length === 0) {
    return { stripId, numPts: pts.length, maxAngleDeg: 0, meanAngleDeg: 0, p95AngleDeg: 0, sharpCount: 0 };
  }

  angles.sort((a, b) => a - b);
  const mean = angles.reduce((s, a) => s + a, 0) / angles.length;
  const p95  = angles[Math.min(angles.length - 1, Math.floor(0.95 * angles.length))];
  const maxA = angles[angles.length - 1];
  const sharp = angles.filter(a => a > MAX_ANGLE_DEG).length;

  return { stripId, numPts: pts.length, maxAngleDeg: maxA, meanAngleDeg: mean, p95AngleDeg: p95, sharpCount: sharp };
}

export function logCurvatureReport(strips: Strip[]): void {
  const stats = strips.map(s => computeAngleStats(s.id, s.centerline));
  const hasSharp = stats.filter(s => s.sharpCount > 0);

  console.group(`[Curvature] ${strips.length} strips — threshold ${MAX_ANGLE_DEG}°`);
  console.log(`All smooth: ${stats.length - hasSharp.length}/${stats.length}`);

  if (hasSharp.length > 0) {
    console.warn(`${hasSharp.length} strip(s) still have sharp angles:`);
    if (typeof console.table === 'function') {
      console.table(hasSharp.map(s => ({
        id: s.stripId,
        pts: s.numPts,
        maxAngle: s.maxAngleDeg.toFixed(1) + '°',
        p95: s.p95AngleDeg.toFixed(1) + '°',
        mean: s.meanAngleDeg.toFixed(1) + '°',
        sharpPts: s.sharpCount,
      })));
    }
  }

  // Global summary
  const allMax = Math.max(...stats.map(s => s.maxAngleDeg));
  const allMean = stats.reduce((s, st) => s + st.meanAngleDeg, 0) / stats.length;
  console.log(`Global: max=${allMax.toFixed(1)}°  mean=${allMean.toFixed(1)}°`);
  console.groupEnd();
}

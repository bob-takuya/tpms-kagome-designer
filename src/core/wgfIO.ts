/**
 * wgfIO.ts
 *
 * Plain-text serialization for the Weaving Geodesic Foliations
 * pipeline. Used to round-trip meshes and results between the
 * in-browser app and a native CLI build (e.g. running on Google
 * Colab — see COLAB.md in the repo root).
 *
 * Input file (what the browser writes, what the CLI reads):
 *
 *   # wgf-input v1
 *   V <nV>
 *   x y z
 *   ...
 *   F <nF>
 *   a b c
 *   ...
 *   OPTS lambdaInit=... lambdaMin=... alg1MaxIter=... mu=... jointIters=...
 *        userScale=... useCover=0|1
 *   END
 *
 * Output file (what the CLI writes, what the browser reads):
 *
 *   # wgf-output v2
 *   META initialCurl=... finalCurl=... iterations=... numSingular=...
 *        segments=... components=... fam0=... fam1=... fam2=...
 *   SEG <nSeg>
 *   ax ay az bx by bz family baseFaceIdx chainId
 *   ...
 *   END
 *
 * v1 compatibility: old 8-column rows (no chainId) still parse; the
 * missing chainId field is filled with -1 ("unknown"). Downstream
 * code falls back to the pre-chain behaviour in that case.
 */

import type { HalfEdgeMesh } from './halfEdge';
import type { WgfOptions } from './wgfClient';

// ─────────────────────────────────────────────────────────────────────────────
// Export mesh + options → text
// ─────────────────────────────────────────────────────────────────────────────

export function exportMeshText(
  mesh: HalfEdgeMesh,
  opts: WgfOptions = {},
): string {
  const lines: string[] = [];
  lines.push('# wgf-input v1');
  lines.push(`V ${mesh.vertices.length}`);
  for (const p of mesh.vertices) {
    // full double precision (17 digits) so the round-trip is exact
    lines.push(`${p.x.toPrecision(17)} ${p.y.toPrecision(17)} ${p.z.toPrecision(17)}`);
  }
  lines.push(`F ${mesh.faces.length}`);
  for (const f of mesh.faces) {
    lines.push(`${f[0]} ${f[1]} ${f[2]}`);
  }
  // The CLI's paper-strict defaults (1000, 1e-3, 50, 10) kick in if OPTS
  // is omitted. We still write the currently-active values explicitly
  // so the round-trip is deterministic and the user can tweak them by
  // hand if desired.
  const optParts: string[] = [];
  if (opts.lambdaInit  !== undefined) optParts.push(`lambdaInit=${opts.lambdaInit}`);
  if (opts.lambdaMin   !== undefined) optParts.push(`lambdaMin=${opts.lambdaMin}`);
  if (opts.alg1MaxIter !== undefined) optParts.push(`alg1MaxIter=${opts.alg1MaxIter}`);
  if (opts.mu          !== undefined) optParts.push(`mu=${opts.mu}`);
  if (opts.jointIters  !== undefined) optParts.push(`jointIters=${opts.jointIters}`);
  if (opts.userScale   !== undefined) optParts.push(`userScale=${opts.userScale}`);
  if (opts.useCover    !== undefined) optParts.push(`useCover=${opts.useCover ? 1 : 0}`);
  if (optParts.length > 0) lines.push('OPTS ' + optParts.join(' '));
  lines.push('END');
  return lines.join('\n') + '\n';
}

// ─────────────────────────────────────────────────────────────────────────────
// Parse result text → structured data
// ─────────────────────────────────────────────────────────────────────────────

export interface ParsedWgfSegment {
  ax: number; ay: number; az: number;
  bx: number; by: number; bz: number;
  family:  number;
  faceIdx: number;
  // Chain identifier from the native tracer (wgf-output v2). Segments
  // with the same chainId form one contiguous polyline. -1 means the
  // field was absent (v1 file) or unknown — readers should fall back
  // to the old per-family bucketing behaviour.
  chainId: number;
}

export interface ParsedWgfResult {
  segments: ParsedWgfSegment[];
  initialCurl: number;
  finalCurl:   number;
  iterations:  number;
  numSingular: number;
  numComponents: number;
  familyCounts: [number, number, number];
}

export function parseResultText(text: string): ParsedWgfResult {
  const lines = text.split(/\r?\n/);
  const result: ParsedWgfResult = {
    segments: [],
    initialCurl: 0,
    finalCurl:   0,
    iterations:  0,
    numSingular: 0,
    numComponents: 0,
    familyCounts: [0, 0, 0],
  };

  let i = 0;
  const n = lines.length;
  while (i < n) {
    const raw = lines[i].trim();
    i++;
    if (!raw || raw.startsWith('#')) continue;

    const tokens = raw.split(/\s+/);
    const tag = tokens[0];

    if (tag === 'META') {
      for (let j = 1; j < tokens.length; j++) {
        const eq = tokens[j].indexOf('=');
        if (eq < 0) continue;
        const k = tokens[j].slice(0, eq);
        const v = tokens[j].slice(eq + 1);
        switch (k) {
          case 'initialCurl':   result.initialCurl   = parseFloat(v); break;
          case 'finalCurl':     result.finalCurl     = parseFloat(v); break;
          case 'iterations':    result.iterations    = parseInt(v, 10); break;
          case 'numSingular':   result.numSingular   = parseInt(v, 10); break;
          case 'components':    result.numComponents = parseInt(v, 10); break;
          case 'fam0':          result.familyCounts[0] = parseInt(v, 10); break;
          case 'fam1':          result.familyCounts[1] = parseInt(v, 10); break;
          case 'fam2':          result.familyCounts[2] = parseInt(v, 10); break;
          default: /* ignore */ break;
        }
      }
      continue;
    }

    if (tag === 'SEG') {
      const nSeg = parseInt(tokens[1], 10);
      result.segments.length = 0;
      for (let k = 0; k < nSeg; k++) {
        // Skip blank / comment lines between SEG and the first data row
        while (i < n) {
          const s = lines[i].trim();
          if (s && !s.startsWith('#')) break;
          i++;
        }
        if (i >= n) throw new Error(`SEG: expected ${nSeg} rows, got ${k}`);
        const p = lines[i].trim().split(/\s+/);
        i++;
        if (p.length < 8) throw new Error(`SEG row ${k}: expected >=8 cols, got ${p.length}`);
        result.segments.push({
          ax: parseFloat(p[0]), ay: parseFloat(p[1]), az: parseFloat(p[2]),
          bx: parseFloat(p[3]), by: parseFloat(p[4]), bz: parseFloat(p[5]),
          family:  parseInt(p[6], 10),
          faceIdx: parseInt(p[7], 10),
          // v2: 9th column. Default to -1 for v1 (8-column) files.
          chainId: p.length >= 9 ? parseInt(p[8], 10) : -1,
        });
      }
      continue;
    }

    if (tag === 'END') break;
  }

  return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// File-download helper (browser only)
// ─────────────────────────────────────────────────────────────────────────────

export function downloadText(filename: string, text: string): void {
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  // Revoke after a short delay so the browser has time to start the download.
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}

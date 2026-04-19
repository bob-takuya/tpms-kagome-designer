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
 *   # wgf-output vN          (N=2,3,4 — see version table below)
 *   META initialCurl=... finalCurl=... iterations=... numSingular=...
 *        segments=... components=... fam0=... fam1=... fam2=...
 *        [numIsoLevels=8]                       (v3+, optional)
 *        [resample=1] [cut=1] [prune=1]         (v3/v4 capability flags, optional)
 *   SEG <nSeg>
 *   ax ay az bx by bz family baseFaceIdx [chainId] [isoLevel] [flag]
 *   ...
 *   END
 *
 * Version table (header + per-segment columns):
 *   v1   8 cols                                  legacy
 *   v2   9 cols  (+chainId)                      one polyline per chain
 *   v3  10 cols  (+isoLevel)                     resampled along chain
 *   v4  11 cols  (+flag: 0=normal,1=cut,2=fast)  cut + prune applied
 *
 * The parser is permissive: missing trailing columns are filled with -1
 * ("unknown") so a v3/v4 reader transparently consumes older files, and
 * a v2-aware reader gracefully ignores the extra v3/v4 columns it does
 * not understand. The detected version is reported back to the caller
 * in `ParsedWgfResult.version` so the UI can light up the matching
 * controls (iso-level filter, cut-chain colouring, …).
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
  // Iso-level index within (family, baseComponent). -1 if absent (v1/v2).
  // All segments of one chain share the same isoLevel.
  isoLevel: number;
  // Per-segment classification flag (v4+). -1 if absent.
  //   0 = regular chain segment
  //   1 = cut segment (split at sharp corner / singularity)
  //   2 = fast-path segment (resampled / topologically straight chain)
  flag: number;
}

export interface ParsedWgfResult {
  /** Detected file version (1..4). 0 means no version header found. */
  version: number;
  segments: ParsedWgfSegment[];
  initialCurl: number;
  finalCurl:   number;
  iterations:  number;
  numSingular: number;
  numComponents: number;
  familyCounts: [number, number, number];
  /** Number of iso-levels declared in META (v3+). 0 if absent. */
  numIsoLevels: number;
  /** Capability flags advertised by the producer in META. */
  hasResample: boolean;
  hasCut:      boolean;
  hasPrune:    boolean;
}

export function parseResultText(text: string): ParsedWgfResult {
  const lines = text.split(/\r?\n/);
  const result: ParsedWgfResult = {
    version: 0,
    segments: [],
    initialCurl: 0,
    finalCurl:   0,
    iterations:  0,
    numSingular: 0,
    numComponents: 0,
    familyCounts: [0, 0, 0],
    numIsoLevels: 0,
    hasResample: false,
    hasCut:      false,
    hasPrune:    false,
  };

  let i = 0;
  const n = lines.length;
  while (i < n) {
    const raw = lines[i].trim();
    i++;
    if (!raw) continue;
    if (raw.startsWith('#')) {
      // "# wgf-output v3" — pull the integer that follows the trailing 'v'.
      const m = raw.match(/wgf-output\s+v(\d+)/i);
      if (m) result.version = parseInt(m[1], 10);
      continue;
    }

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
          case 'numIsoLevels':  result.numIsoLevels  = parseInt(v, 10); break;
          case 'resample':      result.hasResample   = v !== '0'; break;
          case 'cut':           result.hasCut        = v !== '0'; break;
          case 'prune':         result.hasPrune      = v !== '0'; break;
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
          chainId:  p.length >=  9 ? parseInt(p[8],  10) : -1,
          isoLevel: p.length >= 10 ? parseInt(p[9],  10) : -1,
          flag:     p.length >= 11 ? parseInt(p[10], 10) : -1,
        });
      }
      continue;
    }

    if (tag === 'END') break;
  }

  // Fall back to inferring version from per-segment column count when the
  // header was missing or unrecognised, so downstream UI still gets a
  // sensible capability hint.
  if (result.version === 0 && result.segments.length > 0) {
    const s = result.segments[0];
    if      (s.flag     >= 0) result.version = 4;
    else if (s.isoLevel >= 0) result.version = 3;
    else if (s.chainId  >= 0) result.version = 2;
    else                      result.version = 1;
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

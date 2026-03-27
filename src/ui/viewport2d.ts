/**
 * viewport2d.ts – 2D Unfold viewer
 *
 * Renders UnfoldedStrip objects onto a Canvas2D element.
 * Supports pan (drag), zoom (wheel), and fit-to-view.
 */

import * as THREE from 'three';
import { store } from '../store';
import type { KagomePattern, Junction } from '../core/kagome';
import { unfoldStrip, layoutStrips, applyLayout } from '../core/unfold';
import type { UnfoldedStrip } from '../core/unfold';
import type { HalfEdgeMesh } from '../core/halfEdge';

export interface Viewport2DContext {
  canvas:         HTMLCanvasElement;
  ctx:            CanvasRenderingContext2D;
  unfoldedStrips: UnfoldedStrip[];
  scale:          number;
  offset:         THREE.Vector2;
  isDragging:     boolean;
  lastMouse:      THREE.Vector2;
}

// ─────────────────────────────────────────────────────────────────────────────
// Setup
// ─────────────────────────────────────────────────────────────────────────────

export function createViewport2D(container: HTMLElement): Viewport2DContext {
  const canvas = document.createElement('canvas');
  canvas.className = 'canvas-2d';
  container.appendChild(canvas);

  const ctx = canvas.getContext('2d')!;

  const vp: Viewport2DContext = {
    canvas, ctx,
    unfoldedStrips: [],
    scale:  1,
    offset: new THREE.Vector2(0, 0),
    isDragging: false,
    lastMouse:  new THREE.Vector2(0, 0),
  };

  const resize = () => {
    canvas.width  = container.clientWidth;
    canvas.height = container.clientHeight;
    render2D(vp);
  };
  resize();
  window.addEventListener('resize', resize);

  // Zoom
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    // Zoom toward mouse cursor
    const mx = e.clientX - canvas.getBoundingClientRect().left;
    const my = e.clientY - canvas.getBoundingClientRect().top;
    vp.offset.x = mx + (vp.offset.x - mx) * factor;
    vp.offset.y = my + (vp.offset.y - my) * factor;
    vp.scale   *= factor;
    vp.scale    = Math.max(0.05, Math.min(20, vp.scale));
    render2D(vp);
  });

  // Pan
  canvas.addEventListener('mousedown',  (e) => { vp.isDragging = true;  vp.lastMouse.set(e.clientX, e.clientY); });
  canvas.addEventListener('mousemove',  (e) => {
    if (!vp.isDragging) return;
    vp.offset.x += e.clientX - vp.lastMouse.x;
    vp.offset.y += e.clientY - vp.lastMouse.y;
    vp.lastMouse.set(e.clientX, e.clientY);
    render2D(vp);
  });
  canvas.addEventListener('mouseup',    () => { vp.isDragging = false; });
  canvas.addEventListener('mouseleave', () => { vp.isDragging = false; });

  // Double-click → fit view
  canvas.addEventListener('dblclick', () => fitView(vp));

  return vp;
}

// ─────────────────────────────────────────────────────────────────────────────
// Fit view to current content
// ─────────────────────────────────────────────────────────────────────────────

function fitView(vp: Viewport2DContext): void {
  if (vp.unfoldedStrips.length === 0) return;

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const s of vp.unfoldedStrips) {
    minX = Math.min(minX, s.boundingBox.minX); maxX = Math.max(maxX, s.boundingBox.maxX);
    minY = Math.min(minY, s.boundingBox.minY); maxY = Math.max(maxY, s.boundingBox.maxY);
  }

  const cw = vp.canvas.width,  ch = vp.canvas.height;
  const dw = maxX - minX,      dh = maxY - minY;
  if (dw < 1 || dh < 1) return;

  const pad   = 40;
  vp.scale    = Math.min((cw - pad * 2) / dw, (ch - pad * 2) / dh);
  vp.offset.x = cw / 2 - ((minX + maxX) / 2) * vp.scale;
  vp.offset.y = ch / 2 - ((minY + maxY) / 2) * vp.scale;

  render2D(vp);
}

// ─────────────────────────────────────────────────────────────────────────────
// Regenerate unfolding
// ─────────────────────────────────────────────────────────────────────────────

export function regenerateUnfold(
  vp:        Viewport2DContext,
  pattern:   KagomePattern | null,
  mesh:      HalfEdgeMesh | null,
  _junctions: Junction[],
): void {
  if (!pattern || !mesh || pattern.strips.length === 0) {
    vp.unfoldedStrips = [];
    render2D(vp);
    return;
  }

  const state = store.getState();
  const scale = state.develop?.scale ?? 50;  // default 50 pixels per world unit
  const margin = state.develop?.margin ?? 10;

  const raw: UnfoldedStrip[] = pattern.strips.map(strip =>
    unfoldStrip(strip, mesh, pattern.junctions, scale),
  );

  const layout  = layoutStrips(raw, margin, vp.canvas.width / vp.scale * 0.9 || 600);
  vp.unfoldedStrips = applyLayout(layout);

  fitView(vp);
}

// ─────────────────────────────────────────────────────────────────────────────
// Render
// ─────────────────────────────────────────────────────────────────────────────

export function render2D(vp: Viewport2DContext): void {
  const { canvas, ctx, unfoldedStrips, scale, offset } = vp;
  const state = store.getState();

  ctx.fillStyle = '#1a1a1a';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  if (unfoldedStrips.length === 0) {
    ctx.fillStyle = '#555';
    ctx.font = '15px "Segoe UI", system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(
      'Switch to 3D view, click ▶ Calculate, then come back here',
      canvas.width / 2, canvas.height / 2,
    );
    return;
  }

  ctx.save();
  ctx.translate(offset.x, offset.y);
  ctx.scale(scale, scale);

  const lw = 1 / scale;
  const familyColors = state.kagome.layerColors;

  for (const strip of unfoldedStrips) {
    const hexColor = familyColors[strip.family];
    const col = hexColor;

    for (const seg of strip.segments) {
      if (seg.leftBoundary.length < 2) continue;

      // ── Filled strip body ─────────────────────────────────────────────────
      ctx.beginPath();
      ctx.moveTo(seg.leftBoundary[0].x, seg.leftBoundary[0].y);
      for (const p of seg.leftBoundary.slice(1))  ctx.lineTo(p.x, p.y);
      for (const p of [...seg.rightBoundary].reverse()) ctx.lineTo(p.x, p.y);
      ctx.closePath();

      ctx.fillStyle = hexToRgba(col, strip.layer === 2 ? 0.30 : strip.layer === 0 ? 0.15 : 0.22);
      ctx.fill();

      // ── Strip outline ─────────────────────────────────────────────────────
      ctx.strokeStyle = col;
      ctx.lineWidth   = lw * 1.5;

      // Left edge
      ctx.beginPath();
      ctx.moveTo(seg.leftBoundary[0].x, seg.leftBoundary[0].y);
      for (const p of seg.leftBoundary.slice(1)) ctx.lineTo(p.x, p.y);
      ctx.stroke();

      // Right edge
      ctx.beginPath();
      ctx.moveTo(seg.rightBoundary[0].x, seg.rightBoundary[0].y);
      for (const p of seg.rightBoundary.slice(1)) ctx.lineTo(p.x, p.y);
      ctx.stroke();

      // End caps
      ctx.lineWidth = lw;
      ctx.beginPath();
      ctx.moveTo(seg.leftBoundary[0].x,  seg.leftBoundary[0].y);
      ctx.lineTo(seg.rightBoundary[0].x, seg.rightBoundary[0].y);
      ctx.stroke();
      const nL = seg.leftBoundary.length;
      ctx.beginPath();
      ctx.moveTo(seg.leftBoundary[nL - 1].x,  seg.leftBoundary[nL - 1].y);
      ctx.lineTo(seg.rightBoundary[nL - 1].x, seg.rightBoundary[nL - 1].y);
      ctx.stroke();

      // ── Centerline (fold guide) ───────────────────────────────────────────
      if (state.export?.includeFoldLines && seg.centerline.length > 1) {
        ctx.strokeStyle = '#ff66ff';
        ctx.lineWidth   = lw * 0.8;
        ctx.setLineDash([3 / scale, 2 / scale]);
        ctx.beginPath();
        ctx.moveTo(seg.centerline[0].x, seg.centerline[0].y);
        for (const p of seg.centerline.slice(1)) ctx.lineTo(p.x, p.y);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // ── Junction holes ────────────────────────────────────────────────────
      for (const hole of seg.holes) {
        ctx.beginPath();
        ctx.arc(hole.center.x, hole.center.y, hole.radius, 0, Math.PI * 2);
        ctx.strokeStyle = '#00e5ff';
        ctx.lineWidth   = lw * 1.2;
        ctx.stroke();
        ctx.fillStyle   = hexToRgba('#00e5ff', 0.15);
        ctx.fill();

        if (state.export?.includeHoleIds) {
          const fs = Math.max(hole.radius * 0.8, 2 / scale);
          ctx.fillStyle     = '#00e5ff';
          ctx.font          = `${fs}px "Segoe UI", system-ui, sans-serif`;
          ctx.textAlign     = 'center';
          ctx.textBaseline  = 'middle';
          ctx.fillText(String(hole.junctionId), hole.center.x, hole.center.y);
        }
      }
    }

    // ── Strip label ───────────────────────────────────────────────────────
    if (strip.segments.length > 0 && strip.segments[0].centerline.length > 0) {
      const mid = Math.floor(strip.segments[0].centerline.length / 2);
      const lp  = strip.segments[0].centerline[mid];
      const hw  = (strip.segments[0].width / 2) + 2 / scale;
      ctx.fillStyle    = '#ddd';
      ctx.font         = `${Math.max(3 / scale, 0.4)}px "Segoe UI", system-ui, sans-serif`;
      ctx.textAlign    = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(strip.stripId, lp.x, lp.y - hw);
    }
  }

  ctx.restore();

  // Fixed-size ruler in bottom-left
  drawRuler(ctx, vp);
}

// ─────────────────────────────────────────────────────────────────────────────
// Scale ruler
// ─────────────────────────────────────────────────────────────────────────────

function drawRuler(ctx: CanvasRenderingContext2D, vp: Viewport2DContext): void {
  const state = store.getState();
  const scale = state.develop?.scale ?? 50;

  // 1 world unit → how many canvas pixels?
  const pixPerUnit = vp.scale;
  // Show a "1 unit" ruler (or 10 units if too small)
  const rulerUnits = pixPerUnit < 20 ? 10 : 1;
  const rulerPx    = pixPerUnit * rulerUnits;

  const x = 20, y = vp.canvas.height - 20;
  ctx.strokeStyle = '#aaa';
  ctx.lineWidth   = 1.5;
  ctx.beginPath();
  ctx.moveTo(x, y); ctx.lineTo(x + rulerPx, y);
  ctx.moveTo(x, y - 5); ctx.lineTo(x, y + 5);
  ctx.moveTo(x + rulerPx, y - 5); ctx.lineTo(x + rulerPx, y + 5);
  ctx.stroke();

  ctx.fillStyle    = '#aaa';
  ctx.font         = '11px "Segoe UI", system-ui, sans-serif';
  ctx.textAlign    = 'center';
  ctx.textBaseline = 'top';

  const mmPerUnit = (1000 / scale).toFixed(1); // scale = pixels per world unit
  ctx.fillText(
    `${rulerUnits} u = ${(parseFloat(mmPerUnit) * rulerUnits).toFixed(0)} mm`,
    x + rulerPx / 2, y + 7,
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

export function getUnfoldedStrips(vp: Viewport2DContext): UnfoldedStrip[] {
  return vp.unfoldedStrips;
}

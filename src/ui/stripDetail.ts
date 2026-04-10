/**
 * stripDetail.ts – Floating overlay that shows the 2D unfolded shape
 * of a single strip when clicked in the 3D viewport.
 */

import { store } from '../store';
import { unfoldStrip } from '../core/unfold';
import type { UnfoldedStrip } from '../core/unfold';
import type { Strip, Junction } from '../core/kagome';
import type { HalfEdgeMesh } from '../core/halfEdge';

// ─────────────────────────────────────────────────────────────────────────────
// DOM references (lazily created)
// ─────────────────────────────────────────────────────────────────────────────

let overlay: HTMLDivElement | null = null;
let detailCanvas: HTMLCanvasElement | null = null;
let titleEl: HTMLElement | null = null;
let infoEl: HTMLElement | null = null;

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

export function showStripDetail(
  container: HTMLElement,
  strip: Strip,
  mesh: HalfEdgeMesh,
  junctions: Junction[],
): void {
  const state = store.getState();
  const scale = state.develop.scale;

  const unfolded = unfoldStrip(strip, mesh, junctions, scale);

  ensureOverlay(container);

  overlay!.style.display = 'flex';
  overlay!.style.flexDirection = 'column';

  // Header text
  const familyNames = ['A', 'B', 'C'];
  titleEl!.textContent =
    `Strip ${strip.id}  (Family ${familyNames[strip.family]})`;

  // Info bar
  const bb = unfolded.boundingBox;
  const length = (bb.maxX - bb.minX).toFixed(1);
  const width  = (bb.maxY - bb.minY).toFixed(1);
  const holes  = unfolded.segments.reduce((s, seg) => s + seg.holes.length, 0);
  infoEl!.textContent = `L ${length} mm  |  W ${width} mm  |  Holes ${holes}`;

  renderStripDetail(detailCanvas!, unfolded, state.kagome.layerColors[strip.family]);
}

export function hideStripDetail(): void {
  if (overlay) overlay.style.display = 'none';
}

export function isStripDetailVisible(): boolean {
  return overlay?.style.display !== 'none' && overlay?.style.display !== '';
}

// ─────────────────────────────────────────────────────────────────────────────
// Overlay DOM construction (once)
// ─────────────────────────────────────────────────────────────────────────────

function ensureOverlay(container: HTMLElement): void {
  if (overlay) return;

  overlay = document.createElement('div');
  overlay.className = 'strip-detail-overlay';

  // Header
  const header = document.createElement('div');
  header.className = 'strip-detail-header';

  titleEl = document.createElement('span');
  titleEl.className = 'strip-detail-title';

  const closeBtn = document.createElement('button');
  closeBtn.className = 'strip-detail-close';
  closeBtn.textContent = '\u00d7'; // ×
  closeBtn.addEventListener('click', () => {
    hideStripDetail();
    window.dispatchEvent(new CustomEvent('strip-deselected'));
  });

  header.appendChild(titleEl);
  header.appendChild(closeBtn);

  // Canvas
  detailCanvas = document.createElement('canvas');
  detailCanvas.className = 'strip-detail-canvas';

  // Info
  infoEl = document.createElement('div');
  infoEl.className = 'strip-detail-info';

  overlay.appendChild(header);
  overlay.appendChild(detailCanvas);
  overlay.appendChild(infoEl);
  container.appendChild(overlay);
}

// ─────────────────────────────────────────────────────────────────────────────
// Canvas renderer
// ─────────────────────────────────────────────────────────────────────────────

function renderStripDetail(
  canvas: HTMLCanvasElement,
  strip: UnfoldedStrip,
  familyColor: string,
): void {
  const dpr = window.devicePixelRatio || 1;
  const displayW = canvas.clientWidth  || 340;
  const displayH = canvas.clientHeight || 160;
  canvas.width  = displayW * dpr;
  canvas.height = displayH * dpr;

  const ctx = canvas.getContext('2d')!;
  ctx.scale(dpr, dpr);

  // Background
  ctx.fillStyle = '#111';
  ctx.fillRect(0, 0, displayW, displayH);

  if (strip.segments.length === 0) return;

  // Compute transform (fit to canvas with padding)
  const bb = strip.boundingBox;
  const dw = bb.maxX - bb.minX;
  const dh = bb.maxY - bb.minY;
  if (dw < 0.01 || dh < 0.01) return;

  const pad = 16;
  const sc = Math.min((displayW - pad * 2) / dw, (displayH - pad * 2) / dh);
  const cx = displayW / 2 - ((bb.minX + bb.maxX) / 2) * sc;
  const cy = displayH / 2 - ((bb.minY + bb.maxY) / 2) * sc;

  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(sc, sc);

  const lw = 1 / sc;

  for (const seg of strip.segments) {
    if (seg.leftBoundary.length < 2) continue;

    // ── Filled body ───────────────────────────────────────────────────────
    ctx.beginPath();
    moveTo(ctx, seg.leftBoundary[0]);
    for (let i = 1; i < seg.leftBoundary.length; i++) lineTo(ctx, seg.leftBoundary[i]);
    for (let i = seg.rightBoundary.length - 1; i >= 0; i--) lineTo(ctx, seg.rightBoundary[i]);
    ctx.closePath();
    ctx.fillStyle = hexToRgba(familyColor, 0.25);
    ctx.fill();

    // ── Outline ───────────────────────────────────────────────────────────
    ctx.strokeStyle = familyColor;
    ctx.lineWidth = lw * 2;

    // Left edge
    ctx.beginPath();
    moveTo(ctx, seg.leftBoundary[0]);
    for (let i = 1; i < seg.leftBoundary.length; i++) lineTo(ctx, seg.leftBoundary[i]);
    ctx.stroke();

    // Right edge
    ctx.beginPath();
    moveTo(ctx, seg.rightBoundary[0]);
    for (let i = 1; i < seg.rightBoundary.length; i++) lineTo(ctx, seg.rightBoundary[i]);
    ctx.stroke();

    // End caps
    ctx.lineWidth = lw;
    const nL = seg.leftBoundary.length;
    ctx.beginPath();
    moveTo(ctx, seg.leftBoundary[0]);
    lineTo(ctx, seg.rightBoundary[0]);
    ctx.stroke();
    ctx.beginPath();
    moveTo(ctx, seg.leftBoundary[nL - 1]);
    lineTo(ctx, seg.rightBoundary[nL - 1]);
    ctx.stroke();

    // ── Centerline (fold) ─────────────────────────────────────────────────
    if (seg.centerline.length > 1) {
      ctx.strokeStyle = '#ff66ff';
      ctx.lineWidth = lw * 0.8;
      ctx.setLineDash([3 / sc, 2 / sc]);
      ctx.beginPath();
      moveTo(ctx, seg.centerline[0]);
      for (let i = 1; i < seg.centerline.length; i++) lineTo(ctx, seg.centerline[i]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // ── Junction holes ────────────────────────────────────────────────────
    for (const hole of seg.holes) {
      ctx.beginPath();
      ctx.arc(hole.center.x, hole.center.y, hole.radius, 0, Math.PI * 2);
      ctx.strokeStyle = '#00e5ff';
      ctx.lineWidth = lw;
      ctx.stroke();
      ctx.fillStyle = hexToRgba('#00e5ff', 0.2);
      ctx.fill();

      // Hole ID label
      const fs = Math.max(hole.radius * 0.7, 2 / sc);
      ctx.fillStyle = '#00e5ff';
      ctx.font = `${fs}px "Segoe UI", system-ui, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(hole.junctionId), hole.center.x, hole.center.y);
    }
  }

  ctx.restore();
}

// ─────────────────────────────────────────────────────────────────────────────
// Tiny helpers
// ─────────────────────────────────────────────────────────────────────────────

function moveTo(ctx: CanvasRenderingContext2D, p: { x: number; y: number }): void {
  ctx.moveTo(p.x, p.y);
}

function lineTo(ctx: CanvasRenderingContext2D, p: { x: number; y: number }): void {
  ctx.lineTo(p.x, p.y);
}

function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

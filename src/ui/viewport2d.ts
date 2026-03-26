import * as THREE from 'three';
import { store } from '../store';
import type { KagomePattern, Junction } from '../core/kagome';
import { unfoldStrip, layoutStrips, applyLayout } from '../core/unfold';
import type { UnfoldedStrip } from '../core/unfold';
import type { HalfEdgeMesh } from '../core/halfEdge';

export interface Viewport2DContext {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  unfoldedStrips: UnfoldedStrip[];
  scale: number;
  offset: THREE.Vector2;
  isDragging: boolean;
  lastMouse: THREE.Vector2;
}

export function createViewport2D(container: HTMLElement): Viewport2DContext {
  const canvas = document.createElement('canvas');
  canvas.className = 'canvas-2d';
  container.appendChild(canvas);

  const ctx = canvas.getContext('2d')!;

  const viewport: Viewport2DContext = {
    canvas,
    ctx,
    unfoldedStrips: [],
    scale: 1,
    offset: new THREE.Vector2(0, 0),
    isDragging: false,
    lastMouse: new THREE.Vector2(0, 0),
  };

  // Handle resize
  const resize = () => {
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    render2D(viewport);
  };
  resize();
  window.addEventListener('resize', resize);

  // Pan and zoom
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    viewport.scale *= factor;
    viewport.scale = Math.max(0.1, Math.min(10, viewport.scale));
    render2D(viewport);
  });

  canvas.addEventListener('mousedown', (e) => {
    viewport.isDragging = true;
    viewport.lastMouse.set(e.clientX, e.clientY);
  });

  canvas.addEventListener('mousemove', (e) => {
    if (viewport.isDragging) {
      const dx = e.clientX - viewport.lastMouse.x;
      const dy = e.clientY - viewport.lastMouse.y;
      viewport.offset.x += dx;
      viewport.offset.y += dy;
      viewport.lastMouse.set(e.clientX, e.clientY);
      render2D(viewport);
    }
  });

  canvas.addEventListener('mouseup', () => {
    viewport.isDragging = false;
  });

  canvas.addEventListener('mouseleave', () => {
    viewport.isDragging = false;
  });

  return viewport;
}

export function regenerateUnfold(
  viewport: Viewport2DContext,
  pattern: KagomePattern | null,
  mesh: HalfEdgeMesh | null,
  junctions: Junction[]
): void {
  if (!pattern || !mesh) {
    viewport.unfoldedStrips = [];
    render2D(viewport);
    return;
  }

  const state = store.getState();

  // Unfold each strip
  const unfoldedStrips: UnfoldedStrip[] = [];

  for (const strip of pattern.strips) {
    const unfolded = unfoldStrip(strip, mesh, junctions, state.develop.scale);
    unfoldedStrips.push(unfolded);
  }

  // Layout strips
  const layout = layoutStrips(unfoldedStrips, state.develop.margin);
  viewport.unfoldedStrips = applyLayout(layout);

  // Reset view to fit content
  if (viewport.unfoldedStrips.length > 0) {
    viewport.scale = Math.min(
      viewport.canvas.width / (layout.totalWidth * 1.2),
      viewport.canvas.height / (layout.totalHeight * 1.2)
    );
    viewport.offset.set(
      viewport.canvas.width / 2 - (layout.totalWidth * viewport.scale) / 2,
      viewport.canvas.height / 2 - (layout.totalHeight * viewport.scale) / 2
    );
  }

  render2D(viewport);
}

export function render2D(viewport: Viewport2DContext): void {
  const { canvas, ctx, unfoldedStrips, scale, offset } = viewport;
  const state = store.getState();

  // Clear canvas
  ctx.fillStyle = '#1a1a1a';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  if (unfoldedStrips.length === 0) {
    // Draw placeholder text
    ctx.fillStyle = '#666';
    ctx.font = '16px "Segoe UI", system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Generate a pattern to see 2D unfolding', canvas.width / 2, canvas.height / 2);
    return;
  }

  ctx.save();
  ctx.translate(offset.x, offset.y);
  ctx.scale(scale, scale);

  const familyColors = state.kagome.layerColors;

  for (const strip of unfoldedStrips) {
    const color = familyColors[strip.family];

    for (const segment of strip.segments) {
      // Draw strip boundaries
      ctx.strokeStyle = color;
      ctx.lineWidth = 1 / scale;
      ctx.beginPath();

      // Left boundary
      if (segment.leftBoundary.length > 0) {
        ctx.moveTo(segment.leftBoundary[0].x, segment.leftBoundary[0].y);
        for (let i = 1; i < segment.leftBoundary.length; i++) {
          ctx.lineTo(segment.leftBoundary[i].x, segment.leftBoundary[i].y);
        }
      }
      ctx.stroke();

      // Right boundary
      ctx.beginPath();
      if (segment.rightBoundary.length > 0) {
        ctx.moveTo(segment.rightBoundary[0].x, segment.rightBoundary[0].y);
        for (let i = 1; i < segment.rightBoundary.length; i++) {
          ctx.lineTo(segment.rightBoundary[i].x, segment.rightBoundary[i].y);
        }
      }
      ctx.stroke();

      // End caps
      if (segment.leftBoundary.length > 0 && segment.rightBoundary.length > 0) {
        ctx.beginPath();
        ctx.moveTo(segment.leftBoundary[0].x, segment.leftBoundary[0].y);
        ctx.lineTo(segment.rightBoundary[0].x, segment.rightBoundary[0].y);
        ctx.stroke();

        ctx.beginPath();
        const lastL = segment.leftBoundary[segment.leftBoundary.length - 1];
        const lastR = segment.rightBoundary[segment.rightBoundary.length - 1];
        ctx.moveTo(lastL.x, lastL.y);
        ctx.lineTo(lastR.x, lastR.y);
        ctx.stroke();
      }

      // Fold lines (centerline)
      if (state.export.includeFoldLines && segment.centerline.length > 1) {
        ctx.strokeStyle = '#ff00ff';
        ctx.setLineDash([4 / scale, 2 / scale]);
        ctx.beginPath();
        ctx.moveTo(segment.centerline[0].x, segment.centerline[0].y);
        for (let i = 1; i < segment.centerline.length; i++) {
          ctx.lineTo(segment.centerline[i].x, segment.centerline[i].y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Holes
      ctx.strokeStyle = '#00ffff';
      for (const hole of segment.holes) {
        ctx.beginPath();
        ctx.arc(hole.center.x, hole.center.y, hole.radius, 0, Math.PI * 2);
        ctx.stroke();

        // Hole ID
        if (state.export.includeHoleIds) {
          ctx.fillStyle = '#00ffff';
          ctx.font = `${6 / scale}px "Segoe UI", system-ui, sans-serif`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(String(hole.junctionId), hole.center.x, hole.center.y);
        }
      }
    }

    // Strip label
    if (strip.segments.length > 0 && strip.segments[0].centerline.length > 0) {
      const labelPos = strip.segments[0].centerline[0].clone();
      labelPos.y += strip.segments[0].width * 0.7;
      ctx.fillStyle = '#ffffff';
      ctx.font = `${8 / scale}px "Segoe UI", system-ui, sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillText(strip.stripId, labelPos.x, labelPos.y);
    }
  }

  ctx.restore();
}

export function getUnfoldedStrips(viewport: Viewport2DContext): UnfoldedStrip[] {
  return viewport.unfoldedStrips;
}

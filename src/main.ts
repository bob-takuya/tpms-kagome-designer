import './style.css';
import { store } from './store';
import { createSidebar, downloadFile } from './ui/sidebar';
import { createViewport3D, regenerateMesh, regeneratePattern, updateColors } from './ui/viewport3d';
import { createViewport2D, regenerateUnfold, getUnfoldedStrips } from './ui/viewport2d';
import { createStatusBar } from './ui/statusbar';
import { generateDXF, generateJunctionCSV } from './export/dxf';
import { generateSVG } from './export/svg';

// Create main layout
const app = document.getElementById('app')!;
app.innerHTML = `
  <div class="layout">
    <div class="sidebar-container"></div>
    <div class="main-container">
      <div class="viewport-tabs">
        <button class="tab-btn active" data-view="3d">3D View</button>
        <button class="tab-btn" data-view="2d">2D Unfold</button>
      </div>
      <div class="viewport-container">
        <div class="viewport viewport-3d active"></div>
        <div class="viewport viewport-2d"></div>
      </div>
      <div class="status-container"></div>
    </div>
  </div>
`;

// Initialize components
const sidebarContainer = document.querySelector('.sidebar-container')!;
const viewport3DContainer = document.querySelector('.viewport-3d')!;
const viewport2DContainer = document.querySelector('.viewport-2d')!;
const statusContainer = document.querySelector('.status-container')!;

createSidebar(sidebarContainer as HTMLElement);
const ctx3D = createViewport3D(viewport3DContainer as HTMLElement);
const ctx2D = createViewport2D(viewport2DContainer as HTMLElement);
createStatusBar(statusContainer as HTMLElement);

// Tab switching
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const view = btn.getAttribute('data-view');

    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    document.querySelectorAll('.viewport').forEach(v => v.classList.remove('active'));
    document.querySelector(`.viewport-${view}`)?.classList.add('active');

    store.getState().setViewMode(view as '3d' | '2d');

    if (view === '2d') {
      regenerateUnfold(ctx2D, ctx3D.kagomePattern, ctx3D.halfEdgeMesh, ctx3D.kagomePattern?.junctions || []);
    }
  });
});

// Event handlers
window.addEventListener('regenerate-mesh', () => {
  regenerateMesh(ctx3D);
  if (store.getState().viewMode === '2d') {
    regenerateUnfold(ctx2D, ctx3D.kagomePattern, ctx3D.halfEdgeMesh, ctx3D.kagomePattern?.junctions || []);
  }
});

window.addEventListener('regenerate-pattern', () => {
  regeneratePattern(ctx3D);
  if (store.getState().viewMode === '2d') {
    regenerateUnfold(ctx2D, ctx3D.kagomePattern, ctx3D.halfEdgeMesh, ctx3D.kagomePattern?.junctions || []);
  }
});

window.addEventListener('regenerate-unfold', () => {
  regenerateUnfold(ctx2D, ctx3D.kagomePattern, ctx3D.halfEdgeMesh, ctx3D.kagomePattern?.junctions || []);
});

window.addEventListener('update-colors', () => {
  updateColors(ctx3D);
});

window.addEventListener('export-dxf', () => {
  const strips = getUnfoldedStrips(ctx2D);
  if (strips.length === 0) {
    alert('Generate a pattern first');
    return;
  }

  const state = store.getState();
  const dxf = generateDXF(strips, state.export.includeHoleIds, state.export.includeFoldLines);
  downloadFile(dxf, 'kagome-pattern.dxf', 'application/dxf');
});

window.addEventListener('export-svg', () => {
  const strips = getUnfoldedStrips(ctx2D);
  if (strips.length === 0) {
    alert('Generate a pattern first');
    return;
  }

  const state = store.getState();

  // Calculate bounds
  let maxX = 0, maxY = 0;
  for (const strip of strips) {
    maxX = Math.max(maxX, strip.boundingBox.maxX);
    maxY = Math.max(maxY, strip.boundingBox.maxY);
  }

  const svg = generateSVG(strips, maxX + 20, maxY + 20, {
    includeHoleIds: state.export.includeHoleIds,
    includeFoldLines: state.export.includeFoldLines,
    strokeWidth: 0.5,
  });
  downloadFile(svg, 'kagome-pattern.svg', 'image/svg+xml');
});

window.addEventListener('export-csv', () => {
  const strips = getUnfoldedStrips(ctx2D);
  if (strips.length === 0) {
    alert('Generate a pattern first');
    return;
  }

  const csv = generateJunctionCSV(strips);
  downloadFile(csv, 'kagome-junctions.csv', 'text/csv');
});

// Declare custom events for TypeScript
declare global {
  interface WindowEventMap {
    'regenerate-mesh': CustomEvent;
    'regenerate-pattern': CustomEvent;
    'regenerate-unfold': CustomEvent;
    'update-colors': CustomEvent;
    'export-dxf': CustomEvent;
    'export-svg': CustomEvent;
    'export-csv': CustomEvent;
  }
}

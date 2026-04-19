import './style.css';
import { store } from './store';
import { createSidebar, downloadFile } from './ui/sidebar';
import {
  createViewport3D, regenerateMesh, regeneratePattern,
  updateColors, highlightStrip, applyImportedPattern,
  renderIsolines,
} from './ui/viewport3d';
import { createViewport2D, regenerateUnfold, getUnfoldedStrips } from './ui/viewport2d';
import { showStripDetail, hideStripDetail } from './ui/stripDetail';
import { createStatusBar } from './ui/statusbar';
import { generateDXF, generateJunctionCSV } from './export/dxf';
import { generateSVG } from './export/svg';
import { exportMeshText, parseResultText, downloadText } from './core/wgfIO';

// ─── Diagnostic overlay ───────────────────────────────────────────────────
// Lightweight on-screen error/status banner that works even when remote
// debugging is not available. Especially useful for iOS Safari where
// unhandled rejections (e.g. WASM load failures) would otherwise be
// invisible and present as "the viewport is blank".
const banner = document.createElement('div');
banner.id = 'wgf-diag';
banner.style.cssText = [
  'position:fixed', 'top:0', 'left:0', 'right:0',
  'max-height:40vh', 'overflow:auto',
  'background:rgba(150,0,0,0.92)', 'color:#fff',
  'padding:6px 10px', 'font:11px/1.35 ui-monospace,monospace',
  'z-index:99999', 'white-space:pre-wrap', 'display:none',
  'box-shadow:0 2px 8px rgba(0,0,0,0.4)',
].join(';');
document.body.appendChild(banner);
function diag(msg: string, isError = false) {
  banner.style.display = 'block';
  if (isError) banner.style.background = 'rgba(150,0,0,0.92)';
  banner.textContent += msg + '\n';
  banner.scrollTop = banner.scrollHeight;
}
window.addEventListener('error', (e) => {
  diag('[error] ' + (e.error?.stack || e.error?.message || e.message || String(e)), true);
});
window.addEventListener('unhandledrejection', (e) => {
  const r = e.reason;
  diag('[rejection] ' + (r?.stack || r?.message || String(r)), true);
});
// Expose for other modules to push status lines without spamming console.
(window as unknown as { wgfDiag: (m: string) => void }).wgfDiag = (m: string) => diag(m);

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
      // Force canvas resize AFTER the DOM has painted the now-visible container
      requestAnimationFrame(() => {
        const c = ctx2D.canvas;
        const container = c.parentElement!;
        if (container.clientWidth > 0 && (c.width !== container.clientWidth || c.height !== container.clientHeight)) {
          c.width  = container.clientWidth;
          c.height = container.clientHeight;
        }
        regenerateUnfold(ctx2D, ctx3D.kagomePattern, ctx3D.halfEdgeMesh, ctx3D.kagomePattern?.junctions || []);
      });
    }
  });
});

// Event handlers
window.addEventListener('regenerate-mesh', () => {
  // regenerateMesh only builds the surface mesh now. It does NOT
  // trigger the heavy WGF pipeline automatically. The user must press
  // "Generate Pattern" to start that computation.
  regenerateMesh(ctx3D);
  if (store.getState().viewMode === '2d') {
    regenerateUnfold(ctx2D, ctx3D.kagomePattern, ctx3D.halfEdgeMesh, ctx3D.kagomePattern?.junctions || []);
  }
});

// ── Pattern generation driver (progress UI + async WGF pipeline) ────────────
let wgfGenerating = false;
async function runPatternGeneration(): Promise<void> {
  if (wgfGenerating) return;      // prevent re-entry
  if (!ctx3D.halfEdgeMesh) {
    diag('[wgf] no mesh yet — press "Rebuild Mesh" first', true);
    return;
  }
  wgfGenerating = true;

  const genBtn  = document.getElementById('generate-btn') as HTMLButtonElement | null;
  const progBox = document.getElementById('wgf-progress') as HTMLElement | null;
  const label   = progBox?.querySelector('.wgf-progress-label') as HTMLElement | null;
  const fill    = progBox?.querySelector('.wgf-progress-fill')  as HTMLElement | null;
  const count   = progBox?.querySelector('.wgf-progress-count') as HTMLElement | null;

  if (genBtn) { genBtn.disabled = true; genBtn.textContent = '⏳ Generating…'; }
  if (progBox) progBox.style.display = 'block';
  if (label)   label.textContent = 'Starting…';
  if (fill)    fill.style.width = '0%';
  if (count)   count.textContent = '';

  const stageCount = 7;
  try {
    await regeneratePattern(ctx3D, (p) => {
      // Compose overall progress from (coarse stage index) + (intra-stage
      // fraction for the heavy per-component phase). This gives a
      // monotonic bar that advances smoothly through long stages.
      const intra = p.total > 0 ? p.cur / p.total : 0;
      const overall = Math.min(1, Math.max(0, (p.stage + intra) / stageCount));
      if (label) label.textContent = `§${p.stage}. ${p.label}`;
      if (fill)  fill.style.width = `${(overall * 100).toFixed(1)}%`;
      if (count && p.total > 1) count.textContent = `${p.cur} / ${p.total}`;
      else if (count) count.textContent = '';
    });

    const pat = ctx3D.kagomePattern;
    if (store.getState().viewMode === '2d') {
      regenerateUnfold(ctx2D, pat, ctx3D.halfEdgeMesh, pat ? pat.junctions : []);
    }
    if (label)   label.textContent = '✓ Done';
    if (fill)    fill.style.width = '100%';
    // Fade the overlay out after a short delay.
    setTimeout(() => { if (progBox) progBox.style.display = 'none'; }, 1500);
  } catch (err) {
    if (label)   label.textContent = '✗ ' + (err as Error).message;
    if (fill)    fill.style.width = '0%';
  } finally {
    if (genBtn) { genBtn.disabled = false; genBtn.textContent = '✦ Generate Pattern'; }
    wgfGenerating = false;
  }
}

window.addEventListener('generate-pattern', () => { void runPatternGeneration(); });
// Legacy alias — some components still dispatch 'regenerate-pattern'.
window.addEventListener('regenerate-pattern', () => { void runPatternGeneration(); });

// ── Export for Colab ────────────────────────────────────────────────────────
window.addEventListener('export-for-colab', () => {
  const mesh = ctx3D.halfEdgeMesh;
  if (!mesh) {
    diag('[wgf] no mesh yet — press "Rebuild Mesh" first', true);
    return;
  }
  // Paper-strict defaults for the Colab run (native is fast enough).
  const text = exportMeshText(mesh, {
    lambdaInit:  1000,
    lambdaMin:   1e-3,
    alg1MaxIter: 50,
    mu:          1e-4,
    jointIters:  10,
    userScale:   store.getState().strip.numIsolines / 8.0,
    useCover:    true,
  });
  downloadText('wgf-input.txt', text);
});

// ── Import Colab result ─────────────────────────────────────────────────────
window.addEventListener('import-colab-result', ((e: CustomEvent<{ text: string }>) => {
  if (!ctx3D.halfEdgeMesh) {
    diag('[wgf] no mesh — press "Rebuild Mesh" first', true);
    return;
  }
  try {
    const parsed = parseResultText(e.detail.text);
    applyImportedPattern(ctx3D, parsed);
    const pat = ctx3D.kagomePattern;
    if (store.getState().viewMode === '2d') {
      regenerateUnfold(ctx2D, pat, ctx3D.halfEdgeMesh, pat ? pat.junctions : []);
    }
  } catch (err) {
    diag('[wgf import] ' + (err as Error).message, true);
  }
}) as EventListener);

window.addEventListener('regenerate-unfold', () => {
  regenerateUnfold(ctx2D, ctx3D.kagomePattern, ctx3D.halfEdgeMesh, ctx3D.kagomePattern?.junctions || []);
});

window.addEventListener('update-colors', () => {
  updateColors(ctx3D);
});

// Cheap overlay rebuild: ribbon/line toggle, min-chain-segs, tube radius,
// rainbow mode. Does NOT touch Kagome strips/junctions or rerun WGF.
window.addEventListener('isoline-view-changed', () => {
  renderIsolines(ctx3D);
});

// ─── Strip click → show 2D unfold overlay ──────────────────────────────────

window.addEventListener('strip-selected', ((e: CustomEvent<{ stripId: string }>) => {
  const { stripId } = e.detail;
  if (!ctx3D.kagomePattern || !ctx3D.halfEdgeMesh) return;

  const strip = ctx3D.kagomePattern.strips.find(s => s.id === stripId);
  if (!strip) return;

  showStripDetail(
    viewport3DContainer as HTMLElement,
    strip,
    ctx3D.halfEdgeMesh,
    ctx3D.kagomePattern.junctions,
  );
}) as EventListener);

window.addEventListener('strip-deselected', () => {
  hideStripDetail();
  highlightStrip(ctx3D, null);
});

// Hide overlay when pattern is regenerated
window.addEventListener('regenerate-mesh', () => {
  hideStripDetail();
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
    'generate-pattern': CustomEvent;
    'export-for-colab': CustomEvent;
    'import-colab-result': CustomEvent<{ text: string }>;
    'regenerate-unfold': CustomEvent;
    'update-colors': CustomEvent;
    'isoline-view-changed': CustomEvent;
    'wgf-meta-updated': CustomEvent;
    'export-dxf': CustomEvent;
    'export-svg': CustomEvent;
    'export-csv': CustomEvent;
    'strip-selected': CustomEvent<{ stripId: string }>;
    'strip-deselected': CustomEvent;
  }
}

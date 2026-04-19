import { store } from '../store';
import type { SurfaceType } from '../store';

export function createSidebar(container: HTMLElement): void {
  const sidebar = document.createElement('div');
  sidebar.className = 'sidebar';
  sidebar.innerHTML = `
    <div class="sidebar-header">
      <h1>TPMS Kagome Designer</h1>
      <button class="toggle-btn" id="toggle-sidebar">
        <span class="toggle-icon">◀</span>
      </button>
    </div>
    <div class="sidebar-content">
      ${createAccordionSection('TPMS', createTPMSControls())}
      ${createAccordionSection('Noise', createNoiseControls())}
      ${createAccordionSection('Strip', createStripControls())}
      ${createAccordionSection('Kagome', createKagomeControls())}
      ${createAccordionSection('Isoline view', createIsolineViewControls())}
      ${createAccordionSection('Develop', createDevelopControls())}
      ${createAccordionSection('Export', createExportControls())}
    </div>
  `;

  // Inject the two primary action buttons before the sidebar content.
  // "Rebuild Mesh" runs marching cubes + half-edge construction only
  // (cheap). "Generate Pattern" runs the heavy Vekhter 2019 WGF
  // pipeline in a Web Worker and streams progress. Splitting them lets
  // the user tweak TPMS parameters and preview the surface geometry
  // without triggering a minutes-long ribbon computation each time.
  const calcSection = document.createElement('div');
  calcSection.className = 'calculate-section';
  calcSection.innerHTML = `
    <button id="calculate-btn" class="calculate-btn">▶ Rebuild Mesh</button>
    <button id="generate-btn" class="calculate-btn generate-btn">✦ Generate Pattern</button>
    <span id="calc-status" class="calc-status"></span>
    <div id="wgf-progress" class="wgf-progress" style="display:none">
      <div class="wgf-progress-label"></div>
      <div class="wgf-progress-bar"><div class="wgf-progress-fill"></div></div>
      <div class="wgf-progress-count"></div>
    </div>
    <div class="colab-row">
      <button id="export-colab-btn" class="secondary-btn" title="Export the current mesh to wgf-input.txt for running on Google Colab — see COLAB.md">⇩ Export for Colab</button>
      <button id="import-colab-btn" class="secondary-btn" title="Import a wgf-output.txt produced by the Colab CLI">⇪ Import Colab result</button>
      <input type="file" id="import-colab-file" accept=".txt,text/plain" style="display:none">
    </div>
  `;
  sidebar.insertBefore(calcSection, sidebar.querySelector('.sidebar-content')!);

  container.appendChild(sidebar);

  setupEventListeners();
}

function createAccordionSection(title: string, content: string): string {
  return `
    <div class="accordion-section">
      <button class="accordion-header" data-section="${title.toLowerCase()}">
        <span class="section-title">§${title}</span>
        <span class="accordion-icon">▼</span>
      </button>
      <div class="accordion-content" id="section-${title.toLowerCase()}">
        ${content}
      </div>
    </div>
  `;
}

function createTPMSControls(): string {
  const state = store.getState();
  return `
    <div class="control-group">
      <label for="surface-type">Surface Type</label>
      <select id="surface-type">
        <option value="gyroid" ${state.tpms.surfaceType === 'gyroid' ? 'selected' : ''}>Gyroid</option>
        <option value="schwarzP" ${state.tpms.surfaceType === 'schwarzP' ? 'selected' : ''}>Schwarz P</option>
        <option value="schwarzD" ${state.tpms.surfaceType === 'schwarzD' ? 'selected' : ''}>Schwarz D</option>
      </select>
    </div>
    <div class="control-group">
      <label for="period">Period λ</label>
      <input type="range" id="period" min="1" max="20" step="0.1" value="${state.tpms.period}">
      <span class="value-display" id="period-value">${state.tpms.period.toFixed(1)}</span>
    </div>
    <div class="control-group">
      <label for="base-t">Base t₀</label>
      <input type="range" id="base-t" min="-1" max="1" step="0.01" value="${state.tpms.baseT}">
      <span class="value-display" id="base-t-value">${state.tpms.baseT.toFixed(2)}</span>
    </div>
    <div class="control-group">
      <label for="bbox-min">Bounding Box Min</label>
      <input type="range" id="bbox-min" min="-10" max="0" step="0.1" value="${state.tpms.boundingBox.min}">
      <span class="value-display" id="bbox-min-value">${state.tpms.boundingBox.min.toFixed(1)}</span>
    </div>
    <div class="control-group">
      <label for="bbox-max">Bounding Box Max</label>
      <input type="range" id="bbox-max" min="0" max="10" step="0.1" value="${state.tpms.boundingBox.max}">
      <span class="value-display" id="bbox-max-value">${state.tpms.boundingBox.max.toFixed(1)}</span>
    </div>
    <div class="control-group">
      <label for="resolution">Grid Resolution</label>
      <input type="range" id="resolution" min="20" max="150" step="5" value="${state.tpms.gridResolution}">
      <span class="value-display" id="resolution-value">${state.tpms.gridResolution}</span>
    </div>
  `;
}

function createNoiseControls(): string {
  const state = store.getState();
  return `
    <div class="control-group">
      <label for="noise-enabled">
        <input type="checkbox" id="noise-enabled" ${state.noise.enabled ? 'checked' : ''}>
        Enable Noise
      </label>
    </div>
    <div class="control-group">
      <label for="noise-amplitude">Amplitude A</label>
      <input type="range" id="noise-amplitude" min="0" max="1" step="0.01" value="${state.noise.amplitude}">
      <span class="value-display" id="noise-amplitude-value">${state.noise.amplitude.toFixed(2)}</span>
    </div>
    <div class="control-group">
      <label for="noise-frequency">Frequency f</label>
      <input type="range" id="noise-frequency" min="0.1" max="5" step="0.1" value="${state.noise.frequency}">
      <span class="value-display" id="noise-frequency-value">${state.noise.frequency.toFixed(1)}</span>
    </div>
    <div class="control-group">
      <label for="noise-seed">Seed</label>
      <input type="number" id="noise-seed" min="1" max="9999" value="${state.noise.seed}">
    </div>
  `;
}

function createStripControls(): string {
  const state = store.getState();
  return `
    <div class="control-group">
      <label for="num-isolines">アイソライン本数 N</label>
      <input type="range" id="num-isolines" min="2" max="20" step="1" value="${state.strip.numIsolines}">
      <span class="value-display" id="num-isolines-value">${state.strip.numIsolines}</span>
    </div>
    <div class="control-group">
      <label for="strip-method">幅の決め方</label>
      <select id="strip-method">
        <option value="A" ${state.strip.method === 'A' ? 'selected' : ''}>A - アイソライン間隔比（非均一）</option>
        <option value="B" ${state.strip.method === 'B' ? 'selected' : ''}>B - mm 直接指定（均一）</option>
      </select>
    </div>
    <div class="control-group" id="strip-width-mm-group">
      <label for="strip-width-mm">ストリップ幅 (mm) <small>[B案]</small></label>
      <input type="range" id="strip-width-mm" min="1" max="50" step="0.5" value="${state.strip.stripWidthMm}">
      <span class="value-display" id="strip-width-mm-value">${state.strip.stripWidthMm} mm</span>
    </div>
    <div class="control-group" id="width-ratio-group">
      <label for="width-ratio">幅比率 ρ <small>[A案]</small></label>
      <input type="range" id="width-ratio" min="0.1" max="0.95" step="0.05" value="${state.strip.widthRatio}">
      <span class="value-display" id="width-ratio-value">${state.strip.widthRatio.toFixed(2)}</span>
    </div>
  `;
}

function createKagomeControls(): string {
  const state = store.getState();
  return `
    <div class="control-group">
      <label for="hole-radius">ジャンクション穴径 (mm)</label>
      <input type="range" id="hole-radius" min="0.5" max="10" step="0.5" value="${state.kagome.holeRadiusMm}">
      <span class="value-display" id="hole-radius-value">${state.kagome.holeRadiusMm} mm</span>
    </div>
    <div class="control-group">
      <label>Layer Colors</label>
      <div class="color-row">
        <input type="color" id="layer-color-0" value="${state.kagome.layerColors[0]}">
        <input type="color" id="layer-color-1" value="${state.kagome.layerColors[1]}">
        <input type="color" id="layer-color-2" value="${state.kagome.layerColors[2]}">
      </div>
    </div>
  `;
}

function createIsolineViewControls(): string {
  const v = store.getState().isolineView;
  return `
    <div class="control-group">
      <label for="iso-mode">描画モード</label>
      <select id="iso-mode">
        <option value="ribbon" ${v.mode === 'ribbon' ? 'selected' : ''}>Ribbon (TubeGeometry)</option>
        <option value="line"   ${v.mode === 'line'   ? 'selected' : ''}>Line (debug)</option>
      </select>
    </div>
    <div class="control-group">
      <label for="iso-min-segs">最小 chain セグメント数</label>
      <input type="range" id="iso-min-segs" min="1" max="300" step="1" value="${v.minChainSegs}">
      <span class="value-display" id="iso-min-segs-value">${v.minChainSegs}</span>
    </div>
    <div class="control-group">
      <label for="iso-tube-radius">Tube 半径 (world units)</label>
      <input type="range" id="iso-tube-radius" min="0.005" max="0.1" step="0.005" value="${v.tubeRadius}">
      <span class="value-display" id="iso-tube-radius-value">${v.tubeRadius.toFixed(3)}</span>
    </div>
    <div class="control-group">
      <label>Family filter</label>
      <div class="toggle-row" id="iso-family-row">
        ${renderFamilyToggles(v.familyMask)}
      </div>
    </div>
    <div class="control-group" id="iso-level-group" style="display:none">
      <label>
        Iso-level filter
        <small id="iso-level-version-note"></small>
      </label>
      <div class="toggle-row" id="iso-level-row"></div>
    </div>
    <div class="control-group" id="iso-flag-group" style="display:none">
      <label>v4 chain class</label>
      <label for="iso-hl-cut">
        <input type="checkbox" id="iso-hl-cut" ${v.highlightCutChains ? 'checked' : ''}>
        cut chain を強調
      </label>
      <label for="iso-hl-fast">
        <input type="checkbox" id="iso-hl-fast" ${v.highlightFastChains ? 'checked' : ''}>
        fast-path chain を強調
      </label>
    </div>
    <div class="control-group">
      <label for="iso-rainbow">
        <input type="checkbox" id="iso-rainbow" ${v.chainIdColor ? 'checked' : ''}>
        chainId 毎に虹色 (debug)
      </label>
    </div>
    <div class="control-group">
      <label for="iso-adj-eps">Adjacency eps (debug)</label>
      <select id="iso-adj-eps">
        <option value="1e-6" ${v.adjacencyEps === 1e-6 ? 'selected' : ''}>1e-6 (tight)</option>
        <option value="1e-5" ${v.adjacencyEps === 1e-5 ? 'selected' : ''}>1e-5</option>
        <option value="1e-4" ${v.adjacencyEps === 1e-4 ? 'selected' : ''}>1e-4</option>
        <option value="1e-3" ${v.adjacencyEps === 1e-3 ? 'selected' : ''}>1e-3</option>
        <option value="1e-2" ${v.adjacencyEps === 1e-2 ? 'selected' : ''}>1e-2 (loose)</option>
      </select>
    </div>
    <div class="control-group">
      <small id="wgf-version-line" class="muted">wgf-output: (none loaded)</small>
    </div>
  `;
}

function renderFamilyToggles(mask: number): string {
  const labels = ['A', 'B', 'C'];
  return labels.map((lbl, k) => {
    const on = (mask & (1 << k)) !== 0;
    return `<button type="button" class="toggle-btn-pill ${on ? 'on' : 'off'}" data-fam="${k}">${lbl}</button>`;
  }).join('');
}

function renderIsoLevelToggles(mask: number, count: number): string {
  if (count <= 0) return '';
  const buttons: string[] = [];
  for (let k = 0; k < count; k++) {
    const on = (mask & (1 << k)) !== 0;
    buttons.push(
      `<button type="button" class="toggle-btn-pill ${on ? 'on' : 'off'}" data-iso="${k}">${k}</button>`,
    );
  }
  return buttons.join('');
}

function createDevelopControls(): string {
  const state = store.getState();
  return `
    <div class="control-group">
      <label for="develop-scale">スケール (mm/unit)</label>
      <input type="range" id="develop-scale" min="5" max="200" step="5" value="${state.develop.scale}">
      <span class="value-display" id="develop-scale-value">${state.develop.scale} mm</span>
    </div>
    <div class="control-group">
      <label for="develop-margin">Margin (mm)</label>
      <input type="range" id="develop-margin" min="1" max="50" step="1" value="${state.develop.margin}">
      <span class="value-display" id="develop-margin-value">${state.develop.margin}</span>
    </div>
  `;
}

function createExportControls(): string {
  const state = store.getState();
  return `
    <div class="control-group">
      <label for="include-hole-ids">
        <input type="checkbox" id="include-hole-ids" ${state.export.includeHoleIds ? 'checked' : ''}>
        Include Hole IDs
      </label>
    </div>
    <div class="control-group">
      <label for="include-fold-lines">
        <input type="checkbox" id="include-fold-lines" ${state.export.includeFoldLines ? 'checked' : ''}>
        Include Fold Lines
      </label>
    </div>
    <div class="button-group">
      <button class="export-btn" id="export-dxf">Export DXF</button>
      <button class="export-btn" id="export-svg">Export SVG</button>
      <button class="export-btn" id="export-csv">Export CSV</button>
    </div>
    <div class="button-group">
      <button class="secondary-btn" id="export-json">Save JSON</button>
      <button class="secondary-btn" id="import-json">Load JSON</button>
    </div>
  `;
}

function setupEventListeners(): void {
  // ── Rebuild Mesh button (cheap: marching cubes + half-edge) ────────────────
  document.getElementById('calculate-btn')?.addEventListener('click', () => {
    const btn    = document.getElementById('calculate-btn') as HTMLButtonElement;
    const status = document.getElementById('calc-status')!;
    btn.disabled = true;
    btn.textContent = '⏳ Rebuilding…';
    status.textContent = '';

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        window.dispatchEvent(new CustomEvent('regenerate-mesh'));
        btn.disabled = false;
        btn.textContent = '▶ Rebuild Mesh';
        status.textContent = '✓';
        setTimeout(() => { status.textContent = ''; }, 1500);
      });
    });
  });

  // ── Generate Pattern button (heavy: full WGF pipeline in a Worker) ─────────
  // Dispatches `generate-pattern`. main.ts owns the progress-UI wiring.
  document.getElementById('generate-btn')?.addEventListener('click', () => {
    window.dispatchEvent(new CustomEvent('generate-pattern'));
  });

  // ── Export for Colab / Import Colab result ─────────────────────────────────
  // See COLAB.md at the repo root for the full round-trip instructions.
  document.getElementById('export-colab-btn')?.addEventListener('click', () => {
    window.dispatchEvent(new CustomEvent('export-for-colab'));
  });
  const importInput = document.getElementById('import-colab-file') as HTMLInputElement | null;
  document.getElementById('import-colab-btn')?.addEventListener('click', () => {
    importInput?.click();
  });
  importInput?.addEventListener('change', async () => {
    const file = importInput.files?.[0];
    if (!file) return;
    const text = await file.text();
    importInput.value = '';   // allow re-picking the same file
    window.dispatchEvent(new CustomEvent('import-colab-result', { detail: { text } }));
  });

  // ── Accordion toggle ────────────────────────────────────────────────────────
  document.querySelectorAll('.accordion-header').forEach(header => {
    header.addEventListener('click', () => {
      const section = header.getAttribute('data-section');
      const content = document.getElementById(`section-${section}`);
      const icon = header.querySelector('.accordion-icon');
      if (content && icon) {
        content.classList.toggle('collapsed');
        icon.textContent = content.classList.contains('collapsed') ? '▶' : '▼';
      }
    });
  });

  // Sidebar toggle
  document.getElementById('toggle-sidebar')?.addEventListener('click', () => {
    store.getState().toggleSidebar();
    document.querySelector('.sidebar')?.classList.toggle('collapsed');
  });

  // ── TPMS controls – update store only (Calculate triggers recompute) ────────
  setupSlider('surface-type', 'select', (value) => {
    store.getState().setTPMS({ surfaceType: value as SurfaceType });
  });
  setupSlider('period', 'range', (value) => {
    store.getState().setTPMS({ period: parseFloat(value) });
  });
  setupSlider('base-t', 'range', (value) => {
    store.getState().setTPMS({ baseT: parseFloat(value) });
  });
  setupSlider('bbox-min', 'range', (value) => {
    const s = store.getState();
    s.setTPMS({ boundingBox: { ...s.tpms.boundingBox, min: parseFloat(value) } });
  });
  setupSlider('bbox-max', 'range', (value) => {
    const s = store.getState();
    s.setTPMS({ boundingBox: { ...s.tpms.boundingBox, max: parseFloat(value) } });
  });
  setupSlider('resolution', 'range', (value) => {
    store.getState().setTPMS({ gridResolution: parseInt(value) });
  });

  // ── Noise controls (store-only) ─────────────────────────────────────────────
  document.getElementById('noise-enabled')?.addEventListener('change', (e) => {
    store.getState().setNoise({ enabled: (e.target as HTMLInputElement).checked });
  });
  setupSlider('noise-amplitude', 'range', (value) => {
    store.getState().setNoise({ amplitude: parseFloat(value) });
  });
  setupSlider('noise-frequency', 'range', (value) => {
    store.getState().setNoise({ frequency: parseFloat(value) });
  });
  document.getElementById('noise-seed')?.addEventListener('change', (e) => {
    store.getState().setNoise({ seed: parseInt((e.target as HTMLInputElement).value) });
  });

  // ── Strip controls (store-only) ─────────────────────────────────────────────
  setupSlider('num-isolines', 'range', (value) => {
    store.getState().setStrip({ numIsolines: parseInt(value) });
  });
  setupSlider('strip-method', 'select', (value) => {
    store.getState().setStrip({ method: value as 'A' | 'B' });
  });
  setupSlider('strip-width-mm', 'range', (value) => {
    const v = parseFloat(value);
    store.getState().setStrip({ stripWidthMm: v });
    const el = document.getElementById('strip-width-mm-value');
    if (el) el.textContent = `${v} mm`;
  });
  setupSlider('width-ratio', 'range', (value) => {
    const v = parseFloat(value);
    store.getState().setStrip({ widthRatio: v });
    const el = document.getElementById('width-ratio-value');
    if (el) el.textContent = v.toFixed(2);
  });

  // ── Isoline view controls (live → dispatch 'isoline-view-changed') ─────────
  // These toggle the overlay rendering only; no WGF recompute needed.
  document.getElementById('iso-mode')?.addEventListener('change', (e) => {
    const mode = (e.target as HTMLSelectElement).value as 'ribbon' | 'line';
    store.getState().setIsolineView({ mode });
    window.dispatchEvent(new CustomEvent('isoline-view-changed'));
  });
  // Use 'change' (fires on slider release) for the range sliders so we
  // don't rebuild 7k tubes on every intermediate drag frame.
  const minSegsEl = document.getElementById('iso-min-segs') as HTMLInputElement | null;
  minSegsEl?.addEventListener('input', (e) => {
    // Reflect the live value in the label while dragging…
    const v = parseInt((e.target as HTMLInputElement).value, 10);
    const lbl = document.getElementById('iso-min-segs-value');
    if (lbl) lbl.textContent = String(v);
  });
  minSegsEl?.addEventListener('change', (e) => {
    const v = parseInt((e.target as HTMLInputElement).value, 10);
    store.getState().setIsolineView({ minChainSegs: v });
    window.dispatchEvent(new CustomEvent('isoline-view-changed'));
  });
  const radiusEl = document.getElementById('iso-tube-radius') as HTMLInputElement | null;
  radiusEl?.addEventListener('input', (e) => {
    const v = parseFloat((e.target as HTMLInputElement).value);
    const lbl = document.getElementById('iso-tube-radius-value');
    if (lbl) lbl.textContent = v.toFixed(3);
  });
  radiusEl?.addEventListener('change', (e) => {
    const v = parseFloat((e.target as HTMLInputElement).value);
    store.getState().setIsolineView({ tubeRadius: v });
    window.dispatchEvent(new CustomEvent('isoline-view-changed'));
  });
  document.getElementById('iso-rainbow')?.addEventListener('change', (e) => {
    const on = (e.target as HTMLInputElement).checked;
    store.getState().setIsolineView({ chainIdColor: on });
    window.dispatchEvent(new CustomEvent('isoline-view-changed'));
  });
  document.getElementById('iso-adj-eps')?.addEventListener('change', (e) => {
    const v = parseFloat((e.target as HTMLSelectElement).value);
    store.getState().setIsolineView({ adjacencyEps: v });
    window.dispatchEvent(new CustomEvent('isoline-view-changed'));
  });

  // Family-toggle pills
  document.getElementById('iso-family-row')?.addEventListener('click', (e) => {
    const t = e.target as HTMLElement;
    const famAttr = t.getAttribute('data-fam');
    if (famAttr === null) return;
    const k = parseInt(famAttr, 10);
    const cur = store.getState().isolineView.familyMask;
    const next = cur ^ (1 << k);
    store.getState().setIsolineView({ familyMask: next });
    t.classList.toggle('on');
    t.classList.toggle('off');
    window.dispatchEvent(new CustomEvent('isoline-view-changed'));
  });

  // Iso-level toggle pills (delegated; the row is repopulated when META updates)
  document.getElementById('iso-level-row')?.addEventListener('click', (e) => {
    const t = e.target as HTMLElement;
    const isoAttr = t.getAttribute('data-iso');
    if (isoAttr === null) return;
    const k = parseInt(isoAttr, 10);
    const cur = store.getState().isolineView.isoLevelMask;
    const next = (cur ^ (1 << k)) >>> 0;   // keep unsigned
    store.getState().setIsolineView({ isoLevelMask: next });
    t.classList.toggle('on');
    t.classList.toggle('off');
    window.dispatchEvent(new CustomEvent('isoline-view-changed'));
  });

  document.getElementById('iso-hl-cut')?.addEventListener('change', (e) => {
    const on = (e.target as HTMLInputElement).checked;
    store.getState().setIsolineView({ highlightCutChains: on });
    window.dispatchEvent(new CustomEvent('isoline-view-changed'));
  });
  document.getElementById('iso-hl-fast')?.addEventListener('change', (e) => {
    const on = (e.target as HTMLInputElement).checked;
    store.getState().setIsolineView({ highlightFastChains: on });
    window.dispatchEvent(new CustomEvent('isoline-view-changed'));
  });

  // Refresh version-aware widgets when an import/generate publishes new META.
  window.addEventListener('wgf-meta-updated', refreshWgfMetaUI);
  // Initial pass so the version line is consistent at load time.
  refreshWgfMetaUI();

  // ── Kagome controls (store-only) ────────────────────────────────────────────
  setupSlider('hole-radius', 'range', (value) => {
    const v = parseFloat(value);
    store.getState().setKagome({ holeRadiusMm: v });
    const el = document.getElementById('hole-radius-value');
    if (el) el.textContent = `${v} mm`;
  });

  for (let i = 0; i < 3; i++) {
    document.getElementById(`layer-color-${i}`)?.addEventListener('change', (e) => {
      const state = store.getState();
      const colors = [...state.kagome.layerColors] as [string, string, string];
      colors[i] = (e.target as HTMLInputElement).value;
      store.getState().setKagome({ layerColors: colors });
      window.dispatchEvent(new CustomEvent('update-colors'));
    });
  }

  // Develop controls
  setupSlider('develop-scale', 'range', (value) => {
    const v = parseFloat(value);
    store.getState().setDevelop({ scale: v });
    const el = document.getElementById('develop-scale-value');
    if (el) el.textContent = `${v} mm`;
    window.dispatchEvent(new CustomEvent('regenerate-unfold'));
  });

  setupSlider('develop-margin', 'range', (value) => {
    store.getState().setDevelop({ margin: parseFloat(value) });
    window.dispatchEvent(new CustomEvent('regenerate-unfold'));
  });

  // Export controls
  document.getElementById('include-hole-ids')?.addEventListener('change', (e) => {
    store.getState().setExport({ includeHoleIds: (e.target as HTMLInputElement).checked });
  });

  document.getElementById('include-fold-lines')?.addEventListener('change', (e) => {
    store.getState().setExport({ includeFoldLines: (e.target as HTMLInputElement).checked });
  });

  document.getElementById('export-dxf')?.addEventListener('click', () => {
    window.dispatchEvent(new CustomEvent('export-dxf'));
  });

  document.getElementById('export-svg')?.addEventListener('click', () => {
    window.dispatchEvent(new CustomEvent('export-svg'));
  });

  document.getElementById('export-csv')?.addEventListener('click', () => {
    window.dispatchEvent(new CustomEvent('export-csv'));
  });

  document.getElementById('export-json')?.addEventListener('click', () => {
    const json = store.getState().exportJSON();
    downloadFile(json, 'tpms-kagome-settings.json', 'application/json');
  });

  document.getElementById('import-json')?.addEventListener('click', () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          const json = e.target?.result as string;
          store.getState().importJSON(json);
          window.dispatchEvent(new CustomEvent('regenerate-mesh'));
          location.reload();
        };
        reader.readAsText(file);
      }
    };
    input.click();
  });
}

function setupSlider(id: string, type: string, onChange: (value: string) => void): void {
  const element = document.getElementById(id);
  if (!element) return;

  const valueDisplay = document.getElementById(`${id}-value`);

  element.addEventListener(type === 'select' ? 'change' : 'input', (e) => {
    const value = (e.target as HTMLInputElement | HTMLSelectElement).value;
    if (valueDisplay) {
      if (type === 'range') {
        const num = parseFloat(value);
        valueDisplay.textContent = Number.isInteger(num) ? String(num) : num.toFixed(2);
      }
    }
    onChange(value);
  });
}

/**
 * Re-render the version-dependent parts of the Isoline-view section
 * after a wgf-output file is loaded (or the in-browser pipeline runs).
 *
 *   v1/v2 → only family filter + min-segs slider
 *   v3+   → also iso-level toggle row (count from META)
 *   v4    → also cut/fast-path highlight toggles
 *
 * Filters never disappear from the store — they just stop being shown
 * in the sidebar. That way an old session's familyMask stays valid if a
 * v4 file is reloaded later.
 */
function refreshWgfMetaUI(): void {
  const meta = store.getState().wgfMeta;
  const view = store.getState().isolineView;

  const versionLine = document.getElementById('wgf-version-line');
  if (versionLine) {
    if (meta.version === 0) {
      versionLine.textContent = 'wgf-output: (none loaded)';
    } else {
      const caps: string[] = [];
      if (meta.hasResample) caps.push('resample');
      if (meta.hasCut)      caps.push('cut');
      if (meta.hasPrune)    caps.push('prune');
      const capStr = caps.length ? ` [${caps.join(',')}]` : '';
      versionLine.textContent =
        `wgf-output: v${meta.version}` +
        (meta.numIsoLevels > 0 ? ` · ${meta.numIsoLevels} iso-levels` : '') +
        capStr;
    }
  }

  const levelGroup = document.getElementById('iso-level-group') as HTMLElement | null;
  const levelRow   = document.getElementById('iso-level-row');
  const versionNote = document.getElementById('iso-level-version-note');
  if (levelGroup && levelRow) {
    if (meta.numIsoLevels > 0) {
      levelGroup.style.display = '';
      levelRow.innerHTML = renderIsoLevelToggles(view.isoLevelMask, meta.numIsoLevels);
      if (versionNote) versionNote.textContent = ` (v${meta.version})`;
    } else {
      levelGroup.style.display = 'none';
      levelRow.innerHTML = '';
      if (versionNote) versionNote.textContent = '';
    }
  }

  const flagGroup = document.getElementById('iso-flag-group') as HTMLElement | null;
  if (flagGroup) {
    flagGroup.style.display = (meta.version >= 4 || meta.hasCut) ? '' : 'none';
  }
}

export function downloadFile(content: string, filename: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

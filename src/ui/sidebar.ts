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
      ${createAccordionSection('Develop', createDevelopControls())}
      ${createAccordionSection('Export', createExportControls())}
    </div>
  `;

  // Inject Calculate button before sidebar content
  const calcSection = document.createElement('div');
  calcSection.className = 'calculate-section';
  calcSection.innerHTML = `
    <button id="calculate-btn" class="calculate-btn">▶ Calculate</button>
    <span id="calc-status" class="calc-status"></span>
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
      <label for="num-isolines">Number of Isolines N</label>
      <input type="range" id="num-isolines" min="2" max="20" step="1" value="${state.strip.numIsolines}">
      <span class="value-display" id="num-isolines-value">${state.strip.numIsolines}</span>
    </div>
    <div class="control-group">
      <label for="strip-method">Method</label>
      <select id="strip-method">
        <option value="A" ${state.strip.method === 'A' ? 'selected' : ''}>A - Isoline Pair</option>
        <option value="B" ${state.strip.method === 'B' ? 'selected' : ''}>B - Width Ratio</option>
      </select>
    </div>
    <div class="control-group">
      <label for="strip-width">Strip Width</label>
      <input type="range" id="strip-width" min="0.01" max="0.5" step="0.01" value="${state.strip.stripWidth}">
      <span class="value-display" id="strip-width-value">${state.strip.stripWidth.toFixed(2)}</span>
    </div>
    <div class="control-group">
      <label for="width-ratio">Width Ratio ρ</label>
      <input type="range" id="width-ratio" min="0.1" max="0.9" step="0.05" value="${state.strip.widthRatio}">
      <span class="value-display" id="width-ratio-value">${state.strip.widthRatio.toFixed(2)}</span>
    </div>
  `;
}

function createKagomeControls(): string {
  const state = store.getState();
  return `
    <div class="control-group">
      <label for="hole-radius">Hole Radius</label>
      <input type="range" id="hole-radius" min="0.005" max="0.1" step="0.005" value="${state.kagome.holeRadius}">
      <span class="value-display" id="hole-radius-value">${state.kagome.holeRadius.toFixed(3)}</span>
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

function createDevelopControls(): string {
  const state = store.getState();
  return `
    <div class="control-group">
      <label for="develop-scale">Scale (mm/unit)</label>
      <input type="range" id="develop-scale" min="10" max="500" step="10" value="${state.develop.scale}">
      <span class="value-display" id="develop-scale-value">${state.develop.scale}</span>
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
  // ── Calculate button ────────────────────────────────────────────────────────
  // Heavy computation (Phase 1+2+3) is only triggered here, not on slider drag.
  document.getElementById('calculate-btn')?.addEventListener('click', () => {
    const btn    = document.getElementById('calculate-btn') as HTMLButtonElement;
    const status = document.getElementById('calc-status')!;
    btn.disabled = true;
    btn.textContent = '⏳ Calculating…';
    status.textContent = '';

    // Yield to the browser so the button UI updates before blocking work starts
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        window.dispatchEvent(new CustomEvent('regenerate-mesh'));
        btn.disabled = false;
        btn.textContent = '▶ Calculate';
        status.textContent = '✓';
        setTimeout(() => { status.textContent = ''; }, 1500);
      });
    });
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
  setupSlider('strip-width', 'range', (value) => {
    store.getState().setStrip({ stripWidth: parseFloat(value) });
  });
  setupSlider('width-ratio', 'range', (value) => {
    store.getState().setStrip({ widthRatio: parseFloat(value) });
  });

  // ── Kagome controls (store-only) ────────────────────────────────────────────
  setupSlider('hole-radius', 'range', (value) => {
    store.getState().setKagome({ holeRadius: parseFloat(value) });
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
    store.getState().setDevelop({ scale: parseFloat(value) });
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

export function downloadFile(content: string, filename: string, mimeType: string): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

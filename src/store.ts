import { createStore } from 'zustand/vanilla';

export type SurfaceType = 'gyroid' | 'schwarzP' | 'schwarzD';

export interface TPMSParams {
  surfaceType: SurfaceType;
  period: number;
  baseT: number;
  boundingBox: { min: number; max: number };
  gridResolution: number;
}

export interface NoiseParams {
  enabled: boolean;
  amplitude: number;
  frequency: number;
  seed: number;
}

export interface StripParams {
  numIsolines:   number;
  stripWidthMm:  number;  // physical strip width in mm (used by method B)
  widthRatio:    number;  // fraction of isoline spacing (used by method A)
  method:        'A' | 'B';
}

export interface KagomeParams {
  holeRadiusMm:  number;  // junction hole radius in mm
  layerColors:   [string, string, string];
}

export interface IsolineViewParams {
  /** 'ribbon' → TubeGeometry per chain, 'line' → thin LineSegments (debug). */
  mode:         'ribbon' | 'line';
  /** Minimum number of segments a chain must have to be drawn. Short
   *  "tanzaku" chains get pruned so the viewport isn't flooded.
   *  Applies to BOTH the chain overlay and the kagome strip ribbons. */
  minChainSegs: number;
  /** Tube radius in world units (before mm-per-unit scaling). */
  tubeRadius:   number;
  /** Debug: colour each chain by index (HSL rainbow) instead of by family. */
  chainIdColor: boolean;
  /** Quantisation grid spacing used to match endpoint 3D positions when
   *  rebuilding polylines from the flat segment list. With the v2
   *  exporter the base-mesh 3D points of a shared edge are computed
   *  independently per face, so 1e-6 misses every real junction —
   *  exposing this as a user-adjustable slider lets us diagnose the
   *  projection bug without rebuilding the WASM module. */
  adjacencyEps: number;
  /** Bitmask of visible families (bit k → family k). Default 0b111 = all on. */
  familyMask:   number;
  /** Bitmask of visible iso-levels (bit k → iso-level k). Up to 32 levels.
   *  Chains with isoLevel<0 (v1/v2 imports without per-level info) are
   *  always shown. Default 0xFFFFFFFF = all on. */
  isoLevelMask: number;
  /** Highlight cut chains (v4 flag=1) in a distinct colour. */
  highlightCutChains: boolean;
  /** Highlight fast-path chains (v4 flag=2) in a distinct colour. */
  highlightFastChains: boolean;
}

/** Capability + version info detected from the most recently loaded
 *  wgf-output file (or from the in-browser pipeline). Drives which
 *  filter widgets the sidebar exposes. */
export interface WgfMeta {
  /** Detected file version (1..4). 0 = nothing loaded yet. */
  version:      number;
  /** Number of iso-levels reported in META (v3+). 0 means "filter UI off". */
  numIsoLevels: number;
  hasResample:  boolean;
  hasCut:       boolean;
  hasPrune:     boolean;
}

export interface DevelopParams {
  scale: number;
  margin: number;
}

export interface ExportParams {
  includeHoleIds: boolean;
  includeFoldLines: boolean;
}

export interface AppState {
  tpms: TPMSParams;
  noise: NoiseParams;
  strip: StripParams;
  kagome: KagomeParams;
  isolineView: IsolineViewParams;
  develop: DevelopParams;
  export: ExportParams;
  wgfMeta: WgfMeta;
  viewMode: '3d' | '2d';
  sidebarCollapsed: boolean;
  stats: {
    vertices: number;
    faces: number;
    strips: number;
    junctions: number;
  };
}

export interface AppActions {
  setTPMS: (params: Partial<TPMSParams>) => void;
  setNoise: (params: Partial<NoiseParams>) => void;
  setStrip: (params: Partial<StripParams>) => void;
  setKagome: (params: Partial<KagomeParams>) => void;
  setIsolineView: (params: Partial<IsolineViewParams>) => void;
  setDevelop: (params: Partial<DevelopParams>) => void;
  setExport: (params: Partial<ExportParams>) => void;
  setWgfMeta: (meta: Partial<WgfMeta>) => void;
  setViewMode: (mode: '3d' | '2d') => void;
  toggleSidebar: () => void;
  setStats: (stats: Partial<AppState['stats']>) => void;
  exportJSON: () => string;
  importJSON: (json: string) => void;
}

const defaultState: AppState = {
  tpms: {
    surfaceType: 'gyroid',
    period: 2 * Math.PI,
    baseT: 0,
    boundingBox: { min: -Math.PI, max: Math.PI },
    gridResolution: 50,
  },
  noise: {
    enabled: false,
    amplitude: 0.2,
    frequency: 0.5,
    seed: 42,
  },
  strip: {
    numIsolines:  8,
    stripWidthMm: 10,   // 10 mm default
    widthRatio:   0.75, // method A: 75% of inter-isoline spacing
    method:       'B' as 'A' | 'B',
  },
  kagome: {
    holeRadiusMm: 2,    // 2 mm default
    layerColors:  ['#ff4444', '#ffff44', '#44ff44'] as [string, string, string],
  },
  isolineView: {
    mode:         'ribbon' as 'ribbon' | 'line',
    minChainSegs: 10,
    tubeRadius:   0.02,
    chainIdColor: false,
    adjacencyEps: 1e-6,
    familyMask:   0b111,
    isoLevelMask: 0xFFFFFFFF >>> 0,    // unsigned 32-bit, all bits set
    highlightCutChains:  false,
    highlightFastChains: false,
  },
  develop: {
    scale: 50,   // mm per TPMS world unit (1 unit ≈ 50 mm → period ≈ 314 mm)
    margin: 5,
  },
  export: {
    includeHoleIds: true,
    includeFoldLines: true,
  },
  wgfMeta: {
    version:      0,
    numIsoLevels: 0,
    hasResample:  false,
    hasCut:       false,
    hasPrune:     false,
  },
  viewMode: '3d',
  sidebarCollapsed: false,
  stats: {
    vertices: 0,
    faces: 0,
    strips: 0,
    junctions: 0,
  },
};

export const store = createStore<AppState & AppActions>((set, get) => ({
  ...defaultState,

  setTPMS: (params) => set((state) => ({ tpms: { ...state.tpms, ...params } })),
  setNoise: (params) => set((state) => ({ noise: { ...state.noise, ...params } })),
  setStrip: (params) => set((state) => ({ strip: { ...state.strip, ...params } })),
  setKagome: (params) => set((state) => ({ kagome: { ...state.kagome, ...params } })),
  setIsolineView: (params) =>
    set((state) => ({ isolineView: { ...state.isolineView, ...params } })),
  setDevelop: (params) => set((state) => ({ develop: { ...state.develop, ...params } })),
  setExport: (params) => set((state) => ({ export: { ...state.export, ...params } })),
  setWgfMeta: (meta) => set((state) => ({ wgfMeta: { ...state.wgfMeta, ...meta } })),
  setViewMode: (mode) => set({ viewMode: mode }),
  toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
  setStats: (stats) => set((state) => ({ stats: { ...state.stats, ...stats } })),

  exportJSON: () => {
    const state = get();
    const exportData = {
      tpms: state.tpms,
      noise: state.noise,
      strip: state.strip,
      kagome: state.kagome,
      isolineView: state.isolineView,
      develop: state.develop,
      export: state.export,
    };
    return JSON.stringify(exportData, null, 2);
  },

  importJSON: (json) => {
    try {
      const data = JSON.parse(json);
      set({
        tpms: { ...defaultState.tpms, ...data.tpms },
        noise: { ...defaultState.noise, ...data.noise },
        strip: { ...defaultState.strip, ...data.strip },
        kagome: { ...defaultState.kagome, ...data.kagome },
        isolineView: { ...defaultState.isolineView, ...data.isolineView },
        develop: { ...defaultState.develop, ...data.develop },
        export: { ...defaultState.export, ...data.export },
      });
    } catch {
      console.error('Failed to import JSON');
    }
  },
}));

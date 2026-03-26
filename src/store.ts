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
  numIsolines: number;
  stripWidth: number;
  widthRatio: number;
  method: 'A' | 'B';
}

export interface KagomeParams {
  holeRadius: number;
  layerColors: [string, string, string];
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
  develop: DevelopParams;
  export: ExportParams;
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
  setDevelop: (params: Partial<DevelopParams>) => void;
  setExport: (params: Partial<ExportParams>) => void;
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
    numIsolines: 8,
    stripWidth: 0.1,
    widthRatio: 0.3,
    method: 'B',
  },
  kagome: {
    holeRadius: 0.02,
    layerColors: ['#ff4444', '#ffff44', '#44ff44'],
  },
  develop: {
    scale: 100,
    margin: 10,
  },
  export: {
    includeHoleIds: true,
    includeFoldLines: true,
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
  setDevelop: (params) => set((state) => ({ develop: { ...state.develop, ...params } })),
  setExport: (params) => set((state) => ({ export: { ...state.export, ...params } })),
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
        develop: { ...defaultState.develop, ...data.develop },
        export: { ...defaultState.export, ...data.export },
      });
    } catch {
      console.error('Failed to import JSON');
    }
  },
}));

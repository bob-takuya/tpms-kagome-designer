import { createNoise3D } from 'simplex-noise';
import type { SurfaceType } from '../store';

export type ImplicitFunction = (x: number, y: number, z: number) => number;

// Simple seeded random number generator
function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
}

export function createTPMSFunction(
  type: SurfaceType,
  baseT: number,
  noiseEnabled: boolean,
  noiseAmplitude: number,
  noiseFrequency: number,
  noiseSeed: number
): ImplicitFunction {
  const noise3D = createNoise3D(seededRandom(noiseSeed));

  const baseFunctions: Record<SurfaceType, ImplicitFunction> = {
    gyroid: (x, y, z) => Math.sin(x) * Math.cos(y) + Math.sin(y) * Math.cos(z) + Math.sin(z) * Math.cos(x),
    schwarzP: (x, y, z) => Math.cos(x) + Math.cos(y) + Math.cos(z),
    schwarzD: (x, y, z) => Math.cos(x) * Math.cos(y) * Math.cos(z) - Math.sin(x) * Math.sin(y) * Math.sin(z),
  };

  const baseFunc = baseFunctions[type];

  return (x: number, y: number, z: number) => {
    let t = baseT;
    if (noiseEnabled) {
      t += noiseAmplitude * noise3D(x * noiseFrequency, y * noiseFrequency, z * noiseFrequency);
    }
    return baseFunc(x, y, z) - t;
  };
}

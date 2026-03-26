import { defineConfig } from 'vite';

export default defineConfig({
  base: '/tpms-kagome-designer/',
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
  server: {
    open: true,
  },
});

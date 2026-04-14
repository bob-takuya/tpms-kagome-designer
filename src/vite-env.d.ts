/// <reference types="vite/client" />

// Emscripten-generated WGF glue (SINGLE_FILE=1 ES6 module).
// Built by `npm run build:wasm` into src/wasm/wgf.js.
declare module '../wasm/wgf.js' {
  const createWgfModule: (opts?: unknown) => Promise<unknown>;
  export default createWgfModule;
}

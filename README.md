# TPMS Kagome Designer

A browser tool that generates a triaxial (kagome) weaving pattern of strips on triply periodic minimal surfaces (Gyroid, Schwarz P, Schwarz D) and unrolls the strips flat for laser cutting.

TPMS（ジャイロイド等）の曲面上にカゴメ編みのストリップ配置を計算し、レーザーカット用に平面展開する実験的なブラウザツール。

## Status

**Experimental.** The full chain (surface → strip pattern → flat strips → DXF/SVG) runs, but it is a research prototype: several parameters are debug-level, parts of the underlying method are not implemented, and the quality of the resulting patterns has not been validated by fabrication in this repository.

- **Works**
  - Implicit TPMS surfaces (Gyroid, Schwarz P, Schwarz D) with period, offset *t₀*, bounding box, grid resolution and optional noise; meshed with marching cubes
  - Pattern generation with a C++ implementation of the *Weaving Geodesic Foliations* pipeline (Vekhter et al. 2019), compiled to WebAssembly and run in a Web Worker with a progress bar
  - The same C++ code as a native command-line tool, with a Google Colab workflow for large meshes (see [COLAB.md](COLAB.md))
  - Strips unrolled to 2D (arc-length / geodesic-curvature unrolling), with junction holes
  - Export: DXF (R12, layered CUT / HOLE / SCORE / LABEL, laid out for nesting), SVG, junction CSV; save / load parameters as JSON
  - `tsc --noEmit` passes
- **Partial**
  - Post-processing implements resampling, cutting at high geodesic curvature and pruning, but not the "extend to crossings" step from the paper
  - Several sidebar controls are debug options (line/ribbon draw mode, chain filters, adjacency epsilon)
  - The unrolled strips are an approximation (constant width, developable approximation); physical fit on the surface is not checked
  - Older TypeScript stripe-field code (`src/core/connectionLaplacian.ts`, `geodesicField.ts`) remains alongside the WASM pipeline
- **Not implemented**
  - Automated tests (the `test/` scripts are manual harnesses)
  - Surfaces other than the three TPMS types
- **Known issues**
  - The in-browser WASM run is much slower than the native build (roughly 10–30× on a laptop, per COLAB.md); large resolutions can take minutes
  - `npm run dev` / `npm run build` need the WASM module (`src/wasm/wgf.js`), which is not committed — run `npm run build:wasm` first (requires Emscripten)
  - `test/geodesic-smoke.ts` is out of date and fails to import (`computeGeodesicFoliationStripeFields` no longer exists); `test/stripe-quality.ts` runs
  - Debug scripts (`debug-schwarzd.ts`, `local-kagome.ts`) live in the repository root
  - UI labels are a mix of English and Japanese

## Demo

The GitHub Pages workflow (`.github/workflows/deploy.yml`) installs Emscripten, builds the WASM module and publishes `main` to:
https://bob-takuya.github.io/tpms-kagome-designer/

## Background

Developed March–April 2026 as a design tool related to the author's studio work on strip models of gyroid surfaces: the aim was to cover a TPMS with three families of woven strips (a kagome weave) and get flat, cuttable strip outlines.

## Usage

1. **TPMS** / **Noise**: pick the surface type and parameters, then click **▶ Rebuild Mesh**.
2. **Strip** / **Kagome**: set the number of isolines and the strip width (method A: ratio of isoline spacing; method B: fixed width in mm) and the junction hole size.
3. Click **Generate Pattern** and wait for the pipeline to finish.
4. Switch to the **2D Unfold** tab to see the unrolled strips; set scale (mm per unit) and margin under **Develop**.
5. Export **DXF**, **SVG** or **CSV**. Use **Save JSON / Load JSON** to keep parameters.

For large meshes: **Export for Colab** → run the cell in [COLAB.md](COLAB.md) → **Import Colab result**.

## Development

```bash
npm install
npm run build:wasm   # needs Emscripten (emcc/emcmake), cmake and curl; fetches Eigen 3.4.0
npm run dev          # Vite dev server
npm run build        # tsc + vite build
npm run preview
```

Native CLI (used by the Colab workflow): `bash native/build_cli.sh` builds `native/build/wgf_cli`.

Manual quality harness: `npx tsx test/stripe-quality.ts`.

Tech: TypeScript, Vite, Three.js, Zustand, simplex-noise; C++ with Eigen, compiled with Emscripten.

```
src/core/       # TPMS, marching cubes, half-edge mesh, kagome strips, unfolding, WASM client
src/workers/    # Web Worker hosting the WASM pipeline
src/export/     # DXF / SVG
src/ui/         # sidebar, 3D / 2D viewports
native/         # C++ pipeline (header-based), CLI and WASM build scripts
```

## Credits

- Josh Vekhter, Jiacheng Zhuo, Luisa F. Gil Fandino, Qixing Huang, Etienne Vouga. "Weaving Geodesic Foliations." *ACM Transactions on Graphics* 38(4) (SIGGRAPH 2019).
- Felix Knöppel, Keenan Crane, Ulrich Pinkall, Peter Schröder. "Globally Optimal Direction Fields." *ACM TOG* 32(4), 2013; and "Stripe Patterns on Surfaces." *ACM TOG* 34(4), 2015.
- [Eigen](https://eigen.tuxfamily.org/) (downloaded at build time).

This is an independent student project and is not affiliated with the papers' authors.

## Related repos

- [smocking-cad](https://github.com/bob-takuya/smocking-cad) — smocking pattern editor and 3D preview
- [bamboo-gridshell](https://github.com/bob-takuya/bamboo-gridshell) — small bamboo gridshell sketch tool
- [rhinotools](https://github.com/bob-takuya/rhinotools) — RhinoPython scripts for CNC / laser part prep

## License

MIT — see [LICENSE](LICENSE). Eigen, downloaded at build time, is under its own license (MPL-2.0).

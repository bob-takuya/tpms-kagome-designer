# Running the WGF pipeline on Google Colab

The in-browser WASM build of the Weaving Geodesic Foliations pipeline
works, but on a typical laptop it is 10–30× slower than a native build
with SIMD + cache-friendly Eigen. For big meshes (TPMS at resolution
60+, `useCover=true`, and paper-strict λ schedules) this difference
means the difference between a 30-second run and a 10-minute run.

This document shows how to offload the heavy compute to a free Google
Colab instance and import the result back into the browser app. The
flow is:

1. **In the browser**: click **"Export for Colab"** in the sidebar.
   Downloads a `wgf-input.txt` file containing the mesh (`V`, `F`) and
   the current pipeline parameters.
2. **In Colab**: paste the single cell below, run it, upload
   `wgf-input.txt` when prompted, wait for the pipeline to finish,
   then download the emitted `wgf-output.txt`.
3. **In the browser**: click **"Import Colab result"**, pick the
   downloaded `wgf-output.txt`. The ribbon pattern appears and the 2D
   unfold / DXF / SVG exports become available as usual.

---

## Colab cell — paste this into a single code cell

```python
# Weaving Geodesic Foliations — native Colab runner.
#
# What this cell does:
#   1. Clones (or updates) the bob-takuya/tpms-kagome-designer repo.
#   2. Fetches Eigen 3.4.0 into native/third_party/eigen.
#   3. Builds the wgf_cli native binary with -O3 -ffast-math.
#   4. Prompts you to upload wgf-input.txt (exported from the browser).
#   5. Runs the pipeline: ./wgf_cli < input > output (streaming stderr
#      live to the Colab output so you can watch the stages advance).
#   6. Auto-downloads wgf-output.txt to your machine.

import os, subprocess

REPO = "https://github.com/bob-takuya/tpms-kagome-designer.git"
ROOT = "/content/tpms-kagome-designer"

# 1. Clone or pull the repo.
if not os.path.isdir(ROOT):
    subprocess.check_call(["git", "clone", "--depth", "1", REPO, ROOT])
else:
    subprocess.check_call(["git", "-C", ROOT, "pull", "--ff-only"])

# 2 + 3. Build the native CLI.
subprocess.check_call(["bash", f"{ROOT}/native/build_cli.sh"])

# 4. Upload the input file (the one exported from the browser).
from google.colab import files
print("\nUpload your wgf-input.txt (exported from the browser sidebar)")
uploaded = files.upload()
input_name = next(iter(uploaded))

# 5. Run the pipeline. We use Popen + PIPE for stderr instead of
#    stderr=sys.stderr, because Colab's IPython replaces sys.stderr with
#    an IOStream object that lacks a real fileno() — which causes
#    subprocess.run(..., stderr=sys.stderr) to raise UnsupportedOperation.
out_path = "/content/wgf-output.txt"
print(f"\nRunning wgf_cli on {input_name} ...\n")
with open(input_name, "rb") as fin, open(out_path, "wb") as fout:
    proc = subprocess.Popen(
        [f"{ROOT}/native/build/wgf_cli"],
        stdin=fin,
        stdout=fout,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert proc.stderr is not None
    for line in proc.stderr:
        print(line, end="", flush=True)
    rc = proc.wait()
if rc != 0:
    raise RuntimeError(f"wgf_cli exited with {rc}")

# 6. Auto-download the result.
print("\nDone. Downloading wgf-output.txt ...")
files.download(out_path)
```

That is the entire cell. Paste, run, upload, wait, download.

---

## Text formats (for reference)

The CLI reads / writes plain text so both Python and TypeScript can
trivially parse it without pulling in a JSON library.

### Input (`wgf-input.txt`)

```
# wgf-input v1
V <nV>
<x> <y> <z>
<x> <y> <z>
...
F <nF>
<a> <b> <c>
<a> <b> <c>
...
OPTS lambdaInit=1000 lambdaMin=1e-3 alg1MaxIter=50 mu=1e-4 \
     jointIters=10 userScale=1 useCover=1
END
```

### Output (`wgf-output.txt`)

```
# wgf-output v1
META initialCurl=<v> finalCurl=<v> iterations=<v> numSingular=<v> \
     segments=<v> components=<v> fam0=<v> fam1=<v> fam2=<v>
SEG <nSeg>
<ax> <ay> <az> <bx> <by> <bz> <family> <baseFaceIdx>
...
END
```

`family ∈ {0, 1, 2}` — the kagome family (three directed lines per
triaxial weave), `baseFaceIdx` is the index into the input `F` array
that the segment lives on.

---

## Notes

- The CLI defaults to paper-strict parameters (`lambdaInit=1000`,
  `lambdaMin=1e-3`, `alg1MaxIter=50`, `jointIters=10`). Override any
  of them via the `OPTS` line in the input file; the browser export
  writes whatever is currently in the sidebar.
- Expect a few seconds of compile time on the first run and ~1 second
  thereafter (the repo is cached in `/content/`). Reuse the same
  Colab runtime if you want to skip the clone + build.
- A Colab CPU runtime is plenty — you do not need a GPU or TPU.

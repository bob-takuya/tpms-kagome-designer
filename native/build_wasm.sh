#!/usr/bin/env bash
#
# Build the Weaving Geodesic Foliations WASM module.
#
# Prerequisites:
#   - emsdk activated (emcmake, emmake, emcc on PATH)
#   - cmake, curl, tar
#
# Fetches Eigen 3.4.0 on demand into native/third_party/eigen if missing.
# Leaves the resulting wgf.js + wgf.wasm in public/wasm/.

set -euo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$HERE"

if [ ! -d third_party/eigen/Eigen ]; then
    echo "[wgf] fetching Eigen 3.4.0..."
    mkdir -p third_party
    curl -sSL -o /tmp/eigen.tar.gz \
        https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    tar -xzf /tmp/eigen.tar.gz -C third_party
    rm -f /tmp/eigen.tar.gz
    rm -rf third_party/eigen
    mv third_party/eigen-3.4.0 third_party/eigen
fi

mkdir -p build-wasm
cd build-wasm
emcmake cmake -DCMAKE_BUILD_TYPE=Release ..
emmake make wgf_wasm -j2

echo "[wgf] built:"
ls -lh "$HERE/../src/wasm/" || true

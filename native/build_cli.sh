#!/usr/bin/env bash
#
# Build the native Weaving Geodesic Foliations CLI (wgf_cli).
#
# Prerequisites:
#   - cmake, make, g++/clang++ (C++17)
#   - curl, tar (for fetching Eigen 3.4.0)
#
# Fetches Eigen on demand into native/third_party/eigen if missing.
# Produces the binary at native/build/wgf_cli.

set -euo pipefail

HERE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$HERE"

if [ ! -d third_party/eigen/Eigen ]; then
    echo "[wgf-cli] fetching Eigen 3.4.0..."
    mkdir -p third_party
    curl -sSL -o /tmp/eigen.tar.gz \
        https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    tar -xzf /tmp/eigen.tar.gz -C third_party
    rm -f /tmp/eigen.tar.gz
    rm -rf third_party/eigen
    mv third_party/eigen-3.4.0 third_party/eigen
fi

mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. > /dev/null
make wgf_cli -j"$(nproc 2>/dev/null || echo 2)"

echo "[wgf-cli] built:"
ls -lh "$HERE/build/wgf_cli"

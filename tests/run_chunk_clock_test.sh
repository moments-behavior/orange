#!/usr/bin/env bash
# Build and run the synthetic ChunkClock test (no cameras or GPU needed).
set -euo pipefail
cd "$(dirname "$0")/.."
out=${TMPDIR:-/tmp}/chunk_clock_test
CUDA_INC=${CUDA_INC:-/usr/local/cuda/include}
g++ -O2 -std=c++17 -I src -I /opt/EVT/eSDK/include -I "$CUDA_INC" \
    tests/chunk_clock_test.cpp src/recording_chunks.cpp -lpthread -o "$out"
"$out"

#!/usr/bin/env bash
# Example 3: 3-D coupled air-ground — static propeller source
set -euo pipefail

USE_CUDA=${USE_CUDA:-true}
CUDA_FLAG=""
[ "$USE_CUDA" = true ] && CUDA_FLAG="--use-cuda"

python3 examples/run_coupled_static_3d.py \
    --x-min 0 --x-max 100 \
    --y-min 0 --y-max 100 \
    --z-min -5 --z-max 15 \
    --ground-vp 500 --ground-vs 250 --ground-density 1800 \
    --ground-qp 20 --ground-qs 10 \
    --source-x 50 --source-y 50 --source-z 5 \
    --source-signal propeller \
    --array-cx 50 --array-cy 50 --array-radius 2 --array-count 16 \
    --mic-z 1.5 --geo-z -0.05 \
    --total-time 0.3 \
    --fd-order 4 \
    --damping-width 10 --damping-max 0.05 \
    --snapshot-interval 200 \
    --output-dir output/coupled_static_3d \
    $CUDA_FLAG

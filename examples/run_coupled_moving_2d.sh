#!/usr/bin/env bash
# Example 2: 2-D coupled air-ground — moving propeller source
set -euo pipefail

USE_CUDA=${USE_CUDA:-false}
CUDA_FLAG=""
[ "$USE_CUDA" = true ] && CUDA_FLAG="--use-cuda"

python3 examples/run_coupled_moving_2d.py \
    --x-min -100 --x-max 100 \
    --z-min -5 --z-max 15 \
    --ground-vp 500 --ground-vs 250 --ground-density 1800 \
    --ground-qp 20 --ground-qs 10 \
    --source-x0 -80 --source-x1 80 --source-z 5 --source-speed 15 \
    --source-signal propeller \
    --array-cx 0 --array-radius 2 --array-count 16 \
    --mic-z 1.5 --geo-z -0.05 \
    --total-time 0.5 \
    --fd-order 4 \
    --damping-width 20 --damping-max 0.05 \
    --snapshot-interval 200 \
    --output-dir output/coupled_moving_2d \
    $CUDA_FLAG

#!/usr/bin/env bash
# Example 1: 2-D coupled air-ground simulation — static propeller source
#
# A propeller source sits 5 m above a flat ground surface (z=0).
# The elastic FDTD solver handles both the air (acoustic) and ground
# (elastic) domains in a single unified simulation.
set -euo pipefail

# Domain ----------------------------------------------------------------
X_MIN=-100.0
X_MAX=100.0
Z_MIN=-5.0
Z_MAX=15.0

# Ground material (soft soil / chernozem) --------------------------------
GROUND_VP=500.0
GROUND_VS=250.0
GROUND_DENSITY=1800.0
GROUND_QP=20.0
GROUND_QS=10.0

# Source -----------------------------------------------------------------
SOURCE_X=50.0
SOURCE_Z=5.0
SOURCE_SIGNAL=propeller

# Receiver array ---------------------------------------------------------
ARRAY_CX=0.0
ARRAY_RADIUS=2.0
ARRAY_COUNT=16
MIC_Z=1.5           # ear height
GEO_Z=-0.05         # just below ground surface

# Simulation -------------------------------------------------------------
TOTAL_TIME=0.5       # reduced for quick demo; use 3.0 for full run
FD_ORDER=4
DAMPING_WIDTH=20
DAMPING_MAX=0.05
SNAPSHOT_INTERVAL=200
SOURCE_AMPLITUDE=1.0

# Parallelism ------------------------------------------------------------
USE_CUDA=${USE_CUDA:-false}
MPI_RANKS=0

# Output -----------------------------------------------------------------
OUTPUT_DIR=output/coupled_static_2d

# -----------------------------------------------------------------------
CUDA_FLAG=""
if [ "$USE_CUDA" = true ]; then
    CUDA_FLAG="--use-cuda"
fi

CMD="python3 examples/run_coupled_static_2d.py \
    --x-min $X_MIN --x-max $X_MAX \
    --z-min $Z_MIN --z-max $Z_MAX \
    --ground-vp $GROUND_VP --ground-vs $GROUND_VS \
    --ground-density $GROUND_DENSITY \
    --ground-qp $GROUND_QP --ground-qs $GROUND_QS \
    --source-x $SOURCE_X --source-z $SOURCE_Z \
    --source-signal $SOURCE_SIGNAL \
    --array-cx $ARRAY_CX --array-radius $ARRAY_RADIUS \
    --array-count $ARRAY_COUNT \
    --mic-z $MIC_Z --geo-z $GEO_Z \
    --total-time $TOTAL_TIME \
    --fd-order $FD_ORDER \
    --damping-width $DAMPING_WIDTH \
    --damping-max $DAMPING_MAX \
    --snapshot-interval $SNAPSHOT_INTERVAL \
    --source-amplitude $SOURCE_AMPLITUDE \
    --output-dir $OUTPUT_DIR \
    $CUDA_FLAG"

if [ "$MPI_RANKS" -gt 1 ] 2>/dev/null; then
    mpirun --allow-run-as-root -n "$MPI_RANKS" bash -c "$CMD"
else
    eval "$CMD"
fi

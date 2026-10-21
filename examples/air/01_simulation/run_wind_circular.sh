#!/usr/bin/env bash
# Wind domain — moving propeller source, circular receiver array
set -euo pipefail

# Domain ----------------------------------------------------------------
DOMAIN=wind
DX=0.18                # resolves BPF=180 Hz at 343 m/s (10 ppw)
X_MIN=-50.0
X_MAX=50.0
Y_MIN=-50.0
Y_MAX=50.0
WIND_SPEED=5.0
WIND_DIR=45.0          # degrees

# Source ----------------------------------------------------------------
SOURCE_TYPE=moving
SOURCE_SIGNAL=propeller
SOURCE_X=-45.0
SOURCE_Y=0.0
SOURCE_X1=45.0
SOURCE_Y1=0.0
SOURCE_SPEED=50.0
BLADE_COUNT=3
RPM=3600
HARMONICS=14

# Receiver array --------------------------------------------------------
ARRAY=circular
RECEIVER_COUNT=16
RECEIVER_RADIUS=2.0    # tightened from 15 m

# Simulation ------------------------------------------------------------
TOTAL_TIME=3.0
SNAPSHOT_INTERVAL=50
DAMPING_WIDTH=40
DAMPING_MAX=0.15
SOURCE_AMPLITUDE=1.0    # Pa — peak source pressure
AIR_ABSORPTION=0.005    # background damping

# Parallelism -----------------------------------------------------------
USE_CUDA=true           # set to false for CPU-only
MPI_RANKS=0             # 0 = no MPI; >1 = mpirun -n $MPI_RANKS
FD_ORDER=2              # spatial FD accuracy order (2, 4, 6, …)

# Output ----------------------------------------------------------------
OUTPUT_DIR=output/my_test

# -----------------------------------------------------------------------
CUDA_FLAG=""                # the FDTD runs on the GPU by default
if [ "$USE_CUDA" != true ]; then
    CUDA_FLAG="--cpu"
fi

CMD="python examples/air/01_simulation/run_fdtd.py \
    --domain \"$DOMAIN\" \
    --dx \"$DX\" \
    --x-min \"$X_MIN\" --x-max \"$X_MAX\" \
    --y-min \"$Y_MIN\" --y-max \"$Y_MAX\" \
    --wind-speed \"$WIND_SPEED\" --wind-dir \"$WIND_DIR\" \
    --source-type \"$SOURCE_TYPE\" \
    --source-signal \"$SOURCE_SIGNAL\" \
    --source-x \"$SOURCE_X\" --source-y \"$SOURCE_Y\" \
    --source-x1 \"$SOURCE_X1\" --source-y1 \"$SOURCE_Y1\" \
    --source-speed \"$SOURCE_SPEED\" \
    --blade-count \"$BLADE_COUNT\" --rpm \"$RPM\" --harmonics \"$HARMONICS\" \
    --array \"$ARRAY\" \
    --receiver-count \"$RECEIVER_COUNT\" \
    --receiver-radius \"$RECEIVER_RADIUS\" \
    --total-time \"$TOTAL_TIME\" \
    --snapshot-interval \"$SNAPSHOT_INTERVAL\" \
    --damping-width \"$DAMPING_WIDTH\" \
    --damping-max \"$DAMPING_MAX\" \
    --source-amplitude \"$SOURCE_AMPLITUDE\" \
    --air-absorption \"$AIR_ABSORPTION\" \
    --output-dir \"$OUTPUT_DIR\" \
    --fd-order \"$FD_ORDER\" \
    $CUDA_FLAG"

if [ "$MPI_RANKS" -gt 1 ] 2>/dev/null; then
    mpirun --allow-run-as-root -n "$MPI_RANKS" bash -c "$CMD"
else
    eval "$CMD"
fi

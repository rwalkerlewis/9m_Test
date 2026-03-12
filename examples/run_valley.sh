#!/usr/bin/env bash
# Valley domain — moving propeller source, circular receiver array
set -euo pipefail

# Domain ----------------------------------------------------------------
DOMAIN=hills_vegetation
DX=0.18                # resolves BPF=180 Hz at 343 m/s (10 ppw)
X_MIN=-50.0
X_MAX=50.0
Y_MIN=-50.0
Y_MAX=50.0
WIND_SPEED=0.0
WIND_DIR=0.0           # degrees
DIRT_VELOCITY=1500.0

# Source — arc path through the valley, peaking above the array --------
SOURCE_TYPE=moving
SOURCE_SIGNAL=propeller
SOURCE_X=-40.0
SOURCE_Y=0.0
SOURCE_X1=40.0
SOURCE_Y1=0.0
SOURCE_SPEED=50.0
SOURCE_ARC_HEIGHT=15.0  # parabolic arc peaking at y=15 m (above array)
BLADE_COUNT=3
RPM=3600
HARMONICS=14

# Receiver array --------------------------------------------------------
ARRAY=circular
RECEIVER_COUNT=16
RECEIVER_RADIUS=2.0
RECEIVER_CX=0.0
RECEIVER_CY=10.0        # shifted +10 m in y (toward north ridge)

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
OUTPUT_DIR=output/valley_test

# -----------------------------------------------------------------------
CUDA_FLAG=""
if [ "$USE_CUDA" = true ]; then
    CUDA_FLAG="--use-cuda"
fi

CMD="python examples/run_fdtd.py \
    --domain \"$DOMAIN\" \
    --dx \"$DX\" \
    --x-min \"$X_MIN\" --x-max \"$X_MAX\" \
    --y-min \"$Y_MIN\" --y-max \"$Y_MAX\" \
    --wind-speed \"$WIND_SPEED\" --wind-dir \"$WIND_DIR\" \
    --dirt-velocity \"$DIRT_VELOCITY\" \
    --source-type \"$SOURCE_TYPE\" \
    --source-signal \"$SOURCE_SIGNAL\" \
    --source-x \"$SOURCE_X\" --source-y \"$SOURCE_Y\" \
    --source-x1 \"$SOURCE_X1\" --source-y1 \"$SOURCE_Y1\" \
    --source-speed \"$SOURCE_SPEED\" \
    --source-arc-height \"$SOURCE_ARC_HEIGHT\" \
    --blade-count \"$BLADE_COUNT\" --rpm \"$RPM\" --harmonics \"$HARMONICS\" \
    --array \"$ARRAY\" \
    --receiver-count \"$RECEIVER_COUNT\" \
    --receiver-radius \"$RECEIVER_RADIUS\" \
    --receiver-cx \"$RECEIVER_CX\" --receiver-cy \"$RECEIVER_CY\" \
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

# Run detection evaluation if requested
RUN_DETECTION=${RUN_DETECTION:-true}
if [ "$RUN_DETECTION" = true ]; then
    echo ""
    echo "Running detection pipeline evaluation..."
    python examples/run_full_pipeline.py "$OUTPUT_DIR" --source-speed "$SOURCE_SPEED"
fi

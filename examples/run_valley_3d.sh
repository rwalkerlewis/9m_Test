#!/usr/bin/env bash
# 3-D Valley domain — moving propeller source, circular receiver array
#
# This is the 3-D counterpart of run_valley.sh.  The valley terrain
# (two ridges with vegetation) is extruded into three dimensions so
# the source flies at a real altitude and the FDTD wave equation is
# solved on a volumetric grid.
#
# Grid spacing is coarser than the 2-D run (dx=1.0 vs 0.18) to keep
# memory usage practical.  Adjust DX, domain bounds, and TOTAL_TIME
# to trade off resolution vs. compute cost.
set -euo pipefail

# Domain ----------------------------------------------------------------
DOMAIN=hills_vegetation
DX=0.25                # f_max ≈ 274 Hz at 5 PPW, ~1.1 GB memory
X_MIN=-50.0
X_MAX=50.0
Y_MIN=-50.0
Y_MAX=50.0
Z_MIN=-5.0             # small sub-surface margin
Z_MAX=50.0             # altitude ceiling
WIND_SPEED=0.0
WIND_DIR=0.0           # degrees
DIRT_VELOCITY=1500.0

# Source — arc path through the valley at altitude ---------------------
SOURCE_TYPE=moving
SOURCE_SIGNAL=propeller
SOURCE_X=-40.0
SOURCE_Y=0.0
SOURCE_Z=15.0          # start altitude [m]
SOURCE_X1=40.0
SOURCE_Y1=0.0
SOURCE_Z1=15.0         # end altitude [m]
SOURCE_SPEED=50.0
SOURCE_ARC_HEIGHT=10.0 # parabolic arc in z (vertical), peaks at midpoint
BLADE_COUNT=3
RPM=3600
HARMONICS=14

# Receiver array --------------------------------------------------------
ARRAY=circular
RECEIVER_COUNT=16
RECEIVER_RADIUS=2.0
RECEIVER_CX=-5.0       # shifted -5 m in x
RECEIVER_CY=7.0        # shifted -3 m in y (from 10 to 7)
RECEIVER_CZ=5.5        # above terrain at (-5,7) (terrain height ~4.5m)

# Simulation ------------------------------------------------------------
TOTAL_TIME=1.0         # shorter than 2-D (3.0 s) to keep runtime down
SNAPSHOT_INTERVAL=100
DAMPING_WIDTH=40
DAMPING_MAX=0.15
SOURCE_AMPLITUDE=1.0   # Pa — peak source pressure
AIR_ABSORPTION=0.005   # background damping

# Parallelism -----------------------------------------------------------
USE_CUDA=true           # GPU acceleration via CuPy
MPI_RANKS=0            # 0 = no MPI; >1 = mpirun -n $MPI_RANKS
FD_ORDER=8             # spatial FD accuracy order (2, 4, 6, …)

# Field plane (decoupled array placement) --------------------------------
FIELD_PLANE_Z=5.5       # altitude of horizontal slice to save [m]
FIELD_PLANE_SUB=4       # spatial subsampling (4 → 1.0 m with dx=0.25)

# Output ----------------------------------------------------------------
OUTPUT_DIR=output/valley_3d_test

# -----------------------------------------------------------------------
CUDA_FLAG=""
if [ "$USE_CUDA" = true ]; then
    CUDA_FLAG="--use-cuda"
fi

CMD="python3 examples/run_fdtd_3d.py \
    --domain \"$DOMAIN\" \
    --dx \"$DX\" \
    --x-min \"$X_MIN\" --x-max \"$X_MAX\" \
    --y-min \"$Y_MIN\" --y-max \"$Y_MAX\" \
    --z-min \"$Z_MIN\" --z-max \"$Z_MAX\" \
    --wind-speed \"$WIND_SPEED\" --wind-dir \"$WIND_DIR\" \
    --dirt-velocity \"$DIRT_VELOCITY\" \
    --source-type \"$SOURCE_TYPE\" \
    --source-signal \"$SOURCE_SIGNAL\" \
    --source-x \"$SOURCE_X\" --source-y \"$SOURCE_Y\" --source-z \"$SOURCE_Z\" \
    --source-x1 \"$SOURCE_X1\" --source-y1 \"$SOURCE_Y1\" --source-z1 \"$SOURCE_Z1\" \
    --source-speed \"$SOURCE_SPEED\" \
    --source-arc-height \"$SOURCE_ARC_HEIGHT\" \
    --blade-count \"$BLADE_COUNT\" --rpm \"$RPM\" --harmonics \"$HARMONICS\" \
    --array \"$ARRAY\" \
    --receiver-count \"$RECEIVER_COUNT\" \
    --receiver-radius \"$RECEIVER_RADIUS\" \
    --receiver-cx \"$RECEIVER_CX\" --receiver-cy \"$RECEIVER_CY\" \
    --receiver-cz \"$RECEIVER_CZ\" \
    --total-time \"$TOTAL_TIME\" \
    --snapshot-interval \"$SNAPSHOT_INTERVAL\" \
    --damping-width \"$DAMPING_WIDTH\" \
    --damping-max \"$DAMPING_MAX\" \
    --source-amplitude \"$SOURCE_AMPLITUDE\" \
    --air-absorption \"$AIR_ABSORPTION\" \
    --output-dir \"$OUTPUT_DIR\" \
    --fd-order \"$FD_ORDER\" \    --field-plane-z "$FIELD_PLANE_Z" \
    --field-plane-subsample "$FIELD_PLANE_SUB" \    $CUDA_FLAG"

if [ "$MPI_RANKS" -gt 1 ] 2>/dev/null; then
    mpirun --allow-run-as-root -n "$MPI_RANKS" bash -c "$CMD"
else
    eval "$CMD"
fi

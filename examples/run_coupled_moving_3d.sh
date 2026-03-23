#!/usr/bin/env bash
# 3-D Coupled air–ground — moving propeller source
#
# Coupled elastic FDTD with a propeller source flying an arc path
# through the domain.  Microphones sit on a circular array; geophones
# are placed on a slightly wider circle aligned with the eight
# cardinal / inter-cardinal directions (N, NE, E, …).
#
# The source trajectory matches the valley-3D example so results can
# be compared directly.
set -euo pipefail

# Domain ----------------------------------------------------------------
DX=0.25
X_MIN=-50.0
X_MAX=10.0
Y_MIN=-5.0
Y_MAX=15.0
Z_MIN=-5.0             # sub-surface margin
Z_MAX=15.0             # altitude ceiling

# Ground properties -----------------------------------------------------
# Saturated sand / stiff soil.  Vp=1500 gives a 58 ms head-wave
# advantage over direct air-path (c_air=343) at 35 m range.
# The impedance ratio Z_ground/Z_air ≈ 6400:1 is realistic —
# only 0.03% of particle velocity transmits.  The source amplitude
# is set high enough that the transmitted vz is still detectable.
GROUND_VP=1500.0
GROUND_VS=250.0
GROUND_DENSITY=1800.0
GROUND_QP=20.0
GROUND_QS=10.0

# Source — arc path through domain at altitude --------------------------
SOURCE_SIGNAL=propeller
SOURCE_X0=-40.0
SOURCE_Y0=0.0
SOURCE_Z0=8.0          # start altitude [m]
SOURCE_X1=40.0
SOURCE_Y1=0.0
SOURCE_Z1=5.0          # end altitude [m]
SOURCE_SPEED=50.0
SOURCE_ARC_HEIGHT=10.0 # parabolic arc in y, peaks at midpoint
BLADE_COUNT=3
RPM=3600
HARMONICS=14

# Microphones — 10 on a 2 m circle -------------------------------------
ARRAY_CX=-5.0          # shifted −5 m in x
ARRAY_CY=7.0           # shifted to y = 7 m
MIC_RADIUS=2.0
MIC_COUNT=10
MIC_Z=5.5              # above terrain

# Geophones — 8 at cardinal directions on a 3 m circle -----------------
GEO_RADIUS=3.0
GEO_COUNT=8
GEO_Z=-0.05            # just below surface

# Simulation ------------------------------------------------------------
TOTAL_TIME=0.3
FD_ORDER=4
CFL_SAFETY=0.8
DAMPING_WIDTH=10
DAMPING_MAX=0.15
SNAPSHOT_INTERVAL=100
SOURCE_AMPLITUDE=500.0        # [Pa·m³/s] moment-rate source (~81 dB SPL at 1 m)

# Parallelism -----------------------------------------------------------
USE_CUDA=true

# Field plane (decoupled array placement) --------------------------------
FIELD_PLANE_Z=5.5       # altitude of horizontal slice to save [m]
FIELD_PLANE_SUB=4       # spatial subsampling

# Output ----------------------------------------------------------------
OUTPUT_DIR=output/coupled_moving_3d

# -----------------------------------------------------------------------
CUDA_FLAG=""
if [ "$USE_CUDA" = true ]; then
    CUDA_FLAG="--use-cuda"
fi

python3 examples/run_coupled_moving_3d.py \
    --dx "$DX" \
    --x-min "$X_MIN" --x-max "$X_MAX" \
    --y-min "$Y_MIN" --y-max "$Y_MAX" \
    --z-min "$Z_MIN" --z-max "$Z_MAX" \
    --ground-vp "$GROUND_VP" --ground-vs "$GROUND_VS" \
    --ground-density "$GROUND_DENSITY" \
    --ground-qp "$GROUND_QP" --ground-qs "$GROUND_QS" \
    --source-signal "$SOURCE_SIGNAL" \
    --source-x0 "$SOURCE_X0" --source-y0 "$SOURCE_Y0" \
    --source-z0 "$SOURCE_Z0" \
    --source-x1 "$SOURCE_X1" --source-y1 "$SOURCE_Y1" \
    --source-z1 "$SOURCE_Z1" \
    --source-speed "$SOURCE_SPEED" \
    --source-arc-height "$SOURCE_ARC_HEIGHT" \
    --blade-count "$BLADE_COUNT" --rpm "$RPM" --harmonics "$HARMONICS" \
    --array-cx "$ARRAY_CX" --array-cy "$ARRAY_CY" \
    --mic-radius "$MIC_RADIUS" --mic-count "$MIC_COUNT" \
    --mic-z "$MIC_Z" \
    --geo-radius "$GEO_RADIUS" --geo-count "$GEO_COUNT" \
    --geo-z "$GEO_Z" \
    --total-time "$TOTAL_TIME" \
    --fd-order "$FD_ORDER" \
    --cfl-safety "$CFL_SAFETY" \
    --damping-width "$DAMPING_WIDTH" \
    --damping-max "$DAMPING_MAX" \
    --snapshot-interval "$SNAPSHOT_INTERVAL" \
    --source-amplitude "$SOURCE_AMPLITUDE" \
    --field-plane-z "$FIELD_PLANE_Z" \
    --field-plane-subsample "$FIELD_PLANE_SUB" \
    --output-dir "$OUTPUT_DIR" \
    $CUDA_FLAG

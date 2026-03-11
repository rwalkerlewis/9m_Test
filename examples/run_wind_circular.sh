#!/usr/bin/env bash
# Wind domain — moving propeller source, circular receiver array
set -euo pipefail

# MPI -------------------------------------------------------------------
NP=4

# Domain ----------------------------------------------------------------
DOMAIN=wind
VELOCITY=343.0
DX=0.18                # resolves BPF=180 Hz at 343 m/s (10 ppw)
X_MIN=-50.0
X_MAX=50.0
Y_MIN=-50.0
Y_MAX=50.0
WIND_SPEED=15.0
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
RECEIVER_RADIUS=3.0    # tightened from 15 m

# Simulation ------------------------------------------------------------
TOTAL_TIME=2.0
SNAPSHOT_INTERVAL=50

# Output ----------------------------------------------------------------
OUTPUT_DIR=output/my_test

# -----------------------------------------------------------------------
mpirun --oversubscribe -np "$NP" python examples/run_fdtd.py \
    --domain "$DOMAIN" \
    --velocity "$VELOCITY" \
    --dx "$DX" \
    --x-min "$X_MIN" --x-max "$X_MAX" \
    --y-min "$Y_MIN" --y-max "$Y_MAX" \
    --wind-speed "$WIND_SPEED" --wind-dir "$WIND_DIR" \
    --source-type "$SOURCE_TYPE" \
    --source-signal "$SOURCE_SIGNAL" \
    --source-x "$SOURCE_X" --source-y "$SOURCE_Y" \
    --source-x1 "$SOURCE_X1" --source-y1 "$SOURCE_Y1" \
    --source-speed "$SOURCE_SPEED" \
    --blade-count "$BLADE_COUNT" --rpm "$RPM" --harmonics "$HARMONICS" \
    --array "$ARRAY" \
    --receiver-count "$RECEIVER_COUNT" \
    --receiver-radius "$RECEIVER_RADIUS" \
    --total-time "$TOTAL_TIME" \
    --snapshot-interval "$SNAPSHOT_INTERVAL" \
    --output-dir "$OUTPUT_DIR"

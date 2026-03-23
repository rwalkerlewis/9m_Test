#!/usr/bin/env bash
# Pipeline run — erratic_quadcopter (3-D isotropic domain, erratic trajectory)
set -euo pipefail

SIM_DIR=output/erratic_quadcopter
SOURCE_SPEED=8.0        # mean speed from metadata
CONFIG=examples/pipeline.config.json

# ML flags (set to true to enable) -------------------------------------
ENABLE_CLASSIFICATION=${ENABLE_CLASSIFICATION:-false}
ENABLE_MANEUVER=${ENABLE_MANEUVER:-false}
ENABLE_FUSION=${ENABLE_FUSION:-false}
ENABLE_ANOMALY=${ENABLE_ANOMALY:-false}

# Build optional ML flags -----------------------------------------------
ML_FLAGS=""
if [ "$ENABLE_CLASSIFICATION" = true ]; then ML_FLAGS="$ML_FLAGS --enable-classification"; fi
if [ "$ENABLE_MANEUVER" = true ];       then ML_FLAGS="$ML_FLAGS --enable-maneuver"; fi
if [ "$ENABLE_FUSION" = true ];         then ML_FLAGS="$ML_FLAGS --enable-fusion"; fi
if [ "$ENABLE_ANOMALY" = true ];        then ML_FLAGS="$ML_FLAGS --enable-anomaly"; fi

echo "Running detection pipeline on $SIM_DIR ..."
python examples/run_pipeline.py "$SIM_DIR" \
    --config "$CONFIG" \
    --source-speed "$SOURCE_SPEED" \
    $ML_FLAGS

#!/usr/bin/env bash
# All walkthrough figures for one recording: ./examples/walkthrough/run_all.sh multistatic --cuda
set -euo pipefail
cd "$(dirname "$0")"
ds="${1:-multistatic}"; shift || true
for s in 01_recording 02_passive_map 03_tracks 04_ais_crosscheck 05_direct_path_removal 06_echo_map 07_harbor_picture; do
    if [[ "$ds" == harbor && ( "$s" == 05_* || "$s" == 06_* ) ]]; then continue; fi
    python "$s.py" --dataset "$ds" "$@"
done

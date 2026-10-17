#!/usr/bin/env bash
# Development-container helper (Linux, macOS, WSL).  Needs only Docker with the compose plugin.
#
#   ./dev.sh shell            open a shell in the dev container (builds / starts it if needed)
#   ./dev.sh test [args]      run pytest in the container (default: tests/harbor)
#   ./dev.sh run CMD...       run any command in the container
#   ./dev.sh build [--no-cache]   (re)build the image
#   ./dev.sh status           show GPU detection and container state
#   ./dev.sh down             stop and remove the container
#
# Environment:  USE_GPU=auto|1|0 (default auto), EXTRAS=cuda,dev[,ml], BASE_IMAGE=... (see Dockerfile)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export HOST_UID="${HOST_UID:-$(id -u)}"
export HOST_GID="${HOST_GID:-$(id -g)}"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker not found; install Docker first (https://docs.docker.com/get-docker/)" >&2
    exit 1
fi

gpu_usable() {
    docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia \
        && command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

use_gpu=0
case "${USE_GPU:-auto}" in
    1) use_gpu=1 ;;
    0) use_gpu=0 ;;
    auto) if gpu_usable; then use_gpu=1; fi ;;
    *) echo "USE_GPU must be auto, 1 or 0" >&2; exit 1 ;;
esac

files=(-f docker-compose.yml)
if [[ $use_gpu == 1 ]]; then files+=(-f docker-compose.gpu.yml); fi
dc() { docker compose "${files[@]}" "$@"; }

ensure_up() {
    if [[ -z "$(dc ps --status running -q dev 2>/dev/null)" ]]; then
        dc up -d --build dev
    fi
}

tty_flag=()
if [[ ! -t 0 || ! -t 1 ]]; then tty_flag=(-T); fi

cmd="${1:-shell}"
shift || true
case "$cmd" in
    build)  dc build "$@" dev ;;
    up)     dc up -d --build dev ;;
    shell)  ensure_up; dc exec "${tty_flag[@]}" dev bash ;;
    run)    ensure_up; dc exec "${tty_flag[@]}" dev "$@" ;;
    test)   ensure_up; dc exec "${tty_flag[@]}" dev pytest "${@:-tests/harbor}" ;;
    down)   dc down ;;
    status)
        echo "GPU passthrough: $([[ $use_gpu == 1 ]] && echo enabled || echo disabled) (USE_GPU=${USE_GPU:-auto})"
        echo "Host user: $HOST_UID:$HOST_GID"
        dc ps
        ;;
    *) echo "unknown command: $cmd (see the header of dev.sh)" >&2; exit 1 ;;
esac

#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG="legged_gym.envs.base.two_leg_stand_config"
DEFAULT_TASK="lite3_two_leg_stand"
DEFAULT_DEVICE="cuda:0"

usage() {
    cat <<EOF
Usage: $(basename "$0") -s <sweep-id> [-n <runs>] [--headless] [--rl-device DEV] [--sim-device DEV]

Defaults:
  config      ${DEFAULT_CONFIG}
  task        ${DEFAULT_TASK}
  rl-device   ${DEFAULT_DEVICE}
  sim-device  ${DEFAULT_DEVICE}
  physics     physicsX

Additional arguments are passed through to run_agent.py if supplied after "--".
EOF
}

SWEEP_ID=""
NUM_RUNS=1
HEADLESS=0
RL_DEVICE="${DEFAULT_DEVICE}"
SIM_DEVICE="${DEFAULT_DEVICE}"
PHYSICS_ENGINE="physicsX"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--sweep-id)
            SWEEP_ID="$2"
            shift 2
            ;;
        -n|--num-runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        --headless)
            HEADLESS=1
            shift
            ;;
        --rl-device)
            RL_DEVICE="$2"
            shift 2
            ;;
        --sim-device)
            SIM_DEVICE="$2"
            shift 2
            ;;
        --physics-engine)
            PHYSICS_ENGINE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "${SWEEP_ID}" ]]; then
    echo "Error: sweep-id is required." >&2
    usage
    exit 1
fi

PYTHON_BIN=""
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
    PYTHON_BIN="$(command -v python)"
fi

if [[ -z "${PYTHON_BIN}" ]]; then
    echo "Error: could not locate a python executable." >&2
    exit 1
fi

CMD=("${PYTHON_BIN}" "${SCRIPT_DIR}/run_agent.py"
    --sweep-id "${SWEEP_ID}"
    --config "${DEFAULT_CONFIG}"
    --task "${DEFAULT_TASK}"
    --rl-device "${RL_DEVICE}"
    --sim-device "${SIM_DEVICE}"
    --physics-engine "${PHYSICS_ENGINE}"
    --num-runs "${NUM_RUNS}"
)

if [[ ${HEADLESS} -eq 1 ]]; then
    CMD+=(--headless)
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

echo "Using python: ${PYTHON_BIN}"
echo "Running: ${CMD[*]}"
"${CMD[@]}"

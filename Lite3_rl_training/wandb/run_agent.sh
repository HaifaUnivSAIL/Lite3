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

Authentication:
  --wandb-api-key KEY         Explicit W&B API key to use for non-interactive login.
  --wandb-api-key-file FILE   Read the W&B API key from FILE.
                              If neither flag is supplied, the script falls back to the
                              WANDB_API_KEY environment variable or any existing CLI login.
EOF
}

SWEEP_ID=""
NUM_RUNS=1
HEADLESS=0
RL_DEVICE="${DEFAULT_DEVICE}"
SIM_DEVICE="${DEFAULT_DEVICE}"
PHYSICS_ENGINE="physicsX"
EXTRA_ARGS=()
WANDB_API_KEY_OVERRIDE=""
WANDB_API_KEY_FILE=""

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
        --wandb-api-key)
            WANDB_API_KEY_OVERRIDE="$2"
            shift 2
            ;;
        --wandb-api-key-file)
            WANDB_API_KEY_FILE="$2"
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

RESOLVED_WANDB_API_KEY=""
if [[ -n "${WANDB_API_KEY_OVERRIDE}" ]]; then
    RESOLVED_WANDB_API_KEY="${WANDB_API_KEY_OVERRIDE}"
elif [[ -n "${WANDB_API_KEY_FILE}" ]]; then
    if [[ ! -f "${WANDB_API_KEY_FILE}" ]]; then
        echo "Error: W&B API key file not found: ${WANDB_API_KEY_FILE}" >&2
        exit 1
    fi
    RESOLVED_WANDB_API_KEY="$(<"${WANDB_API_KEY_FILE}")"
elif [[ -n "${WANDB_API_KEY:-}" ]]; then
    RESOLVED_WANDB_API_KEY="${WANDB_API_KEY}"
fi

if [[ -n "${RESOLVED_WANDB_API_KEY}" ]]; then
    # strip common newline characters to avoid accidental breaks
    RESOLVED_WANDB_API_KEY="${RESOLVED_WANDB_API_KEY//$'\r'/}"
    RESOLVED_WANDB_API_KEY="${RESOLVED_WANDB_API_KEY//$'\n'/}"
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

if [[ -n "${RESOLVED_WANDB_API_KEY}" ]]; then
    echo "Logging into W&B with provided API key."
    export WANDB_API_KEY="${RESOLVED_WANDB_API_KEY}"
    if ! "${PYTHON_BIN}" -m wandb login --relogin "${RESOLVED_WANDB_API_KEY}" >/dev/null 2>&1; then
        echo "Error: failed to log into W&B. Please verify the API key." >&2
        exit 1
    fi
else
    echo "W&B API key not provided via flags; relying on existing wandb login state."
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

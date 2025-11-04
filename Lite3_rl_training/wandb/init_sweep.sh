#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG="legged_gym.envs.base.two_leg_stand_config"
TEMPLATE_DIR="${SCRIPT_DIR}/templates"

usage() {
    cat <<EOF
Usage: $(basename "$0") -t <template> -e <entity> -p <project> [--dry-run] [--output <file>]

Templates:
  shallow      -> ${TEMPLATE_DIR}/two_leg_stand_shallow.json
  exhaustive   -> ${TEMPLATE_DIR}/two_leg_stand_exhaustive.json
  curriculum   -> ${TEMPLATE_DIR}/two_leg_stand_curriculum.json
  <path>       -> explicit path to a custom sweep template

Optional flags map directly to sweep_init.py; defaults assume the Lite3 two-leg stand config.
EOF
}

TEMPLATE_KEY=""
ENTITY=""
PROJECT=""
DRY_RUN=0
OUTPUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--template)
            TEMPLATE_KEY="$2"
            shift 2
            ;;
        -e|--entity)
            ENTITY="$2"
            shift 2
            ;;
        -p|--project)
            PROJECT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "${TEMPLATE_KEY}" || -z "${ENTITY}" || -z "${PROJECT}" ]]; then
    echo "Error: template, entity, and project are required." >&2
    usage
    exit 1
fi

case "${TEMPLATE_KEY}" in
    shallow)
        TEMPLATE="${TEMPLATE_DIR}/two_leg_stand_shallow.json"
        ;;
    exhaustive)
        TEMPLATE="${TEMPLATE_DIR}/two_leg_stand_exhaustive.json"
        ;;
    curriculum)
        TEMPLATE="${TEMPLATE_DIR}/two_leg_stand_curriculum.json"
        ;;
    *)
        TEMPLATE="${TEMPLATE_KEY}"
        ;;
esac

if [[ ! -f "${TEMPLATE}" ]]; then
    echo "Error: template not found at ${TEMPLATE}" >&2
    exit 1
fi

CMD=(python "${SCRIPT_DIR}/sweep_init.py"
    --template "${TEMPLATE}"
    --config "${DEFAULT_CONFIG}"
    --entity "${ENTITY}"
    --project "${PROJECT}"
)

if [[ ${DRY_RUN} -eq 1 ]]; then
    CMD+=(--dry-run)
fi

if [[ -n "${OUTPUT}" ]]; then
    CMD+=(--output "${OUTPUT}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

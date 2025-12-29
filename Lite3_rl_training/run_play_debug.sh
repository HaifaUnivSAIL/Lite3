#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_play_debug.sh /full/path/to/model_XXXX.pt [--task TASK] [--num-envs N] [--headless] [--debug-dumps N] [--dump-dir PATH]

Defaults:
  TASK = lite3_two_leg_stand_still_safe
  LITE3_FIXED_CMD = "0 0 0"
  LITE3_DISABLE_NEAR_GOAL = 1
  LITE3_DISABLE_DOMAIN_RAND = 1
  LITE3_DEBUG_DUMPS = 10
  LITE3_DEBUG_DEFAULT_RESET = 1
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 1
fi

ckpt_path="$1"
shift

if [[ ! -f "$ckpt_path" ]]; then
  echo "Checkpoint not found: $ckpt_path" >&2
  exit 1
fi

task="lite3_two_leg_stand_still_safe"
num_envs=""
headless_flag=""
debug_dumps="${LITE3_DEBUG_DUMPS:-10}"
dump_dir="${LITE3_DEBUG_DUMP_DIR:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      task="$2"
      shift 2
      ;;
    --num-envs)
      num_envs="$2"
      shift 2
      ;;
    --headless)
      headless_flag="--headless"
      shift
      ;;
    --debug-dumps)
      debug_dumps="$2"
      shift 2
      ;;
    --dump-dir)
      dump_dir="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

ckpt_dir="$(dirname "$ckpt_path")"
ckpt_file="$(basename "$ckpt_path")"

export LITE3_FIXED_CMD="${LITE3_FIXED_CMD:-0 0 0}"
export LITE3_DISABLE_NEAR_GOAL="${LITE3_DISABLE_NEAR_GOAL:-1}"
export LITE3_DISABLE_DOMAIN_RAND="${LITE3_DISABLE_DOMAIN_RAND:-1}"
export LITE3_DEBUG_DUMPS="$debug_dumps"
export LITE3_DEBUG_DEFAULT_RESET="${LITE3_DEBUG_DEFAULT_RESET:-1}"
if [[ -n "$dump_dir" ]]; then
  export LITE3_DEBUG_DUMP_DIR="$dump_dir"
fi

extra_args=()
if [[ -n "$num_envs" ]]; then
  extra_args+=(--num_envs "$num_envs")
fi
if [[ -n "$headless_flag" ]]; then
  extra_args+=("$headless_flag")
fi

python /workspace/Lite3_rl_training/legged_gym/legged_gym/scripts/play.py \
  --task "$task" \
  --load_run "$ckpt_dir" \
  --checkpoint "$ckpt_file" \
  "${extra_args[@]}"

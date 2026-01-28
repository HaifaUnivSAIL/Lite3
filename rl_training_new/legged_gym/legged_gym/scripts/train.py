import os
currentdir = os.path.dirname(os.path.abspath(__file__))
legged_gym_dir = os.path.dirname(os.path.dirname(currentdir))
isaacgym_dir = os.path.join(os.path.dirname(legged_gym_dir), "isaacgym/python")
rsl_rl_dir = os.path.join(os.path.dirname(legged_gym_dir), "rsl_rl")
os.sys.path.insert(0, legged_gym_dir)
os.sys.path.insert(0, isaacgym_dir)
os.sys.path.insert(0, rsl_rl_dir)
import numpy as np
import json
from datetime import datetime
import isaacgym
import shutil
from legged_gym.envs import *
from legged_gym.utils import get_args, Logger, register
from legged_gym.utils.task_registry import task_registry
from legged_gym.utils.helpers import class_to_dict


def _make_world_writable(path):
    """Best-effort chmod to allow collaborative writes to log directories."""
    try:
        os.chmod(path, 0o777)
    except OSError:
        # Non-fatal: continue training even if permissions cannot be changed
        pass


def train(args):
    register(args.task, task_registry)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for training
    env_cfg.commands.fixed_commands = None
    if args.near_goal_init_prob is not None:
        env_cfg.init_state.near_goal_init_prob = min(
            max(args.near_goal_init_prob, 0.0), 1.0)

    # prepare environment
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # load model
    if args.load_run:
        train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg,
        enable_summary_writer=True)

    # if this is a fresh run and a previous run folder exists, wipe it
    if not train_cfg.runner.resume and os.path.isdir(ppo_runner.log_dir):
        shutil.rmtree(ppo_runner.log_dir)

    # record configs as log files
    os.makedirs(ppo_runner.log_dir, exist_ok=True)
    # ensure collaborative write access for logs (experiment and run folders)
    log_parent = os.path.dirname(ppo_runner.log_dir)
    logs_root = os.path.dirname(log_parent)
    for p in (logs_root, log_parent, ppo_runner.log_dir):
        if os.path.isdir(p):
            _make_world_writable(p)
    # drop a helper script to replay this run easily
    run_dir_abs = os.path.abspath(ppo_runner.log_dir)
    run_play_path = os.path.join(ppo_runner.log_dir, "run_play.sh")
    run_resume_path = os.path.join(ppo_runner.log_dir, "run_resume.sh")
    run_evolution_path = os.path.join(ppo_runner.log_dir, "run_evolution.sh")
    exp_name = os.path.basename(os.path.dirname(run_dir_abs))
    run_name = os.path.basename(run_dir_abs)

    play_cmd = f"""#!/usr/bin/env bash
set -euo pipefail
# Resolve repo/log roots relative to this run directory
THIS_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 && pwd)"
LOGS_ROOT="$(cd "$THIS_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$LOGS_ROOT/.." && pwd)"

HEADLESS_FLAG=""
CHECKPOINT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --headless) HEADLESS_FLAG="--headless"; shift ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done
CKPT_FLAG=""
if [[ -n "$CHECKPOINT" ]]; then
  CKPT_FLAG="--checkpoint $CHECKPOINT"
fi

python "$REPO_ROOT/legged_gym/scripts/play.py" \\
  --task {args.task} \\
  --experiment_name "{exp_name}" \\
  --run_name "{run_name}" \\
  --load_run "{run_name}" \\
  $CKPT_FLAG \\
  $HEADLESS_FLAG
"""
    resume_cmd = f"""#!/usr/bin/env bash
set -euo pipefail
# Resolve repo/log roots relative to this run directory
THIS_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 && pwd)"
LOGS_ROOT="$(cd "$THIS_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$LOGS_ROOT/.." && pwd)"

CHECKPOINT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done
CKPT_FLAG=""
if [[ -n "$CHECKPOINT" ]]; then
  CKPT_FLAG="--checkpoint $CHECKPOINT"
fi

# Always headless for resume scripts
    python "$REPO_ROOT/legged_gym/scripts/train.py" \\
      --task {args.task} \\
      --resume \\
      --experiment_name "{exp_name}" \\
      --run_name "{run_name}" \\
      --load_run "{run_name}" \\
      $CKPT_FLAG \\
      --headless
"""
    evolution_cmd = f"""#!/usr/bin/env bash
set -euo pipefail

# Resolve repo/log roots relative to this run directory
THIS_DIR="$(cd -- "$(dirname "$0")" >/dev/null 2>&1 && pwd)"
LOGS_ROOT="$(cd "$THIS_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$LOGS_ROOT/.." && pwd)"

EXP_NAME="$(basename "$(dirname "$THIS_DIR")")"
RUN_NAME="$(basename "$THIS_DIR")"

# Default checkpoints to showcase policy evolution
DEFAULT_CKPTS=("model_500.pt" "model_1000.pt" "model_2500.pt" "model_3500.pt" "model_5000.pt")
CHECKPOINTS=("${{DEFAULT_CKPTS[@]}}")
FPS=30
SECONDS_PER_CKPT=5
FRAMES_PER_CKPT=""
NUM_ENVS=20
CAMERA_POS="7,4,1"
CAMERA_LOOKAT="0,0,0"
RECORD_SOURCE="viewer"   # viewer | camera
OUTPUT_NAME="policy_evolution.mp4"
KEEP_FRAMES="${{KEEP_FRAMES:-1}}"
SIM_DEVICE="${{SIM_DEVICE:-cuda:0}}"
RL_DEVICE="${{RL_DEVICE:-cuda:0}}"
LABEL_FRAMES="${{LABEL_FRAMES:-1}}"
LABEL_FONT_SIZE="${{LABEL_FONT_SIZE:-24}}"

usage() {{
  cat <<'USAGE'
Usage: ./run_evolution.sh [options]
  --checkpoints <ckpt1 ckpt2 ...>   Space-separated checkpoint names (default: model_500.pt ... model_7000.pt)
  --fps <int>                       Frames per second for the output video (default: 30)
  --seconds <float>                 Seconds to record per checkpoint (default: 5)
  --frames <int>                    Override total frames per checkpoint (overrides --seconds)
  --num-envs <int>                  Number of envs to roll out (default: 20)
  --camera-pos <x,y,z>              Camera position in env0 frame (default: 7,4,1)
  --camera-lookat <x,y,z>           Camera target in env0 frame (default: 0,0,0)
  --record-source <viewer|camera>   What to record into PNGs (default: viewer)
  --record-viewer                   Alias for --record-source viewer
  --record-camera                   Alias for --record-source camera
  --output <filename.mp4>           Final combined video name (default: policy_evolution.mp4)
  --keep-frames                     Keep individual PNG frames (default: kept)
  --label-frames                    Overlay checkpoint name on each frame (default: on)
  --no-labels                       Disable frame labeling
  --label-font-size <int>           Font size for labels (default: 24)
  --sim-device <device>             Simulation device (default: cuda:0, respects SIM_DEVICE env)
  --rl-device <device>              RL device (default: cuda:0, respects RL_DEVICE env)
USAGE
}}

# Parse CLI flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoints)
      CHECKPOINTS=()
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        CHECKPOINTS+=("$1")
        shift
      done
      ;;
    --fps) FPS="$2"; shift 2 ;;
    --seconds) SECONDS_PER_CKPT="$2"; shift 2 ;;
    --frames) FRAMES_PER_CKPT="$2"; shift 2 ;;
    --num-envs) NUM_ENVS="$2"; shift 2 ;;
    --camera-pos) CAMERA_POS="$2"; shift 2 ;;
    --camera-lookat) CAMERA_LOOKAT="$2"; shift 2 ;;
    --record-source) RECORD_SOURCE="$2"; shift 2 ;;
    --record-viewer) RECORD_SOURCE="viewer"; shift ;;
    --record-camera) RECORD_SOURCE="camera"; shift ;;
    --output) OUTPUT_NAME="$2"; shift 2 ;;
    --keep-frames) KEEP_FRAMES=1; shift ;;
    --label-frames) LABEL_FRAMES=1; shift ;;
    --no-labels) LABEL_FRAMES=0; shift ;;
    --label-font-size) LABEL_FONT_SIZE="$2"; shift 2 ;;
    --sim-device) SIM_DEVICE="$2"; shift 2 ;;
    --rl-device) RL_DEVICE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

TASK="{args.task}"

export EVOLVE_RUN_DIR="$THIS_DIR"
export EVOLVE_REPO_ROOT="$REPO_ROOT"
export EVOLVE_TASK="$TASK"
export EVOLVE_EXP_NAME="$EXP_NAME"
export EVOLVE_RUN_NAME="$RUN_NAME"
export EVOLVE_CHECKPOINTS="${{CHECKPOINTS[*]}}"
export EVOLVE_FPS="$FPS"
export EVOLVE_SECONDS="$SECONDS_PER_CKPT"
export EVOLVE_FRAMES="$FRAMES_PER_CKPT"
export EVOLVE_NUM_ENVS="$NUM_ENVS"
export EVOLVE_CAMERA_POS="$CAMERA_POS"
export EVOLVE_CAMERA_LOOKAT="$CAMERA_LOOKAT"
export EVOLVE_RECORD_SOURCE="$RECORD_SOURCE"
export EVOLVE_OUTPUT_NAME="$OUTPUT_NAME"
export EVOLVE_KEEP_FRAMES="$KEEP_FRAMES"
export EVOLVE_LABEL_FRAMES="$LABEL_FRAMES"
export EVOLVE_LABEL_FONT_SIZE="$LABEL_FONT_SIZE"
export EVOLVE_SIM_DEVICE="$SIM_DEVICE"
export EVOLVE_RL_DEVICE="$RL_DEVICE"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/legged_gym:$(cd "$REPO_ROOT/.." && pwd)/isaacgym/python:$(cd "$REPO_ROOT/.." && pwd)/rsl_rl:${{PYTHONPATH:-}}"

python - <<'PY'
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional
from PIL import Image, ImageDraw, ImageFont

# ffmpeg binary resolution
ffmpeg_bin = os.environ.get("FFMPEG_BIN", "ffmpeg")
if shutil.which(ffmpeg_bin) is None:
    sys.stderr.write(
        f"[error] ffmpeg not found (looked for '{{ffmpeg_bin}}'). "
        "Install it (e.g., `apt-get update && apt-get install -y ffmpeg`) "
        "or set FFMPEG_BIN=/path/to/ffmpeg.\\n"
    )
    sys.exit(1)
ffmpeg_crf = os.environ.get("FFMPEG_CRF", "18")
ffmpeg_preset = os.environ.get("FFMPEG_PRESET", "medium")


def _ensure_writable(path: Path, is_dir: bool = False):
    \"\"\"Best-effort permission relaxer.\"\"\"
    mode = 0o777 if is_dir else 0o666
    try:
        path.chmod(mode)
    except OSError:
        pass


# Resolve inputs from environment (keeps bash parser simple)
run_dir = Path(os.environ["EVOLVE_RUN_DIR"]).resolve()
repo_root = Path(os.environ["EVOLVE_REPO_ROOT"]).resolve()
task = os.environ.get("EVOLVE_TASK", "lite3")
exp_name = os.environ["EVOLVE_EXP_NAME"]
run_name = os.environ["EVOLVE_RUN_NAME"]
ckpt_list = os.environ.get("EVOLVE_CHECKPOINTS", "").split()
if not ckpt_list:
    print("No checkpoints provided. Nothing to record.", file=sys.stderr)
    sys.exit(1)
fps = int(os.environ.get("EVOLVE_FPS", "30"))
frames_env = os.environ.get("EVOLVE_FRAMES", "").strip()
if frames_env:
    frames_per_ckpt = int(float(frames_env))
else:
    seconds = float(os.environ.get("EVOLVE_SECONDS", "5"))
    frames_per_ckpt = max(1, int(fps * seconds))
num_envs = int(os.environ.get("EVOLVE_NUM_ENVS", "1"))
camera_pos = [float(x) for x in os.environ.get("EVOLVE_CAMERA_POS", "7,4,1").split(",")]
camera_lookat = [float(x) for x in os.environ.get("EVOLVE_CAMERA_LOOKAT", "0,0,0").split(",")]
record_source = os.environ.get("EVOLVE_RECORD_SOURCE", "viewer").strip().lower()
if record_source not in ("viewer", "camera"):
    print(f"[error] EVOLVE_RECORD_SOURCE must be 'viewer' or 'camera' (got: {{record_source!r}})", file=sys.stderr)
    sys.exit(1)
output_name = os.environ.get("EVOLVE_OUTPUT_NAME", "policy_evolution.mp4")
keep_frames = os.environ.get("EVOLVE_KEEP_FRAMES", "1") == "1"  # default: keep frames for debugging/quality checks
if keep_frames:
    print("[info] KEEP_FRAMES=1 -> PNG frames will be kept for inspection.")
else:
    print("[info] KEEP_FRAMES=0 -> PNG frames will be deleted after encoding.")
label_frames = os.environ.get("EVOLVE_LABEL_FRAMES", "1") == "1"
label_font_size = int(os.environ.get("EVOLVE_LABEL_FONT_SIZE", "24"))
sim_device = os.environ.get("EVOLVE_SIM_DEVICE", "cuda:0")
rl_device = os.environ.get("EVOLVE_RL_DEVICE", "cuda:0")

if len(camera_pos) != 3 or len(camera_lookat) != 3:
    print("Camera vectors must be 3 numbers each (got pos=%s lookat=%s)" % (camera_pos, camera_lookat), file=sys.stderr)
    sys.exit(1)

frames_root = run_dir / "evolution_frames"
videos_root = run_dir / "evolution_videos"
frames_root.mkdir(exist_ok=True)
videos_root.mkdir(exist_ok=True)
_ensure_writable(frames_root, is_dir=True)
_ensure_writable(videos_root, is_dir=True)

# Add repo deps to path just like train/play (redundant with PYTHONPATH but safe)
legged_gym_dir = repo_root  # parent of the legged_gym package folder
isaacgym_dir = repo_root.parent / "isaacgym" / "python"
rsl_rl_dir = repo_root.parent / "rsl_rl"
sys.path[:0] = [str(legged_gym_dir), str(legged_gym_dir / "legged_gym"), str(isaacgym_dir), str(rsl_rl_dir)]

from isaacgym import gymapi  # noqa: E402
import torch  # noqa: E402
from legged_gym.utils import register  # noqa: E402
from legged_gym.utils.task_registry import task_registry  # noqa: E402

register(task, task_registry)


def make_play_args(checkpoint: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        task=task,
        resume=False,
        experiment_name=exp_name,
        run_name=run_name,
        load_run=run_name,
        checkpoint=checkpoint,
        headless=False,  # needs a graphics device for camera sensors
        horovod=False,
        rl_device=rl_device,
        num_envs=num_envs,
        seed=None,
        max_iterations=None,
        near_goal_init_prob=None,
        save_rewards=False,
        physics_engine="physx",
        sim_device=sim_device,
        use_npu=False,
        num_threads=0,
        subscenes=0,
        slices=None,
    )


args = make_play_args()
env_cfg, train_cfg = task_registry.get_cfgs(name=task)
env_cfg.env.num_envs = num_envs
env_cfg.viewer.real_time_step = True
env_cfg.pmtg.train_mode = False
env_cfg.commands.fixed_commands = [0.8, 0.0, 0.0]
env_cfg.env.episode_length_s = max(env_cfg.env.episode_length_s, 1.0)

env, env_cfg = task_registry.make_env(name=task, args=args, env_cfg=env_cfg)
train_cfg.runner.resume = False  # manual checkpoint loads below
train_cfg.runner.checkpoint = -1
ppo_runner, train_cfg = task_registry.make_alg_runner(
    env=env, name=task, args=args, train_cfg=train_cfg, enable_summary_writer=False
)
policy_fn = ppo_runner.get_inference_policy(device=env.device)

# Single camera anchored in env0
gym = env.gym
sim = env.sim
viewer = getattr(env, "viewer", None)

use_viewer_capture = (
    record_source == "viewer"
    and viewer is not None
    and hasattr(gym, "write_viewer_image_to_file")
)
if record_source == "viewer" and not use_viewer_capture:
    why = []
    if viewer is None:
        why.append("env.viewer is None (headless?)")
    if not hasattr(gym, "write_viewer_image_to_file"):
        why.append("gym.write_viewer_image_to_file is unavailable")
    why_str = "; ".join(why) if why else "unknown reason"
    print(f"[warn] Viewer capture requested but unavailable ({{why_str}}); falling back to camera sensor capture.", file=sys.stderr)
    use_viewer_capture = False

cam_props = gymapi.CameraProperties()
cam_props.width = 1280
cam_props.height = 720
cam_handle = gym.create_camera_sensor(env.envs[0], cam_props)
gym.set_camera_location(
    cam_handle,
    env.envs[0],
    gymapi.Vec3(*camera_pos),
    gymapi.Vec3(*camera_lookat),
)


def record_checkpoint(checkpoint: str) -> Optional[Path]:
    ckpt_path = run_dir / checkpoint
    if not ckpt_path.exists():
        print(f"[skip] checkpoint {{checkpoint}} not found in {{run_dir}}", file=sys.stderr)
        return None

    # Load weights for this checkpoint
    ppo_runner.load(str(ckpt_path), load_optimizer=False)
    policy = ppo_runner.get_inference_policy(device=env.device)
    env.reset()

    # Refresh observations after reset for this checkpoint
    obs_dict = env.get_observations()
    obs, priv_obs, obs_history = (
        obs_dict["obs"],
        obs_dict["privileged_obs"],
        obs_dict["obs_history"],
    )

    frame_dir = frames_root / checkpoint.replace(".pt", "")
    frame_dir.mkdir(parents=True, exist_ok=True)

    print(f"[record] {{checkpoint}}: {{frames_per_ckpt}} frames @ {{fps}} fps")
    written_frames = 0
    for idx in range(frames_per_ckpt):
        with torch.no_grad():
            actions = policy(obs, obs_history)
        obs_dict, _, _, _ = env.step(actions)
        obs, priv_obs, obs_history = (
            obs_dict["obs"],
            obs_dict["privileged_obs"],
            obs_dict["obs_history"],
        )

        frame_path = frame_dir / f"frame_{{idx:05d}}.png"
        if use_viewer_capture:
            # Ensure the viewer has rendered the *current* sim state before grabbing a screenshot.
            # (env.step() renders at the beginning of the step, so we render once more here.)
            try:
                env.render(sync_frame_time=False)
            except TypeError:
                # Older BaseTask.render signature may not accept the kwarg.
                env.render(False)

            # Isaac Gym bindings differ slightly across versions; try a few common signatures.
            wrote = False
            for attempt in (
                lambda: gym.write_viewer_image_to_file(viewer, str(frame_path)),
                lambda: gym.write_viewer_image_to_file(viewer, sim, str(frame_path)),
                lambda: gym.write_viewer_image_to_file(sim, viewer, str(frame_path)),
            ):
                try:
                    attempt()
                    wrote = True
                    break
                except TypeError:
                    continue
            if not wrote:
                raise RuntimeError(
                    "Failed to call gym.write_viewer_image_to_file with expected signatures; "
                    "use --record-camera or update Isaac Gym bindings."
                )
        else:
            # Update graphics and capture from a camera sensor
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.render_all_camera_sensors(sim)
            gym.write_camera_image_to_file(
                sim, env.envs[0], cam_handle, gymapi.IMAGE_COLOR, str(frame_path)
            )
        written_frames += 1

    if written_frames == 0:
        print(f"[error] No frames were written for {{checkpoint}} (camera capture failed).", file=sys.stderr)
        return None

    if label_frames:
        import re
        digits = re.findall(r"\\d+", checkpoint)
        iter_str = digits[-1] if digits else checkpoint
        label_text = f"Behavior after {{iter_str}} learning iterations"
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", label_font_size)
        except Exception:
            font = ImageFont.load_default()
        for png_path in frame_dir.glob("frame_*.png"):
            with Image.open(png_path).convert("RGBA") as img:
                draw = ImageDraw.Draw(img)
                margin = 8
                text_bbox = draw.textbbox((0, 0), label_text, font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
                banner_height = text_h + margin * 2
                overlay = Image.new("RGBA", (img.width, banner_height), (0, 0, 0, 120))
                img.paste(overlay, (0, 0), overlay)
                x = max(margin, (img.width - text_w) // 2)
                y = margin
                draw.text((x, y), label_text, font=font, fill=(255, 255, 255, 255))
                img.convert("RGB").save(png_path)
        print(f"[label] Applied label '{{label_text}}' to {{written_frames}} frames in {{frame_dir}}")

    _ensure_writable(frame_dir, is_dir=True)
    return frame_dir


frame_dirs: List[Path] = []
total_frames = 0
for ckpt in ckpt_list:
    path = record_checkpoint(ckpt)
    if path:
        frame_dirs.append(path)
        total_frames += len(list(path.glob("frame_*.png")))

if not frame_dirs or total_frames == 0:
    print("No frames produced. Ensure checkpoints exist and Isaac Gym viewer can render.", file=sys.stderr)
    sys.exit(1)

combined_frames = frames_root / "combined_frames"
if combined_frames.exists():
    shutil.rmtree(combined_frames, ignore_errors=True)
combined_frames.mkdir(parents=True, exist_ok=True)

# Flatten all frames into a single sequential sequence for ffmpeg
frame_index = 0
for frame_dir in frame_dirs:
    for png_path in sorted(frame_dir.glob("frame_*.png")):
        target = combined_frames / f"frame_{{frame_index:06d}}.png"
        try:
            os.link(png_path, target)  # hardlink to avoid extra copies
        except OSError:
            shutil.copy2(png_path, target)
        frame_index += 1

if frame_index == 0:
    print("No frames found after flattening; aborting.", file=sys.stderr)
    sys.exit(1)

final_path = run_dir / output_name
ffmpeg_cmd = [
    ffmpeg_bin,
    "-y",
    "-framerate",
    str(fps),
    "-i",
    str(combined_frames / "frame_%06d.png"),
    "-c:v",
    "libx264",
    "-crf",
    str(ffmpeg_crf),
    "-preset",
    str(ffmpeg_preset),
    "-pix_fmt",
    "yuv420p",
    "-movflags",
    "+faststart",
    str(final_path),
]
subprocess.run(ffmpeg_cmd, check=True)

_ensure_writable(final_path, is_dir=False)
_ensure_writable(final_path.parent, is_dir=True)
if keep_frames:
    _ensure_writable(combined_frames, is_dir=True)
else:
    # cleanup combined frames and per-ckpt frames if user opted out of keeping them
    shutil.rmtree(combined_frames, ignore_errors=True)
    for frame_dir in frame_dirs:
        shutil.rmtree(frame_dir, ignore_errors=True)

print(f"[done] Video saved to: {{final_path}}")
PY
"""
    for path, content in [(run_play_path, play_cmd), (run_resume_path, resume_cmd), (run_evolution_path, evolution_cmd)]:
        with open(path, "w") as fp:
            fp.write(content)
        os.chmod(path, 0o755)
    with open(os.path.join(ppo_runner.log_dir, 'env_cfg.json'), 'w') as fp:
        json.dump(class_to_dict(env_cfg), fp)
    with open(os.path.join(ppo_runner.log_dir, 'train_cfg.json'), 'w') as fp:
        json.dump(class_to_dict(train_cfg), fp)

    # train ppo policy
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == '__main__':
    args = get_args()
    args.save_rewards = True
    train(args)

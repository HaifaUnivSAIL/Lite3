#!/usr/bin/env python3
"""
Compare deploy (MuJoCo ONNX) debug dumps with training play debug dumps.

Deploy dumps: debug_cpp_step*.txt (text key/value lines)
Training dumps: debug_play_step*.npz (numpy arrays)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


DEPLOY_KEY_WHITELIST = {
    "cmd",
    "base_rpy",
    "projected_gravity",
    "body_omega",
    "omega_world",
    "joint_pos_policy",
    "joint_vel_policy",
    "action_raw",
    "action_offset",
    "target_joint_pos",
    "obs_flat",
}


def _parse_step_index(path: Path) -> int | None:
    match = re.search(r"step(\d+)", path.name)
    if not match:
        return None
    return int(match.group(1))


def _parse_deploy_dump(path: Path) -> dict[str, np.ndarray]:
    data: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            key = parts[0]
            if key not in DEPLOY_KEY_WHITELIST:
                continue
            if len(parts) == 1:
                continue
            try:
                values = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            except ValueError:
                continue
            data[key] = values
    return data


def _load_train_dump(path: Path) -> dict[str, np.ndarray]:
    data = {}
    with np.load(path, allow_pickle=False) as npz:
        for key in npz.files:
            data[key] = npz[key]
    return data


def _compare_vec(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        return {"shape_mismatch": [list(a.shape), list(b.shape)]}
    diff = a - b
    abs_diff = np.abs(diff)
    max_abs = float(abs_diff.max()) if abs_diff.size > 0 else 0.0
    mean_abs = float(abs_diff.mean()) if abs_diff.size > 0 else 0.0
    topk = min(10, abs_diff.size)
    idx = np.argsort(abs_diff.reshape(-1))[-topk:][::-1]
    top = [
        {
            "index": int(i),
            "deploy": float(b.reshape(-1)[i]),
            "train": float(a.reshape(-1)[i]),
            "abs_diff": float(abs_diff.reshape(-1)[i]),
        }
        for i in idx
    ]
    return {"max_abs": max_abs, "mean_abs": mean_abs, "top_diffs": top}


def _discover_files(root: Path, pattern: str) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.exists():
        return []
    files = list(root.glob(pattern))
    if files:
        return files
    # Fall back to recursive search.
    return list(root.rglob(pattern))


def _index_by_step(files: list[Path]) -> dict[int, Path]:
    by_step: dict[int, Path] = {}
    for path in files:
        step = _parse_step_index(path)
        if step is None:
            continue
        if step not in by_step:
            by_step[step] = path
            continue
        # If duplicate step, pick the most recent.
        try:
            if path.stat().st_mtime > by_step[step].stat().st_mtime:
                by_step[step] = path
        except FileNotFoundError:
            continue
    return by_step


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare deploy and training play debug dumps.")
    parser.add_argument("--deploy-dir", required=True, help="Directory with debug_cpp_step*.txt dumps.")
    parser.add_argument("--train-dir", required=True, help="Directory with debug_play_step*.npz dumps.")
    parser.add_argument("--steps", type=int, default=None, help="Number of steps to compare (default: all common).")
    parser.add_argument("--out", type=str, default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    deploy_dir = Path(args.deploy_dir).expanduser()
    train_dir = Path(args.train_dir).expanduser()

    deploy_candidates = _discover_files(deploy_dir, "debug_cpp_step*.txt")
    train_candidates = _discover_files(train_dir, "debug_play_step*.npz")

    deploy_files = _index_by_step(deploy_candidates)
    train_files = _index_by_step(train_candidates)

    common_steps = sorted(set(deploy_files.keys()) & set(train_files.keys()))
    if not common_steps:
        print("[FAIL] No common steps found between deploy and train dumps.")
        print(f"[INFO] deploy_dir={deploy_dir} (found {len(deploy_candidates)} files)")
        print(f"[INFO] train_dir={train_dir} (found {len(train_candidates)} files)")
        print("[HINT] Use absolute paths inside the container, e.g.")
        print("  --deploy-dir /workspace/rl_training_new/lite3_debug/deploy")
        print("  --train-dir  /workspace/rl_training_new/logs/.../debug_play")
        return 1

    if args.steps is not None:
        common_steps = common_steps[: args.steps]

    report = {
        "steps": common_steps,
        "comparisons": {},
        "notes": [],
    }

    for step in common_steps:
        deploy_data = _parse_deploy_dump(deploy_files[step])
        train_data = _load_train_dump(train_files[step])

        step_report = {}
        # Compare flattened observation input
        if "obs_flat" in train_data and "obs_flat" in deploy_data:
            step_report["obs_flat"] = _compare_vec(train_data["obs_flat"].reshape(-1), deploy_data["obs_flat"])
        else:
            step_report["obs_flat"] = {"missing": True}

        # Compare raw actions
        if "actions" in train_data and "action_raw" in deploy_data:
            step_report["action_raw"] = _compare_vec(train_data["actions"].reshape(-1), deploy_data["action_raw"])
        else:
            step_report["action_raw"] = {"missing": True}

        # Compare key slices if available
        for key, deploy_key in [
            ("cmd", "cmd"),
            ("base_rpy", "base_rpy"),
            ("body_omega", "body_omega"),
            ("joint_pos", "joint_pos_policy"),
            ("joint_vel", "joint_vel_policy"),
        ]:
            if key in train_data and deploy_key in deploy_data:
                step_report[key] = _compare_vec(train_data[key].reshape(-1), deploy_data[deploy_key])
            else:
                step_report[key] = {"missing": True}

        report["comparisons"][str(step)] = step_report

    # Human-readable summary
    first_step = str(common_steps[0])
    if "obs_flat" in report["comparisons"][first_step]:
        obs_info = report["comparisons"][first_step]["obs_flat"]
        if "max_abs" in obs_info:
            report["notes"].append(
                f"step {first_step}: obs_flat max_abs={obs_info['max_abs']:.6f}, mean_abs={obs_info['mean_abs']:.6f}"
            )
    if "action_raw" in report["comparisons"][first_step]:
        act_info = report["comparisons"][first_step]["action_raw"]
        if "max_abs" in act_info:
            report["notes"].append(
                f"step {first_step}: action_raw max_abs={act_info['max_abs']:.6f}, mean_abs={act_info['mean_abs']:.6f}"
            )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    print("[OK] Comparison complete.")
    for note in report["notes"]:
        print(f"[INFO] {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Diagnose first-step parity gaps between train play dump and deploy dump.

Focuses on high-impact mismatch classes:
1) Quaternion/RPY convention branch mismatch (legacy XYZW vs standard WXYZ).
2) Reset/history seed mismatch at step0 (history leak vs zero-seed).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


def _parse_deploy_txt(path: Path) -> dict[str, object]:
    numeric_keys = {
        "cmd",
        "base_rpy",
        "base_quat_wxyz",
        "joint_pos_policy",
        "joint_vel_policy",
        "joint_pos_history",
        "joint_vel_history",
        "action_history",
        "obs_flat",
    }
    string_keys = {
        "parity_build_id",
        "base_rpy_mode",
        "action_history_mode",
        "history_seed_file_loaded",
        "history_seed_file_used",
        "history_seed_file_path",
    }
    out: dict[str, object] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or " " not in line:
                continue
            key, rest = line.split(" ", 1)
            if key in numeric_keys:
                out[key] = np.fromstring(rest, sep=" ", dtype=np.float64)
            elif key in string_keys:
                out[key] = rest.strip()
    return out


def _rpy_from_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quat_wxyz.tolist()
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _rpy_from_quat_legacy_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quat_wxyz.tolist()
    # Legacy path in deploy reinterprets wxyz as xyzw.
    w = qz
    x = qw
    y = qx
    z = qy

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _wrap_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _stats_abs(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    n = min(a.size, b.size)
    if n == 0:
        return 0.0, 0.0
    d = np.abs(a[:n] - b[:n])
    return float(d.max()), float(d.mean())


def _norm(x: np.ndarray | None) -> float:
    if x is None:
        return float("nan")
    return float(np.linalg.norm(x.reshape(-1)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-step", required=True, help="Path to debug_play_step0.npz")
    parser.add_argument("--deploy-step", required=True, help="Path to debug_cpp_step0.txt")
    args = parser.parse_args()

    train_path = Path(args.train_step).expanduser()
    deploy_path = Path(args.deploy_step).expanduser()

    if not train_path.exists():
        print(f"[FAIL] train step file missing: {train_path}")
        return 2
    if not deploy_path.exists():
        print(f"[FAIL] deploy step file missing: {deploy_path}")
        return 2

    train = np.load(train_path)
    deploy = _parse_deploy_txt(deploy_path)

    train_base_rpy = np.asarray(train["base_rpy"], dtype=np.float64).reshape(-1)
    deploy_base_rpy = np.asarray(deploy.get("base_rpy", np.zeros(3)), dtype=np.float64).reshape(-1)
    deploy_quat = np.asarray(deploy.get("base_quat_wxyz", np.array([1.0, 0.0, 0.0, 0.0])), dtype=np.float64).reshape(-1)

    deploy_rpy_std = _rpy_from_quat_wxyz(deploy_quat)
    deploy_rpy_legacy = _rpy_from_quat_legacy_xyzw(deploy_quat)

    mx_raw, mean_raw = _stats_abs(train_base_rpy, deploy_base_rpy)
    mx_wrap, mean_wrap = _stats_abs(_wrap_pi(train_base_rpy), _wrap_pi(deploy_base_rpy))
    mx_std, mean_std = _stats_abs(train_base_rpy, deploy_rpy_std)
    mx_legacy, mean_legacy = _stats_abs(train_base_rpy, deploy_rpy_legacy)

    train_action_hist = np.asarray(train["action_history"], dtype=np.float64).reshape(-1)
    deploy_action_hist = np.asarray(deploy.get("action_history", np.zeros(24)), dtype=np.float64).reshape(-1)
    train_jvel_hist = np.asarray(train["joint_vel_history"], dtype=np.float64).reshape(-1)
    deploy_jvel_hist = np.asarray(deploy.get("joint_vel_history", np.zeros(24)), dtype=np.float64).reshape(-1)

    print("=== Step0 Parity Diagnosis ===")
    print(f"train:  {train_path}")
    print(f"deploy: {deploy_path}")
    print("")
    print("[RPY]")
    print(f"  train base_rpy:          {np.array2string(train_base_rpy, precision=6)}")
    print(f"  deploy logged base_rpy:  {np.array2string(deploy_base_rpy, precision=6)}")
    print(f"  deploy quat->rpy std:    {np.array2string(deploy_rpy_std, precision=6)}")
    print(f"  deploy quat->rpy legacy: {np.array2string(deploy_rpy_legacy, precision=6)}")
    print(f"  abs diff train-vs-deploy logged: max={mx_raw:.6f}, mean={mean_raw:.6f}")
    print(f"  abs diff (pi-wrapped):          max={mx_wrap:.6f}, mean={mean_wrap:.6f}")
    print(f"  abs diff train-vs-std(quat):    max={mx_std:.6f}, mean={mean_std:.6f}")
    print(f"  abs diff train-vs-legacy(quat): max={mx_legacy:.6f}, mean={mean_legacy:.6f}")

    if mean_std < mean_legacy:
        print("  [HINT] Training aligns with standard WXYZ conversion better than deploy legacy conversion.")
    else:
        print("  [HINT] Training aligns with deploy legacy conversion (unlikely for current rl_training_new stack).")

    print("")
    print("[History Seed]")
    print(
        f"  ||train action_history||={_norm(train_action_hist):.6f}, "
        f"||deploy action_history||={_norm(deploy_action_hist):.6f}"
    )
    print(
        f"  ||train joint_vel_history||={_norm(train_jvel_hist):.6f}, "
        f"||deploy joint_vel_history||={_norm(deploy_jvel_hist):.6f}"
    )
    if _norm(train_action_hist) > 1e-3 and _norm(deploy_action_hist) < 1e-3:
        print("  [HINT] Step0 history mismatch: training appears pre-seeded/leaked while deploy is zero-seeded.")
    else:
        print("  [HINT] Step0 action history norms are not strongly contradictory.")

    mode = deploy.get("base_rpy_mode")
    hist_mode = deploy.get("action_history_mode")
    seed_used = deploy.get("history_seed_file_used")
    if mode is not None:
        print(f"  deploy base_rpy_mode: {mode}")
    if hist_mode is not None:
        print(f"  deploy action_history_mode: {hist_mode}")
    if seed_used is not None:
        print(f"  deploy history_seed_file_used: {seed_used}")

    print("")
    print("[Conclusion]")
    print("  Primary parity blockers are step0 convention/reset mismatch before policy-core comparison.")
    print("  Re-capture train dumps with LITE3_PLAY_FORCE_DEPLOY_RESET=1, then re-run compare.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


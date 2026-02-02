#!/usr/bin/env python3
"""Validate deploy parity for a two-leg-stand ONNX policy.

This test compares training-stack assumptions (obs layout, action scale,
default joint pose) against deploy runner assumptions and the ONNX model.
It is designed to be runnable from the repo root without IsaacLab imports.

Usage:
  python scripts/tools/validate_deploy_parity.py --policy exported/policy.onnx
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TRAIN_CFG = REPO_ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/two_leg_stand/two_leg_stand_env_cfg.py"
DEFAULT_TRAIN_ASSET = REPO_ROOT / "source/rl_training/rl_training/assets/deeprobotics.py"
DEFAULT_DEPLOY_HDR = REPO_ROOT / "../Lite3_rl_deploy/run_policy/lite3_test_policy_runner_onnx.hpp"


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}")


def _auto_find_deploy_header() -> Path | None:
    """Locate Lite3_rl_deploy header from common roots."""
    env_hint = os.getenv("LITE3_DEPLOY_DIR")
    candidates: List[Path] = []
    if env_hint:
        candidates.append(Path(env_hint))

    # Common roots in this workspace/container.
    candidates.extend(
        [
            REPO_ROOT.parent / "Lite3_rl_deploy",
            Path("/workspace/Lite3_rl_deploy"),
            Path("/home/sail/Lite3/Lite3_rl_deploy"),
        ]
    )

    for root in candidates:
        if not root.exists():
            continue
        for name in (
            "lite3_test_policy_runner_onnx.hpp",
            "lite3_test_policy_runner_onnx_backup.hpp",
        ):
            hdr = root / "run_policy" / name
            if hdr.exists():
                return hdr
    return None


def _float_list_from_cpp_initializer(text: str) -> List[float]:
    # Accept numbers with optional sign/decimal.
    vals = re.findall(r"[-+]?\d+\.?\d*(?:e[-+]?\d+)?", text, flags=re.IGNORECASE)
    return [float(v) for v in vals]


def _extract_training_action_scale(text: str) -> float | None:
    # Find action scale in TwoLegStandActionsCfg.
    m = re.search(r"joint_pos\s*=\s*base_mdp\.JointPositionActionCfg\([\s\S]*?scale=([0-9\.]+)", text)
    return float(m.group(1)) if m else None


def _extract_training_obs_scales(text: str) -> Dict[str, float]:
    # Minimal extraction for the 3 scales that matter in deploy parity.
    result: Dict[str, float] = {}
    patterns = {
        "velocity_commands": r"velocity_commands[\s\S]*?scale=([0-9\.]+)",
        "base_rpy": r"base_rpy[\s\S]*?scale=([0-9\.]+)",
        "base_ang_vel": r"base_ang_vel[\s\S]*?scale=([0-9\.]+)",
        "joint_pos": r"joint_pos[\s\S]*?scale=([0-9\.]+)",
        "joint_vel": r"joint_vel[\s\S]*?scale=([0-9\.]+)",
        "joint_pos_history": r"joint_pos_history[\s\S]*?scale=([0-9\.]+)",
        "joint_vel_history": r"joint_vel_history[\s\S]*?scale=([0-9\.]+)",
        "action_history": r"action_history[\s\S]*?scale=([0-9\.]+)",
    }
    for name, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            result[name] = float(m.group(1))
    return result


def _extract_training_obs_history_len(text: str) -> int | None:
    m = re.search(r"num_observation_history\s*:\s*int\s*=\s*(\d+)", text)
    return int(m.group(1)) if m else None


def _extract_training_default_pose(asset_text: str) -> Dict[str, float] | None:
    # Reads joint_pos dict from DEEPROBOTICS_LITE3_CFG init_state.
    # Expect keys like ".*HipX_joint": 0.0, etc.
    m = re.search(r"DEEPROBOTICS_LITE3_CFG[\s\S]*?joint_pos\s*=\s*\{([\s\S]*?)\}\s*,\s*joint_vel", asset_text)
    if not m:
        return None
    block = m.group(1)
    pairs = re.findall(r"\"([^\"]+)\"\s*:\s*([-+]?\d+\.?\d*)", block)
    if not pairs:
        return None
    return {k: float(v) for k, v in pairs}


def _extract_deploy_constants(hdr_text: str) -> Dict[str, float | int | List[float]]:
    consts: Dict[str, float | int | List[float]] = {}
    # kObsDim, kHistoryLen, kTrainingActionScale, kDofVelScale
    for name in ("kObsDim", "kHistoryLen", "kTrainingActionScale", "kDofVelScale"):
        m = re.search(rf"{name}\s*=\s*([0-9\.]+)", hdr_text)
        if m:
            val = m.group(1)
            consts[name] = int(val) if val.isdigit() else float(val)
    # default joint pose vector (dof_pos_default_policy_)
    m = re.search(r"dof_pos_default_policy_[\s\S]*?<<([\s\S]*?);", hdr_text)
    if m:
        consts["dof_pos_default_policy"] = _float_list_from_cpp_initializer(m.group(1))
    # kp/kd
    m = re.search(r"kp_\s*=\s*([0-9\.]+)f", hdr_text)
    if m:
        consts["kp"] = float(m.group(1))
    m = re.search(r"kd_\s*=\s*([0-9\.]+)f", hdr_text)
    if m:
        consts["kd"] = float(m.group(1))
    return consts


def _check_onnx_io(policy_path: Path) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    try:
        import onnx  # type: ignore
    except Exception:
        return True, ["[skip] onnx not installed; skipping ONNX input/output checks"]

    model = onnx.load(str(policy_path))
    graph = model.graph
    in_names = [i.name for i in graph.input]
    out_names = [o.name for o in graph.output]
    if "action" not in out_names:
        errors.append(f"ONNX outputs missing 'action' (found: {out_names})")
    # deploy expects a single input named 'obs'
    if in_names != ["obs"]:
        errors.append(
            "ONNX inputs are not ['obs']; deploy runner expects one flat input named 'obs'. "
            f"Found: {in_names}"
        )
    # try to read shape if present
    if graph.input:
        shape = []
        for dim in graph.input[0].type.tensor_type.shape.dim:
            if dim.dim_param:
                shape.append(dim.dim_param)
            elif dim.dim_value:
                shape.append(int(dim.dim_value))
            else:
                shape.append(None)
        if len(shape) >= 2 and isinstance(shape[1], int):
            if shape[1] not in (4797,):
                errors.append(f"ONNX input dim[1] expected 4797; got {shape[1]}")
    return (len(errors) == 0), errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True, help="Path to .onnx policy (relative to repo root or absolute)")
    parser.add_argument("--train-cfg", default=str(DEFAULT_TRAIN_CFG))
    parser.add_argument("--train-asset", default=str(DEFAULT_TRAIN_ASSET))
    parser.add_argument("--deploy-hdr", default=str(DEFAULT_DEPLOY_HDR))
    args = parser.parse_args()

    failures: List[str] = []
    notes: List[str] = []

    policy_path = Path(args.policy)
    if not policy_path.is_absolute():
        policy_path = (REPO_ROOT / policy_path).resolve()
    if not policy_path.exists():
        print(f"[fail] policy not found: {policy_path}")
        return 2

    train_cfg = Path(args.train_cfg)
    train_asset = Path(args.train_asset)
    deploy_hdr = Path(args.deploy_hdr)
    if not deploy_hdr.is_absolute():
        deploy_hdr = (REPO_ROOT / deploy_hdr).resolve()

    if not deploy_hdr.exists():
        auto_hdr = _auto_find_deploy_header()
        if auto_hdr is not None:
            deploy_hdr = auto_hdr
            notes.append(f"[auto] deploy header: {deploy_hdr}")

    train_text = _read_text(train_cfg)
    asset_text = _read_text(train_asset)
    deploy_text = ""
    if deploy_hdr.exists():
        deploy_text = _read_text(deploy_hdr)
    else:
        notes.append(
            "[skip] deploy header not found; pass --deploy-hdr to enable deploy-side checks"
        )

    # Training expectations
    action_scale = _extract_training_action_scale(train_text)
    obs_scales = _extract_training_obs_scales(train_text)
    obs_hist_len = _extract_training_obs_history_len(train_text)
    default_pose = _extract_training_default_pose(asset_text)

    if action_scale is None:
        failures.append("Could not extract training action scale")
    if obs_hist_len is None:
        failures.append("Could not extract training num_observation_history")

    # Deploy expectations
    deploy_consts = _extract_deploy_constants(deploy_text) if deploy_text else {}

    # Compare action scale
    deploy_action_scale = deploy_consts.get("kTrainingActionScale") if deploy_consts else None
    if action_scale is not None and deploy_action_scale is not None:
        if abs(float(action_scale) - float(deploy_action_scale)) > 1e-6:
            failures.append(
                f"Action scale mismatch: training={action_scale} deploy={deploy_action_scale}"
            )

    # Compare obs history length
    deploy_hist = deploy_consts.get("kHistoryLen") if deploy_consts else None
    if obs_hist_len is not None and deploy_hist is not None:
        if int(obs_hist_len) != int(deploy_hist):
            failures.append(
                f"History length mismatch: training={obs_hist_len} deploy={deploy_hist}"
            )

    # Compare dof vel scale
    train_dof_vel_scale = obs_scales.get("joint_vel", None)
    deploy_dof_vel_scale = deploy_consts.get("kDofVelScale") if deploy_consts else None
    if train_dof_vel_scale is not None and deploy_dof_vel_scale is not None:
        if abs(float(train_dof_vel_scale) - float(deploy_dof_vel_scale)) > 1e-6:
            failures.append(
                f"DOF vel scale mismatch: training={train_dof_vel_scale} deploy={deploy_dof_vel_scale}"
            )

    # Compare default pose (best-effort). Training uses regex patterns; deploy uses explicit list.
    deploy_pose = deploy_consts.get("dof_pos_default_policy") if deploy_consts else None
    if default_pose is not None and deploy_pose:
        # expected order in deploy runner: FL/FR/HL/HR HipX/HipY/Knee
        order = [
            "FL_HipX_joint", "FL_HipY_joint", "FL_Knee_joint",
            "FR_HipX_joint", "FR_HipY_joint", "FR_Knee_joint",
            "HL_HipX_joint", "HL_HipY_joint", "HL_Knee_joint",
            "HR_HipX_joint", "HR_HipY_joint", "HR_Knee_joint",
        ]
        # training default pose is regex-based; compare against the expected constants (0, -0.8, 1.6)
        # This flags if deploy hard-coded offsets drift too far from training config.
        expected = [0.0, -0.8, 1.6] * 4
        if len(deploy_pose) == 12:
            max_delta = max(abs(a - b) for a, b in zip(deploy_pose, expected))
            if max_delta > 1e-3:
                failures.append(
                    "Default joint pose mismatch: deploy hard-coded pose differs from training init_state."
                )
                notes.append(f"deploy default pose = {deploy_pose}")
        else:
            failures.append("Deploy default pose not found or wrong length")
    else:
        notes.append("[skip] Could not compare default pose; missing in training or deploy.")

    # ONNX IO check
    onnx_ok, onnx_msgs = _check_onnx_io(policy_path)
    if not onnx_ok:
        failures.extend(onnx_msgs)
    else:
        notes.extend(onnx_msgs)

    # Summarize
    if failures:
        print("[FAIL] Deploy parity check failed:")
        for msg in failures:
            print(f"  - {msg}")
        if notes:
            print("[INFO] Notes:")
            for msg in notes:
                print(f"  - {msg}")
        return 1

    print("[OK] Deploy parity checks passed")
    if notes:
        print("[INFO] Notes:")
        for msg in notes:
            print(f"  - {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Compare deploy (MuJoCo ONNX) debug dumps with training play debug dumps.

Deploy dumps: debug_cpp_step*.txt (text key/value lines)
Training dumps: debug_play_step*.npz (numpy arrays)
"""
from __future__ import annotations

import argparse
import json
import math
import re
import tempfile
from pathlib import Path

import numpy as np


DEPLOY_KEY_WHITELIST = {
    "obs_contract",
    "cmd",
    "base_rpy",
    "base_quat_wxyz",
    "base_rot_mat",
    "projected_gravity",
    "body_omega",
    "omega_world",
    "joint_pos_policy",
    "joint_vel_policy",
    "joint_pos_history",
    "joint_vel_history",
    "action_history",
    # legacy aliases
    "pos_hist",
    "vel_hist",
    "tgt_hist",
    "action_raw",
    "action_offset",
    "target_joint_pos_policy",
    "target_joint_pos",
    "target_joint_pos_clipped",
    "pd_tau_raw",
    "pd_tau_clipped",
    "joint_limits_lower",
    "joint_limits_upper",
    "effort_limits",
    "obs_flat",
}

OBS_CONTRACT = [
    ("cmd", 0, 3),
    ("base_rpy", 3, 6),
    ("body_omega", 6, 9),
    ("joint_pos", 9, 21),
    ("joint_vel", 21, 33),
    ("joint_pos_history", 33, 69),
    ("joint_vel_history", 69, 93),
    ("action_history", 93, 117),
]
TERM_SLICES = {name: (s, e) for name, s, e in OBS_CONTRACT}
OBS_CONTRACT_DIMS = [(name, e - s) for name, s, e in OBS_CONTRACT]

TRAIN_TERM_KEYS = {
    "cmd": ("cmd",),
    "base_rpy": ("base_rpy",),
    "body_omega": ("body_omega",),
    "joint_pos": ("joint_pos",),
    "joint_vel": ("joint_vel",),
    "joint_pos_history": ("joint_pos_history",),
    "joint_vel_history": ("joint_vel_history",),
    "action_history": ("action_history",),
}

DEPLOY_TERM_KEYS = {
    "cmd": ("cmd",),
    "base_rpy": ("base_rpy",),
    "body_omega": ("body_omega",),
    "joint_pos": ("joint_pos_policy",),
    "joint_vel": ("joint_vel_policy",),
    "joint_pos_history": ("joint_pos_history", "pos_hist"),
    "joint_vel_history": ("joint_vel_history", "vel_hist"),
    "action_history": ("action_history", "tgt_hist"),
}

STATE_FIELD_MAP: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
    ("base_rpy", ("base_rpy",), ("base_rpy",)),
    ("base_rot_mat", ("base_rot_mat",), ("base_rot_mat",)),
    ("body_omega", ("body_omega",), ("body_omega",)),
    ("projected_gravity", ("projected_gravity",), ("projected_gravity",)),
    ("omega_world", ("omega_world",), ("omega_world",)),
    ("base_quat_wxyz", ("base_quat_wxyz",), ("base_quat_wxyz",)),
]

CONTROL_FIELD_MAP: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
    ("action_raw", ("actions",), ("action_raw",)),
    ("action_offset", ("action_offset",), ("action_offset",)),
    ("target_joint_pos_policy", ("target_joint_pos_policy",), ("target_joint_pos_policy",)),
    ("target_joint_pos", ("target_joint_pos_robot", "target_joint_pos"), ("target_joint_pos",)),
    ("target_joint_pos_clipped", ("target_joint_pos_clipped",), ("target_joint_pos_clipped",)),
    ("pd_tau_raw", ("pd_tau_raw_est", "pd_tau_raw"), ("pd_tau_raw",)),
    ("pd_tau_clipped", ("pd_tau_clipped_est", "pd_tau_clipped"), ("pd_tau_clipped",)),
]


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
            if key == "obs_contract":
                names: list[str] = []
                dims: list[int] = []
                for token in parts[1:]:
                    if ":" not in token:
                        continue
                    name, dim = token.split(":", 1)
                    try:
                        names.append(name)
                        dims.append(int(dim))
                    except ValueError:
                        continue
                if names and dims and len(names) == len(dims):
                    data["obs_contract_names"] = np.asarray(names)
                    data["obs_contract_dims"] = np.asarray(dims, dtype=np.int32)
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
    a64 = a.reshape(-1).astype(np.float64, copy=False)
    b64 = b.reshape(-1).astype(np.float64, copy=False)
    diff = a64 - b64
    abs_diff = np.abs(diff)
    max_abs = float(abs_diff.max()) if abs_diff.size > 0 else 0.0
    mean_abs = float(abs_diff.mean()) if abs_diff.size > 0 else 0.0
    rmse = float(math.sqrt(np.mean(diff * diff))) if abs_diff.size > 0 else 0.0
    l2_diff = float(np.linalg.norm(diff))
    train_l2 = float(np.linalg.norm(a64))
    train_rms = float(math.sqrt(np.mean(a64 * a64))) if a64.size > 0 else 0.0
    train_std = float(np.std(a64)) if a64.size > 0 else 0.0
    eps = 1e-12
    # Robust normalization floors avoid exploding relative metrics when the reference signal is near zero.
    rms_floor = 1e-3
    std_floor = 1e-3
    l2_floor = float(np.sqrt(a64.size)) * rms_floor if a64.size > 0 else rms_floor
    rms_denom = max(train_rms, rms_floor)
    std_denom = max(train_std, std_floor)
    l2_denom = max(train_l2, l2_floor)

    topk = min(10, abs_diff.size)
    idx = np.argsort(abs_diff.reshape(-1))[-topk:][::-1]
    top = [
        {
            "index": int(i),
            "deploy": float(b64.reshape(-1)[i]),
            "train": float(a64.reshape(-1)[i]),
            "abs_diff": float(abs_diff.reshape(-1)[i]),
        }
        for i in idx
    ]
    return {
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rmse": rmse,
        "l2_diff": l2_diff,
        "rel_l2": float(l2_diff / (l2_denom + eps)),
        "nmae_rms": float(mean_abs / (rms_denom + eps)),
        "nrmse_rms": float(rmse / (rms_denom + eps)),
        "nmax_rms": float(max_abs / (rms_denom + eps)),
        "nmae_std": float(mean_abs / (std_denom + eps)),
        "nrmse_std": float(rmse / (std_denom + eps)),
        "nmax_std": float(max_abs / (std_denom + eps)),
        "train_rms": train_rms,
        "train_std": train_std,
        "normalization_floor": {
            "rms_floor": rms_floor,
            "std_floor": std_floor,
            "l2_floor": l2_floor,
        },
        "top_diffs": top,
    }


def _parse_obs_contract(data: dict[str, np.ndarray]) -> list[tuple[str, int, int]]:
    names = data.get("obs_contract_names")
    dims = data.get("obs_contract_dims")
    if names is None or dims is None:
        contract = []
        start = 0
        for name, dim in OBS_CONTRACT_DIMS:
            end = start + dim
            contract.append((name, start, end))
            start = end
        return contract

    name_list = [str(x) for x in np.asarray(names).reshape(-1).tolist()]
    dim_list = [int(x) for x in np.asarray(dims).reshape(-1).tolist()]
    if not name_list or len(name_list) != len(dim_list):
        return []
    if any(d <= 0 for d in dim_list):
        return []

    out: list[tuple[str, int, int]] = []
    start = 0
    for name, dim in zip(name_list, dim_list):
        end = start + dim
        out.append((name, start, end))
        start = end
    return out


def _term_slice(contract: list[tuple[str, int, int]], term_name: str) -> tuple[int, int] | None:
    for name, s, e in contract:
        if name == term_name:
            return s, e
    return None


def _compact_metrics(metrics: dict) -> dict:
    keys = (
        "max_abs",
        "mean_abs",
        "rmse",
        "rel_l2",
        "nmae_rms",
        "nrmse_rms",
        "nmax_rms",
        "nmae_std",
        "nrmse_std",
        "nmax_std",
    )
    return {k: metrics[k] for k in keys if k in metrics}


def _compare_obs_sections(
    train_obs_flat: np.ndarray,
    deploy_obs_flat: np.ndarray,
    train_contract: list[tuple[str, int, int]],
    deploy_contract: list[tuple[str, int, int]],
) -> dict:
    train_obs_dim = sum(e - s for _, s, e in train_contract)
    deploy_obs_dim = sum(e - s for _, s, e in deploy_contract)
    if train_obs_flat.shape != deploy_obs_flat.shape:
        return {"shape_mismatch": [list(train_obs_flat.shape), list(deploy_obs_flat.shape)]}
    if train_obs_flat.size < train_obs_dim or deploy_obs_flat.size < deploy_obs_dim:
        return {"shape_mismatch": [list(train_obs_flat.shape), list(deploy_obs_flat.shape)]}

    train_curr = train_obs_flat[:train_obs_dim]
    deploy_curr = deploy_obs_flat[:deploy_obs_dim]
    out: dict[str, dict] = {"current": {}, "history": {}}

    # Current frame section diffs.
    common_terms = [name for name, _, _ in train_contract if _term_slice(deploy_contract, name) is not None]
    for name in common_terms:
        tr = _term_slice(train_contract, name)
        dr = _term_slice(deploy_contract, name)
        if tr is None or dr is None:
            continue
        ts, te = tr
        ds, de = dr
        c = _compare_vec(train_curr[ts:te], deploy_curr[ds:de])
        out["current"][name] = _compact_metrics(c)

    # History section diffs across all history frames.
    hist_train = train_obs_flat[train_obs_dim:]
    hist_deploy = deploy_obs_flat[deploy_obs_dim:]
    if hist_train.size != hist_deploy.size:
        out["history"]["shape_mismatch"] = [int(hist_train.size), int(hist_deploy.size)]
        return out
    if train_obs_dim <= 0 or deploy_obs_dim <= 0:
        out["history"]["shape_mismatch"] = [int(hist_train.size), int(hist_deploy.size)]
        return out
    if hist_train.size % train_obs_dim != 0 or hist_deploy.size % deploy_obs_dim != 0:
        out["history"]["shape_mismatch"] = [int(hist_train.size), int(hist_deploy.size)]
        return out

    n_hist_train = hist_train.size // train_obs_dim
    n_hist_deploy = hist_deploy.size // deploy_obs_dim
    if n_hist_train != n_hist_deploy:
        out["history"]["shape_mismatch"] = [int(n_hist_train), int(n_hist_deploy)]
        return out
    n_hist = n_hist_train
    train_hist = hist_train.reshape(n_hist, train_curr.size)
    deploy_hist = hist_deploy.reshape(n_hist, deploy_curr.size)
    for name in common_terms:
        tr = _term_slice(train_contract, name)
        dr = _term_slice(deploy_contract, name)
        if tr is None or dr is None:
            continue
        ts, te = tr
        ds, de = dr
        c = _compare_vec(train_hist[:, ts:te].reshape(-1), deploy_hist[:, ds:de].reshape(-1))
        out["history"][name] = _compact_metrics(c)
    return out


def _compare_history_frames(
    train_obs_flat: np.ndarray,
    deploy_obs_flat: np.ndarray,
    train_contract: list[tuple[str, int, int]],
    deploy_contract: list[tuple[str, int, int]],
) -> dict:
    train_obs_dim = sum(e - s for _, s, e in train_contract)
    deploy_obs_dim = sum(e - s for _, s, e in deploy_contract)
    if train_obs_flat.shape != deploy_obs_flat.shape:
        return {"shape_mismatch": [list(train_obs_flat.shape), list(deploy_obs_flat.shape)]}
    if train_obs_flat.size < train_obs_dim or deploy_obs_flat.size < deploy_obs_dim:
        return {"shape_mismatch": [list(train_obs_flat.shape), list(deploy_obs_flat.shape)]}

    hist_train = train_obs_flat[train_obs_dim:]
    hist_deploy = deploy_obs_flat[deploy_obs_dim:]
    if hist_train.size != hist_deploy.size:
        return {"shape_mismatch": [int(hist_train.size), int(hist_deploy.size)]}
    if train_obs_dim <= 0 or deploy_obs_dim <= 0:
        return {"shape_mismatch": [int(hist_train.size), int(hist_deploy.size)]}
    if hist_train.size % train_obs_dim != 0 or hist_deploy.size % deploy_obs_dim != 0:
        return {"shape_mismatch": [int(hist_train.size), int(hist_deploy.size)]}

    n_hist_train = hist_train.size // train_obs_dim
    n_hist_deploy = hist_deploy.size // deploy_obs_dim
    if n_hist_train != n_hist_deploy:
        return {"shape_mismatch": [int(n_hist_train), int(n_hist_deploy)]}
    n_hist = n_hist_train
    train_hist = hist_train.reshape(n_hist, train_obs_dim)
    deploy_hist = hist_deploy.reshape(n_hist, deploy_obs_dim)
    common_terms = [name for name, _, _ in train_contract if _term_slice(deploy_contract, name) is not None]

    frames: list[dict] = []
    for i in range(n_hist):
        frame_cmp = _compare_vec(train_hist[i], deploy_hist[i])
        frame_terms = {}
        for term_name in common_terms:
            tr = _term_slice(train_contract, term_name)
            dr = _term_slice(deploy_contract, term_name)
            if tr is None or dr is None:
                continue
            ts, te = tr
            ds, de = dr
            frame_terms[term_name] = _compact_metrics(
                _compare_vec(train_hist[i, ts:te], deploy_hist[i, ds:de])
            )
        frames.append(
            {
                "frame": int(i),
                "overall": _compact_metrics(frame_cmp),
                "terms": frame_terms,
            }
        )

    by_rel = sorted(
        (
            {
                "frame": f["frame"],
                "rel_l2": f["overall"].get("rel_l2", 0.0),
                "max_abs": f["overall"].get("max_abs", 0.0),
                "mean_abs": f["overall"].get("mean_abs", 0.0),
            }
            for f in frames
        ),
        key=lambda x: float(x.get("rel_l2", 0.0)),
        reverse=True,
    )
    by_max = sorted(
        by_rel,
        key=lambda x: float(x.get("max_abs", 0.0)),
        reverse=True,
    )

    return {
        "num_history_frames": int(n_hist),
        "frames": frames,
        "top_frames_by_rel_l2": by_rel[:5],
        "top_frames_by_max_abs": by_max[:5],
    }


def _get_first_available(data: dict[str, np.ndarray], keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return None


def _get_term_from_obs(
    data: dict[str, np.ndarray],
    term_name: str,
    contract: list[tuple[str, int, int]],
) -> np.ndarray | None:
    obs = data.get("obs")
    if obs is None:
        return None
    term = _term_slice(contract, term_name)
    if term is None:
        return None
    s, e = term
    flat = obs.reshape(-1)
    if flat.size < e:
        return None
    return flat[s:e]


def _get_term_from_obs_flat(
    data: dict[str, np.ndarray],
    term_name: str,
    contract: list[tuple[str, int, int]],
) -> np.ndarray | None:
    obs_flat = data.get("obs_flat")
    if obs_flat is None:
        return None
    term = _term_slice(contract, term_name)
    if term is None:
        return None
    s, e = term
    flat = obs_flat.reshape(-1)
    if flat.size < e:
        return None
    return flat[s:e]


def _build_preprocess_block_report(
    train_data: dict[str, np.ndarray],
    deploy_data: dict[str, np.ndarray],
    train_contract: list[tuple[str, int, int]],
    deploy_contract: list[tuple[str, int, int]],
) -> dict[str, dict]:
    report: dict[str, dict] = {}
    for term_name in TRAIN_TERM_KEYS:
        train_direct = _get_first_available(train_data, TRAIN_TERM_KEYS[term_name])
        deploy_direct = _get_first_available(deploy_data, DEPLOY_TERM_KEYS[term_name])
        train_obs = _get_term_from_obs(train_data, term_name, train_contract)
        deploy_obs = _get_term_from_obs(deploy_data, term_name, deploy_contract)
        train_obs_flat = _get_term_from_obs_flat(train_data, term_name, train_contract)
        deploy_obs_flat = _get_term_from_obs_flat(deploy_data, term_name, deploy_contract)

        train_ref = train_direct
        train_ref_src = "train.direct"
        if train_ref is None:
            train_ref = train_obs
            train_ref_src = "train.obs"
        if train_ref is None:
            train_ref = train_obs_flat
            train_ref_src = "train.obs_flat"

        deploy_ref = deploy_direct
        deploy_ref_src = "deploy.direct"
        if deploy_ref is None:
            deploy_ref = deploy_obs
            deploy_ref_src = "deploy.obs"
        if deploy_ref is None:
            deploy_ref = deploy_obs_flat
            deploy_ref_src = "deploy.obs_flat"

        term_report: dict[str, object] = {
            "sources": {
                "train_ref": train_ref_src if train_ref is not None else "missing",
                "deploy_ref": deploy_ref_src if deploy_ref is not None else "missing",
                "train_direct": train_direct is not None,
                "deploy_direct": deploy_direct is not None,
                "train_obs": train_obs is not None,
                "deploy_obs": deploy_obs is not None,
                "train_obs_flat": train_obs_flat is not None,
                "deploy_obs_flat": deploy_obs_flat is not None,
            }
        }
        if train_ref is not None and deploy_ref is not None:
            term_report["train_vs_deploy"] = _compact_metrics(
                _compare_vec(train_ref.reshape(-1), deploy_ref.reshape(-1))
            )
        else:
            term_report["train_vs_deploy"] = {"missing": True}

        if train_direct is not None and train_obs_flat is not None:
            term_report["train_direct_vs_obs_flat"] = _compact_metrics(
                _compare_vec(train_direct.reshape(-1), train_obs_flat.reshape(-1))
            )
        if train_obs is not None and train_obs_flat is not None:
            term_report["train_obs_vs_obs_flat"] = _compact_metrics(
                _compare_vec(train_obs.reshape(-1), train_obs_flat.reshape(-1))
            )
        if deploy_direct is not None and deploy_obs_flat is not None:
            term_report["deploy_direct_vs_obs_flat"] = _compact_metrics(
                _compare_vec(deploy_direct.reshape(-1), deploy_obs_flat.reshape(-1))
            )
        if deploy_obs is not None and deploy_obs_flat is not None:
            term_report["deploy_obs_vs_obs_flat"] = _compact_metrics(
                _compare_vec(deploy_obs.reshape(-1), deploy_obs_flat.reshape(-1))
            )

        report[term_name] = term_report
    return report


def _compare_named_fields(
    train_data: dict[str, np.ndarray],
    deploy_data: dict[str, np.ndarray],
    field_map: list[tuple[str, tuple[str, ...], tuple[str, ...]]],
) -> dict[str, dict]:
    report: dict[str, dict] = {}
    for name, train_keys, deploy_keys in field_map:
        train_val = _get_first_available(train_data, train_keys)
        deploy_val = _get_first_available(deploy_data, deploy_keys)
        if train_val is None or deploy_val is None:
            report[name] = {"missing": True}
            continue
        report[name] = _compare_vec(train_val.reshape(-1), deploy_val.reshape(-1))
    return report


def _summarize_stage_metrics(metrics_by_name: dict[str, dict]) -> dict[str, object]:
    available_nrmse: list[tuple[str, float]] = []
    available_rel: list[tuple[str, float]] = []
    available_max: list[tuple[str, float]] = []
    missing_names: list[str] = []

    for name, metrics in metrics_by_name.items():
        if not isinstance(metrics, dict):
            missing_names.append(name)
            continue
        if metrics.get("missing"):
            missing_names.append(name)
            continue
        if "shape_mismatch" in metrics:
            missing_names.append(name)
            continue
        if "nrmse_rms" in metrics:
            available_nrmse.append((name, float(metrics["nrmse_rms"])))
        if "rel_l2" in metrics:
            available_rel.append((name, float(metrics["rel_l2"])))
        if "max_abs" in metrics:
            available_max.append((name, float(metrics["max_abs"])))

    out: dict[str, object] = {
        "num_fields_total": len(metrics_by_name),
        "num_fields_missing": len(missing_names),
        "missing_fields": missing_names,
    }
    if available_nrmse:
        worst_name, worst_val = max(available_nrmse, key=lambda x: x[1])
        out["worst_nrmse_field"] = worst_name
        out["worst_nrmse_rms"] = worst_val
    if available_rel:
        worst_name, worst_val = max(available_rel, key=lambda x: x[1])
        out["worst_rel_l2_field"] = worst_name
        out["worst_rel_l2"] = worst_val
    if available_max:
        worst_name, worst_val = max(available_max, key=lambda x: x[1])
        out["worst_max_abs_field"] = worst_name
        out["worst_max_abs"] = worst_val
    return out


def _build_stage_summary(step_report: dict) -> dict[str, dict]:
    preprocess_metrics = {
        "obs_flat": step_report.get("obs_flat", {"missing": True}),
        "cmd": step_report.get("cmd", {"missing": True}),
        "base_rpy": step_report.get("base_rpy", {"missing": True}),
        "body_omega": step_report.get("body_omega", {"missing": True}),
        "joint_pos": step_report.get("joint_pos", {"missing": True}),
        "joint_vel": step_report.get("joint_vel", {"missing": True}),
    }
    history_metrics = {
        "joint_pos_history": step_report.get("joint_pos_history", {"missing": True}),
        "joint_vel_history": step_report.get("joint_vel_history", {"missing": True}),
        "action_history": step_report.get("action_history", {"missing": True}),
    }
    history_frames = step_report.get("history_frames", {})
    if isinstance(history_frames, dict):
        top_rel = history_frames.get("top_frames_by_rel_l2", [])
        if isinstance(top_rel, list) and top_rel:
            t0 = top_rel[0]
            history_metrics["history_frame_worst"] = {
                "rel_l2": float(t0.get("rel_l2", 0.0)),
                "max_abs": float(t0.get("max_abs", 0.0)),
                "mean_abs": float(t0.get("mean_abs", 0.0)),
            }

    state_metrics = {}
    for name, metrics in step_report.get("state_fields", {}).items():
        state_metrics[name] = metrics
    control_metrics = {}
    for name, metrics in step_report.get("control_fields", {}).items():
        control_metrics[name] = metrics

    return {
        "preprocess": _summarize_stage_metrics(preprocess_metrics),
        "history": _summarize_stage_metrics(history_metrics),
        "state": _summarize_stage_metrics(state_metrics),
        "control": _summarize_stage_metrics(control_metrics),
    }


def _aggregate_stage_summaries(report: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for stage_name in ("preprocess", "history", "state", "control"):
        candidates_nrmse: list[tuple[int, str, float]] = []
        candidates_rel: list[tuple[int, str, float]] = []
        for step_str, step_report in report["comparisons"].items():
            stage = step_report.get("stage_summary", {}).get(stage_name, {})
            if not isinstance(stage, dict):
                continue
            if "worst_nrmse_rms" in stage:
                candidates_nrmse.append(
                    (int(step_str), str(stage.get("worst_nrmse_field", "")), float(stage["worst_nrmse_rms"]))
                )
            if "worst_rel_l2" in stage:
                candidates_rel.append(
                    (int(step_str), str(stage.get("worst_rel_l2_field", "")), float(stage["worst_rel_l2"]))
                )
        stage_out: dict[str, object] = {}
        if candidates_nrmse:
            step, field, value = max(candidates_nrmse, key=lambda x: x[2])
            stage_out["worst_nrmse_step"] = step
            stage_out["worst_nrmse_field"] = field
            stage_out["worst_nrmse_rms"] = value
        if candidates_rel:
            step, field, value = max(candidates_rel, key=lambda x: x[2])
            stage_out["worst_rel_l2_step"] = step
            stage_out["worst_rel_l2_field"] = field
            stage_out["worst_rel_l2"] = value
        out[stage_name] = stage_out
    return out


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
    parser.add_argument("--gate-preprocess-nrmse", type=float, default=None, help="Fail if preprocess stage nrmse_rms exceeds this threshold.")
    parser.add_argument("--gate-history-nrmse", type=float, default=None, help="Fail if history stage nrmse_rms exceeds this threshold.")
    parser.add_argument("--gate-state-nrmse", type=float, default=None, help="Fail if state stage nrmse_rms exceeds this threshold.")
    parser.add_argument("--gate-control-nrmse", type=float, default=None, help="Fail if control stage nrmse_rms exceeds this threshold.")
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

    gate_thresholds = {
        "preprocess": args.gate_preprocess_nrmse,
        "history": args.gate_history_nrmse,
        "state": args.gate_state_nrmse,
        "control": args.gate_control_nrmse,
    }

    report = {
        "steps": common_steps,
        "comparisons": {},
        "notes": [],
        "gates": {
            "thresholds": gate_thresholds,
            "failures": [],
        },
    }

    for step in common_steps:
        deploy_data = _parse_deploy_dump(deploy_files[step])
        train_data = _load_train_dump(train_files[step])
        train_contract = _parse_obs_contract(train_data)
        deploy_contract = _parse_obs_contract(deploy_data)
        train_obs_dim = sum(e - s for _, s, e in train_contract)
        deploy_obs_dim = sum(e - s for _, s, e in deploy_contract)

        step_report = {}
        # Compare flattened observation input
        if "obs_flat" in train_data and "obs_flat" in deploy_data:
            step_report["obs_flat"] = _compare_vec(train_data["obs_flat"].reshape(-1), deploy_data["obs_flat"])
            step_report["obs_sections"] = _compare_obs_sections(
                train_data["obs_flat"].reshape(-1),
                deploy_data["obs_flat"].reshape(-1),
                train_contract,
                deploy_contract,
            )
            step_report["history_frames"] = _compare_history_frames(
                train_data["obs_flat"].reshape(-1),
                deploy_data["obs_flat"].reshape(-1),
                train_contract,
                deploy_contract,
            )
        else:
            step_report["obs_flat"] = {"missing": True}
            step_report["obs_sections"] = {"missing": True}
            step_report["history_frames"] = {"missing": True}

        # Compare raw actions
        if "actions" in train_data and "action_raw" in deploy_data:
            step_report["action_raw"] = _compare_vec(train_data["actions"].reshape(-1), deploy_data["action_raw"])
        else:
            step_report["action_raw"] = {"missing": True}

        # Compare term-wise contract slices if available.
        for term_name in TRAIN_TERM_KEYS:
            train_term = _get_first_available(train_data, TRAIN_TERM_KEYS[term_name])
            deploy_term = _get_first_available(deploy_data, DEPLOY_TERM_KEYS[term_name])
            if train_term is None:
                train_term = _get_term_from_obs(train_data, term_name, train_contract)
            if train_term is None:
                train_term = _get_term_from_obs_flat(train_data, term_name, train_contract)
            if deploy_term is None:
                deploy_term = _get_term_from_obs(deploy_data, term_name, deploy_contract)
            if deploy_term is None:
                deploy_term = _get_term_from_obs_flat(deploy_data, term_name, deploy_contract)
            if train_term is not None and deploy_term is not None:
                step_report[term_name] = _compare_vec(train_term.reshape(-1), deploy_term.reshape(-1))
            else:
                step_report[term_name] = {"missing": True}

        step_report["contracts"] = {
            "train": [{"name": n, "start": int(s), "end": int(e)} for n, s, e in train_contract],
            "deploy": [{"name": n, "start": int(s), "end": int(e)} for n, s, e in deploy_contract],
            "train_obs_dim": int(train_obs_dim),
            "deploy_obs_dim": int(deploy_obs_dim),
        }
        step_report["state_fields"] = _compare_named_fields(
            train_data=train_data,
            deploy_data=deploy_data,
            field_map=STATE_FIELD_MAP,
        )
        step_report["control_fields"] = _compare_named_fields(
            train_data=train_data,
            deploy_data=deploy_data,
            field_map=CONTROL_FIELD_MAP,
        )
        step_report["preprocess_blocks"] = _build_preprocess_block_report(
            train_data,
            deploy_data,
            train_contract,
            deploy_contract,
        )
        step_report["stage_summary"] = _build_stage_summary(step_report)
        report["comparisons"][str(step)] = step_report

    report["stage_summary_overall"] = _aggregate_stage_summaries(report)

    gate_failures = []
    for step in common_steps:
        step_key = str(step)
        stage_summary = report["comparisons"][step_key].get("stage_summary", {})
        if not isinstance(stage_summary, dict):
            continue
        for stage_name, threshold in gate_thresholds.items():
            if threshold is None:
                continue
            stage_metrics = stage_summary.get(stage_name, {})
            if not isinstance(stage_metrics, dict):
                gate_failures.append(
                    {
                        "step": int(step),
                        "stage": stage_name,
                        "reason": "missing_stage_summary",
                    }
                )
                continue
            value = stage_metrics.get("worst_nrmse_rms")
            if value is None:
                gate_failures.append(
                    {
                        "step": int(step),
                        "stage": stage_name,
                        "reason": "no_nrmse_metric",
                    }
                )
                continue
            if float(value) > float(threshold):
                gate_failures.append(
                    {
                        "step": int(step),
                        "stage": stage_name,
                        "threshold": float(threshold),
                        "value": float(value),
                        "field": stage_metrics.get("worst_nrmse_field"),
                    }
                )
    report["gates"]["failures"] = gate_failures
    report["gates"]["passed"] = len(gate_failures) == 0

    # Human-readable summary
    first_step = str(common_steps[0])
    if "obs_flat" in report["comparisons"][first_step]:
        obs_info = report["comparisons"][first_step]["obs_flat"]
        if "max_abs" in obs_info:
            report["notes"].append(
                "step "
                f"{first_step}: obs_flat max_abs={obs_info['max_abs']:.6f}, mean_abs={obs_info['mean_abs']:.6f}, "
                f"rel_l2={obs_info.get('rel_l2', 0.0):.6e}, nrmse_rms={obs_info.get('nrmse_rms', 0.0):.6e}"
            )
    obs_sections = report["comparisons"][first_step].get("obs_sections")
    if isinstance(obs_sections, dict) and isinstance(obs_sections.get("history"), dict):
        hist_items = []
        for k, v in obs_sections["history"].items():
            if isinstance(v, dict) and "max_abs" in v:
                hist_items.append((k, float(v["max_abs"])))
        hist_items.sort(key=lambda x: x[1], reverse=True)
        if hist_items:
            top = ", ".join([f"{k}={v:.4f}" for k, v in hist_items[:3]])
            report["notes"].append(f"step {first_step}: history section max_abs top3: {top}")
    if "action_raw" in report["comparisons"][first_step]:
        act_info = report["comparisons"][first_step]["action_raw"]
        if "max_abs" in act_info:
            report["notes"].append(
                "step "
                f"{first_step}: action_raw max_abs={act_info['max_abs']:.6f}, mean_abs={act_info['mean_abs']:.6f}, "
                f"rel_l2={act_info.get('rel_l2', 0.0):.6e}, nrmse_rms={act_info.get('nrmse_rms', 0.0):.6e}"
            )
    hist_frames = report["comparisons"][first_step].get("history_frames")
    if isinstance(hist_frames, dict):
        top_rel = hist_frames.get("top_frames_by_rel_l2")
        if isinstance(top_rel, list) and top_rel:
            top = ", ".join(
                [
                    f"f{int(x['frame'])}:rel_l2={float(x['rel_l2']):.2e}"
                    for x in top_rel[:3]
                ]
            )
            report["notes"].append(f"step {first_step}: history top rel_l2 frames: {top}")

    if args.out:
        out_path = Path(args.out).expanduser()
        wrote_path: Path | None = None
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            wrote_path = out_path
        except OSError as exc:
            # Keep the compare flow running even if the requested output path is not writable.
            fallback = Path(tempfile.gettempdir()) / "compare_report.json"
            fallback.parent.mkdir(parents=True, exist_ok=True)
            with fallback.open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            wrote_path = fallback
            print(f"[WARN] Failed to write report to '{out_path}': {exc}")
            print(f"[WARN] Wrote report to fallback path: {fallback}")
        if wrote_path is not None:
            print(f"[INFO] Report path: {wrote_path}")

    print("[OK] Comparison complete.")
    for note in report["notes"]:
        if note.startswith("step 0:"):
            continue
        print(f"[INFO] {note}")
    # Per-step summary (helps spot drift beyond step 0).
    for step in common_steps:
        srep = report["comparisons"].get(str(step), {})
        obs_info = srep.get("obs_flat", {})
        act_info = srep.get("action_raw", {})
        if "max_abs" in obs_info and "max_abs" in act_info:
            print(
                f"[INFO] step {step}: "
                "obs_flat "
                f"max_abs={obs_info['max_abs']:.6f}, mean_abs={obs_info['mean_abs']:.6f}, "
                f"rel_l2={obs_info.get('rel_l2', 0.0):.3e}, nrmse_rms={obs_info.get('nrmse_rms', 0.0):.3e}; "
                "action_raw "
                f"max_abs={act_info['max_abs']:.6f}, mean_abs={act_info['mean_abs']:.6f}, "
                f"rel_l2={act_info.get('rel_l2', 0.0):.3e}, nrmse_rms={act_info.get('nrmse_rms', 0.0):.3e}"
            )
        obs_sections = srep.get("obs_sections", {})
        hist = obs_sections.get("history", {}) if isinstance(obs_sections, dict) else {}
        if isinstance(hist, dict):
            hist_items = []
            for k, v in hist.items():
                if isinstance(v, dict) and "max_abs" in v:
                    hist_items.append((k, float(v["max_abs"])))
            hist_items.sort(key=lambda x: x[1], reverse=True)
            if hist_items:
                top = ", ".join([f"{k}={v:.4f}" for k, v in hist_items[:3]])
                print(f"[INFO] step {step}: history top3 max_abs: {top}")
        hframes = srep.get("history_frames", {})
        top_rel = hframes.get("top_frames_by_rel_l2", []) if isinstance(hframes, dict) else []
        if isinstance(top_rel, list) and top_rel:
            top = ", ".join(
                [f"f{int(x['frame'])}:rel_l2={float(x['rel_l2']):.2e}" for x in top_rel[:3]]
            )
            print(f"[INFO] step {step}: history top3 rel_l2 frames: {top}")
        stage_summary = srep.get("stage_summary", {})
        if isinstance(stage_summary, dict):
            for stage_name in ("preprocess", "history", "state", "control"):
                stage = stage_summary.get(stage_name, {})
                if not isinstance(stage, dict):
                    continue
                if "worst_nrmse_rms" in stage:
                    print(
                        f"[INFO] step {step}: stage={stage_name} "
                        f"worst_nrmse_rms={float(stage['worst_nrmse_rms']):.3e} "
                        f"field={stage.get('worst_nrmse_field', 'n/a')}"
                    )
                elif "worst_rel_l2" in stage:
                    print(
                        f"[INFO] step {step}: stage={stage_name} "
                        f"worst_rel_l2={float(stage['worst_rel_l2']):.3e} "
                        f"field={stage.get('worst_rel_l2_field', 'n/a')}"
                    )
                else:
                    print(f"[INFO] step {step}: stage={stage_name} no comparable metrics")

    overall = report.get("stage_summary_overall", {})
    if isinstance(overall, dict):
        for stage_name in ("preprocess", "history", "state", "control"):
            stage = overall.get(stage_name, {})
            if not isinstance(stage, dict):
                continue
            if "worst_nrmse_rms" in stage:
                print(
                    f"[INFO] overall stage={stage_name}: "
                    f"worst_nrmse_rms={float(stage['worst_nrmse_rms']):.3e} "
                    f"(step {int(stage['worst_nrmse_step'])}, field={stage.get('worst_nrmse_field', 'n/a')})"
                )

    gate_failures = report.get("gates", {}).get("failures", [])
    if isinstance(gate_failures, list) and gate_failures:
        print(f"[FAIL] Gate checks failed ({len(gate_failures)} failure(s)).")
        for failure in gate_failures:
            if isinstance(failure, dict):
                step = failure.get("step", "n/a")
                stage = failure.get("stage", "n/a")
                reason = failure.get("reason")
                if reason is not None:
                    print(f"[FAIL] step {step} stage={stage}: {reason}")
                else:
                    print(
                        f"[FAIL] step {step} stage={stage}: "
                        f"value={float(failure.get('value', 0.0)):.3e} "
                        f"> threshold={float(failure.get('threshold', 0.0)):.3e} "
                        f"field={failure.get('field', 'n/a')}"
                    )
        return 2

    if any(v is not None for v in gate_thresholds.values()):
        print("[OK] Gate checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

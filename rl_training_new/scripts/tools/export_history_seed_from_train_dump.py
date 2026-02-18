#!/usr/bin/env python3
"""
Export reset history seed from a training debug dump (.npz) into a text file
that deploy can consume with LITE3_HISTORY_SEED_FILE.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _pick(npz: np.lib.npyio.NpzFile, *keys: str) -> np.ndarray | None:
    for key in keys:
        if key in npz:
            return np.asarray(npz[key], dtype=np.float32).reshape(-1)
    return None


def _vec_to_text(vec: np.ndarray) -> str:
    return " ".join(f"{float(v):.9g}" for v in vec.reshape(-1))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export deploy history seed from a training debug_play_step*.npz dump."
    )
    parser.add_argument("--train-dump", required=True, help="Path to debug_play_step*.npz")
    parser.add_argument("--out", required=True, help="Output text seed file path")
    parser.add_argument(
        "--include-obs-history",
        action="store_true",
        help="Also export obs_history (40x117) for full history replay.",
    )
    args = parser.parse_args()

    in_path = Path(args.train_dump)
    out_path = Path(args.out)
    if not in_path.exists():
        raise FileNotFoundError(f"Missing train dump: {in_path}")

    with np.load(in_path, allow_pickle=False) as npz:
        pos_hist = _pick(npz, "joint_pos_history", "pos_hist")
        vel_hist = _pick(npz, "joint_vel_history", "vel_hist")
        act_hist = _pick(npz, "action_history", "tgt_hist")
        obs_hist = _pick(npz, "obs_history")

    if pos_hist is None or pos_hist.size != 36:
        raise ValueError(f"joint_pos_history missing or invalid size (expected 36, got {None if pos_hist is None else pos_hist.size})")
    if vel_hist is None or vel_hist.size != 24:
        raise ValueError(f"joint_vel_history missing or invalid size (expected 24, got {None if vel_hist is None else vel_hist.size})")
    if act_hist is None or act_hist.size != 24:
        raise ValueError(f"action_history missing or invalid size (expected 24, got {None if act_hist is None else act_hist.size})")
    if args.include_obs_history:
        if obs_hist is None or obs_hist.size != 4680:
            raise ValueError(f"obs_history missing or invalid size (expected 4680, got {None if obs_hist is None else obs_hist.size})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Lite3 deploy history seed v1\n")
        f.write(f"# source_train_dump {in_path}\n")
        f.write("joint_pos_history ")
        f.write(_vec_to_text(pos_hist))
        f.write("\n")
        f.write("joint_vel_history ")
        f.write(_vec_to_text(vel_hist))
        f.write("\n")
        f.write("action_history ")
        f.write(_vec_to_text(act_hist))
        f.write("\n")
        if args.include_obs_history and obs_hist is not None:
            f.write("obs_history ")
            f.write(_vec_to_text(obs_hist))
            f.write("\n")

    print(f"[OK] Wrote seed file: {out_path}")
    print(f"[INFO] pos_hist={pos_hist.size} vel_hist={vel_hist.size} action_hist={act_hist.size}")
    if args.include_obs_history:
        print(f"[INFO] obs_history={obs_hist.size if obs_hist is not None else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

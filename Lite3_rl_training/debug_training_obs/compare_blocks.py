import argparse
import glob
import numpy as np
from pathlib import Path

BLOCKS = (
    ("cmd", 0, 3),
    ("orient", 3, 6),
    ("ang_vel", 6, 9),
    ("dof_pos", 9, 21),
    ("dof_vel", 21, 33),
    ("pos_hist", 33, 69),
    ("vel_hist", 69, 93),
    ("tgt_hist", 93, 117),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare training npz dumps against deployment cpp logs block-by-block."
    )
    parser.add_argument("--env-idx", type=int, default=0, help="Environment index to inspect.")
    parser.add_argument("--steps", type=int, default=3, help="How many steps to inspect.")
    parser.add_argument("--history-len", type=int, default=40, help="Observation history length.")
    parser.add_argument("--obs-dim", type=int, default=117, help="Single-frame observation dimension.")
    parser.add_argument("--all-history", action="store_true", help="Print stats for every history frame.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing dumps.",
    )
    return parser.parse_args()


def load_cpp(path: Path):
    with open(path) as f:
        lines = f.readlines()
    obs = np.array([float(x) for x in lines[0].split()[1:]], dtype=np.float32)
    act = np.array([float(x) for x in lines[1].split()[1:]], dtype=np.float32)
    return obs, act


def load_npz(path: Path, env_idx: int):
    data = np.load(path)
    obs = data["obs"]
    hist = data["obs_history"]
    action = data["action"]
    if obs.ndim == 2:
        obs = obs[env_idx]
    if hist.ndim == 3:
        hist = hist[env_idx]
    if action.ndim == 2:
        action = action[env_idx]
    return obs.astype(np.float32), hist.astype(np.float32), action.astype(np.float32)


def reshape_history(history: np.ndarray, frame_dim: int, history_len: int) -> np.ndarray:
    history = history.reshape(-1, frame_dim)
    if history.shape[0] > history_len:
        history = history[-history_len:]
    elif history.shape[0] < history_len:
        pad = np.zeros((history_len - history.shape[0], frame_dim), dtype=history.dtype)
        history = np.vstack([pad, history])
    return history


def summarize_block(name, start, end, ref, other):
    diff = np.abs(ref[start:end] - other[start:end])
    return diff.mean(), diff.max()


def first_large_diff(a, b, tol=1e-5):
    diff = np.abs(a - b)
    idx = np.argmax(diff > tol)
    if diff.size == 0 or diff.flat[idx] <= tol:
        return None
    return idx // a.shape[1], idx % a.shape[1], diff.flat[idx]


def describe_history(py_hist, cpp_hist, obs_dim, verbose):
    diff = np.abs(py_hist - cpp_hist)
    overall = diff.mean(), diff.max()
    mismatch = first_large_diff(py_hist, cpp_hist)
    if verbose:
        for frame_idx in range(py_hist.shape[0]):
            frame_diff = diff[frame_idx]
            print(
                f"    hist[{frame_idx:02d}] mean {frame_diff.mean():.6f} max {frame_diff.max():.6f}"
            )
    if mismatch:
        frame, elem, val = mismatch
        print(f"  first mismatch at frame {frame}, elem {elem}, abs diff {val:.6f}")
    return overall


def main():
    args = parse_args()
    root = args.root
    npz_files = sorted(root.glob("step_*.npz"))
    cpp_files = sorted(root.glob("debug_cpp_step*.txt"))
    steps = min(len(npz_files), len(cpp_files), args.steps)
    if steps == 0:
        raise SystemExit("No overlapping npz/txt files found.")

    for i in range(steps):
        py_obs, py_hist, py_action = load_npz(npz_files[i], args.env_idx)
        cpp_obs, cpp_action = load_cpp(cpp_files[i])

        py_hist = reshape_history(py_hist, args.obs_dim, args.history_len)
        cpp_frames = reshape_history(cpp_obs, args.obs_dim, args.history_len)
        latest_py = py_hist[-1]
        latest_cpp = cpp_frames[-1]

        print(f"\n=== Step {i} ({npz_files[i].name} vs {cpp_files[i].name}) ===")
        for name, start, end in BLOCKS:
            mean_diff, max_diff = summarize_block(name, start, end, latest_py, latest_cpp)
            print(f"{name:9s} mean {mean_diff:.6f} max {max_diff:.6f}")

        act_diff = np.abs(py_action - cpp_action)
        print(f"action   mean {act_diff.mean():.6f} max {act_diff.max():.6f}")

        hist_mean, hist_max = describe_history(py_hist, cpp_frames, args.obs_dim, args.all_history)
        print(f"history  mean {hist_mean:.6f} max {hist_max:.6f}")


if __name__ == "__main__":
    main()

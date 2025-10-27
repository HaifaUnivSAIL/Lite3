import numpy as np
import glob

ENV_IDX = 0


def load_cpp_txt(path):
    with open(path) as f:
        lines = f.readlines()
    obs = np.array([float(x) for x in lines[0].split()[1:]])
    act = np.array([float(x) for x in lines[1].split()[1:]])
    return obs, act


def select_env(array, idx=0):
    """Return the slice for a specific environment if the array has an env axis."""
    if array.ndim == 0:
        return array
    if array.ndim == 1:
        return array
    if idx >= array.shape[0]:
        raise IndexError(f"Env index {idx} out of range for shape {array.shape}")
    return array[idx]


npz_files = sorted(glob.glob("step_*.npz"))
cpp_files = sorted(glob.glob("debug_cpp_step*.txt"))

if not npz_files or not cpp_files:
    raise FileNotFoundError("Couldn't find .npz or .txt files. Check paths!")

num_steps = min(len(npz_files), len(cpp_files))

for i in range(num_steps):
    py = np.load(npz_files[i])
    cpp_obs, cpp_act = load_cpp_txt(cpp_files[i])

    obs = np.asarray(select_env(py["obs"], ENV_IDX))
    obs_history = np.asarray(select_env(py["obs_history"], ENV_IDX))
    action = np.asarray(select_env(py["action"], ENV_IDX))

    if obs_history.ndim == 1:
        obs_dim = obs.size
        if obs_dim == 0 or obs_history.size % obs_dim != 0:
            raise ValueError(
                f"Cannot reshape obs_history of size {obs_history.size} with obs size {obs_dim}"
            )
        hist_steps = obs_history.size // obs_dim
        obs_history = obs_history.reshape(hist_steps, obs_dim)

    # --- FIX: only keep the first 40 history steps (not all envs * steps)
    # expected obs_history shape is (40, 117)
    if obs_history.shape[0] > 40:
        obs_history = obs_history[:40]

    action = action.reshape(-1)

    # Flatten
    obs_flat = np.concatenate([obs.ravel(), obs_history.ravel()])

    print(f"\n--- Step {i} ---")
    print("Obs shapes:", obs_flat.shape, cpp_obs.shape)
    print("Action shapes:", action.shape, cpp_act.shape)
    print("Obs diff mean:", np.mean(np.abs(obs_flat - cpp_obs)))
    print("Obs diff max :", np.max(np.abs(obs_flat - cpp_obs)))
    print("Action diff mean:", np.mean(np.abs(action - cpp_act)))
    print("Action diff max :", np.max(np.abs(action - cpp_act)))

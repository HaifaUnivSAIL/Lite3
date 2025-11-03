import argparse
import csv
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
LEG_GYM_ROOT = PROJECT_ROOT / "legged_gym"
ISAACGYM_ROOT = PROJECT_ROOT / "isaacgym" / "python"
RSL_RL_ROOT = PROJECT_ROOT / "rsl_rl"

for candidate in (LEG_GYM_ROOT, ISAACGYM_ROOT, RSL_RL_ROOT, PROJECT_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

_UTILS_SPEC = importlib.util.spec_from_file_location(
    "lite3_wandb_utils", PACKAGE_ROOT / "utils.py"
)
_UTILS_MODULE = importlib.util.module_from_spec(_UTILS_SPEC)
assert _UTILS_SPEC.loader is not None
_UTILS_SPEC.loader.exec_module(_UTILS_MODULE)
sys.modules.setdefault("lite3_wandb_utils", _UTILS_MODULE)

try:
    import isaacgym  # noqa: F401 - ensure isaacgym loads before torch
except ImportError as exc:  # pragma: no cover - user guidance
    raise SystemExit(
        "Failed to import isaacgym. Make sure the Isaac Gym python bindings are available."
    ) from exc

def _remove_path_entries(target: Path) -> List[Tuple[int, str]]:
    removed: List[Tuple[int, str]] = []
    resolved_target = target.resolve()
    for index in range(len(sys.path) - 1, -1, -1):
        entry = sys.path[index]
        try:
            entry_path = Path(entry).resolve()
        except (OSError, RuntimeError, ValueError):
            continue
        if entry_path == resolved_target:
            removed.append((index, entry))
            del sys.path[index]
    return removed


def _import_external_wandb():
    removed_project = _remove_path_entries(PROJECT_ROOT)
    try:
        return importlib.import_module("wandb")
    except ModuleNotFoundError as exc:  # pragma: no cover - user guidance
        raise SystemExit(
            "wandb is required to run sweeps. Install it with `pip install wandb`."
        ) from exc
    finally:
        for index, value in reversed(removed_project):
            sys.path.insert(index, value)


wandb = _import_external_wandb()
os.environ.setdefault("WANDB_CONSOLE", "off")

from legged_gym.utils.helpers import class_to_dict, register
from legged_gym.utils.task_registry import task_registry
from lite3_wandb_utils import (
    WandbConfigError,
    apply_overrides,
    dotted_to_nested,
    group_by_prefix,
    instantiate_cfgs,
    load_config_module,
    to_serializable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a W&B sweep agent around the Lite3 training stack.",
    )
    parser.add_argument(
        "--sweep-id",
        required=True,
        help="W&B sweep identifier to join.",
    )
    parser.add_argument(
        "-n",
        "--num-runs",
        type=int,
        default=1,
        help="Number of runs for this agent to execute (default: 1).",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Python module path or file path for the target config (e.g. legged_gym.envs.base.two_leg_stand_config).",
    )
    parser.add_argument(
        "--env-class",
        help="Optional override for the LeggedRobotCfg subclass name inside the config module.",
    )
    parser.add_argument(
        "--train-class",
        help="Optional override for the LeggedRobotCfgPPO subclass name inside the config module.",
    )
    parser.add_argument(
        "--task",
        default="lite3_two_leg_stand",
        help="Task name to register with the task registry. Must be recognised by legged_gym.utils.helpers.register.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from a checkpoint.",
    )
    parser.add_argument(
        "--load-run",
        help="Run directory to resume from when --resume is set.",
    )
    parser.add_argument(
        "--checkpoint",
        help="Specific checkpoint filename to load when resuming (-1 means latest).",
    )
    parser.add_argument(
        "--experiment-name",
        help="Optional experiment name override for the PPO runner.",
    )
    parser.add_argument(
        "--run-name",
        help="Optional run name override for the PPO runner.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Force simulation to run headless.",
    )
    parser.add_argument(
        "--rl-device",
        default="cuda:0",
        help="Device used by the RL algorithm (e.g. cpu, cuda:0).",
    )
    parser.add_argument(
        "--sim-device",
        default="cuda:0",
        help="Device used by the simulator (e.g. cpu, cuda:0).",
    )
    parser.add_argument(
        "--physics-engine",
        default="physicsX",
        help="Physics engine backend to use.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        help="Override the number of environments. Updates both CLI args and config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed override. Applies to the PPO config.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        help="Maximum number of PPO iterations.",
    )
    parser.add_argument(
        "--use-npu",
        action="store_true",
        help="Use NPU for inference.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help="PhysX num_threads override.",
    )
    parser.add_argument(
        "--subscenes",
        type=int,
        default=0,
        help="Number of PhysX subscenes.",
    )
    parser.add_argument(
        "--slices",
        type=int,
        help="Number of client threads that process env slices.",
    )
    return parser.parse_args()


def build_train_args(cli_args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        task=cli_args.task,
        resume=cli_args.resume,
        experiment_name=cli_args.experiment_name,
        run_name=cli_args.run_name,
        load_run=cli_args.load_run,
        checkpoint=cli_args.checkpoint,
        headless=cli_args.headless,
        horovod=False,
        rl_device=cli_args.rl_device,
        num_envs=cli_args.num_envs,
        seed=cli_args.seed,
        max_iterations=cli_args.max_iterations,
        save_rewards=True,
        physics_engine=cli_args.physics_engine,
        sim_device=cli_args.sim_device,
        use_npu=cli_args.use_npu,
        num_threads=cli_args.num_threads,
        subscenes=cli_args.subscenes,
        slices=cli_args.slices,
    )


def extract_flat_config(wandb_config: wandb.sdk.wandb_config.Config) -> Dict[str, Any]:
    as_dict = wandb_config.as_dict()
    return {key: value for key, value in as_dict.items() if not key.startswith("_")}


def warn_for_unhandled(keys: Sequence[str], prefixes: Sequence[str]) -> None:
    handled = set()
    for key in keys:
        for prefix in prefixes:
            if key.startswith(f"{prefix}."):
                handled.add(key)
                break
    unused = sorted(set(keys) - handled)
    if unused:
        wandb.termwarn(
            "Ignoring sweep parameters outside recognised prefixes: "
            + ", ".join(unused)
        )


def apply_cli_overrides(train_cfg: Any, cli_args: argparse.Namespace) -> None:
    runner = getattr(train_cfg, "runner", None)
    if runner is None:
        return
    if cli_args.resume:
        runner.resume = True
    if cli_args.load_run is not None:
        runner.load_run = cli_args.load_run
    if cli_args.checkpoint is not None:
        runner.checkpoint = cli_args.checkpoint
    if cli_args.experiment_name is not None:
        runner.experiment_name = cli_args.experiment_name
    if cli_args.run_name is not None:
        runner.run_name = cli_args.run_name
    if cli_args.max_iterations is not None:
        runner.max_iterations = cli_args.max_iterations


def ensure_log_dir(log_dir: Optional[str]) -> None:
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)


def write_config_snapshots(log_dir: str, env_cfg: Any, train_cfg: Any) -> None:
    env_path = os.path.join(log_dir, "env_cfg.json")
    train_path = os.path.join(log_dir, "train_cfg.json")
    with open(env_path, "w", encoding="utf-8") as env_file:
        json.dump(to_serializable(class_to_dict(env_cfg)), env_file)
    with open(train_path, "w", encoding="utf-8") as train_file:
        json.dump(to_serializable(class_to_dict(train_cfg)), train_file)


def main() -> None:
    args = parse_args()
    try:
        config_module = load_config_module(args.config)
    except WandbConfigError as exc:
        raise SystemExit(str(exc)) from exc

    def sweep_train() -> None:
        with wandb.init(settings=wandb.Settings(console="off")) as run:
            flat_config = extract_flat_config(wandb.config)
            warn_for_unhandled(flat_config.keys(), prefixes=("env_cfg", "train_cfg"))

            register(args.task, task_registry)
            try:
                env_cfg, train_cfg = instantiate_cfgs(
                    config_module,
                    env_class_name=args.env_class,
                    train_class_name=args.train_class,
                )
            except WandbConfigError as exc:
                raise SystemExit(str(exc)) from exc

            grouped = group_by_prefix(flat_config, prefixes=("env_cfg", "train_cfg"))
            env_overrides = dotted_to_nested(grouped["env_cfg"]) if grouped["env_cfg"] else {}
            train_overrides = dotted_to_nested(grouped["train_cfg"]) if grouped["train_cfg"] else {}
            if env_overrides:
                apply_overrides(env_cfg, env_overrides)
            if train_overrides:
                apply_overrides(train_cfg, train_overrides)

            if args.num_envs is not None:
                env_cfg.env.num_envs = args.num_envs

            if args.seed is not None and hasattr(train_cfg, "seed"):
                train_cfg.seed = args.seed

            apply_cli_overrides(train_cfg, args)

            if hasattr(train_cfg, "seed"):
                setattr(env_cfg, "seed", getattr(train_cfg, "seed"))
            else:
                setattr(env_cfg, "seed", getattr(env_cfg, "seed", 0))

            run_name = getattr(train_cfg.runner, "run_name", "") if hasattr(train_cfg, "runner") else ""
            if not run_name and run is not None:
                auto_name = run.name or run.id
                if hasattr(train_cfg, "runner"):
                    train_cfg.runner.run_name = auto_name

            task_registry.env_cfgs[args.task] = env_cfg
            task_registry.train_cfgs[args.task] = train_cfg

            if hasattr(env_cfg, "commands"):
                env_cfg.commands.fixed_commands = None

            train_args = build_train_args(args)
            env, resolved_env_cfg = task_registry.make_env(
                name=args.task,
                args=train_args,
                env_cfg=env_cfg,
            )
            if getattr(train_args, "load_run", None):
                train_cfg.runner.resume = True

            ppo_runner, resolved_train_cfg = task_registry.make_alg_runner(
                env=env,
                name=args.task,
                args=train_args,
                train_cfg=train_cfg,
                enable_summary_writer=True,
            )

            ensure_log_dir(ppo_runner.log_dir)
            write_config_snapshots(ppo_runner.log_dir, resolved_env_cfg, resolved_train_cfg)

            if ppo_runner.log_dir:
                wandb.summary["log_dir"] = ppo_runner.log_dir
                env_cfg_path = Path(ppo_runner.log_dir) / "env_cfg.json"
                train_cfg_path = Path(ppo_runner.log_dir) / "train_cfg.json"
                if env_cfg_path.exists():
                    wandb.save(str(env_cfg_path))
                if train_cfg_path.exists():
                    wandb.save(str(train_cfg_path))

            ppo_runner.learn(
                num_learning_iterations=resolved_train_cfg.runner.max_iterations,
                init_at_random_ep_len=True,
            )

            if ppo_runner.log_dir:
                rewards_csv = Path(ppo_runner.log_dir) / "rewards.csv"
                if rewards_csv.exists():
                    with rewards_csv.open("r", encoding="utf-8") as reward_file:
                        reader = csv.DictReader(reward_file)
                        reward_values = [
                            float(row["total_reward"])
                            for row in reader
                            if "total_reward" in row and row["total_reward"] not in ("", "nan")
                        ]
                    if reward_values:
                        wandb.summary["episode_reward_mean"] = sum(reward_values) / len(reward_values)
                        wandb.summary["episode_reward_max"] = max(reward_values)
                        wandb.summary["episode_reward_last"] = reward_values[-1]

    wandb.agent(args.sweep_id, function=sweep_train, count=args.num_runs)


if __name__ == "__main__":
    main()

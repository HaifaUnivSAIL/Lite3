import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
LEG_GYM_ROOT = PROJECT_ROOT / "legged_gym"
RSL_RL_ROOT = PROJECT_ROOT / "rsl_rl"

for candidate in (LEG_GYM_ROOT, RSL_RL_ROOT, PROJECT_ROOT):
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
    except ModuleNotFoundError as exc:  # pragma: no cover - guidance for the user
        raise SystemExit(
            "wandb is required for sweep initialisation. Install it with "
            "`pip install wandb` before running this script."
        ) from exc
    finally:
        for index, value in reversed(removed_project):
            sys.path.insert(index, value)


wandb = _import_external_wandb()

from lite3_wandb_utils import (
    WandbConfigError,
    build_sweep_parameters,
    instantiate_cfgs,
    load_config_module,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialise a W&B sweep from a Lite3 config module.",
    )
    parser.add_argument(
        "--template",
        type=Path,
        required=True,
        help="Path to a JSON sweep template. The script populates its parameters and enforces bayes search.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Python module path or file path to a config module (e.g. legged_gym.envs.base.two_leg_stand_config).",
    )
    parser.add_argument(
        "--env-class",
        help="Optional override for the LeggedRobotCfg subclass name to use from the config module.",
    )
    parser.add_argument(
        "--train-class",
        help="Optional override for the LeggedRobotCfgPPO subclass name to use from the config module.",
    )
    parser.add_argument(
        "--entity",
        help="Optional W&B entity (team or username). Overrides any template value.",
    )
    parser.add_argument(
        "--project",
        help="Optional W&B project. Overrides any template value.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the resolved sweep JSON. Implies --dry-run if provided without --no-dry-run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the expanded sweep config without creating it on W&B.",
    )
    parser.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Create the sweep on W&B even if --output is supplied.",
    )
    parser.add_argument(
        "--no-defaults",
        dest="include_defaults",
        action="store_false",
        help="Do not seed the sweep config with all default parameters from the Lite3 config.",
    )
    parser.add_argument(
        "--include-defaults",
        dest="include_defaults",
        action="store_true",
        help="Seed the sweep config with default values for every parameter (current behaviour).",
    )
    parser.set_defaults(include_defaults=True)
    parser.set_defaults(dry_run=False)
    return parser.parse_args()


def load_template(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Sweep template not found: {path}")
    with path.open("r", encoding="utf-8") as fp:
        try:
            template = json.load(fp)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Failed to parse JSON template at {path}: {exc}") from exc
    if not isinstance(template, dict):
        raise SystemExit("Sweep template must contain a JSON object at the top level.")
    return template


def merge_parameters(template: Dict[str, Any], defaults: Dict[str, Any], include_defaults: bool) -> Dict[str, Any]:
    merged = dict(template)
    params = merged.setdefault("parameters", {})
    if not isinstance(params, dict):
        raise SystemExit("Expected 'parameters' in the template to be a JSON object.")
    if include_defaults:
        for key, value in defaults.items():
            param_block = params.get(key)
            if isinstance(param_block, dict):
                params.setdefault(key, param_block)
                uses_distribution = "distribution" in param_block or "values" in param_block
                if not uses_distribution:
                    params[key].setdefault("value", value)
            else:
                params[key] = {"value": value}
    merged["method"] = "bayes"
    return merged


def main() -> None:
    args = parse_args()
    try:
        module = load_config_module(args.config)
        env_cfg, train_cfg = instantiate_cfgs(
            module,
            env_class_name=args.env_class,
            train_class_name=args.train_class,
        )
    except WandbConfigError as exc:
        raise SystemExit(str(exc)) from exc

    defaults = build_sweep_parameters(env_cfg, train_cfg)
    template = load_template(args.template)
    merged = merge_parameters(template, defaults, include_defaults=args.include_defaults)
    export_payload: Dict[str, Any] = dict(merged)
    if args.entity or args.project:
        export_payload.setdefault("__wandb", {})
        if args.entity:
            export_payload["__wandb"]["entity"] = args.entity
        if args.project:
            export_payload["__wandb"]["project"] = args.project

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fp:
            json.dump(export_payload, fp, indent=2, sort_keys=True)
        print(f"Wrote expanded sweep template to {args.output}")

    if args.dry_run:
        print(json.dumps(export_payload, indent=2, sort_keys=True))
        return

    sweep_kwargs = {}
    if args.entity:
        sweep_kwargs["entity"] = args.entity
    if args.project:
        sweep_kwargs["project"] = args.project

    sweep_id = wandb.sweep(merged, **sweep_kwargs)
    print(f"Created sweep: {sweep_id}")


if __name__ == "__main__":
    main()

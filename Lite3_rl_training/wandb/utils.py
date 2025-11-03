import importlib
import importlib.util
import inspect
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from legged_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)
from legged_gym.utils.helpers import class_to_dict


class WandbConfigError(RuntimeError):
    """Raised when a configuration module cannot be resolved."""


def load_config_module(config_path: str) -> ModuleType:
    """
    Load a Python module that defines LeggedRobot configs.

    The caller may specify either a dotted import path (e.g.
    ``legged_gym.envs.base.two_leg_stand_config``) or a filesystem path to a
    Python file.
    """
    if not config_path:
        raise WandbConfigError("Config path must be a non-empty string")

    if config_path.endswith(".py") or "/" in config_path or config_path.startswith("."):
        path = Path(config_path).expanduser().resolve()
        if not path.exists():
            raise WandbConfigError(f"Config file does not exist: {path}")
        if not path.is_file():
            raise WandbConfigError(f"Config path is not a file: {path}")
        module_name = path.stem
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise WandbConfigError(f"Unable to create import spec from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    try:
        return importlib.import_module(config_path)
    except ModuleNotFoundError as exc:
        raise WandbConfigError(f"Unable to import config '{config_path}'") from exc


def _resolve_cfg_class(
    module: ModuleType,
    base_cls: type,
    explicit_name: Optional[str],
) -> type:
    if explicit_name:
        try:
            cls = getattr(module, explicit_name)
        except AttributeError as exc:
            raise WandbConfigError(
                f"Module '{module.__name__}' has no attribute '{explicit_name}'"
            ) from exc
        if not inspect.isclass(cls) or not issubclass(cls, base_cls):
            raise WandbConfigError(
                f"Attribute '{explicit_name}' in '{module.__name__}' is not a "
                f"subclass of {base_cls.__name__}"
            )
        return cls

    candidates = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, base_cls)
        and obj is not base_cls
        and obj.__module__ == module.__name__
    ]
    if not candidates:
        raise WandbConfigError(
            f"No subclasses of {base_cls.__name__} found in '{module.__name__}'"
        )
    if len(candidates) > 1:
        candidate_names = ", ".join(cls.__name__ for cls in candidates)
        raise WandbConfigError(
            f"Multiple subclasses of {base_cls.__name__} found in "
            f"'{module.__name__}': {candidate_names}. Please specify which one to use."
        )
    return candidates[0]


def instantiate_cfgs(
    module: ModuleType,
    env_class_name: Optional[str] = None,
    train_class_name: Optional[str] = None,
) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
    """
    Instantiate the environment and PPO configuration objects from a module.
    """
    env_cls = _resolve_cfg_class(module, LeggedRobotCfg, env_class_name)
    train_cls = _resolve_cfg_class(module, LeggedRobotCfgPPO, train_class_name)
    return env_cls(), train_cls()


def to_serializable(value: Any) -> Any:
    """Recursively convert config values to JSON-serializable Python objects."""
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    return value


def flatten_dict(tree: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Flatten a nested dictionary using dotted keys."""
    items: Dict[str, Any] = {}
    for key, value in tree.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict) and value:
            items.update(flatten_dict(value, new_key, sep=sep))
        else:
            items[new_key] = value
    return items


def flatten_cfg(cfg_obj: Any, prefix: str) -> Dict[str, Any]:
    """
    Flatten a configuration object into dotted-path parameters.

    The returned keys include the provided ``prefix`` so they can be grouped by
    config type later on. Example key: ``env_cfg.rewards.scales.torso_upright``.
    """
    raw_dict = class_to_dict(cfg_obj)
    serializable = to_serializable(raw_dict)
    flattened = flatten_dict(serializable)
    return {f"{prefix}.{key}": value for key, value in flattened.items()}


def group_by_prefix(
    flat_params: Dict[str, Any],
    prefixes: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {prefix: {} for prefix in prefixes}
    for key, value in flat_params.items():
        for prefix in prefixes:
            prefix_token = f"{prefix}."
            if key.startswith(prefix_token):
                inner_key = key[len(prefix_token):]
                grouped[prefix][inner_key] = value
                break
    return grouped


def dotted_to_nested(flat_dict: Dict[str, Any]) -> Dict[str, Any]:
    nested: Dict[str, Any] = {}
    for dotted_key, value in flat_dict.items():
        cursor = nested
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return nested


def apply_overrides(target: Any, overrides: Dict[str, Any]) -> None:
    """
    Recursively apply nested overrides to a config object.
    """
    for key, value in overrides.items():
        if not hasattr(target, key):
            raise WandbConfigError(
                f"Cannot apply override '{key}' - attribute not found on '{type(target).__name__}'"
            )
        attr = getattr(target, key)
        if isinstance(value, dict) and hasattr(attr, "__dict__"):
            apply_overrides(attr, value)
        else:
            setattr(target, key, value)


def build_sweep_parameters(
    env_cfg: LeggedRobotCfg,
    train_cfg: LeggedRobotCfgPPO,
) -> Dict[str, Any]:
    """
    Prepare a flattened parameter dictionary for both configs.
    """
    params = {}
    params.update(flatten_cfg(env_cfg, "env_cfg"))
    params.update(flatten_cfg(train_cfg, "train_cfg"))
    return params


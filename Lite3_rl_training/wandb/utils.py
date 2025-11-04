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

    def _parse_part(part: str) -> Tuple[str, Optional[int]]:
        if "[" in part and part.endswith("]"):
            base, index_str = part[:-1].split("[", 1)
            try:
                index = int(index_str)
            except ValueError as exc:
                raise WandbConfigError(f"Invalid list index in key segment '{part}'") from exc
            return base, index
        return part, None

    for dotted_key, value in flat_dict.items():
        cursor = nested
        parts = dotted_key.split(".")
        for i, part in enumerate(parts):
            base_name, list_index = _parse_part(part)
            is_last = i == len(parts) - 1
            if list_index is None:
                if is_last:
                    cursor[base_name] = value
                else:
                    cursor = cursor.setdefault(base_name, {})
            else:
                container = cursor.setdefault(base_name, {})
                if not isinstance(container, dict):
                    raise WandbConfigError(
                        f"Expected dict container for list overrides at '{part}', got {type(container).__name__}"
                    )
                if is_last:
                    container[list_index] = value
                else:
                    cursor = container.setdefault(list_index, {})
    return nested


def apply_overrides(target: Any, overrides: Dict[str, Any]) -> None:
    """
    Recursively apply nested overrides to a config object.
    """
    def _update_dict_recursive(target_dict: Dict[str, Any], updates: Dict[str, Any]) -> None:
        for sub_key, sub_value in updates.items():
            if isinstance(sub_value, dict) and isinstance(target_dict.get(sub_key), dict):
                _update_dict_recursive(target_dict[sub_key], sub_value)
            else:
                target_dict[sub_key] = sub_value

    for key, value in overrides.items():
        if not hasattr(target, key):
            raise WandbConfigError(
                f"Cannot apply override '{key}' - attribute not found on '{type(target).__name__}'"
            )
        attr = getattr(target, key)

        if isinstance(value, dict) and value and all(isinstance(k, int) for k in value.keys()):
            if not isinstance(attr, list):
                raise WandbConfigError(
                    f"Override targets list indices for '{key}' but attribute is {type(attr).__name__}"
                )
            for index, sub_value in value.items():
                if index >= len(attr):
                    raise WandbConfigError(
                        f"List index {index} out of range for '{key}' (length {len(attr)})"
                    )
                if isinstance(sub_value, dict):
                    element = attr[index]
                    if hasattr(element, "__dict__"):
                        apply_overrides(element, sub_value)
                    elif isinstance(element, dict):
                        _update_dict_recursive(element, sub_value)
                    else:
                        attr[index] = sub_value
                else:
                    attr[index] = sub_value
            continue

        if isinstance(value, dict):
            if hasattr(attr, "__dict__"):
                apply_overrides(attr, value)
            elif isinstance(attr, dict):
                _update_dict_recursive(attr, value)
            else:
                for sub_key, sub_val in value.items():
                    setattr(attr, sub_key, sub_val)
            continue

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

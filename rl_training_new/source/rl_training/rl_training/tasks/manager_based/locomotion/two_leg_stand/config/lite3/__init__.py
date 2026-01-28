# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
#
# Lite3 two-leg standing task registration.

import gymnasium as gym

from . import agents
from .base_env_cfg import (
    Lite3TwoLegStandEnvCfg,
    Lite3TwoLegStandStillEnvCfg,
    Lite3TwoLegStandStillV2EnvCfg,
    Lite3TwoLegStandSafeEnvCfg,
    Lite3TwoLegStandDeployAlignedEnvCfg,
    Lite3TwoLegStandDeployR1EnvCfg,
)

##
# Register Gym environments
##

# Basic two-leg standing
gym.register(
    id="TwoLegStand-Deeprobotics-Lite3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.base_env_cfg:Lite3TwoLegStandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Lite3TwoLegStandPPORunnerCfg",
    },
)

# Two-leg standing with stillness focus
gym.register(
    id="TwoLegStandStill-Deeprobotics-Lite3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.base_env_cfg:Lite3TwoLegStandStillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Lite3TwoLegStandStillPPORunnerCfg",
    },
)

# Two-leg standing with extended stillness curriculum (v2)
gym.register(
    id="TwoLegStandStillV2-Deeprobotics-Lite3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.base_env_cfg:Lite3TwoLegStandStillV2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Lite3TwoLegStandStillV2PPORunnerCfg",
    },
)

# Two-leg standing with safety constraints
gym.register(
    id="TwoLegStandSafe-Deeprobotics-Lite3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.base_env_cfg:Lite3TwoLegStandSafeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Lite3TwoLegStandSafePPORunnerCfg",
    },
)

# Two-leg standing aligned with deployment
gym.register(
    id="TwoLegStandDeployAligned-Deeprobotics-Lite3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.base_env_cfg:Lite3TwoLegStandDeployAlignedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Lite3TwoLegStandDeployAlignedPPORunnerCfg",
    },
)

# Two-leg standing deploy/r1 (matches Lite3_rl_training log config)
gym.register(
    id="TwoLegStandDeployR1-Deeprobotics-Lite3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.base_env_cfg:Lite3TwoLegStandDeployR1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Lite3TwoLegStandDeployR1PPORunnerCfg",
    },
)

##
# Legacy task IDs (Lite3_rl_training compatibility)
##

gym.register(
    id="lite3_two_leg_stand",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.base_env_cfg:Lite3TwoLegStandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Lite3TwoLegStandPPORunnerCfg",
    },
)

gym.register(
    id="lite3_two_leg_stand_still",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.base_env_cfg:Lite3TwoLegStandStillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Lite3TwoLegStandStillPPORunnerCfg",
    },
)

gym.register(
    id="lite3_two_leg_stand_still_v2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.base_env_cfg:Lite3TwoLegStandStillV2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Lite3TwoLegStandStillV2PPORunnerCfg",
    },
)

gym.register(
    id="lite3_two_leg_stand_still_safe",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.base_env_cfg:Lite3TwoLegStandSafeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Lite3TwoLegStandSafePPORunnerCfg",
    },
)

gym.register(
    id="lite3_two_leg_stand_deploy_aligned",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.base_env_cfg:Lite3TwoLegStandDeployAlignedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Lite3TwoLegStandDeployAlignedPPORunnerCfg",
    },
)

gym.register(
    id="lite3_two_leg_stand_deploy_r1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.base_env_cfg:Lite3TwoLegStandDeployR1EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Lite3TwoLegStandDeployR1PPORunnerCfg",
    },
)

gym.register(
    id="two_leg_stand_deploy_aligned",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.base_env_cfg:Lite3TwoLegStandDeployAlignedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Lite3TwoLegStandDeployAlignedPPORunnerCfg",
    },
)

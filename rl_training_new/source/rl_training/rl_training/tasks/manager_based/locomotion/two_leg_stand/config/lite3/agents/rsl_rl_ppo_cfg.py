# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
#
# PPO runner configurations for Lite3 two-leg standing task.

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class Lite3TwoLegStandPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for basic two-leg standing."""

    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 500
    experiment_name = "two_leg_stand"
    empirical_normalization = False
    clip_actions = 12.0

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Lite3TwoLegStandStillPPORunnerCfg(Lite3TwoLegStandPPORunnerCfg):
    """PPO runner configuration for stillness-focused two-leg standing."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "two_leg_stand_still"


@configclass
class Lite3TwoLegStandStillV2PPORunnerCfg(Lite3TwoLegStandStillPPORunnerCfg):
    """PPO runner configuration for stillness v2 two-leg standing."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "two_leg_stand_still_v2"


@configclass
class Lite3TwoLegStandSafePPORunnerCfg(Lite3TwoLegStandStillV2PPORunnerCfg):
    """PPO runner configuration for safety-focused two-leg standing."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "two_leg_stand_still_safe"


@configclass
class Lite3TwoLegStandDeployAlignedPPORunnerCfg(Lite3TwoLegStandSafePPORunnerCfg):
    """PPO runner configuration for deployment-aligned two-leg standing."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "two_leg_stand_deploy_aligned"

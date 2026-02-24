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
class Lite3TwoLegStandSafeSlowLowPowerPPORunnerCfg(Lite3TwoLegStandSafePPORunnerCfg):
    """PPO runner configuration for isolated safe/slow/low-power training."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "two_leg_stand_safe_slow_low_power"


@configclass
class Lite3TwoLegStandSafeSlowLowPowerDomainRandPPORunnerCfg(Lite3TwoLegStandSafeSlowLowPowerPPORunnerCfg):
    """PPO runner for safe/slow/low-power with domain randomization."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "two_leg_stand_safe_slow_low_power_domain_rand"
        self.max_iterations = 20000


@configclass
class Lite3TwoLegStandDeployAlignedPPORunnerCfg(Lite3TwoLegStandSafePPORunnerCfg):
    """PPO runner configuration for deployment-aligned two-leg standing."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "two_leg_stand_deploy_aligned"


@configclass
class Lite3TwoLegStandDeployR1PPORunnerCfg(Lite3TwoLegStandDeployAlignedPPORunnerCfg):
    """PPO runner configuration matching Lite3_rl_training deploy/r1."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "deploy"
        self.run_name = "r1"
        self.num_steps_per_env = 24
        self.max_iterations = 15000
        self.save_interval = 500
        self.seed = 2024
        # Mirror legacy resume settings (can be overridden from CLI).
        self.resume = False
        self.load_run = "r1"
        self.load_checkpoint = "model_14500.pt"
        # Explicitly mirror adaptation network config
        self.policy.adaptation_hidden_dims = [256, 32]
        self.policy.encoder_latent_dims = 18


@configclass
class Lite3TwoLegStandDeployR12MimicPPORunnerCfg(Lite3TwoLegStandDeployR1PPORunnerCfg):
    """PPO runner for isolated deploy/r1-r2 mimic experiments."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "deploy_r12_mimic"
        # Keep task isolated from historical r1 run naming by default.
        self.run_name = ""
        self.resume = False


@configclass
class Lite3TwoLegStandDeploySafeV2PPORunnerCfg(Lite3TwoLegStandDeployR1PPORunnerCfg):
    """PPO runner for deploy-safe-v2 curriculum."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "two_leg_stand_deploy_safe_v2"
        self.run_name = ""


@configclass
class Lite3TwoLegStandRobustPPORunnerCfg(Lite3TwoLegStandSafePPORunnerCfg):
    """PPO runner for robust two-leg stand domain randomization."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "two_leg_stand_robust"
        self.max_iterations = 20000

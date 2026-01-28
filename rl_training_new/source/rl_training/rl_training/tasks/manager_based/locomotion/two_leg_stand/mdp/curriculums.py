# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
#
# Multi-phase curriculum controller for two-leg standing task.

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@dataclass
class CurriculumPhase:
    """Configuration for a single curriculum phase."""

    name: str
    """Name of the phase for logging."""

    trigger_thresh: int
    """Training iteration threshold to enter this phase."""

    near_goal_init_prob: float = 0.0
    """Probability of resetting to near-goal state in this phase."""

    reward_scales: dict[str, float] = field(default_factory=dict)
    """Reward scale overrides for this phase."""


class TwoLegStandCurriculumManager:
    """Multi-phase curriculum manager for two-leg standing task.

    This manager tracks training progress and adjusts reward scales and
    reset probabilities based on predefined phases. It mirrors the curriculum
    system from the original Lite3_rl_training implementation.

    Example phases:
        Phase 0: Legs up warmup (0-500 iterations)
        Phase 1: Basic stability (500-1000 iterations)
        Phase 2: Posture alignment (1000-2500 iterations)
        Phase 3: Fine standing (2500+ iterations)
    """

    def __init__(
        self,
        phases: list[CurriculumPhase],
        base_reward_scales: dict[str, float] = None,
        log_curriculum: bool = True,
        merge_base_scales: bool = True,
    ):
        """Initialize the curriculum manager.

        Args:
            phases: List of curriculum phases in order.
            base_reward_scales: Default reward scales to use when phase doesn't override.
            log_curriculum: Whether to log phase transitions.
        """
        self.phases = phases
        self.base_reward_scales = base_reward_scales or {}
        self.log_curriculum = log_curriculum
        self.merge_base_scales = merge_base_scales

        self.current_phase_idx = 0
        self.current_iteration = 0

        # Cache current phase's settings
        self._update_current_phase()

    def _update_current_phase(self) -> None:
        """Update cached values from current phase."""
        phase = self.phases[self.current_phase_idx]
        self.current_phase_name = phase.name
        self.current_near_goal_prob = phase.near_goal_init_prob

        # Merge base scales with phase overrides if enabled
        if self.merge_base_scales:
            self.current_reward_scales = self.base_reward_scales.copy()
            self.current_reward_scales.update(phase.reward_scales)
        else:
            self.current_reward_scales = phase.reward_scales.copy()

    def update(self, iteration: int) -> bool:
        """Update curriculum based on training iteration.

        Args:
            iteration: Current training iteration.

        Returns:
            True if phase changed, False otherwise.
        """
        self.current_iteration = iteration
        phase_changed = False

        # Check for phase transition
        while (
            self.current_phase_idx < len(self.phases) - 1
            and iteration >= self.phases[self.current_phase_idx + 1].trigger_thresh
        ):
            self.current_phase_idx += 1
            self._update_current_phase()
            phase_changed = True

            if self.log_curriculum:
                print(f"[Curriculum] Entering phase {self.current_phase_idx}: {self.current_phase_name}")

        return phase_changed

    def get_reward_scale(self, reward_name: str) -> float:
        """Get the current reward scale for a given reward term.

        Args:
            reward_name: Name of the reward term.

        Returns:
            Current scale value (0.0 if not defined).
        """
        return self.current_reward_scales.get(reward_name, 0.0)

    def get_near_goal_prob(self) -> float:
        """Get current near-goal initialization probability."""
        return self.current_near_goal_prob

    def get_current_phase_name(self) -> str:
        """Get name of current phase."""
        return self.current_phase_name

    def get_state_dict(self) -> dict[str, Any]:
        """Get state for checkpointing."""
        return {
            "current_phase_idx": self.current_phase_idx,
            "current_iteration": self.current_iteration,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state from checkpoint."""
        self.current_phase_idx = state_dict["current_phase_idx"]
        self.current_iteration = state_dict["current_iteration"]
        self._update_current_phase()


class LegacyCurriculumShim:
    """Shim to expose curriculum state in legacy legged_gym-style API."""

    def __init__(self, phases: list[CurriculumPhase]) -> None:
        self.enabled = True
        self.phases = [
            {
                "name": p.name,
                "trigger_thresh": p.trigger_thresh,
                "near_goal_init_prob": p.near_goal_init_prob,
                "reward_scales": p.reward_scales,
            }
            for p in phases
        ]
        self.current_phase = 0
        self.current_scales: dict[str, float] = {}
        self.progress_buf = 0

    def get_progress_buf(self, buf_element):
        # Stored for compatibility; progression handled by IsaacLab curriculum.
        self.progress_buf = buf_element


# =============================================================================
# Default Phase Configurations
# =============================================================================

def get_two_leg_stand_phases() -> list[CurriculumPhase]:
    """Get default curriculum phases for basic two-leg standing."""
    return [
        CurriculumPhase(
            name="phase_0_legs_up_warmup",
            trigger_thresh=0,
            near_goal_init_prob=0.0,
            reward_scales={
                "front_legs_up_warmup": 18.0,
                "torso_upright_warmup": 8.0,
                "base_height_bonus": 6.0,
                "front_tap_penalty": 0.0,
                "termination": -10.0,
            },
        ),
        CurriculumPhase(
            name="phase_0_basic",
            trigger_thresh=500,
            near_goal_init_prob=0.0,
            reward_scales={
                "front_legs_up_warmup": 14.0,
                "torso_upright_warmup": 10.0,
                "base_height_bonus": 8.0,
                "stand_still": 0.0,
                "front_tap_penalty": -1.0,
                "termination": -10.0,
            },
        ),
        CurriculumPhase(
            name="phase_1_posture_alignment",
            trigger_thresh=1000,
            near_goal_init_prob=0.45,
            reward_scales={
                "front_legs_up_warmup": 14.0,
                "torso_upright_warmup": 10.0,
                "base_height_bonus": 8.0,
                "stand_still_roll_only": 1.0,
                "hind_legs_calmness": 1.0,
                "front_tap_penalty": -1.0,
                "termination": -10.0,
            },
        ),
        CurriculumPhase(
            name="phase_2_fine_standing_roll_supression",
            trigger_thresh=2500,
            near_goal_init_prob=0.7,
            reward_scales={
                "front_legs_up_warmup": 14.0,
                "torso_upright_warmup": 10.0,
                "base_height_bonus": 8.0,
                "stand_still_roll_only": 10.0,
                "front_tap_penalty": -1.0,
                "termination": -10.0,
            },
        ),
    ]


def get_two_leg_stand_still_phases() -> list[CurriculumPhase]:
    """Get curriculum phases for stillness-focused two-leg standing."""
    return [
        CurriculumPhase(
            name="phase_0_legs_up_warmup",
            trigger_thresh=0,
            near_goal_init_prob=0.0,
            reward_scales={
                "front_legs_up_warmup": 18.0,
                "torso_upright_warmup": 8.0,
                "base_height_bonus": 6.0,
                "front_tap_penalty": -0.5,
                "deploy_posture_gate": -5.0,
                "termination": -10.0,
            },
        ),
        CurriculumPhase(
            name="phase_1_basic_stability",
            trigger_thresh=500,
            near_goal_init_prob=0.25,
            reward_scales={
                "front_legs_up_warmup": 14.0,
                "torso_upright_soften": 8.0,
                "base_height_bonus": 8.0,
                "hind_leg_extension_geom": 2.0,
                "hind_legs_calmness": 2.0,
                "stand_still": 2.0,
                "stand_still_yaw_only": 1.5,
                "stand_still_roll_only": 0.5,
                "feet_velocity": -0.2,
                "action_rate": -0.01,
                "front_tap_penalty": -1.5,
                "deploy_posture_gate": -5.0,
                "termination": -10.0,
            },
        ),
        CurriculumPhase(
            name="phase_2_reduce_spin",
            trigger_thresh=3000,
            near_goal_init_prob=0.6,
            reward_scales={
                "front_legs_up_continuous": 6.0,
                "torso_upright_continuous": 8.0,
                "human_posture_warmup": 4.0,
                "hind_leg_extension_geom": 3.0,
                "hind_legs_calmness": 3.0,
                "stand_still": 4.0,
                "stand_still_yaw_only": 3.0,
                "feet_velocity": -0.4,
                "action_rate": -0.02,
                "front_tap_penalty": -2.0,
                "base_height_bonus": 6.0,
                "deploy_posture_gate": -5.0,
                "termination": -10.0,
            },
        ),
        CurriculumPhase(
            name="phase_3_fine_still_stand",
            trigger_thresh=5000,
            near_goal_init_prob=0.75,
            reward_scales={
                "front_legs_up_continuous": 8.0,
                "torso_upright_continuous": 10.0,
                "human_posture": 6.0,
                "hind_leg_extension_geom": 4.0,
                "hind_legs_calmness": 4.0,
                "stand_still": 6.0,
                "stand_still_yaw_only": 5.0,
                "stand_still_roll_only": 1.0,
                "feet_velocity": -0.6,
                "action_rate": -0.03,
                "front_tap_penalty": -3.0,
                "base_height_bonus": 6.0,
                "deploy_posture_gate": -5.0,
                "termination": -10.0,
            },
        ),
    ]


def get_two_leg_stand_still_v2_phases() -> list[CurriculumPhase]:
    """Get curriculum phases for stillness v2 (longer exploration)."""
    return [
        CurriculumPhase(
            name="phase_0_legs_up_warmup",
            trigger_thresh=0,
            near_goal_init_prob=0.0,
            reward_scales={
                "front_legs_up_warmup": 18.0,
                "torso_upright_warmup": 8.0,
                "base_height_bonus": 6.0,
                "hind_leg_extension_geom": 1.0,
                "hind_legs_calmness": 0.5,
                "stand_still_yaw_only": 0.3,
                "stand_still_roll_only": 0.2,
                "stand_still_lin_x": 0.2,
                "stand_still_lin_y": 0.2,
                "front_tap_penalty": -0.3,
                "deploy_posture_gate": -5.0,
                "termination": -10.0,
            },
        ),
        CurriculumPhase(
            name="phase_1_explore_stable_two_leg",
            trigger_thresh=500,
            near_goal_init_prob=0.2,
            reward_scales={
                "front_legs_up_warmup": 14.0,
                "torso_upright_soften": 8.0,
                "base_height_bonus": 8.0,
                "hind_leg_extension_geom": 3.0,
                "hind_legs_calmness": 1.5,
                "stand_still_yaw_only": 0.8,
                "stand_still_roll_only": 0.4,
                "stand_still_lin_x": 0.4,
                "stand_still_lin_y": 0.4,
                "stand_still_lin_z": 0.2,
                "feet_velocity": -0.15,
                "action_rate": -0.01,
                "front_tap_penalty": -1.0,
                "deploy_posture_gate": -5.0,
                "termination": -10.0,
            },
        ),
        CurriculumPhase(
            name="phase_2_transition_reduce_spin",
            trigger_thresh=3000,
            near_goal_init_prob=0.45,
            reward_scales={
                "front_legs_up_warmup": 10.0,
                "front_legs_up_continuous": 4.0,
                "torso_upright_soften": 6.0,
                "torso_upright_continuous": 5.0,
                "base_height_bonus": 7.0,
                "hind_leg_extension_geom": 4.0,
                "human_posture_warmup": 2.5,
                "hind_legs_calmness": 2.0,
                "stand_still_yaw_only": 1.5,
                "stand_still_roll_only": 0.6,
                "stand_still_lin_x": 0.7,
                "stand_still_lin_y": 0.7,
                "stand_still_lin_z": 0.4,
                "feet_velocity": -0.25,
                "action_rate": -0.02,
                "front_tap_penalty": -1.8,
                "deploy_posture_gate": -5.0,
                "termination": -10.0,
            },
        ),
        CurriculumPhase(
            name="phase_3_refine_still_stand",
            trigger_thresh=10000,
            near_goal_init_prob=0.7,
            reward_scales={
                "front_legs_up_continuous": 7.0,
                "torso_upright_continuous": 9.0,
                "human_posture": 5.0,
                "hind_leg_extension_geom": 5.0,
                "hind_legs_calmness": 3.0,
                "stand_still_yaw_only": 3.5,
                "stand_still_roll_only": 1.0,
                "stand_still_lin_x": 1.2,
                "stand_still_lin_y": 1.2,
                "stand_still_lin_z": 0.8,
                "torques": -0.0002,
                "dof_vel": -0.0002,
                "feet_velocity": -0.65,
                "action_rate": -0.04,
                "front_tap_penalty": -2.8,
                "base_height_bonus": 6.0,
                "deploy_posture_gate": -5.0,
                "termination": -10.0,
            },
        ),
        CurriculumPhase(
            name="phase_4_final_polish",
            trigger_thresh=15000,
            near_goal_init_prob=0.75,
            reward_scales={
                "front_legs_up_continuous": 8.0,
                "torso_upright_continuous": 10.0,
                "human_posture": 6.0,
                "hind_leg_extension_geom": 6.0,
                "hind_legs_calmness": 3.5,
                "stand_still_yaw_only": 4.0,
                "stand_still_roll_only": 1.2,
                "stand_still_lin_x": 1.4,
                "stand_still_lin_y": 1.4,
                "stand_still_lin_z": 1.0,
                "torques": -0.0003,
                "dof_vel": -0.0003,
                "feet_velocity": -0.75,
                "action_rate": -0.05,
                "front_tap_penalty": -3.2,
                "base_height_bonus": 6.0,
                "deploy_posture_gate": -5.0,
                "termination": -10.0,
            },
        ),
    ]


def get_two_leg_stand_safe_phases() -> list[CurriculumPhase]:
    """Get curriculum phases for safety-focused two-leg standing."""
    return [
        CurriculumPhase(
            name="phase_0_legs_up_safe_warmup",
            trigger_thresh=0,
            near_goal_init_prob=0.0,
            reward_scales={
                "front_legs_up_warmup_safe": 18.0,
                "torso_upright_warmup": 8.0,
                "base_height_bonus": 6.0,
                "hind_leg_extension_geom": 1.0,
                "hind_legs_calmness": 0.5,
                "stand_still_yaw_only": 0.3,
                "stand_still_roll_only": 0.2,
                "stand_still_lin_x": 0.2,
                "stand_still_lin_y": 0.2,
                "front_tap_penalty": -0.3,
                "deploy_posture_gate": -5.0,
                # Safety shaping
                "lin_vel_z": -0.1,
                "ang_vel_xy": -0.05,
                "feet_velocity": -0.15,
                "action_rate": -0.02,
                "target_smoothness": -0.002,
                "dof_acc": -2.0e-6,
                "power": -0.00002,
                "torque_limits": -0.05,
                "dof_vel_limits": -0.05,
                "termination": -10.0,
            },
        ),
        CurriculumPhase(
            name="phase_1_explore_stable_two_leg_safe",
            trigger_thresh=500,
            near_goal_init_prob=0.15,
            reward_scales={
                "front_legs_up_warmup_safe": 12.0,
                "front_legs_up_continuous_safe": 6.0,
                "torso_upright_soften": 8.0,
                "torso_upright_continuous": 5.0,
                "base_height_bonus": 8.0,
                "hind_leg_extension_geom": 3.0,
                "hind_legs_calmness": 1.5,
                "stand_still_yaw_only": 0.8,
                "stand_still_roll_only": 0.4,
                "stand_still_lin_x": 0.4,
                "stand_still_lin_y": 0.4,
                "stand_still_lin_z": 0.2,
                "two_leg_stability_safe": 1.5,
                "lin_vel_z": -0.12,
                "ang_vel_xy": -0.06,
                "feet_velocity": -0.22,
                "front_tap_penalty": -1.0,
                "deploy_posture_gate": -5.0,
                # Safety shaping (medium)
                "action_rate": -0.03,
                "target_smoothness": -0.004,
                "dof_acc": -3.0e-6,
                "power": -0.00004,
                "torque_limits": -0.1,
                "dof_vel_limits": -0.1,
                "termination": -10.0,
            },
        ),
        CurriculumPhase(
            name="phase_2_transition_reduce_spin_safe",
            trigger_thresh=3000,
            near_goal_init_prob=0.25,
            reward_scales={
                "front_legs_up_warmup_safe": 8.0,
                "front_legs_up_continuous_safe": 7.0,
                "torso_upright_soften": 6.0,
                "torso_upright_continuous": 5.0,
                "base_height_bonus": 7.0,
                "hind_leg_extension_geom": 4.0,
                "human_posture_warmup": 2.5,
                "hind_legs_calmness": 2.0,
                "stand_still_yaw_only": 1.5,
                "stand_still_roll_only": 0.6,
                "stand_still_lin_x": 0.7,
                "stand_still_lin_y": 0.7,
                "stand_still_lin_z": 0.4,
                "two_leg_stability_safe": 3.0,
                "lin_vel_z": -0.15,
                "ang_vel_xy": -0.08,
                "feet_velocity": -0.3,
                "front_tap_penalty": -1.8,
                "deploy_posture_gate": -5.0,
                # Safety shaping (strong)
                "action_rate": -0.04,
                "target_smoothness": -0.01,
                "dof_acc": -4.0e-6,
                "power": -0.00006,
                "torque_limits": -0.15,
                "dof_vel_limits": -0.15,
                "termination": -10.0,
            },
        ),
        CurriculumPhase(
            name="phase_3_refine_still_stand_safe",
            trigger_thresh=10000,
            near_goal_init_prob=0.3,
            reward_scales={
                "front_legs_up_continuous_safe": 8.0,
                "torso_upright_continuous": 9.0,
                "human_posture": 5.0,
                "hind_leg_extension_geom": 5.0,
                "hind_legs_calmness": 3.0,
                "stand_still_yaw_only": 3.5,
                "stand_still_roll_only": 1.0,
                "stand_still_lin_x": 1.2,
                "stand_still_lin_y": 1.2,
                "stand_still_lin_z": 0.8,
                "two_leg_stability_safe": 5.0,
                "lin_vel_z": -0.18,
                "ang_vel_xy": -0.12,
                "feet_velocity": -1.5,
                "front_tap_penalty": -4.0,
                "base_height_bonus": 6.0,
                "deploy_posture_gate": -5.0,
                # Safety shaping (very strong)
                "action_rate": -0.08,
                "target_smoothness": -0.04,
                "dof_vel": -0.002,
                "dof_acc": -2.0e-5,
                "power": -0.0002,
                "torque_limits": -0.35,
                "dof_vel_limits": -0.5,
                "feet_contact_forces": -0.05,
                "termination": -10.0,
            },
        ),
        CurriculumPhase(
            name="phase_4_final_polish_safe",
            trigger_thresh=15000,
            near_goal_init_prob=0.35,
            reward_scales={
                "front_legs_up_continuous_safe": 8.0,
                "torso_upright_continuous": 10.0,
                "human_posture": 6.0,
                "hind_leg_extension_geom": 6.0,
                "hind_legs_calmness": 3.5,
                "stand_still_yaw_only": 4.0,
                "stand_still_roll_only": 1.2,
                "stand_still_lin_x": 1.4,
                "stand_still_lin_y": 1.4,
                "stand_still_lin_z": 1.0,
                "two_leg_stability_safe": 6.0,
                "lin_vel_z": -0.2,
                "ang_vel_xy": -0.15,
                "feet_velocity": -1.8,
                "front_tap_penalty": -5.0,
                "base_height_bonus": 6.0,
                "deploy_posture_gate": -5.0,
                # Safety shaping (max)
                "action_rate": -0.1,
                "target_smoothness": -0.06,
                "dof_vel": -0.003,
                "dof_acc": -3.0e-5,
                "power": -0.0003,
                "torque_limits": -0.45,
                "dof_vel_limits": -0.6,
                "feet_contact_forces": -0.08,
                "termination": -10.0,
            },
        ),
    ]


def get_two_leg_stand_deploy_r1_phases() -> list[CurriculumPhase]:
    """Get curriculum phases matching Lite3_rl_training deploy/r1 run."""
    return [
        CurriculumPhase(
            name="phase_0_legs_up_safe_warmup",
            trigger_thresh=0,
            near_goal_init_prob=0.0,
            reward_scales={
                "front_legs_up_warmup_safe": 18.0,
                "torso_upright_warmup": 8.0,
                "base_height_bonus": 6.0,
                "hind_leg_extension_geom": 1.0,
                "hind_legs_calmness": 0.5,
                "stand_still_yaw_only": 0.3,
                "stand_still_roll_only": 0.2,
                "stand_still_lin_x": 0.2,
                "stand_still_lin_y": 0.2,
                "front_tap_penalty": -0.3,
                "deploy_posture_gate": -5.0,
                "lin_vel_z": -0.1,
                "ang_vel_xy": -0.05,
                "feet_velocity": -0.15,
                "action_rate": -0.02,
                "target_smoothness": -0.002,
                "dof_acc": -2.0e-6,
                "power": -2.0e-5,
                "torque_limits": -0.05,
                "dof_vel_limits": -0.05,
                "termination": -10.0,
            },
        ),
        CurriculumPhase(
            name="phase_1_explore_stable_two_leg_safe",
            trigger_thresh=500,
            near_goal_init_prob=0.15,
            reward_scales={
                "front_legs_up_warmup_safe": 12.0,
                "front_legs_up_continuous_safe": 6.0,
                "torso_upright_soften": 8.0,
                "torso_upright_continuous": 5.0,
                "base_height_bonus": 8.0,
                "hind_leg_extension_geom": 3.0,
                "hind_legs_calmness": 1.5,
                "stand_still_yaw_only": 0.8,
                "stand_still_roll_only": 0.4,
                "stand_still_lin_x": 0.4,
                "stand_still_lin_y": 0.4,
                "stand_still_lin_z": 0.2,
                "two_leg_stability_safe": 1.5,
                "lin_vel_z": -0.12,
                "ang_vel_xy": -0.06,
                "feet_velocity": -0.22,
                "front_tap_penalty": -1.0,
                "deploy_posture_gate": -5.0,
                "action_rate": -0.03,
                "target_smoothness": -0.004,
                "dof_acc": -3.0e-6,
                "power": -4.0e-5,
                "torque_limits": -0.1,
                "dof_vel_limits": -0.1,
                "termination": -10.0,
            },
        ),
        CurriculumPhase(
            name="phase_2_transition_reduce_spin_safe",
            trigger_thresh=3000,
            near_goal_init_prob=0.25,
            reward_scales={
                "front_legs_up_warmup_safe": 8.0,
                "front_legs_up_continuous_safe": 7.0,
                "torso_upright_soften": 6.0,
                "torso_upright_continuous": 5.0,
                "base_height_bonus": 7.0,
                "hind_leg_extension_geom": 4.0,
                "human_posture_warmup": 2.5,
                "hind_legs_calmness": 2.0,
                "stand_still_yaw_only": 1.5,
                "stand_still_roll_only": 0.6,
                "stand_still_lin_x": 0.7,
                "stand_still_lin_y": 0.7,
                "stand_still_lin_z": 0.4,
                "two_leg_stability_safe": 3.0,
                "lin_vel_z": -0.15,
                "ang_vel_xy": -0.08,
                "feet_velocity": -0.3,
                "front_tap_penalty": -1.8,
                "deploy_posture_gate": -5.0,
                "action_rate": -0.04,
                "target_smoothness": -0.01,
                "dof_acc": -4.0e-6,
                "power": -6.0e-5,
                "torque_limits": -0.15,
                "dof_vel_limits": -0.15,
                "termination": -10.0,
            },
        ),
        CurriculumPhase(
            name="phase_3_refine_still_stand_safe",
            trigger_thresh=10000,
            near_goal_init_prob=0.3,
            reward_scales={
                "front_legs_up_continuous_safe": 8.0,
                "torso_upright_continuous": 9.0,
                "human_posture": 5.0,
                "hind_leg_extension_geom": 5.0,
                "hind_legs_calmness": 3.0,
                "stand_still_yaw_only": 3.5,
                "stand_still_roll_only": 1.0,
                "stand_still_lin_x": 1.2,
                "stand_still_lin_y": 1.2,
                "stand_still_lin_z": 0.8,
                "two_leg_stability_safe": 5.0,
                "lin_vel_z": -0.18,
                "ang_vel_xy": -0.12,
                "feet_velocity": -1.5,
                "front_tap_penalty": -4.0,
                "base_height_bonus": 6.0,
                "deploy_posture_gate": -5.0,
                "action_rate": -0.08,
                "target_smoothness": -0.04,
                "dof_vel": -0.002,
                "dof_acc": -2.0e-5,
                "power": -0.0002,
                "torque_limits": -0.35,
                "dof_vel_limits": -0.5,
                "feet_contact_forces": -0.05,
                "termination": -10.0,
            },
        ),
        CurriculumPhase(
            name="phase_4_final_polish_safe",
            trigger_thresh=15000,
            near_goal_init_prob=0.35,
            reward_scales={
                "front_legs_up_continuous_safe": 8.0,
                "torso_upright_continuous": 10.0,
                "human_posture": 6.0,
                "hind_leg_extension_geom": 6.0,
                "hind_legs_calmness": 3.5,
                "stand_still_yaw_only": 4.0,
                "stand_still_roll_only": 1.2,
                "stand_still_lin_x": 1.4,
                "stand_still_lin_y": 1.4,
                "stand_still_lin_z": 1.0,
                "two_leg_stability_safe": 6.0,
                "lin_vel_z": -0.2,
                "ang_vel_xy": -0.15,
                "feet_velocity": -1.8,
                "front_tap_penalty": -5.0,
                "base_height_bonus": 6.0,
                "deploy_posture_gate": -5.0,
                "action_rate": -0.1,
                "target_smoothness": -0.06,
                "dof_vel": -0.003,
                "dof_acc": -3.0e-5,
                "power": -0.0003,
                "torque_limits": -0.45,
                "dof_vel_limits": -0.6,
                "feet_contact_forces": -0.08,
                "termination": -10.0,
            },
        ),
    ]


def two_leg_stand_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids,
    phases: list[CurriculumPhase],
    steps_per_env: int = 24,
    log_curriculum: bool = True,
    front_touch_termination: dict | None = None,
) -> None:
    """Update reward scales and near-goal reset probability based on training iteration."""
    if not hasattr(env, "_two_leg_curriculum"):
        env._two_leg_curriculum = TwoLegStandCurriculumManager(
            phases=phases,
            base_reward_scales={},
            log_curriculum=log_curriculum,
            merge_base_scales=False,
        )
        env.front_touch_termination_active = False
        env._two_leg_front_touch_cfg = front_touch_termination or {}
        # Legacy-style curriculum controller for runner logging.
        env.curriculum_controller = LegacyCurriculumShim(phases)

    steps_per_env = max(int(steps_per_env), 1)
    iteration = int(env.common_step_counter // steps_per_env)
    env._two_leg_curriculum.update(iteration)

    current_scales = env._two_leg_curriculum.current_reward_scales
    term_names = getattr(env.reward_manager, "_episode_sums", {}).keys()

    for name in term_names:
        term_cfg = env.reward_manager.get_term_cfg(name)
        if name == "is_terminated":
            if "termination" in current_scales:
                term_cfg.weight = current_scales["termination"]
            continue
        term_cfg.weight = current_scales.get(name, 0.0)

    # Update near-goal reset probability
    near_goal_prob = env._two_leg_curriculum.get_near_goal_prob()
    if hasattr(env, "event_manager"):
        try:
            term_cfg = env.event_manager.get_term_cfg("reset_to_near_goal")
        except Exception:
            term_cfg = None
        if term_cfg is not None:
            term_cfg.params["near_goal_prob"] = near_goal_prob
    # Expose for legacy logger
    try:
        env.goal_state_prob = near_goal_prob
    except Exception:
        pass

    # Update legacy-style curriculum state for logging
    if hasattr(env, "curriculum_controller"):
        env.curriculum_controller.current_phase = env._two_leg_curriculum.current_phase_idx
        env.curriculum_controller.current_scales = current_scales

    # Enable front-touch termination when metrics exceed thresholds
    front_cfg = env._two_leg_front_touch_cfg
    if front_cfg and front_cfg.get("enabled", False) and not env.front_touch_termination_active:
        thresholds = front_cfg.get("metrics", {})
        if isinstance(thresholds, dict):
            items = thresholds.items()
        else:
            items = thresholds.__dict__.items()
        for name, threshold in items:
            if name == "two_leg_stability":
                if "two_leg_stand_metric" not in env.reward_manager._episode_sums:
                    return
                metric_val = (
                    torch.mean(env.reward_manager._episode_sums["two_leg_stand_metric"][env_ids])
                    / env.max_episode_length_s
                )
            elif name in env.reward_manager._episode_sums:
                metric_val = (
                    torch.mean(env.reward_manager._episode_sums[name][env_ids]) / env.max_episode_length_s
                )
            else:
                return
            if metric_val < threshold:
                return
        env.front_touch_termination_active = True
        if front_cfg.get("log_enable", True):
            print("[Curriculum] Front-leg termination enabled based on metrics.")

# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
#
# Two-leg standing task for quadruped robots.

from .two_leg_stand_env_cfg import (
    TwoLegStandEnvCfg,
    TwoLegStandSceneCfg,
    TwoLegStandActionsCfg,
    TwoLegStandCommandsCfg,
    TwoLegStandObservationsCfg,
    TwoLegStandEventCfg,
    TwoLegStandRewardsCfg,
    TwoLegStandTerminationsCfg,
    TwoLegStandCurriculumCfg,
)

from . import mdp
from . import config  # Import config to trigger gym registration

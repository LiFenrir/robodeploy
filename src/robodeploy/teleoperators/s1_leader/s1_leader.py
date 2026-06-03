#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from functools import cached_property
from typing import Any

from robodeploy.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_s1_leader import S1LeaderConfig

logger = logging.getLogger(__name__)


class S1Leader(Teleoperator):
    """
    Single S1 Leader Arm using the Theseus S1 SDK for teleoperation.
    The leader arm is used to record human demonstrations by tracking
    joint positions which are then sent as actions to the follower arm.
    """

    config_class = S1LeaderConfig
    name = "s1_leader"

    # Motor names for S1 arm (6 joints + optional 7th for gripper/teach)
    MOTOR_NAMES = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "gripper",
    ]

    def __init__(self, config: S1LeaderConfig):
        super().__init__(config)
        self.config = config

        # Import S1 SDK
        from S1_SDK.S1_arm import S1_arm, control_mode

        # Create arm instance (leader mode - no collision checking needed)
        self.arm = S1_arm(
            mode=control_mode.only_real,
            dev=config.port,
            end_effector=config.end_effector,
            check_collision=False,
            arm_version=config.arm_version,
        )

        self._is_connected = False

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.MOTOR_NAMES}

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        # S1 leader doesn't support force feedback
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Enable motors for reading
        self.arm.enable()

        self._is_connected = True
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        # S1 leader uses zero position setting
        return True

    def calibrate(self) -> None:
        """Set current position as zero for the leader arm."""
        logger.info(f"Setting zero position for {self}")
        self.arm.set_zero_position()

    def configure(self) -> None:
        """Configure leader arm. S1 SDK handles this internally."""
        pass

    def setup_motors(self) -> None:
        """Set up motor IDs. S1 arms have fixed motor IDs."""
        logger.info("S1 arms have fixed motor IDs, no setup required.")

    def get_action(self) -> dict[str, float]:
        """Get the current joint positions of the leader arm.

        Returns:
            Dictionary with keys like 'joint1.pos', 'gripper.pos', etc.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
            
         
        self.arm.control_teach(0.08)
        self.arm.gravity()
        action_dict = {}

        # Read arm position
        start = time.perf_counter()
        pos = self.arm.get_pos()
        for i, motor in enumerate(self.MOTOR_NAMES):
            action_dict[f"{motor}.pos"] = pos[i]
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read arm action: {dt_ms:.1f}ms")

        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Send feedback to the leader arm (not supported for S1).

        Args:
            feedback: Dictionary of feedback values (ignored for S1 leader)
        """
        # S1 leader doesn't support force feedback
        # This method is required by the abstract base class but is no-op for this implementation
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Disable motors
        self.arm.close()

        self._is_connected = False
        logger.info(f"{self} disconnected.")

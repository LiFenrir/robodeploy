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
from .config_bi_s1_leader import BiS1LeaderConfig

logger = logging.getLogger(__name__)


class BiS1Leader(Teleoperator):
    """
    Bimanual S1 Leader Arms using the Theseus S1 SDK for teleoperation.
    The leader arms are used to record human demonstrations by tracking
    joint positions which are then sent as actions to the follower arms.
    """

    config_class = BiS1LeaderConfig
    name = "bi_s1_leader"

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

    def __init__(self, config: BiS1LeaderConfig):
        super().__init__(config)
        self.config = config

        # Import S1 SDK
        from S1_SDK.S1_arm import S1_arm, control_mode

        # Create left arm instance (leader mode - no collision checking needed)
        self.left_arm = S1_arm(
            mode=control_mode.only_real,
            dev=config.left_arm_port,
            end_effector=config.left_end_effector,
            check_collision=False,
            # arm_version=config.left_arm_version,
        )

        # Create right arm instance
        self.right_arm = S1_arm(
            mode=control_mode.only_real,
            dev=config.right_arm_port,
            end_effector=config.right_end_effector,
            check_collision=False,
            # arm_version=config.right_arm_version,
        )

        self._is_connected = False

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"left_{motor}.pos": float for motor in self.MOTOR_NAMES} | {
            f"right_{motor}.pos": float for motor in self.MOTOR_NAMES
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        # S1 leader doesn't support force feedback
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        """Check if calibration data exists."""
        return True

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Enable motors for reading
        self.right_arm.enable()
        self.left_arm.enable()

        self._is_connected = True
        logger.info(f"{self} connected.")
        # self.calibrate()

    def calibrate(self) -> None:
       """Set current positions as zero for both leader arms."""
       pass

    def configure(self) -> None:
        """Configure leader arms. S1 SDK handles this internally."""
        pass

    def setup_motors(self) -> None:
        """Set up motor IDs. S1 arms have fixed motor IDs."""
        logger.info("S1 arms have fixed motor IDs, no setup required.")

    def get_action(self) -> dict[str, float]:
        """Get the current joint positions of both leader arms.

        Returns:
            Dictionary with keys like 'left_joint1.pos', 'right_gripper.pos', etc.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.left_arm.control_teach(0.08)
        self.left_arm.gravity()
        self.right_arm.control_teach(0.08)
        self.right_arm.gravity()

        action_dict = {}

        # Read left arm position
        start = time.perf_counter()
        left_pos = self.left_arm.get_pos()
        for i, motor in enumerate(self.MOTOR_NAMES):
            action_dict[f"left_{motor}.pos"] = left_pos[i]
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read left arm action: {dt_ms:.1f}ms")

        # Read right arm position
        start = time.perf_counter()
        right_pos = self.right_arm.get_pos()
        
        for i, motor in enumerate(self.MOTOR_NAMES):
            action_dict[f"right_{motor}.pos"] = right_pos[i]
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read right arm action: {dt_ms:.1f}ms")

        return action_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command both leader arms to move to target joint configurations.

        Used for leader-follower alignment (e.g., after zero-reset).
        Mirrors BiS1Follower.send_action with the same sign conventions.

        Args:
            action: Dictionary with keys like 'left_joint1.pos', 'right_gripper.pos', etc.

        Returns:
            The action actually sent to the motors.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        
        right_action = {
            key.removeprefix("right_"): action[key]
            for key in action
            if key.startswith("right_")
        }

        right_pos = [right_action.get(f"{motor}.pos", 0.0) for motor in self.MOTOR_NAMES]

        left_gripper_pos = left_pos[-1]
        left_joint_pos = left_pos[:6]
        right_gripper_pos = right_pos[-1]
        right_joint_pos = right_pos[:6]

        self.right_arm.joint_control_mit(right_joint_pos)
        self.right_arm.control_gripper(right_gripper_pos, 0.3)

        sent_action = {}
        for i, motor in enumerate(self.MOTOR_NAMES):
            sent_action[f"left_{motor}.pos"] = left_pos[i]
            sent_action[f"right_{motor}.pos"] = right_pos[i]

        return sent_action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """Send feedback to the leader arms (not supported for S1).

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
    
        self.right_arm.close()
        self.left_arm.close()

        self._is_connected = False
        logger.info(f"{self} disconnected.")

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
# See the the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from functools import cached_property
from typing import Any

import numpy as np

from robodeploy.cameras.utils import make_cameras_from_configs
from robodeploy.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_bi_s1_follower import BiS1FollowerConfig

logger = logging.getLogger(__name__)


class BiS1Follower(Robot):
    """
    Bimanual S1 Follower Arms using the Theseus S1 SDK.
    Each arm has 6 joints + optional gripper (7 DOF total per arm).
    """

    config_class = BiS1FollowerConfig
    name = "bi_s1_follower"

    # Motor names for S1 arm (6 joints + gripper)
    MOTOR_NAMES = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "gripper",
    ]

    def __init__(self, config: BiS1FollowerConfig):
        super().__init__(config)
        self.config = config

        # Import S1 SDK
        from S1_SDK.S1_arm import S1_arm, control_mode

        # Create left arm instance
        self.left_arm = S1_arm(
            mode=control_mode.only_real,
            dev=config.left_arm_port,
            end_effector=config.left_end_effector,
            check_collision=config.check_collision,
            arm_version=config.left_arm_version,
        )

        # Create right arm instance
        self.right_arm = S1_arm(
            mode=control_mode.only_real,
            dev=config.right_arm_port,
            end_effector=config.right_end_effector,
            check_collision=config.check_collision,
            arm_version=config.right_arm_version,
        )

        self.cameras = make_cameras_from_configs(config.cameras)

        # Flag to track connection state
        self._is_connected = False

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"left_{motor}.pos": float for motor in self.MOTOR_NAMES} | {
            f"right_{motor}.pos": float for motor in self.MOTOR_NAMES
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Enable motors
        self.left_arm.enable()
        self.right_arm.enable()

        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()

        self._is_connected = True
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """S1 follower uses zero position setting."""
        return True

    def calibrate(self) -> None:
        """Set current position as zero for both arms."""
        logger.info(f"Setting zero position for {self}")
        self.left_arm.set_zero_position()
        self.right_arm.set_zero_position()

    def configure(self) -> None:
        """Configure motors. S1 SDK handles this internally."""
        pass

    def setup_motors(self) -> None:
        """Set up motor IDs. S1 arms have fixed motor IDs."""
        logger.info("S1 arms have fixed motor IDs, no setup required.")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}

        # Read left arm position
        start = time.perf_counter()
        left_pos = self.left_arm.get_pos()
        for i, motor in enumerate(self.MOTOR_NAMES):
            obs_dict[f"left_{motor}.pos"] = left_pos[i]
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read left arm: {dt_ms:.1f}ms")

        # Read right arm position
        start = time.perf_counter()
        right_pos = self.right_arm.get_pos()
        for i, motor in enumerate(self.MOTOR_NAMES):
            obs_dict[f"right_{motor}.pos"] = right_pos[i]
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read right arm: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: np.ndarray) -> np.ndarray:
        """Command both arms to move to target joint configurations.

        Args:
            action: Numpy array of 14 joint positions [left_7 + right_7]

        Returns:
            The action actually sent to the motors.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Ensure action is numpy array
        action = np.asarray(action, dtype=np.float32)

        # Split into left and right arms (each 7 values)
        left_action = action[:7]
        right_action = action[7:]

        # Separate gripper control (last element)
        left_joint_pos = left_action[:6].tolist()
        left_gripper_pos = float(left_action[6])
        right_joint_pos = right_action[:6].tolist()
        right_gripper_pos = float(right_action[6])

        # Send commands to each arm
        self.left_arm.joint_control_mit(left_joint_pos)
        self.left_arm.control_gripper(left_gripper_pos, 0.3)
        self.right_arm.joint_control_mit(right_joint_pos)
        self.right_arm.control_gripper(right_gripper_pos, 0.3)

        return action

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Close motors
        self.left_arm.close()
        self.right_arm.close()

        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()

        self._is_connected = False
        logger.info(f"{self} disconnected.")

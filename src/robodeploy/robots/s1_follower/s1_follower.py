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

import numpy as np

from robodeploy.cameras.utils import make_cameras_from_configs
from robodeploy.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_s1_follower import S1FollowerConfig

logger = logging.getLogger(__name__)


class S1Follower(Robot):
    """
    Single S1 Follower Arm using the Theseus S1 SDK.
    Has 6 joints + optional gripper (7 DOF total).
    """

    config_class = S1FollowerConfig
    name = "s1_follower"

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

    def __init__(self, config: S1FollowerConfig):
        super().__init__(config)
        self.config = config

        # Import S1 SDK
        from S1_SDK.S1_arm import S1_arm, control_mode

        # Create arm instance
        self.arm = S1_arm(
            mode=control_mode.only_real,
            dev=config.port,
            end_effector=config.end_effector,
            check_collision=config.check_collision,
            arm_version=config.arm_version,
        )

        self.cameras = make_cameras_from_configs(config.cameras)

        # Flag to track connection state
        self._is_connected = False

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.MOTOR_NAMES}

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
        self.arm.enable()

        # Connect cameras
        for cam in self.cameras.values():
            cam.connect()

        self._is_connected = True
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        # S1 arms don't have the same calibration concept as Feetech motors
        # They use zero position setting instead
        return True

    def calibrate(self) -> None:
        """Set current position as zero for the arm."""
        logger.info(f"Setting zero position for {self}")
        self.arm.set_zero_position()

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

        # Read arm position
        start = time.perf_counter()
        pos = self.arm.get_pos()
        for i, motor in enumerate(self.MOTOR_NAMES):
            obs_dict[f"{motor}.pos"] = pos[i]
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read arm: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: np.ndarray) -> np.ndarray:
        """Command the arm to move to a target joint configuration.

        Args:
            action: Numpy array of 7 joint positions (6 joints + gripper)

        Returns:
            The action actually sent to the motors.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Ensure action is numpy array
        action = np.asarray(action, dtype=np.float32)

        # Separate gripper control (last element)
        joint_pos = action[:6].tolist()
        gripper_pos = float(action[6])

        # Send commands to the arm
        self.arm.joint_control_mit(joint_pos)
        self.arm.control_gripper(gripper_pos, 0.3)

        return action

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Disable motors
        self.arm.disable()

        # Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()

        self._is_connected = False
        logger.info(f"{self} disconnected.")

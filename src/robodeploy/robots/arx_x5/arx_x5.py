import logging
import time
from functools import cached_property
from typing import Any

import numpy as np

from robodeploy.cameras.utils import make_cameras_from_configs
from robodeploy.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_arx_x5 import ARXX5Config

logger = logging.getLogger(__name__)

MOTOR_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]


class ARXX5Robot(Robot):
    config_class = ARXX5Config
    name = "arx_x5"

    def __init__(self, config: ARXX5Config):
        super().__init__(config)
        self.config = config

        from arx_x5_python.bimanual import SingleArm

        arm_config = {
            "can_port": config.can_port,
            "type": config.arm_type,
            "num_joints": 7,
            "dt": 0.05,
        }
        self.arm = SingleArm(arm_config)
        self.cameras = make_cameras_from_configs(config.cameras)
        self._is_connected = False

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in MOTOR_NAMES}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
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

        for cam in self.cameras.values():
            cam.connect()

        if self.config.mode == "collect":
            self.arm.gravity_compensation()
            logger.info(f"{self} connected (gravity compensation mode).")
        else:
            self.arm.go_home()
            logger.info(f"{self} connected (position control mode).")

        self._is_connected = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}

        start = time.perf_counter()
        pos = self.arm.get_joint_positions()
        for i, motor in enumerate(MOTOR_NAMES):
            obs_dict[f"{motor}.pos"] = pos[i]
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read arm: {dt_ms:.1f}ms")

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        joint_pos = [float(action[f"{motor}.pos"]) for motor in MOTOR_NAMES[:6]]
        gripper_pos = float(action["gripper.pos"])

        self.arm.set_joint_positions(joint_pos)
        self.arm.set_catch_pos(gripper_pos)

        return action

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.arm.protect_mode()

        for cam in self.cameras.values():
            cam.disconnect()

        self._is_connected = False
        logger.info(f"{self} disconnected.")

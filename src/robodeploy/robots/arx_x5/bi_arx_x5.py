import logging
import time
from functools import cached_property
from typing import Any

import numpy as np

from robodeploy.cameras.utils import make_cameras_from_configs
from robodeploy.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .arx_x5 import MOTOR_NAMES
from .config_arx_x5 import BiARXX5Config

logger = logging.getLogger(__name__)


class BiARXX5Robot(Robot):
    config_class = BiARXX5Config
    name = "bi_arx_x5"

    def __init__(self, config: BiARXX5Config):
        super().__init__(config)
        self.config = config

        from arx_x5_python.bimanual import SingleArm

        left_arm_config = {
            "can_port": config.left_can_port,
            "type": config.left_arm_type,
            "num_joints": 7,
            "dt": 0.05,
        }
        right_arm_config = {
            "can_port": config.right_can_port,
            "type": config.right_arm_type,
            "num_joints": 7,
            "dt": 0.05,
        }
        self.left_arm = SingleArm(left_arm_config)
        self.right_arm = SingleArm(right_arm_config)
        self.cameras = make_cameras_from_configs(config.cameras)
        self._is_connected = False

        # Buffer: init to zeros, updated after each get_observation()
        self._state_buffer_left = np.zeros(7, dtype=np.float64)
        self._state_buffer_right = np.zeros(7, dtype=np.float64)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"left_{motor}.pos": float for motor in MOTOR_NAMES} | {
            f"right_{motor}.pos": float for motor in MOTOR_NAMES
        }

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
            self.left_arm.gravity_compensation()
            self.right_arm.gravity_compensation()
            logger.info(f"{self} connected (gravity compensation mode).")
        else:
            self.left_arm.go_home()
            self.right_arm.go_home()
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

        # Return buffered state (zeros on first call, previous arm position on subsequent calls)
        for i, motor in enumerate(MOTOR_NAMES):
            obs_dict[f"left_{motor}.pos"] = float(self._state_buffer_left[i])
            obs_dict[f"right_{motor}.pos"] = float(self._state_buffer_right[i])

        # Update buffer with current arm positions (for next call)
        start = time.perf_counter()
        left_pos = self.left_arm.get_joint_positions()
        self._state_buffer_left = np.array(left_pos, dtype=np.float64)
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read left arm: {dt_ms:.1f}ms")

        start = time.perf_counter()
        right_pos = self.right_arm.get_joint_positions()
        self._state_buffer_right = np.array(right_pos, dtype=np.float64)
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read right arm: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        left_joint_pos = [float(action[f"left_{motor}.pos"]) for motor in MOTOR_NAMES[:6]]
        left_gripper_pos = float(action["left_gripper.pos"])
        right_joint_pos = [float(action[f"right_{motor}.pos"]) for motor in MOTOR_NAMES[:6]]
        right_gripper_pos = float(action["right_gripper.pos"])

        self.left_arm.set_joint_positions(left_joint_pos)
        self.left_arm.set_catch_pos(left_gripper_pos)
        self.right_arm.set_joint_positions(right_joint_pos)
        self.right_arm.set_catch_pos(right_gripper_pos)

        return action

    def get_action(self) -> dict[str, float]:
        """Read joint positions from both arms (body teaching)."""
        left_pos = self.left_arm.get_joint_positions()
        right_pos = self.right_arm.get_joint_positions()
        action = {}
        for i, motor in enumerate(MOTOR_NAMES):
            action[f"left_{motor}.pos"] = left_pos[i]
            action[f"right_{motor}.pos"] = right_pos[i]
        return action

    def set_mode(self, mode: str) -> None:
        """Switch between collect (gravity compensation) and control (position control)."""
        if mode == "collect":
            self.left_arm.gravity_compensation()
            self.right_arm.gravity_compensation()
            logger.info(f"{self} switched to collect mode (gravity compensation).")
        elif mode == "control":
            self.left_arm.go_home()
            self.right_arm.go_home()
            logger.info(f"{self} switched to control mode (position control).")
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.left_arm.protect_mode()
        self.right_arm.protect_mode()

        for cam in self.cameras.values():
            cam.disconnect()

        self._is_connected = False
        logger.info(f"{self} disconnected.")

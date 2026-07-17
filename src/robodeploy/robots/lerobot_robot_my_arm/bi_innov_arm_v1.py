import logging
import time
from functools import cached_property
from typing import Any

import numpy as np

from robodeploy.cameras.utils import make_cameras_from_configs
from robodeploy.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from .ArmDriver import RobotController

from ..robot import Robot
from .config_innov_arm import BiInnovArmV1Config
from .innov_arm_v1 import MOTOR_NAMES

logger = logging.getLogger(__name__)


class BiInnovArmV1Robot(Robot):
    config_class = BiInnovArmV1Config
    name = "bi_innov_arm_v1"

    def __init__(self, config: BiInnovArmV1Config):
        super().__init__(config)
        self.config = config

        self.left_arm = RobotController(config.left_port, type="bi_innov_arm_v1_left")
        self.right_arm = RobotController(config.right_port, type="bi_innov_arm_v1_right")
        self.cameras = make_cameras_from_configs(config.cameras)
        self._is_connected = False

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

        self._configure_arm(self.left_arm)
        self._configure_arm(self.right_arm)

        self._is_connected = True

        if self.config.mode == "collect":
            self.left_arm.gravity_compensation()
            self.right_arm.gravity_compensation()
            logger.info(f"{self} connected (gravity compensation mode).")
        else:
            logger.info(f"{self} connected (position control mode).")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _configure_arm(self, arm) -> None:
        arm.enable()
        time.sleep(0.1)
        if self.config.mode == "collect":
            arm.type = "Grivity_arm"
            arm.set_mit_mode()
        else:
            arm.type = "follower"
            arm.set_pos_vel_mode()
        time.sleep(0.1)
        arm.enable()

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}

        for i, motor in enumerate(MOTOR_NAMES):
            obs_dict[f"left_{motor}.pos"] = float(self._state_buffer_left[i])
            obs_dict[f"right_{motor}.pos"] = float(self._state_buffer_right[i])

        start = time.perf_counter()
        left_pos = self.left_arm.get_current_joint_angles()
        left_gripper = self.left_arm.get_current_gripper_angles()
        self._state_buffer_left[:6] = np.array(left_pos, dtype=np.float64)
        self._state_buffer_left[6] = float(left_gripper)
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read left arm: {dt_ms:.1f}ms")

        start = time.perf_counter()
        right_pos = self.right_arm.get_current_joint_angles()
        right_gripper = self.right_arm.get_current_gripper_angles()
        self._state_buffer_right[:6] = np.array(right_pos, dtype=np.float64)
        self._state_buffer_right[6] = float(right_gripper)
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read right arm: {dt_ms:.1f}ms")

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

        if self.config.mode == "collect":
            self.left_arm.gravity_compensation()
            self.right_arm.gravity_compensation()
        else:
            self.left_arm.set_joint_angles(left_joint_pos, 2)
            self.left_arm.set_gripper_angles(gripper_angle=left_gripper_pos, v=2, tau_limit=0.1)
            self.right_arm.set_joint_angles(right_joint_pos, 2)
            self.right_arm.set_gripper_angles(gripper_angle=right_gripper_pos, v=2, tau_limit=0.1)

        return action

    def get_action(self) -> dict[str, float]:
        """Read joint positions from both arms (body teaching)."""
        left_pos = self.left_arm.get_current_joint_angles()
        left_gripper = self.left_arm.get_current_gripper_angles()
        right_pos = self.right_arm.get_current_joint_angles()
        right_gripper = self.right_arm.get_current_gripper_angles()
        action = {}
        for i, motor in enumerate(MOTOR_NAMES[:6]):
            action[f"left_{motor}.pos"] = left_pos[i]
        action["left_gripper.pos"] = float(left_gripper)
        for i, motor in enumerate(MOTOR_NAMES[:6]):
            action[f"right_{motor}.pos"] = right_pos[i]
        action["right_gripper.pos"] = float(right_gripper)
        return action

    def set_mode(self, mode: str) -> None:
        """Switch between collect (gravity compensation) and control (position control)."""
        if mode == "collect":
            self.left_arm.disable()
            self.right_arm.disable()
            time.sleep(0.1)
            self.left_arm.type = "Grivity_arm"
            self.right_arm.type = "Grivity_arm"
            self.left_arm.set_mit_mode()
            self.right_arm.set_mit_mode()
            time.sleep(0.1)
            self.left_arm.enable()
            self.right_arm.enable()
            time.sleep(0.1)
            self.left_arm.gravity_compensation()
            self.right_arm.gravity_compensation()
            self.config.mode = "collect"
            logger.info(f"{self} switched to collect mode (gravity compensation).")
        elif mode == "control":
            self.left_arm.disable()
            self.right_arm.disable()
            time.sleep(0.1)
            self.left_arm.type = "follower"
            self.right_arm.type = "follower"
            self.left_arm.set_pos_vel_mode()
            self.right_arm.set_pos_vel_mode()
            time.sleep(0.1)
            self.left_arm.enable()
            self.right_arm.enable()
            self.config.mode = "control"
            logger.info(f"{self} switched to control mode (position control).")
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.left_arm.set_pos_vel_mode()
        self.right_arm.set_pos_vel_mode()
        time.sleep(0.1)
        self.left_arm.disable()
        self.right_arm.disable()

        for cam in self.cameras.values():
            cam.disconnect()

        self._is_connected = False
        logger.info(f"{self} disconnected.")

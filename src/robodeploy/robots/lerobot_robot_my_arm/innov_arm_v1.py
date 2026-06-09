import logging
import time
from functools import cached_property
from typing import Any

from robodeploy.cameras.utils import make_cameras_from_configs
from robodeploy.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..robot import Robot
from .config_innov_arm import InnovArmV1Config

logger = logging.getLogger(__name__)

MOTOR_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]


class InnovArmV1Robot(Robot):
    config_class = InnovArmV1Config
    name = "innov_arm_v1"

    def __init__(self, config: InnovArmV1Config):
        super().__init__(config)
        self.config = config

        from .ArmDriver import RobotController

        self.arm = RobotController(config.port, type="innov_arm_v1")
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

        if not self.arm.RobotCtrl.serial_.is_open:
            raise DeviceNotConnectedError(f"{self} serial port not open")

        self._is_connected = True
        self.configure()

        for cam in self.cameras.values():
            cam.connect()

        if self.config.mode == "collect":
            self.arm.gravity_compensation()
            logger.info(f"{self} connected (gravity compensation mode).")
        else:
            logger.info(f"{self} connected (position control mode).")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        self.arm.enable()
        time.sleep(0.1)
        if self.config.mode == "collect":
            self.arm.type = "Grivity_arm"
            self.arm.set_mit_mode()
        else:
            self.arm.type = "follower"
            self.arm.set_pos_vel_mode()
        time.sleep(0.1)
        self.arm.enable()
        logger.info(f"configure {self.name} done (mode={self.config.mode})")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict = {}

        start = time.perf_counter()
        pos = self.arm.get_current_joint_angles()
        gripper_pos = self.arm.get_current_gripper_angles()
        for i, motor in enumerate(MOTOR_NAMES[:6]):
            obs_dict[f"{motor}.pos"] = pos[i]
        obs_dict["gripper.pos"] = float(gripper_pos)
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

        if self.config.mode == "collect":
            self.arm.gravity_compensation()
        else:
            self.arm.set_joint_angles(joint_pos, 2)
            self.arm.set_gripper_angles(gripper_angle=gripper_pos, v=2, tau_limit=0.1)

        return action

    def get_action(self) -> dict[str, float]:
        """Read joint positions from arm (body teaching)."""
        pos = self.arm.get_current_joint_angles()
        gripper_pos = self.arm.get_current_gripper_angles()
        action = {}
        for i, motor in enumerate(MOTOR_NAMES[:6]):
            action[f"{motor}.pos"] = pos[i]
        action["gripper.pos"] = float(gripper_pos)
        return action

    def set_mode(self, mode: str) -> None:
        """Switch between collect (gravity compensation) and control (position control)."""
        if mode == "collect":
            self.arm.type = "Grivity_arm"
            self.arm.set_mit_mode()
            self.arm.gravity_compensation()
            self.config.mode = "collect"
            logger.info(f"{self} switched to collect mode (gravity compensation).")
        elif mode == "control":
            self.arm.type = "follower"
            self.arm.set_pos_vel_mode()
            self.config.mode = "control"
            logger.info(f"{self} switched to control mode (position control).")
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.arm.set_pos_vel_mode()
        time.sleep(0.1)
        self.arm.disable()

        for cam in self.cameras.values():
            cam.disconnect()

        self._is_connected = False
        logger.info(f"{self} disconnected.")

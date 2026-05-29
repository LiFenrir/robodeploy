import logging
import time
from functools import cached_property
from typing import Any

from robodeploy.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_arx_x5_leader import ARXX5LeaderConfig

logger = logging.getLogger(__name__)

MOTOR_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]


class ARXX5Leader(Teleoperator):
    config_class = ARXX5LeaderConfig
    name = "arx_x5_leader"

    def __init__(self, config: ARXX5LeaderConfig):
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
        self._is_connected = False

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in MOTOR_NAMES}

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.arm.gravity_compensation()
        self._is_connected = True
        logger.info(f"{self} connected (gravity compensation mode).")

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, float]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start = time.perf_counter()
        pos = self.arm.get_joint_positions()
        action = {f"{MOTOR_NAMES[i]}.pos": pos[i] for i in range(7)}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.arm.protect_mode()
        self._is_connected = False
        logger.info(f"{self} disconnected.")

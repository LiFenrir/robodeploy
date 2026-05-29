import logging
import time
from functools import cached_property
from typing import Any

from robodeploy.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .arx_x5_leader import MOTOR_NAMES
from .config_arx_x5_leader import BiARXX5LeaderConfig

logger = logging.getLogger(__name__)


class BiARXX5Leader(Teleoperator):
    config_class = BiARXX5LeaderConfig
    name = "bi_arx_x5_leader"

    def __init__(self, config: BiARXX5LeaderConfig):
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
        self._is_connected = False

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"left_{motor}.pos": float for motor in MOTOR_NAMES} | {
            f"right_{motor}.pos": float for motor in MOTOR_NAMES
        }

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

        self.left_arm.gravity_compensation()
        self.right_arm.gravity_compensation()
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
        left_pos = self.left_arm.get_joint_positions()
        right_pos = self.right_arm.get_joint_positions()
        dt_ms = (time.perf_counter() - start) * 1e3

        action = {}
        for i, motor in enumerate(MOTOR_NAMES):
            action[f"left_{motor}.pos"] = left_pos[i]
            action[f"right_{motor}.pos"] = right_pos[i]

        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.left_arm.protect_mode()
        self.right_arm.protect_mode()
        self._is_connected = False
        logger.info(f"{self} disconnected.")

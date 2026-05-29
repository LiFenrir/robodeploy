from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("arx_x5_leader")
@dataclass
class ARXX5LeaderConfig(TeleoperatorConfig):
    can_port: str
    arm_type: int = 1  # 1=master URDF for leader arm


@TeleoperatorConfig.register_subclass("bi_arx_x5_leader")
@dataclass
class BiARXX5LeaderConfig(TeleoperatorConfig):
    left_can_port: str
    right_can_port: str
    left_arm_type: int = 1
    right_arm_type: int = 1

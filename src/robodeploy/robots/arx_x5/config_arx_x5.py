from dataclasses import dataclass, field

from robodeploy.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("arx_x5")
@dataclass
class ARXX5Config(RobotConfig):
    can_port: str = "can0"
    arm_type: int = 0  # 0=standard, 1=master, 2=2025
    end_effector: str = "composite"
    mode: str = "collect"  # "collect"=gravity comp for body teaching, "control"=position control
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


@RobotConfig.register_subclass("bi_arx_x5")
@dataclass
class BiARXX5Config(RobotConfig):
    left_can_port: str
    right_can_port: str
    left_arm_type: int = 0
    right_arm_type: int = 0
    left_end_effector: str = "composite"
    right_end_effector: str = "composite"
    mode: str = "collect"  # "collect"=gravity comp for body teaching, "control"=position control
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

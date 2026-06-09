from dataclasses import dataclass, field

from robodeploy.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("innov_arm_v1")
@dataclass
class InnovArmV1Config(RobotConfig):
    port: str
    mode: str = "collect"  # "collect"=gravity comp for body teaching, "control"=position control
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


@RobotConfig.register_subclass("bi_innov_arm_v1")
@dataclass
class BiInnovArmV1Config(RobotConfig):
    left_port: str
    right_port: str
    mode: str = "collect"  # "collect"=gravity comp for body teaching, "control"=position control
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

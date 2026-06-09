from dataclasses import dataclass, field
from functools import cached_property
import serial
import time
import logging
from typing import Any
import re

from lerobot.cameras import CameraConfig
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots import Robot, RobotConfig
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .ArmDriver import RobotController

logger = logging.getLogger(__name__)


@RobotConfig.register_subclass("Grivity_arm")
@dataclass
class GrivityRobotConfig(RobotConfig):
    port: str
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


class GrivityRobot(Robot):
    """
    DM Arm Follower Arm designed by The Robot Learning Company.
    """

    config_class = GrivityRobotConfig
    name = "Grivity_arm"

    def __init__(self, config: GrivityRobotConfig):
        super().__init__(config)

        self.config = config
        # self._ser = None
        self._is_connected = False
        self.obs_dict = {}
        self.arm =None
        self.first_action_received = False
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
      return {
        "joint_1.pos": float,  # 数据类型是 float，不是默认值 0
        "joint_2.pos": float,
        "joint_3.pos": float,
        "joint_4.pos": float,
        "joint_5.pos": float,
        "joint_6.pos": float,
        "gripper": float,  # 夹具也是 float 类型
        }

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

    def connect(self) -> None:
        if self._is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.arm = RobotController(self.config.port, type='Grivity_arm')
        if self.arm.RobotCtrl.serial_.is_open:
            self._is_connected = True
        else:
            print("Grivity_arm connected fail")

        self.configure()

        for cam in self.cameras.values():
            cam.connect()
        
        print("Camera connect down")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        self.arm.enable()
        time.sleep(0.1)
        self.arm.set_mit_mode()   # 先暂时设置为mit模式 
        time.sleep(0.1)
        self.arm.enable()      
        print("configure Grivity_arm down")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()

        pos = self.arm.get_current_joint_angles()

        self.obs_dict["joint_1.pos"] = pos[0]
        self.obs_dict["joint_2.pos"] = pos[1]
        self.obs_dict["joint_3.pos"] = pos[2]
        self.obs_dict["joint_4.pos"] = pos[3]
        self.obs_dict["joint_5.pos"] = pos[4]
        self.obs_dict["joint_6.pos"] = pos[5]
        self.obs_dict["gripper"] = self.arm.get_current_gripper_angles()

        dt_ms = (time.perf_counter() - start) * 1e3
        #print(f"read state: {dt_ms:.1f}ms")   #这个是打印延时？
        # print(self.obs_dict)

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            self.obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            # print(f"{self} read {cam_key}: {dt_ms:.1f} ms")   #

        return self.obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        #开启重力补偿
        self.arm.gravity_compensation()
        #print("***************重力补偿开启***********")
          # 返回当前观测的关节位置作为动作
        return {
            "joint_1.pos": self.obs_dict.get("joint_1.pos", 0.0),
            "joint_2.pos": self.obs_dict.get("joint_2.pos", 0.0),
            "joint_3.pos": self.obs_dict.get("joint_3.pos", 0.0),
            "joint_4.pos": self.obs_dict.get("joint_4.pos", 0.0),
            "joint_5.pos": self.obs_dict.get("joint_5.pos", 0.0),
            "joint_6.pos": self.obs_dict.get("joint_6.pos", 0.0),
            "gripper": self.obs_dict.get("gripper", 0.0),
        }


    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError("DmArmFollower is not connected.")
        time.sleep(0.1)
        self.arm.set_pos_vel_mode()  #再设置为位置模式
        time.sleep(0.1)
        self.arm.disable()
        self._is_connected = False

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info("Grivity disconnected.")

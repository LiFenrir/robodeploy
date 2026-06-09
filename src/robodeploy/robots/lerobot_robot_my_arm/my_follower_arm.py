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


@RobotConfig.register_subclass("my_follower_arm")
@dataclass
class FollowerRobotConfig(RobotConfig):
    port: str
    cameras: dict[str, CameraConfig] = field(default_factory=dict)


class FollowerRobot(Robot):
    """
    DM Arm Follower Arm designed by The Robot Learning Company.
    """

    config_class = FollowerRobotConfig
    name = "my_follower_arm"

    def __init__(self, config: FollowerRobotConfig):
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
        pos_dict = {
            "joint_1.pos": 0,
            "joint_2.pos": 0,
            "joint_3.pos": 0,
            "joint_4.pos": 0,
            "joint_5.pos": 0,
            "joint_6.pos": 0,
            "gripper": 0,
        }

        # pos = self.arm.get_q()

        # if pos:
            # pos_dict["joint_1.pos"] = pos[0]
            # pos_dict["joint_2.pos"] = pos[1]
            # pos_dict["joint_3.pos"] = pos[2]
            # pos_dict["joint_4.pos"] = pos[3]
            # pos_dict["joint_5.pos"] = pos[4]
            # pos_dict["joint_6.pos"] = pos[5]
            # pos_dict["gripper"] = pos[6]

        return pos_dict

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

        self.arm = RobotController(self.config.port, type='my_follower_arm')
        if self.arm.RobotCtrl.serial_.is_open:
            self._is_connected = True
        else:
            print("my_follower_arm connected fail")

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
        self.arm.set_pos_vel_mode()   # 设置为位置模式 
        time.sleep(0.1)
        self.arm.enable()      
        print("configure my_follower_arm down")

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
        #self.arm.gravity_compensation()
        pos = [
            action["joint_1.pos"],
            action["joint_2.pos"],
            action["joint_3.pos"],
            action["joint_4.pos"],
            action["joint_5.pos"],
            action["joint_6.pos"],
        ]
        gripper = action["gripper"]

        if not self.first_action_received:
            self.first_action_received = True
            start = time.perf_counter()
            self.arm.set_joint_angles(pos, 1)
            self.arm.set_gripper_angles(gripper_angle=gripper*2, v=2, tau_limit=0.1)
            time.sleep(1)
            dt_ms = (time.perf_counter() - start) * 1e3
            print(f"Run to start position of first action: {dt_ms:.1f} ms")
            time.sleep(1)

        # Send goal position to the arm

        self.arm.set_joint_angles(pos, 2)   #########这里设置速度########
        self.arm.set_gripper_angles(gripper_angle=gripper, v=2, tau_limit=0.1)



        return action

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError("my_follower_arm is not connected.")
        time.sleep(0.1)
        self.arm.disable()
        time.sleep(0.1)
        self._is_connected = False

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info("my_follower_arm disconnected")

#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""S1 主臂直接遥操作 Innov 从臂。

不经过 Robot/Teleoperator 抽象，直接调用 S1 SDK 与 Innov ArmDriver。
S1 get_pos() 读到的 6 关节 + 夹爪原样发送给 Innov 从臂。
Usage:
    python s1_to_innov_teleop.py --s1-port=/dev/ttyUSB0 --innov-port=/dev/ttyUSB1
"""

import argparse
import logging
import signal
import sys
import time

from S1_SDK.S1_arm import S1_arm, control_mode

from robodeploy.robots.lerobot_robot_my_arm.ArmDriver import RobotController

logger = logging.getLogger(__name__)

STOP_FLAG = False


def signal_handler(_sig, _frame):
    """收到 Ctrl+C 后设置停止标志，让主循环优雅退出。"""
    global STOP_FLAG
    STOP_FLAG = True
    print("\n[Signal] 收到中断，准备退出...")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="S1 主臂遥操作 Innov 从臂（直接 SDK 版）")
    parser.add_argument("--s1-port", type=str, default="/dev/ttyUSB0", help="S1 主臂串口设备")
    parser.add_argument("--innov-port", type=str, default="/dev/ttyUSB1", help="Innov 从臂串口设备")
    parser.add_argument("--fps", type=int, default=30, help="控制循环频率")
    parser.add_argument("--innov-speed", type=float, default=2.0, help="Innov 关节运动速度")
    parser.add_argument("--log-level", type=str, default="INFO", help="日志级别: DEBUG/INFO/WARNING")
    return parser.parse_args()


def connect_s1(args: argparse.Namespace) -> S1_arm:
    """初始化并连接 S1 主臂。"""
    logger.info(f"连接 S1 主臂: port={args.s1_port}")
    arm = S1_arm(
        mode=control_mode.only_real,
        dev=args.s1_port,
        end_effector="teach",
        check_collision=False,
        arm_version="V2",
    )
    arm.enable()
    logger.info("S1 主臂已使能")
    return arm


def connect_innov(args: argparse.Namespace) -> RobotController:
    """初始化并连接 Innov 从臂。"""
    logger.info(f"连接 Innov 从臂: port={args.innov_port}")
    arm = RobotController(args.innov_port, type="follower")
    if not arm.RobotCtrl.serial_.is_open:
        raise ConnectionError(f"Innov 从臂串口 {args.innov_port} 未打开")

    arm.enable()
    time.sleep(0.1)
    arm.set_pos_vel_mode()
    time.sleep(0.1)
    arm.enable()
    logger.info("Innov 从臂已使能（关节 POS_VEL + 夹爪 Torque_Pos）")
    return arm


def run_teleop(s1_arm: S1_arm, innov_arm: RobotController, args: argparse.Namespace) -> None:
    """主循环：读取 S1 位置并原样发送到 Innov。"""
    period = 1.0 / args.fps
    logger.info(f"开始遥操作循环，频率 {args.fps}Hz (周期 {period * 1000:.1f}ms)")
    logger.info("按 Ctrl+C 停止")

    while not STOP_FLAG:
        t0 = time.perf_counter()

        # 读取 S1 主臂位置
        s1_pos = s1_arm.get_pos()

        # 维持 S1 主臂在可拖动状态
        s1_arm.control_teach(0.08)
        s1_arm.gravity()

        # 拆分 6 关节 + 夹爪，关节 2/4/5（索引 1/3/4）取反
        s1_joints = list(s1_pos[:6])
        for idx in (1, 3, 4):
            s1_joints[idx] = -s1_joints[idx]
        s1_gripper = float(s1_pos[6]) if len(s1_pos) > 6 else 0.0

        logger.debug(f"S1: joints={[f'{p:7.3f}' for p in s1_joints]}  gripper={s1_gripper:7.3f}")

        innov_arm.set_joint_angles(s1_joints, args.innov_speed)
        innov_arm.set_gripper_angles(gripper_angle=s1_gripper, v=2, tau_limit=0.1)

        # 按目标频率休眠
        elapsed = time.perf_counter() - t0
        sleep_time = period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)


def main() -> int:
    """入口函数。"""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    signal.signal(signal.SIGINT, signal_handler)

    s1_arm = None
    innov_arm = None
    try:
        s1_arm = connect_s1(args)
        innov_arm = connect_innov(args)

        print("=" * 60)
        print(f"  S1 主臂: {args.s1_port}")
        print(f"  Innov 从臂: {args.innov_port}")
        print(f"  频率: {args.fps}Hz")
        print(f"  Innov 速度: {args.innov_speed}")
        print("=" * 60)

        run_teleop(s1_arm, innov_arm, args)
    except KeyboardInterrupt:
        logger.info("用户中断")
    except Exception as e:
        logger.exception(f"运行出错: {e}")
        return 1
    finally:
        logger.info("关闭设备...")
        if innov_arm is not None:
            try:
                innov_arm.set_pos_vel_mode()
                time.sleep(0.1)
                innov_arm.disable()
                innov_arm.close_serial()
            except Exception as e:
                logger.warning(f"Innov 关闭异常: {e}")
        if s1_arm is not None:
            try:
                s1_arm.disable()
                s1_arm.close()
            except Exception as e:
                logger.warning(f"S1 关闭异常: {e}")
        logger.info("已退出")
    return 0


if __name__ == "__main__":
    sys.exit(main())

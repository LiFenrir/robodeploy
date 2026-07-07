#!/usr/bin/env python
"""主从控制测试：主臂自由拖动（MIT+重力补偿），从臂实时跟随（位置模式）。

Usage:
    python examples/innov/test_master_slave.py \
        --master_port /dev/ttyACM0 --slave_port /dev/ttyACM1

    # 调整跟随速度
    python examples/innov/test_master_slave.py \
        --master_port /dev/ttyACM0 --slave_port /dev/ttyACM1 \
        --speed 3

按键:
    Esc   退出
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from robodeploy.robots.lerobot_robot_my_arm.ArmDriver import RobotController
from robodeploy.utils.keyboard_control import get_keypress

JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]


def main():
    parser = argparse.ArgumentParser(description="主从控制测试")
    parser.add_argument("--master_port", type=str, required=True, help="主臂串口")
    parser.add_argument("--slave_port", type=str, required=True, help="从臂串口")
    parser.add_argument("--speed", type=int, default=2, help="从臂跟随速度 (1-10)")
    parser.add_argument("--fps", type=int, default=50, help="控制循环频率")
    args = parser.parse_args()

    # ── 连接 ──
    master = RobotController(args.master_port, type="leader")
    slave = RobotController(args.slave_port, type="follower")
    if not master.RobotCtrl.serial_.is_open:
        raise ConnectionError(f"主臂串口 {args.master_port} 打开失败")
    if not slave.RobotCtrl.serial_.is_open:
        raise ConnectionError(f"从臂串口 {args.slave_port} 打开失败")
    print(f"主臂 {args.master_port} 已连接")
    print(f"从臂 {args.slave_port} 已连接")

    try:
        # ── 主臂：MIT + 重力补偿（可自由拖动） ──
        master.enable()
        time.sleep(0.1)
        master.set_mit_mode()
        time.sleep(0.1)
        master.enable()
        master.gravity_compensation()
        print("主臂 MIT 模式 + 重力补偿已启用")

        # ── 从臂：位置模式（关节 POS_VEL + 夹爪 Torque_Pos） ──
        slave.enable()
        time.sleep(0.1)
        slave.set_pos_vel_mode()
        time.sleep(0.1)
        slave.enable()
        print("从臂位置模式已启用")

        print("\n" + "=" * 80)
        print("  跟随中... Esc 退出")
        print("=" * 80 + "\n")

        frame = 0

        while True:
            t_loop = time.perf_counter()

            # ── Esc 退出 ──
            key = get_keypress()
            if key == "\x1b":
                break

            # ── 读主臂 ──
            m_joints = master.get_current_joint_angles()
            m_gripper = master.get_current_gripper_angles()

            # ── 从臂跟随 ──
            slave.set_joint_angles(m_joints, args.speed)
            slave.set_gripper_angles(m_gripper, v=args.speed, tau_limit=0.1)

            # ── 读从臂当前角度 ──
            s_joints = slave.get_current_joint_angles()
            s_gripper = slave.get_current_gripper_angles()

            # ── 刷新主臂重力补偿 ──
            master.gravity_compensation()

            # ── 打印角度信息 ──
            m_j_str = " ".join(f"{v:7.3f}" for v in m_joints)
            s_j_str = " ".join(f"{v:7.3f}" for v in s_joints)
            print(f"[{frame:4d}] "
                  f"主臂关节: [{m_j_str}]  夹爪: {m_gripper:7.3f} | "
                  f"从臂关节: [{s_j_str}]  夹爪: {s_gripper:7.3f}")

            # ── 帧率控制 ──
            dt = time.perf_counter() - t_loop
            sleep = 1.0 / args.fps - dt
            if sleep > 0:
                time.sleep(sleep)

            frame += 1

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        print("清理中...")
        try:
            master.set_pos_vel_mode()
            time.sleep(0.05)
            master.disable()
            master.close_serial()
        except Exception as e:
            print(f"主臂断开异常: {e}")
        try:
            slave.disable()
            slave.close_serial()
        except Exception as e:
            print(f"从臂断开异常: {e}")
        print("退出")


if __name__ == "__main__":
    main()

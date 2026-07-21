# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""单臂关节极值测试：开启重力补偿后手动拖动机械臂，记录各关节最大/最小角度。

Usage:
    python examples/innov/test_joint_limits.py --port /dev/ttyACM0

按键:
    Esc   退出并保存结果
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from robodeploy.robots.lerobot_robot_my_arm.ArmDriver import RobotController
from robodeploy.utils.keyboard_control import get_keypress

JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]


def main():
    parser = argparse.ArgumentParser(description="单臂关节极值测试")
    parser.add_argument("--port", type=str, required=True, help="机械臂串口")
    parser.add_argument("--fps", type=int, default=50, help="读取循环频率")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 JSON 文件路径，默认 joint_limits_<timestamp>.json",
    )
    args = parser.parse_args()

    limits = {name: {"min": float("inf"), "max": float("-inf")} for name in JOINT_NAMES}
    gripper_limits = {"min": float("inf"), "max": float("-inf")}

    arm = RobotController(args.port, type="leader")
    if not arm.RobotCtrl.serial_.is_open:
        raise ConnectionError(f"串口 {args.port} 打开失败")
    print(f"机械臂 {args.port} 已连接")

    try:
        arm.enable()
        time.sleep(0.1)
        arm.set_mit_mode()
        time.sleep(0.1)
        arm.enable()
        arm.gravity_compensation()
        print("MIT 模式 + 重力补偿已启用，可手动拖动关节")

        print("\n" + "=" * 80)
        print("  拖动到极限位置后按 Esc 退出并保存结果")
        print("=" * 80 + "\n")

        frame = 0
        while True:
            t_loop = time.perf_counter()

            key = get_keypress()
            if key == "\x1b":
                break

            joints = arm.get_current_joint_angles()
            gripper = arm.get_current_gripper_angles()

            for name, value in zip(JOINT_NAMES, joints, strict=True):
                limits[name]["min"] = min(limits[name]["min"], value)
                limits[name]["max"] = max(limits[name]["max"], value)

            gripper_limits["min"] = min(gripper_limits["min"], gripper)
            gripper_limits["max"] = max(gripper_limits["max"], gripper)

            arm.gravity_compensation()

            j_str = " ".join(f"{v:7.3f}" for v in joints)
            print(
                f"[{frame:4d}] 关节: [{j_str}]  夹爪: {gripper:7.3f} | "
                f"当前极值: "
                + " ".join(f"{n}[{limits[n]['min']:+.2f},{limits[n]['max']:+.2f}]" for n in JOINT_NAMES)
            )

            dt = time.perf_counter() - t_loop
            sleep = 1.0 / args.fps - dt
            if sleep > 0:
                time.sleep(sleep)

            frame += 1

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        print("\n清理中...")
        try:
            arm.set_pos_vel_mode()
            time.sleep(0.05)
            arm.disable()
            arm.close_serial()
        except Exception as e:
            print(f"断开异常: {e}")

        print("\n" + "=" * 80)
        print("  关节极值")
        print("=" * 80)
        for name in JOINT_NAMES:
            lo = limits[name]["min"]
            hi = limits[name]["max"]
            print(f"  {name}:  min = {lo:+.4f}, max = {hi:+.4f}")
        print(f"  gripper: min = {gripper_limits['min']:+.4f}, max = {gripper_limits['max']:+.4f}")

        result = {
            "port": args.port,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "joints": limits,
            "gripper": gripper_limits,
        }

        output_path = args.output
        if output_path is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_path = f"joint_limits_{timestamp}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_path}")
        print("退出")


if __name__ == "__main__":
    main()

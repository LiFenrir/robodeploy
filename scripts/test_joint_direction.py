"""Test joint torque direction to verify inverted_axes configuration.

For each joint, applies a small positive MIT torque and asks the user to observe
which direction the joint moves. Based on the observation, determines whether the
joint should be in the inverted_axes list.

Usage:
    # Single arm
    python scripts/test_joint_direction.py --port /dev/ttyACM0

    # Dual arm
    python scripts/test_joint_direction.py --left_port /dev/ttyACM0 --right_port /dev/ttyACM1

Safety:
    - Torque is small (default 0.15, adjustable via --torque)
    - Only one joint tested at a time
    - Each test lasts ~1 second
    - Press Ctrl+C to abort at any time
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from robodeploy.robots.lerobot_robot_my_arm.ArmDriver import RobotController

JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
CURRENT_INVERTED = [1, 2, 3, 4]  # current inverted_axes (0-based indices)


def test_single_joint(arm, joint_idx: int, torque: float, arm_name: str) -> str:
    """Apply positive torque to one joint and ask user which way it moved.

    Returns: "invert" (needs inversion), "keep" (no inversion), or "unclear".
    """
    joint = arm.joints[joint_idx]
    name = JOINT_NAMES[joint_idx]

    # Read initial position via ArmDriver (properly refreshes all motors)
    pos_before = arm.get_current_joint_angles()[joint_idx]
    print(f"\n  [{arm_name}] {name}: pos_before={pos_before:.4f}, applying torque={torque:.3f} for 1s...")

    # Apply positive torque (no kp/kd/q/dq, pure torque control)
    arm.RobotCtrl.controlMIT(joint, 0.0, 0.0, 0.0, 0.0, torque)
    time.sleep(1.0)

    # Stop torque first, then read position
    arm.RobotCtrl.controlMIT(joint, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Read position after via ArmDriver (properly refreshes all motors)
    pos_after = arm.get_current_joint_angles()[joint_idx]

    delta = pos_after - pos_before
    direction = "+" if delta > 0 else "-"

    print(f"  [{arm_name}] {name}: pos_after={pos_after:.4f}, delta={delta:+.4f} ({direction})")

    while True:
        answer = input(f"  [{arm_name}] {name}: Which direction did joint {joint_idx+1} move? (+ / - / ?): ").strip()
        if answer == "+":
            print(f"  [{arm_name}] {name}: → positive torque → positive movement → KEEP (no inversion)")
            return "keep"
        elif answer == "-":
            print(f"  [{arm_name}] {name}: → positive torque → negative movement → INVERT")
            return "invert"
        elif answer == "?":
            print(f"  [{arm_name}] {name}: → unclear, retry recommended")
            return "unclear"
        else:
            print("  Enter '+' for positive direction, '-' for negative, '?' for unclear")


def main():
    parser = argparse.ArgumentParser(description="Test joint torque direction for inverted_axes config")
    parser.add_argument("--port", type=str, default=None, help="Serial port (single arm)")
    parser.add_argument("--left_port", type=str, default=None, help="Left arm serial port (dual arm)")
    parser.add_argument("--right_port", type=str, default=None, help="Right arm serial port (dual arm)")
    parser.add_argument("--baudrate", type=int, default=921600, help="Serial baudrate")
    parser.add_argument("--torque", type=float, default=0.15, help="Test torque amplitude")
    parser.add_argument("--joints", type=str, default="1,2,3,4,5,6", help="Joints to test (1-indexed, comma-separated)")
    args = parser.parse_args()

    ports = []
    if args.left_port and args.right_port:
        ports = [("left", args.left_port), ("right", args.right_port)]
    elif args.port:
        ports = [("arm", args.port)]
    else:
        parser.error("Specify --port or --left_port/--right_port")

    joint_indices = [int(j.strip()) - 1 for j in args.joints.split(",")]

    results = {}
    for arm_name, port in ports:
        print(f"\n{'='*60}")
        print(f"[{arm_name}] Connecting to {port} ...")
        arm = RobotController(port, baudrate=args.baudrate, type="test")
        print(f"[{arm_name}] Enabling motors in MIT mode ...")
        arm.enable()
        time.sleep(0.1)
        arm.set_mit_mode()
        time.sleep(0.1)

        arm_results = {}
        for j in joint_indices:
            result = test_single_joint(arm, j, args.torque, arm_name)
            arm_results[j] = result

        arm.disable()
        arm.close_serial()
        results[arm_name] = arm_results

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Current inverted_axes (0-based): {CURRENT_INVERTED}")
    print(f"(Indices: 0=joint_1, 1=joint_2, 2=joint_3, 3=joint_4, 4=joint_5, 5=joint_6)")
    print()

    for arm_name, arm_results in results.items():
        invert_list = sorted([j for j, r in arm_results.items() if r == "invert"])
        keep_list = sorted([j for j, r in arm_results.items() if r == "keep"])
        unclear_list = sorted([j for j, r in arm_results.items() if r == "unclear"])

        print(f"[{arm_name}]:")
        print(f"  Should INVERT (positive torque → negative move): {[j+1 for j in invert_list]} → indices: {invert_list}")
        print(f"  Should KEEP  (positive torque → positive move): {[j+1 for j in keep_list]} → indices: {keep_list}")
        if unclear_list:
            print(f"  UNCLEAR: {[j+1 for j in unclear_list]} → retry these")
        print(f"  → Suggested inverted_axes: {invert_list}")


if __name__ == "__main__":
    main()

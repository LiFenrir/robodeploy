#!/usr/bin/env python
"""Set motor zero position for innov_arm / bi_innov_arm.

Usage:
    # Single arm
    python examples/innov/set_motor_zero.py --port /dev/ttyACM0

    # Dual arm
    python examples/innov/set_motor_zero.py --left_port /dev/ttyACM0 --right_port /dev/ttyACM1
"""

import argparse
import sys
from pathlib import Path

# Ensure robodeploy is importable when run directly from examples/innov/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from robodeploy.robots.lerobot_robot_my_arm.ArmDriver import RobotController


def main():
    parser = argparse.ArgumentParser(description="Set motor zero position for innov_arm")
    parser.add_argument("--port", type=str, default=None, help="Serial port for single arm")
    parser.add_argument("--left_port", type=str, default=None, help="Left arm serial port (dual arm)")
    parser.add_argument("--right_port", type=str, default=None, help="Right arm serial port (dual arm)")
    parser.add_argument("--baudrate", type=int, default=921600, help="Serial baudrate")
    args = parser.parse_args()

    ports = []
    if args.left_port and args.right_port:
        ports = [("left", args.left_port), ("right", args.right_port)]
    elif args.port:
        ports = [("arm", args.port)]
    else:
        parser.error("Specify --port for single arm or --left_port/--right_port for dual arm.")

    for name, port in ports:
        print(f"[{name}] Connecting to {port} ...")
        arm = RobotController(port, baudrate=args.baudrate, type="Grivity_arm")
        print(f"[{name}] Setting zero position ...")
        ok = arm.set_zero()
        arm.close_serial()
        status = "OK" if ok else f"WARNING: some motors did not zero cleanly"
        print(f"[{name}] Done. {status}")


if __name__ == "__main__":
    main()

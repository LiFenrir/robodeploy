#!/usr/bin/env python
"""
Synchronous inference script for analyzing policy action outputs.

Runs control loop at 30Hz: consumes action chunk frame-by-frame, re-infers
when chunk exhausted. No temporal smoothing, no recording, no teleop.

Uses draccus for configuration parsing with ChoiceRegistry support.

Usage:
    # S1 bimanual
    python inspect_policy_action.py \
        --robot.type=bi_s1_follower \
        --robot.left_arm_port=/dev/ttyUSB0 --robot.right_arm_port=/dev/ttyUSB1 \
        --robot.cameras='{"front":{"type":"intelrealsense","width":848,"height":480,"fps":30}}' \
        --policy.type=openpi \
        --policy.host=localhost --policy.port=8000 \
        --task="fold the box"

    # Single arm S1
    python inspect_policy_action.py \
        --robot.type=s1_follower \
        --robot.port=/dev/ttyUSB0 \
        --robot.cameras='{"front":{"type":"intelrealsense","width":848,"height":480,"fps":30}}' \
        --policy.type=openpi \
        --policy.host=localhost --policy.port=8000 \
        --task="pick and place" --max_steps=300
"""

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Register camera / robot / policy types with draccus ChoiceRegistry
from robodeploy.cameras import CameraConfig  # noqa: F401, E402
from robodeploy.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401, E402
from robodeploy.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401, E402
from robodeploy.policy_clients import (  # noqa: F401, E402
    PolicyClientConfig,  # noqa: F401, E402
    openpi,
)
from robodeploy.robots import (  # noqa: F401, E402
    RobotConfig,  # noqa: F401, E402
    bi_s1_follower,
    bi_so100_follower,
    s1_follower,
    so100_follower,
)


@dataclass
class InspectConfig:
    """Configuration for policy action inspection."""

    # Robot (draccus ChoiceRegistry, use --robot.type=...)
    robot: RobotConfig | None = None

    # Policy client (draccus ChoiceRegistry, use --policy.type=...)
    policy: PolicyClientConfig | None = None

    # Task description
    task: str = "fold the box"

    # Control loop
    fps: float = 30.0
    max_steps: int = 300
    log_file: str = "policy_actions.jsonl"


def _stack_front_cameras(images: dict) -> dict:
    """Vertically stack front + front_1 (front_1 rotated 180° first), in-place."""
    if "front" in images and "front_1" in images:
        front = np.asarray(images["front"])
        front_1 = np.asarray(images["front_1"])
        images["front"] = np.concatenate([front, np.rot90(front_1, 2)], axis=0)
        del images["front_1"]
    return images


def numpy_to_action_dict(action_np: np.ndarray, action_features: dict) -> dict:
    """Convert policy output [D] to robot.send_action() dict format."""
    keys = list(action_features.keys())
    if len(action_np) != len(keys):
        raise ValueError(f"Action dim mismatch: policy={len(action_np)}, robot={len(keys)}")
    return {key: float(action_np[i]) for i, key in enumerate(keys)}


def run_inspect(cfg: InspectConfig) -> None:
    """Main inference loop."""
    from robodeploy.policy_clients import make_policy_client_from_config
    from robodeploy.robots import make_robot_from_config

    # Robot
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    logger.info(f"Robot '{robot.name}' connected")

    camera_names = list(getattr(robot, "cameras", {}).keys())
    action_features = robot.action_features
    action_keys = list(action_features.keys())

    # Policy
    policy = make_policy_client_from_config(cfg.policy)
    if not policy.connected:
        logger.error("Policy not connected, exiting.")
        sys.exit(1)

    # Log file
    log_path = Path(cfg.log_file)
    log_file = open(log_path, "w")  # noqa: SIM115
    logger.info(f"Logging actions to {log_path}")

    # ---------- inference loop ----------
    step = 0
    infer_count = 0
    chunk = None
    chunk_idx = 0
    infer_ms = 0.0
    last_chunk_full = None
    rate = 1.0 / cfg.fps

    print(f"\n{'='*50}")
    print(f"  Robot: {robot.name}  |  Policy: OpenPI")
    print(f"  Task: {cfg.task}  |  FPS: {cfg.fps}  |  Max steps: {cfg.max_steps}")
    print(f"  Log: {cfg.log_file}")
    print(f"{'='*50}\n")
    input("Press [Enter] to start...")

    try:
        while step < cfg.max_steps:
            t0 = time.perf_counter()

            obs = robot.get_observation()
            state = np.array([obs.get(k, 0.0) for k in action_keys], dtype=np.float64)
            images = {cam: np.asarray(obs[cam]) for cam in camera_names if cam in obs}
            _stack_front_cameras(images)

            # Re-infer when chunk exhausted
            if chunk is None or chunk_idx >= len(chunk):
                result = policy.infer(images, state, cfg.task)
                actions = result.get("actions")
                infer_ms = result.get("policy_timing", {}).get("infer_ms", 0.0)
                infer_count += 1
                if actions is not None and len(actions) > 0:
                    chunk = np.asarray(actions)
                    last_chunk_full = chunk.tolist()
                    chunk_idx = 0
                else:
                    chunk = None
                    chunk_idx = 0

            if chunk is not None and chunk_idx < len(chunk):
                action_np = chunk[chunk_idx]
                action_0 = action_np.tolist()
                act_dict = numpy_to_action_dict(action_np, action_features)
                sent = robot.send_action(act_dict)
                sent_action = {k: float(sent[k]) for k in sent} if sent else {}
                chunk_idx += 1
            else:
                action_0 = None
                sent_action = {}

            entry = {
                "step": step,
                "timestamp": time.time(),
                "action": action_0,
                "actions_full": last_chunk_full,
                "chunk_infer": infer_count - 1,
                "chunk_frame": chunk_idx - 1 if chunk_idx > 0 else 0,
                "sent_action": sent_action,
                "state": state.tolist(),
                "infer_ms": infer_ms,
            }
            log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            log_file.flush()

            if step % 30 == 0:
                a_preview = [f"{v:.4f}" for v in action_0[:6]] if action_0 else "None"
                print(f"[{step:4d}] infer=#{infer_count} chunk_frame={chunk_idx - 1} "
                      f"action[:6]=[{', '.join(a_preview)}]  infer_ms={infer_ms:.0f}ms")

            step += 1
            elapsed = time.perf_counter() - t0
            sleep_t = rate - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[Interrupted]")
    finally:
        log_file.close()
        robot.disconnect()
        print(f"\nDone. {step} steps logged to {log_path}")


def main() -> None:
    from robodeploy.configs.parser import wrap

    @wrap()
    def _main(cfg: InspectConfig) -> None:
        run_inspect(cfg)

    _main()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Replay actions from a LeRobot dataset parquet file to the robot.

Loads actions from a specific episode's parquet and sends them frame-by-frame
to verify action correctness. No inference, no teleop, no recording.

Usage:
    # Single arm
    python replay_actions.py \\
        --robot.type=s1_follower --robot.port=/dev/ttyUSB0 \\
        --data_dir=./s1_data/bi_s1_0525_full_fixed \\
        --episode=0 --fps=30

    # Bimanual
    python replay_actions.py \\
        --robot.type=bi_s1_follower \\
        --robot.left_arm_port=/dev/left_follower --robot.right_arm_port=/dev/right_follower \\
        --data_dir=./s1_data/bi_s1_0525_full_fixed \\
        --episode=0 --max_frames=100
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from robodeploy.robots import RobotConfig, make_robot_from_config  # noqa: F401, E402

# Register robot types with draccus ChoiceRegistry
from robodeploy.robots import (  # noqa: F401, E402
    bi_s1_follower,
    bi_so100_follower,
    s1_follower,
    so100_follower,
)


@dataclass
class ReplayConfig:
    robot: RobotConfig | None = None
    data_dir: str = ""
    episode: int = 0
    fps: float = 30.0
    max_frames: int = 0


def numpy_to_action_dict(action_np: np.ndarray, action_features: dict) -> dict:
    keys = list(action_features.keys())
    if len(action_np) != len(keys):
        raise ValueError(f"Action dim mismatch: data={len(action_np)}, robot={len(keys)}")
    return {key: float(action_np[i]) for i, key in enumerate(keys)}


def run_replay(cfg: ReplayConfig) -> None:
    # Load parquet
    data_dir = Path(cfg.data_dir)
    parquet_path = data_dir / "data" / "chunk-000" / f"episode_{cfg.episode:06d}.parquet"
    if not parquet_path.exists():
        logger.error(f"Parquet not found: {parquet_path}")
        return

    df = pd.read_parquet(parquet_path, columns=["action"])
    actions = np.stack([np.asarray(v) for v in df["action"].values])
    total = len(actions)
    if cfg.max_frames > 0:
        actions = actions[:cfg.max_frames]
    logger.info(f"Loaded {len(actions)} actions from episode {cfg.episode} (total={total})")

    # Robot
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    logger.info(f"Robot '{robot.name}' connected")

    action_features = robot.action_features
    preview_keys = list(action_features.keys())[:6]

    # Confirm
    print(f"\n{'='*50}")
    print(f"  Replay: episode {cfg.episode} | {len(actions)} frames | {cfg.fps} fps")
    print(f"  Robot: {robot.name}")
    print(f"{'='*50}\n")
    input("Press [Enter] to start (robot will move)...")

    rate = 1.0 / cfg.fps

    try:
        for step, action_np in enumerate(actions):
            t0 = time.perf_counter()
            act_dict = numpy_to_action_dict(action_np, action_features)
            sent = robot.send_action(act_dict)

            if step % 30 == 0:
                preview = [f"{action_np[i]:.4f}" for i in range(min(6, len(action_np)))]
                data_keys = list(action_features.keys())
                sent_preview = [f"{sent.get(k, 0):.4f}" for k in data_keys[:3]]
                print(f"[{step:4d}/{len(actions)}] "
                      f"action[:6]=[{', '.join(preview)}]  sent[:3]=[{', '.join(sent_preview)}]")

            elapsed = time.perf_counter() - t0
            sleep_t = rate - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[Interrupted]")
    finally:
        robot.disconnect()
        print(f"\nDone. {step + 1} actions sent.")


def main() -> None:
    from robodeploy.configs.parser import wrap

    @wrap()
    def _main(cfg: ReplayConfig) -> None:
        run_replay(cfg)

    _main()


if __name__ == "__main__":
    main()

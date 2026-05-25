#!/usr/bin/env python
"""
Teleoperation-only data recording script with memory-efficient image writing.

Unlike record_s1_inference.py which accumulates all video frames in RAM before
encoding (causing OOM on small-memory machines), this script writes each camera
image to disk immediately via AsyncImageWriter background threads. Only file
paths are kept in the episode buffer, so RAM usage is constant regardless of
episode length or camera count.

Usage:
    # Bimanual S1
    python scripts/record_teleop.py \
        --robot.type=bi_s1_follower \
        --robot.left_arm_port=/dev/ttyUSB0 --robot.right_arm_port=/dev/ttyUSB1 \
        --robot.cameras='{"front":{"type":"intelrealsense","width":848,"height":480,"fps":30}}' \
        --teleop.type=bi_s1_leader \
        --teleop.left_arm_port=/dev/ttyUSB2 --teleop.right_arm_port=/dev/ttyUSB3 \
        --repo_id=my_dataset --task="fold the box"

    # Single arm S1
    python scripts/record_teleop.py \
        --robot.type=s1_follower --robot.port=/dev/ttyUSB0 \
        --robot.cameras='{"front":{"type":"intelrealsense","width":848,"height":480,"fps":30}}' \
        --teleop.type=s1_leader --teleop.port=/dev/ttyUSB1 \
        --repo_id=my_dataset --task="pick and place"
"""

import logging
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Register robot / teleop types with draccus ChoiceRegistry
from robodeploy.cameras import CameraConfig  # noqa: F401
from robodeploy.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from robodeploy.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from robodeploy.robots import (  # noqa: F401
    bi_s1_follower,
    bi_so100_follower,
    s1_follower,
    so100_follower,
)
from robodeploy.teleoperators import (  # noqa: F401
    bi_s1_leader,
    bi_so100_leader,
    s1_leader,
    so100_leader,
)
from robodeploy.robots import make_robot_from_config, RobotConfig
from robodeploy.teleoperators import make_teleoperator_from_config, TeleoperatorConfig
from robodeploy.datasets.lerobot_dataset import LeRobotDataset
from robodeploy.datasets.utils import build_dataset_frame, hw_to_dataset_features


@dataclass
class TeleopRecordConfig:
    robot: RobotConfig
    teleop: TeleoperatorConfig

    # Dataset
    output_dir: str = "./s1_data"
    repo_id: str = "dataset"
    task: str = "task"
    fps: int = 30
    episode_time_s: float = 120.0

    # Image writer tuning (0 processes = threads only, recommended)
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4

    # Allow recording more episodes after the first
    allow_multiple_episodes: bool = True


def _get_keypress() -> str | None:
    """Non-blocking single-key read from stdin."""
    if select.select([sys.stdin], [], [], 0.0)[0]:
        return sys.stdin.read(1)
    return None


def _prompt_label() -> int:
    """Block until user labels the episode. Returns 1=success, 0=failure, -1=discard."""
    print()
    print("=" * 50)
    print("  Label: [1] Success  |  [0] Failure  |  [2] Discard")
    print("=" * 50)

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        # drain any buffered input
        while select.select([sys.stdin], [], [], 0.0)[0]:
            sys.stdin.read(1)
        result = None
        while result is None:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                c = sys.stdin.read(1)
                if c == "1":
                    print("  => SUCCESS")
                    result = 1
                elif c == "0":
                    print("  => FAILURE")
                    result = 0
                elif c == "2":
                    print("  => DISCARDED")
                    result = -1
                else:
                    print(f"  Invalid: '{c}' — press 1, 0, or 2")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return result


def run_record(cfg: TeleopRecordConfig) -> None:
    # --- Robot & Teleop ---
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    logger.info(f"Robot '{robot.name}' connected.")

    teleop = make_teleoperator_from_config(cfg.teleop)
    teleop.connect()
    logger.info(f"Teleop '{teleop.name}' connected.")

    # --- Dataset features ---
    action_fts = hw_to_dataset_features(robot.action_features, "action", use_video=True)
    obs_fts = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)
    dataset_features = {**action_fts, **obs_fts}

    num_cameras = len(getattr(robot, "cameras", {}))
    num_threads = cfg.num_image_writer_threads_per_camera * max(num_cameras, 1)

    dataset_root = Path(cfg.output_dir) / cfg.repo_id

    dataset = LeRobotDataset.create(
        cfg.repo_id,
        cfg.fps,
        root=str(dataset_root),
        robot_type=robot.name,
        features=dataset_features,
        use_videos=True,
        image_writer_processes=cfg.num_image_writer_processes,
        image_writer_threads=num_threads,
    )
    logger.info(f"Dataset created at {dataset_root}")

    # --- Terminal setup ---
    fd = sys.stdin.fileno()
    old_termios = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    # --- State ---
    recording = False
    episode_start_t = 0.0
    saved_episodes = 0
    camera_names = list(getattr(robot, "cameras", {}).keys())

    def _buffer_size() -> int:
        if dataset.episode_buffer is None:
            return 0
        return dataset.episode_buffer.get("size", 0)

    def _stop_recording(discard: bool = True) -> None:
        nonlocal recording
        if recording:
            recording = False
            n = _buffer_size()
            print(f"\n[Recording] OFF  ({n} frames {'discarded' if discard else 'buffered'})")
            if discard and n > 0:
                dataset.clear_episode_buffer()

    def _save_episode() -> None:
        nonlocal recording, saved_episodes
        recording = False

        n_frames = _buffer_size()
        if n_frames == 0:
            print("[Save] No frames to save.")
            return

        print(f"\n[Save] {n_frames} frames captured.")

        label = _prompt_label()
        if label < 0:
            print("[Save] Discarded episode.")
            dataset.clear_episode_buffer()
            return

        dataset.save_episode()
        saved_episodes += 1
        print(f"[Save] Episode {saved_episodes - 1} saved (success={label}). Ready for next episode.")

    # --- Print header ---
    print("\n" + "=" * 60)
    print(f"  Robot:   {robot.name}")
    print(f"  Teleop:  {teleop.name}")
    print(f"  Cameras: {camera_names}")
    print(f"  Task:    {cfg.task}")
    print(f"  Output:  {dataset_root}")
    print(f"  FPS:     {cfg.fps}  |  Episode limit: {cfg.episode_time_s}s")
    print(f"  Controls: R=rec  S=save+label  Esc=quit")
    print("=" * 60 + "\n")
    input("Press [Enter] to start...")

    print(f"[Ready] Press R to start recording episode {saved_episodes}.")

    try:
        while True:
            k = _get_keypress()
            if k is not None:
                if k == '\x1b' or k == '\x03':  # Esc or Ctrl-C
                    _stop_recording()
                    print("\n[Exit]")
                    break
                elif k == 'r':
                    if recording:
                        _stop_recording(discard=True)
                    else:
                        recording = True
                        episode_start_t = time.perf_counter()
                        print(f"[Recording] ON  (episode {saved_episodes})")
                elif k == 's':
                    _save_episode()
                    if not cfg.allow_multiple_episodes and saved_episodes > 0:
                        print("[Done] Single episode mode.")
                        break

            if recording:
                loop_start = time.perf_counter()

                observation = robot.get_observation()
                teleop_action = teleop.get_action()
                sent_action = robot.send_action(teleop_action)

                obs_frame = build_dataset_frame(dataset.features, observation, "observation")
                action_frame = build_dataset_frame(dataset.features, sent_action, "action")
                frame = {**obs_frame, **action_frame}
                dataset.add_frame(frame, task=cfg.task)

                elapsed = time.perf_counter() - episode_start_t
                if elapsed >= cfg.episode_time_s:
                    print(f"\n[Time limit] {cfg.episode_time_s}s reached.")
                    _save_episode()
                    if not cfg.allow_multiple_episodes:
                        break
                    print(f"[Ready] Press R to record episode {saved_episodes}.")

                # Maintain target FPS
                dt = time.perf_counter() - loop_start
                sleep_t = 1.0 / cfg.fps - dt
                if sleep_t > 0:
                    time.sleep(sleep_t)

                # Periodic status
                n = _buffer_size()
                if recording and n > 0 and n % 60 == 0:
                    print(f"[REC] episode={saved_episodes} frames={n} elapsed={elapsed:.0f}s")
            else:
                time.sleep(0.02)  # Idle polling

    except KeyboardInterrupt:
        print("\n[Interrupted]")
        _stop_recording()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_termios)

        # Wait for async image writer to finish, then disconnect
        print("[Shutdown] Waiting for image writer...")
        if dataset.image_writer is not None:
            dataset.image_writer.stop()
            dataset.image_writer = None
        robot.disconnect()
        teleop.disconnect()
        print(f"[Shutdown] Output: {dataset_root}")
        print("[Shutdown] Done.")


def main() -> None:
    from robodeploy.configs.parser import wrap

    @wrap()
    def _main(cfg: TeleopRecordConfig) -> None:
        run_record(cfg)

    _main()


if __name__ == "__main__":
    main()

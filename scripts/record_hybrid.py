#!/usr/bin/env python
"""
Unified data collection + inference script using HybridLeRobotWriter (SharedMemory + native LeRobot).

Based on record_s1_inference.py but replaces BackgroundLeRobotWriter with HybridLeRobotWriter
for zero-copy IPC and native video encoding/stats computation.

Features:
  1. Teleoperation recording with background encoding via SharedMemory (async, non-blocking)
  2. OpenPI policy inference with temporal smoothing (StreamActionBuffer)
  3. Success/failure labeling via keyboard (stored as is_failure_data field)
  4. Teleop/inference mode switching + is_infer_data field
  5. Leader-follower alignment on inference->teleop switch (cosine interpolation)
  6. Native LeRobot video encoding (encode_video_frames) and stats (compute_episode_stats)

Usage:
    # S1 bimanual (cameras configured via robot.cameras)
    python record_hybrid.py \
        --robot.type=bi_s1_follower \
        --robot.left_arm_port=/dev/ttyUSB0 --robot.right_arm_port=/dev/ttyUSB1 \
        --robot.cameras='{"front":{"type":"intelrealsense","width":848,"height":480,"fps":30}}' \
        --teleop.type=bi_s1_leader \
        --teleop.left_arm_port=/dev/ttyUSB2 --teleop.right_arm_port=/dev/ttyUSB3 \
        --policy.type=openpi \
        --policy.host=localhost --policy.port=8000 \
        --task="fold the box"

    # SO100 bimanual (just change type)
    python record_hybrid.py \
        --robot.type=bi_so100_follower --teleop.type=bi_so100_leader \
        ...

    # Single arm S1
    python record_hybrid.py \
        --robot.type=s1_follower \
        --robot.port=/dev/ttyUSB0 \
        --robot.cameras='{"front":{"type":"intelrealsense","width":848,"height":480,"fps":30}}' \
        --teleop.type=s1_leader \
        --teleop.port=/dev/ttyUSB1 \
        --policy.type=openpi \
        --policy.host=localhost --policy.port=8000 \
        --task="pick and place"
"""

import json
import logging
import shutil
import sys
import tempfile
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Register camera / robot / teleop / policy types with draccus ChoiceRegistry
from lerobot_mini.cameras import CameraConfig  # noqa: F401, E402
from lerobot_mini.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401, E402
from lerobot_mini.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401, E402
from lerobot_mini.robots import (  # noqa: F401, E402
    bi_s1_follower,
    bi_so100_follower,
    s1_follower,
    so100_follower,
)
from lerobot_mini.teleoperators import (  # noqa: F401, E402
    bi_s1_leader,
    bi_so100_leader,
    s1_leader,
    so100_leader,
)
from lerobot_mini.policy_clients import (  # noqa: F401, E402
    openpi,
)
from lerobot_mini.webui.server import WebUIServer
from lerobot_mini.utils.stream_buffer import StreamActionBuffer  # noqa: F401, E402
from lerobot_mini.utils.leader_follower_align import (  # noqa: F401, E402
    interpolate_leader_to_follower,
    reset_to_zero,
)
from lerobot_mini.utils.keyboard_control import get_keypress, prompt_success_failure  # noqa: F401, E402

# NEW: Hybrid writer with SharedMemory
from lerobot_mini.datasets.async_writer import HybridLeRobotWriter, CustomFeatureConfig  # noqa: E402

SENTINEL = None


class ControlMode(Enum):
    TELEOP = "teleop"
    POLICY = "policy"
    MIXED = "mixed"


# ==============================================================================
# Adapter: numpy policy output -> robot action dict
# ==============================================================================

def numpy_to_action_dict(action_np: np.ndarray, action_features: dict[str, type]) -> dict[str, float]:
    """Convert policy output [D] numpy array to robot.send_action() dict format."""
    keys = list(action_features.keys())
    if len(action_np) != len(keys):
        raise ValueError(f"Action dim mismatch: policy={len(action_np)}, robot={len(keys)}")
    return {key: float(action_np[i]) for i, key in enumerate(keys)}


# ==============================================================================
# Episode buffer (in-memory, flushes to HybridLeRobotWriter)
# ==============================================================================

class EpisodeBuffer:
    """Collects frames in memory, flushes to background writer on episode end."""

    def __init__(self, camera_names: list[str]):
        self.camera_names = camera_names
        self.images: dict[str, list] = {cam: [] for cam in camera_names}
        self.states: list = []
        self.actions: list = []
        self.is_infer_data: list = []
        self.label: int = 1  # 1=success, 0=failure; default success

    def add(
        self,
        observation: dict,
        sent_action: dict,
        action_features: dict[str, type],
        is_inference_val: int,
    ) -> None:
        for cam in self.camera_names:
            if cam in observation:
                img = np.asarray(observation[cam])
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                self.images[cam].append(img.copy())
            else:
                if self.images[cam]:
                    self.images[cam].append(np.zeros_like(self.images[cam][-1]))
        state = np.array([observation.get(k, 0.0) for k in action_features], dtype=np.float64)
        action = np.array([sent_action.get(k, 0.0) for k in action_features], dtype=np.float64)
        self.states.append(state)
        self.actions.append(action)
        self.is_infer_data.append(is_inference_val)

    def reset(self) -> None:
        self.images = {cam: [] for cam in self.camera_names}
        self.states = []
        self.actions = []
        self.is_infer_data = []
        self.label = 1

    @property
    def frame_count(self) -> int:
        return len(self.actions)

    def to_episode_data(self) -> dict:
        """Build episode_data dict for HybridLeRobotWriter.submit()."""
        return {
            "images": {cam: np.stack(frames) for cam, frames in self.images.items() if frames},
            "observation.state": np.stack(self.states) if self.states else np.empty((0,)),
            "action": np.stack(self.actions) if self.actions else np.empty((0,)),
            "is_inference": np.array(self.is_infer_data, dtype=np.int64),
            "label": self.label,
        }


# ==============================================================================
# Inference thread
# ==============================================================================

def _start_inference_thread(
    policy,
    buffer: StreamActionBuffer,
    state_ref: dict,
    recording_ref: dict,
    action_features: dict[str, type],
    camera_names: list[str],
    task: str,
    inference_rate: float,
    latency_k: int,
    min_smooth_steps: int,
) -> threading.Thread:
    """Start a daemon thread for async policy inference."""

    def _run() -> None:
        rate = 1.0 / inference_rate
        was_recording = False
        while not state_ref["stop"]:
            if state_ref["mode"] != ControlMode.POLICY:
                was_recording = False
                time.sleep(0.1)
                continue

            is_recording = recording_ref.get("recording", False)
            if not is_recording:
                if was_recording:
                    buffer.clear()
                was_recording = False
                time.sleep(0.1)
                continue

            obs = state_ref.get("obs")
            if obs is None:
                time.sleep(0.1)
                continue

            try:
                state = np.array([obs.get(k, 0.0) for k in action_features], dtype=np.float64)
                images = {cam: np.asarray(obs[cam]) for cam in camera_names if cam in obs}

                result = policy.infer(images, state, task)
                actions = result.get("actions", None)
                if actions is not None and len(actions) > 0:
                    buffer.integrate_new_chunk(
                        np.asarray(actions), max_k=latency_k, min_m=min_smooth_steps,
                    )
                state_ref["inference_ok"] = True
            except Exception as e:
                logger.warning(f"Inference error: {e}")
                state_ref["inference_ok"] = False
            time.sleep(rate)

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    return th


# ==============================================================================
# Main record loop
# ==============================================================================

def record_loop(
    robot,
    fps: int,
    control_time_s: float,
    teleop,
    stream_buffer: StreamActionBuffer | None,
    episode_buffer: EpisodeBuffer,
    action_features: dict[str, type],
    state_ref: dict,
    recording_ref: dict,
    stop_ref: dict,
    obs_lock: threading.Lock,
    on_key: callable = None,
) -> None:
    """Main control loop using lerobot abstract interfaces."""
    start_episode_t = time.perf_counter()
    timestamp = 0.0
    was_recording = recording_ref.get("recording", False)

    while timestamp < control_time_s and not stop_ref["stop"]:
        start_loop_t = time.perf_counter()

        is_recording = recording_ref.get("recording", False)
        if is_recording and not was_recording:
            start_episode_t = time.perf_counter()
            timestamp = 0.0
        was_recording = is_recording

        observation = robot.get_observation()
        with obs_lock:
            state_ref["obs"] = observation
        if recording_ref.get("recording", False):
            recording_ref["frames"] = episode_buffer.frame_count

        if on_key is not None:
            k = get_keypress()
            if k is not None:
                on_key(k)
            if stop_ref["stop"]:
                break

        teleop_action = None
        if teleop is not None:
            teleop_action = teleop.get_action()

        is_inference = 0
        action = None
        if state_ref["mode"] == ControlMode.POLICY and stream_buffer is not None:
            act_np = stream_buffer.pop_next_action()
            if act_np is not None:
                action = numpy_to_action_dict(act_np, action_features)
                is_inference = 1
        elif teleop_action is not None and recording_ref.get("recording", False):
            action = teleop_action

        if action is not None:
            sent_action = robot.send_action(action)
            if recording_ref.get("recording", False):
                episode_buffer.add(observation, sent_action, action_features, is_inference)

        dt_s = time.perf_counter() - start_loop_t
        sleep_time = 1.0 / fps - dt_s
        if sleep_time > 0:
            time.sleep(sleep_time)
        timestamp = time.perf_counter() - start_episode_t

        fc = episode_buffer.frame_count
        if recording_ref.get("recording", False) and fc > 0 and fc % 60 == 0:
            mode_str = "POL" if state_ref["mode"] == ControlMode.POLICY else "TEL"
            switch_hint = "P=switch " if state_ref["control_mode"] == ControlMode.MIXED else ""
            inf_ok = " ERR!" if not state_ref.get("inference_ok", True) else ""
            ep = recording_ref.get("episode", 0)
            print(f"[{mode_str}{inf_ok}] ep={ep} frames={fc} elapsed={timestamp:.0f}s | {switch_hint}R=rec S=save Esc=quit")


# ==============================================================================
# Main entry
# ==============================================================================

def run_record(cfg) -> None:
    """Main entry point, receives a RecordConfig from draccus."""
    from lerobot_mini.robots import make_robot_from_config
    from lerobot_mini.teleoperators import make_teleoperator_from_config
    from lerobot_mini.policy_clients import make_policy_client_from_config
    from lerobot_mini.datasets.utils import hw_to_dataset_features

    # Create robot
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    logger.info(f"Robot '{robot.name}' connected.")
    is_bimanual = cfg.robot.type.startswith("bi_")

    # Determine control mode
    control_mode = ControlMode(cfg.control_mode)

    # Create teleop -- skip for pure POLICY mode
    teleop = None
    if control_mode != ControlMode.POLICY and cfg.teleop is not None:
        teleop = make_teleoperator_from_config(cfg.teleop)
        teleop.connect()
        logger.info(f"Teleop '{teleop.name}' connected.")
    elif control_mode == ControlMode.POLICY:
        logger.info("Pure POLICY mode, skipping teleop connection.")

    # Detect camera names from robot
    camera_names = list(getattr(robot, "cameras", {}).keys())

    # Build dataset features from hardware features
    action_features = robot.action_features
    action_fts = hw_to_dataset_features(action_features, "action", use_video=True)
    obs_fts = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)

    # Add video info sub-dicts
    for key, feat in obs_fts.items():
        if feat.get("dtype") == "video":
            cam_name = key.rsplit(".", 1)[-1]
            cam = robot.cameras.get(cam_name)
            if cam is not None:
                h = getattr(cam, "height", 480)
                w = getattr(cam, "width", 640)
                feat["info"] = {
                    "video.height": h,
                    "video.width": w,
                    "video.codec": "av1",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": cfg.fps,
                    "video.channels": 3,
                    "has_audio": False,
                }

    # Define custom features using CustomFeatureConfig
    # These replace the hardcoded standard_fts from record_s1_inference.py
    custom_features = [
        CustomFeatureConfig(
            name="is_failure_data",
            dtype="int64",
            shape=(1,),
            value_fn=lambda ed: 1 - ed.get("label", 1),  # label=1 success -> is_failure=0
        ),
        CustomFeatureConfig(
            name="is_infer_data",
            dtype="int64",
            shape=(1,),
            value_fn=lambda ed: ed.get("is_inference", np.zeros(len(ed.get("action", [])), dtype=np.int64)),
        ),
    ]

    info_features = {**action_fts, **obs_fts}

    # Hybrid writer with SharedMemory
    writer = HybridLeRobotWriter(
        output_dir=cfg.output_dir,
        info_features=info_features,
        fps=cfg.fps,
        repo_id=cfg.repo_id,
        robot_type=robot.name,
        custom_features=custom_features,
        video_codec="libsvtav1",
        video_crf=30,
        video_pix_fmt="yuv420p",
        video_gop=2,
    )

    # Policy client -- skip for pure TELEOP mode
    policy = None
    if control_mode != ControlMode.TELEOP and cfg.policy is not None:
        policy = make_policy_client_from_config(cfg.policy)

    # Initial effective mode
    if control_mode == ControlMode.TELEOP:
        initial_mode = ControlMode.TELEOP
    elif control_mode == ControlMode.POLICY:
        initial_mode = ControlMode.POLICY
    else:
        initial_mode = ControlMode.TELEOP if cfg.control_mode_initial == "teleop" else ControlMode.POLICY

    # State references
    state_ref = {
        "obs": None,
        "control_mode": control_mode,
        "mode": initial_mode,
        "stop": False,
        "inference_ok": True,
    }
    recording_ref = {"recording": False, "episode": 0, "frames": 0}
    stop_ref = {"stop": False}
    obs_lock = threading.Lock()

    # Stream buffer
    stream_buffer = StreamActionBuffer(
        state_dim=len(action_features)
    ) if (cfg.use_temporal_smoothing and control_mode != ControlMode.TELEOP) else None

    # Episode buffer
    episode_buffer = EpisodeBuffer(camera_names)

    # Start inference thread
    inference_thread = None
    if stream_buffer is not None and policy is not None and policy.connected:
        inference_thread = _start_inference_thread(
            policy=policy,
            buffer=stream_buffer,
            state_ref=state_ref,
            recording_ref=recording_ref,
            action_features=action_features,
            camera_names=camera_names,
            task=cfg.task,
            inference_rate=cfg.inference_rate,
            latency_k=cfg.latency_k,
            min_smooth_steps=cfg.min_smooth_steps,
        )

    # Guard: fail fast if required component is missing
    if control_mode == ControlMode.POLICY and (policy is None or not policy.connected):
        logger.error("POLICY mode requires a running policy server. Exiting.")
        sys.exit(1)
    if control_mode == ControlMode.TELEOP and teleop is None:
        logger.error("TELEOP mode requires a teleoperator. Exiting.")
        sys.exit(1)

    # WebUI server
    webui = None
    if cfg.webui_port > 0:
        def _cmd_switch_mode(_data=None):
            if state_ref["control_mode"] != ControlMode.MIXED:
                return {"error": f"Fixed to {state_ref['control_mode'].value.upper()}"}
            if state_ref["mode"] == ControlMode.POLICY:
                if teleop is not None:
                    leader_pos = teleop.get_action()
                    follower_pos = state_ref.get("obs") or robot.get_observation()
                    interpolate_leader_to_follower(
                        teleop, leader_pos, follower_pos, action_features,
                        max_step=cfg.align_max_step,
                    )
                state_ref["mode"] = ControlMode.TELEOP
                if stream_buffer:
                    stream_buffer.clear()
                print("[Mode] TELEOP")
            else:
                state_ref["mode"] = ControlMode.POLICY
                if stream_buffer:
                    stream_buffer.clear()
                print("[Mode] POLICY")
            return None

        def _cmd_toggle_record(_data=None):
            recording_ref["recording"] = not recording_ref["recording"]
            if recording_ref["recording"]:
                if stream_buffer:
                    stream_buffer.clear()
                print(f"[Recording] ON  (episode {recording_ref['episode']})")
                if webui is not None:
                    webui.on_recording_started()
            else:
                print("[Recording] OFF")
                if webui is not None:
                    webui.on_recording_stopped()
            return None

        def _cmd_save(data):
            label = data.get("label", -1) if data else -1
            if recording_ref.get("recording", False) and episode_buffer.frame_count > 0:
                recording_ref["recording"] = False
                print(f"\n[Save] Episode {recording_ref['episode']}: {episode_buffer.frame_count} frames")
                if webui is not None:
                    webui.on_recording_stopped()
                if label >= 0:
                    episode_buffer.label = label
                    writer.submit(episode_buffer.to_episode_data(), cfg.task)
                    if webui is not None:
                        webui.on_episode_saved(recording_ref["episode"], episode_buffer.frame_count, label)
                    recording_ref["episode"] += 1
                    print(f"[Save] Submitted (success={label}). Ready for next episode.")
                else:
                    print("[Save] Discarded.")
                episode_buffer.reset()
            else:
                print("[Save] Nothing to save.")
            return None

        def _cmd_reset_zero(_data=None):
            if recording_ref.get("recording", False):
                print("[Reset] Cannot reset while recording. Stop recording first.")
                return {"error": "Cannot reset while recording"}
            print("[Reset] Moving arms to zero position...")
            if stream_buffer:
                stream_buffer.clear()
            reset_to_zero(robot, teleop, action_features, max_step=cfg.align_max_step)
            return None

        def _cmd_stop(_data=None):
            print("\n[Exit]")
            stop_ref["stop"] = True
            return None

        command_handlers = {
            "switch_mode": _cmd_switch_mode,
            "toggle_record": _cmd_toggle_record,
            "save": _cmd_save,
            "reset_zero": _cmd_reset_zero,
            "stop": _cmd_stop,
        }

        webui = WebUIServer(
            state_ref=state_ref,
            recording_ref=recording_ref,
            stop_ref=stop_ref,
            obs_lock=obs_lock,
            camera_names=camera_names,
            port=cfg.webui_port,
            command_handlers=command_handlers,
        )
        webui.start()
        logger.info(f"WebUI started at http://0.0.0.0:{cfg.webui_port}")

    # Keyboard handling
    import termios as _termios
    import tty as _tty

    _stdin_fd = sys.stdin.fileno()
    _old_termios = None

    def handle_keypress(k: str):
        nonlocal state_ref, recording_ref, stop_ref, teleop, robot, is_bimanual, action_features
        try:
            if k == '\x1b' or k == '\x03':
                print("\n[Exit]")
                stop_ref["stop"] = True
            elif k == 'p' or k == '\t':
                if state_ref["control_mode"] != ControlMode.MIXED:
                    print(f"[Mode] Fixed to {state_ref['control_mode'].value.upper()}, switching disabled.")
                elif state_ref["mode"] == ControlMode.POLICY:
                    if teleop is not None:
                        leader_pos = teleop.get_action()
                        follower_pos = state_ref.get("obs") or robot.get_observation()
                        interpolate_leader_to_follower(
                            teleop, leader_pos, follower_pos, action_features,
                            max_step=cfg.align_max_step,
                        )
                    state_ref["mode"] = ControlMode.TELEOP
                    if stream_buffer:
                        stream_buffer.clear()
                    print(f"[Mode] TELEOP")
                else:
                    state_ref["mode"] = ControlMode.POLICY
                    if stream_buffer:
                        stream_buffer.clear()
                    print(f"[Mode] POLICY")
            elif k == 'r':
                recording_ref["recording"] = not recording_ref["recording"]
                if recording_ref["recording"]:
                    if stream_buffer:
                        stream_buffer.clear()
                    print(f"[Recording] ON  (episode {recording_ref['episode']})")
                else:
                    print(f"[Recording] OFF")
            elif k == 's':
                if recording_ref.get("recording", False) and episode_buffer.frame_count > 0:
                    recording_ref["recording"] = False
                    print(f"\n[Save] Episode {recording_ref['episode']}: {episode_buffer.frame_count} frames")
                    label = prompt_success_failure()
                    if label >= 0:
                        episode_buffer.label = label
                        writer.submit(episode_buffer.to_episode_data(), cfg.task)
                        recording_ref["episode"] += 1
                        print(f"[Save] Submitted (success={label}). Ready for next episode.")
                    else:
                        print("[Save] Discarded.")
                    episode_buffer.reset()
                else:
                    print("[Save] Nothing to save.")
            elif k == 'z':
                if recording_ref.get("recording", False):
                    print("[Reset] Cannot reset while recording. Stop recording first.")
                else:
                    print("[Reset] Moving arms to zero position...")
                    if stream_buffer:
                        stream_buffer.clear()
                    reset_to_zero(robot, teleop, action_features, max_step=cfg.align_max_step)
        except Exception as e:
            print(f"Key error: {e}")

    # Main loop
    switch_hint = "P=switch  " if state_ref["control_mode"] == ControlMode.MIXED else ""
    print("\n" + "=" * 60)
    print(f"  Robot: {robot.name}  |  Teleop: {teleop.name if teleop else 'None'}")
    print(f"  Control: {state_ref['control_mode'].value.upper()}  |  Start: {state_ref['mode'].value.upper()}")
    print(f"  Policy: {'Connected' if (policy and policy.connected) else 'N/A'}  |  Output: {cfg.output_dir}")
    print(f"  Task: {cfg.task}")
    print(f"  Controls: {switch_hint}R=rec  S=save+label  Z=zero-reset  Esc=exit")
    print("=" * 60 + "\n")
    input("Press [Enter] to start...")

    _old_termios = _termios.tcgetattr(_stdin_fd)
    _tty.setcbreak(_stdin_fd)

    print(f"[Control] {state_ref['control_mode'].value.upper()}  [Start] {state_ref['mode'].value.upper()}")
    print(f"[Recording] OFF  (next episode {recording_ref['episode']})")

    try:
        while not stop_ref["stop"]:
            record_loop(
                robot=robot,
                fps=cfg.fps,
                control_time_s=cfg.episode_time_s,
                teleop=teleop,
                stream_buffer=stream_buffer,
                episode_buffer=episode_buffer,
                action_features=action_features,
                state_ref=state_ref,
                recording_ref=recording_ref,
                stop_ref=stop_ref,
                obs_lock=obs_lock,
                on_key=handle_keypress,
            )
            if recording_ref.get("recording", False) and episode_buffer.frame_count > 0:
                recording_ref["recording"] = False
                print(f"\n[Auto-save] Episode {recording_ref['episode']}: {episode_buffer.frame_count} frames (time limit)")
                label = prompt_success_failure()
                if label >= 0:
                    episode_buffer.label = label
                    writer.submit(episode_buffer.to_episode_data(), cfg.task)
                    recording_ref["episode"] += 1
                episode_buffer.reset()
    except KeyboardInterrupt:
        print("\n[Interrupted]")
    finally:
        print("\n" + "=" * 50)
        print("  Shutting down...")
        print("=" * 50)
        state_ref["stop"] = True
        if webui is not None:
            webui.stop()
        if inference_thread is not None:
            inference_thread.join(timeout=5)
        if _old_termios is not None:
            _termios.tcsetattr(_stdin_fd, _termios.TCSADRAIN, _old_termios)
        writer.shutdown()
        robot.disconnect()
        if teleop is not None:
            teleop.disconnect()
        print(f"  Output: {cfg.output_dir}/{cfg.repo_id}")
        print(f"  Log:    {cfg.output_dir}/{cfg.repo_id}/writer.log")
        print("  Done.\n")


def main() -> None:
    from lerobot_mini.configs.parser import wrap
    from scripts.record_config import RecordConfig

    @wrap()
    def _main(cfg: RecordConfig) -> None:
        run_record(cfg)

    _main()


if __name__ == "__main__":
    main()

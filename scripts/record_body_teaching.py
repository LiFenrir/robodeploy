#!/usr/bin/env python
"""
Body-teaching data collection + inference script with NPY-based storage backend.

Same features as record_s1_inference_npy.py but designed for body-teaching robots
(本体示教) where the same physical arm serves as both the teaching device (human
backdrives it in gravity compensation mode) and the execution device (policy
commands joint positions). No separate teleoperator hardware needed.

The robot config's `mode` field controls behavior:
  - "collect": gravity compensation ON, arm backdrivable by human
  - "control": gravity compensation OFF, arm follows position commands

Features:
  1. Body-teaching recording with memory-efficient NPY -> MP4 encoding
  2. OpenPI policy inference with temporal smoothing (StreamActionBuffer)
  3. Success/failure labeling via keyboard (stored as is_failure_data field, 1=failure)
  4. Collect/policy mode switching + recording toggle + is_infer_data field (1=inference)
  5. No leader-follower alignment needed (same arm for teach and execute)

Usage:
    # Bimanual ARX X5 body teaching with 3 RealSense cameras
    python record_body_teaching.py \
        --robot.type=bi_arx_x5 \
        --robot.left_can_port=can0 --robot.right_can_port=can1 \
        --robot.mode=collect \
        --robot.cameras='{"top":{"type":"intelrealsense","width":848,"height":480,"fps":30,"serial_number_or_name":"135"},"left_hand":{"type":"intelrealsense","width":848,"height":480,"fps":30,"serial_number_or_name":"260"},"right_hand":{"type":"intelrealsense","width":848,"height":480,"fps":30,"serial_number_or_name":"352"}}' \
        --policy.type=openpi \
        --policy.host=localhost --policy.port=8000 \
        --task="fold the box"

    # Single arm ARX X5
    python record_body_teaching.py \
        --robot.type=arx_x5 \
        --robot.can_port=can0 \
        --robot.mode=collect \
        --robot.cameras='{"front":{"type":"intelrealsense","width":848,"height":480,"fps":30,"serial_number_or_name":"123"}}' \
        --task="pick and place"
"""

# ── 在任何 import 之前：仅过滤 C++ .so 输出中的 "ARX方舟无限" 噪声 ──
import os as _os
import threading as _threading

_REAL_STDERR_FD = _os.dup(2)

_rfd, _wfd = _os.pipe()


def _filter_stderr():
    with _os.fdopen(_rfd, "r", buffering=1, errors="replace") as _reader:
        for _line in _reader:
            if "ARX方舟无限" not in _line:
                _os.write(_REAL_STDERR_FD, _line.encode(errors="replace"))


_threading.Thread(target=_filter_stderr, daemon=True).start()

_os.dup2(_wfd, 1)  # stdout → pipe (经 filter 线程)
_os.dup2(_wfd, 2)  # stderr → pipe
_os.close(_wfd)

import logging
import sys
import threading
import time
from enum import Enum
from pathlib import Path

# Add project root to sys.path so that `from scripts.xxx` imports work
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Register camera / robot / policy types with draccus ChoiceRegistry
from robodeploy.cameras import CameraConfig  # noqa: F401, E402
from robodeploy.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401, E402
from robodeploy.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401, E402
from robodeploy.robots import make_robot_from_config  # noqa: F401, E402
from robodeploy.robots.arx_x5 import arx_x5, bi_arx_x5  # noqa: F401, E402
from robodeploy.policy_clients import (  # noqa: F401, E402
    openpi,
)
from robodeploy.policy_clients.utils import make_policy_client_from_config  # noqa: F401, E402
from robodeploy.webui.server import WebUIServer
from robodeploy.utils.stream_buffer import StreamActionBuffer
from robodeploy.utils.leader_follower_align import reset_to_zero
from robodeploy.utils.keyboard_control import get_keypress, prompt_success_failure
from robodeploy.datasets.utils import build_dataset_frame, hw_to_dataset_features
from robodeploy.datasets.npy_backend import BackgroundVideoEncoder, LeRobotDatasetNPY


class ControlMode(Enum):
    COLLECT = "collect"
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
        while not state_ref["stop"]:
            if state_ref["mode"] != ControlMode.POLICY:
                time.sleep(0.1)
                continue

            is_recording = recording_ref.get("recording", False)
            if not is_recording:
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
    stream_buffer: StreamActionBuffer | None,
    dataset: LeRobotDatasetNPY,
    action_features: dict[str, type],
    state_ref: dict,
    recording_ref: dict,
    stop_ref: dict,
    obs_lock: threading.Lock,
    task: str,
    on_key: callable = None,
) -> None:
    """Main control loop for body-teaching robots with NPY dataset storage."""
    start_episode_t = time.perf_counter()
    timestamp = 0.0
    was_recording = recording_ref.get("recording", False)

    while timestamp < control_time_s and not stop_ref["stop"]:
        start_loop_t = time.perf_counter()

        is_recording = recording_ref.get("recording", False)
        if is_recording and (dataset.episode_buffer or {}).get("size", 0) == 0:
            start_episode_t = time.perf_counter()
        was_recording = is_recording

        observation = robot.get_observation()
        with obs_lock:
            state_ref["obs"] = observation
        if recording_ref.get("recording", False):
            recording_ref["frames"] = (dataset.episode_buffer or {}).get("size", 0)

        if on_key is not None:
            k = get_keypress()
            if k is not None:
                on_key(k)
            if stop_ref["stop"]:
                break

        is_inference = 0
        action = None
        if state_ref["mode"] == ControlMode.POLICY and stream_buffer is not None:
            act_np = stream_buffer.pop_next_action()
            if act_np is not None:
                action = numpy_to_action_dict(act_np, action_features)
                is_inference = 1
        elif state_ref["mode"] == ControlMode.COLLECT and recording_ref.get("recording", False):
            if hasattr(robot, "get_action"):
                action = robot.get_action()

        if action is not None:
            if is_inference:
                sent_action = robot.send_action(action)
            else:
                # Collect mode: arm is in gravity compensation, human moves it.
                # Don't send_action — it would fight the human.
                sent_action = action

            if recording_ref.get("recording", False):
                obs_frame = build_dataset_frame(dataset.features, observation, "observation")
                action_frame = build_dataset_frame(dataset.features, sent_action, "action")
                frame = {**obs_frame, **action_frame}
                frame["is_infer_data"] = np.array([is_inference], dtype=np.int64)
                frame["is_failure_data"] = np.array([0], dtype=np.int64)
                dataset.add_frame(frame, task=task)

        dt_s = time.perf_counter() - start_loop_t
        sleep_time = 1.0 / fps - dt_s
        if sleep_time > 0:
            time.sleep(sleep_time)
        timestamp = time.perf_counter() - start_episode_t

        fc = (dataset.episode_buffer or {}).get("size", 0)
        if recording_ref.get("recording", False) and fc > 0 and fc % 60 == 0:
            mode_str = "POL" if state_ref["mode"] == ControlMode.POLICY else "COL"
            switch_hint = "P=switch " if state_ref["control_mode"] == ControlMode.MIXED else ""
            inf_ok = " ERR!" if not state_ref.get("inference_ok", True) else ""
            ep = recording_ref.get("episode", 0)
            print(f"[{mode_str}{inf_ok}] ep={ep} frames={fc} elapsed={timestamp:.0f}s | {switch_hint}R=rec S=save Esc=quit")


# ==============================================================================
# Main entry
# ==============================================================================

def _dataset_buffer_size(dataset: LeRobotDatasetNPY) -> int:
    if dataset.episode_buffer is None:
        return 0
    return dataset.episode_buffer.get("size", 0)


def _save_dataset_episode(dataset: LeRobotDatasetNPY, label: int, task: str) -> None:
    """Fix is_failure_data in buffer, then save parquet + metadata (video encoding in background)."""
    if dataset.episode_buffer is None or _dataset_buffer_size(dataset) == 0:
        return
    n = len(dataset.episode_buffer["is_failure_data"])
    dataset.episode_buffer["is_failure_data"] = [np.array([1 - label], dtype=np.int64)] * n
    dataset.save_episode_async()


def _discard_dataset_episode(dataset: LeRobotDatasetNPY) -> None:
    """Discard current episode buffer and clean up NPY files."""
    if dataset.episode_buffer is not None and _dataset_buffer_size(dataset) > 0:
        dataset.clear_episode_buffer()


def run_record(cfg) -> None:
    """Main entry point, receives a RecordConfig from draccus."""
    from robodeploy.policy_clients.utils import make_policy_client_from_config

    # Create robot
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    logger.info(f"Robot '{robot.name}' connected (mode={cfg.robot.mode}).")

    # Determine control mode
    control_mode = ControlMode(cfg.control_mode)

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

    # Standard feature entries matching parquet columns
    standard_fts = {
        "timestamp": {"dtype": "float64", "shape": (1,), "names": None},
        "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
        "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
        "index": {"dtype": "int64", "shape": (1,), "names": None},
        "task_index": {"dtype": "int64", "shape": (1,), "names": None},
        "is_failure_data": {"dtype": "int64", "shape": (1,), "names": None},
        "is_infer_data": {"dtype": "int64", "shape": (1,), "names": None},
    }
    info_features = {**action_fts, **obs_fts, **standard_fts}

    # Create NPY-backed dataset
    dataset_root = Path(cfg.output_dir) / cfg.repo_id
    dataset = LeRobotDatasetNPY.create(
        cfg.repo_id,
        cfg.fps,
        root=str(dataset_root),
        robot_type=robot.name,
        features=info_features,
        use_videos=True,
    )
    logger.info(f"Dataset created at {dataset_root} (NPY backend)")

    # Create background video encoder and attach to dataset
    info_path = str(dataset_root / "meta" / "info.json")
    video_keys = dataset.meta.video_keys
    bg_encoder = BackgroundVideoEncoder(info_path, video_keys)
    dataset.set_video_encoder(bg_encoder)

    # Policy client — skip for pure COLLECT mode
    policy = None
    if control_mode != ControlMode.COLLECT and cfg.policy is not None:
        policy = make_policy_client_from_config(cfg.policy)

    # Initial effective mode
    if control_mode == ControlMode.COLLECT:
        initial_mode = ControlMode.COLLECT
    elif control_mode == ControlMode.POLICY:
        initial_mode = ControlMode.POLICY
    else:
        initial_mode = ControlMode.COLLECT if cfg.control_mode_initial == "collect" else ControlMode.POLICY

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
    ) if (cfg.use_temporal_smoothing and control_mode != ControlMode.COLLECT) else None

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
    if not hasattr(robot, "get_action"):
        logger.error("Robot does not support body teaching (no get_action method). Exiting.")
        sys.exit(1)

    # WebUI server
    webui = None
    if cfg.webui_port > 0:
        def _cmd_switch_mode(_data=None):
            if state_ref["control_mode"] != ControlMode.MIXED:
                return {"error": f"Fixed to {state_ref['control_mode'].value.upper()}"}
            if state_ref["mode"] == ControlMode.POLICY:
                # Switch to collect mode: enable gravity compensation
                robot.set_mode("collect")
                state_ref["mode"] = ControlMode.COLLECT
                if stream_buffer:
                    stream_buffer.clear()
                print("[Mode] COLLECT (gravity compensation ON)")
            else:
                # Switch to policy mode: disable gravity compensation
                robot.set_mode("control")
                state_ref["mode"] = ControlMode.POLICY
                if stream_buffer:
                    stream_buffer.clear()
                print("[Mode] POLICY (position control)")
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
            if recording_ref.get("recording", False) and _dataset_buffer_size(dataset) > 0:
                recording_ref["recording"] = False
                fc = _dataset_buffer_size(dataset)
                print(f"\n[Save] Episode {recording_ref['episode']}: {fc} frames")
                if webui is not None:
                    webui.on_recording_stopped()
                if label >= 0:
                    _save_dataset_episode(dataset, label, cfg.task)
                    if webui is not None:
                        webui.on_episode_saved(recording_ref["episode"], fc, label)
                    recording_ref["episode"] += 1
                    print(f"[Save] Submitted (success={label}). Ready for next episode.")
                else:
                    print("[Save] Discarded.")
                    _discard_dataset_episode(dataset)
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
            reset_to_zero(robot, None, action_features, max_step=cfg.align_max_step)
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
        nonlocal state_ref, recording_ref, stop_ref, robot, action_features
        try:
            if k == '\x1b' or k == '\x03':
                print("\n[Exit]")
                stop_ref["stop"] = True
            elif k == 'p' or k == '\t':
                if state_ref["control_mode"] != ControlMode.MIXED:
                    print(f"[Mode] Fixed to {state_ref['control_mode'].value.upper()}, switching disabled.")
                elif state_ref["mode"] == ControlMode.POLICY:
                    robot.set_mode("collect")
                    state_ref["mode"] = ControlMode.COLLECT
                    if stream_buffer:
                        stream_buffer.clear()
                    print(f"[Mode] COLLECT (gravity compensation ON)")
                else:
                    robot.set_mode("control")
                    state_ref["mode"] = ControlMode.POLICY
                    if stream_buffer:
                        stream_buffer.clear()
                    print(f"[Mode] POLICY (position control)")
            elif k == 'r':
                recording_ref["recording"] = not recording_ref["recording"]
                if recording_ref["recording"]:
                    if stream_buffer:
                        stream_buffer.clear()
                    print(f"[Recording] ON  (episode {recording_ref['episode']})")
                else:
                    print(f"[Recording] OFF")
            elif k == 's':
                if recording_ref.get("recording", False) and _dataset_buffer_size(dataset) > 0:
                    recording_ref["recording"] = False
                    fc = _dataset_buffer_size(dataset)
                    print(f"\n[Save] Episode {recording_ref['episode']}: {fc} frames")
                    label = prompt_success_failure()
                    if label >= 0:
                        _save_dataset_episode(dataset, label, cfg.task)
                        recording_ref["episode"] += 1
                        print(f"[Save] Submitted (success={label}). Ready for next episode.")
                    else:
                        print("[Save] Discarded.")
                        _discard_dataset_episode(dataset)
                else:
                    print("[Save] Nothing to save.")
            elif k == 'z':
                if recording_ref.get("recording", False):
                    print("[Reset] Cannot reset while recording. Stop recording first.")
                else:
                    print("[Reset] Moving arms to zero position...")
                    if stream_buffer:
                        stream_buffer.clear()
                    reset_to_zero(robot, None, action_features, max_step=cfg.align_max_step)
        except Exception as e:
            print(f"Key error: {e}")

    # Main loop
    switch_hint = "P=switch  " if state_ref["control_mode"] == ControlMode.MIXED else ""
    mode_label = "collect" if state_ref["mode"] == ControlMode.COLLECT else "policy"
    print("\n" + "=" * 60)
    print(f"  Robot: {robot.name}  |  Mode: {cfg.robot.mode}")
    print(f"  Control: {state_ref['control_mode'].value.upper()}  |  Start: {mode_label.upper()}")
    print(f"  Policy: {'Connected' if (policy and policy.connected) else 'N/A'}  |  Output: {cfg.output_dir}")
    print(f"  Task: {cfg.task}")
    print(f"  Storage: NPY (O(1) RAM)")
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
                stream_buffer=stream_buffer,
                dataset=dataset,
                action_features=action_features,
                state_ref=state_ref,
                recording_ref=recording_ref,
                stop_ref=stop_ref,
                obs_lock=obs_lock,
                task=cfg.task,
                on_key=handle_keypress,
            )
            if recording_ref.get("recording", False) and _dataset_buffer_size(dataset) > 0:
                recording_ref["recording"] = False
                fc = _dataset_buffer_size(dataset)
                print(f"\n[Auto-save] Episode {recording_ref['episode']}: {fc} frames (time limit)")
                label = prompt_success_failure()
                if label >= 0:
                    _save_dataset_episode(dataset, label, cfg.task)
                    recording_ref["episode"] += 1
                else:
                    _discard_dataset_episode(dataset)
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
        robot.disconnect()
        bg_encoder.shutdown()
        print(f"  Output: {dataset_root}")
        print("  Done.\n")


def main() -> None:
    from robodeploy.configs.parser import wrap
    from scripts.record_config_body_teaching import RecordBodyTeachingConfig

    @wrap()
    def _main(cfg: RecordBodyTeachingConfig) -> None:
        run_record(cfg)

    _main()


if __name__ == "__main__":
    main()

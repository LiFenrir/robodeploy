#!/usr/bin/env python
"""
机器人数据采集 + 推理脚本，NPY 存储后端，O(1) 内存。

控制模式: teleop / policy / mixed。 支持 RTC 和 temporal smoothing，
leader-follower 对齐，WebUI 监控，键盘标注成功/失败。

Usage:
    python record_dataset.py --robot.type=s1_follower --robot.port=/dev/ttyUSB0 \\
        --teleop.type=s1_leader --teleop.port=/dev/ttyUSB1 --task="pick and place"
"""

import logging
import sys
import threading
import time
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Register camera / robot / teleop / policy types with draccus ChoiceRegistry
from robodeploy.cameras import CameraConfig  # noqa: F401, E402
from robodeploy.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401, E402
from robodeploy.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401, E402
from robodeploy.policy_clients import (  # noqa: F401, E402
    make_policy_client_from_config,
    openpi,
)
from robodeploy.robots import (  # noqa: F401, E402
    bi_s1_follower,
    bi_so100_follower,
    make_robot_from_config,
    s1_follower,
    so100_follower,
)
from robodeploy.robots.lerobot_robot_my_arm import (  # noqa: F401, E402
    bi_innov_arm_v1,
    innov_arm_v1,
)
from robodeploy.teleoperators import (  # noqa: F401, E402
    bi_s1_leader,
    bi_so100_leader,
    make_teleoperator_from_config,
    s1_leader,
    so100_leader,
)
from robodeploy.utils.stream_buffer import StreamActionBuffer  # noqa: F401, E402
from robodeploy.webui.server import WebUIServer  # noqa: E402

try:
    import termios
    import tty
except ImportError:
    termios = None  # type: ignore[assignment]
    tty = None  # type: ignore[assignment]

try:
    from robodeploy.rtc import ActionQueue, RTCConfig
except ImportError:
    ActionQueue = None  # type: ignore[assignment]
    RTCConfig = None  # type: ignore[assignment]

from robodeploy.configs.parser import wrap  # noqa: E402
from robodeploy.datasets.npy_backend import BackgroundVideoEncoder, LeRobotDatasetNPY  # noqa: E402
from robodeploy.datasets.utils import (  # noqa: E402
    DEFAULT_FEATURES,
    RECORDING_EXTRA_FEATURES,
    build_dataset_frame,
    hw_to_dataset_features,
)
from robodeploy.scripts.record_config import RecordConfig  # noqa: E402
from robodeploy.utils.keyboard_control import get_keypress, prompt_success_failure  # noqa: F401, E402
from robodeploy.utils.leader_follower_align import (  # noqa: F401, E402
    interpolate_leader_to_follower,
    reset_to_zero,
    smooth_inference_action,
)


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
# Inference thread
# ==============================================================================


def _stack_front_cameras(images: dict) -> dict:
    """Vertically stack front + front_1 (front_1 rotated 180° first), in-place."""
    if "front" in images and "front_1" in images:
        front = np.asarray(images["front"])
        front_1 = np.asarray(images["front_1"])
        images["front"] = np.concatenate([front, np.rot90(front_1, 2)], axis=0)
        del images["front_1"]
    return images


def _binarize_gripper_state(state: np.ndarray) -> None:
    """Binarize gripper joints (indices 6, 13): < 0.2 → 0, >= 0.2 → 1, in-place."""
    for gi in (6, 13):
        if gi < len(state):
            state[gi] = 0.0 if state[gi] < 0.2 else 1.0


def _prepare_inference_input(
    obs: dict, action_features: dict[str, type], camera_names: list[str]
) -> tuple[np.ndarray, dict]:
    """Extract state vector and image dict from observation dict for policy inference."""
    state = np.array([obs.get(k, 0.0) for k in action_features], dtype=np.float64)
    _binarize_gripper_state(state)
    images = {cam: np.asarray(obs[cam]) for cam in camera_names if cam in obs}
    _stack_front_cameras(images)
    return state, images


def _start_inference_thread(
    policy,
    buffer: StreamActionBuffer | None,
    state_ref: dict,
    recording_ref: dict,
    action_features: dict[str, type],
    camera_names: list[str],
    task: str,
    inference_rate: float,
    latency_k: int,
    min_smooth_steps: int,
    action_queue: "ActionQueue | None" = None,
    rtc_execution_horizon: int = 10,
) -> threading.Thread:
    """异步推理线程。

    RTC 模式：收发驱动 —— 收到推理结果后立即发送新观测，通道中仅一个请求。
    Smoothing 模式：轮询 + 限速（legacy）。
    """

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
                    if buffer is not None:
                        buffer.clear()
                    if action_queue is not None:
                        action_queue.clear()
                was_recording = False
                time.sleep(0.1)
                continue

            # Smoothing mode: rate-limited polling
            if action_queue is None:
                was_recording = True
                obs = state_ref.get("obs")
                if obs is None:
                    time.sleep(0.1)
                    continue

                try:
                    state, images = _prepare_inference_input(obs, action_features, camera_names)
                    if buffer is not None:
                        result = policy.infer(images, state, task)
                        actions = result.get("actions", None)
                        if actions is not None and len(actions) > 0:
                            buffer.integrate_new_chunk(
                                np.asarray(actions),
                                max_k=latency_k,
                                min_m=min_smooth_steps,
                            )
                    state_ref["inference_ok"] = True
                except Exception as e:
                    logger.warning(f"Inference error: {e}")
                    state_ref["inference_ok"] = False
                time.sleep(rate)
                continue

            # RTC mode: request-response cycle, one packet in flight
            was_recording = True
            obs = state_ref.get("obs")
            if obs is None:
                time.sleep(0.1)
                continue

            try:
                state, images = _prepare_inference_input(obs, action_features, camera_names)

                # 快照发送时刻 + 全部剩余 action（服务端取前 horizon 步约束）
                send_index = action_queue.last_index
                prev_leftover = action_queue.get_left_over()

                rtc_kwargs = {}
                if prev_leftover is not None:
                    rtc_kwargs["prev_chunk_left_over"] = prev_leftover.cpu().numpy()
                    rtc_kwargs["inference_delay"] = 0
                    rtc_kwargs["execution_horizon"] = rtc_execution_horizon

                result = policy.infer(images, state, task, **rtc_kwargs)
                actions = result.get("actions", None)
                if actions is not None and len(actions) > 0:
                    # 等待期间主循环消耗的步数
                    wait_steps = action_queue.last_index - send_index
                    actions = actions[wait_steps:]
                    if len(actions) > 0:
                        action_queue.merge(
                            actions=torch.from_numpy(np.asarray(actions)),
                            execution_horizon=rtc_execution_horizon,
                        )

                state_ref["inference_ok"] = True
            except Exception as e:
                logger.warning(f"Inference error: {e}")
                state_ref["inference_ok"] = False

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
    dataset: LeRobotDatasetNPY,
    action_features: dict[str, type],
    state_ref: dict,
    recording_ref: dict,
    stop_ref: dict,
    obs_lock: threading.Lock,
    task: str,
    on_key: callable = None,
    process_pending: callable = None,
    action_queue: "ActionQueue | None" = None,
    action_smooth_max_step: float = 0.0,
) -> None:
    """Main control loop using lerobot abstract interfaces with NPY dataset storage."""
    start_episode_t = time.perf_counter()
    timestamp = 0.0
    was_recording = recording_ref.get("recording", False)
    prev_infer_action: dict | None = None
    was_policy = False

    while timestamp < control_time_s and not stop_ref["stop"]:
        start_loop_t = time.perf_counter()

        is_recording = recording_ref.get("recording", False)
        if is_recording and not was_recording:
            start_episode_t = time.perf_counter()
            timestamp = 0.0
            prev_infer_action = None
        elif not is_recording:
            start_episode_t = time.perf_counter()
            timestamp = 0.0
        was_recording = is_recording

        observation = robot.get_observation()
        with obs_lock:
            state_ref["obs"] = observation
        if recording_ref.get("recording", False):
            recording_ref["frames"] = _dataset_buffer_size(dataset)

        if on_key is not None:
            k = get_keypress()
            if k is not None:
                on_key(k)
            if stop_ref["stop"]:
                break

        if process_pending is not None:
            process_pending()

        teleop_action = None
        if teleop is not None:
            teleop_action = teleop.get_action()

        is_inference = 0
        action = None
        is_policy = state_ref["mode"] == ControlMode.POLICY
        if is_policy and not was_policy:
            prev_infer_action = None
        was_policy = is_policy

        if is_policy:
            if action_queue is not None:
                act_tensor = action_queue.get()
                act_np = act_tensor.cpu().numpy() if act_tensor is not None else None
            elif stream_buffer is not None:
                act_np = stream_buffer.pop_next_action()
            else:
                act_np = None
            if act_np is not None:
                action = numpy_to_action_dict(act_np, action_features)
                is_inference = 1
        elif teleop_action is not None and recording_ref.get("recording", False):
            action = teleop_action

        if action is not None:
            if is_inference and prev_infer_action is not None and action_smooth_max_step > 0:
                smooth_inference_action(
                    robot,
                    prev_infer_action,
                    action,
                    action_features,
                    max_step=action_smooth_max_step,
                )
            sent_action = robot.send_action(action)
            if is_inference:
                prev_infer_action = action
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

        fc = _dataset_buffer_size(dataset)
        if recording_ref.get("recording", False) and fc > 0 and fc % 60 == 0:
            mode_str = "POL" if state_ref["mode"] == ControlMode.POLICY else "TEL"
            switch_hint = "P=switch " if state_ref["control_mode"] == ControlMode.MIXED else ""
            inf_ok = " ERR!" if not state_ref.get("inference_ok", True) else ""
            ep = recording_ref.get("episode", 0)
            print(
                f"[{mode_str}{inf_ok}] ep={ep} frames={fc} elapsed={fc / fps:.1f}s | {switch_hint}R=rec S=save Esc=quit"
            )


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
    dataset.episode_buffer["is_failure_data"] = [np.int64(1 - label)] * n
    dataset.save_episode_async()


def _discard_dataset_episode(dataset: LeRobotDatasetNPY) -> None:
    """Discard current episode buffer and clean up NPY files."""
    if dataset.episode_buffer is not None and _dataset_buffer_size(dataset) > 0:
        dataset.clear_episode_buffer()


def run_record(cfg) -> None:
    """Main entry point, receives a RecordConfig from draccus."""

    # Create robot
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    logger.info(f"Robot '{robot.name}' connected.")

    if cfg.output_dir == "auto":
        now = datetime.now()
        cfg.output_dir = f"outputs/{robot.name}/{now.strftime('%m%d_%H%M')}"

    # Determine control mode
    control_mode = ControlMode(cfg.control_mode)

    # Create teleop — skip for pure POLICY mode
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

    info_features = {**action_fts, **obs_fts, **DEFAULT_FEATURES, **RECORDING_EXTRA_FEATURES}

    # Create NPY-backed dataset (no image_writer — np.save is fast enough synchronously)
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
    bg_encoder = BackgroundVideoEncoder(info_path, video_keys, log_dir=str(dataset_root))
    dataset.set_video_encoder(bg_encoder)

    # Policy client — skip for pure TELEOP mode
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
    pending_ref = {"cmd": None, "data": None}
    obs_lock = threading.Lock()

    # Stream buffer / ActionQueue — mutually exclusive
    stream_buffer = None
    action_queue = None
    if control_mode != ControlMode.TELEOP:
        if cfg.use_rtc:
            if ActionQueue is None:
                logger.error(
                    "RTC requested but robodeploy.rtc failed to import. Falling back to temporal smoothing."
                )
                stream_buffer = StreamActionBuffer(state_dim=len(action_features))
            else:
                rtc_cfg = RTCConfig(
                    enabled=True,
                    execution_horizon=cfg.rtc_execution_horizon,
                )
                action_queue = ActionQueue(
                    cfg=rtc_cfg,
                    queue_size=cfg.rtc_queue_size,
                )
                logger.info("RTC mode enabled (ActionQueue), temporal smoothing disabled.")
        elif cfg.use_temporal_smoothing:
            stream_buffer = StreamActionBuffer(state_dim=len(action_features))

    # ---- 推理预热 ----
    if cfg.warmup_rounds > 0 and policy is not None and policy.connected:
        logger.info("=" * 60)
        logger.info("推理预热 (%d rounds)...", cfg.warmup_rounds)
        obs = robot.get_observation()
        warmup_state, warmup_images = _prepare_inference_input(obs, action_features, camera_names)
        warmup_times = []
        for w in range(cfg.warmup_rounds):
            t0 = time.monotonic()
            try:
                policy.infer(warmup_images, warmup_state, cfg.task)
                elapsed = (time.monotonic() - t0) * 1000
                warmup_times.append(elapsed)
                logger.info("  预热 %2d/%d: %.0fms", w + 1, cfg.warmup_rounds, elapsed)
            except Exception as e:
                logger.warning("  预热 %2d/%d 失败: %s", w + 1, cfg.warmup_rounds, e)
        if warmup_times:
            logger.info(
                "预热完成: avg=%.0fms, min=%.0fms, max=%.0fms",
                np.mean(warmup_times),
                np.min(warmup_times),
                np.max(warmup_times),
            )

    # Start inference thread
    inference_thread = None
    need_inference = policy is not None and policy.connected
    if need_inference and (stream_buffer is not None or action_queue is not None):
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
            action_queue=action_queue,
            rtc_execution_horizon=cfg.rtc_execution_horizon,
        )

    # Guard: fail fast if required component is missing
    if control_mode == ControlMode.POLICY and (policy is None or not policy.connected):
        logger.error("POLICY mode requires a running policy server. Exiting.")
        sys.exit(1)
    if control_mode == ControlMode.TELEOP and teleop is None:
        logger.error("TELEOP mode requires a teleoperator. Exiting.")
        sys.exit(1)

    # Shared command handlers — called both from keyboard (main thread) and
    # WebUI (via pending_ref, executed synchronously in record_loop).
    webui = None  # set below if webui_port > 0

    def _handle_switch_mode():
        if state_ref["control_mode"] != ControlMode.MIXED:
            print(f"[Mode] Fixed to {state_ref['control_mode'].value.upper()}, switching disabled.")
            return
        if state_ref["mode"] == ControlMode.POLICY:
            if teleop is not None:
                leader_pos = teleop.get_action()
                follower_pos = state_ref.get("obs") or robot.get_observation()
                interpolate_leader_to_follower(
                    teleop,
                    leader_pos,
                    follower_pos,
                    action_features,
                    max_step=cfg.align_max_step,
                )
            state_ref["mode"] = ControlMode.TELEOP
            if stream_buffer:
                stream_buffer.clear()
            if action_queue:
                action_queue.clear()
            print("[Mode] TELEOP")
        else:
            state_ref["mode"] = ControlMode.POLICY
            if stream_buffer:
                stream_buffer.clear()
            if action_queue:
                action_queue.clear()
            print("[Mode] POLICY")

    def _handle_toggle_record():
        recording_ref["recording"] = not recording_ref["recording"]
        if recording_ref["recording"]:
            if stream_buffer:
                stream_buffer.clear()
            if action_queue:
                action_queue.clear()
            print(f"[Recording] ON  (episode {recording_ref['episode']})")
            if webui is not None:
                webui.on_recording_started()
        else:
            print("[Recording] OFF")
            # 停止录制时清空动作队列，避免机器人继续执行缓存动作
            if action_queue:
                action_queue.clear()
            if stream_buffer:
                stream_buffer.clear()
            if webui is not None:
                webui.on_recording_stopped()

    def _handle_save(label: int = -1):
        if _dataset_buffer_size(dataset) > 0:
            was_recording = recording_ref.get("recording", False)
            if was_recording:
                recording_ref["recording"] = False
                # 停止录制时清空动作队列，避免机器人继续执行缓存动作
                if action_queue:
                    action_queue.clear()
                if stream_buffer:
                    stream_buffer.clear()
            fc = _dataset_buffer_size(dataset)
            print(f"[Save] Episode {recording_ref['episode']}: {fc} frames")
            if was_recording and webui is not None:
                webui.on_recording_stopped()
            if label >= 0:
                _save_dataset_episode(dataset, label, cfg.task)
                if webui is not None:
                    webui.on_episode_saved(recording_ref["episode"], fc, label)
                recording_ref["episode"] += 1
                print(f"[Save] Submitted (label={label}). Ready for next episode.")
            else:
                print("[Save] Discarded.")
                _discard_dataset_episode(dataset)
        else:
            print("[Save] Nothing to save.")

    def _handle_reset_zero():
        if recording_ref.get("recording", False):
            print("[Reset] Cannot reset while recording. Stop recording first.")
            return
        print("[Reset] Moving arms to zero position...")
        if stream_buffer:
            stream_buffer.clear()
        if action_queue:
            action_queue.clear()
        reset_to_zero(robot, teleop, action_features, max_step=cfg.align_max_step)

    # WebUI server — handlers only set pending_ref; main loop executes them.
    if cfg.webui_port > 0:

        def _cmd_switch_mode(_data=None):
            pending_ref["cmd"] = "switch_mode"
            return None

        def _cmd_toggle_record(_data=None):
            pending_ref["cmd"] = "toggle_record"
            return None

        def _cmd_save(data):
            pending_ref["cmd"] = "save"
            pending_ref["data"] = data
            return None

        def _cmd_reset_zero(_data=None):
            pending_ref["cmd"] = "reset_zero"
            return None

        def _cmd_stop(_data=None):
            print("[Exit]")
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
    _stdin_fd = sys.stdin.fileno()
    _old_termios = None

    def handle_keypress(k: str):
        try:
            if k == "\x1b" or k == "\x03":
                print("[Exit]")
                stop_ref["stop"] = True
            elif k == "p" or k == "\t":
                _handle_switch_mode()
            elif k == "r":
                _handle_toggle_record()
            elif k == "s":
                label = prompt_success_failure()
                _handle_save(label)
            elif k == "z":
                _handle_reset_zero()
            # Clear any pending WebUI command since we just handled one via keyboard
            pending_ref["cmd"] = None
            pending_ref["data"] = None
        except Exception as e:
            print(f"Key error: {e}")

    # Main loop — process pending WebUI commands synchronously
    def _process_pending():
        if pending_ref["cmd"] is None:
            return
        cmd = pending_ref["cmd"]
        data = pending_ref["data"]
        pending_ref["cmd"] = None
        pending_ref["data"] = None

        if cmd == "save":
            label = data.get("label", -1) if data else -1
            _handle_save(label)
        elif cmd == "reset_zero":
            _handle_reset_zero()
        elif cmd == "switch_mode":
            _handle_switch_mode()
        elif cmd == "toggle_record":
            _handle_toggle_record()

    switch_hint = "P=switch  " if state_ref["control_mode"] == ControlMode.MIXED else ""
    print("=" * 60)
    print(f"  Robot: {robot.name}  |  Teleop: {teleop.name if teleop else 'None'}")
    print(
        f"  Control: {state_ref['control_mode'].value.upper()}  |  Start: {state_ref['mode'].value.upper()}"
    )
    print(f"  Policy: {'Connected' if (policy and policy.connected) else 'N/A'}  |  Output: {cfg.output_dir}")
    print(f"  Task: {cfg.task}")
    print("  Storage: NPY (O(1) RAM)")
    print(f"  Controls: {switch_hint}R=rec  S=save+label  Z=zero-reset  Esc=exit")
    print("=" * 60)
    input("Press [Enter] to start...")

    _old_termios = termios.tcgetattr(_stdin_fd)
    tty.setcbreak(_stdin_fd)

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
                dataset=dataset,
                action_features=action_features,
                state_ref=state_ref,
                recording_ref=recording_ref,
                stop_ref=stop_ref,
                obs_lock=obs_lock,
                task=cfg.task,
                on_key=handle_keypress,
                process_pending=_process_pending,
                action_queue=action_queue,
                action_smooth_max_step=cfg.action_smooth_max_step,
            )
            if recording_ref.get("recording", False) and _dataset_buffer_size(dataset) > 0:
                recording_ref["recording"] = False
                fc = _dataset_buffer_size(dataset)
                print(f"[Auto-save] Episode {recording_ref['episode']}: {fc} frames (time limit)")
                if webui is not None:
                    webui.on_recording_stopped()
                label = prompt_success_failure()
                if label >= 0:
                    _save_dataset_episode(dataset, label, cfg.task)
                    if webui is not None:
                        webui.on_episode_saved(recording_ref["episode"], fc, label)
                    recording_ref["episode"] += 1
                else:
                    _discard_dataset_episode(dataset)
                pending_ref["cmd"] = None
                pending_ref["data"] = None
    except KeyboardInterrupt:
        print("[Interrupted]")
    finally:
        print("=" * 50)
        print("  Shutting down...")
        print("=" * 50)
        state_ref["stop"] = True
        if webui is not None:
            webui.stop()
        if inference_thread is not None:
            inference_thread.join(timeout=5)
        if _old_termios is not None:
            termios.tcsetattr(_stdin_fd, termios.TCSADRAIN, _old_termios)
        robot.disconnect()
        if teleop is not None:
            teleop.disconnect()
        bg_encoder.shutdown()
        print(f"  Output: {dataset_root}")
        print("  Done.")


def main() -> None:
    @wrap()
    def _main(cfg: RecordConfig) -> None:
        run_record(cfg)

    _main()


if __name__ == "__main__":
    main()

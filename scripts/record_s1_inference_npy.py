#!/usr/bin/env python
"""
Unified data collection + inference script with NPY-based storage backend.

Same features as record_s1_inference.py (three control modes, policy inference,
temporal smoothing, leader-follower alignment, WebUI, labeling) but uses
LeRobotDataset with NPY intermediate storage instead of in-memory accumulation:

  - Each camera frame is written to disk immediately as a raw .npy file
    (no PNG compression overhead, O(1) RAM regardless of episode length).
  - At episode end, .npy files are stream-encoded directly to MP4 via
    av.VideoFrame.from_ndarray() (no PNG decode step).
  - Parquet + metadata managed by LeRobotDataset.

Supports any robot platform registered in lerobot (bi_s1_follower, so100_follower,
etc.) by simply changing --robot.type and --teleop.type.

Uses draccus for configuration parsing with ChoiceRegistry support.

Features:
  1. Teleoperation recording with memory-efficient NPY -> MP4 encoding
  2. OpenPI policy inference with temporal smoothing (StreamActionBuffer)
  3. Success/failure labeling via keyboard (stored as is_failure_data field, 1=failure)
  4. Teleop/inference mode switching + recording toggle + is_infer_data field (1=inference)
  5. Leader-follower alignment on inference->teleop switch (cosine interpolation)

Usage:
    # S1 bimanual (cameras configured via robot.cameras)
    python record_s1_inference_npy.py \
        --robot.type=bi_s1_follower \
        --robot.left_arm_port=/dev/ttyUSB0 --robot.right_arm_port=/dev/ttyUSB1 \
        --robot.cameras='{"front":{"type":"intelrealsense","width":848,"height":480,"fps":30}}' \
        --teleop.type=bi_s1_leader \
        --teleop.left_arm_port=/dev/ttyUSB2 --teleop.right_arm_port=/dev/ttyUSB3 \
        --policy.type=openpi \
        --policy.host=localhost --policy.port=8000 \
        --task="fold the box"

    # Single arm S1
    python record_s1_inference_npy.py \
        --robot.type=s1_follower \
        --robot.port=/dev/ttyUSB0 \
        --robot.cameras='{"front":{"type":"intelrealsense","width":848,"height":480,"fps":30}}' \
        --teleop.type=s1_leader \
        --teleop.port=/dev/ttyUSB1 \
        --task="pick and place"
"""

import json
import logging
import multiprocessing as mp
import shutil
import sys
import threading
import time
from enum import Enum
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Register camera / robot / teleop / policy types with draccus ChoiceRegistry
from robodeploy.cameras import CameraConfig  # noqa: F401, E402
from robodeploy.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401, E402
from robodeploy.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401, E402
from robodeploy.robots import (  # noqa: F401, E402
    bi_s1_follower,
    bi_so100_follower,
    s1_follower,
    so100_follower,
)
from robodeploy.teleoperators import (  # noqa: F401, E402
    bi_s1_leader,
    bi_so100_leader,
    s1_leader,
    so100_leader,
)
from robodeploy.policy_clients import (  # noqa: F401, E402
    openpi,
)
from robodeploy.webui.server import WebUIServer
from robodeploy.utils.stream_buffer import StreamActionBuffer  # noqa: F401, E402
from robodeploy.utils.leader_follower_align import (  # noqa: F401, E402
    interpolate_leader_to_follower,
    reset_to_zero,
)
from robodeploy.utils.keyboard_control import get_keypress, prompt_success_failure  # noqa: F401, E402
from robodeploy.datasets.lerobot_dataset import LeRobotDataset
from robodeploy.datasets.utils import build_dataset_frame, hw_to_dataset_features, write_info


class ControlMode(Enum):
    TELEOP = "teleop"
    POLICY = "policy"
    MIXED = "mixed"


# ==============================================================================
# NPY-based video encoder (streaming, O(1) RAM)
# ==============================================================================

def encode_video_from_npy(npy_dir: Path, dst: Path, fps: int, vcodec: str = "libsvtav1", crf: int = 30) -> None:
    """Encode a directory of frame_*.npy files to MP4. Reads one frame at a time — O(1) RAM."""
    import av

    npy_files = sorted(npy_dir.glob("frame_*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No frame_*.npy files found in {npy_dir}")

    first = np.load(str(npy_files[0]))
    if first.dtype != np.uint8:
        first = first.astype(np.uint8)
    h, w = first.shape[:2]

    dst.parent.mkdir(parents=True, exist_ok=True)
    options = {"crf": str(crf), "preset": "8"}

    with av.open(str(dst), "w") as output:
        stream = output.add_stream(vcodec, fps, options=options)
        stream.pix_fmt = "yuv420p"
        stream.width = w
        stream.height = h

        for i, fpath in enumerate(npy_files):
            img = np.load(str(fpath))
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for pkt in stream.encode(frame):
                output.mux(pkt)
            if i > 0 and i % 300 == 0:
                print(f"    Encoding {npy_dir.name}: {i}/{len(npy_files)}")

        for pkt in stream.encode():
            output.mux(pkt)


# ==============================================================================
# Background video encoder (mp.Process, async — doesn't block recording)
# ==============================================================================

def _video_encoder_loop(queue: mp.Queue, info_path: str, video_keys: list[str]) -> None:
    """Background process: encodes NPY dirs to MP4, cleans up, updates metadata."""
    while True:
        item = queue.get()
        if item is None:
            break
        npy_dir, video_path, fps = item
        try:
            encode_video_from_npy(npy_dir, video_path, fps)
            shutil.rmtree(npy_dir, ignore_errors=True)
            # Update total_videos in info.json
            if Path(info_path).exists():
                with open(info_path) as f:
                    info = json.load(f)
                info["total_videos"] += 1
                if info["total_videos"] == 1:
                    # First video encoded — update video info from the file
                    from robodeploy.datasets.video_utils import get_video_info
                    for key in video_keys:
                        if not info["features"].get(key, {}).get("info"):
                            vpath = str(Path(info_path).parent.parent / info["video_path"].format(
                                episode_chunk=0, video_key=key, episode_index=0))
                            if Path(vpath).is_file():
                                info["features"][key]["info"] = get_video_info(vpath)
                with open(info_path, "w") as f:
                    json.dump(info, f, indent=2)
            print(f"[Encoder] Video done: {video_path.name}")
        except Exception as e:
            print(f"[Encoder] Error encoding {video_path}: {e}")
            import traceback
            traceback.print_exc()


class BackgroundVideoEncoder:
    """Manages a background process for async NPY -> MP4 video encoding."""

    def __init__(self, info_path: str, video_keys: list[str]):
        self.queue: mp.Queue = mp.Queue(maxsize=8)
        self.info_path = info_path
        self.video_keys = video_keys
        self.process = mp.Process(
            target=_video_encoder_loop,
            args=(self.queue, info_path, video_keys),
            daemon=True,
        )
        self.process.start()
        logger.info("[Encoder] Background video encoder started.")

    def submit(self, npy_dir: Path, video_path: Path, fps: int) -> None:
        """Non-blocking: submit an encoding job."""
        self.queue.put((npy_dir, video_path, fps))

    def pending(self) -> int:
        """Number of jobs awaiting encoding."""
        return self.queue.qsize()

    def shutdown(self) -> None:
        """Wait for all pending encoding jobs to finish."""
        pending = self.queue.qsize()
        if pending > 0:
            print(f"\n[Encoder] {pending} video(s) pending, waiting for encoding...")
        self.queue.put(None)
        print("[Encoder] Waiting for background encoding to complete...")
        self.process.join(timeout=300)
        if self.process.is_alive():
            logger.warning("[Encoder] Process did not exit, terminating.")
            self.process.terminate()
        else:
            print("[Encoder] Encoding finished, all videos saved.")


# ==============================================================================
# LeRobotDataset subclass with NPY intermediate storage (no PNG overhead)
# ==============================================================================

class LeRobotDatasetNPY(LeRobotDataset):
    """LeRobotDataset variant that writes raw .npy files instead of PNG.

    Overrides _save_image, _get_image_file_path, and encode_episode_videos.
    Supports async video encoding via a BackgroundVideoEncoder.

    Use save_episode_async() to write parquet + metadata and hand off video
    encoding to the background process (non-blocking). The sync save_episode()
    encodes videos inline if no background encoder is attached.
    """

    def set_video_encoder(self, encoder: BackgroundVideoEncoder) -> None:
        """Attach a background video encoder for async encoding."""
        self._bg_encoder = encoder  # type: ignore[attr-defined]

    def _get_image_file_path(self, episode_index: int, image_key: str, frame_index: int) -> Path:
        fpath = "images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.npy"
        return self.root / fpath.format(
            image_key=image_key,
            episode_index=episode_index,
            frame_index=frame_index,
        )

    def _save_image(self, image, fpath: Path) -> None:
        """Write raw numpy array as .npy — no compression, very fast."""
        if hasattr(image, "cpu"):  # torch.Tensor
            image = image.cpu().numpy()
        fpath.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(fpath), np.asarray(image, dtype=np.uint8))

    def encode_episode_videos(self, episode_index: int) -> None:
        """Encode NPY frame directories to MP4.

        If a background encoder is attached, submits jobs asynchronously.
        Otherwise encodes inline (blocking).
        """
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            if video_path.is_file():
                continue
            img_dir = self._get_image_file_path(episode_index, key, 0).parent
            if getattr(self, '_bg_encoder', None) is not None:
                getattr(self, '_bg_encoder').submit(img_dir, video_path, self.fps)
            else:
                encode_video_from_npy(img_dir, video_path, self.fps)
                shutil.rmtree(img_dir)

        if len(self.meta.video_keys) > 0 and episode_index == 0 and getattr(self, '_bg_encoder', None) is None:
            self.meta.update_video_info()
            write_info(self.meta.info, self.meta.root)

    def save_episode_async(self) -> None:
        """Write parquet + update metadata + reset buffer. Video encoding is handed
        off to the background encoder (must call set_video_encoder() first).

        This returns immediately — the main thread can start the next episode while
        the background process encodes videos.
        """
        if getattr(self, '_bg_encoder', None) is None:
            # Fall back to sync encoding
            self.save_episode()
            return

        # --- Steps 1-5: buffer processing + parquet write (same as parent) ---
        episode_buffer = self.episode_buffer
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(
            self.meta.total_frames, self.meta.total_frames + episode_length
        )
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        for task in episode_tasks:
            if self.meta.get_task_index(task) is None:
                self.meta.add_task(task)
        episode_buffer["task_index"] = np.array(
            [self.meta.get_task_index(t) for t in tasks]
        )

        from robodeploy.datasets.utils import validate_episode_buffer
        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        for key, ft in self.features.items():
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self._save_episode_table(episode_buffer, episode_index)

        # Compute stats (state/action only — skip images since they're NPY)
        from robodeploy.datasets.compute_stats import compute_episode_stats
        try:
            ep_stats = compute_episode_stats(episode_buffer, self.features)
        except Exception:
            # compute_episode_stats may fail trying to read NPY paths as images;
            # fall back to state/action-only stats
            ep_stats = {}
            for key in episode_buffer:
                arr = episode_buffer[key]
                if isinstance(arr, np.ndarray) and arr.dtype.kind in ('f', 'i'):
                    ep_stats[key] = {
                        "mean": arr.mean(axis=0).tolist() if arr.ndim > 1 else float(arr.mean()),
                        "std": arr.std(axis=0).tolist() if arr.ndim > 1 else float(arr.std()),
                        "min": arr.min(axis=0).tolist() if arr.ndim > 1 else float(arr.min()),
                        "max": arr.max(axis=0).tolist() if arr.ndim > 1 else float(arr.max()),
                        "count": [episode_length],
                    }

        # --- Step 6: Submit video encoding to background ---
        # Don't increment total_videos yet — the background encoder will.
        saved_total_videos = self.meta.info.get("total_videos", 0)
        self.encode_episode_videos(episode_index)

        # --- Step 7: Update metadata ---
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)
        # Restore total_videos (background encoder increments it after each video)
        self.meta.info["total_videos"] = saved_total_videos
        write_info(self.meta.info, self.meta.root)

        # --- Step 8: Check timestamps, reset buffer ---
        from robodeploy.datasets.utils import (
            check_timestamps_sync,
            get_episode_data_index,
        )
        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        self.episode_buffer = self.create_episode_buffer()

        pending = getattr(self, '_bg_encoder', None).pending() if getattr(self, '_bg_encoder', None) else 0
        print(f"[Save] Episode {episode_index}: {episode_length} frames. "
              f"Video encoding in background ({pending} jobs in queue).")


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
    dataset: LeRobotDatasetNPY,
    action_features: dict[str, type],
    state_ref: dict,
    recording_ref: dict,
    stop_ref: dict,
    obs_lock: threading.Lock,
    task: str,
    on_key: callable = None,
) -> None:
    """Main control loop using lerobot abstract interfaces with NPY dataset storage."""
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
            recording_ref["frames"] = (dataset.episode_buffer or {}).get("size", 0)

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
                obs_frame = build_dataset_frame(dataset.features, observation, "observation")
                action_frame = build_dataset_frame(dataset.features, sent_action, "action")
                frame = {**obs_frame, **action_frame}
                frame["is_infer_data"] = np.int64(is_inference)
                frame["is_failure_data"] = np.int64(0)
                dataset.add_frame(frame, task=task)

        dt_s = time.perf_counter() - start_loop_t
        sleep_time = 1.0 / fps - dt_s
        if sleep_time > 0:
            time.sleep(sleep_time)
        timestamp = time.perf_counter() - start_episode_t

        fc = (dataset.episode_buffer or {}).get("size", 0)
        if recording_ref.get("recording", False) and fc > 0 and fc % 60 == 0:
            mode_str = "POL" if state_ref["mode"] == ControlMode.POLICY else "TEL"
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
    dataset.episode_buffer["is_failure_data"] = [np.int64(1 - label)] * n
    dataset.save_episode_async()


def _discard_dataset_episode(dataset: LeRobotDatasetNPY) -> None:
    """Discard current episode buffer and clean up NPY files."""
    if dataset.episode_buffer is not None and _dataset_buffer_size(dataset) > 0:
        dataset.clear_episode_buffer()


def run_record(cfg) -> None:
    """Main entry point, receives a RecordConfig from draccus."""
    from robodeploy.robots import make_robot_from_config
    from robodeploy.teleoperators import make_teleoperator_from_config
    from robodeploy.policy_clients import make_policy_client_from_config

    # Create robot
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    logger.info(f"Robot '{robot.name}' connected.")
    is_bimanual = cfg.robot.type.startswith("bi_")

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
    bg_encoder = BackgroundVideoEncoder(info_path, video_keys)
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
    obs_lock = threading.Lock()

    # Stream buffer
    stream_buffer = StreamActionBuffer(
        state_dim=len(action_features)
    ) if (cfg.use_temporal_smoothing and control_mode != ControlMode.TELEOP) else None

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
        if teleop is not None:
            teleop.disconnect()
        bg_encoder.shutdown()
        print(f"  Output: {dataset_root}")
        print("  Done.\n")


def main() -> None:
    from robodeploy.configs.parser import wrap
    from scripts.record_config import RecordConfig

    @wrap()
    def _main(cfg: RecordConfig) -> None:
        run_record(cfg)

    _main()


if __name__ == "__main__":
    main()

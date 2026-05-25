#!/usr/bin/env python
"""
Hybrid async dataset writer with SharedMemory for zero-copy inter-process transfer.

Combines:
- SharedMemory for passing image arrays without disk I/O
- Native LeRobot video encoding (encode_video_frames)
- Native LeRobot stats computation (compute_episode_stats, aggregate_stats)
- Dynamic custom features via CustomFeatureConfig
- Full LeRobot v2.1 compatibility

Usage:
    from robodeploy.datasets.async_writer import HybridLeRobotWriter, CustomFeatureConfig

    custom_features = [
        CustomFeatureConfig("is_failure_data", "int64", (1,),
            value_fn=lambda ed: 1 - ed.get("label", 1)),
        CustomFeatureConfig("is_infer_data", "int64", (1,),
            value_fn=lambda ed: ed.get("is_inference", np.zeros(len(ed["action"]), dtype=np.int64))),
    ]

    writer = HybridLeRobotWriter(
        output_dir="./data",
        info_features={**action_fts, **obs_fts},
        fps=30,
        repo_id="my_dataset",
        robot_type="bi_s1_follower",
        custom_features=custom_features,
    )

    # Submit episode data
    writer.submit(episode_data, task="fold the box")

    # Shutdown
    writer.shutdown()
"""

import json
import logging
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import threading
import time
import warnings
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from robodeploy.datasets.compute_stats import compute_episode_stats
from robodeploy.datasets.utils import (
    DEFAULT_FEATURES,
    get_hf_features_from_features,
    serialize_dict,
)
from robodeploy.datasets.video_utils import encode_video_frames, get_video_info

logger = logging.getLogger(__name__)
SENTINEL = None

# Suppress known deprecation warnings from shared_memory on some platforms
warnings.filterwarnings("ignore", category=DeprecationWarning, module="multiprocessing")


# ==============================================================================
# Custom Feature Configuration
# ==============================================================================

class CustomFeatureConfig:
    """Configuration for a user-defined feature to be added to each frame.

    Args:
        name: Feature key in the dataset (e.g. "is_failure_data").
        dtype: Numpy dtype string ("int64", "float32", etc.).
        shape: Feature shape tuple. Use (1,) for scalars per frame.
        value_fn: Callable receiving the episode_data dict and returning either
            a scalar (broadcast to all frames) or an array of length N_frames.
        names: Optional dimension names for vector features.
    """

    def __init__(
        self,
        name: str,
        dtype: str,
        shape: tuple,
        value_fn: Callable[[dict], Any],
        names: list[str] | None = None,
    ):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.value_fn = value_fn
        self.names = names


# ==============================================================================
# SharedMemory helpers
# ==============================================================================

def _create_shared_array(array: np.ndarray, prefix: str = "shm") -> tuple[str, tuple]:
    """Create a shared_memory buffer from a numpy array.

    Returns:
        (shm_name, array_shape) so the consumer can reconstruct the array.
    """
    # Use a unique name to avoid collisions across episodes
    shm_name = f"{prefix}_{os.getpid()}_{threading.current_thread().ident}_{id(array)}_{time.time_ns()}"
    # Some platforms limit SHM name length; keep it reasonable
    shm_name = shm_name[-128:] if len(shm_name) > 128 else shm_name
    shm_name = shm_name.replace(".", "_")

    nbytes = array.nbytes
    shm = shared_memory.SharedMemory(create=True, size=nbytes, name=shm_name)
    try:
        dst = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
        dst[:] = array[:]
    except Exception:
        shm.close()
        shm.unlink()
        raise
    # We close in this process but do NOT unlink yet — the child will open it.
    shm.close()
    return shm_name, array.shape, str(array.dtype)


def _open_shared_array(shm_name: str, shape: tuple, dtype: str) -> np.ndarray:
    """Open a shared_memory buffer created by _create_shared_array."""
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    # Return a copy so we can safely close/unlink the shm afterwards
    return arr.copy(), shm


def _unlink_shared(shm_name: str) -> None:
    """Best-effort unlink of a shared_memory segment."""
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        shm.close()
        shm.unlink()
    except Exception:
        pass


# ==============================================================================
# Background writer loop (runs in child process)
# ==============================================================================

def _writer_loop(
    queue: mp.Queue,
    output_dir: str,
    fps: int,
    info_features: dict,
    repo_id: str,
    robot_type: str,
    custom_features: list[CustomFeatureConfig],
    video_codec: str,
    video_crf: int,
    video_pix_fmt: str,
    video_gop: int | None,
    video_fast_decode: int,
) -> None:
    """Background process: receives shm descriptors, encodes videos, writes parquet + metadata."""
    import jsonlines

    root = Path(output_dir) / repo_id
    root.mkdir(parents=True, exist_ok=True)

    # Redirect C-level stderr to log file (SVT-AV1 encoder is noisy)
    log_file = root / "writer.log"
    try:
        _fd = os.open(str(log_file), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.dup2(_fd, 2)
        sys.stderr = open(str(log_file), "a")
    except Exception:
        pass

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )

    # Metadata paths
    meta_dir = root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    info_path = meta_dir / "info.json"
    episodes_jsonl_path = meta_dir / "episodes.jsonl"
    tasks_jsonl_path = meta_dir / "tasks.jsonl"
    episodes_stats_jsonl_path = meta_dir / "episodes_stats.jsonl"

    video_keys = [k for k, v in info_features.items() if v.get("dtype") == "video"]

    # Init info.json if not present
    if not info_path.exists():
        info = {
            "codebase_version": "v2.1",
            "robot_type": robot_type,
            "total_episodes": 0,
            "total_frames": 0,
            "total_tasks": 0,
            "total_videos": 0,
            "total_chunks": 0,
            "chunks_size": 1000,
            "fps": fps,
            "splits": {},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": info_features,
        }
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    # Restore state from existing episodes
    task_index_map: dict[str, int] = {}
    total_frames = 0
    total_episodes = 0
    if episodes_jsonl_path.exists():
        with jsonlines.open(episodes_jsonl_path, "r") as reader:
            for ep in reader:
                idx = ep["episode_index"]
                total_episodes = max(total_episodes, idx + 1)
                total_frames += ep["length"]
    if tasks_jsonl_path.exists():
        with jsonlines.open(tasks_jsonl_path, "r") as reader:
            for t in reader:
                task_index_map[t["task"]] = t["task_index"]

    while True:
        item = queue.get()
        if item is SENTINEL:
            break

        # item = (shm_desc, custom_values, task)
        shm_desc, custom_values, task = item

        try:
            # Reconstruct arrays from shared memory
            episode_data: dict[str, np.ndarray] = {}
            shm_handles = []
            for key, (shm_name, shape, dtype) in shm_desc.items():
                arr, shm = _open_shared_array(shm_name, shape, dtype)
                episode_data[key] = arr
                shm_handles.append(shm)

            episode_length = len(next(iter(episode_data.values())))
            episode_index = total_episodes
            episode_chunk = episode_index // 1000

            data_chunk_dir = root / f"data/chunk-{episode_chunk:03d}"
            video_chunk_dir = root / f"videos/chunk-{episode_chunk:03d}"
            data_chunk_dir.mkdir(parents=True, exist_ok=True)
            video_chunk_dir.mkdir(parents=True, exist_ok=True)

            # Encode videos using native encode_video_frames
            # We need to save frames as PNGs first because encode_video_frames expects them.
            # This is still disk I/O but only for the encoded episode, not per-frame temp files.
            with tempfile.TemporaryDirectory(prefix="vid_") as tmp_img_dir:
                tmp_img_path = Path(tmp_img_dir)
                for video_key in video_keys:
                    cam_short = video_key.rsplit(".", 1)[-1]
                    if cam_short not in episode_data:
                        continue
                    images = episode_data[cam_short]  # [N, H, W, 3] uint8
                    if len(images) == 0:
                        continue

                    # Write PNGs for native encoder
                    cam_img_dir = tmp_img_path / cam_short
                    cam_img_dir.mkdir(parents=True, exist_ok=True)
                    for i, img in enumerate(images):
                        Image.fromarray(img.astype(np.uint8)).save(
                            cam_img_dir / f"frame_{i:06d}.png"
                        )

                    video_dst = video_chunk_dir / video_key / f"episode_{episode_index:06d}.mp4"
                    video_dst.parent.mkdir(parents=True, exist_ok=True)

                    encode_video_frames(
                        imgs_dir=cam_img_dir,
                        video_path=video_dst,
                        fps=fps,
                        vcodec=video_codec,
                        pix_fmt=video_pix_fmt,
                        g=video_gop,
                        crf=video_crf,
                        fast_decode=video_fast_decode,
                        overwrite=True,
                    )

                    # Store video path for stats computation
                    episode_data[video_key] = [str(video_dst)] * episode_length

            # Task index
            if task not in task_index_map:
                task_index_map[task] = len(task_index_map)
            task_idx = task_index_map[task]

            # Build columns dynamically
            columns: dict[str, Any] = {}

            # User data columns (action, observation.state, etc.)
            for key, feat in info_features.items():
                if key in DEFAULT_FEATURES:
                    continue
                if feat["dtype"] in ["image", "video"]:
                    columns[key] = episode_data.get(key, [""] * episode_length)
                elif key in custom_values:
                    val = custom_values[key]
                    if np.isscalar(val):
                        columns[key] = np.full(episode_length, val, dtype=feat["dtype"])
                    else:
                        columns[key] = np.asarray(val, dtype=feat["dtype"])
                elif key in episode_data:
                    columns[key] = episode_data[key]

            # Standard columns
            frame_indices = np.arange(episode_length, dtype=np.int64)
            columns["timestamp"] = frame_indices.astype(np.float64) / fps
            columns["frame_index"] = frame_indices
            columns["episode_index"] = np.full(episode_length, episode_index, dtype=np.int64)
            columns["index"] = np.arange(total_frames, total_frames + episode_length, dtype=np.int64)
            columns["task_index"] = np.full(episode_length, task_idx, dtype=np.int64)

            # Build parquet
            arrays = []
            field_names = []
            hf_features = get_hf_features_from_features(info_features)

            for key in columns:
                data = columns[key]
                # Handle FixedSizeList for vector features (action, observation.state)
                if isinstance(data, np.ndarray) and data.ndim == 2:
                    dim = data.shape[1]
                    arr = pa.FixedSizeListArray.from_arrays(data.ravel(), dim)
                elif isinstance(data, np.ndarray):
                    arr = pa.array(data)
                else:
                    arr = pa.array(data)
                arrays.append(arr)
                field_names.append(key)

            table = pa.Table.from_arrays(arrays, field_names)
            parquet_path = data_chunk_dir / f"episode_{episode_index:06d}.parquet"
            pq.write_table(table, parquet_path)

            # Compute stats using native compute_episode_stats
            stats_input = {}
            for key, feat in info_features.items():
                if key in DEFAULT_FEATURES:
                    continue
                if feat["dtype"] in ["image", "video"]:
                    stats_input[key] = episode_data.get(key, [""] * episode_length)
                elif key in episode_data:
                    stats_input[key] = episode_data[key]
                elif key in custom_values:
                    val = custom_values[key]
                    if np.isscalar(val):
                        stats_input[key] = np.full(episode_length, val, dtype=feat["dtype"])
                    else:
                        stats_input[key] = np.asarray(val, dtype=feat["dtype"])

            ep_stats = compute_episode_stats(stats_input, info_features)
            serialized_stats = {
                "episode_index": episode_index,
                "stats": serialize_dict(ep_stats),
            }
            with jsonlines.open(episodes_stats_jsonl_path, "a") as stats_writer:
                stats_writer.write(serialized_stats)

            # Append episode summary
            with jsonlines.open(episodes_jsonl_path, "a") as ep_writer:
                ep_writer.write({
                    "episode_index": episode_index,
                    "tasks": [task],
                    "length": episode_length,
                })

            total_episodes += 1
            total_frames += episode_length
            logger.info(
                f"[HybridWriter] Episode {episode_index}: {episode_length} frames, task={task}"
            )

        except Exception as e:
            logger.error(f"[HybridWriter] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up shared memory
            for key, (shm_name, _shape, _dtype) in shm_desc.items():
                _unlink_shared(shm_name)

    # Finalize metadata
    with jsonlines.open(tasks_jsonl_path, "w") as tw:
        for task_name, tidx in sorted(task_index_map.items(), key=lambda x: x[1]):
            tw.write({"task_index": tidx, "task": task_name})

    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        info["total_episodes"] = total_episodes
        info["total_frames"] = total_frames
        info["total_tasks"] = len(task_index_map)
        info["total_videos"] = total_episodes * len(video_keys)
        info["total_chunks"] = (total_episodes - 1) // 1000 + 1 if total_episodes > 0 else 0
        info["splits"] = {"train": f"0:{total_episodes}"}
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    logger.info(f"[HybridWriter] Shutdown. Total: {total_episodes} episodes, {total_frames} frames.")


# ==============================================================================
# Main writer class
# ==============================================================================

class HybridLeRobotWriter:
    """Async dataset writer using SharedMemory for zero-copy IPC and native LeRobot utilities.

    Args:
        output_dir: Root directory for dataset output.
        info_features: Feature definitions (from hw_to_dataset_features).
        fps: Recording frame rate.
        repo_id: Dataset repository identifier.
        robot_type: Robot type string.
        custom_features: List of CustomFeatureConfig for extra fields.
        video_codec: Video codec ("libsvtav1", "h264", "hevc").
        video_crf: Constant rate factor for quality.
        video_pix_fmt: Pixel format.
        video_gop: Group of pictures (keyframe interval).
        video_fast_decode: Enable fast-decode tuning.
    """

    def __init__(
        self,
        output_dir: str,
        info_features: dict,
        fps: int = 30,
        repo_id: str = "dataset",
        robot_type: str = "",
        custom_features: list[CustomFeatureConfig] | None = None,
        video_codec: str = "libsvtav1",
        video_crf: int = 30,
        video_pix_fmt: str = "yuv420p",
        video_gop: int | None = 2,
        video_fast_decode: int = 0,
    ):
        self.output_dir = output_dir
        self.fps = fps
        self.repo_id = repo_id
        self.robot_type = robot_type
        self.custom_features = custom_features or []
        self.video_codec = video_codec
        self.video_crf = video_crf
        self.video_pix_fmt = video_pix_fmt
        self.video_gop = video_gop
        self.video_fast_decode = video_fast_decode

        # Merge custom features into info_features
        self.info_features = self._build_merged_features(info_features)

        self.queue: mp.Queue = mp.Queue(maxsize=4)
        self.process = mp.Process(
            target=_writer_loop,
            args=(
                self.queue,
                output_dir,
                fps,
                self.info_features,
                repo_id,
                robot_type,
                self.custom_features,
                video_codec,
                video_crf,
                video_pix_fmt,
                video_gop,
                video_fast_decode,
            ),
            daemon=True,
        )
        self.process.start()
        logger.info(f"[HybridWriter] Started, output: {output_dir}/{repo_id}")

    def _build_merged_features(self, base_features: dict) -> dict:
        features = dict(base_features)
        for cf in self.custom_features:
            features[cf.name] = {
                "dtype": cf.dtype,
                "shape": cf.shape,
                "names": cf.names,
            }
        for key, val in DEFAULT_FEATURES.items():
            if key not in features:
                features[key] = val
        return features

    def submit(self, episode_data: dict, task: str) -> None:
        """Submit an episode for background encoding.

        Args:
            episode_data: dict with keys:
                - images: {cam_name: np.ndarray [N, H, W, 3] uint8}
                - observation.state: np.ndarray [N, state_dim]
                - action: np.ndarray [N, action_dim]
                - Any custom fields needed by custom_features value_fn
            task: Task description string.
        """
        # Build shared memory descriptors for numeric arrays
        shm_desc: dict[str, tuple[str, tuple, str]] = {}

        for key, data in episode_data.items():
            if key == "images" or not isinstance(data, np.ndarray):
                continue
            shm_name, shape, dtype = _create_shared_array(data, prefix=f"ep_{key}")
            shm_desc[key] = (shm_name, shape, dtype)

        # Images are passed by camera name directly
        for cam_name, images in episode_data.get("images", {}).items():
            if isinstance(images, np.ndarray):
                shm_name, shape, dtype = _create_shared_array(images, prefix=f"img_{cam_name}")
                shm_desc[cam_name] = (shm_name, shape, dtype)

        # Evaluate custom feature values
        custom_values = {}
        for cf in self.custom_features:
            try:
                custom_values[cf.name] = cf.value_fn(episode_data)
            except Exception as e:
                logger.warning(f"Custom feature '{cf.name}' evaluation failed: {e}")
                custom_values[cf.name] = 0 if "int" in cf.dtype else 0.0

        self.queue.put((shm_desc, custom_values, task))

    def shutdown(self, timeout: float = 120.0) -> None:
        """Send sentinel and wait for background encoding to finish."""
        pending = self.queue.qsize()
        if pending > 0:
            print(f"\n[HybridWriter] {pending} episode(s) pending, waiting...")
        self.queue.put(SENTINEL)
        print("[HybridWriter] Waiting for background encoding to complete...")
        self.process.join(timeout=timeout)
        if self.process.is_alive():
            logger.warning("[HybridWriter] Process did not exit, terminating.")
            self.process.terminate()
        else:
            print("[HybridWriter] Encoding finished, all data saved.")

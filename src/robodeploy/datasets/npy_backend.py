#!/usr/bin/env python
"""NPY-backed dataset storage for body-teaching and low-memory recording.

Provides LeRobotDatasetNPY (writes raw .npy instead of PNG) and BackgroundVideoEncoder
(async NPY->MP4 encoding via a separate process).
"""

import json
import logging
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import numpy as np

from robodeploy.datasets.compute_stats import compute_episode_stats
from robodeploy.datasets.lerobot_dataset import LeRobotDataset
from robodeploy.datasets.utils import (
    check_timestamps_sync,
    get_episode_data_index,
    validate_episode_buffer,
    write_info,
)
from robodeploy.datasets.video_utils import encode_video_from_npy, get_video_info

logger = logging.getLogger(__name__)


# ==============================================================================
# Background video encoder (mp.Process, async -- doesn't block recording)
# ==============================================================================

def _video_encoder_loop(queue: mp.Queue, info_path: str, video_keys: list[str], log_path: str) -> None:
    """Background process: encodes NPY dirs to MP4, cleans up, updates metadata."""
    log_file = open(log_path, "a", buffering=1)  # line-buffered

    # Redirect stderr to log file to capture SVT-AV1 and other C-library output
    os.dup2(log_file.fileno(), 2)

    def _log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        log_file.write(f"[Encoder {ts}] {msg}\n")

    _log(f"Started (PID {os.getpid()})")

    while True:
        item = queue.get()
        if item is None:
            break
        npy_dir, video_path, fps = item
        try:
            encode_video_from_npy(npy_dir, video_path, fps)
            shutil.rmtree(npy_dir, ignore_errors=True)
            if Path(info_path).exists():
                with open(info_path) as f:
                    info = json.load(f)
                info["total_videos"] += 1
                if info["total_videos"] == 1:
                    for key in video_keys:
                        if not info["features"].get(key, {}).get("info"):
                            vpath = str(
                                Path(info_path).parent.parent
                                / info["video_path"].format(
                                    episode_chunk=0, video_key=key, episode_index=0
                                )
                            )
                            if Path(vpath).is_file():
                                info["features"][key]["info"] = get_video_info(vpath)
                with open(info_path, "w") as f:
                    json.dump(info, f, indent=2)
            _log(f"Video done: {video_path.name}")
        except Exception as e:
            _log(f"Error encoding {video_path}: {e}")
            import traceback

            log_file.write(traceback.format_exc())
            log_file.flush()


class BackgroundVideoEncoder:
    """Manages a background process for async NPY -> MP4 video encoding."""

    def __init__(self, info_path: str, video_keys: list[str], log_dir: str | None = None):
        self.queue: mp.Queue = mp.Queue(maxsize=8)
        self.info_path = info_path
        self.video_keys = video_keys
        log_path = str(Path(log_dir or Path(info_path).parent.parent) / "encoder.log")
        self.process = mp.Process(
            target=_video_encoder_loop,
            args=(self.queue, info_path, video_keys, log_path),
            daemon=True,
        )
        self.process.start()
        logger.info("[Encoder] Background video encoder started (log: %s).", log_path)

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
            print(f"[Encoder] {pending} video(s) pending, waiting for encoding...")
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
        """Write raw numpy array as .npy -- no compression, very fast."""
        if hasattr(image, "cpu"):  # torch.Tensor
            image = image.cpu().numpy()
        fpath.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(fpath), np.asarray(image, dtype=np.uint8))

    def clear_episode_buffer(self) -> None:
        episode_index = self.episode_buffer["episode_index"]
        for key in self.meta.video_keys:
            img_dir = self._get_image_file_path(episode_index, key, 0).parent
            if img_dir.is_dir():
                shutil.rmtree(img_dir)
        self.episode_buffer = self.create_episode_buffer()

    def encode_episode_videos(self, episode_index: int) -> None:
        """Encode NPY frame directories to MP4."""
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            if video_path.is_file():
                continue
            img_dir = self._get_image_file_path(episode_index, key, 0).parent
            if getattr(self, "_bg_encoder", None) is not None:
                self._bg_encoder.submit(img_dir, video_path, self.fps)
            else:
                encode_video_from_npy(img_dir, video_path, self.fps)
                shutil.rmtree(img_dir)

        if (
            len(self.meta.video_keys) > 0
            and episode_index == 0
            and getattr(self, "_bg_encoder", None) is None
        ):
            self.meta.update_video_info()
            write_info(self.meta.info, self.meta.root)

    def save_episode_async(self) -> None:
        """Write parquet + update metadata + reset buffer. Video encoding is handed
        off to the background encoder (must call set_video_encoder() first).
        """
        if getattr(self, "_bg_encoder", None) is None:
            self.save_episode()
            return

        episode_buffer = self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        try:
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

            for key, ft in self.features.items():
                if key in ["index", "episode_index", "task_index"] or ft["dtype"] in [
                    "image",
                    "video",
                ]:
                    continue
                episode_buffer[key] = np.stack(episode_buffer[key])
                if ft["shape"] == (1,) and episode_buffer[key].ndim == 2 and episode_buffer[key].shape[-1] == 1:
                    episode_buffer[key] = episode_buffer[key].squeeze(-1)

            self._save_episode_table(episode_buffer, episode_index)

            ep_stats = compute_episode_stats(episode_buffer, self.features)

            saved_total_videos = self.meta.info.get("total_videos", 0)
            self.encode_episode_videos(episode_index)

            self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)
            self.meta.info["total_videos"] = saved_total_videos
            write_info(self.meta.info, self.meta.root)

            ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
            ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
            check_timestamps_sync(
                episode_buffer["timestamp"],
                episode_buffer["episode_index"],
                ep_data_index_np,
                self.fps,
                self.tolerance_s,
            )

            pending = (
                getattr(self, "_bg_encoder", None).pending()
                if getattr(self, "_bg_encoder", None)
                else 0
            )
            print(
                f"[Save] Episode {episode_index}: {episode_length} frames. "
                f"Video encoding in background ({pending} jobs in queue)."
            )
        finally:
            self.episode_buffer = self.create_episode_buffer()

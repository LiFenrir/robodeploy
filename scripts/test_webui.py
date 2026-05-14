#!/usr/bin/env python
"""
Test script for WebUI — replays a LeRobot v2.1 dataset as if it were live,
streaming decoded video frames through WebUIServer.

Usage:
    python scripts/test_webui.py \
        --dataset_dir deploy/data_collection/s1_data/lerobot/bi_s1_full_mixed \
        --webui_port 8080
"""

import argparse
import json
import logging
import threading
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_video_frames(video_path: str, fps: int, num_frames: int) -> np.ndarray:
    """Decode all frames from an MP4 video into a numpy array [N, H, W, 3] uint8.

    Uses PyAV directly since torchvision.io.VideoReader may not be available.
    """
    import av

    container = av.open(video_path)
    stream = container.streams.video[0]

    tb = stream.time_base  # Fraction
    target_pts_list = [int(i / fps * tb.denominator / tb.numerator)
                       for i in range(num_frames)]

    # Decode all frames, collecting only those near our target PTS values
    frames_out = []
    target_idx = 0

    for packet in container.demux(stream):
        for frame in packet.decode():
            if target_idx >= len(target_pts_list):
                break
            current_pts = frame.pts
            target_pts = target_pts_list[target_idx]
            # pyav demuxes sequentially; frame pts should closely match targets
            if current_pts >= target_pts:
                img = frame.to_ndarray(format="rgb24")  # [H, W, 3] uint8
                frames_out.append(img)
                target_idx += 1
        if target_idx >= len(target_pts_list):
            break

    container.close()

    if len(frames_out) < num_frames:
        logger.warning("Decoded %d/%d frames from %s", len(frames_out), num_frames, video_path)

    if not frames_out:
        return np.empty((0,), dtype=np.uint8)

    return np.stack(frames_out)  # [N, H, W, 3] uint8


def load_episode(dataset_dir: Path, episode_idx: int, video_keys: list[str], fps: int):
    """Load parquet metadata + all video frames for one episode.

    Returns:
        video_frames: {cam_short_name: np.ndarray [N, H, W, 3] uint8}
        other_data: dict with keys like action, observation.state, etc.
    """
    import pyarrow.parquet as pq

    chunk = episode_idx // 1000
    parquet_path = dataset_dir / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_idx:06d}.parquet"

    if not parquet_path.exists():
        return None

    table = pq.read_table(parquet_path)
    num_frames = table.num_rows

    video_frames = {}
    for vk in video_keys:
        cam_short = vk.rsplit(".", 1)[-1]
        video_path = (
            dataset_dir / "videos" / f"chunk-{chunk:03d}" / vk / f"episode_{episode_idx:06d}.mp4"
        )
        if video_path.exists():
            logger.info("  Loading %s (%s frames)...", cam_short, num_frames)
            video_frames[cam_short] = load_video_frames(str(video_path), fps, num_frames)
        else:
            logger.warning("  Video not found: %s", video_path)

    return {"video_frames": video_frames, "num_frames": num_frames}


def run_test(dataset_dir: str, port: int = 8080, fps: int | None = None,
             loop: bool = True, start_episode: int = 0):
    """Main: start WebUI and replay dataset episodes."""
    from lerobot_mini.webui.server import WebUIServer

    root = Path(dataset_dir)
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        logger.error("Dataset not found at %s (missing meta/info.json)", root)
        return

    with open(info_path) as f:
        info = json.load(f)

    if fps is None:
        fps = info.get("fps", 30)

    # Find video keys from features
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    camera_names = [k.rsplit(".", 1)[-1] for k in video_keys]

    total_episodes = info["total_episodes"]
    logger.info("Dataset: %s episodes, %d cameras, %d FPS",
                total_episodes, len(camera_names), fps)
    logger.info("Cameras: %s", camera_names)

    # Shared state (mirrors record_hybrid.py pattern)
    state_ref = {"obs": None, "mode": "teleop", "control_mode": "mixed", "stop": False, "inference_ok": True}
    recording_ref = {"recording": False, "episode": 0, "frames": 0}
    stop_ref = {"stop": False}
    obs_lock = threading.Lock()

    # Command handlers
    def _cmd_switch_mode(_data=None):
        current = state_ref["mode"]
        state_ref["mode"] = "policy" if current == "teleop" else "teleop"
        print(f"[Mode] {state_ref['mode'].upper()}")
        return None

    def _cmd_toggle_record(_data=None):
        recording_ref["recording"] = not recording_ref["recording"]
        if recording_ref["recording"]:
            print(f"[Recording] ON (episode {recording_ref['episode']})")
            webui.on_recording_started()
        else:
            print("[Recording] OFF")
            webui.on_recording_stopped()
        return None

    def _cmd_save(data):
        label = data.get("label", -1) if data else -1
        if recording_ref.get("recording", False):
            recording_ref["recording"] = False
            webui.on_recording_stopped()
            if label >= 0:
                webui.on_episode_saved(recording_ref["episode"], recording_ref.get("frames", 0), label)
                recording_ref["episode"] += 1
                print(f"[Save] Episode saved (label={label}). Next: {recording_ref['episode']}")
            else:
                print("[Save] Discarded")
            recording_ref["frames"] = 0
        else:
            print("[Save] Not recording, nothing to save")
        return None

    def _cmd_reset_zero(_data=None):
        if recording_ref.get("recording", False):
            return {"error": "Cannot reset while recording"}
        print("[Reset] Zero position (simulated)")
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
        port=port,
        command_handlers=command_handlers,
        fps=fps,
    )
    webui.start()
    logger.info("WebUI at http://localhost:%d", port)
    logger.info("Press Ctrl+C to stop.")

    frame_interval = 1.0 / fps
    episode_idx = start_episode

    try:
        while not stop_ref["stop"]:
            # Load current episode
            logger.info("--- Episode %d/%d ---", episode_idx, total_episodes - 1)
            ep_data = load_episode(root, episode_idx, video_keys, fps)
            if ep_data is None:
                logger.warning("Episode %d not found, skipping", episode_idx)
                episode_idx += 1
                if episode_idx >= total_episodes:
                    episode_idx = 0 if loop else None
                    if episode_idx is None:
                        break
                continue

            num_frames = ep_data["num_frames"]
            recording_ref["frames"] = 0

            for fi in range(num_frames):
                if stop_ref["stop"]:
                    break

                loop_start = time.perf_counter()

                # Build obs dict
                obs = {}
                for cam_name in camera_names:
                    if cam_name in ep_data["video_frames"]:
                        obs[cam_name] = ep_data["video_frames"][cam_name][fi]

                with obs_lock:
                    state_ref["obs"] = obs

                if recording_ref.get("recording", False):
                    recording_ref["frames"] = fi + 1

                # Status line
                if fi > 0 and fi % 60 == 0:
                    rec_str = "[REC]" if recording_ref.get("recording") else "[IDLE]"
                    mode_str = state_ref["mode"].upper()
                    print(f"{rec_str} [{mode_str}] ep={episode_idx} frame={fi}/{num_frames}")

                # Sleep to match FPS
                dt = time.perf_counter() - loop_start
                sleep_time = frame_interval - dt
                if sleep_time > 0:
                    time.sleep(sleep_time)

            logger.info("Episode %d finished (%d frames)", episode_idx, num_frames)

            episode_idx += 1
            if episode_idx >= total_episodes:
                if loop:
                    episode_idx = 0
                    logger.info("Looping back to episode 0")
                else:
                    break

    except KeyboardInterrupt:
        print("\n[Interrupted]")
    finally:
        print("Shutting down...")
        stop_ref["stop"] = True
        webui.stop()
        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebUI test with dataset replay")
    parser.add_argument("--dataset_dir", type=str,
                        default="deploy/data_collection/s1_data/lerobot/bi_s1_full_mixed",
                        help="Path to LeRobot v2.1 dataset directory")
    parser.add_argument("--webui_port", type=int, default=8080)
    parser.add_argument("--fps", type=int, default=None,
                        help="Replay FPS (default: read from dataset info.json)")
    parser.add_argument("--no-loop", action="store_true",
                        help="Stop after last episode instead of looping")
    parser.add_argument("--start_episode", type=int, default=0,
                        help="Episode to start from")
    args = parser.parse_args()

    run_test(
        dataset_dir=args.dataset_dir,
        port=args.webui_port,
        fps=args.fps,
        loop=not args.no_loop,
        start_episode=args.start_episode,
    )

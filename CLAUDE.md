# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Robodeploy is a robotics deployment & data collection toolkit based on LeRobot, adapted for real-world robot control, teleoperation recording, and policy inference. Supports S1 (single/bimanual), Innov Arm, and ARX X5 robots.

## Build/Run/Test Commands

```bash
# Install (base + dev/test)
pip install -e ".[dev,test]"

# Robot-specific extras: [innov] (mujoco), [feetech], [dynamixel], [intelrealsense], [qt] (PyQt6 UI)
# Full install with all optional drivers
pip install -e ".[all]"

# Lint & format
ruff check src/
ruff format src/

# Note: tests/ suite has been removed; no pytest commands (dev/test deps kept in pyproject.toml for future use)

# Start WebUI monitoring dashboard
python -m robodeploy.webui           # defaults to http://localhost:5000
```

## Architecture

### Core abstraction layers

Three pluggable interfaces connected by `draccus.ChoiceRegistry` (dataclass-based CLI + registry pattern):

- **Robot** (`src/robodeploy/robots/`) — `get_observation()` → `dict`, `send_action(action)` → `None`, `connect()`/`disconnect()`. Configs subclass `RobotConfig`. Instantiated via `make_robot_from_config(cfg)`.
- **Teleoperator** (`src/robodeploy/teleoperators/`) — `get_action()` → `dict`, `connect()`/`disconnect()`. Configs subclass `TeleoperatorConfig`. Instantiated via `make_teleoperator_from_config(cfg)`.
- **Policy** (`src/robodeploy/policy_clients/`) — `BasePolicy.infer(obs)` → action `dict`, `reset()`. Primary implementation: OpenPI WebSocket client.

New robot/teleoperator types are auto-discovered — just place a package under `robots/` or `teleoperators/` with a config class that inherits from the base `*Config`.

### Data recording pipeline (`src/robodeploy/scripts/record_dataset.py`)

Main entry point for data collection. The 30fps control loop runs three phases per tick:
1. `robot.get_obs()` → populate `state_ref["obs"]` and camera images
2. Depending on mode: read leader position (teleop) or `stream_buffer.pop_next_action()` (policy)
3. `robot.send_action(action)` + `dataset.add_frame(frame_dict)`

Three control modes — `teleop` (pure leader-follower), `policy` (pure inference), `mixed` (P key hot-switch between them). When switching policy→teleop, `interpolate_leader_to_follower()` cosine-blends the leader arms to match follower positions for smooth handoff.

### Temporal smoothing (`src/robodeploy/utils/stream_buffer.py`)

`StreamActionBuffer` smooths overlapping action chunks from the policy server. New chunks are linearly crossfaded with the tail of the previous chunk. Key tuning knobs: `latency_k` (drop first k steps of new chunk), `min_smooth_steps` (minimum overlap for blending). Use the `action-analyzer` skill to tune these from action logs.

### Dataset storage

Two backends:
- **Standard** (`LeRobotDataset` in `lerobot_dataset.py`) — parquet + video files (PNG → MP4), HuggingFace-compatible format.
- **NPY** (`LeRobotDatasetNPY` in `npy_backend.py`) — O(1) RAM, writes raw `.npy` frames, `BackgroundVideoEncoder` subprocess asynchronously encodes NPY→MP4 after episode save.

### WebUI (`src/robodeploy/webui/server.py`)

FastAPI server in a background thread. Pure WebSocket: JSON for status/commands, binary for video frames. Shares state with the main loop via `state_ref`/`recording_ref`/`stop_ref` dicts and an `obs_lock`.

### Qt UI (`src/robodeploy/qtui/`)

PyQt6 desktop app (extra `[qt]`, entry `robodeploy-qt` / `python -m robodeploy.qtui`). Four tabs: 相机调试 (enumerate/preview/intrinsics export), 数据采集, 数据检查 (browse/replay/validate/scripts/v3.0 export), 部署 (policy connect test + launch).

- `record_dataset.py --front_end=qt` runs the control loop in a worker thread; main thread runs QApplication. Qt reuses the same `pending_ref` command entry as WebUI — `record_loop()` internals are untouched. `QtRecordBridge` mirrors the `WebUIServer` hook interface (`on_recording_started`/`on_episode_saved`/...) and provides blocking `request_label()` for the episode-time-limit save dialog.
- Threading rule: hardware IO never on the Qt main thread; frames/states cross threads via Signals or 30Hz QTimer polling of `state_ref["obs"]` under `obs_lock` (drop-frame strategy, never blocks the control loop).
- URDF live view: `UrdfWorker` (mujoco offscreen renderer in a QThread) maps `state_ref` joint values → `mjData.qpos`; config via `urdf_path`/`urdf_joint_indices`/`urdf_joint_scale`. Degrades to numeric joint readout if GL init fails.
- Dataset replay decodes video via ffmpeg subprocess pipe (`FfmpegVideoReader`, rawvideo RGB24 + `-ss` input seek) — cv2 cannot decode the AV1 recordings.
- Standalone tabs launch `record_dataset` and dataset maintenance scripts as QProcess subprocesses with live logs; v3.0 export calls lerobot's `convert_dataset_v21_to_v30.py` (path configurable).

### Offline testing without hardware

When no physical motors/cameras are connected, reuse existing LeRobot-format datasets for offline testing. `conftest.py` auto-detects missing hardware and skips relevant tests.

## Code Conventions

- **Double quotes** for all Python strings (ruff-enforced)
- **110-char line width**
- **Google-style docstrings**
- **Apache 2.0 license header** required on every `.py` file
- **`src` layout** — all imports use `from robodeploy import ...` or `from robodeploy.xxx import ...`
- **`scripts/` is standalone** — hardware-interaction deploy/test tools, not part of the Python package. Data collection & dataset processing scripts live in `src/robodeploy/scripts/` (invoked via `python -m robodeploy.scripts.<module>`)
- Configuration via **draccus** CLI args: `--robot.type=bi_s1_follower`, `--policy.host=localhost`, etc.
- `torchcodec` is unavailable on Windows, ARM Linux, and macOS x86_64 — this is expected
- `[feetech]`, `[dynamixel]`, `[intelrealsense]` are optional extras, not in the base install

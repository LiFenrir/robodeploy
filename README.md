# robodeploy

Robotics deployment & data collection toolkit — based on LeRobot, adapted for real-world robot control and policy inference.

---

## Features

- **Multi-robot support** — S1 (single/bimanual), Innov Arm, ARX X5
- **Bimanual teleoperation** — single-arm and dual-arm leader-follower setups
- **Multi-mode data collection** — teleop, policy inference, mixed (hot-switch with P key)
- **Camera integration** — OpenCV USB cameras, Intel RealSense depth cameras
- **Policy inference** — OpenPI WebSocket client with temporal smoothing and DAgger alignment
- **WebUI** — FastAPI WebSocket — real-time monitoring and control dashboard

## Project Layout

```
robodeploy/
├── src/robodeploy/         # core package: cameras, datasets, motors, robots, teleoperators, webui, scripts
├── scripts/                   # hardware deploy/test tools (replay, inference check, RTC smoke test)
├── examples/                  # usage examples
└── pyproject.toml             # project metadata & dependencies
```

## Quick Start

### Install

```bash
pip install -e ".[dev,test]"           # base + dev/test
pip install -e ".[feetech]"            # Feetech servo driver
pip install -e ".[dynamixel]"          # Dynamixel servo driver
pip install -e ".[intelrealsense]"     # Intel RealSense camera driver
pip install -e ".[innov]"              # Innov Arm (mujoco renderer)
pip install -e ".[qt]"                 # Qt desktop UI (PyQt6)
pip install -e ".[all]"                # everything
```

### Lint & format

```bash
ruff check src/
ruff format src/
```

(The `tests/` suite has been removed; dev/test deps remain in `pyproject.toml` for future use.)

### Start WebUI

```bash
python -m robodeploy.webui           # defaults to http://localhost:5000
```

### Record data

```bash
# Bimanual S1 — pure teleoperation (NPY storage, O(1) RAM)
python -m robodeploy.scripts.record_dataset \
    --robot.type=bi_s1_follower \
    --robot.left_arm_port=/dev/ttyUSB0 --robot.right_arm_port=/dev/ttyUSB1 \
    --teleop.type=bi_s1_leader \
    --teleop.left_arm_port=/dev/ttyUSB2 --teleop.right_arm_port=/dev/ttyUSB3 \
    --control_mode teleop \
    --task="pick and place"

# Mixed mode — policy inference + teleop (P key to toggle)
python -m robodeploy.scripts.record_dataset \
    --robot.type=bi_s1_follower \
    --teleop.type=bi_s1_leader \
    --policy.type=openpi \
    --policy.host=localhost --policy.port=8000 \
    --task="pick and place"
```

See [scripts/README.md](scripts/README.md) for details.

## Tech Stack

| Component | Stack |
|-----------|-------|
| Language | Python 3.10+ |
| ML Backend | PyTorch, HuggingFace datasets & hub |
| Vision | OpenCV, Intel RealSense |
| Config | draccus (dataclass-based CLI) |
| WebUI | FastAPI / uvicorn |
| Video | PyAV (av) |
| Serial | pyserial, Dynamixel SDK, Feetech SDK |
| Policy | OpenPI WebSocket client, msgpack |

## Conventions

- **Double quotes** for all Python strings (ruff-enforced)
- **110-char line width**
- **Google-style docstrings**
- **Apache 2.0 header** on every `.py` file
- **`src` layout** — import from `from robodeploy import ...`

## Notes

- Hardware tests auto-skip in `conftest.py` when no physical motor/camera is connected
- `torchcodec` is unavailable on Windows, ARM Linux, and macOS x86_64 — this is expected
- `[feetech]`, `[dynamixel]`, `[intelrealsense]` are optional extras, not in the base install
- Top-level `scripts/` is **not** part of the Python package — standalone hardware deploy/test entry points only. Data collection & dataset processing scripts live in `src/robodeploy/scripts/` (invoked via `python -m robodeploy.scripts.<module>`)

## License

Apache 2.0 — derived from [LeRobot](https://github.com/huggingface/lerobot).

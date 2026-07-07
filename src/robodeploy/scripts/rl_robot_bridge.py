#!/usr/bin/env python
"""RL training bridge — connects a robodeploy robot to a Training PC for Stage 2 online RL.

This script runs on the **Robot PC** (the machine physically connected to the robot).
It:

1. Connects to the robot hardware via robodeploy.
2. Opens a WebSocket connection to the Training PC (which runs
   ``RemoteWebSocketEnv``).
3. Streams robot observations and receives action chunks in a
   request-response loop.
4. Collects human reward signals (success / failure) via keyboard.

Protocol (msgpack + numpy over WebSocket)::

    Robot PC → Training PC:  {state, images, prompt, reward, done, success}
    Training PC → Robot PC:  {actions, reset}

Usage::

    # bi_s1 robot, Training PC at 192.168.1.100
    python rl_robot_bridge.py \
        --robot.type=bi_s1_follower \
        --robot.left_arm_port=/dev/ttyUSB0 --robot.right_arm_port=/dev/ttyUSB1 \
        --robot.cameras='{"front":{"type":"intelrealsense","width":848,"height":480,"fps":30}}' \
        --host=192.168.1.100 --port=5556 \
        --action_dim=14 --chunk_length=10 \
        --task="fold the box"

    # Without cameras (simulated or test mode)
    python rl_robot_bridge.py --robot.type=s1_follower --robot.port=/dev/ttyUSB0 \
        --host=localhost --port=5556
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np

from robodeploy.policy_clients.openpi import resize_with_pad

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from openpi_client.rl_client import pack_rl_observation, unpack_rl_response  # noqa: E402
from openpi_client.websocket_client_policy import WebsocketClientPolicy  # noqa: E402

# 注册 robot/camera 类型到 draccus ChoiceRegistry
from robodeploy.cameras import CameraConfig  # noqa: F401, E402
from robodeploy.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401, E402
from robodeploy.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401, E402
from robodeploy.robots import (  # noqa: F401, E402
    RobotConfig,
    bi_s1_follower,
    make_robot_from_config,
   
)
from robodeploy.teleoperators import (  # noqa: F401, E402
    TeleoperatorConfig,
    make_teleoperator_from_config,
    bi_s1_leader,
)

from robodeploy.utils.leader_follower_align import (  # noqa: E402
    interpolate_leader_to_follower,
    reset_to_zero,
)
from robodeploy.utils.stream_buffer import StreamActionBuffer  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RLBridgeConfig:
    """Configuration for the RL robot bridge."""

    # Robot hardware (draccus ChoiceRegistry)
    robot: RobotConfig | None = None

    # Teleoperator for human intervention (None = disabled)
    teleoperator: TeleoperatorConfig | None = None

    # Training PC connection
    host: str = "localhost"
    port: int = 5556

    # Action space (must match Stage 2 config)
    action_dim: int = 14
    chunk_length: int = 10

    # Task
    task: str = ""

    # RL toggle: press this key to toggle RL mode on/off.
    # RL ON (default): actor controls, transitions stored in replay buffer.
    # RL OFF: VLA reference controls, transitions NOT stored.
    rl_toggle_key: str = "t"

    # Human intervention (teleoperation takeover)
    # Press this key to toggle teleop control on/off.
    intervention_key: str = "i"

    # Temporal smoothing (StreamActionBuffer)
    latency_k: int = 2  # drop first k steps of new chunk for latency compensation
    min_smooth_steps: int = 8  # minimum overlap length for linear crossfade
    inference_rate: float = 7.0  # policy inference frequency (Hz), controls chunk overlap depth

    # Control
    fps: int = 30  # robot control frequency

    # 每轮 reset 后丢弃前 N 个 chunk，确保流水线排空
    skip_chunks_after_reset: int = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _numpy_to_action_dict(action_np: np.ndarray, action_features: dict) -> dict:
    """Convert [action_dim] numpy array to robot send_action() dict."""
    keys = list(action_features.keys())
    if len(action_np) != len(keys):
        raise ValueError(f"Action dim mismatch: {len(action_np)} vs {len(keys)}")
    return {key: float(action_np[i]) for i, key in enumerate(keys)}


def _stack_front_cameras(images: dict) -> dict:
    """Vertically stack front + front_1 images (front_1 rotated 180°), in-place.

    Some robot setups have two front cameras (stereo pair).  The VLA was
    trained with vertically stacked images, so we must apply the same
    transformation before sending observations.
    """
    if "front" in images and "front_1" in images:
        front = np.asarray(images["front"])
        front_1 = np.asarray(images["front_1"])
        images["front"] = np.concatenate([front, np.rot90(front_1, 2)], axis=0)
        del images["front_1"]
    return images


def _extract_observation(robot) -> dict:
    """Extract an observation dict in the format expected by the VLA.

    Returns a dict with:
        - ``"state"``: joint positions [action_dim]
        - ``"images"``: camera_name → np.ndarray (only if robot has cameras)
        - ``"prompt"``: task instruction
    """
    raw = robot.get_observation()

    # State: use action_features directly (insertion order, matching record_dataset.py)
    state = np.array([raw.get(k, 0.0) for k in robot.action_features], dtype=np.float64)

    # Binarize gripper state for bi_s1 (indices 6 and 13 in 14-dim state)
    for gi in (6, 13):
        if gi < len(state):
            state[gi] = 0.0 if state[gi] < 0.2 else 1.0

    # Images: camera_name → array
    images: dict[str, np.ndarray] = {}
    if hasattr(robot, "cameras"):
        for cam_name in robot.cameras:
            if cam_name in raw:
                images[cam_name] = np.asarray(raw[cam_name])

    # Stack front cameras if stereo pair present
    _stack_front_cameras(images)

    # Image preprocessing: BGR→RGB, resize 224, CHW (matching record_dataset.py → OpenPIPolicyClient)
    rgb_images: dict[str, np.ndarray] = {}
    for cam_name, img in images.items():
        if img is not None:
            rgb_images[cam_name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    payload_images: dict[str, np.ndarray] = {}
    for cam_name, img in rgb_images.items():
        payload_images[cam_name] = resize_with_pad(
            np.array([img]),
            224,
            224,
        )[0].transpose(2, 0, 1)

    return {"state": state, "images": payload_images}


def _check_keypress(extra_keys: str = "") -> str | None:
    """非阻塞读取按键（需外层已设置 cbreak 模式）。

    Returns the key character if mapped, ``None`` otherwise.
    Mapped: ``"s"``, ``"f"``, ``"\\x1b"``, ``"\\x03"``, plus ``extra_keys``.
    """
    import select as _select

    r, _w, _x = _select.select([sys.stdin], [], [], 0)
    if r:
        ch = sys.stdin.read(1)
        if ch in ("s", "f", "\x1b", "\x03") or ch in extra_keys:
            return ch
        logger.info("Key pressed but not mapped: %r (ord=%d)", ch, ord(ch))
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_bridge(cfg: RLBridgeConfig) -> None:
    """Main entry point."""
    # ---- Connect to robot ----
    if cfg.robot is None:
        logger.error("--robot.type is required")
        raise SystemExit(1)

    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    logger.info("Robot '%s' connected.", robot.name)

    # ---- Connect teleoperator (if configured) ----
    teleop = None
    if cfg.teleoperator is not None:
        teleop = make_teleoperator_from_config(cfg.teleoperator)
        teleop.connect()
        logger.info("Teleoperator '%s' connected for human intervention.", teleop.name)

    # ---- Connect to Training PC ----
    logger.info("Connecting to Training PC at %s:%d ...", cfg.host, cfg.port)
    client = WebsocketClientPolicy(host=cfg.host, port=cfg.port)
    logger.info("Connected to Training PC.")

    # ---- State ----
    action_features = robot.action_features
    control_period = 1.0 / cfg.fps
    obs = _extract_observation(robot)
    obs["prompt"] = cfg.task
    reward = 0.0
    done = False
    success = False
    chunk_idx = 0
    episode = 0

    # ---- Intervention state ----
    intervention_active = False  # toggled by intervention_key
    actual_action_chunk: np.ndarray | None = None  # actual executed action [C, d]

    # ---- RL toggle state ----
    rl_active: bool = True  # toggled by rl_toggle_key

    print("=" * 60)
    print("  RL Robot Bridge")
    print(f"  Robot: {robot.name}  |  Training PC: {cfg.host}:{cfg.port}")
    print(f"  Task: {cfg.task}  |  Action dim: {cfg.action_dim}")
    print(f"  Chunk length: {cfg.chunk_length}  |  FPS: {cfg.fps}")
    steps_per_inference = max(1, int(cfg.fps / cfg.inference_rate))
    print(f"  Smoothing: latency_k={cfg.latency_k}  min_smooth={cfg.min_smooth_steps}  "
          f"inf_rate={cfg.inference_rate}Hz  →  {steps_per_inference} steps/inference")
    print(f"  Controls: s=success  f=fail  {cfg.rl_toggle_key}=toggle RL [{('ON' if rl_active else 'OFF')}]")
    if teleop is not None:
        print(f"  Intervention: '{cfg.intervention_key}'=toggle  |  teleop={teleop.name}")
    print("=" * 60)

    # 时序平滑缓冲：线性交叉淡化相邻动作块，消除块间跳变
    action_buffer = StreamActionBuffer(state_dim=cfg.action_dim)

    # 终端 cbreak 模式：按键即时生效无需回车，全程只切换一次
    import termios as _termios
    import tty as _tty

    _stdin_fd = sys.stdin.fileno()
    _old_termios = _termios.tcgetattr(_stdin_fd)
    _tty.setcbreak(_stdin_fd)
    logger.info("Terminal set to cbreak mode, keypress detection active")
    _stop = False

    try:
        while not _stop:
            # ---- Send observation + RL metadata ----
            msg = pack_rl_observation(
                obs, reward=reward, done=done, success=success,
                intervention=intervention_active,
                action=actual_action_chunk,
                rl_active=rl_active,
            )
            resp = client.infer(msg)
            actions, vla_actions, reset_cmd = unpack_rl_response(resp)

            # ---- Handle reset ----
            if reset_cmd:
                logger.info("Reset command received — moving arms to zero")
                action_buffer.clear()
                reset_to_zero(robot, teleop=teleop, action_features=action_features)
                # 等待机械臂稳定 + 训练端状态同步：发送归零后的观测，确保训练端
                # 基于当前关节位置生成动作，而非基于 reset 前的旧状态
                time.sleep(2.0)
                obs = _extract_observation(robot)
                obs["prompt"] = cfg.task
                reward = 0.0
                done = False
                success = False
                chunk_idx = 0
                intervention_active = False
                actual_action_chunk = None
                skip_remaining = cfg.skip_chunks_after_reset
                episode += 1
                logger.info("Episode %d ready, will skip %d chunks", episode, skip_remaining)

                # 等待操作员按 Enter 开始新 episode
                input("Press Enter to start episode...")
                logger.info("\nOperator confirmed, starting episode %d", episode)
                continue

            # Reset 后丢弃前 N 个 chunk，等待流水线排空
            if skip_remaining > 0:
                skip_remaining -= 1
                action_buffer.clear()
                obs = _extract_observation(robot)
                obs["prompt"] = cfg.task
                continue

            # Select actions based on RL toggle
            if rl_active and actions is not None:
                chosen_actions = actions
            elif not rl_active and vla_actions is not None:
                chosen_actions = vla_actions
            else:
                chosen_actions = actions  # fallback

            if chosen_actions is None:
                logger.warning("Empty response from Training PC, retrying...")
                time.sleep(0.5)
                continue

            # ---- Select action source ----
            if intervention_active and teleop is not None:
                # 连续遥操循环：fps 跟随 leader，每 steps_per_inference 步
                # 向 Training PC 发送一次数据（inference_rate），避免卡顿
                teleop_actions: list[np.ndarray] = []
                done = False
                success = False
                step = 0

                while not _stop:
                    t_start = time.time()
                    raw = teleop.get_action()
                    act_np = np.array([float(raw.get(key, 0.0)) for key in action_features], dtype=np.float64)
                    teleop_actions.append(act_np)
                    action_dict = _numpy_to_action_dict(act_np, action_features)
                    robot.send_action(action_dict)

                    key = _check_keypress(extra_keys=cfg.intervention_key + cfg.rl_toggle_key)
                    if key == "\x1b" or key == "\x03":
                        logger.info("Esc/Ctrl-C: stopping bridge")
                        _stop = True
                        break
                    elif key == "f":
                        done = True
                        success = False
                        logger.info("FAILURE during intervention — episode %d", episode)
                        break
                    elif key == "s":
                        reward += 1.0
                        done = True
                        success = True
                        logger.info("SUCCESS during intervention — episode %d", episode)
                        break
                    elif key == cfg.rl_toggle_key:
                        rl_active = not rl_active
                        logger.info("RL toggled %s during intervention",
                                    "ON" if rl_active else "OFF")
                    elif key == cfg.intervention_key:
                        intervention_active = False
                        logger.info("Intervention OFF")
                        break

                    elapsed = time.time() - t_start
                    sleep_t = control_period - elapsed
                    if sleep_t > 0:
                        time.sleep(sleep_t)

                    step += 1
                    if step >= steps_per_inference:
                        _obs = _extract_observation(robot)
                        _obs["prompt"] = cfg.task
                        _chunk = np.stack(teleop_actions[-steps_per_inference:], axis=0)
                        _msg = pack_rl_observation(
                            _obs, reward=0.0, done=False, success=False,
                            intervention=True, action=_chunk, rl_active=rl_active,
                        )
                        client.infer(_msg)  # 发送数据，忽略返回动作
                        step = 0

                action_buffer.clear()
                actual_action_chunk = None  # 数据已在循环内发送
                obs = _extract_observation(robot)
                obs["prompt"] = cfg.task
            else:
                # Normal mode: use Training PC action chunk
                chosen = np.asarray(chosen_actions)  # [C, action_dim]
                action_buffer.integrate_new_chunk(
                    chosen,
                    max_k=cfg.latency_k,
                    min_m=cfg.min_smooth_steps,
                )
                actual_action_chunk = None  # Training PC knows what it sent

                # ---- Execute action chunk ----
                done = False
                success = False

                for k in range(steps_per_inference):
                    act_np = action_buffer.pop_next_action()
                    if act_np is None:
                        break
                    t_start = time.time()

                    action_dict = _numpy_to_action_dict(act_np, action_features)
                    robot.send_action(action_dict)

                    # Check for human reward / toggle signal
                    key = _check_keypress(extra_keys=cfg.intervention_key + cfg.rl_toggle_key)
                    if key == "\x1b" or key == "\x03":
                        logger.info("Esc/Ctrl-C: stopping bridge")
                        _stop = True
                        break
                    elif key == "f":
                        done = True
                        success = False
                        action_buffer.clear()
                        logger.info("FAILURE — episode %d, chunk %d, step %d", episode, chunk_idx, k)
                        break
                    elif key == "s":
                        reward += 1.0
                        done = True
                        success = True
                        action_buffer.clear()
                        logger.info("SUCCESS — episode %d (reward=%.3f)", episode, reward)
                        break
                    elif key == cfg.rl_toggle_key:
                        rl_active = not rl_active
                        logger.info("RL toggled %s (chunk %d, step %d)",
                                    "ON" if rl_active else "OFF", chunk_idx, k)
                    elif key == cfg.intervention_key and teleop is not None:
                        action_buffer.clear()
                        # Align leader to follower before takeover to avoid joint jumps
                        logger.info("Aligning leader to follower before intervention...")
                        leader_pos = teleop.get_action()
                        follower_pos = robot.get_observation()
                        interpolate_leader_to_follower(
                            teleop, leader_pos, follower_pos,
                            action_features, dt=0.05, max_step=0.02,
                        )
                        intervention_active = True
                        logger.info("Intervention ON — teleoperator takeover")
                        break

                    # Enforce control frequency
                    elapsed = time.time() - t_start
                    sleep_t = control_period - elapsed
                    if sleep_t > 0:
                        time.sleep(sleep_t)

            if _stop:
                break

            chunk_idx += 1

            # ---- Get next observation ----
            obs = _extract_observation(robot)
            obs["prompt"] = cfg.task

            # Note: DO NOT auto-reset here.  The Training PC controls the
            # episode lifecycle — it will send a ``reset: true`` command at
            # the start of the next episode via ``env.reset()``.

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    except Exception:
        logger.exception("Fatal error")
    finally:
        _termios.tcsetattr(_stdin_fd, _termios.TCSADRAIN, _old_termios)
        logger.info("Terminal restored.")
        if teleop is not None:
            teleop.disconnect()
            logger.info("Teleoperator disconnected.")
        robot.disconnect()
        logger.info("Robot disconnected. Done.")


def main() -> None:
    from robodeploy.configs.parser import wrap

    @wrap()
    def _main(cfg: RLBridgeConfig) -> None:
        run_bridge(cfg)

    _main()


if __name__ == "__main__":
    main()

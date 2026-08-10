"""OpenPI policy 持续推理脚本。

循环执行：获取观测 -> 策略推理 -> 发送动作。
支持键盘控制：p 暂停/继续，r 重置到零位，q 退出。

Usage:
    python inference_step.py \
        --robot.type=bi_innov_arm_v1 \
        --robot.left_port=/dev/ttyACM0 --robot.right_port=/dev/ttyACM1 \
        --robot.mode=control \
        --robot.cameras='{"top":{"type":"intelrealsense",...}}' \
        --policy.type=openpi --policy.host=192.168.201.203 --policy.port=8000 \
        --task="Pick up the block in front and place it into the black box in the middle." 
"""

import logging
import sys
import termios
import time
import tty
from dataclasses import dataclass

import numpy as np

# Register camera / robot / policy types with draccus ChoiceRegistry
from robodeploy.cameras import CameraConfig  # noqa: F401
from robodeploy.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from robodeploy.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from robodeploy.configs.parser import wrap
from robodeploy.policy_clients import (  # noqa: F401
    PolicyClientConfig,
    openpi,
)
from robodeploy.policy_clients.utils import make_policy_client_from_config
from robodeploy.robots import RobotConfig, make_robot_from_config
from robodeploy.robots.arx_x5 import arx_x5, bi_arx_x5  # noqa: F401
from robodeploy.robots.lerobot_robot_my_arm import bi_innov_arm_v1, innov_arm_v1  # noqa: F401
from robodeploy.utils.keyboard_control import get_keypress
from robodeploy.utils.leader_follower_align import reset_to_zero, smooth_inference_action

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class InferenceStepConfig:
    """Configuration for OpenPI single-step inference."""

    robot: RobotConfig | None = None
    policy: PolicyClientConfig | None = None
    task: str = "fold the box"
    warmup_rounds: int = 10
    action_smooth_max_step: float = 0.05
    reset_to_zero: bool = False
    align_max_step: float = 0.02
    loop_sleep_s: float = 0.01
    pause_key: str = "p"
    reset_key: str = "r"
    quit_key: str = "q"


def numpy_to_action_dict(action_np: np.ndarray, action_features: dict[str, type]) -> dict[str, float]:
    """Convert policy output [D] numpy array to robot.send_action() dict format."""
    keys = list(action_features.keys())
    if len(action_np) != len(keys):
        raise ValueError(f"Action dim mismatch: policy={len(action_np)}, robot={len(keys)}")
    return {key: float(action_np[i]) for i, key in enumerate(keys)}


def _prepare_inference_input(
    obs: dict, action_features: dict[str, type], camera_names: list[str]
) -> tuple[np.ndarray, dict]:
    """Extract state vector and image dict from observation dict for policy inference."""
    state = np.array([obs.get(k, 0.0) for k in action_features], dtype=np.float64)
    images = {cam: np.asarray(obs[cam]) for cam in camera_names if cam in obs}
    return state, images


def _warmup_policy(
    policy, robot, action_features: dict[str, type], camera_names: list[str], rounds: int
) -> None:
    """Run a few dummy inferences to warm up the policy server."""
    logger.info("=" * 60)
    logger.info("推理预热 (%d rounds)...", rounds)
    times = []
    for i in range(rounds):
        obs = robot.get_observation()
        state, images = _prepare_inference_input(obs, action_features, camera_names)
        t0 = time.monotonic()
        try:
            policy.infer(images, state, "")
            elapsed = (time.monotonic() - t0) * 1000
            times.append(elapsed)
            logger.info("  预热 %2d/%d: %.0fms", i + 1, rounds, elapsed)
        except Exception as e:
            logger.warning("  预热 %2d/%d 失败: %s", i + 1, rounds, e)
    if times:
        logger.info(
            "预热完成: avg=%.0fms, min=%.0fms, max=%.0fms",
            np.mean(times),
            np.min(times),
            np.max(times),
        )


def run_inference_step(cfg: InferenceStepConfig) -> None:
    """Connect robot and policy, run one inference step, execute the first action."""
    if cfg.robot is None:
        logger.error("No robot config provided. Use --robot.type=...")
        sys.exit(1)
    if cfg.policy is None:
        logger.error("No policy config provided. Use --policy.type=openpi ...")
        sys.exit(1)

    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    logger.info(f"Robot '{robot.name}' connected (mode={getattr(robot.config, 'mode', 'unknown')}).")

    policy = make_policy_client_from_config(cfg.policy)
    if not policy.connected:
        logger.error("Failed to connect to policy server. Exiting.")
        robot.disconnect()
        sys.exit(1)
    logger.info(
        f"Policy '{cfg.policy.type}' connected at {getattr(cfg.policy, 'host', '?')}:{getattr(cfg.policy, 'port', '?')}"
    )

    camera_names = list(getattr(robot, "cameras", {}).keys())
    action_features = robot.action_features

    if cfg.warmup_rounds > 0:
        _warmup_policy(policy, robot, action_features, camera_names, cfg.warmup_rounds)

    # Set terminal to cbreak mode for single-key commands without Enter.
    fd = sys.stdin.fileno()
    old_tty = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    paused = False
    step = 0
    prev_action: dict[str, float] | None = None
    try:
        while True:
            key = get_keypress()
            if key == cfg.quit_key.lower() or key == cfg.quit_key.upper():
                logger.info(f"Quit key '{key}' pressed, exiting loop.")
                break
            elif key == cfg.pause_key.lower() or key == cfg.pause_key.upper():
                paused = not paused
                logger.info(f"{'Paused' if paused else 'Resumed'}.")
                continue
            elif key == cfg.reset_key.lower() or key == cfg.reset_key.upper():
                logger.info(f"Reset key '{key}' pressed, resetting to zero...")
                reset_to_zero(robot, None, action_features, max_step=cfg.align_max_step)
                prev_action = None
                continue


            step += 1
            obs = robot.get_observation()
            state, images = _prepare_inference_input(obs, action_features, camera_names)

            logger.info(f"Running inference step {step}...")
            t0 = time.monotonic()
            result = policy.infer(images, state, cfg.task)
            elapsed = (time.monotonic() - t0) * 1000
            logger.info(f"Inference took {elapsed:.0f}ms")

            actions = result.get("actions", None)
            if actions is None or len(actions) == 0:
                logger.error("Policy returned no actions.")
                break

            first_action = np.asarray(actions)[0]
            action_dict = numpy_to_action_dict(first_action, action_features)
            logger.info(f"Inferred action: {action_dict}")

            if prev_action is not None and cfg.action_smooth_max_step > 0:
                smooth_inference_action(
                    robot,
                    prev_action,
                    action_dict,
                    action_features,
                    max_step=cfg.action_smooth_max_step,
                )
            sent_action = robot.send_action(action_dict)
            prev_action = action_dict
            logger.info(f"Sent action: {sent_action}")
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, exiting loop.")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_tty)

    if cfg.reset_to_zero:
        logger.info("Resetting to zero position...")
        reset_to_zero(robot, None, action_features, max_step=cfg.align_max_step)

    robot.disconnect()
    logger.info("Done.")


def main() -> None:
    @wrap()
    def _main(cfg: InferenceStepConfig) -> None:
        run_inference_step(cfg)

    _main()


if __name__ == "__main__":
    main()

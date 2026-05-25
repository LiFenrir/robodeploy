# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Leader-follower alignment utilities for teleoperation.

Provides cosine interpolation for aligning leader (teaching) arms to
follower arms when switching from policy inference back to teleoperation.
"""

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


def _build_joint_array(action_dict: dict[str, float], prefix: str, motor_count: int = 7) -> np.ndarray:
    """Extract joints + gripper for one arm from action dict.

    Supports lerobot naming: '{prefix}joint{i}.pos' and '{prefix}gripper.pos'.
    """
    result = np.zeros(motor_count, dtype=np.float64)
    for i in range(1, motor_count):
        key = f"{prefix}joint{i}.pos"
        if key in action_dict:
            result[i - 1] = action_dict[key]
    gripper_key = f"{prefix}gripper.pos"
    if gripper_key in action_dict:
        result[motor_count - 1] = action_dict[gripper_key]
    return result


def _get_arm_prefixes(action_features: dict[str, type]) -> tuple[list[str], int]:
    """Detect arm prefixes from action feature keys.

    E.g., ['left_', 'right_'] for bimanual, [''] for single arm.
    """
    prefixes = set()
    for key in action_features:
        if key.endswith(".pos"):
            parts = key.split("joint")
            if len(parts) > 1 and "_" in parts[0]:
                prefixes.add(parts[0])  # e.g., "left_", "right_"
            elif "_" in key and "gripper" in key:
                prefixes.add(key.split("gripper")[0])
    if not prefixes:
        return [""], 7
    return sorted(prefixes), 7


def interpolate_leader_to_follower(
    teleop,
    leader_pos: dict[str, float],
    follower_pos: dict[str, float],
    action_features: dict[str, type],
    dt: float = 0.05,
    max_step: float = 0.02,
) -> None:
    """Cosine-interpolate leader arms to match follower positions.

    Steps are computed from the largest single-joint displacement so that
    large-range moves are slower and small corrections are fast.
    Only the 6 arm joints are interpolated; the leader gripper (teach
    pendant) is left untouched to avoid disturbing the follower gripper.

    Auto-detects arm prefixes from action_features so it works for
    bimanual (left_/right_) and single-arm (no prefix) robots.

    Args:
        teleop: Teleoperator instance with joint_control_mit method.
        leader_pos: Current leader position dict.
        follower_pos: Current follower position dict.
        action_features: Action feature dict from robot.action_features.
        dt: Time step between interpolation steps (seconds).
        max_step: Maximum joint displacement per step (radians).
    """
    prefixes, motor_count = _get_arm_prefixes(action_features)

    starts = {}
    ends = {}
    for prefix in prefixes:
        starts[prefix] = _build_joint_array(leader_pos, prefix, motor_count)
        ends[prefix] = _build_joint_array(follower_pos, prefix, motor_count)

    max_disp = max(abs(ends[p][:6] - starts[p][:6]).max() for p in prefixes)
    steps = max(1, int(np.ceil(max_disp / max_step)))

    logger.info(
        f"[Align] Interpolating {len(prefixes)} arm(s), max disp={max_disp:.3f} rad, steps={steps}, ~{steps * dt:.1f}s"
    )
    for i in range(steps):
        t_val = i / (steps - 1) if steps > 1 else 1.0
        t_smoothed = (1 - np.cos(t_val * np.pi)) / 2

        for prefix in prefixes:
            joint_interp = starts[prefix][:6] * (1 - t_smoothed) + ends[prefix][:6] * t_smoothed
            joint_vals = joint_interp.tolist()

            arm_name = prefix.rstrip("_") if prefix else "arm"
            try:
                if hasattr(teleop, f"{arm_name}_arm"):
                    arm = getattr(teleop, f"{arm_name}_arm")
                elif hasattr(teleop, arm_name):
                    arm = getattr(teleop, arm_name)
                else:
                    continue
                arm.joint_control_mit(joint_vals)
            except Exception:
                pass

        time.sleep(dt)
    logger.info("[Align] Done.")


def reset_to_zero(
    robot,
    teleop,
    action_features: dict[str, type],
    dt: float = 0.05,
    max_step: float = 0.02,
) -> None:
    """Cosine-interpolate follower arms to zero position, then align leader.

    Steps are computed from the largest single-joint displacement.
    Used when inference fails and the arms need to return to a safe home position.

    Args:
        robot: Robot instance.
        teleop: Teleoperator instance (can be None).
        action_features: Action feature dict from robot.action_features.
        dt: Time step between interpolation steps (seconds).
        max_step: Maximum joint displacement per step (radians).
    """
    keys = list(action_features.keys())
    obs = robot.get_observation()
    start_pos = np.array([obs.get(k, 0.0) for k in keys], dtype=np.float64)
    zero_pos = np.zeros(len(keys), dtype=np.float64)

    max_disp = abs(start_pos).max()
    steps = max(1, int(np.ceil(max_disp / max_step)))

    logger.info(f"[Reset] Moving follower to zero, max disp={max_disp:.3f}, steps={steps}")
    for i in range(steps):
        t_val = i / (steps - 1) if steps > 1 else 1.0
        t_smoothed = (1 - np.cos(t_val * np.pi)) / 2
        interp = start_pos * (1 - t_smoothed) + zero_pos * t_smoothed
        action = {key: float(interp[j]) for j, key in enumerate(keys)}
        robot.send_action(action)
        time.sleep(dt)

    logger.info("[Reset] Follower at zero.")

    if teleop is not None:
        logger.info("[Reset] Aligning leader to zero...")
        leader_pos = teleop.get_action()
        follower_pos = robot.get_observation()
        interpolate_leader_to_follower(teleop, leader_pos, follower_pos, action_features, dt=dt, max_step=max_step)
        logger.info("[Reset] Leader aligned. Done.")

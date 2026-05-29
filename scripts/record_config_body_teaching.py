"""Configuration dataclass for record_body_teaching.py.

Uses draccus for CLI parsing with nested config support.
No separate teleoperator — the robot provides get_action() for body teaching.

Example:
    python record_body_teaching.py \
        --robot.type=bi_arx_x5 \
        --robot.left_can_port=can0 --robot.right_can_port=can1 \
        --robot.mode=collect \
        --robot.cameras='{"top":{"type":"intelrealsense","width":848,"height":480,"fps":30,"serial_number_or_name":"135"}}' \
        --policy.type=openpi \
        --task="fold the box"
"""

from dataclasses import dataclass, field

from robodeploy.robots.arx_x5 import arx_x5, bi_arx_x5  # noqa: F401
from robodeploy.policy_clients import (  # noqa: F401
    lingbot,
    openpi,
)
from robodeploy.policy_clients import PolicyClientConfig
from robodeploy.robots import RobotConfig


@dataclass
class RecordBodyTeachingConfig:
    """Configuration for body-teaching data collection + inference script."""

    # Robot (draccus ChoiceRegistry, use --robot.type=arx_x5 or bi_arx_x5)
    robot: RobotConfig | None = None

    # Policy client (draccus ChoiceRegistry, use --policy.type=...)
    policy: PolicyClientConfig | None = None

    # Output settings
    output_dir: str = "./s1_data"
    repo_id: str = "dataset"
    task: str = "fold the box"
    fps: int = 30
    episode_time_s: float = 120.0

    # Temporal smoothing
    use_temporal_smoothing: bool = True
    inference_rate: float = 3.0
    latency_k: int = 8
    min_smooth_steps: int = 8

    # Alignment
    align_max_step: float = 0.02

    # Control mode
    control_mode: str = "mixed"
    control_mode_initial: str = "collect"

    # WebUI
    webui_port: int = 8080

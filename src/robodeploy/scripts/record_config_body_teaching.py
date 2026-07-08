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

from dataclasses import dataclass

from robodeploy.policy_clients import (  # noqa: F401
    PolicyClientConfig,
    lingbot,
    openpi,
)
from robodeploy.robots import RobotConfig
from robodeploy.robots.arx_x5 import arx_x5, bi_arx_x5  # noqa: F401
from robodeploy.robots.lerobot_robot_my_arm import bi_innov_arm_v1, innov_arm_v1  # noqa: F401


@dataclass
class RecordBodyTeachingConfig:
    """Configuration for body-teaching data collection + inference script."""

    # Robot (draccus ChoiceRegistry: arx_x5, bi_arx_x5, innov_arm_v1, bi_innov_arm_v1)
    robot: RobotConfig | None = None

    # Policy client (draccus ChoiceRegistry, use --policy.type=...)
    policy: PolicyClientConfig | None = None

    # Output settings
    output_dir: str = "./s1_data"
    repo_id: str = "dataset"
    task: str = "fold the box"
    fps: int = 30
    episode_time_s: float = 120.0

    # Temporal smoothing (min_smooth_steps is used only when use_temporal_smoothing=True)
    use_temporal_smoothing: bool = True
    min_smooth_steps: int = 8

    # RTC (Real-Time Chunking) — 替代 StreamBuffer，收发驱动
    use_rtc: bool = False
    rtc_execution_horizon: int = 13  # 服务端约束窗口 + 客户端 blend overlap
    warmup_rounds: int = 10  # 推理预热轮数，0 跳过
    action_smooth_max_step: float = 0.05  # 推理动作单步最大变化(rad)，0 关闭

    # Alignment
    align_max_step: float = 0.02

    # Control mode
    control_mode: str = "mixed"
    control_mode_initial: str = "collect"

    # WebUI
    webui_port: int = 8080

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
"""Configuration dataclass for record_dataset.py.

Uses draccus for CLI parsing with nested config support.
Example:
    python record_dataset.py \
        --robot.type=s1_follower \
        --robot.port=/dev/ttyUSB0 \
        --teleop.type=s1_leader \
        --teleop.port=/dev/ttyUSB1 \
        --policy.type=openpi \
        --policy.host=localhost \
        --policy.port=8000 \
        --task="fold the box"
"""

from dataclasses import dataclass

from robodeploy.policy_clients import (  # noqa: F401
    PolicyClientConfig,
    lingbot,
    openpi,
)

# Import config modules to trigger draccus ChoiceRegistry registration
from robodeploy.robots import (  # noqa: F401
    RobotConfig,
    bi_s1_follower,
    s1_follower,
)
from robodeploy.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    bi_s1_leader,
    s1_leader,
)


@dataclass
class RecordConfig:
    """Configuration for unified data collection + inference script."""

    # Robot (draccus ChoiceRegistry, use --robot.type=...)
    robot: RobotConfig | None = None

    # Teleop (draccus ChoiceRegistry, use --teleop.type=...)
    teleop: TeleoperatorConfig | None = None

    # Policy client (draccus ChoiceRegistry, use --policy.type=...)
    policy: PolicyClientConfig | None = None

    # Output settings
    output_dir: str = "auto"  # "auto" → outputs/<robot_name>/<MMDD>_<HHMM>
    repo_id: str = "dataset"
    task: str = "fold the box"
    fps: int = 30
    episode_time_s: float = 120.0

    # Temporal smoothing (ignored when use_rtc=True)
    use_temporal_smoothing: bool = True
    inference_rate: float = 3.0
    latency_k: int = 8
    min_smooth_steps: int = 8

    # RTC (Real-Time Chunking) — replaces temporal smoothing when enabled
    use_rtc: bool = False
    rtc_execution_horizon: int = 13  # guidance constraint window + client blend overlap

    # Warmup
    warmup_rounds: int = 10  # 推理预热轮数，0 跳过

    # Alignment
    align_max_step: float = 0.02

    # Action smoothing（推理动作插值平滑，0 关闭）
    action_smooth_max_step: float = 0.05

    # Control mode
    control_mode: str = "mixed"
    control_mode_initial: str = "teleop"

    # WebUI
    webui_port: int = 8080

    # 前端：web（浏览器）| qt（桌面内嵌）| none（仅键盘）
    front_end: str = "web"

    # 部署模式：Qt 前端隐藏录制/保存按钮（control_mode=policy 纯推理）
    deploy_mode: bool = False

    # URDF 随动视图（front_end=qt 时生效）
    urdf_path: str = "/home/kemove/INNOV/infra/robot_SDK/robot-arm-4340/urdf/urdf/urdf.urdf"
    urdf_joint_indices: str = "7,8,9,10,11,12"  # state 向量中映射到 URDF qpos 的维度
    urdf_joint_scale: float = 1.0  # 关节角 → 弧度换算系数

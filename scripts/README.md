# 部署与测试工具

`scripts/` 目录保留部署/测试相关的独立工具脚本。数据采集与数据集处理脚本已迁移至 `src/robodeploy/scripts/`。

| 文件 | 用途 |
|------|------|
| `replay_actions.py` | 从 LeRobot 数据集 parquet 回放 action 到机器人 |
| `inspect_policy_action.py` | 策略 action 裸推理检查（无平滑、无录制），输出 JSONL 日志 |
| `test_webui.py` | WebUI 离线测试（重放已录制的数据集） |
| `test_joint_direction.py` | 关节方向测试（验证 inverted_axes 配置） |

## 数据采集与数据集处理

数据采集核心脚本和数据集处理工具已迁移至 Python 包内：

```
src/robodeploy/scripts/
├── record_dataset.py            # 统一数据采集（遥操作 + 策略推理）
├── record_body_teaching.py      # 本体示教采集
├── record_config.py             # 采集配置 dataclass
├── record_config_body_teaching.py  # 本体示教配置 dataclass
├── binarize_gripper.py          # 夹爪二值化
├── filter_valid_episodes.py     # 过滤有效 episodes
├── filter_lerobot_dataset.py    # LeRobot 数据集过滤
├── merge_lerobot_datasets.py    # 数据集合并
├── space_mirroring.py           # 视频镜像
├── stack_front_cameras.py       # 前摄像头堆叠
├── data_augment.py              # 数据增强
├── regenerate_stats.py          # 重新生成统计量
├── reassign_tasks.py            # 重新分配任务标签
└── split_by_position.py         # 按位置拆分数据集
```

## 机器人专属采集入口

各机器人的 shell 启动脚本集中在 `examples/` 目录：

```
examples/
├── s1/
│   ├── record_dataset.sh        # S1 遥操作 + 策略推理
│   └── inspect_policy_action.sh # S1 策略 action 检查
├── arx/
│   └── record_arx_bimanual.sh   # ARX X5 双臂本体示教
└── innov/
    ├── record_innov.sh          # Innov Arm 双臂本体示教
    ├── set_motor_zero.py        # Innov Arm 电机归零
    └── README.md
```

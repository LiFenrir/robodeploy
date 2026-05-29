# 数据采集脚本

统一数据采集 + 策略推理工具，基于 LeRobot 抽象接口 (Robot/Teleoperator)，支持任意已注册的机器人平台。

**入口：**

| 入口 | 用途 |
|------|------|
| `python record_s1_inference_npy.py` | S1/SO100 遥操作 + 策略推理（NPY 存储，O(1) 内存） |
| `python record_body_teaching.py` | 本体示教（ARX X5 等，无独立遥操作器） |

## 功能

1. **遥操作录制**：后台子进程异步编码视频 + 写入 parquet（非阻塞）
2. **策略推理**：OpenPI WebSocket 客户端 + StreamActionBuffer 时序平滑
3. **成功/失败标注**：每个 episode 结束时键盘标注，存储为 `is_success` 字段
4. **三模式切换**：teleop（纯遥操作）/ policy（纯推理）/ mixed（P 键实时切换），逐帧 `is_inference` 标记
5. **DAgger 对齐**：策略切换到遥控时，余弦插值对齐 leader→follower
6. **归零复位**：Z 键将机械臂平滑归零并重新对齐主端（不录制时可用）

## 架构

```
record_s1_inference_npy.py
        │
键盘监听 ──(P/R/S/Z/Esc)──→ 主循环 (30fps)
                                │
         ┌──────────────────────┼──────────────────────┐
         ▼                      ▼                      ▼
    robot.get_obs()    stream_buffer.pop()    robot.send_action()
         │                      ▲                      │
         ▼                      │                      ▼
    state_ref["obs"]   integrate_new_chunk    dataset.add_frame()
         │                      ▲                      │
         ▼                      │                      ▼
    推理线程 ──→ policy.infer() ──┘     LeRobotDatasetNPY (逐帧 NPY)
                                                   │
                                     ┌─────────────┼─────────────┐
                                     ▼             ▼             ▼
                              BackgroundVideoEncoder  parquet    metadata
                               (NPY → MP4 后台编码)
```

## Python CLI 参数（draccus 风格）

### Robot（机器人硬件）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--robot.type` | str | `bi_s1_follower` | 机器人类型：`s1_follower`, `bi_s1_follower`, `so100_follower` 等 |
| `--robot.port` | str | — | 单臂串口 |
| `--robot.left_arm_port` | str | — | 双臂左手串口 |
| `--robot.right_arm_port` | str | — | 双臂右手串口 |
| `--robot.cameras` | str (JSON) | `{}` | 摄像头配置 JSON |

### Teleop（遥操作主端）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--teleop.type` | str | `bi_s1_leader` | 遥操作设备类型，纯推理模式可不设 |
| `--teleop.port` | str | — | 单臂 leader 串口 |
| `--teleop.left_arm_port` | str | — | 双臂左手 leader 串口 |
| `--teleop.right_arm_port` | str | — | 双臂右手 leader 串口 |

### OpenPI（策略推理）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--policy.type` | str | `openpi` | 策略客户端类型 |
| `--policy.host` | str | `localhost` | 推理服务器 IP |
| `--policy.port` | int | `8000` | 推理服务器端口 |

### Output（数据输出）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--output_dir` | str | `./s1_data` | 输出目录（LeRobot 格式） |
| `--repo_id` | str | `dataset` | 数据集标识符 |
| `--task` | str | `fold the box` | 任务描述（存入 meta/tasks.jsonl） |
| `--fps` | int | `30` | 控制/录像帧率 |
| `--episode_time_s` | float | `120.0` | 每个 episode 最大时长（秒），超时自动保存 |

### Temporal Smoothing（时序平滑）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--use_temporal_smoothing` | flag | True | 启用时序平滑（默认开） |
| `--no_temporal_smoothing` | flag | — | 关闭时序平滑 |
| `--inference_rate` | float | `3.0` | 推理频率（Hz），策略模型请求间隔 |
| `--latency_k` | int | `8` | 新动作块到达时丢弃旧块的前 k 步 |
| `--min_smooth_steps` | int | `8` | 重叠区最小长度，保证平滑过渡 |

### Control（控制模式）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--control_mode` | str | `mixed` | 启动模式：`teleop`、`policy`、`mixed` |
| `--control_mode_initial` | str | `teleop` | mixed 模式下的初始控制方式 |

## 用法示例

### 双臂 S1 + mixed 模式（可 P 键切换）

```bash
python record_s1_inference_npy.py \
    --robot.type=bi_s1_follower \
    --robot.left_arm_port=/dev/ttyUSB0 --robot.right_arm_port=/dev/ttyUSB1 \
    --robot.cameras='{"front":{"type":"intelrealsense","width":848,"height":480,"fps":30}}' \
    --teleop.type=bi_s1_leader \
    --teleop.left_arm_port=/dev/ttyUSB2 --teleop.right_arm_port=/dev/ttyUSB3 \
    --policy.type=openpi --policy.host=localhost --policy.port=8000 \
    --task="fold the box" \
    --control_mode mixed --control_mode_initial teleop
```

### 纯遥操作（无需 OpenPI）

```bash
python record_s1_inference_npy.py \
    --robot.type=bi_s1_follower \
    --robot.left_arm_port=/dev/ttyUSB0 --robot.right_arm_port=/dev/ttyUSB1 \
    --robot.cameras='{"front":{...}}' \
    --teleop.type=bi_s1_leader \
    --teleop.left_arm_port=/dev/ttyUSB2 --teleop.right_arm_port=/dev/ttyUSB3 \
    --task="fold the box" \
    --control_mode teleop
```

### 纯策略推理（无需遥操臂）

```bash
python record_s1_inference_npy.py \
    --robot.type=bi_s1_follower \
    --robot.left_arm_port=/dev/ttyUSB0 --robot.right_arm_port=/dev/ttyUSB1 \
    --robot.cameras='{"front":{...}}' \
    --policy.type=openpi --policy.host=192.168.1.100 --policy.port=8000 \
    --task="pick and place" \
    --control_mode policy --episode_time_s 60
```

### 单臂 S1

```bash
python record_s1_inference_npy.py \
    --robot.type=s1_follower --robot.port=/dev/ttyUSB0 \
    --teleop.type=s1_leader --teleop.port=/dev/ttyUSB2 \
    --robot.cameras='{"front":{...}}' \
    --task="pick and place"
```

### 本体示教（ARX X5）

```bash
python record_body_teaching.py \
    --robot.type=bi_arx_x5 \
    --robot.left_can_port=can0 --robot.right_can_port=can1 \
    --robot.mode=collect \
    --robot.cameras='{"top":{"type":"intelrealsense","width":848,"height":480,"fps":30}}' \
    --task="fold the box"
```

## 键盘操作

| 按键 | 适用模式 | 功能 |
|------|----------|------|
| `P` / `Tab` | mixed | 切换 Teleop ↔ Policy 模式（切换时自动对齐 leader↔follower） |
| `R` | 全部 | 开始/停止录像 |
| `S` | 全部 | 保存当前 episode + 弹出成功/失败标注 |
| `Z` | 全部 | 机械臂归零复位并重新对齐主端（录制中不可用） |
| `Esc` | 全部 | 退出程序 |

保存时标注对话框：

- `1` → 成功 (is_success=1)
- `0` → 失败 (is_success=0)
- `2` → 丢弃该 episode

## 输出目录结构

```
output_dir/
├── meta/
│   ├── info.json          # 数据集元信息（fps, features, 统计等）
│   ├── episodes.jsonl     # 逐 episode 摘要（index, task, length）
│   └── tasks.jsonl        # 任务索引映射
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
└── videos/
    └── chunk-000/
        ├── front/
        │   ├── episode_000000.mp4
        │   └── ...
        ├── left_wrist/
        │   └── ...
        └── right_wrist/
            └── ...
```

## 依赖

- `lerobot`（Robot/Teleoperator 抽象接口）
- `openpi_client`（WebSocket 策略客户端 + 图像预处理）
- `av`（PyAV，视频编码）
- `pyarrow`（parquet 写入）
- `numpy`, `opencv-python`
- `flask`（WebUI 服务）

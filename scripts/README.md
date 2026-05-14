# record_s1_inference

统一数据采集 + 策略推理工具，基于 LeRobot 抽象接口 (Robot/Teleoperator)，支持任意已注册的机器人平台。

**两个入口：**

| 入口 | 用途 |
|------|------|
| `bash record_s1_inference.sh` | Shell 封装脚本，快速配置 + 交互式菜单选择模式（推荐） |
| `python record_s1_inference.py` | Python 脚本直接调用，适合自动化/脚本化 |

## 功能

1. **遥操作录制**：后台子进程异步编码视频 + 写入 parquet（非阻塞）
2. **策略推理**：OpenPI WebSocket 客户端 + StreamActionBuffer 时序平滑
3. **成功/失败标注**：每个 episode 结束时键盘标注，存储为 `is_success` 字段
4. **三模式切换**：teleop（纯遥操作）/ policy（纯推理）/ mixed（P 键实时切换），逐帧 `is_inference` 标记
5. **DAgger 对齐**：策略切换到遥控时，余弦插值对齐 leader→follower
6. **归零复位**：Z 键将机械臂平滑归零并重新对齐主端（不录制时可用）

## 架构

```
record_s1_inference.sh  (可选：选择模式 + 打印配置)
        │
        ▼
record_s1_inference.py
        │
键盘监听 ──(P/R/S/Z/Esc)──→ 主循环 (30fps)
                                │
         ┌──────────────────────┼──────────────────────┐
         ▼                      ▼                      ▼
    robot.get_obs()    stream_buffer.pop()    robot.send_action()
         │                      ▲                      │
         ▼                      │                      ▼
    state_ref["obs"]   integrate_new_chunk    episode_buffer.add()
         │                      ▲                      │
         ▼                      │                      ▼
    推理线程 ──→ policy.infer() ──┘        BackgroundLeRobotWriter
                                            (mp.Queue → 子进程)
                                                   │
                                     ┌─────────────┼─────────────┐
                                     ▼             ▼             ▼
                                _encode_video  write_table   jsonl
```

## Shell 脚本快速配置 (record_s1_inference.sh)

Shell 脚本顶部提供了快速配置区，修改变量即可调整运行参数。启动时弹出交互菜单选择控制模式：

```
==============================================
       选择控制模式
==============================================

  1) teleop  — 纯遥操作（无需推理服务，不连 OpenPI）
  2) policy  — 纯策略推理（无需遥操臂，不连 leader）
  3) mixed   — 混合模式（遥操作 + 策略推理，P 键切换）
```

**Shell 脚本特有配置变量：**

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `CONTROL_MODE` | 运行时由菜单选择，留空即可 | `""` |
| `CONTROL_MODE_INITIAL` | mixed 模式下的初始控制方式 | `teleop` |
| `LEFT_FOLLOWER_PORT` | 双臂左手 follower 串口 | `/dev/left_follower` |
| `RIGHT_FOLLOWER_PORT` | 双臂右手 follower 串口 | `/dev/right_follower` |
| `LEFT_LEADER_PORT` | 双臂左手 leader 串口 | `/dev/left_leader` |
| `RIGHT_LEADER_PORT` | 双臂右手 leader 串口 | `/dev/right_leader` |

## Python CLI 参数

### Robot（机器人硬件）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--robot_type` | str | `bi_s1_follower` | 机器人类型：`s1_follower`, `bi_s1_follower`, `so100_follower`, `bi_so100_follower` 等 |
| `--follower_port` | str | `/dev/ttyUSB0` | 单臂 follower 串口 |
| `--left_follower_port` | str | `/dev/ttyUSB0` | 双臂左手 follower 串口 |
| `--right_follower_port` | str | `/dev/ttyUSB1` | 双臂右手 follower 串口 |
| `--follower_version` | str | `V2` | 臂硬件版本 |

### Teleop（遥操作主端）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--teleop_type` | str | `bi_s1_leader` | 遥操作设备类型，设为空字符串则仅推理不遥控 |
| `--leader_port` | str | `/dev/ttyUSB2` | 单臂 leader 串口 |
| `--left_leader_port` | str | `/dev/ttyUSB2` | 双臂左手 leader 串口 |
| `--right_leader_port` | str | `/dev/ttyUSB3` | 双臂右手 leader 串口 |
| `--leader_version` | str | `V2` | 主端硬件版本 |

### Cameras（摄像头）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--cameras` | str (JSON) | `{}` | 摄像头配置 JSON |

摄像头 JSON 格式：

```json
{
  "front": {
    "type": "intelrealsense",
    "serial_number_or_name": "243522075502",
    "width": 848, "height": 480, "fps": 30
  },
  "left_wrist": {
    "type": "intelrealsense",
    "serial_number_or_name": "239222071739",
    "width": 848, "height": 480, "fps": 30
  },
  "right_wrist": {
    "type": "intelrealsense",
    "serial_number_or_name": "239222071740",
    "width": 848, "height": 480, "fps": 30
  }
}
```

### OpenPI（策略推理）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--openpi_host` | str | `localhost` | 推理服务器 IP |
| `--openpi_port` | int | `8000` | 推理服务器端口 |

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
| `--control_mode` | str | `mixed` | 启动模式：`teleop`（纯遥控）、`policy`（纯推理）、`mixed`（可切换） |
| `--control_mode_initial` | str | `teleop` | mixed 模式下的初始控制方式：`teleop` 或 `policy` |

## 用法示例

### 推荐：Shell 脚本快速启动

```bash
# 1. 编辑 record_s1_inference.sh 顶部配置区（机器人类型、串口、摄像头等）
# 2. 运行脚本，根据交互菜单选择模式
bash record_s1_inference.sh
```

运行流程：
1. 打印配置摘要并弹出模式选择菜单（1=teleop / 2=policy / 3=mixed）
2. 确认配置后按 Enter 启动
3. 使用键盘控制录制、保存、模式切换等

### Python 脚本直接调用

适用于自动化脚本、远程启动等场景。

#### 双臂 S1 + mixed 模式（可 P 键切换）

```bash
python record_s1_inference.py \
    --robot_type bi_s1_follower \
    --left_follower_port /dev/ttyUSB0 \
    --right_follower_port /dev/ttyUSB1 \
    --teleop_type bi_s1_leader \
    --left_leader_port /dev/ttyUSB2 \
    --right_leader_port /dev/ttyUSB3 \
    --cameras '{"front":{...},"left_wrist":{...},"right_wrist":{...}}' \
    --task "fold the box" \
    --control_mode mixed \
    --control_mode_initial teleop \
    --openpi_host localhost --openpi_port 8000 \
    --output_dir ./s1_data \
    --episode_time_s 120
```

#### 纯遥操作（无需 OpenPI）

```bash
python record_s1_inference.py \
    --robot_type bi_s1_follower \
    --left_follower_port /dev/ttyUSB0 \
    --right_follower_port /dev/ttyUSB1 \
    --teleop_type bi_s1_leader \
    --left_leader_port /dev/ttyUSB2 \
    --right_leader_port /dev/ttyUSB3 \
    --cameras '{"front":{...}}' \
    --task "fold the box" \
    --control_mode teleop
```

#### 纯策略推理（无需遥操臂）

```bash
python record_s1_inference.py \
    --robot_type bi_s1_follower \
    --left_follower_port /dev/ttyUSB0 \
    --right_follower_port /dev/ttyUSB1 \
    --cameras '{"front":{...}}' \
    --task "pick and place" \
    --control_mode policy \
    --openpi_host 192.168.1.100 --openpi_port 8000 \
    --episode_time_s 60
```

#### 单臂 S1

```bash
python record_s1_inference.py \
    --robot_type s1_follower \
    --follower_port /dev/ttyUSB0 \
    --teleop_type s1_leader \
    --leader_port /dev/ttyUSB2 \
    --cameras '{"front":{...}}' \
    --task "pick and place"
```

#### SO100 双臂

```bash
python record_s1_inference.py \
    --robot_type bi_so100_follower \
    --left_follower_port /dev/ttyUSB0 \
    --right_follower_port /dev/ttyUSB1 \
    --teleop_type bi_so100_leader \
    --left_leader_port /dev/ttyUSB2 \
    --right_leader_port /dev/ttyUSB3 \
    --cameras '{"front":{...}}' \
    --task "sort the blocks"
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
- `pynput`（键盘监听）
- `numpy`, `opencv-python`

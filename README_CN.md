# robodeploy 中文说明

> 机器人部署与数据采集工具包，基于 LeRobot 改造，适用于真实机器人硬件控制、遥操作录制和策略推理。

## 功能介绍

- **多机器人支持** — S1、SO100、SO101、Koch、Aloha、LeKiwi、HopeJr、Stretch3、ViperX
- **双臂遥操作** — 支持单臂 / 双臂 leader-follower 遥操作
- **多模式数据采集** — 纯遥操作、纯策略推理、混合模式（P 键热切换）
- **摄像头集成** — OpenCV USB 摄像头、Intel RealSense 深度摄像头
- **策略推理** — OpenPI WebSocket 客户端，支持时序平滑与 DAgger 对齐
- **Web 管理界面** — Flask 实时监控与控制面板
- **ROS 部署** — Piper 机器人 ROS noetic 部署脚本

## 项目结构

```
robodeploy/
├── src/robodeploy/         # 核心包：cameras, datasets, motors, robots, teleoperators, webui
├── scripts/                   # 数据采集与推理脚本
├── deploy/                    # 部署工具：piper_deploy.py, ROS launch, data_collection
├── tests/                     # pytest 测试套件
├── examples/                  # 使用示例
└── pyproject.toml             # 项目元数据与依赖配置
```

## 快速开始

### 安装

```bash
pip install -e ".[dev,test]"           # 基础 + 开发/测试
pip install -e ".[feetech]"            # Feetech 舵机驱动
pip install -e ".[dynamixel]"          # Dynamixel 舵机驱动
pip install -e ".[intelrealsense]"     # Intel RealSense 摄像头驱动
pip install -e ".[all]"                # 全部
```

### 运行测试

```bash
pytest                                 # 硬件相关测试会自动跳过
```

### 代码检查与格式化

```bash
ruff check src/ tests/                 # 代码检查
ruff format src/ tests/                # 代码格式化
```

### 启动 Web 界面

```bash
python -m robodeploy.webui           # 默认地址 http://localhost:5000
```

### 数据采集

```bash
# 双臂 S1 — 纯遥操作
python scripts/record_s1_inference.py \
    --robot_type bi_s1_follower \
    --follower_port /dev/ttyUSB0 \
    --leader_port /dev/ttyUSB1 \
    --control_mode teleop

# 混合模式 — 策略推理 + 遥操作（P 键切换）
python scripts/record_hybrid.py \
    --robot.type=bi_s1_follower \
    --teleop.type=bi_s1_leader \
    --policy.type=openpi \
    --policy.host=localhost --policy.port=8000 \
    --task="pick and place"
```

详情见 [scripts/README.md](scripts/README.md)。

## 技术栈

| 组件 | 方案 |
|------|------|
| 语言 | Python 3.10+ |
| 机器学习 | PyTorch, HuggingFace datasets & hub |
| 视觉 | OpenCV, Intel RealSense |
| 配置 | draccus（数据类驱动的 CLI 解析） |
| Web 界面 | Flask + FastAPI / uvicorn |
| 视频编码 | PyAV (av) |
| 串口通信 | pyserial, Dynamixel SDK, Feetech SDK |
| 策略推理 | OpenPI WebSocket 客户端, msgpack |

## 代码规范

- **双引号** (`"`) 用于所有 Python 字符串（ruff 强制）
- **行宽 110 字符**
- **Google 风格 docstring**
- **Apache 2.0 协议头** 在每个 `.py` 文件
- **`src` 布局** — 导入路径 `from robodeploy import ...`

## 注意事项

- 硬件测试会自动跳过 — `conftest.py` 检测到未连接物理电机/摄像头时跳过
- `torchcodec` 在 Windows / ARM Linux / macOS x86_64 上不可用，此为预期行为
- `[feetech]`、`[dynamixel]`、`[intelrealsense]` 是可选驱动，不在默认安装中
- 顶层 `scripts/` 不属于 Python 包，是独立的入口脚本

## 开源协议

Apache 2.0 — 基于 [LeRobot](https://github.com/huggingface/lerobot) 改造。

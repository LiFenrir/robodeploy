#!/bin/bash
#===============================================================================
# S1 数据采集 + 策略推理 启动脚本 (RTC 版本)
#
# RTC (Real-Time Chunking) 服务端时序平滑，替代 StreamBuffer 线性 crossfade。
# 新增推理预热阶段（warmup_rounds），降低首帧推理延迟。
#
# 用法:
#   bash record_dataset_rtc.sh
#
# 如果使用单臂 S1，需确保 S1 SDK 在 Python 路径中:
#   export PYTHONPATH=/path/to/YHRG/S1_SDK_V2/src:$PYTHONPATH
#===============================================================================

set -e

# ==============================================================================
# 快速配置区 — 修改以下变量即可
# ==============================================================================

# Robot（机器人硬件）
#------------------------------------------------------------------------------
ROBOT_TYPE="bi_s1_follower"                  # 单臂: s1_follower, so100_follower  双臂: bi_s1_follower, bi_so100_follower
# 双臂时取消下面两行注释
LEFT_FOLLOWER_PORT="/dev/ttyUSB0"
RIGHT_FOLLOWER_PORT="/dev/ttyUSB1"


# Teleop（遥操作主端）
#------------------------------------------------------------------------------
TELEOP_TYPE="bi_s1_leader"                   # 单臂: s1_leader, so100_leader  双臂: bi_s1_leader, bi_so100_leader  纯推理留空: ""
# 双臂时取消下面两行注释
LEFT_LEADER_PORT="/dev/ttyUSB3"
RIGHT_LEADER_PORT="/dev/ttyUSB2"

# Policy（策略客户端）
#------------------------------------------------------------------------------
POLICY_TYPE="openpi"                         # 策略类型，目前支持 openpi
OPENPI_HOST="192.168.1.17"
OPENPI_PORT=8000

# Cameras（摄像头 — 严格 JSON 格式）
#------------------------------------------------------------------------------
# 三摄像头示例:
CAMERA_CONFIG='{"front":{"type":"intelrealsense","serial_number_or_name":"135122077817","width":848,"height":480,"fps":30},"front_1":{"type":"intelrealsense","serial_number_or_name":"935422072733","width":848,"height":480,"fps":30},"left_wrist":{"type":"intelrealsense","serial_number_or_name":"409122273564","width":640,"height":480,"fps":30},"right_wrist":{"type":"intelrealsense","serial_number_or_name":"409122273228","width":640,"height":480,"fps":30}}'

# Output（数据输出）
#------------------------------------------------------------------------------
OUTPUT_DIR="./outputs"
REPO_ID="bi_s1/bi_s1_$(date +%m%d_%H%M)"
TASK="Grasp a single layer of the cloth with the gripper, then place the cloth onto the board"
FPS=30
EPISODE_TIME_S=6000                         # 每个 episode 最大时长（秒），超时自动触发保存

# Control（控制模式）— 由终端交互菜单选择，无需手动修改
#------------------------------------------------------------------------------
CONTROL_MODE=""                              # 运行时选择: teleop | policy | mixed
CONTROL_MODE_INITIAL="teleop"                # mixed 模式下的初始控制方式

# RTC（Real-Time Chunking — 服务端时序平滑）
#------------------------------------------------------------------------------
USE_RTC=true                                 # true 启用 RTC | false 回退 StreamBuffer
INFERENCE_RATE=3.0                           # 推理频率（Hz），控制线程轮询间隔
RTC_EXECUTION_HORIZON=15                     # 约束窗口步数，需 > max(infer_ms/33ms)
WARMUP_ROUNDS=10                             # 推理预热轮数，0 跳过

# Alignment（对齐参数）
#------------------------------------------------------------------------------
ALIGN_MAX_STEP=0.02                          # 归零/对齐时单步最大关节角度变化

# Action Smoothing（推理动作插值平滑，0 关闭）
#------------------------------------------------------------------------------
ACTION_SMOOTH_MAX_STEP=0.05                  # 推理动作单步最大变化(rad)，超过则插值

# WebUI（实时监控面板）
#------------------------------------------------------------------------------
WEBUI_PORT=8080

# ==============================================================================
# 脚本逻辑 — 一般无需修改
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
RECORD_SCRIPT="$PROJECT_ROOT/src/robodeploy/scripts/record_dataset.py"

# 检查 record_dataset.py 是否存在
if [ ! -f "$RECORD_SCRIPT" ]; then
    echo "错误: 找不到 $RECORD_SCRIPT"
    exit 1
fi

# 给所有串口赋予读写权限
SERIAL_PORTS=()
[ -n "${LEFT_FOLLOWER_PORT:-}" ] && SERIAL_PORTS+=("$LEFT_FOLLOWER_PORT")
[ -n "${RIGHT_FOLLOWER_PORT:-}" ] && SERIAL_PORTS+=("$RIGHT_FOLLOWER_PORT")
[ -n "${LEFT_LEADER_PORT:-}" ] && SERIAL_PORTS+=("$LEFT_LEADER_PORT")
[ -n "${RIGHT_LEADER_PORT:-}" ] && SERIAL_PORTS+=("$RIGHT_LEADER_PORT")
[ -n "${FOLLOWER_PORT:-}" ] && SERIAL_PORTS+=("$FOLLOWER_PORT")
[ -n "${LEADER_PORT:-}" ] && SERIAL_PORTS+=("$LEADER_PORT")

for port in "${SERIAL_PORTS[@]}"; do
    if [ -e "$port" ]; then
        sudo chmod 777 "$port"
        echo "  [OK] chmod 777 $port"
    else
        echo "  [WARN] $port 不存在，跳过"
    fi
done
echo ""

# 构建 robot 参数 (draccus 风格 --robot.xxx=yyy)
#------------------------------------------------------------------------------
ROBOT_ARGS=(
    --robot.type "$ROBOT_TYPE"
)

if [ "$ROBOT_TYPE" = "bi_s1_follower" ] || [ "$ROBOT_TYPE" = "bi_so100_follower" ]; then
    ROBOT_ARGS+=(--robot.left_arm_port "${LEFT_FOLLOWER_PORT:-/dev/left_follower}")
    ROBOT_ARGS+=(--robot.right_arm_port "${RIGHT_FOLLOWER_PORT:-/dev/right_follower}")
else
    ROBOT_ARGS+=(--robot.port "${FOLLOWER_PORT:-/dev/ttyUSB0}")
fi

if [ -n "$CAMERA_CONFIG" ]; then
    ROBOT_ARGS+=(--robot.cameras "$CAMERA_CONFIG")
fi

# 构建 teleop 参数 (draccus 风格 --teleop.xxx=yyy)
# pure policy 模式下跳过
#------------------------------------------------------------------------------
TELEOP_ARGS=()
if [ "$CONTROL_MODE" != "policy" ] && [ -n "$TELEOP_TYPE" ]; then
    TELEOP_ARGS+=(--teleop.type "$TELEOP_TYPE")

    if [[ "$TELEOP_TYPE" == bi_* ]]; then
        TELEOP_ARGS+=(--teleop.left_arm_port "${LEFT_LEADER_PORT:-/dev/left_leader}")
        TELEOP_ARGS+=(--teleop.right_arm_port "${RIGHT_LEADER_PORT:-/dev/right_leader}")
    else
        TELEOP_ARGS+=(--teleop.port "${LEADER_PORT:-/dev/ttyUSB1}")
    fi
fi

# 构建 policy 参数 (draccus 风格 --policy.xxx=yyy)
# pure teleop 模式下跳过
#------------------------------------------------------------------------------
POLICY_ARGS=()
if [ "$CONTROL_MODE" != "teleop" ]; then
    POLICY_ARGS+=(--policy.type "$POLICY_TYPE")
    POLICY_ARGS+=(--policy.host "$OPENPI_HOST")
    POLICY_ARGS+=(--policy.port "$OPENPI_PORT")
fi

# 构建 RTC / 推理参数 (draccus 风格)
# pure teleop 模式下跳过
#------------------------------------------------------------------------------
RTC_ARGS=()
if [ "$CONTROL_MODE" != "teleop" ]; then
    if [ "$USE_RTC" = true ]; then
        RTC_ARGS+=(--use_rtc true)
        RTC_ARGS+=(--rtc_execution_horizon "$RTC_EXECUTION_HORIZON")
    else
        RTC_ARGS+=(--use_rtc false)
    fi
    RTC_ARGS+=(--inference_rate "$INFERENCE_RATE")
    RTC_ARGS+=(--warmup_rounds "$WARMUP_ROUNDS")
    RTC_ARGS+=(--action_smooth_max_step "$ACTION_SMOOTH_MAX_STEP")
fi

# 打印配置
#------------------------------------------------------------------------------
print_config() {
    echo "=============================================="
    echo "    S1 数据采集 + 策略推理 (RTC)"
    echo "=============================================="
    echo "Robot:        $ROBOT_TYPE"
    if [ "$ROBOT_TYPE" = "bi_s1_follower" ] || [ "$ROBOT_TYPE" = "bi_so100_follower" ]; then
        echo "  Left:       ${LEFT_FOLLOWER_PORT:-/dev/left_follower}"
        echo "  Right:      ${RIGHT_FOLLOWER_PORT:-/dev/right_follower}"
    else
        echo "  Port:       ${FOLLOWER_PORT:-/dev/ttyUSB0}"
    fi
    echo ""
    if [ "$CONTROL_MODE" != "policy" ] && [ -n "$TELEOP_TYPE" ]; then
        echo "Teleop:       $TELEOP_TYPE"
        if [[ "$TELEOP_TYPE" == bi_* ]]; then
            echo "  Left:       ${LEFT_LEADER_PORT:-/dev/left_leader}"
            echo "  Right:      ${RIGHT_LEADER_PORT:-/dev/right_leader}"
        else
            echo "  Port:       ${LEADER_PORT:-/dev/ttyUSB1}"
        fi
    elif [ "$CONTROL_MODE" = "policy" ]; then
        echo "Teleop:       (无 — 纯策略推理模式)"
    else
        echo "Teleop:       (未配置)"
    fi
    echo ""
    if [ "$CONTROL_MODE" != "teleop" ]; then
        echo "Policy:       ${POLICY_TYPE} @ ${OPENPI_HOST}:${OPENPI_PORT}"
    fi
    echo "Cameras:      $CAMERA_CONFIG"
    echo ""
    echo "Output:       $OUTPUT_DIR"
    echo "Repo:         $REPO_ID"
    echo "Task:         $TASK"
    echo "FPS:          $FPS"
    echo "Episode:      ${EPISODE_TIME_S}s"
    echo ""
    echo "Control:      $CONTROL_MODE"
    if [ "$CONTROL_MODE" = "mixed" ]; then
        echo "  Initial:    $CONTROL_MODE_INITIAL"
    fi
    if [ "$CONTROL_MODE" != "teleop" ]; then
        echo "RTC:          $USE_RTC"
        if [ "$USE_RTC" = true ]; then
            echo "  Horizon:    ${RTC_EXECUTION_HORIZON}   Warmup: ${WARMUP_ROUNDS}"
        fi
        echo "  Inference:  ${INFERENCE_RATE}Hz"
        echo "  ActionSmooth: ${ACTION_SMOOTH_MAX_STEP} rad"
    fi
    echo "Align step:   $ALIGN_MAX_STEP"
    echo "WebUI:        http://localhost:${WEBUI_PORT}"
    echo "=============================================="
    echo ""
}

# 控制模式交互菜单
#------------------------------------------------------------------------------
select_control_mode() {
    echo ""
    echo "=============================================="
    echo "       选择控制模式"
    echo "=============================================="
    echo ""
    echo "  1) teleop  — 纯遥操作（无需推理服务，不连 OpenPI）"
    echo "  2) policy  — 纯策略推理（无需遥操臂，不连 leader）"
    echo "  3) mixed   — 混合模式（遥操作 + 策略推理，P 键切换）"
    echo ""
    while true; do
        read -p "请输入选项 [1-3]: " choice
        case "$choice" in
            1) CONTROL_MODE="teleop"; break;;
            2) CONTROL_MODE="policy"; break;;
            3) CONTROL_MODE="mixed"; break;;
            *) echo "  无效选项，请输入 1、2 或 3";;
        esac
    done
    echo ""
    echo "  => 已选择: $CONTROL_MODE"
    echo ""
}

# 主流程
#------------------------------------------------------------------------------
echo ""
select_control_mode

# 根据控制模式重新构建参数（teleop/policy/RTC 依赖 CONTROL_MODE）
if [ "$CONTROL_MODE" != "policy" ] && [ -n "$TELEOP_TYPE" ]; then
    TELEOP_ARGS=()
    if [[ "$TELEOP_TYPE" == bi_* ]]; then
        TELEOP_ARGS+=(--teleop.type "$TELEOP_TYPE")
        TELEOP_ARGS+=(--teleop.left_arm_port "${LEFT_LEADER_PORT:-/dev/left_leader}")
        TELEOP_ARGS+=(--teleop.right_arm_port "${RIGHT_LEADER_PORT:-/dev/right_leader}")
    else
        TELEOP_ARGS+=(--teleop.type "$TELEOP_TYPE")
        TELEOP_ARGS+=(--teleop.port "${LEADER_PORT:-/dev/ttyUSB1}")
    fi
fi

if [ "$CONTROL_MODE" != "teleop" ]; then
    POLICY_ARGS=(--policy.type "$POLICY_TYPE" --policy.host "$OPENPI_HOST" --policy.port "$OPENPI_PORT")
    if [ "$USE_RTC" = true ]; then
        RTC_ARGS=(--use_rtc true --rtc_execution_horizon "$RTC_EXECUTION_HORIZON")
    else
        RTC_ARGS=(--use_rtc false)
    fi
    RTC_ARGS+=(--inference_rate "$INFERENCE_RATE" --warmup_rounds "$WARMUP_ROUNDS" --action_smooth_max_step "$ACTION_SMOOTH_MAX_STEP")
else
    POLICY_ARGS=()
    RTC_ARGS=()
fi

print_config

read -p "确认配置无误后按 Enter 键开始 (Ctrl+C 取消)..."

echo ""
if [ "$CONTROL_MODE" = "mixed" ]; then
    echo "提示: P=切换模式  R=录像  S=保存+标注  Z=归零复位  Esc=退出"
else
    echo "提示: R=录像  S=保存+标注  Esc=退出"
fi
echo ""

python "$RECORD_SCRIPT" \
    "${ROBOT_ARGS[@]}" \
    "${TELEOP_ARGS[@]}" \
    "${POLICY_ARGS[@]}" \
    --output_dir "$OUTPUT_DIR" \
    --repo_id "$REPO_ID" \
    --task "$TASK" \
    --fps "$FPS" \
    --episode_time_s "$EPISODE_TIME_S" \
    --control_mode "$CONTROL_MODE" \
    --control_mode_initial "$CONTROL_MODE_INITIAL" \
    --align_max_step "$ALIGN_MAX_STEP" \
    --webui_port "$WEBUI_PORT" \
    "${RTC_ARGS[@]}"

echo ""
echo "数据采集完成!"

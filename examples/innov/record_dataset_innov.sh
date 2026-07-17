#!/bin/bash
#===============================================================================
# InnovArmV1 数据采集 + 策略推理 启动脚本
#
# 使用 bi_s1_leader 遥操作双臂控制 bi_innov_arm_v1。
#
# 用法:
#   bash record_dataset_innov.sh
#
# 需确保 innov_arm_v1 驱动在 Python 路径中:
#   export PYTHONPATH=/path/to/lerobot_robot_my_arm:$PYTHONPATH
#===============================================================================

set -e

# ==============================================================================
# 快速配置区 — 修改以下变量即可
# ==============================================================================

# Robot（机器人硬件）
#------------------------------------------------------------------------------
ROBOT_TYPE="bi_innov_arm_v1"
LEFT_FOLLOWER_PORT="/dev/ttyACM0"
RIGHT_FOLLOWER_PORT="/dev/ttyACM1"
ROBOT_MODE="control"                          # collect=重力补偿示教  control=位置控制(策略推理时用)


# Teleop（遥操作主端）
#------------------------------------------------------------------------------
TELEOP_TYPE="bi_s1_leader"
LEFT_LEADER_PORT="/dev/ttyUSB0"
RIGHT_LEADER_PORT="/dev/ttyUSB1"

# Policy（策略客户端）
#------------------------------------------------------------------------------
POLICY_TYPE="openpi"                         # 策略类型，目前支持 openpi
OPENPI_HOST="localhost"
OPENPI_PORT=8000

# Cameras（摄像头 — 严格 JSON 格式）
#------------------------------------------------------------------------------
CAMERA_CONFIG='{"front":{"type":"intelrealsense","serial_number_or_name":"135122077817","width":848,"height":480,"fps":30},"front_1":{"type":"intelrealsense","serial_number_or_name":"935422072733","width":848,"height":480,"fps":30},"left_wrist":{"type":"intelrealsense","serial_number_or_name":"409122273564","width":640,"height":480,"fps":30},"right_wrist":{"type":"intelrealsense","serial_number_or_name":"409122273228","width":640,"height":480,"fps":30}}'

# Output（数据输出）
#------------------------------------------------------------------------------
OUTPUT_DIR="./outputs"
REPO_ID="bi_innov/bi_innov_$(date +%m%d_%H%M)"
TASK="test"
FPS=30
EPISODE_TIME_S=6000                         # 每个 episode 最大时长（秒），超时自动触发保存

# Control（控制模式）— 由终端交互菜单选择，无需手动修改
#------------------------------------------------------------------------------
CONTROL_MODE=""                              # 运行时选择: teleop | policy | mixed
CONTROL_MODE_INITIAL="teleop"                # mixed 模式下的初始控制方式

# Temporal Smoothing（时序平滑 — 策略推理时生效）
#------------------------------------------------------------------------------
USE_TEMPORAL_SMOOTHING=true
INFERENCE_RATE=7
LATENCY_K=4
MIN_SMOOTH_STEPS=12

# Alignment（对齐参数）
#------------------------------------------------------------------------------
ALIGN_MAX_STEP=0.02                          # 归零/对齐时单步最大关节角度变化

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

# 构建 robot 参数 (draccus 风格 --robot.xxx=yyy)
#------------------------------------------------------------------------------
ROBOT_ARGS=(
    --robot.type "$ROBOT_TYPE"
)

if [ "$ROBOT_TYPE" = "bi_innov_arm_v1" ]; then
    ROBOT_ARGS+=(--robot.left_port "${LEFT_FOLLOWER_PORT:-/dev/left_follower}")
    ROBOT_ARGS+=(--robot.right_port "${RIGHT_FOLLOWER_PORT:-/dev/right_follower}")
    ROBOT_ARGS+=(--robot.mode "${ROBOT_MODE:-collect}")
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

# 构建时序平滑参数 (draccus 风格)
# pure teleop 模式下跳过
#------------------------------------------------------------------------------
SMOOTH_ARGS=()
if [ "$CONTROL_MODE" != "teleop" ]; then
    if [ "$USE_TEMPORAL_SMOOTHING" = true ]; then
        SMOOTH_ARGS+=(--use_temporal_smoothing true)
        SMOOTH_ARGS+=(--inference_rate "$INFERENCE_RATE")
        SMOOTH_ARGS+=(--latency_k "$LATENCY_K")
        SMOOTH_ARGS+=(--min_smooth_steps "$MIN_SMOOTH_STEPS")
    else
        SMOOTH_ARGS+=(--use_temporal_smoothing false)
    fi
fi

# 打印配置
#------------------------------------------------------------------------------
print_config() {
    echo "=============================================="
    echo "    InnovArmV1 数据采集 + 策略推理"
    echo "=============================================="
    echo "Robot:        $ROBOT_TYPE"
    if [ "$ROBOT_TYPE" = "bi_innov_arm_v1" ]; then
        echo "  Left:       ${LEFT_FOLLOWER_PORT:-/dev/left_follower}"
        echo "  Right:      ${RIGHT_FOLLOWER_PORT:-/dev/right_follower}"
        echo "  Mode:       ${ROBOT_MODE:-collect}"
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
        echo "Smoothing:    $USE_TEMPORAL_SMOOTHING"
        if [ "$USE_TEMPORAL_SMOOTHING" = true ]; then
            echo "  Inference:  ${INFERENCE_RATE}Hz  latency_k=$LATENCY_K  min_m=$MIN_SMOOTH_STEPS"
        fi
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
    echo "  1) teleop  — 纯遥操作（示教采集，重力补偿模式）"
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

# 根据控制模式重新构建参数（teleop/policy/smooth 依赖 CONTROL_MODE）
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
    if [ "$USE_TEMPORAL_SMOOTHING" = true ]; then
        SMOOTH_ARGS=(--use_temporal_smoothing true --inference_rate "$INFERENCE_RATE" --latency_k "$LATENCY_K" --min_smooth_steps "$MIN_SMOOTH_STEPS")
    else
        SMOOTH_ARGS=(--use_temporal_smoothing false)
    fi
else
    POLICY_ARGS=()
    SMOOTH_ARGS=()
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
    "${SMOOTH_ARGS[@]}"

echo ""
echo "数据采集完成!"

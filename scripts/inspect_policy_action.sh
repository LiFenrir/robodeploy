#!/bin/bash
#===============================================================================
# 策略 Action 检查 启动脚本
#
# 封装 inspect_policy_action.py，快速启动裸推理（无平滑、无录制、无遥操作），
# 将 action 日志写入 JSONL 文件供 action-analyzer 分析。
#
# 用法:
#   bash inspect_policy_action.sh
#===============================================================================

set -e

# ==============================================================================
# 快速配置区 — 修改以下变量即可
# ==============================================================================

# Robot（机器人硬件）
#------------------------------------------------------------------------------
ROBOT_TYPE="bi_s1_follower"                  # 单臂: s1_follower, so100_follower  双臂: bi_s1_follower, bi_so100_follower
# 双臂时取消下面两行注释
LEFT_FOLLOWER_PORT="/dev/left_follower"
RIGHT_FOLLOWER_PORT="/dev/right_follower"
# 单臂时取消下一行注释
# FOLLOWER_PORT="/dev/ttyUSB0"

# Policy（策略客户端）
#------------------------------------------------------------------------------
POLICY_TYPE="openpi"
OPENPI_HOST="localhost"
OPENPI_PORT=8000

# Cameras（摄像头 — 严格 JSON 格式）
#------------------------------------------------------------------------------
CAMERA_CONFIG='{"front":{"type":"intelrealsense","serial_number_or_name":"135122077817","width":848,"height":480,"fps":30},"front_1":{"type":"intelrealsense","serial_number_or_name":"935422072733","width":848,"height":480,"fps":30},"left_wrist":{"type":"intelrealsense","serial_number_or_name":"409122273564","width":640,"height":480,"fps":30},"right_wrist":{"type":"intelrealsense","serial_number_or_name":"409122273228","width":640,"height":480,"fps":30}}'

# Task & Control
#------------------------------------------------------------------------------
TASK="hang cloths"
FPS=30
MAX_STEPS=300
LOG_FILE="policy_actions.jsonl"

# ==============================================================================
# 脚本逻辑 — 一般无需修改
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSPECT_SCRIPT="$SCRIPT_DIR/inspect_policy_action.py"

if [ ! -f "$INSPECT_SCRIPT" ]; then
    echo "错误: 找不到 $INSPECT_SCRIPT"
    exit 1
fi

# 构建 robot 参数
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

# 构建 policy 参数
#------------------------------------------------------------------------------
POLICY_ARGS=(
    --policy.type "$POLICY_TYPE"
    --policy.host "$OPENPI_HOST"
    --policy.port "$OPENPI_PORT"
)

# 打印配置
#------------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "    策略 Action 检查 (Inspect)"
echo "=============================================="
echo "Robot:        $ROBOT_TYPE"
if [ "$ROBOT_TYPE" = "bi_s1_follower" ] || [ "$ROBOT_TYPE" = "bi_so100_follower" ]; then
    echo "  Left:       ${LEFT_FOLLOWER_PORT:-/dev/left_follower}"
    echo "  Right:      ${RIGHT_FOLLOWER_PORT:-/dev/right_follower}"
else
    echo "  Port:       ${FOLLOWER_PORT:-/dev/ttyUSB0}"
fi
echo ""
echo "Policy:       ${POLICY_TYPE} @ ${OPENPI_HOST}:${OPENPI_PORT}"
echo "Task:         $TASK"
echo "FPS:          $FPS"
echo "Max Steps:    $MAX_STEPS"
echo "Log:          $LOG_FILE"
echo "=============================================="
echo ""

read -p "确认配置无误后按 Enter 键开始 (Ctrl+C 取消)..."

python "$INSPECT_SCRIPT" \
    "${ROBOT_ARGS[@]}" \
    "${POLICY_ARGS[@]}" \
    --task "$TASK" \
    --fps "$FPS" \
    --max_steps "$MAX_STEPS" \
    --log_file "$LOG_FILE"

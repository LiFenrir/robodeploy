#!/bin/bash
#===============================================================================
# Innov Arm 本体示教数据采集 + 策略推理 (RTC 版本)
#
# RTC (Real-Time Chunking) 服务端时序平滑，替代 StreamBuffer 线性 crossfade。
# 同一机械臂即当示教器又当执行器，无需独立 teleoperator。
#
# 用法:
#   bash examples/innov/record_innov.sh
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
LEFT_PORT="/dev/ttyACM1"
RIGHT_PORT="/dev/ttyACM0"
ROBOT_MODE="collect"                          # collect=重力补偿示教  control=位置控制(策略推理时用)

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
REPO_ID="innov/innov_$(date +%m%d_%H%M)"
TASK="${TASK:-pick cloth, load into fabric printer}"
FPS=30
EPISODE_TIME_S="${EPISODE_TIME_S:-120}"

# Control（控制模式）— 由终端交互菜单选择，无需手动修改
#------------------------------------------------------------------------------
CONTROL_MODE=""                              # 运行时选择: collect | policy | mixed
CONTROL_MODE_INITIAL="collect"               # mixed 模式下的初始控制方式

# RTC（Real-Time Chunking — 服务端约束 + 客户端 blend，收发驱动）
#------------------------------------------------------------------------------
USE_RTC=false                                 # true 启用 RTC | false 回退 StreamBuffer
RTC_EXECUTION_HORIZON=15                     # 约束窗口大小（服务端取前 N 步约束，客户端 blend overlap）
WARMUP_ROUNDS=10                             # 推理预热轮数，0 跳过

# Temporal Smoothing（仅非 RTC 模式生效）
#------------------------------------------------------------------------------
USE_TEMPORAL_SMOOTHING=true                 # RTC 启用时自动忽略
MIN_SMOOTH_STEPS=8

# Action Smoothing（推理动作插值平滑，0 关闭）
#------------------------------------------------------------------------------
ACTION_SMOOTH_MAX_STEP=0.05                  # 推理动作单步最大变化(rad)，超过则插值

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
RECORD_SCRIPT="$PROJECT_ROOT/src/robodeploy/scripts/record_body_teaching.py"

# 检查 record_body_teaching.py 是否存在
if [ ! -f "$RECORD_SCRIPT" ]; then
    echo "错误: 找不到 $RECORD_SCRIPT"
    exit 1
fi

# 给所有串口赋予读写权限
SERIAL_PORTS=()
[ -n "${LEFT_PORT:-}" ] && SERIAL_PORTS+=("$LEFT_PORT")
[ -n "${RIGHT_PORT:-}" ] && SERIAL_PORTS+=("$RIGHT_PORT")

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
    --robot.left_port "${LEFT_PORT:-/dev/left_arm}"
    --robot.right_port "${RIGHT_PORT:-/dev/right_arm}"
    --robot.mode "${ROBOT_MODE:-collect}"
)

if [ -n "$CAMERA_CONFIG" ]; then
    ROBOT_ARGS+=(--robot.cameras "$CAMERA_CONFIG")
fi

# 构建 policy 参数 (draccus 风格 --policy.xxx=yyy)
# pure collect 模式下跳过
#------------------------------------------------------------------------------
POLICY_ARGS=()
if [ "$CONTROL_MODE" != "collect" ]; then
    POLICY_ARGS+=(--policy.type "$POLICY_TYPE")
    POLICY_ARGS+=(--policy.host "$OPENPI_HOST")
    POLICY_ARGS+=(--policy.port "$OPENPI_PORT")
fi

# 构建 RTC / 推理参数 (draccus 风格)
# pure collect 模式下跳过
#------------------------------------------------------------------------------
RTC_ARGS=()
if [ "$CONTROL_MODE" != "collect" ]; then
    if [ "$USE_RTC" = true ]; then
        RTC_ARGS+=(--use_rtc true --rtc_execution_horizon "$RTC_EXECUTION_HORIZON")
        RTC_ARGS+=(--use_temporal_smoothing false)
    else
        RTC_ARGS+=(--use_rtc false)
        if [ "$USE_TEMPORAL_SMOOTHING" = true ]; then
            RTC_ARGS+=(--use_temporal_smoothing true --min_smooth_steps "$MIN_SMOOTH_STEPS")
        else
            RTC_ARGS+=(--use_temporal_smoothing false)
        fi
    fi
    RTC_ARGS+=(--warmup_rounds "$WARMUP_ROUNDS")
    RTC_ARGS+=(--action_smooth_max_step "$ACTION_SMOOTH_MAX_STEP")
fi

# 打印配置
#------------------------------------------------------------------------------
print_config() {
    echo "=============================================="
    echo "  Innov Arm 本体示教 + 策略推理 (RTC)"
    echo "=============================================="
    echo "Robot:        $ROBOT_TYPE"
    echo "  Left:       ${LEFT_PORT:-/dev/left_arm}"
    echo "  Right:      ${RIGHT_PORT:-/dev/right_arm}"
    echo "  Mode:       ${ROBOT_MODE:-collect}"
    echo ""
    if [ "$CONTROL_MODE" != "collect" ]; then
        echo "Policy:       ${POLICY_TYPE} @ ${OPENPI_HOST}:${OPENPI_PORT}"
    else
        echo "Policy:       (无 — 纯示教模式)"
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
    if [ "$CONTROL_MODE" != "collect" ]; then
        echo "RTC:          $USE_RTC"
        if [ "$USE_RTC" = true ]; then
            echo "  Horizon:    ${RTC_EXECUTION_HORIZON}"
            echo "  收发驱动    Warmup: ${WARMUP_ROUNDS}"
        else
            echo "  Smoothing:  $USE_TEMPORAL_SMOOTHING"
            if [ "$USE_TEMPORAL_SMOOTHING" = true ]; then
                echo "    min_m=$MIN_SMOOTH_STEPS"
            fi
        fi
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
    echo "  1) collect — 纯示教采集（重力补偿模式，人拖拽机械臂示教）"
    echo "  2) policy  — 纯策略推理（无需示教，策略自主控制）"
    echo "  3) mixed   — 混合模式（示教 + 策略推理，P 键切换）"
    echo ""
    while true; do
        read -p "请输入选项 [1-3]: " choice
        case "$choice" in
            1) CONTROL_MODE="collect"; break;;
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

# 根据控制模式重新构建参数（policy/RTC 依赖 CONTROL_MODE）
if [ "$CONTROL_MODE" != "collect" ]; then
    POLICY_ARGS=(--policy.type "$POLICY_TYPE" --policy.host "$OPENPI_HOST" --policy.port "$OPENPI_PORT")
    if [ "$USE_RTC" = true ]; then
        RTC_ARGS=(--use_rtc true --rtc_execution_horizon "$RTC_EXECUTION_HORIZON" --use_temporal_smoothing false)
    else
        RTC_ARGS=(--use_rtc false)
        if [ "$USE_TEMPORAL_SMOOTHING" = true ]; then
            RTC_ARGS+=(--use_temporal_smoothing true --min_smooth_steps "$MIN_SMOOTH_STEPS")
        else
            RTC_ARGS+=(--use_temporal_smoothing false)
        fi
    fi
    RTC_ARGS+=(--warmup_rounds "$WARMUP_ROUNDS" --action_smooth_max_step "$ACTION_SMOOTH_MAX_STEP")
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

cd "${PROJECT_ROOT}"

python "$RECORD_SCRIPT" \
    "${ROBOT_ARGS[@]}" \
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

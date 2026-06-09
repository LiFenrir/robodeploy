#!/usr/bin/env bash
#
# Innov Arm bimanual body-teaching data collection.
#
# The same physical arms serve as both the teaching device (human backdrives
# them in gravity compensation mode) and the execution device. No separate
# leader hardware required.
#
# Usage:
#   bash examples/innov/record_innov.sh
#
# Adjust serial ports below to match your hardware.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# ---------- hardware config (edit as needed) ----------
LEFT_PORT="/dev/ttyACM1"
RIGHT_PORT="/dev/ttyACM0"

# ---------- camera config ----------
# Adjust type/serial_number_or_name/width/height/fps as needed
CAMERAS_JSON='{
  "front":       {"type": "intelrealsense", "serial_number_or_name": "135122077817", "width": 848, "height": 480, "fps": 30},
  "left_wrist":  {"type": "intelrealsense", "serial_number_or_name": "409122273564", "width": 640, "height": 480, "fps": 30},
  "right_wrist": {"type": "intelrealsense", "serial_number_or_name": "409122273228", "width": 640, "height": 480, "fps": 30}
}'

# ---------- output ----------
OUTPUT_DIR="${PROJECT_ROOT}/output"
REPO_ID="innov/innov_$(date +%m%d_%H%M)"
TASK="${TASK:-pick and place}"

# ---------- control ----------
ROBOT_MODE="${ROBOT_MODE:-collect}"          # collect=gravity comp (body teaching), control=position control
CONTROL_MODE="collect"                       # collect | policy | mixed
INITIAL_MODE="collect"                       # collect | policy
EPISODE_TIME_S="${EPISODE_TIME_S:-120}"
FPS="${FPS:-30}"
WEBUI_PORT="${WEBUI_PORT:-8080}"

echo "============================================"
echo "  Innov Arm Bimanual Body-Teaching Collection"
echo "============================================"
echo "  Robot ports:   ${LEFT_PORT} / ${RIGHT_PORT}"
echo "  Robot mode:    ${ROBOT_MODE}"
echo "  Output dir:    ${OUTPUT_DIR}"
echo "  Repo ID:       ${REPO_ID}"
echo "  Task:          ${TASK}"
echo "  Control mode:  ${CONTROL_MODE} (initial: ${INITIAL_MODE})"
echo "  WebUI:         http://0.0.0.0:${WEBUI_PORT}"
echo "============================================"
echo ""

cd "${PROJECT_ROOT}"

python src/robodeploy/scripts/record_body_teaching.py \
    --robot.type=bi_innov_arm_v1 \
    --robot.left_port="${LEFT_PORT}" \
    --robot.right_port="${RIGHT_PORT}" \
    --robot.mode="${ROBOT_MODE}" \
    # --robot.cameras="${CAMERAS_JSON}" \
    --output_dir="${OUTPUT_DIR}" \
    --repo_id="${REPO_ID}" \
    --task="${TASK}" \
    --fps="${FPS}" \
    --episode_time_s="${EPISODE_TIME_S}" \
    --control_mode "${CONTROL_MODE}" \
    --control_mode_initial "${INITIAL_MODE}" \
    --webui_port="${WEBUI_PORT}"

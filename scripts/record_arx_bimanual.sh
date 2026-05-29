#!/usr/bin/env bash
#
# ARX X5 bimanual body-teaching data collection with 3 RealSense cameras.
#
# The same physical arms serve as both the teaching device (human backdrives
# them in gravity compensation mode) and the execution device. No separate
# leader hardware required.
#
# Cameras (by serial number):
#   135  -> top
#   260  -> left_hand
#   352  -> right_hand
#
# Usage:
#   ./scripts/record_arx_bimanual.sh
#
# Adjust CAN ports below to match your hardware.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ---------- hardware config (edit as needed) ----------
LEFT_CAN="can1"
RIGHT_CAN="can3"

# ---------- ARX SDK paths ----------
export PYTHONPATH="/home/arx/ARX_X5/py:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/arx/ARX_X5/py/arx_x5_python/bimanual/api/arx_x5_src:/home/arx/ARX_X5/py/arx_x5_python/bimanual/api:/usr/local/lib:${LD_LIBRARY_PATH:-}"

# ---------- camera config ----------
# serial_number_or_name: RealSense serial number string
# Adjust width/height/fps as needed
CAMERAS_JSON='{
  "top":        {"type": "intelrealsense", "serial_number_or_name": "135222070706", "width": 848, "height": 480, "fps": 30},
  "left_hand":  {"type": "intelrealsense", "serial_number_or_name": "260322272375", "width": 848, "height": 480, "fps": 30},
  "right_hand": {"type": "intelrealsense", "serial_number_or_name": "352122274412", "width": 848, "height": 480, "fps": 30}
}'

# ---------- output ----------
OUTPUT_DIR="${PROJECT_ROOT}/outputs/arx5"
REPO_ID="arx_bimanual_$(date +%m%d_%H%M)"
TASK="${TASK:-pick and place}"

# ---------- control ----------
ROBOT_MODE="${ROBOT_MODE:-collect}"          # collect=gravity comp (body teaching), control=position control
CONTROL_MODE="collect"         # collect | policy | mixed
INITIAL_MODE="collect"       # collect | policy
EPISODE_TIME_S="${EPISODE_TIME_S:-120}"
FPS="${FPS:-30}"
WEBUI_PORT="${WEBUI_PORT:-8080}"

echo "============================================"
echo "  ARX X5 Bimanual Body-Teaching Collection"
echo "============================================"
echo "  Robot CAN:     ${LEFT_CAN} / ${RIGHT_CAN}"
echo "  Robot mode:    ${ROBOT_MODE}"
echo "  Output dir:    ${OUTPUT_DIR}"
echo "  Repo ID:       ${REPO_ID}"
echo "  Task:          ${TASK}"
echo "  Control mode:  ${CONTROL_MODE} (initial: ${INITIAL_MODE})"
echo "  WebUI:         http://0.0.0.0:${WEBUI_PORT}"
echo "============================================"
echo ""

cd "${PROJECT_ROOT}"

python scripts/record_body_teaching.py \
    --robot.type=bi_arx_x5 \
    --robot.left_can_port="${LEFT_CAN}" \
    --robot.right_can_port="${RIGHT_CAN}" \
    --robot.mode="${ROBOT_MODE}" \
    --robot.cameras="${CAMERAS_JSON}" \
    --output_dir="${OUTPUT_DIR}" \
    --repo_id="${REPO_ID}" \
    --task="${TASK}" \
    --fps="${FPS}" \
    --episode_time_s="${EPISODE_TIME_S}" \
    --control_mode "${CONTROL_MODE}" \
    --control_mode_initial "${INITIAL_MODE}" \
    --webui_port="${WEBUI_PORT}"

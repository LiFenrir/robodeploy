#!/bin/bash

python src/robodeploy/scripts/inference_step.py \
        --robot.type=bi_innov_arm_v1 \
        --robot.left_port=/dev/ttyACM0 --robot.right_port=/dev/ttyACM1 \
        --robot.mode=control \
        --robot.cameras='{"top":{"type":"intelrealsense",...}}' \
        --policy.type=openpi --policy.host=192.168.200.203 --policy.port=8000 \
        --task="Put the water flosser into the box" 

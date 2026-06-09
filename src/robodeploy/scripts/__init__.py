#!/usr/bin/env python
"""Data collection and dataset processing scripts for robodeploy.

This subpackage provides:
  - record_dataset: Unified data collection (teleop + policy inference) with NPY backend
  - record_body_teaching: Body-teaching data collection (no separate teleoperator)
  - record_config / record_config_body_teaching: draccus configuration dataclasses
  - filter_valid_episodes / filter_lerobot_dataset: Episode filtering
  - merge_lerobot_datasets: Dataset merging
  - space_mirroring / stack_front_cameras: Video processing
  - data_augment: Data augmentation pipeline
  - regenerate_stats: Statistics regeneration
  - reassign_tasks: Task label reassignment
  - split_by_position: Position-based dataset splitting
  - binarize_gripper: Gripper state binarization
"""

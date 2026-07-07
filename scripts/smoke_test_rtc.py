#!/usr/bin/env python
"""
RTC 冒烟测试：加载离线数据集，通过 openpi-client 打包发送到 openpi_server，
记录推理时间、RTC 平滑效果、预测值与真实值对比。

推理流程最大程度与 record_dataset.py 保持一致：
  - 使用 ActionQueue 管理 action chunk 生命周期
  - 传递 prev_chunk_left_over / inference_delay / execution_horizon 给 RTC
  - 图像预处理匹配 OpenPIPolicyClient（resize 224x224 + pad + CHW）

Usage:
    # 基本用法
    python scripts/smoke_test_rtc.py \
        --dataset_path /path/to/dataset \
        --host localhost --port 8000 \
        --task "fold the box"

    # 指定 episode 和帧数
    python scripts/smoke_test_rtc.py \
        --dataset_path /path/to/dataset \
        --max_episodes 5 \
        --max_frames_per_episode 200 \
        --no_rtc  # 禁用 RTC，仅测试基础推理

    # 使用 BGR 输入（当数据集图像来自 OpenCV 相机直录时）
    python scripts/smoke_test_rtc.py \
        --dataset_path /path/to/dataset \
        --bgr_input
"""

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RTC 支持检测
# ---------------------------------------------------------------------------
try:
    from openpi.policies.rtc.action_queue import ActionQueue
    from openpi.policies.rtc.configuration_rtc import RTCConfig

    HAS_RTC = True
except ImportError:
    ActionQueue = None  # type: ignore[assignment]
    RTCConfig = None  # type: ignore[assignment]
    HAS_RTC = False


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class SmokeTestConfig:
    """冒烟测试配置（命令行参数 + 默认值）。"""

    # 数据集
    dataset_path: str = ""
    max_episodes: int = 3
    max_frames_per_episode: int = 300

    # Policy 服务器
    host: str = "localhost"
    port: int = 8000
    task: str = "fold the box"

    # 图像预处理
    bgr_input: bool = False  # 数据集图像是否为 BGR（OpenCV 格式），默认 RGB

    # RTC
    use_rtc: bool = True
    rtc_execution_horizon: int = 10
    compare_rtc: bool = False  # 同帧 A/B 对比：RTC vs baseline 推理

    # 控制参数（匹配 record_dataset 默认值）
    fps: int = 30

    # 输出
    output_dir: str = "smoke_test_results"


# ---------------------------------------------------------------------------
# 数据集加载
# ---------------------------------------------------------------------------
def load_dataset(dataset_path: str, max_episodes: int):
    """从本地磁盘路径加载 LeRobot 数据集。

    数据集目录结构：
        dataset_path/
        ├── data/chunk-000/episode_000000.parquet
        ├── meta/info.json
        └── videos/chunk-000/...

    Args:
        dataset_path: 数据集根目录路径（包含 meta/ data/ videos/ 的目录）。
        max_episodes: 最多加载的 episode 数量。

    Returns:
        LeRobotDataset 实例。
    """
    from robodeploy.datasets.lerobot_dataset import LeRobotDataset

    root = Path(dataset_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {root}")
    if not (root / "meta" / "info.json").exists():
        raise FileNotFoundError(
            f"meta/info.json not found in {root}. "
            "Please provide the root directory of a LeRobot dataset "
            "(the directory containing meta/, data/, videos/)."
        )

    # repo_id 取目录名，仅用于本地标识，不影响数据加载
    repo_id = root.name
    logger.info("Loading dataset repo_id=%s from %s", repo_id, root)

    ds = LeRobotDataset(
        repo_id,
        root=str(root),
        episodes=list(range(max_episodes)),
    )
    return ds


# ---------------------------------------------------------------------------
# 图像预处理（匹配 OpenPIPolicyClient.infer）
# ---------------------------------------------------------------------------
def preprocess_images(
    images: dict[str, np.ndarray], bgr_input: bool = False
) -> dict[str, np.ndarray]:
    """图像预处理，精确匹配 OpenPIPolicyClient.infer() 的管线。

    BGR(Robot) → RGB → resize 224x224 + pad (PIL) → CHW transpose → float32/255

    Args:
        images: {cam_key: np.ndarray(H, W, C)}，uint8 或 float。
        bgr_input: True 表示输入是 BGR（OpenCV 机器人相机），
                   False 表示输入是 RGB（数据集视频帧，默认）。

    Returns:
        {short_cam_name: np.ndarray(C, 224, 224)}，float32 范围 [0,1]。
    """
    import cv2
    from robodeploy.policy_clients.openpi import resize_with_pad

    payload_images = {}
    for cam_name, img in images.items():
        if img is None:
            continue

        # 确保 uint8 HWC
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # BGR → RGB（匹配 OpenPIPolicyClient: cv2.cvtColor(img, cv2.COLOR_BGR2RGB)）
        if bgr_input:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # resize_with_pad: 输入 (H, W, C) → 输出 (224, 224, C) uint8
        # 内部使用 PIL.Image.BILINEAR，与 OpenPIPolicyClient 完全一致
        img = resize_with_pad(np.array([img]), 224, 224)[0]

        # HWC → CHW, float32 [0,1]
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0

        # camera 短名: "observation.images.front" → "front"
        short_name = cam_name.rsplit(".", 1)[-1] if "." in cam_name else cam_name
        payload_images[short_name] = img

    return payload_images


# ---------------------------------------------------------------------------
# 辅助：从数据集帧提取 observation
# ---------------------------------------------------------------------------
def extract_observation(frame: dict, camera_keys: list[str], state_key: str) -> tuple:
    """从 LeRobot 数据集帧中提取 (images, state, gt_action)。

    Args:
        frame: hf_dataset[idx] 返回的字典。
        camera_keys: 图像特征 key 列表，如 ["observation.images.front"]。
        state_key: 状态 key，如 "observation.state"。

    Returns:
        (images, state, gt_action): images 是 {cam_name: np.ndarray}，
            state 是 (D,) float64，gt_action 是 (A,) float64。
    """
    images = {}
    for cam_key in camera_keys:
        val = frame.get(cam_key)
        if val is not None:
            img = np.asarray(val)
            # 如果是 torch.Tensor，已经在 hf_transform_to_torch 中转为 CPU tensor
            if hasattr(img, "numpy"):
                img = img.numpy()
            # 处理 CHW → HWC
            if img.ndim == 3 and img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
            images[cam_key] = img

    state_val = frame.get(state_key)
    if state_val is not None:
        state = np.asarray(state_val, dtype=np.float64).flatten()
    else:
        state = np.array([], dtype=np.float64)

    action_val = frame.get("action")
    if action_val is not None:
        gt_action = np.asarray(action_val, dtype=np.float64).flatten()
    else:
        gt_action = np.array([], dtype=np.float64)

    return images, state, gt_action


# ---------------------------------------------------------------------------
# 指标收集器
# ---------------------------------------------------------------------------
class MetricsCollector:
    """收集和汇总测试指标。"""

    def __init__(self):
        self.entries: list[dict] = []

    def record(self, **kwargs):
        self.entries.append(kwargs)

    def summary(self) -> dict:
        if not self.entries:
            return {}

        infer_entries = [e for e in self.entries if e.get("need_infer")]
        all_with_mse = [e for e in self.entries if "mse" in e]

        s = {
            "total_frames": len(self.entries),
            "total_inferences": len(infer_entries),
        }

        if infer_entries:
            s["avg_infer_ms"] = float(np.mean([e["infer_ms"] for e in infer_entries]))
            s["avg_roundtrip_ms"] = float(
                np.mean([e["roundtrip_ms"] for e in infer_entries if e["roundtrip_ms"] > 0])
            )
            s["p95_infer_ms"] = float(np.percentile([e["infer_ms"] for e in infer_entries], 95))
            s["max_infer_ms"] = float(np.max([e["infer_ms"] for e in infer_entries]))
            s["min_infer_ms"] = float(np.min([e["infer_ms"] for e in infer_entries]))

        if all_with_mse:
            s["avg_mse"] = float(np.mean([e["mse"] for e in all_with_mse]))
            s["avg_mae"] = float(np.mean([e["mae"] for e in all_with_mse]))
            s["max_mse"] = float(np.max([e["mse"] for e in all_with_mse]))

            # 按维度统计（仅 inference 帧）
            infer_with_mse = [e for e in all_with_mse if e.get("need_infer")]
            if infer_with_mse and "per_dim_mse" in infer_with_mse[0]:
                dim_count = len(infer_with_mse[0]["per_dim_mse"])
                s["per_dim_avg_mse"] = [
                    float(np.mean([e["per_dim_mse"][d] for e in infer_with_mse]))
                    for d in range(dim_count)
                ]

        # RTC 指标
        rtc_delays = [e["inference_delay"] for e in infer_entries if e.get("inference_delay") is not None]
        if rtc_delays:
            s["rtc_avg_delay"] = float(np.mean(rtc_delays))
            s["rtc_max_delay"] = int(np.max(rtc_delays))
            s["rtc_delay_distribution"] = {
                str(k): int(v) for k, v in zip(*np.unique(rtc_delays, return_counts=True))
            }

        leftover_lens = [e["leftover_len"] for e in infer_entries if e.get("leftover_len") is not None]
        if leftover_lens:
            s["rtc_avg_leftover"] = float(np.mean(leftover_lens))
            s["rtc_max_leftover"] = int(np.max(leftover_lens))

        # RTC vs Baseline 对比指标
        cmp_entries = [e for e in infer_entries if "rtc_vs_baseline_mse" in e]
        if cmp_entries:
            s["rtc_vs_baseline_avg_mse"] = float(np.mean([e["rtc_vs_baseline_mse"] for e in cmp_entries]))
            s["rtc_vs_baseline_max_mse"] = float(np.max([e["rtc_vs_baseline_mse"] for e in cmp_entries]))
            s["rtc_vs_baseline_count"] = len(cmp_entries)

            # 按 step 聚合（step 0 表示 RTC 对 chunk 第一步的修正量）
            all_steps = [e["rtc_vs_baseline_per_step_mse"] for e in cmp_entries]
            max_steps = max(len(s) for s in all_steps)
            s["rtc_vs_baseline_avg_per_step"] = [
                float(np.mean([s[t] for s in all_steps if t < len(s)]))
                for t in range(max_steps)
            ]

            # RTC 修正幅度（RTC action 与 baseline action 的 L2 距离随时间发散程度）
            s["rtc_smoothing_divergence"] = float(
                s["rtc_vs_baseline_avg_per_step"][-1] / (s["rtc_vs_baseline_avg_per_step"][0] + 1e-10)
                if s["rtc_vs_baseline_avg_per_step"][0] > 0 else 0.0
            )

        return s


# ---------------------------------------------------------------------------
# 主测试流程
# ---------------------------------------------------------------------------
def run_smoke_test(cfg: SmokeTestConfig):
    """执行冒烟测试。

    推理流程与 record_dataset._start_inference_thread 中的 RTC 模式对齐：
      1. 当 ActionQueue 为空时，发起一次 inference
      2. 传入 prev_chunk_left_over, inference_delay, execution_horizon
      3. 收到 action chunk 后调用 action_queue.merge()
      4. 逐帧从 action_queue.get() 消费动作
    """
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    # ---- 加载数据集 ----
    logger.info("=" * 60)
    logger.info("Step 1/4: 加载数据集...")
    ds = load_dataset(cfg.dataset_path, cfg.max_episodes)
    logger.info(
        "数据集已加载: %d episodes, %d frames, fps=%d",
        ds.meta.total_episodes,
        ds.num_frames,
        ds.fps,
    )

    # 检测 camera / state / action keys
    camera_keys = sorted(
        [k for k in ds.meta.camera_keys if k.startswith("observation.")]
    )
    state_key = "observation.state"

    if not camera_keys:
        logger.error("未找到 camera keys！可用的 features: %s", list(ds.features.keys()))
        sys.exit(1)

    logger.info("Camera keys: %s", camera_keys)
    logger.info("State key: %s, Action dim: %s", state_key, ds.meta.shapes.get("action", "N/A"))

    # ---- 连接 Policy 服务器 ----
    logger.info("=" * 60)
    logger.info("Step 2/4: 连接 Policy 服务器 ws://%s:%d ...", cfg.host, cfg.port)
    try:
        policy = WebsocketClientPolicy(cfg.host, cfg.port)
    except Exception as e:
        logger.error("无法连接到 Policy 服务器: %s", e)
        logger.error("请确保 openpi_server 正在运行: python scripts/serve_policy.py ...")
        sys.exit(1)

    server_meta = policy.get_server_metadata()
    logger.info("已连接。服务器元数据: %s", json.dumps(server_meta, indent=2, default=str))

    # ---- RTC 初始化 ----
    logger.info("=" * 60)
    logger.info("Step 3/4: 初始化 RTC...")
    action_queue = None
    rtc_enabled = False
    if cfg.use_rtc and HAS_RTC:
        rtc_cfg = RTCConfig(
            enabled=True,
            execution_horizon=cfg.rtc_execution_horizon,
        )
        action_queue = ActionQueue(rtc_cfg)
        rtc_enabled = True
        logger.info("RTC 模式已启用 (execution_horizon=%d)", cfg.rtc_execution_horizon)
    elif cfg.use_rtc and not HAS_RTC:
        logger.warning("RTC 模块不可用，回退到非 RTC 模式。")
    else:
        logger.info("非 RTC 模式（无 temporal smoothing）。")

    # ---- 输出目录 ----
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "smoke_test_results.jsonl"
    summary_path = output_dir / "smoke_test_summary.json"

    # ---- 处理 Episodes ----
    logger.info("=" * 60)
    logger.info("Step 4/4: 开始冒烟测试...")
    logger.info("  Episodes: %d, Max frames/ep: %d", cfg.max_episodes, cfg.max_frames_per_episode)
    logger.info("  RTC: %s, Image mode: %s", rtc_enabled, "BGR" if cfg.bgr_input else "RGB")
    logger.info("=" * 60)

    metrics = MetricsCollector()
    ep_data_index = ds.episode_data_index
    total_infer_count = 0

    with open(log_path, "w", encoding="utf-8") as log_file:
        for ep_idx in range(min(cfg.max_episodes, ds.meta.total_episodes)):
            if ep_idx not in ep_data_index:
                logger.warning("Episode %d 不在 episode_data_index 中，跳过", ep_idx)
                continue

            ep_start, ep_end = ep_data_index[ep_idx]
            ep_start = ep_start.item() if hasattr(ep_start, "item") else int(ep_start)
            ep_end = ep_end.item() if hasattr(ep_end, "item") else int(ep_end)
            ep_length = ep_end - ep_start
            effective_length = min(cfg.max_frames_per_episode, ep_length)

            logger.info("\n" + "=" * 60)
            logger.info("Episode %d/%d: %d frames (使用 %d), indices [%d, %d)",
                        ep_idx + 1, min(cfg.max_episodes, ds.meta.total_episodes),
                        ep_length, effective_length, ep_start, ep_end)
            logger.info("Task: %s", ds.meta.tasks.get(ep_idx, cfg.task))
            logger.info("=" * 60)

            # 每个 episode 重置 RTC 状态
            if action_queue is not None:
                action_queue.clear()
            chunk = None
            chunk_idx = 0

            for fi in range(effective_length):
                frame_idx = ep_start + fi
                frame = ds[frame_idx]

                # 提取 observation 和 ground truth
                images, state, gt_action = extract_observation(frame, camera_keys, state_key)

                # ---- 判断是否需要 inference ----
                if action_queue is not None:
                    need_infer = action_queue.empty()
                else:
                    need_infer = (chunk is None or chunk_idx >= len(chunk))

                infer_ms = 0.0
                rtc_roundtrip_ms = 0.0
                prev_total_ms = None
                inference_delay = None
                leftover_len = None
                actions_np = None

                if need_infer:
                    # ---- Preprocess ----
                    payload_images = preprocess_images(images, bgr_input=cfg.bgr_input)
                    payload = {
                        "state": state,
                        "images": payload_images,
                        "prompt": cfg.task,
                    }

                    # ---- RTC kwargs (匹配 record_dataset) ----
                    rtc_kwargs = {}
                    if action_queue is not None:
                        action_index_before = action_queue.get_action_index()
                        inference_delay = action_index_before
                        prev_leftover = action_queue.get_left_over()
                        if prev_leftover is not None:
                            leftover_len = prev_leftover.shape[0]
                            rtc_kwargs["prev_chunk_left_over"] = (
                                prev_leftover.cpu().numpy()
                            )
                            rtc_kwargs["inference_delay"] = inference_delay
                            rtc_kwargs["execution_horizon"] = cfg.rtc_execution_horizon
                        else:
                            leftover_len = 0

                    # ---- RTC Inference ----
                    t0 = time.monotonic()
                    try:
                        result = policy.infer(payload, **rtc_kwargs)
                    except Exception as e:
                        logger.error("Inference 失败 (frame %d): %s", frame_idx, e)
                        metrics.record(
                            episode=ep_idx, frame=fi, global_frame=int(frame_idx),
                            need_infer=True, error=str(e),
                            infer_ms=0.0, roundtrip_ms=0.0,
                        )
                        continue
                    rtc_roundtrip_ms = (time.monotonic() - t0) * 1000
                    total_infer_count += 1

                    # ---- 提取 timing ----
                    server_timing = result.get("server_timing", {})
                    infer_ms = server_timing.get(
                        "infer_ms", result.get("policy_timing", {}).get("infer_ms", 0.0)
                    )
                    prev_total_ms = server_timing.get("prev_total_ms", None)

                    # ---- 处理 action chunk ----
                    actions = result.get("actions")
                    rtc_actions_np = np.asarray(actions) if (actions is not None and len(actions) > 0) else None
                    actions_np = rtc_actions_np

                    # ---- Baseline 对比推理（同一输入，无 RTC kwargs）----
                    baseline_actions_np = None
                    baseline_infer_ms = 0.0
                    baseline_roundtrip_ms = 0.0
                    rtc_vs_baseline_mse = None
                    rtc_vs_baseline_per_step_mse = None

                    if cfg.compare_rtc and rtc_enabled and rtc_actions_np is not None:
                        t1 = time.monotonic()
                        try:
                            baseline_result = policy.infer(payload)  # 无 RTC kwargs
                        except Exception as e:
                            logger.warning("Baseline inference 失败: %s", e)
                            baseline_result = {}
                        baseline_roundtrip_ms = (time.monotonic() - t1) * 1000
                        total_infer_count += 1

                        baseline_actions = baseline_result.get("actions")
                        if baseline_actions is not None and len(baseline_actions) > 0:
                            baseline_actions_np = np.asarray(baseline_actions)
                            baseline_infer_ms = baseline_result.get("server_timing", {}).get(
                                "infer_ms", baseline_result.get("policy_timing", {}).get("infer_ms", 0.0)
                            )

                            # 对比 RTC vs baseline（截取相同长度）
                            cmp_len = min(len(rtc_actions_np), len(baseline_actions_np))
                            rtc_cmp = rtc_actions_np[:cmp_len]
                            baseline_cmp = baseline_actions_np[:cmp_len]
                            diff_chunk = rtc_cmp - baseline_cmp
                            rtc_vs_baseline_mse = round(float(np.mean(diff_chunk**2)), 8)
                            rtc_vs_baseline_per_step_mse = [
                                round(float(np.mean(diff_chunk[t]**2)), 8)
                                for t in range(cmp_len)
                            ]
                            logger.info(
                                "  [A/B] RTC vs baseline: chunk_mse=%.6f  per_step=%s",
                                rtc_vs_baseline_mse,
                                ", ".join(f"{v:.4f}" for v in rtc_vs_baseline_per_step_mse[:6]),
                            )

                    if rtc_actions_np is not None:
                        if action_queue is not None:
                            actions_tensor = torch.from_numpy(rtc_actions_np)
                            action_queue.merge(
                                original_actions=actions_tensor,
                                processed_actions=actions_tensor,
                                real_delay=action_index_before,
                                action_index_before_inference=action_index_before,
                            )
                            act_tensor = action_queue.get()
                            predicted = (
                                act_tensor.cpu().numpy() if act_tensor is not None else None
                            )
                        else:
                            chunk = rtc_actions_np
                            chunk_idx = 0
                            predicted = chunk[chunk_idx] if len(chunk) > 0 else None
                            chunk_idx += 1
                    else:
                        predicted = None
                else:
                    # ---- 从 ActionQueue / chunk 消费 ----
                    if action_queue is not None:
                        act_tensor = action_queue.get()
                        predicted = (
                            act_tensor.cpu().numpy() if act_tensor is not None else None
                        )
                    elif chunk is not None and chunk_idx < len(chunk):
                        predicted = chunk[chunk_idx]
                        chunk_idx += 1
                    else:
                        predicted = None

                # ---- 记录指标 ----
                entry = {
                    "episode": int(ep_idx),
                    "frame": int(fi),
                    "global_frame": int(frame_idx),
                    "infer_count": total_infer_count,
                    "need_infer": need_infer,
                    "infer_ms": round(float(infer_ms), 2),
                    "roundtrip_ms": round(float(rtc_roundtrip_ms), 2),
                    "prev_total_ms": round(float(prev_total_ms), 2) if prev_total_ms else None,
                    "inference_delay": int(inference_delay) if inference_delay is not None else None,
                    "leftover_len": int(leftover_len) if leftover_len is not None else None,
                    "chunk_size": int(len(actions_np)) if actions_np is not None else None,
                }

                # RTC vs Baseline 对比指标
                if rtc_vs_baseline_mse is not None:
                    entry.update({
                        "baseline_infer_ms": round(float(baseline_infer_ms), 2),
                        "baseline_roundtrip_ms": round(float(baseline_roundtrip_ms), 2),
                        "rtc_vs_baseline_mse": rtc_vs_baseline_mse,
                        "rtc_vs_baseline_per_step_mse": rtc_vs_baseline_per_step_mse,
                    })

                if predicted is not None and len(gt_action) > 0:
                    min_len = min(len(predicted), len(gt_action))
                    p = predicted[:min_len]
                    g = gt_action[:min_len]
                    diff = p - g
                    entry.update({
                        "mse": round(float(np.mean(diff**2)), 8),
                        "mae": round(float(np.mean(np.abs(diff))), 8),
                        "per_dim_mse": [round(float(d**2), 8) for d in diff],
                        "per_dim_diff": [round(float(d), 6) for d in diff],
                        "pred_mean": round(float(np.mean(p)), 6),
                        "gt_mean": round(float(np.mean(g)), 6),
                        "pred_first3": [round(float(x), 6) for x in p[:3]],
                        "gt_first3": [round(float(x), 6) for x in g[:3]],
                    })

                metrics.record(**entry)
                log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

                # ---- 进度打印 ----
                if fi % 30 == 0 or (need_infer and fi > 0):
                    parts = [
                        f"[ep={ep_idx} f={fi:4d}]",
                        f"infer=#{total_infer_count}",
                    ]
                    if need_infer:
                        parts.append(f"infer_ms={infer_ms:.0f}")
                        parts.append(f"rtt_ms={rtc_roundtrip_ms:.0f}")
                        if leftover_len is not None:
                            parts.append(f"leftover={leftover_len}")
                        if rtc_vs_baseline_mse is not None:
                            parts.append(f"A/B_mse={rtc_vs_baseline_mse:.4f}")
                    mse_val = entry.get("mse")
                    if mse_val is not None:
                        parts.append(f"mse={mse_val:.6f}")
                    logger.info(" ".join(parts))

                log_file.flush()

            # ---- Episode 小结 ----
            ep_entries = [e for e in metrics.entries if e["episode"] == ep_idx]
            ep_infers = [e for e in ep_entries if e.get("need_infer")]
            if ep_infers:
                avg_infer = np.mean([e["infer_ms"] for e in ep_infers])
                avg_rtt = np.mean([e["roundtrip_ms"] for e in ep_infers if e["roundtrip_ms"] > 0])
                logger.info(
                    "--- Episode %d 小结: %d frames, %d inferences, "
                    "avg_infer=%.1fms, avg_rtt=%.1fms ---",
                    ep_idx, len(ep_entries), len(ep_infers), avg_infer, avg_rtt,
                )

    # ---- 最终汇总 ----
    summary = metrics.summary()
    logger.info("\n" + "=" * 60)
    logger.info("冒烟测试结果汇总")
    logger.info("=" * 60)
    logger.info("  总帧数:            %d", summary.get("total_frames", 0))
    logger.info("  总推理次数:        %d", summary.get("total_inferences", 0))
    logger.info("  平均推理时间:      %.1f ms", summary.get("avg_infer_ms", 0))
    logger.info("  平均往返时间:      %.1f ms", summary.get("avg_roundtrip_ms", 0))
    logger.info("  P95 推理时间:      %.1f ms", summary.get("p95_infer_ms", 0))
    logger.info("  最大推理时间:      %.1f ms", summary.get("max_infer_ms", 0))
    logger.info("  平均 MSE:          %.8f", summary.get("avg_mse", 0))
    logger.info("  平均 MAE:          %.8f", summary.get("avg_mae", 0))
    logger.info("  最大 MSE:          %.8f", summary.get("max_mse", 0))

    if "per_dim_avg_mse" in summary:
        dim_mses = summary["per_dim_avg_mse"]
        logger.info("  各维度 MSE:        %s", ", ".join(f"[{i}]{v:.6f}" for i, v in enumerate(dim_mses[:8])))
        if len(dim_mses) > 8:
            logger.info("                    ... (%d dims total)", len(dim_mses))

    if "rtc_avg_delay" in summary:
        logger.info("  --- RTC 指标 ---")
        logger.info("  RTC 平均延迟:      %.1f frames", summary["rtc_avg_delay"])
        logger.info("  RTC 最大延迟:      %d frames", summary["rtc_max_delay"])
        logger.info("  RTC 延迟分布:      %s", summary.get("rtc_delay_distribution", {}))
        logger.info("  RTC 平均残留:      %.1f frames", summary.get("rtc_avg_leftover", 0))

    if "rtc_vs_baseline_avg_mse" in summary:
        logger.info("  --- RTC vs Baseline (平滑效果) ---")
        logger.info("  对比次数:          %d", summary["rtc_vs_baseline_count"])
        logger.info("  平均 chunk MSE:    %.8f", summary["rtc_vs_baseline_avg_mse"])
        logger.info("  最大 chunk MSE:    %.8f", summary["rtc_vs_baseline_max_mse"])
        per_step = summary.get("rtc_vs_baseline_avg_per_step", [])
        if per_step:
            logger.info("  逐 step MSE:       %s",
                        ", ".join(f"[t{t}]{v:.6f}" for t, v in enumerate(per_step[:8])))
            if len(per_step) > 8:
                logger.info("                    ... (%d steps total)", len(per_step))
        div = summary.get("rtc_smoothing_divergence", 0)
        if abs(div) > 1e-6 and div < 100:
            logger.info("  平滑发散比:        %.2f (RTC 修正在 chunk 末端约为首端的 %.0f%%)", div, div * 100)
        else:
            logger.info("  平滑发散比:        %.2f", div)
        logger.info("  (step_mse[-1]/step_mse[0], >1 表示 RTC 修正随时间衰减)")

    # 保存汇总
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    logger.info("\n详细日志: %s", log_path)
    logger.info("汇总报告: %s", summary_path)
    logger.info("完成。")

    return summary


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="RTC 冒烟测试 — 离线数据集 → openpi_client → openpi_server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本 RTC 测试
  python scripts/smoke_test_rtc.py --dataset_path /data/datasets/my_robot

  # 无 RTC 的基础推理测试
  python scripts/smoke_test_rtc.py --dataset_path /data/datasets/my_robot --no_rtc

  # BGR 输入 + 大范围测试
  python scripts/smoke_test_rtc.py --dataset_path /data/datasets/my_robot \\
      --bgr_input --max_episodes 10 --max_frames_per_episode 500
        """,
    )

    # 数据集
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="LeRobot 数据集根目录路径",
    )
    parser.add_argument(
        "--max_episodes", type=int, default=3,
        help="最大测试 episode 数 (default: 3)",
    )
    parser.add_argument(
        "--max_frames_per_episode", type=int, default=300,
        help="每 episode 最大处理帧数 (default: 300)",
    )

    # 服务器
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task", type=str, default="fold the box")

    # 图像
    parser.add_argument(
        "--bgr_input", action="store_true",
        help="输入图像是 BGR（OpenCV 格式），默认 RGB（数据集视频）",
    )

    # RTC
    parser.add_argument(
        "--no_rtc", action="store_true",
        help="禁用 RTC，使用简单 chunk 模式",
    )
    parser.add_argument(
        "--compare_rtc", action="store_true",
        help="同帧 A/B 对比：每次推理同时发 RTC + baseline 两路请求，量化 RTC 平滑效果",
    )
    parser.add_argument(
        "--rtc_execution_horizon", type=int, default=10,
        help="RTC execution horizon (default: 10)",
    )

    # 输出
    parser.add_argument(
        "--output_dir", type=str, default="smoke_test_results",
        help="输出目录 (default: smoke_test_results)",
    )

    args = parser.parse_args()

    cfg = SmokeTestConfig(
        dataset_path=args.dataset_path,
        max_episodes=args.max_episodes,
        max_frames_per_episode=args.max_frames_per_episode,
        host=args.host,
        port=args.port,
        task=args.task,
        bgr_input=args.bgr_input,
        use_rtc=not args.no_rtc,
        compare_rtc=args.compare_rtc,
        rtc_execution_horizon=args.rtc_execution_horizon,
        output_dir=args.output_dir,
    )

    run_smoke_test(cfg)


if __name__ == "__main__":
    main()

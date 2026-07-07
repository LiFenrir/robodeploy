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
import threading
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

    # 预热
    warmup_rounds: int = 10  # 首帧预热推理次数，0 表示跳过

    # RTC
    use_rtc: bool = True
    rtc_execution_horizon: int = 10
    compare_rtc: bool = False  # 同帧 A/B 对比：RTC vs baseline 推理

    # 多线程推理
    inference_rate: float = 3.0  # 推理线程频率 (Hz)
    action_smooth_max_step: float = 0.05  # 单步最大关节位移 (rad)，0 关闭
    use_closest_lookup: bool = False  # 推理时用 closest-action 查找观测帧

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


def _extract_infer_ms(result: dict) -> float:
    """从 policy 响应中提取 server 侧推理耗时。"""
    server_timing = result.get("server_timing", {})
    return server_timing.get("infer_ms", result.get("policy_timing", {}).get("infer_ms", 0.0))


# ---------------------------------------------------------------------------
# 工具函数：最近帧查找 + 插值平滑
# ---------------------------------------------------------------------------
def build_action_lookup(
    ds, ep_start: int, ep_end: int,
) -> list[tuple[int, np.ndarray]]:
    """预建 episode 的 (global_frame_idx, gt_action) 列表，供推理线程做 NN 搜索。"""
    lookup = []
    for fi in range(ep_end - ep_start):
        frame_idx = ep_start + fi
        frame = ds[frame_idx]
        action_val = frame.get("action")
        if action_val is not None:
            gt_action = np.asarray(action_val, dtype=np.float64).flatten()
            lookup.append((frame_idx, gt_action))
    return lookup


def find_closest_observation_frame(
    current_action: np.ndarray,
    action_lookup: list[tuple[int, np.ndarray]],
) -> int:
    """L2 距离查找与当前执行动作最接近的数据集帧，返回 global_frame_idx。

    若 current_action 为 None 或 lookup 为空，返回 lookup 首帧。
    """
    if current_action is None or not action_lookup:
        return action_lookup[0][0] if action_lookup else -1
    gt_actions = np.stack([a for _, a in action_lookup])
    distances = np.sum((gt_actions - current_action) ** 2, axis=1)
    closest_idx = int(np.argmin(distances))
    return action_lookup[closest_idx][0]


def smooth_predicted_action(
    prev_action: np.ndarray | None,
    curr_action: np.ndarray,
    max_step: float,
) -> np.ndarray:
    """单步位移限制，模拟 record_dataset 的 smooth_inference_action 但无物理机器人。

    若 prev_action 为 None 或 max_step <= 0 或最大位移 ≤ max_step，直接返回 curr_action。
    否则等比例缩放 diff 使最大关节位移 = max_step。
    """
    if prev_action is None or max_step <= 0 or curr_action is None:
        return curr_action
    diff = curr_action - prev_action
    max_disp = float(np.abs(diff).max())
    if max_disp <= max_step:
        return curr_action
    ratio = max_step / max_disp
    return prev_action + diff * ratio


# ---------------------------------------------------------------------------
# 推理线程（参考 record_dataset._start_inference_thread）
# ---------------------------------------------------------------------------
def _start_inference_thread(
    policy,
    action_queue,
    shared_state: dict,
    ds,
    action_lookup: list[tuple[int, np.ndarray]],
    camera_keys: list[str],
    state_key: str,
    task: str,
    inference_rate: float,
    rtc_execution_horizon: int,
    bgr_input: bool,
    compare_rtc: bool = False,
    use_closest_lookup: bool = False,
) -> threading.Thread:
    """启动 daemon 线程，按 inference_rate 频率异步发推理。

    触发条件: action_queue.qsize() < execution_horizon（保证 leftover > 0）。
    观测来源: 默认用 shared_state["latest_frame"]（当前回放位置），
              use_closest_lookup=True 时用 action 最近邻查找。
    """

    def _run() -> None:
        rate = 1.0 / inference_rate
        while not shared_state["stop"].is_set():
            qs = action_queue.qsize()
            if qs >= rtc_execution_horizon:
                time.sleep(0.01)
                continue

            # 确定观测帧
            with shared_state["lock"]:
                latest_frame_val = shared_state.get("latest_frame", -1)

            if use_closest_lookup:
                with shared_state["lock"]:
                    curr_action = shared_state.get("current_action")
                obs_idx = find_closest_observation_frame(curr_action, action_lookup)
            else:
                obs_idx = int(latest_frame_val) if latest_frame_val is not None else -1

            if obs_idx < 0:
                time.sleep(0.1)
                continue

            frame = ds[obs_idx]
            images, state, _ = extract_observation(frame, camera_keys, state_key)

            try:
                payload_images = preprocess_images(images, bgr_input=bgr_input)
                payload = {
                    "state": state,
                    "images": payload_images,
                    "prompt": task,
                }

                # RTC kwargs
                action_index_before = action_queue.get_action_index()
                rtc_kwargs: dict = {}
                prev_leftover = action_queue.get_left_over()
                if prev_leftover is not None:
                    rtc_kwargs["prev_chunk_left_over"] = prev_leftover.cpu().numpy()
                    rtc_kwargs["inference_delay"] = action_index_before
                    rtc_kwargs["execution_horizon"] = rtc_execution_horizon

                result = policy.infer(payload, **rtc_kwargs)
                actions = result.get("actions", None)
                if actions is not None and len(actions) > 0:
                    actions_tensor = torch.from_numpy(np.asarray(actions))
                    action_queue.merge(
                        original_actions=actions_tensor,
                        processed_actions=actions_tensor,
                        real_delay=action_index_before,
                        action_index_before_inference=action_index_before,
                    )

                # 保存结果供主线程记录
                with shared_state["lock"]:
                    shared_state["inference_ok"] = True
                    shared_state["last_infer_result"] = result
                    shared_state["last_infer_obs_idx"] = obs_idx
                    shared_state["last_infer_delay"] = action_index_before
                    shared_state["last_infer_leftover_len"] = (
                        prev_leftover.shape[0] if prev_leftover is not None else 0
                    )

                # A/B 对比
                if compare_rtc:
                    t1 = time.monotonic()
                    try:
                        baseline_result = policy.infer(payload)
                    except Exception:
                        baseline_result = {}
                    baseline_rtt = (time.monotonic() - t1) * 1000
                    with shared_state["lock"]:
                        shared_state["baseline_result"] = baseline_result
                        shared_state["baseline_roundtrip_ms"] = baseline_rtt

            except Exception as e:
                logger.warning("Inference error: %s", e)
                with shared_state["lock"]:
                    shared_state["inference_ok"] = False

            time.sleep(rate)

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    return th


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

        # 推理帧：显式 need_infer 或有非零 infer_ms（多线程 RTC 模式）
        infer_entries = [
            e for e in self.entries
            if e.get("need_infer") or (e.get("infer_ms") or 0) > 0
        ]
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

            # 按维度统计（仅推理相关帧）
            infer_with_mse = [
                e for e in all_with_mse
                if e.get("need_infer") or (e.get("infer_ms") or 0) > 0
            ]
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
                str(k): int(v) for k, v in zip(*np.unique(rtc_delays, return_counts=True), strict=False)
            }

        leftover_lens = [e["leftover_len"] for e in infer_entries if e.get("leftover_len") is not None]
        if leftover_lens:
            s["rtc_avg_leftover"] = float(np.mean(leftover_lens))
            s["rtc_max_leftover"] = int(np.max(leftover_lens))

        # 多线程指标
        queue_sizes = [e["queue_size"] for e in self.entries if e.get("queue_size") is not None]
        if queue_sizes:
            s["avg_queue_size"] = float(np.mean(queue_sizes))
            s["min_queue_size"] = int(np.min(queue_sizes))
            s["max_queue_size"] = int(np.max(queue_sizes))

        smoothed_entries = [e for e in self.entries if e.get("smoothed")]
        if smoothed_entries:
            s["pct_smoothed"] = round(len(smoothed_entries) / max(len(self.entries), 1) * 100, 1)

        closest_offsets = [e["closest_frame_offset"] for e in infer_entries if "closest_frame_offset" in e]
        if closest_offsets:
            s["avg_closest_offset"] = float(np.mean([abs(o) for o in closest_offsets]))

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
    from robodeploy.policy_clients.openpi.websocket_client import WebsocketClientPolicy

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

    # ---- 预热 ----
    if cfg.warmup_rounds > 0:
        logger.info("=" * 60)
        logger.info("推理预热 (%d rounds)...", cfg.warmup_rounds)
        ep_data_index = ds.episode_data_index
        first_ep_start = int(ep_data_index["from"][0])
        first_frame = ds[first_ep_start]
        warmup_images, warmup_state, _ = extract_observation(first_frame, camera_keys, state_key)
        warmup_payload = {
            "state": warmup_state,
            "images": preprocess_images(warmup_images, bgr_input=cfg.bgr_input),
            "prompt": cfg.task,
        }
        warmup_times = []
        for w in range(cfg.warmup_rounds):
            t0 = time.monotonic()
            try:
                policy.infer(warmup_payload)
                elapsed = (time.monotonic() - t0) * 1000
                warmup_times.append(elapsed)
                logger.info("  预热 %2d/%d: %.0fms", w + 1, cfg.warmup_rounds, elapsed)
            except Exception as e:
                logger.warning("  预热 %2d/%d 失败: %s", w + 1, cfg.warmup_rounds, e)
        if warmup_times:
            logger.info(
                "预热完成: avg=%.0fms, min=%.0fms, max=%.0fms",
                np.mean(warmup_times), np.min(warmup_times), np.max(warmup_times),
            )

    # ---- 处理 Episodes ----
    logger.info("=" * 60)
    logger.info("Step 4/4: 开始冒烟测试...")
    logger.info("  Episodes: %d, Max frames/ep: %d", cfg.max_episodes, cfg.max_frames_per_episode)
    logger.info("  RTC: %s, Image mode: %s", rtc_enabled, "BGR" if cfg.bgr_input else "RGB")
    if rtc_enabled:
        logger.info("  Inference rate: %.1f Hz, Smooth max step: %.3f",
                     cfg.inference_rate, cfg.action_smooth_max_step)
    logger.info("=" * 60)

    metrics = MetricsCollector()
    ep_data_index = ds.episode_data_index
    loaded_episodes = ds.episodes if ds.episodes else list(range(len(ep_data_index["from"])))
    total_infer_count = 0
    replay_interval = 1.0 / ds.fps  # 按数据集原生 fps 回放

    with open(log_path, "w", encoding="utf-8") as log_file:
        for i in range(min(cfg.max_episodes, len(loaded_episodes))):
            ep_idx = loaded_episodes[i]
            ep_start = int(ep_data_index["from"][i])
            ep_end = int(ep_data_index["to"][i])
            ep_length = ep_end - ep_start

            logger.info("\n" + "=" * 60)
            logger.info("Episode %d/%d: %d frames (使用 %d), indices [%d, %d)",
                        i + 1, min(cfg.max_episodes, len(loaded_episodes)),
                        ep_length, ep_length, ep_start, ep_end)
            ep_tasks = ds.meta.episodes.get(ep_idx, {}).get("tasks", [])
            ep_task = ep_tasks[0] if ep_tasks else cfg.task
            logger.info("Task: %s", ep_task)
            logger.info("=" * 60)

            # ---- Episode 初始化 ----
            if action_queue is not None:
                action_queue.clear()
            chunk = None  # 非 RTC 模式用
            chunk_idx = 0
            ep_frame_count = 0
            ep_infer_count = 0
            ep_infer_total_ms = 0.0
            ep_rtt_total_ms = 0.0
            prev_action = None
            inf_thread = None

            # ---- 多线程 RTC 模式：构建 lookup + 启动推理线程 ----
            shared_state: dict = {}
            if rtc_enabled:
                action_lookup = build_action_lookup(ds, ep_start, ep_start + ep_length)
                shared_state = {
                    "stop": threading.Event(),
                    "lock": threading.Lock(),
                    "current_action": None,   # 原始预测动作（closest lookup 用）
                    "latest_frame": -1,       # 当前回放帧索引（默认观测源）
                    "inference_ok": True,
                    "last_infer_result": None,
                    "last_infer_obs_idx": -1,
                    "last_infer_delay": 0,
                    "last_infer_leftover_len": 0,
                    "baseline_result": None,
                    "baseline_roundtrip_ms": 0.0,
                }
                inf_thread = _start_inference_thread(
                    policy=policy,
                    action_queue=action_queue,
                    shared_state=shared_state,
                    ds=ds,
                    action_lookup=action_lookup,
                    camera_keys=camera_keys,
                    state_key=state_key,
                    task=ep_task,
                    inference_rate=cfg.inference_rate,
                    rtc_execution_horizon=cfg.rtc_execution_horizon,
                    bgr_input=cfg.bgr_input,
                    compare_rtc=cfg.compare_rtc,
                    use_closest_lookup=cfg.use_closest_lookup,
                )
                logger.info("推理线程已启动 (rate=%.1f Hz, horizon=%d).",
                             cfg.inference_rate, cfg.rtc_execution_horizon)

            # ---- 主消费循环 ----
            for fi in range(ep_length):
                loop_start = time.monotonic()

                frame_idx = ep_start + fi
                frame = ds[frame_idx]
                _, _, gt_action = extract_observation(frame, camera_keys, state_key)

                # ---- 消费一步 action ----
                predicted = None
                need_infer = False
                queue_size_val = None

                if rtc_enabled:
                    # 多线程 RTC: 从队列消费
                    queue_size_val = action_queue.qsize()
                    act_tensor = action_queue.get()
                    predicted = act_tensor.cpu().numpy() if act_tensor is not None else None
                elif chunk is not None and chunk_idx < len(chunk):
                    predicted = chunk[chunk_idx]
                    chunk_idx += 1
                else:
                    # 非 RTC 同步推理
                    need_infer = True

                # ---- 非 RTC 同步推理（仅 --no_rtc 路径）----
                infer_ms = 0.0
                rtc_roundtrip_ms = 0.0
                inference_delay = None
                leftover_len = None
                rtc_actions_np = None
                rtc_vs_baseline_mse = None
                rtc_vs_baseline_per_step_mse = None
                baseline_infer_ms = 0.0
                baseline_roundtrip_ms = 0.0

                if need_infer and not rtc_enabled:
                    images, state, _ = extract_observation(frame, camera_keys, state_key)
                    payload_images = preprocess_images(images, bgr_input=cfg.bgr_input)
                    payload = {"state": state, "images": payload_images, "prompt": ep_task}

                    t0 = time.monotonic()
                    try:
                        result = policy.infer(payload)
                    except Exception as e:
                        logger.error("Inference 失败 (frame %d): %s", frame_idx, e)
                        metrics.record(
                            episode=ep_idx, frame=fi, global_frame=int(frame_idx),
                            need_infer=True, error=str(e),
                            infer_ms=0.0, roundtrip_ms=0.0, queue_size=queue_size_val,
                        )
                        break
                    rtc_roundtrip_ms = (time.monotonic() - t0) * 1000
                    total_infer_count += 1
                    infer_ms = _extract_infer_ms(result)
                    ep_infer_count += 1
                    ep_infer_total_ms += infer_ms
                    ep_rtt_total_ms += rtc_roundtrip_ms

                    actions = result.get("actions")
                    rtc_actions_np = np.asarray(actions) if (actions is not None and len(actions) > 0) else None
                    if rtc_actions_np is not None:
                        chunk = rtc_actions_np
                        chunk_idx = 0
                        predicted = chunk[chunk_idx] if len(chunk) > 0 else None
                        chunk_idx += 1

                # ---- 消费线程获取推理结果（多线程 RTC 模式）----
                infer_obs_idx = -1
                infer_delay = 0
                infer_leftover = 0
                last_result = None
                bl_result = None
                bl_rtt = 0.0
                if rtc_enabled:
                    with shared_state["lock"]:  # type: ignore[index]
                        last_result = shared_state.get("last_infer_result")  # type: ignore[index]
                        infer_obs_idx = int(shared_state.get("last_infer_obs_idx", -1))  # type: ignore[index]
                        infer_delay = int(shared_state.get("last_infer_delay", 0))  # type: ignore[index]
                        infer_leftover = int(shared_state.get("last_infer_leftover_len", 0))  # type: ignore[index]
                        bl_result = shared_state.get("baseline_result")  # type: ignore[index]
                        bl_rtt = float(shared_state.get("baseline_roundtrip_ms", 0.0))  # type: ignore[index]
                        # 标记已消费（避免同一结果重复记录）
                        shared_state["last_infer_result"] = None  # type: ignore[index]
                        shared_state["baseline_result"] = None  # type: ignore[index]

                    if last_result is not None:
                        ep_infer_count += 1
                        total_infer_count += 1
                        infer_ms = _extract_infer_ms(last_result)
                        inference_delay = infer_delay
                        leftover_len = infer_leftover

                        server_timing = last_result.get("server_timing", {})
                        rtc_roundtrip_ms = server_timing.get("total_ms", 0.0)
                        ep_infer_total_ms += infer_ms
                        ep_rtt_total_ms += rtc_roundtrip_ms

                        # 处理 A/B 对比结果
                        if bl_result is not None and last_result is not None:
                            rtc_a = np.asarray(last_result.get("actions", []))
                            bl_a = np.asarray(bl_result.get("actions", []))
                            if len(rtc_a) > 0 and len(bl_a) > 0:
                                cmp_len = min(len(rtc_a), len(bl_a))
                                diff_chunk = rtc_a[:cmp_len] - bl_a[:cmp_len]
                                rtc_vs_baseline_mse = round(float(np.mean(diff_chunk ** 2)), 8)
                                rtc_vs_baseline_per_step_mse = [
                                    round(float(np.mean(diff_chunk[t] ** 2)), 8)
                                    for t in range(cmp_len)
                                ]
                            baseline_roundtrip_ms = bl_rtt
                            baseline_infer_ms = _extract_infer_ms(bl_result or {})

                # ---- 插值平滑 ----
                smoothed = False
                if predicted is not None:
                    raw_pred = predicted.copy()
                    predicted = smooth_predicted_action(
                        prev_action, predicted, cfg.action_smooth_max_step,
                    )
                    if not np.array_equal(predicted, raw_pred):
                        smoothed = True
                    if rtc_enabled:
                        with shared_state["lock"]:  # type: ignore[index]
                            shared_state["current_action"] = raw_pred  # type: ignore[index] - 原始动作（closest lookup 用）
                            shared_state["latest_frame"] = frame_idx  # type: ignore[index]
                    prev_action = predicted

                # ---- 记录指标 ----
                closest_offset = None
                if rtc_enabled:
                    closest_offset = infer_obs_idx - frame_idx if infer_obs_idx >= 0 else None  # type: ignore[possibly-used-undefined]

                entry: dict = {
                    "episode": int(ep_idx),
                    "frame": int(fi),
                    "global_frame": int(frame_idx),
                    "infer_count": total_infer_count,
                    "need_infer": need_infer,  # 多线程模式始终 False；非 RTC 模式同步推理时为 True
                    "infer_ms": round(float(infer_ms), 2),
                    "roundtrip_ms": round(float(rtc_roundtrip_ms), 2),
                    "inference_delay": int(inference_delay) if inference_delay is not None else None,
                    "leftover_len": int(leftover_len) if leftover_len is not None else None,
                    "chunk_size": int(len(rtc_actions_np)) if rtc_actions_np is not None else None,
                    "queue_size": queue_size_val,
                    "smoothed": smoothed,
                    "closest_frame_offset": closest_offset,
                }

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
                ep_frame_count += 1

                # ---- 进度打印 ----
                if fi % 30 == 0:
                    parts = [
                        f"[ep={ep_idx} f={fi:4d}]",
                        f"inf=#{total_infer_count}",
                    ]
                    if rtc_enabled and queue_size_val is not None:
                        parts.append(f"q={queue_size_val}")
                    if closest_offset is not None:
                        parts.append(f"Δ={closest_offset:+d}")
                    mse_val = entry.get("mse")
                    if mse_val is not None:
                        parts.append(f"mse={mse_val:.6f}")
                    if smoothed:
                        parts.append("sm")
                    logger.info(" ".join(parts))

                log_file.flush()

                # ---- 维持回放 fps ----
                elapsed = time.monotonic() - loop_start
                sleep_time = replay_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            # ---- Episode 清理 ----
            if inf_thread is not None:
                shared_state["stop"].set()  # type: ignore[index]
                inf_thread.join(timeout=10.0)

            if ep_infer_count > 0:
                avg_infer = ep_infer_total_ms / ep_infer_count
                avg_rtt = ep_rtt_total_ms / ep_infer_count
                logger.info(
                    "--- Episode %d 小结: %d frames, %d inferences, "
                    "avg_infer=%.1fms, avg_rtt=%.1fms ---",
                    ep_idx, ep_frame_count, ep_infer_count, avg_infer, avg_rtt,
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

    if "avg_queue_size" in summary:
        logger.info("  --- 多线程指标 ---")
        logger.info("  平均队列深度:      %.1f", summary["avg_queue_size"])
        logger.info("  队列深度范围:      %d - %d",
                     summary.get("min_queue_size", 0), summary.get("max_queue_size", 0))
        logger.info("  平滑比例:          %.1f%%", summary.get("pct_smoothed", 0))
        logger.info("  平均查找偏移:      %.1f frames", summary.get("avg_closest_offset", 0))

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
    parser.add_argument(
        "--task", type=str, default="fold the box",
        help="回退 task prompt，仅当数据集中某 episode 无 task 时使用",
    )

    # 图像
    parser.add_argument(
        "--bgr_input", action="store_true",
        help="输入图像是 BGR（OpenCV 格式），默认 RGB（数据集视频）",
    )

    # 预热
    parser.add_argument(
        "--warmup_rounds", type=int, default=10,
        help="首帧预热推理次数，0 表示跳过 (default: 10)",
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

    # 多线程推理
    parser.add_argument(
        "--inference_rate", type=float, default=3.0,
        help="推理线程频率 Hz (default: 3.0)",
    )
    parser.add_argument(
        "--action_smooth_max_step", type=float, default=0.05,
        help="单步最大关节位移 rad，0 关闭 (default: 0.05)",
    )
    parser.add_argument(
        "--use_closest_lookup", action="store_true",
        help="推理时用 closest-action 查找观测帧（默认关闭，使用当前回放帧）",
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
        warmup_rounds=args.warmup_rounds,
        use_rtc=not args.no_rtc,
        compare_rtc=args.compare_rtc,
        rtc_execution_horizon=args.rtc_execution_horizon,
        inference_rate=args.inference_rate,
        action_smooth_max_step=args.action_smooth_max_step,
        use_closest_lookup=args.use_closest_lookup,
        output_dir=args.output_dir,
    )

    run_smoke_test(cfg)


if __name__ == "__main__":
    main()

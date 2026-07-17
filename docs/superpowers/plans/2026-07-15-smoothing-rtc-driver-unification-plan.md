# Smoothing-RTC 驱动统一化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `StreamActionBuffer` 改造为与 `ActionQueue` 一致的 request-response 驱动，移除 `inference_rate` / `latency_k`，使 Smoothing 与 RTC 仅区别于平滑发生的位置。

**Architecture:** 统一 `StreamActionBuffer` 的 API（`get_action_index()` + `integrate_new_chunk(actions, real_delay, min_m)`），在 `record_body_teaching.py`、`record_dataset.py` 的推理线程中用与 RTC 相同的 request-response 模式驱动 Smoothing，`rl_robot_bridge.py` 同步适配新 API。

**Tech Stack:** Python 3.10+, numpy, ruff, pytest

## Global Constraints

- 所有导入必须在文件顶部完成，禁止函数内联导入。
- 使用双引号字符串；行宽 110 字符。
- 每个 `.py` 文件保留 Apache 2.0 license header。
- 修改后必须运行 `ruff check src/ tests/` 和 `ruff format src/ tests/`。
- Smoothing 的 crossfade 算法必须保留，仅替换丢弃步数的决策来源。
- RTC 分支在本次重构中保持不变。

---

## File Map

| 文件 | 职责 | 变更 |
|------|------|------|
| `src/robodeploy/utils/stream_buffer.py` | `StreamActionBuffer` 核心实现 | API 改造：新增 `get_action_index()`，`integrate_new_chunk` 改为 `(actions, real_delay, min_m)`，移除 `self.k` |
| `tests/utils/test_stream_buffer.py` | `StreamActionBuffer` 单元测试 | 新增，覆盖 delay 丢弃、crossfade、空 buffer、短 chunk |
| `src/robodeploy/scripts/record_body_teaching.py` | body-teaching 录制入口 | Smoothing 分支改为 request-response；移除 `_start_inference_thread` 的 `inference_rate`/`latency_k` 参数 |
| `src/robodeploy/scripts/record_config_body_teaching.py` | body-teaching 配置 | 删除 `inference_rate`、`latency_k` |
| `src/robodeploy/scripts/record_dataset.py` | 标准录制入口 | Smoothing 分支改为 request-response；移除 `_start_inference_thread` 的 `inference_rate`/`latency_k` 参数 |
| `src/robodeploy/scripts/record_config.py` | 标准录制配置 | 删除 `inference_rate`、`latency_k` |
| `src/robodeploy/scripts/rl_robot_bridge.py` | RL 训练桥接 | 删除 `inference_rate`、`latency_k`；用实际执行步数作为 `real_delay` |

---

## Task 1: 改造 `StreamActionBuffer`

**Files:**
- Create: `tests/utils/test_stream_buffer.py`
- Modify: `src/robodeploy/utils/stream_buffer.py`

**Interfaces:**
- Produces:
  - `StreamActionBuffer.get_action_index() -> int`
  - `StreamActionBuffer.integrate_new_chunk(actions_chunk: np.ndarray, real_delay: int, min_m: int = 8) -> None`
  - `StreamActionBuffer.pop_next_action() -> np.ndarray | None`（行为不变，内部不再维护 `self.k`）
  - `StreamActionBuffer.clear() -> None`

- [ ] **Step 1: 创建测试文件**

创建 `tests/utils/test_stream_buffer.py`：

```python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import pytest

from robodeploy.utils.stream_buffer import StreamActionBuffer


def test_get_action_index_initially_zero():
    buffer = StreamActionBuffer(state_dim=2)
    assert buffer.get_action_index() == 0


def test_pop_increments_action_index():
    buffer = StreamActionBuffer(state_dim=2)
    buffer.integrate_new_chunk(np.array([[1.0, 1.0], [2.0, 2.0]]), real_delay=0, min_m=1)
    assert buffer.get_action_index() == 0
    buffer.pop_next_action()
    assert buffer.get_action_index() == 1
    buffer.pop_next_action()
    assert buffer.get_action_index() == 2


def test_real_delay_drops_prefix():
    buffer = StreamActionBuffer(state_dim=2)
    chunk = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    buffer.integrate_new_chunk(chunk, real_delay=2, min_m=1)
    first = buffer.pop_next_action()
    np.testing.assert_array_equal(first, [3.0, 3.0])


def test_real_delay_larger_than_chunk_drops_all():
    buffer = StreamActionBuffer(state_dim=2)
    chunk = np.array([[1.0, 1.0], [2.0, 2.0]])
    buffer.integrate_new_chunk(chunk, real_delay=5, min_m=1)
    assert buffer.pop_next_action() is None


def test_crossfade_preserves_overlap():
    buffer = StreamActionBuffer(state_dim=1)
    # First chunk: 0, 10
    buffer.integrate_new_chunk(np.array([[0.0], [10.0]]), real_delay=0, min_m=2)
    # Pop one action, then integrate a new chunk with real_delay=1.
    buffer.pop_next_action()  # pops 0.0
    # Remaining in buffer: [10.0]. New chunk after dropping first step: [20.0, 30.0].
    # With min_m=2, old_list is padded with 10.0 -> [10.0, 10.0].
    # overlap_len = min(2, 3) = 2.
    # smoothed[0] = 1.0 * 10.0 + 0.0 * 20.0 = 10.0
    # smoothed[1] = 0.5 * 10.0 + 0.5 * 30.0 = 20.0
    buffer.integrate_new_chunk(np.array([[20.0], [30.0], [40.0]]), real_delay=1, min_m=2)
    np.testing.assert_allclose(buffer.pop_next_action(), [10.0], atol=1e-7)
    np.testing.assert_allclose(buffer.pop_next_action(), [20.0], atol=1e-7)
    np.testing.assert_allclose(buffer.pop_next_action(), [40.0], atol=1e-7)


def test_clear_resets_index():
    buffer = StreamActionBuffer(state_dim=2)
    buffer.integrate_new_chunk(np.array([[1.0, 1.0], [2.0, 2.0]]), real_delay=0, min_m=1)
    buffer.pop_next_action()
    assert buffer.get_action_index() == 1
    buffer.clear()
    assert buffer.get_action_index() == 0
    assert buffer.pop_next_action() is None
```

- [ ] **Step 2: 运行测试确认失败**

Run:
```bash
pytest tests/utils/test_stream_buffer.py -v
```

Expected: 全部 FAIL，因为 `get_action_index` 不存在，`integrate_new_chunk` 不接受 `real_delay`。

- [ ] **Step 3: 修改 `StreamActionBuffer` 实现**

修改 `src/robodeploy/utils/stream_buffer.py`，替换 `__init__`、`integrate_new_chunk`、`pop_next_action`、`clear`：

```python
class StreamActionBuffer:
    """Sliding-window action chunk buffer with linear overlap blending.

    When a new action chunk arrives from the policy server, it overlaps
    with the tail of the previous chunk. The overlap region is blended
    using linear weights (100% old → 0% old) for smooth transitions.

    Args:
        state_dim: Dimension of the action vector (e.g., 14 for bimanual).
    """

    def __init__(self, state_dim: int = 14):
        self.lock = threading.Lock()
        self.state_dim = state_dim
        self.cur_chunk: deque = deque()
        self.last_action: np.ndarray | None = None
        self._action_index: int = 0

    def get_action_index(self) -> int:
        """Return the number of actions already popped for execution."""
        with self.lock:
            return self._action_index

    def integrate_new_chunk(
        self,
        actions_chunk: np.ndarray,
        real_delay: int,
        min_m: int = 8,
    ) -> None:
        """Integrate a new action chunk with temporal smoothing.

        Args:
            actions_chunk: New action chunk [N, state_dim].
            real_delay: Steps already executed since the observation was sent;
                drop this many steps from the front of the new chunk.
            min_m: Minimum overlap length for smoothing.
        """
        with self.lock:
            if actions_chunk is None or len(actions_chunk) == 0:
                return
            real_delay = max(0, int(real_delay))
            min_m = max(1, int(min_m))
            drop_n = min(real_delay, len(actions_chunk))
            new_chunk = [a.copy() for a in actions_chunk[drop_n:]]

            if len(self.cur_chunk) == 0 and self.last_action is not None:
                old_list = [np.asarray(self.last_action, dtype=float).copy() for _ in range(min_m)]
                self.last_action = None
            else:
                old_list = list(self.cur_chunk)
                if len(old_list) > 0 and len(old_list) < min_m:
                    tail = np.asarray(old_list[-1], dtype=float).copy()
                    old_list.extend([tail.copy() for _ in range(min_m - len(old_list))])
                elif len(old_list) == 0:
                    self.cur_chunk = deque(new_chunk, maxlen=None)
                    return

            overlap_len = min(len(old_list), len(new_chunk))
            if overlap_len <= 0:
                self.cur_chunk = deque(new_chunk, maxlen=None)
                return
            if len(old_list) > len(new_chunk):
                old_list = old_list[: len(new_chunk)]
                overlap_len = len(new_chunk)

            w_old = np.array([1.0]) if overlap_len == 1 else np.linspace(1.0, 0.0, overlap_len)
            w_new = 1.0 - w_old
            smoothed = [
                w_old[i] * np.asarray(old_list[i], dtype=float)
                + w_new[i] * np.asarray(new_chunk[i], dtype=float)
                for i in range(overlap_len)
            ]
            self.cur_chunk = deque([a.copy() for a in smoothed + new_chunk[overlap_len:]], maxlen=None)

    def pop_next_action(self) -> np.ndarray | None:
        """Pop and return the next action to execute.

        Returns:
            Action vector [state_dim,] or None if buffer is empty.
        """
        with self.lock:
            if len(self.cur_chunk) == 0:
                return None
            if len(self.cur_chunk) == 1:
                self.last_action = np.asarray(self.cur_chunk[0], dtype=float).copy()
            act = np.asarray(self.cur_chunk.popleft(), dtype=float)
            self._action_index += 1
            return act

    def clear(self) -> None:
        """Clear the buffer and reset state."""
        with self.lock:
            self.cur_chunk.clear()
            self.last_action = None
            self._action_index = 0
```

- [ ] **Step 4: 运行测试确认通过**

Run:
```bash
pytest tests/utils/test_stream_buffer.py -v
```

Expected: 6 tests PASS。

- [ ] **Step 5: 提交**

```bash
git add src/robodeploy/utils/stream_buffer.py tests/utils/test_stream_buffer.py
git commit -m "refactor: StreamActionBuffer API unified with ActionQueue driver

- Add get_action_index() for real-delay computation.
- Replace max_k/self.k with explicit real_delay parameter.
- Keep linear crossfade blending unchanged."
```

---

## Task 2: 改造 `record_body_teaching.py`

**Files:**
- Modify: `src/robodeploy/scripts/record_body_teaching.py`
- Modify: `src/robodeploy/scripts/record_config_body_teaching.py`

**Interfaces:**
- Consumes: `StreamActionBuffer.get_action_index()`、`StreamActionBuffer.integrate_new_chunk(actions, real_delay, min_m)`
- Produces: `_start_inference_thread(...)` 不再接收 `inference_rate`、`latency_k`

- [ ] **Step 1: 修改 `_start_inference_thread` 函数签名与 Smoothing 分支**

在 `src/robodeploy/scripts/record_body_teaching.py` 中：

1. 修改函数签名，删除 `inference_rate: float` 和 `latency_k: int`：

```python
def _start_inference_thread(
    policy,
    buffer: StreamActionBuffer | None,
    state_ref: dict,
    recording_ref: dict,
    action_features: dict[str, type],
    camera_names: list[str],
    task: str,
    min_smooth_steps: int,
    action_queue: "ActionQueue | None" = None,
    rtc_execution_horizon: int = 10,
) -> threading.Thread:
```

2. 修改函数体，把 Smoothing 分支替换为 request-response 驱动：

原代码（约 168-195 行）：
```python
            # Smoothing mode: rate-limited polling
            if action_queue is None:
                was_recording = True
                obs = state_ref.get("obs")
                if obs is None:
                    time.sleep(0.1)
                    continue

                try:
                    state = np.array([obs.get(k, 0.0) for k in action_features], dtype=np.float64)
                    images = {cam: np.asarray(obs[cam]) for cam in camera_names if cam in obs}
                    _stack_front_cameras(images)

                    if buffer is not None:
                        result = policy.infer(images, state, task)
                        actions = result.get("actions", None)
                        if actions is not None and len(actions) > 0:
                            buffer.integrate_new_chunk(
                                np.asarray(actions),
                                max_k=latency_k,
                                min_m=min_smooth_steps,
                            )
                    state_ref["inference_ok"] = True
                except Exception as e:
                    logger.warning(f"Inference error: {e}")
                    state_ref["inference_ok"] = False
                time.sleep(rate)
                continue
```

替换为：
```python
            # Smoothing mode: request-response driver, same as RTC
            if action_queue is None:
                was_recording = True
                obs = state_ref.get("obs")
                if obs is None:
                    time.sleep(0.1)
                    continue

                try:
                    state = np.array([obs.get(k, 0.0) for k in action_features], dtype=np.float64)
                    images = {cam: np.asarray(obs[cam]) for cam in camera_names if cam in obs}
                    _stack_front_cameras(images)

                    if buffer is not None:
                        send_index = buffer.get_action_index()
                        result = policy.infer(images, state, task)
                        actions = result.get("actions", None)
                        if actions is not None and len(actions) > 0:
                            wait_steps = buffer.get_action_index() - send_index
                            buffer.integrate_new_chunk(
                                np.asarray(actions),
                                real_delay=wait_steps,
                                min_m=min_smooth_steps,
                            )
                    state_ref["inference_ok"] = True
                except Exception as e:
                    logger.warning(f"Inference error: {e}")
                    state_ref["inference_ok"] = False
                continue
```

3. 删除 `_run()` 中不再使用的 `rate = 1.0 / inference_rate`。

- [ ] **Step 2: 修改 `run_record` 中的调用**

找到 `_start_inference_thread(...)` 调用处（约 537-550 行），删除 `inference_rate` 和 `latency_k` 参数：

```python
        inference_thread = _start_inference_thread(
            policy=policy,
            buffer=stream_buffer,
            state_ref=state_ref,
            recording_ref=recording_ref,
            action_features=action_features,
            camera_names=camera_names,
            task=cfg.task,
            min_smooth_steps=cfg.min_smooth_steps,
            action_queue=action_queue,
            rtc_execution_horizon=cfg.rtc_execution_horizon,
        )
```

- [ ] **Step 3: 修改 `RecordBodyTeachingConfig`**

在 `src/robodeploy/scripts/record_config_body_teaching.py` 中：

```python
    # Temporal smoothing (ignored when use_rtc=True)
    use_temporal_smoothing: bool = True
    min_smooth_steps: int = 8
```

删除 `inference_rate: float = 3.0` 和 `latency_k: int = 8`。

- [ ] **Step 4: 运行 lint 检查**

Run:
```bash
ruff check src/robodeploy/scripts/record_body_teaching.py src/robodeploy/scripts/record_config_body_teaching.py
ruff format src/robodeploy/scripts/record_body_teaching.py src/robodeploy/scripts/record_config_body_teaching.py
```

Expected: 无错误。

- [ ] **Step 5: 提交**

```bash
git add src/robodeploy/scripts/record_body_teaching.py src/robodeploy/scripts/record_config_body_teaching.py
git commit -m "refactor: record_body_teaching Smoothing uses RTC-style request-response driver

- Remove inference_rate and latency_k.
- Compute real_delay from actual executed steps via buffer.get_action_index()."
```

---

## Task 3: 改造 `record_dataset.py`

**Files:**
- Modify: `src/robodeploy/scripts/record_dataset.py`
- Modify: `src/robodeploy/scripts/record_config.py`

**Interfaces:**
- Consumes: `StreamActionBuffer.get_action_index()`、`StreamActionBuffer.integrate_new_chunk(actions, real_delay, min_m)`
- Produces: `_start_inference_thread(...)` 不再接收 `inference_rate`、`latency_k`

- [ ] **Step 1: 修改 `_start_inference_thread` 函数签名与 Smoothing 分支**

在 `src/robodeploy/scripts/record_dataset.py` 中：

1. 修改函数签名，删除 `inference_rate: float` 和 `latency_k: int`：

```python
def _start_inference_thread(
    policy,
    buffer: StreamActionBuffer | None,
    state_ref: dict,
    recording_ref: dict,
    action_features: dict[str, type],
    camera_names: list[str],
    task: str,
    min_smooth_steps: int,
    action_queue: "ActionQueue | None" = None,
    rtc_execution_horizon: int = 10,
) -> threading.Thread:
```

2. 修改函数体，把 Smoothing 分支替换为 request-response 驱动：

原代码（约 179-203 行）：
```python
            # Smoothing mode: rate-limited polling
            if action_queue is None:
                was_recording = True
                obs = state_ref.get("obs")
                if obs is None:
                    time.sleep(0.1)
                    continue

                try:
                    state, images = _prepare_inference_input(obs, action_features, camera_names)
                    if buffer is not None:
                        result = policy.infer(images, state, task)
                        actions = result.get("actions", None)
                        if actions is not None and len(actions) > 0:
                            buffer.integrate_new_chunk(
                                np.asarray(actions),
                                max_k=latency_k,
                                min_m=min_smooth_steps,
                            )
                    state_ref["inference_ok"] = True
                except Exception as e:
                    logger.warning(f"Inference error: {e}")
                    state_ref["inference_ok"] = False
                time.sleep(rate)
                continue
```

替换为：
```python
            # Smoothing mode: request-response driver, same as RTC
            if action_queue is None:
                was_recording = True
                obs = state_ref.get("obs")
                if obs is None:
                    time.sleep(0.1)
                    continue

                try:
                    state, images = _prepare_inference_input(obs, action_features, camera_names)
                    if buffer is not None:
                        send_index = buffer.get_action_index()
                        result = policy.infer(images, state, task)
                        actions = result.get("actions", None)
                        if actions is not None and len(actions) > 0:
                            wait_steps = buffer.get_action_index() - send_index
                            buffer.integrate_new_chunk(
                                np.asarray(actions),
                                real_delay=wait_steps,
                                min_m=min_smooth_steps,
                            )
                    state_ref["inference_ok"] = True
                except Exception as e:
                    logger.warning(f"Inference error: {e}")
                    state_ref["inference_ok"] = False
                continue
```

3. 删除 `_run()` 中不再使用的 `rate = 1.0 / inference_rate`。

- [ ] **Step 2: 修改 `run_record` 中的调用**

找到 `_start_inference_thread(...)` 调用处，删除 `inference_rate` 和 `latency_k` 参数：

```python
        inference_thread = _start_inference_thread(
            policy=policy,
            buffer=stream_buffer,
            state_ref=state_ref,
            recording_ref=recording_ref,
            action_features=action_features,
            camera_names=camera_names,
            task=cfg.task,
            min_smooth_steps=cfg.min_smooth_steps,
            action_queue=action_queue,
            rtc_execution_horizon=cfg.rtc_execution_horizon,
        )
```

- [ ] **Step 3: 修改 `RecordConfig`**

在 `src/robodeploy/scripts/record_config.py` 中：

```python
    # Temporal smoothing (ignored when use_rtc=True)
    use_temporal_smoothing: bool = True
    min_smooth_steps: int = 8
```

删除 `inference_rate: float = 3.0` 和 `latency_k: int = 8`。

- [ ] **Step 4: 运行 lint 检查**

Run:
```bash
ruff check src/robodeploy/scripts/record_dataset.py src/robodeploy/scripts/record_config.py
ruff format src/robodeploy/scripts/record_dataset.py src/robodeploy/scripts/record_config.py
```

Expected: 无错误。

- [ ] **Step 5: 提交**

```bash
git add src/robodeploy/scripts/record_dataset.py src/robodeploy/scripts/record_config.py
git commit -m "refactor: record_dataset Smoothing uses RTC-style request-response driver

- Remove inference_rate and latency_k.
- Compute real_delay from actual executed steps via buffer.get_action_index()."
```

---

## Task 4: 改造 `rl_robot_bridge.py`

**Files:**
- Modify: `src/robodeploy/scripts/rl_robot_bridge.py`

**Interfaces:**
- Consumes: `StreamActionBuffer.get_action_index()`、`StreamActionBuffer.integrate_new_chunk(actions, real_delay, min_m)`
- Produces: `RLBridgeConfig` 不再包含 `inference_rate`、`latency_k`

- [ ] **Step 1: 修改 `RLBridgeConfig`**

在 `src/robodeploy/scripts/rl_robot_bridge.py` 中，将配置段：

```python
    # Temporal smoothing (StreamActionBuffer)
    latency_k: int = 2  # drop first k steps of new chunk for latency compensation
    min_smooth_steps: int = 8  # minimum overlap length for linear crossfade
    inference_rate: float = 7.0  # policy inference frequency (Hz), controls chunk overlap depth
```

改为：

```python
    # Temporal smoothing (StreamActionBuffer)
    min_smooth_steps: int = 8  # minimum overlap length for linear crossfade
```

- [ ] **Step 2: 修改启动输出与 `steps_per_inference` 计算**

原代码（约 263-265 行）：
```python
    steps_per_inference = max(1, int(cfg.fps / cfg.inference_rate))
    print(f"  Smoothing: latency_k={cfg.latency_k}  min_smooth={cfg.min_smooth_steps}  "
          f"inf_rate={cfg.inference_rate}Hz  →  {steps_per_inference} steps/inference")
```

改为：
```python
    planned_steps = max(1, cfg.chunk_length - cfg.min_smooth_steps)
    print(f"  Smoothing: min_smooth={cfg.min_smooth_steps}  "
          f"planned_steps={planned_steps}")
```

- [ ] **Step 3: 引入 `prev_executed_steps` 并修改 Normal mode 执行循环**

在 `run_bridge` 函数开头（约 246 行 `chunk_idx = 0` 附近）新增状态变量：

```python
    chunk_idx = 0
    prev_executed_steps = 0  # 用于下一轮 integrate 的 real_delay
```

原 Normal mode 代码（约 407-469 行）：

```python
            else:
                # Normal mode: use Training PC action chunk
                chosen = np.asarray(chosen_actions)  # [C, action_dim]
                action_buffer.integrate_new_chunk(
                    chosen,
                    max_k=cfg.latency_k,
                    min_m=cfg.min_smooth_steps,
                )
                actual_action_chunk = None  # Training PC knows what it sent

                # ---- Execute action chunk ----
                done = False
                success = False

                for k in range(steps_per_inference):
                    act_np = action_buffer.pop_next_action()
                    if act_np is None:
                        break
                    t_start = time.time()

                    action_dict = _numpy_to_action_dict(act_np, action_features)
                    robot.send_action(action_dict)

                    # Check for human reward / toggle signal
                    key = _check_keypress(extra_keys=cfg.intervention_key + cfg.rl_toggle_key)
                    if key == "\x1b" or key == "\x03":
                        logger.info("Esc/Ctrl-C: stopping bridge")
                        _stop = True
                        break
                    elif key == "f":
                        done = True
                        success = False
                        action_buffer.clear()
                        logger.info("FAILURE — episode %d, chunk %d, step %d", episode, chunk_idx, k)
                        break
                    elif key == "s":
                        reward += 1.0
                        done = True
                        success = True
                        action_buffer.clear()
                        logger.info("SUCCESS — episode %d (reward=%.3f)", episode, reward)
                        break
                    elif key == cfg.rl_toggle_key:
                        rl_active = not rl_active
                        logger.info("RL toggled %s (chunk %d, step %d)",
                                    "ON" if rl_active else "OFF", chunk_idx, k)
                    elif key == cfg.intervention_key and teleop is not None:
                        action_buffer.clear()
                        # Align leader to follower before takeover to avoid joint jumps
                        logger.info("Aligning leader to follower before intervention...")
                        leader_pos = teleop.get_action()
                        follower_pos = robot.get_observation()
                        interpolate_leader_to_follower(
                            teleop, leader_pos, follower_pos,
                            action_features, dt=0.05, max_step=0.02,
                        )
                        intervention_active = True
                        logger.info("Intervention ON — teleoperator takeover")
                        break

                    # Enforce control frequency
                    elapsed = time.time() - t_start
                    sleep_t = control_period - elapsed
                    if sleep_t > 0:
                        time.sleep(sleep_t)
```

改为：

```python
            else:
                # Normal mode: use Training PC action chunk
                chosen = np.asarray(chosen_actions)  # [C, action_dim]
                actual_action_chunk = None  # Training PC knows what it sent

                # Integrate the new chunk, dropping the steps actually executed
                # since the previous observation was sent.
                action_buffer.integrate_new_chunk(
                    chosen,
                    real_delay=prev_executed_steps,
                    min_m=cfg.min_smooth_steps,
                )

                # ---- Execute action chunk ----
                done = False
                success = False
                executed_steps = 0

                for k in range(planned_steps):
                    act_np = action_buffer.pop_next_action()
                    if act_np is None:
                        break
                    t_start = time.time()

                    action_dict = _numpy_to_action_dict(act_np, action_features)
                    robot.send_action(action_dict)
                    executed_steps += 1

                    # Check for human reward / toggle signal
                    key = _check_keypress(extra_keys=cfg.intervention_key + cfg.rl_toggle_key)
                    if key == "\x1b" or key == "\x03":
                        logger.info("Esc/Ctrl-C: stopping bridge")
                        _stop = True
                        break
                    elif key == "f":
                        done = True
                        success = False
                        action_buffer.clear()
                        prev_executed_steps = 0
                        logger.info("FAILURE — episode %d, chunk %d, step %d", episode, chunk_idx, k)
                        break
                    elif key == "s":
                        reward += 1.0
                        done = True
                        success = True
                        action_buffer.clear()
                        prev_executed_steps = 0
                        logger.info("SUCCESS — episode %d (reward=%.3f)", episode, reward)
                        break
                    elif key == cfg.rl_toggle_key:
                        rl_active = not rl_active
                        logger.info("RL toggled %s (chunk %d, step %d)",
                                    "ON" if rl_active else "OFF", chunk_idx, k)
                    elif key == cfg.intervention_key and teleop is not None:
                        action_buffer.clear()
                        prev_executed_steps = 0
                        # Align leader to follower before takeover to avoid joint jumps
                        logger.info("Aligning leader to follower before intervention...")
                        leader_pos = teleop.get_action()
                        follower_pos = robot.get_observation()
                        interpolate_leader_to_follower(
                            teleop, leader_pos, follower_pos,
                            action_features, dt=0.05, max_step=0.02,
                        )
                        intervention_active = True
                        logger.info("Intervention ON — teleoperator takeover")
                        break

                    # Enforce control frequency
                    elapsed = time.time() - t_start
                    sleep_t = control_period - elapsed
                    if sleep_t > 0:
                        time.sleep(sleep_t)

                prev_executed_steps = executed_steps
```

注意：
- `action_buffer.integrate_new_chunk` 仍位于执行循环之前，但 `real_delay` 使用上一轮实际执行步数 `prev_executed_steps`。
- 当 buffer 被 `clear()`（失败、成功、人工干预、reset）时，`prev_executed_steps` 必须同步重置为 0。
- reset 分支（约 297-319 行）中已有 `action_buffer.clear()`，需在同一处添加 `prev_executed_steps = 0`：

```python
            if reset_cmd:
                logger.info("Reset command received — moving arms to zero")
                action_buffer.clear()
                prev_executed_steps = 0
                reset_to_zero(robot, teleop=teleop, action_features=action_features)
```

- intervention 结束后的循环（约 401 行）也有 `action_buffer.clear()`，需在同一处添加 `prev_executed_steps = 0`：

```python
                action_buffer.clear()
                prev_executed_steps = 0
                actual_action_chunk = None  # 数据已在循环内发送
```

- [ ] **Step 4: 检查 `steps_per_inference` 是否还有残留引用**

Run:
```bash
grep -n "steps_per_inference\|inference_rate\|latency_k" src/robodeploy/scripts/rl_robot_bridge.py
```

Expected: 无输出。

- [ ] **Step 5: 运行 lint 检查**

Run:
```bash
ruff check src/robodeploy/scripts/rl_robot_bridge.py
ruff format src/robodeploy/scripts/rl_robot_bridge.py
```

Expected: 无错误。

- [ ] **Step 6: 提交**

```bash
git add src/robodeploy/scripts/rl_robot_bridge.py
git commit -m "refactor: rl_robot_bridge adapts to new StreamActionBuffer API

- Remove inference_rate and latency_k from RLBridgeConfig.
- Use planned_steps = chunk_length - min_smooth_steps.
- Track prev_executed_steps as real_delay for integrate_new_chunk.
- Reset prev_executed_steps whenever action_buffer is cleared."
```

---

## Task 5: 最终验证

**Files:**
- 全局检查

- [ ] **Step 1: 运行全部新增与现有测试**

Run:
```bash
pytest tests/utils/test_stream_buffer.py -v
```

Expected: 6 tests PASS。

- [ ] **Step 2: 全量 lint 与 format**

Run:
```bash
ruff check src/ tests/
ruff format src/ tests/
```

Expected: 无错误。

- [ ] **Step 3: 检查无残留引用**

Run:
```bash
grep -R "inference_rate\|latency_k" src/robodeploy/scripts/record_body_teaching.py \
    src/robodeploy/scripts/record_config_body_teaching.py \
    src/robodeploy/scripts/record_dataset.py \
    src/robodeploy/scripts/record_config.py \
    src/robodeploy/scripts/rl_robot_bridge.py \
    src/robodeploy/utils/stream_buffer.py
```

Expected: 无输出（无残留）。

- [ ] **Step 4: 提交最终验证结果（可选）**

```bash
git commit --allow-empty -m "chore: verify Smoothing-RTC driver unification passes lint and tests"
```

---

## Self-Review Checklist

- [x] Spec coverage: 所有设计点（API 改造、record_body_teaching、record_dataset、rl_robot_bridge、配置删除、测试）均已对应到任务。
- [x] Placeholder scan: 无 TBD/TODO/"implement later"/"similar to Task N"。
- [x] Type consistency:
  - `StreamActionBuffer.get_action_index() -> int`
  - `StreamActionBuffer.integrate_new_chunk(actions_chunk: np.ndarray, real_delay: int, min_m: int = 8) -> None`
  - `_start_inference_thread(..., min_smooth_steps: int, ...)` 已删除 `inference_rate`/`latency_k`
- [x] DRY: 两个 record 脚本的改造模式相同，但各自独立成 Task，避免跨文件耦合。
- [x] YAGNI: 不修改 RTC 分支、不改动 `ActionQueue`、不新增无关配置。

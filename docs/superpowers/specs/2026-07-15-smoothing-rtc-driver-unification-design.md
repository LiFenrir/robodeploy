# Smoothing 与 RTC 驱动统一化设计

**日期**: 2026-07-15  
**范围**: `src/robodeploy/rtc`、`src/robodeploy/utils/stream_buffer.py`、`src/robodeploy/scripts/record_body_teaching.py`、`src/robodeploy/scripts/record_dataset.py`、`src/robodeploy/scripts/rl_robot_bridge.py`  
**目标**: 让 Smoothing 模式与 RTC 模式共用同一套 request-response 异步驱动，区别仅在于平滑发生的位置：RTC 在服务端做约束平滑，Smoothing 在部署端做算术 crossfade。

---

## 1. 现状

### 1.1 RTC 模式（已统一）

- 推理线程采用**收发驱动**：发送观测 → 阻塞等待 `policy.infer()` 返回 → 立即处理结果。
- 新 chunk 到来时，按**实际已执行步数**丢弃前缀：
  ```python
  send_index = action_queue.get_action_index()
  result = policy.infer(...)
  wait_steps = action_queue.get_action_index() - send_index
  action_queue.merge(..., real_delay=wait_steps)
  ```
- 重叠混合由 `ActionQueue._replace_actions_queue()` 完成。

### 1.2 Smoothing 模式（待改造）

- 推理线程采用**轮询 + 限速**：以固定 `inference_rate` 调用 `policy.infer()`，调用间隔由 `time.sleep(rate)` 控制。
- 丢弃步数由内部计数器 `self.k` 与外部参数 `latency_k` 共同决定：
  ```python
  drop_n = min(self.k, max_k)
  ```
- 重叠混合由 `StreamActionBuffer.integrate_new_chunk()` 完成，权重为线性 `old → new`。

### 1.3 问题

- 两种模式的**异步驱动逻辑不同**：RTC 是事件驱动，Smoothing 是定时轮询。
- Smoothing 的丢弃步数是**预设上限**，不是本轮推理期间实际执行的步数，导致延迟补偿与真实执行脱节。
- `inference_rate` 与 `latency_k` 属于 legacy 参数，在 request-response 驱动下不再需要。

---

## 2. 设计目标

1. **统一驱动方式**：Smoothing 改为与 RTC 完全一致的 request-response 收发驱动。
2. **统一丢弃依据**：丢弃步数由实际执行指定，即 `wait_steps = current_index - send_index`。
3. **保留混合逻辑**：`StreamActionBuffer` 的线性 crossfade 算法保持不变，仅替换丢弃决策来源。
4. **删除 legacy 参数**：移除 `inference_rate` 和 `latency_k`。
5. **统一调用方 API**：所有使用 `StreamActionBuffer` 的脚本都按新 API 调用。

---

## 3. API 设计

### 3.1 `StreamActionBuffer`

当前签名：

```python
class StreamActionBuffer:
    def __init__(self, state_dim: int = 14): ...
    def integrate_new_chunk(self, actions_chunk: np.ndarray, max_k: int = 8, min_m: int = 8) -> None: ...
    def pop_next_action(self) -> np.ndarray | None: ...
    def clear(self) -> None: ...
```

新签名：

```python
class StreamActionBuffer:
    def __init__(self, state_dim: int = 14): ...
    def get_action_index(self) -> int: ...
    def integrate_new_chunk(self, actions_chunk: np.ndarray, real_delay: int, min_m: int = 8) -> None: ...
    def pop_next_action(self) -> np.ndarray | None: ...
    def clear(self) -> None: ...
```

语义变更：

- 新增 `get_action_index()`：返回已 `pop_next_action()` 出去的动作总数，与 `ActionQueue.get_action_index()` 对齐。
- `integrate_new_chunk` 的 `max_k` 参数改为 `real_delay`：
  - 调用方必须传入本轮推理期间实际执行的步数。
  - 内部丢弃计算：`drop_n = min(real_delay, len(actions_chunk))`。
- 移除内部 `self.k` 计数器：`pop_next_action()` 不再维护 `self.k`。
- `min_m` 保留：最小重叠长度，用于 crossfade。

### 3.2 实现要点

```python
def integrate_new_chunk(self, actions_chunk: np.ndarray, real_delay: int, min_m: int = 8) -> None:
    with self.lock:
        if actions_chunk is None or len(actions_chunk) == 0:
            return
        real_delay = max(0, int(real_delay))
        min_m = max(1, int(min_m))
        drop_n = min(real_delay, len(actions_chunk))
        new_chunk = [a.copy() for a in actions_chunk[drop_n:]]

        # 以下 crossfade 逻辑保持不变
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
        # ... crossfade 计算不变
```

---

## 4. Inference 线程改造

### 4.1 `record_body_teaching.py` / `record_dataset.py`

`_start_inference_thread` 函数签名变更：

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

移除 `inference_rate: float` 和 `latency_k: int`。

Smoothing 分支改为 request-response：

```python
# Smoothing mode: request-response driver, same as RTC
was_recording = True
obs = state_ref.get("obs")
if obs is None:
    time.sleep(0.1)
    continue

try:
    state, images = _prepare_inference_input(obs, action_features, camera_names)

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
```

RTC 分支保持不变。

### 4.2 数据流对比

| 阶段 | RTC 模式 | Smoothing 模式（改造后） |
|------|----------|------------------------|
| 发送观测前 | `send_index = action_queue.get_action_index()` | `send_index = buffer.get_action_index()` |
| 推理 | `result = policy.infer(..., prev_chunk_left_over=...)` | `result = policy.infer(images, state, task)` |
| 计算实际延迟 | `wait_steps = action_queue.get_action_index() - send_index` | `wait_steps = buffer.get_action_index() - send_index` |
| 合并新 chunk | `action_queue.merge(..., real_delay=wait_steps)` | `buffer.integrate_new_chunk(..., real_delay=wait_steps)` |
| 平滑位置 | 服务端（RTC guidance + ActionQueue crossfade） | 部署端（StreamActionBuffer crossfade） |

---

## 5. RL Bridge 改造

`src/robodeploy/scripts/rl_robot_bridge.py` 本身已是 request-response 驱动，但使用 `inference_rate` 计算每轮执行步数 `steps_per_inference`。

### 5.1 配置变更

- 删除 `inference_rate: float`。
- 删除 `latency_k: int`。
- 保留 `min_smooth_steps: int`。

### 5.2 执行步数规划

原逻辑：

```python
steps_per_inference = max(1, int(cfg.fps / cfg.inference_rate))
```

新逻辑：用 chunk 长度减去最小重叠步数，保证每轮请求之间仍有 `min_smooth_steps` 步可用于 crossfade：

```python
planned_steps = max(1, cfg.chunk_length - cfg.min_smooth_steps)
```

如果 `chunk_length <= min_smooth_steps`，则计划执行 1 步，依赖 `StreamActionBuffer` 自动用最后一帧补齐重叠区。

### 5.3 `integrate_new_chunk` 调用

在执行循环中统计**实际执行步数**（人工干预、成功/失败按键会提前中断循环）：

```python
executed_steps = 0
for k in range(planned_steps):
    act_np = action_buffer.pop_next_action()
    if act_np is None:
        break
    # ... send_action + key check ...
    executed_steps += 1

# 下一轮 integrate 时，按实际执行步数丢弃前缀
action_buffer.integrate_new_chunk(
    chosen,
    real_delay=executed_steps,
    min_m=cfg.min_smooth_steps,
)
```

注意：RL bridge 的协议与 policy server 不同，不传递 `prev_chunk_left_over` 等 RTC 字段，因此 Smoothing 仅做部署端 crossfade。

---

## 6. 配置项变更

### 6.1 `RecordConfig` (`record_config.py`)

```python
# 变更前
use_temporal_smoothing: bool = True
inference_rate: float = 3.0
latency_k: int = 8
min_smooth_steps: int = 8

# 变更后
use_temporal_smoothing: bool = True
min_smooth_steps: int = 8
```

### 6.2 `RecordBodyTeachingConfig` (`record_config_body_teaching.py`)

同上，删除 `inference_rate` 和 `latency_k`。

### 6.3 `RLBridgeConfig` (`rl_robot_bridge.py`)

```python
# 变更前
latency_k: int = 2
min_smooth_steps: int = 8
inference_rate: float = 7.0

# 变更后
min_smooth_steps: int = 8
```

---

## 7. 调用点清单

| 文件 | 当前调用 | 改造后调用 |
|------|----------|-----------|
| `src/robodeploy/utils/stream_buffer.py` | `integrate_new_chunk(actions, max_k=k, min_m=m)` | `integrate_new_chunk(actions, real_delay=d, min_m=m)` |
| `src/robodeploy/scripts/record_body_teaching.py` | `_start_inference_thread(..., inference_rate=..., latency_k=...)` | 移除这两个参数；Smoothing 分支用 request-response |
| `src/robodeploy/scripts/record_dataset.py` | 同上 | 同上 |
| `src/robodeploy/scripts/rl_robot_bridge.py` | `integrate_new_chunk(chosen, max_k=latency_k, min_m=min_smooth_steps)` | `integrate_new_chunk(chosen, real_delay=steps_per_inference, min_m=min_smooth_steps)` |

---

## 8. 边界条件

1. **首轮推理**：`send_index = 0`，`wait_steps = 0`，不丢弃新 chunk 前缀。
2. **推理失败**：异常被捕获，`state_ref["inference_ok"]` 置 `False`，下一轮重新发送观测，不补偿丢失的步数。
3. **模式切换 / 停止录制**：调用 `buffer.clear()`，`get_action_index()` 重置为 0。
4. **新 chunk 短于 `real_delay`**：`drop_n = min(real_delay, len(chunk))`，丢弃后可能为空，直接返回。
5. **buffer 为空时 integrate**：退化为直接保存新 chunk，crossfade 不生效。

---

## 9. 风险与回退

- **风险 1**：移除 `inference_rate` 后，如果 policy server 返回速度极快，推理线程会连续发送请求，可能对服务端造成压力。缓解：request-response 本身会自然限速（服务端处理时间），且主控制循环仍以 `fps` 消费动作。
- **风险 2**：RL bridge 的 `steps_per_inference = chunk_length - min_smooth_steps` 会改变原有行为。如果训练端期望固定频率，需要额外调整。本设计假设 RL bridge 与 record 脚本采用一致的 overlap 语义。
- **回退**：若新行为不稳定，可临时将 `StreamActionBuffer` 的 `real_delay` 默认设为 0，恢复为不丢弃前缀的旧行为，同时保留新 API。

---

## 10. 验收标准

- [ ] `StreamActionBuffer` 新 API 通过单元测试：不同 `real_delay`、`min_m`、空 buffer、短 chunk 等场景。
- [ ] `record_body_teaching.py` Smoothing 分支走 request-response，不再使用 `time.sleep(rate)`。
- [ ] `record_dataset.py` Smoothing 分支同样改造。
- [ ] `rl_robot_bridge.py` 使用新 API，`inference_rate` 和 `latency_k` 不再出现。
- [ ] 配置文件中删除 `inference_rate` 和 `latency_k`。
- [ ] `ruff check` 与 `ruff format` 通过。

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

from robodeploy.utils.stream_buffer import StreamActionBuffer


def test_get_action_index_initially_zero():
    buffer = StreamActionBuffer(state_dim=2)
    assert buffer.get_action_index() == 0


def test_pop_increments_action_index():
    buffer = StreamActionBuffer(state_dim=2)
    buffer.integrate_new_chunk(np.array([[1.0, 1.0], [2.0, 2.0]]), max_k=8, min_m=1)
    assert buffer.get_action_index() == 0
    buffer.pop_next_action()
    assert buffer.get_action_index() == 1
    buffer.pop_next_action()
    assert buffer.get_action_index() == 2


def test_max_k_drops_prefix_after_consumption():
    """max_k 配合 self.k 使用：先消费 N 步，再 integrate 时自动丢弃 min(self.k, max_k) 步。"""
    buffer = StreamActionBuffer(state_dim=2)
    chunk = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    buffer.integrate_new_chunk(chunk, max_k=8, min_m=1)
    # 消费 2 步
    buffer.pop_next_action()  # [1.0, 1.0]
    buffer.pop_next_action()  # [2.0, 2.0]
    assert buffer.get_action_index() == 2  # self.k == 2
    # 新 chunk，max_k=2, self.k=2 → drop_n = min(2,2) = 2，丢弃前 2 步
    buffer.integrate_new_chunk(chunk, max_k=2, min_m=1)
    first = buffer.pop_next_action()
    np.testing.assert_array_equal(first, [3.0, 3.0])


def test_max_k_capped_drops_no_more_than_limit():
    """max_k 限制最大丢弃步数，即使 self.k 更大也只丢弃 max_k 步。"""
    buffer = StreamActionBuffer(state_dim=2)
    chunk = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]])
    buffer.integrate_new_chunk(chunk, max_k=8, min_m=1)
    # 消费 5 步（耗尽全部 chunk）
    for _ in range(5):
        buffer.pop_next_action()
    assert buffer.get_action_index() == 5  # self.k == 5
    # 新 chunk，max_k=3, self.k=5 → drop_n = min(5,3) = 3，只丢弃 3 步
    new_chunk = np.array([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [40.0, 40.0]])
    buffer.integrate_new_chunk(new_chunk, max_k=3, min_m=1)
    first = buffer.pop_next_action()
    np.testing.assert_array_equal(first, [40.0, 40.0])


def test_max_k_larger_than_chunk_drops_all():
    """drop_n >= len(chunk) 时 integrate_new_chunk 直接 return，不修改 buffer。"""
    buffer = StreamActionBuffer(state_dim=2)
    chunk = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    buffer.integrate_new_chunk(chunk, max_k=8, min_m=1)
    # 消费 3 步
    for _ in range(3):
        buffer.pop_next_action()
    assert buffer.get_action_index() == 3
    # 新 chunk 只有 2 步，max_k=8, self.k=3 → drop_n=min(3,8)=3 >= len(2) → return
    buffer.integrate_new_chunk(np.array([[4.0, 4.0], [5.0, 5.0]]), max_k=8, min_m=1)
    assert buffer.pop_next_action() is None


def test_max_k_drops_all_preserves_existing_buffer():
    """drop_n >= len(new_chunk) 时旧 buffer 不受影响。"""
    buffer = StreamActionBuffer(state_dim=2)
    chunk1 = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    buffer.integrate_new_chunk(chunk1, max_k=8, min_m=1)
    # 消费 5 步（多于 chunk1 长度，但 buffer 是空的所以不影响）
    # 实际：popleft 3 次后 buffer 空，self.k=3
    for _ in range(5):
        buffer.pop_next_action()  # 后两次返回 None
    assert buffer.get_action_index() == 5
    # 重新集成 chunk1 让 buffer 有数据
    buffer.integrate_new_chunk(chunk1, max_k=0, min_m=1)  # max_k=0 让 self.k=5 不影响（drop_n=min(5,0)=0）
    # 此时 self.k 被重置为 0
    # 新 chunk2 只有 2 步，先消费 5 步（耗尽 buffer），self.k=5
    for _ in range(3):
        buffer.pop_next_action()
    for _ in range(2):
        buffer.pop_next_action()  # None
    assert buffer.get_action_index() == 5  # 3 pop + 2 None pop，但 None 不增 k
    # Wait, pop_next_action only increments k when cur_chunk is non-empty.
    # So after 3 successful pops and 2 None returns, k=3.
    # Let me just test more simply:
    buffer.clear()
    buffer.integrate_new_chunk(chunk1, max_k=8, min_m=1)
    buffer.pop_next_action()  # self.k=1
    # 新 chunk 只有 1 步，max_k=8 → drop_n=min(1,8)=1 >= len(1) → return
    chunk2 = np.array([[4.0, 4.0]])
    buffer.integrate_new_chunk(chunk2, max_k=8, min_m=1)
    # 旧 buffer 不受影响
    np.testing.assert_array_equal(buffer.pop_next_action(), [2.0, 2.0])
    np.testing.assert_array_equal(buffer.pop_next_action(), [3.0, 3.0])
    assert buffer.pop_next_action() is None


def test_crossfade_preserves_overlap():
    buffer = StreamActionBuffer(state_dim=1)
    # First chunk: 0, 10
    buffer.integrate_new_chunk(np.array([[0.0], [10.0]]), max_k=8, min_m=3)
    # Pop one action, then integrate a new chunk with max_k=0 to exercise blending.
    buffer.pop_next_action()  # pops 0.0, self.k=1
    # Remaining in buffer: [10.0]. New chunk: [20.0, 30.0, 40.0].
    # With min_m=3, old_list is padded with 10.0 -> [10.0, 10.0, 10.0].
    # overlap_len = min(3, 3) = 3.
    # smoothed[0] = 1.0 * 10.0 + 0.0 * 20.0 = 10.0
    # smoothed[1] = 0.5 * 10.0 + 0.5 * 30.0 = 20.0
    # smoothed[2] = 0.0 * 10.0 + 1.0 * 40.0 = 40.0
    buffer.integrate_new_chunk(np.array([[20.0], [30.0], [40.0]]), max_k=0, min_m=3)
    np.testing.assert_allclose(buffer.pop_next_action(), [10.0], atol=1e-7)
    np.testing.assert_allclose(buffer.pop_next_action(), [20.0], atol=1e-7)
    np.testing.assert_allclose(buffer.pop_next_action(), [40.0], atol=1e-7)


def test_clear_resets_index():
    buffer = StreamActionBuffer(state_dim=2)
    buffer.integrate_new_chunk(np.array([[1.0, 1.0], [2.0, 2.0]]), max_k=8, min_m=1)
    buffer.pop_next_action()
    assert buffer.get_action_index() == 1
    buffer.clear()
    assert buffer.get_action_index() == 0
    assert buffer.pop_next_action() is None

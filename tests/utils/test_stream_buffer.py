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
    buffer.integrate_new_chunk(np.array([[0.0], [10.0]]), real_delay=0, min_m=3)
    # Pop one action, then integrate a new chunk with real_delay=0 to exercise blending.
    buffer.pop_next_action()  # pops 0.0
    # Remaining in buffer: [10.0]. New chunk: [20.0, 30.0, 40.0].
    # With min_m=3, old_list is padded with 10.0 -> [10.0, 10.0, 10.0].
    # overlap_len = min(3, 3) = 3.
    # smoothed[0] = 1.0 * 10.0 + 0.0 * 20.0 = 10.0
    # smoothed[1] = 0.5 * 10.0 + 0.5 * 30.0 = 20.0
    # smoothed[2] = 0.0 * 10.0 + 1.0 * 40.0 = 40.0
    buffer.integrate_new_chunk(np.array([[20.0], [30.0], [40.0]]), real_delay=0, min_m=3)
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

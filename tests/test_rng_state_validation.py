# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import os
import random
import shutil
import tempfile

import numpy as np
import pytest
import torch

from accelerate import PartialState
from accelerate.checkpointing import load_accelerator_state, save_accelerator_state
from accelerate.utils import RNG_STATE_DIGEST_KEY, RNGStateValidationError, compute_rng_state_digest


PartialState()


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    for _ in range(3):
        random.random()
        np.random.rand()
        torch.randn(2)


def _gen_sequence():
    return (
        [random.random() for _ in range(4)]
        + [float(np.random.rand()) for _ in range(4)]
        + [float(torch.randn(1)) for _ in range(4)]
    )


def _save(output_dir, process_index, seed):
    _seed(seed)
    save_accelerator_state(output_dir, [], [], [], [], process_index, step=7)


def _load(input_dir, process_index):
    return load_accelerator_state(input_dir, [], [], [], [], process_index)


def _rng_state_snapshot():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state())


def _assert_rng_state_unchanged(before):
    after = _rng_state_snapshot()
    assert before[0] == after[0]
    assert before[1][0] == after[1][0]
    assert np.array_equal(before[1][1], after[1][1])
    assert before[1][2:] == after[1][2:]
    assert torch.equal(before[2], after[2])


class TestRNGStateValidation:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.rank0_dir = os.path.join(self.tmpdir, "rank0")
        self.rank1_dir = os.path.join(self.tmpdir, "rank1")
        os.makedirs(self.rank0_dir)
        os.makedirs(self.rank1_dir)
        _save(self.rank0_dir, process_index=0, seed=111)
        _save(self.rank1_dir, process_index=1, seed=222)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    def test_digest_is_stored(self):
        states = torch.load(os.path.join(self.rank0_dir, "random_states_0.pkl"), weights_only=False)
        assert RNG_STATE_DIGEST_KEY in states
        summary = states[RNG_STATE_DIGEST_KEY]
        assert summary["process_index"] == 0
        assert set(summary["states"]) == {"step", "random_state", "numpy_random_seed", "torch_manual_seed"}

    def test_rank_swap_is_rejected(self):
        # Rank 1 restores rank 0's state file (renamed to random_states_1.pkl): the rank must mismatch.
        swap_dir = os.path.join(self.tmpdir, "swap")
        os.makedirs(swap_dir)
        shutil.copy(
            os.path.join(self.rank0_dir, "random_states_0.pkl"), os.path.join(swap_dir, "random_states_1.pkl")
        )
        _seed(999)
        before = _rng_state_snapshot()
        with pytest.raises(RNGStateValidationError, match="saved by process 0 but is being restored by process 1"):
            _load(swap_dir, process_index=1)
        # The rejected load must not have touched the running process' RNG state.
        _assert_rng_state_unchanged(before)

    def test_corrupted_digest_is_rejected(self):
        # Tamper with the digest of rank 0's state file: the generator digest must mismatch.
        corrupt_dir = os.path.join(self.tmpdir, "corrupt")
        os.makedirs(corrupt_dir)
        states = torch.load(os.path.join(self.rank0_dir, "random_states_0.pkl"), weights_only=False)
        digest = states[RNG_STATE_DIGEST_KEY]
        digest["states"]["torch_manual_seed"] = "0" * 64
        states[RNG_STATE_DIGEST_KEY] = digest
        torch.save(states, os.path.join(corrupt_dir, "random_states_0.pkl"))
        _seed(999)
        before = _rng_state_snapshot()
        with pytest.raises(RNGStateValidationError, match="torch_manual_seed.*does not match"):
            _load(corrupt_dir, process_index=0)
        _assert_rng_state_unchanged(before)

    def test_missing_generator_state_is_rejected(self):
        # Drop a generator state from the file but keep the digest: validation must flag the missing state.
        missing_dir = os.path.join(self.tmpdir, "missing")
        os.makedirs(missing_dir)
        states = torch.load(os.path.join(self.rank0_dir, "random_states_0.pkl"), weights_only=False)
        del states["random_state"]
        torch.save(states, os.path.join(missing_dir, "random_states_0.pkl"))
        _seed(999)
        before = _rng_state_snapshot()
        with pytest.raises(RNGStateValidationError, match="random_state.*missing"):
            _load(missing_dir, process_index=0)
        _assert_rng_state_unchanged(before)

    def test_old_format_without_digest_still_loads(self):
        # Checkpoints saved before this change have no digest: they must keep loading and restore the sequence.
        old_dir = os.path.join(self.tmpdir, "old")
        os.makedirs(old_dir)
        states = torch.load(os.path.join(self.rank0_dir, "random_states_0.pkl"), weights_only=False)
        del states[RNG_STATE_DIGEST_KEY]
        torch.save(states, os.path.join(old_dir, "random_states_0.pkl"))
        _seed(111)
        _load(old_dir, process_index=0)
        restored = _gen_sequence()
        _seed(111)
        assert _gen_sequence() == restored

    def test_valid_load_restores_sequence(self):
        _seed(111)
        _load(self.rank0_dir, process_index=0)
        restored = _gen_sequence()
        _seed(111)
        assert _gen_sequence() == restored

    def test_digest_helpers_are_deterministic_and_detect_corruption(self):
        _seed(5)
        states = {
            "random_state": random.getstate(),
            "numpy_random_seed": np.random.get_state(),
            "torch_manual_seed": torch.get_rng_state(),
        }
        first = compute_rng_state_digest(states, process_index=3)
        second = compute_rng_state_digest(states, process_index=3)
        assert first == second
        # A different rank must produce a different summary even for the same states.
        assert compute_rng_state_digest(states, process_index=4) != first
        # Corrupting a state must change its digest.
        corrupted = dict(states)
        corrupted["torch_manual_seed"] = corrupted["torch_manual_seed"].clone()
        corrupted["torch_manual_seed"][0] = 255
        assert compute_rng_state_digest(corrupted, process_index=3) != first

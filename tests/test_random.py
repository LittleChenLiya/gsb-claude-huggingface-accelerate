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

import copy

import pytest
import torch

from accelerate.utils import random as random_utils
from accelerate.utils.random import (
    RNG_STATE_DIGEST_KEY,
    RNG_STATE_RANK_KEY,
    RNGType,
    compute_rng_states_checksum,
    restore_rng_state,
    restore_rng_states,
    save_rng_state,
    save_rng_states,
    verify_rng_states,
)


def _sequence(seed=None, n=5, generator=None):
    if generator is None:
        if seed is not None:
            torch.manual_seed(seed)
        return torch.rand(n).tolist()
    if seed is not None:
        generator.manual_seed(seed)
    return torch.rand(n, generator=generator).tolist()


def test_save_restore_roundtrip_torch_rng():
    expected = _sequence(1234)
    torch.manual_seed(1234)
    payload = save_rng_state(RNGType.TORCH, rank=0)
    assert RNG_STATE_DIGEST_KEY in payload
    # move the live RNG far away
    _sequence(999)
    [torch.rand(1000) for _ in range(1)]
    restore_rng_state(payload, rank=0)
    assert _sequence() == expected


def test_rank_swap_is_rejected_and_rng_untouched():
    torch.manual_seed(111)
    rank0 = save_rng_state("torch", rank=0)
    torch.manual_seed(222)
    rank1 = save_rng_state("torch", rank=1)

    torch.manual_seed(333)
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="rank mismatch"):
        restore_rng_state(rank1, "torch", rank=0)
    after = torch.get_rng_state().clone()
    assert torch.equal(before, after)

    # each payload still restores on its own rank and reproduces its stream
    restore_rng_state(rank0, "torch", rank=0)
    assert _sequence() == _sequence(111)
    restore_rng_state(rank1, "torch", rank=1)
    assert _sequence() == _sequence(222)


@pytest.mark.parametrize("mutate", ["digest", "state"])
def test_corruption_rejected_and_rng_untouched(mutate):
    torch.manual_seed(777)
    payload = save_rng_state("torch", rank=0)
    bad = copy.deepcopy(payload)
    if mutate == "digest":
        bad[RNG_STATE_DIGEST_KEY] = "0" * len(bad[RNG_STATE_DIGEST_KEY])
    else:
        bad["state"][0] ^= 0xFF

    torch.manual_seed(555)
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="verification failed"):
        restore_rng_state(bad, "torch", rank=0)
    after = torch.get_rng_state().clone()
    assert torch.equal(before, after)


def test_rng_type_mismatch_rejected():
    payload = save_rng_state("torch", rank=0)
    with pytest.raises(ValueError, match="RNG type mismatch"):
        restore_rng_state(payload, "generator", generator=torch.Generator(), rank=0)


def test_generator_mismatch_rejected_and_roundtrip_works():
    g1 = torch.Generator().manual_seed(1001)
    g2 = torch.Generator().manual_seed(2002)
    payload = save_rng_state("generator", generator=g1, rank=0)
    expected = _sequence(seed=1001, generator=torch.Generator())

    with pytest.raises(ValueError, match="generator mismatch"):
        restore_rng_state(payload, "generator", generator=g2, rank=0)

    g1 = torch.Generator().manual_seed(1001)
    restore_rng_state(payload, "generator", generator=g1, rank=0)
    assert torch.rand(5, generator=g1).tolist() == expected


def test_legacy_raw_payload_is_accepted():
    # old format: a bare state tensor without envelope metadata
    torch.manual_seed(4242)
    raw = torch.get_rng_state()
    expected = _sequence(4242)
    torch.manual_seed(1)
    restore_rng_state(raw, "torch", rank=0)
    assert _sequence() == expected

    with pytest.raises(ValueError):
        restore_rng_state(raw)  # raw payload without rng_type cannot resolve


def test_checksum_is_deterministic_and_order_independent():
    a = torch.tensor([1, 2, 3], dtype=torch.uint8)
    b = torch.tensor([4, 5, 6], dtype=torch.uint8)
    states = {"a": a, "b": b}
    d1 = compute_rng_states_checksum(states, ["a", "b"])
    # same key order but different dict insertion order -> same digest
    d2 = compute_rng_states_checksum({"b": b, "a": a}, ["a", "b"])
    assert d1 == d2
    # and no-arg call sorts keys, so it is also insertion-order independent
    assert compute_rng_states_checksum({"b": b, "a": a}) == compute_rng_states_checksum(states)
    # changing a value changes the digest
    states2 = {"a": a.clone(), "b": b.clone()}
    states2["b"][0] = 9
    assert compute_rng_states_checksum(states2, ["a", "b"]) != d1
    # and so does swapping in a different value
    states3 = {"a": b, "b": a}
    assert compute_rng_states_checksum(states3, ["a", "b"]) != d1


def test_save_restore_rng_states_envelope_roundtrip_and_swap():
    payload = save_rng_states(["torch"], rank=0)
    assert RNG_STATE_RANK_KEY in payload and RNG_STATE_DIGEST_KEY in payload
    expected = _sequence(1234)
    torch.manual_seed(1234)
    payload = save_rng_states(["torch"], rank=0)
    _sequence(999)
    restore_rng_states(payload, rng_types=["torch"], rank=0)
    assert _sequence() == expected

    # tamper with the batch checksum -> rejected, nothing applied
    torch.manual_seed(1234)
    payload = save_rng_states(["torch"], rank=0)
    torch.manual_seed(555)
    before = torch.get_rng_state().clone()
    payload[RNG_STATE_DIGEST_KEY] = "0" * 64
    with pytest.raises(ValueError, match="checksum mismatch"):
        restore_rng_states(payload, rng_types=["torch"], rank=0)
    assert torch.equal(torch.get_rng_state(), before)


def test_save_rng_states_annotates_existing_mapping():
    states = {
        "torch_manual_seed": torch.get_rng_state(),
    }
    annotated = save_rng_states(states=states, rank=2)
    assert annotated[RNG_STATE_RANK_KEY] == 2
    verify_rng_states(annotated, rank=2)
    with pytest.raises(ValueError, match="rank mismatch"):
        verify_rng_states(annotated, rank=3)
    annotated["torch_manual_seed"][0] ^= 0xFF
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_rng_states(annotated, rank=2)
    # original mapping is not mutated
    assert RNG_STATE_DIGEST_KEY not in states


def test_verify_legacy_mapping_without_metadata_passes():
    legacy = {"torch_manual_seed": torch.get_rng_state()}
    verify_rng_states(legacy, rank=0)  # should be a no-op


def test_restore_rng_states_legacy_mapping_uses_applier():
    calls = []

    def applier(states):
        calls.append(states)

    legacy = {"torch": torch.get_rng_state()}
    restore_rng_states(legacy, rng_types=["torch"], rank=0, applier=applier)
    assert calls == [legacy]


def test_restore_rng_states_applier_only_called_after_verification():
    # If any verification fails, applier must never run (no partial overwrite)
    payload = save_rng_states(["torch"], rank=0)
    payload[RNG_STATE_RANK_KEY] = 5  # rank swap
    calls = []
    with pytest.raises(ValueError, match="rank mismatch"):
        restore_rng_states(payload, rng_types=["torch"], rank=0, applier=lambda s: calls.append(s))
    assert calls == []

    # happy path: applier runs once after verification succeeds
    payload = save_rng_states(["torch"], rank=0)
    restore_rng_states(payload, rng_types=["torch"], rank=0, applier=lambda s: calls.append(s))
    assert len(calls) == 1


def test_synchronize_rng_state_helpers_roundtrip(monkeypatch):
    # exercise the shared get/set helpers used by synchronize_rng_state
    torch.manual_seed(31337)
    state = random_utils._get_rng_state(RNGType.TORCH)
    expected = _sequence(31337)
    torch.manual_seed(0)
    random_utils._set_rng_state(state, RNGType.TORCH)
    assert _sequence() == expected

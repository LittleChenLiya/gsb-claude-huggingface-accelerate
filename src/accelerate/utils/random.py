# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import hashlib
import hmac
import logging
import pickle
import random
from typing import Optional, Union

import numpy as np
import torch

from ..state import AcceleratorState
from .constants import CUDA_DISTRIBUTED_TYPES
from .dataclasses import DistributedType, RNGType
from .imports import (
    is_hpu_available,
    is_mlu_available,
    is_musa_available,
    is_neuron_available,
    is_npu_available,
    is_sdaa_available,
    is_torch_xla_available,
    is_xpu_available,
)


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

logger = logging.getLogger(__name__)


# Version tag for RNG state payloads that carry a verification digest.
# Payloads without these keys are treated as the old, digest-less format.
RNG_STATE_VERSION_KEY = "_accelerate_rng_state_version"
RNG_STATE_RANK_KEY = "_accelerate_rng_state_rank"
RNG_STATE_GENERATOR_KEY = "_accelerate_rng_state_generator"
RNG_STATE_DIGEST_KEY = "_accelerate_rng_state_digest"
RNG_STATE_KEYS_KEY = "_accelerate_rng_state_keys"
RNG_STATE_DIGEST_VERSION = 1
# Keys that make up a wrapped (`save_rng_state`) payload.
_RNG_STATE_ENVELOPE_KEYS = ("rng_type", "rank", "generator", "state")


def set_seed(seed: int, device_specific: bool = False, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        device_specific (`bool`, *optional*, defaults to `False`):
            Whether to differ the seed on each device slightly with `self.process_index`.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    if device_specific:
        seed += AcceleratorState().process_index
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if is_xpu_available():
        torch.xpu.manual_seed_all(seed)
    elif is_npu_available():
        torch.npu.manual_seed_all(seed)
    elif is_mlu_available():
        torch.mlu.manual_seed_all(seed)
    elif is_sdaa_available():
        torch.sdaa.manual_seed_all(seed)
    elif is_musa_available():
        torch.musa.manual_seed_all(seed)
    elif is_hpu_available():
        torch.hpu.manual_seed_all(seed)
    elif is_neuron_available():
        torch.neuron.manual_seed_all(seed)
    else:
        torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available
    if is_torch_xla_available():
        xm.set_rng_state(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


def _get_rng_state(rng_type: RNGType, generator: Optional[torch.Generator] = None):
    """Read the current state of the RNG described by `rng_type`."""
    if rng_type == RNGType.TORCH:
        return torch.get_rng_state()
    elif rng_type == RNGType.CUDA:
        return torch.cuda.get_rng_state()
    elif rng_type == RNGType.XLA:
        assert is_torch_xla_available(), "Can't synchronize XLA seeds as torch_xla is unavailable."
        return torch.tensor(xm.get_rng_state())
    elif rng_type == RNGType.NPU:
        assert is_npu_available(), "Can't synchronize NPU seeds on an environment without NPUs."
        return torch.npu.get_rng_state()
    elif rng_type == RNGType.MLU:
        assert is_mlu_available(), "Can't synchronize MLU seeds on an environment without MLUs."
        return torch.mlu.get_rng_state()
    elif rng_type == RNGType.SDAA:
        assert is_sdaa_available(), "Can't synchronize SDAA seeds on an environment without SDAAs."
        return torch.sdaa.get_rng_state()
    elif rng_type == RNGType.MUSA:
        assert is_musa_available(), "Can't synchronize MUSA seeds on an environment without MUSAs."
        return torch.musa.get_rng_state()
    elif rng_type == RNGType.XPU:
        assert is_xpu_available(), "Can't synchronize XPU seeds on an environment without XPUs."
        return torch.xpu.get_rng_state()
    elif rng_type == RNGType.HPU:
        assert is_hpu_available(), "Can't synchronize HPU seeds on an environment without HPUs."
        return torch.hpu.get_rng_state()
    elif rng_type == RNGType.NEURON:
        assert is_neuron_available(), "Can't synchronize Neuron seeds on an environment without Neuron Cores."
        return torch.neuron.get_rng_state()
    elif rng_type == RNGType.GENERATOR:
        assert generator is not None, "Need a generator to synchronize its seed."
        return generator.get_state()
    raise ValueError(f"Invalid rng_type passed: {rng_type}, should be one of {[r.value for r in RNGType]}")


def _set_rng_state(rng_state, rng_type: RNGType, generator: Optional[torch.Generator] = None) -> None:
    """Write `rng_state` into the RNG described by `rng_type`."""
    if rng_type == RNGType.TORCH:
        torch.set_rng_state(rng_state)
    elif rng_type == RNGType.CUDA:
        torch.cuda.set_rng_state(rng_state)
    elif rng_type == RNGType.NPU:
        torch.npu.set_rng_state(rng_state)
    elif rng_type == RNGType.MLU:
        torch.mlu.set_rng_state(rng_state)
    elif rng_type == RNGType.SDAA:
        torch.sdaa.set_rng_state(rng_state)
    elif rng_type == RNGType.MUSA:
        torch.musa.set_rng_state(rng_state)
    elif rng_type == RNGType.XPU:
        torch.xpu.set_rng_state(rng_state)
    elif rng_type == RNGType.HPU:
        torch.hpu.set_rng_state(rng_state)
    elif rng_type == RNGType.NEURON:
        torch.neuron.set_rng_state(rng_state)
    elif rng_type == RNGType.XLA:
        xm.set_rng_state(rng_state.item())
    elif rng_type == RNGType.GENERATOR:
        generator.set_state(rng_state)
    else:
        raise ValueError(f"Invalid rng_type passed: {rng_type}, should be one of {[r.value for r in RNGType]}")


def synchronize_rng_state(rng_type: Optional[RNGType] = None, generator: Optional[torch.Generator] = None):
    # Get the proper rng state
    rng_type = RNGType(rng_type) if not isinstance(rng_type, RNGType) else rng_type
    rng_state = _get_rng_state(rng_type, generator)

    # Broadcast the rng state from device 0 to other devices
    state = AcceleratorState()
    if state.distributed_type == DistributedType.XLA:
        rng_state = rng_state.to(xm.xla_device())
        xm.collective_broadcast([rng_state])
        xm.mark_step()
        rng_state = rng_state.cpu()
    elif (
        state.distributed_type in CUDA_DISTRIBUTED_TYPES
        or state.distributed_type == DistributedType.MULTI_MLU
        or state.distributed_type == DistributedType.MULTI_SDAA
        or state.distributed_type == DistributedType.MULTI_MUSA
        or state.distributed_type == DistributedType.MULTI_NPU
        or state.distributed_type == DistributedType.MULTI_XPU
        or state.distributed_type == DistributedType.MULTI_HPU
        or state.distributed_type == DistributedType.MULTI_NEURON
    ):
        rng_state = rng_state.to(state.device)
        torch.distributed.broadcast(rng_state, 0)
        rng_state = rng_state.cpu()
    elif state.distributed_type == DistributedType.MULTI_CPU:
        torch.distributed.broadcast(rng_state, 0)

    # Set the broadcast rng state
    _set_rng_state(rng_state, rng_type, generator)


def synchronize_rng_states(rng_types: list[Union[str, RNGType]], generator: Optional[torch.Generator] = None):
    for rng_type in rng_types:
        synchronize_rng_state(RNGType(rng_type), generator=generator)


# ---------------------------------------------------------------------------
# RNG state capture / restore with verification digests
# ---------------------------------------------------------------------------
#
# RNG states are often dumped per rank (checkpointing) or shipped between
# processes (synchronization). Restoring a state produced for a different
# rank, by a different generator, or corrupted on disk silently breaks
# reproducibility. The helpers below wrap captured states with a small
# metadata envelope (rank, generator identity, digest) and verify *everything*
# before touching any RNG, so a failed check can never partially overwrite the
# live RNG states. Payloads without the envelope keys are accepted as the old
# digest-less format for backward compatibility.


def _canonical_rng_bytes(value) -> bytes:
    """Deterministic byte representation of an RNG state value."""
    if isinstance(value, torch.Tensor):
        # `Tensor.numpy()` is a view; copy ensures the buffer stays alive.
        return value.detach().cpu().contiguous().numpy().tobytes()
    # Python / numpy RNG states (tuples with ndarrays) and plain tensors
    # serialized via the fixed protocol 2, which is stable across Python 3.
    return pickle.dumps(value, protocol=2)


def _generator_identity(generator: Optional[torch.Generator]) -> Optional[tuple]:
    """
    Best-effort stable identity of a `torch.Generator`: its device and initial
    seed. `None` for the global RNGs (or when unavailable, e.g. old torch).
    """
    if generator is None:
        return None
    try:
        return (str(generator.device), int(generator.initial_seed()))
    except (AttributeError, RuntimeError, TypeError):
        return None


def compute_rng_states_checksum(states: dict, keys: Optional[list] = None) -> str:
    """
    Compute a hex digest over the RNG state entries of `states`.

    Args:
        states (`dict`): A mapping holding RNG states (e.g. a checkpoint dict).
        keys (`List[str]`, *optional*):
            The entry names to include. When unset, every entry is included
            (with a stable key ordering), which is also what `save` uses when
            the key list is not known at save time.

    Returns:
        `str`: A hex SHA-256 digest of the selected `(key, value)` pairs.
    """
    if keys is None:
        keys = sorted(states.keys())
    hasher = hashlib.sha256()
    for key in keys:
        hasher.update(repr(str(key)).encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(_canonical_rng_bytes(states[key]))
        hasher.update(b"\x00")
    return hasher.hexdigest()


def save_rng_state(
    rng_type: Union[str, RNGType],
    generator: Optional[torch.Generator] = None,
    rank: Optional[int] = None,
) -> dict:
    """
    Capture one RNG state together with a verification digest.

    The returned dict can later be passed to [`restore_rng_state`]. It contains
    the raw state plus envelope metadata (`rank`, generator identity and a
    SHA-256 digest) that makes rank/generator mismatches and on-disk corruption
    detectable at restore time.

    Args:
        rng_type (`str` or `RNGType`): Which RNG to capture (e.g. `"torch"`).
        generator (`torch.Generator`, *optional*):
            Required when `rng_type == RNGType.GENERATOR`.
        rank (`int`, *optional*):
            Rank this state belongs to. Defaults to the current process index.

    Returns:
        `dict`: The wrapped RNG state payload.
    """
    rng_type = RNGType(rng_type)
    if rank is None:
        rank = AcceleratorState().process_index
    state = _get_rng_state(rng_type, generator)
    payload = {
        "rng_type": rng_type.value,
        "rank": int(rank),
        "generator": _generator_identity(generator),
        "state": state,
    }
    digest = hashlib.sha256()
    for key in _RNG_STATE_ENVELOPE_KEYS:
        digest.update(key.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(_canonical_rng_bytes(payload[key]))
        digest.update(b"\x00")
    payload[RNG_STATE_VERSION_KEY] = RNG_STATE_DIGEST_VERSION
    payload[RNG_STATE_DIGEST_KEY] = digest.hexdigest()
    return payload


def _verify_rng_state_payload(
    payload: dict,
    target_type: RNGType,
    generator: Optional[torch.Generator],
    rank: int,
) -> None:
    """Validate an enveloped RNG state without modifying any RNG."""
    expected = payload.get(RNG_STATE_DIGEST_KEY)
    digest = hashlib.sha256()
    for key in _RNG_STATE_ENVELOPE_KEYS:
        if key not in payload:
            raise ValueError(f"Malformed RNG state payload: missing '{key}'.")
        digest.update(key.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(_canonical_rng_bytes(payload[key]))
        digest.update(b"\x00")
    actual = digest.hexdigest()
    if not hmac.compare_digest(str(expected), actual):
        raise ValueError(
            "RNG state verification failed: digest mismatch. The saved state is corrupted "
            "or was tampered with; no RNG state has been modified."
        )
    if target_type.value != payload["rng_type"]:
        raise ValueError(
            f"RNG type mismatch: refusing to restore a '{payload['rng_type']}' state into "
            f"the '{target_type.value}' RNG; no RNG state has been modified."
        )
    if int(rank) != int(payload["rank"]):
        raise ValueError(
            f"RNG state rank mismatch: the state was saved for rank {payload['rank']} but "
            f"rank {rank} is restoring it; no RNG state has been modified."
        )
    saved_generator = payload.get("generator")
    current_generator = _generator_identity(generator)
    if saved_generator is not None and current_generator is not None and saved_generator != current_generator:
        raise ValueError(
            f"RNG state generator mismatch: the state was saved for generator {saved_generator} "
            f"but generator {current_generator} was provided; no RNG state has been modified."
        )


def restore_rng_state(
    payload: Union[dict, object],
    rng_type: Optional[Union[str, RNGType]] = None,
    generator: Optional[torch.Generator] = None,
    rank: Optional[int] = None,
) -> None:
    """
    Restore one RNG state captured by [`save_rng_state`].

    Rank, generator identity and digest are all verified *before* any RNG is
    modified, so a failed verification never partially overwrites the live RNG
    state. A bare (un-enveloped) state tensor/tuple -- the old format -- is
    accepted when `rng_type` is provided.

    Args:
        payload (`dict` or raw RNG state): Output of [`save_rng_state`], or a
            legacy raw state.
        rng_type (`str` or `RNGType`, *optional*):
            Required for legacy raw payloads. For enveloped payloads it is
            checked against the recorded type when provided.
        generator (`torch.Generator`, *optional*):
            Target generator for `RNGType.GENERATOR`; also checked against the
            recorded generator identity when applicable.
        rank (`int`, *optional*):
            Expected rank. Defaults to the current process index; mismatch
            raises an error rather than silently restoring the wrong rank's
            stream.
    """
    # Legacy / raw payload: nothing to verify, just restore directly.
    if not isinstance(payload, dict) or RNG_STATE_DIGEST_KEY not in payload:
        if rng_type is None:
            raise ValueError(
                "A raw (old-format) RNG state was passed without `rng_type`; "
                "pass `rng_type` explicitly or use a payload from `save_rng_state`."
            )
        _set_rng_state(payload, RNGType(rng_type), generator)
        return

    if rank is None:
        rank = AcceleratorState().process_index
    target_type = RNGType(rng_type) if rng_type is not None else RNGType(payload["rng_type"])
    _verify_rng_state_payload(payload, target_type, generator, rank)
    _set_rng_state(payload["state"], target_type, generator)


def save_rng_states(
    rng_types: Optional[list[Union[str, RNGType]]] = None,
    generator: Optional[torch.Generator] = None,
    rank: Optional[int] = None,
    states: Optional[dict] = None,
    keys: Optional[list] = None,
) -> dict:
    """
    Capture several RNG states (or annotate an existing state dict) with
    verification metadata.

    Either pass `rng_types` (the named RNGs to capture) or an existing
    `states` mapping (e.g. the checkpoint dict assembled in
    `save_accelerator_state`). The returned mapping gains:

      - a `rank` entry,
      - a `checksum` entry covering every captured RNG state,
      - (when capturing named RNGs) one enveloped payload per RNG type from
        [`save_rng_state`].

    Args:
        rng_types (`List[Union[str, RNGType]]`, *optional*): RNGs to capture.
        generator (`torch.Generator`, *optional*): Generator for the
            `"generator"` RNG type.
        rank (`int`, *optional*): Owner rank; defaults to the process index.
        states (`dict`, *optional*): Existing mapping to annotate (copied, not
            mutated).
        keys (`List[str]`, *optional*): Entries of `states` covered by the
            checksum. Defaults to every key except the metadata keys.

    Returns:
        `dict`: The mapping with rank and checksum metadata added.
    """
    if rank is None:
        rank = AcceleratorState().process_index
    output = dict(states) if states else {}

    captured_keys = []
    if rng_types is not None:
        for rng_type in rng_types:
            rng_enum = RNGType(rng_type)
            payload = save_rng_state(rng_enum, generator=generator, rank=rank)
            output[rng_enum.value] = payload
            captured_keys.append(rng_enum.value)

    if keys is None:
        keys = [
            key
            for key in (captured_keys or sorted(output.keys()))
            if key not in (RNG_STATE_RANK_KEY, RNG_STATE_DIGEST_KEY, RNG_STATE_KEYS_KEY)
        ]
    output[RNG_STATE_RANK_KEY] = int(rank)
    output[RNG_STATE_DIGEST_KEY] = compute_rng_states_checksum(output, keys)
    return output


def verify_rng_states(states: dict, rank: Optional[int] = None, keys: Optional[list] = None) -> None:
    """
    Verify the rank and checksum of a multi-RNG state mapping produced by
    [`save_rng_states`] (or annotated checkpoint dict) without modifying any
    RNG. Mappings without a checksum are treated as the old format and are
    accepted as-is.

    Raises `ValueError` on rank swap, checksum mismatch or a missing covered
    entry.
    """
    if RNG_STATE_DIGEST_KEY not in states:
        # Old format: nothing to verify.
        return
    if rank is None:
        rank = AcceleratorState().process_index
    saved_rank = states.get(RNG_STATE_RANK_KEY)
    if saved_rank is not None and int(saved_rank) != int(rank):
        raise ValueError(
            f"RNG states rank mismatch: saved for rank {saved_rank}, loaded on rank {rank}; "
            "no RNG state has been modified."
        )
    if keys is None:
        keys = [
            key
            for key in sorted(states.keys())
            if key not in (RNG_STATE_RANK_KEY, RNG_STATE_DIGEST_KEY, RNG_STATE_KEYS_KEY)
        ]
    for key in keys:
        if key not in states:
            raise ValueError(f"RNG states verification failed: expected entry '{key}' is missing.")
    expected = str(states[RNG_STATE_DIGEST_KEY])
    actual = compute_rng_states_checksum(states, keys)
    if not hmac.compare_digest(expected, actual):
        raise ValueError(
            "RNG states verification failed: checksum mismatch (corrupted or mixed-rank data); "
            "no RNG state has been modified."
        )


def restore_rng_states(
    states: dict,
    rng_types: Optional[list[Union[str, RNGType]]] = None,
    generator: Optional[torch.Generator] = None,
    rank: Optional[int] = None,
    keys: Optional[list] = None,
    applier=None,
) -> None:
    """
    Verify then restore a multi-RNG state mapping from [`save_rng_states`].

    Verification (rank + checksum, and per-state envelopes) runs to completion
    *before* any RNG is touched, guaranteeing that a rejected payload cannot
    partially overwrite live RNG states. Mappings without metadata are treated
    as the old format and handed straight to `applier` (backward compatible).

    Args:
        states (`dict`): Mapping from [`save_rng_states`] or an annotated
            checkpoint dict.
        rng_types (`List[Union[str, RNGType]]`, *optional*): RNGs present as
            enveloped payloads; each is verified and restored individually.
        generator (`torch.Generator`, *optional*): Generator for the
            `"generator"` RNG type.
        rank (`int`, *optional*): Expected rank; defaults to the process index.
        keys (`List[str]`, *optional*): Entries covered by the checksum.
        applier (`Callable[[dict], None]`, *optional*):
            Callback that performs the actual state writes. Called only after
            every verification has succeeded (also used for legacy dicts).
    """
    has_metadata = RNG_STATE_DIGEST_KEY in states or any(
        isinstance(states.get(RNGType(rng_type).value), dict)
        and RNG_STATE_DIGEST_KEY in states[RNGType(rng_type).value]
        for rng_type in (rng_types or [])
    )
    if not has_metadata:
        if applier is not None:
            applier(states)
        return

    if rank is None:
        rank = AcceleratorState().process_index
    # Phase 1: verify everything, mutate nothing.
    verify_rng_states(states, rank=rank, keys=keys)
    target_enum_types = [RNGType(rng_type) for rng_type in (rng_types or [])]
    for rng_enum in target_enum_types:
        payload = states.get(rng_enum.value)
        # Enveloped per-state payload: re-verify its own envelope (rank /
        # generator / digest) without applying yet.
        if isinstance(payload, dict) and RNG_STATE_DIGEST_KEY in payload:
            _verify_rng_state_payload(payload, rng_enum, generator if rng_enum == RNGType.GENERATOR else None, rank)

    # Phase 2: all checks passed, apply.
    if applier is not None:
        applier(states)
    for rng_enum in target_enum_types:
        payload = states.get(rng_enum.value)
        if isinstance(payload, dict) and RNG_STATE_DIGEST_KEY in payload:
            _set_rng_state(payload["state"], rng_enum, generator if rng_enum == RNGType.GENERATOR else None)

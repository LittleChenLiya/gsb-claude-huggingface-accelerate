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
import pickle
import random
from typing import Optional, Union

import numpy as np
import torch

from ..state import AcceleratorState
from .constants import CUDA_DISTRIBUTED_TYPES, RNG_STATE_NAME
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


# Key under which a validation summary (digest) is stored alongside the saved RNG states.
RNG_STATE_DIGEST_KEY = "rng_state_digest"


class RNGStateValidationError(Exception):
    """
    Raised when RNG states restored from a checkpoint fail validation.

    This typically means the checkpoint was saved by a different process (rank) than the one restoring it, one of the
    generator states is corrupted, or the checkpoint was tampered with. When this is raised, no RNG state has been
    overwritten yet, so the running process keeps its own random sequence.
    """

    pass


def _rng_state_to_bytes(value) -> bytes:
    """
    Convert a single RNG state value to a canonical byte representation used for digests.

    Torch tensors are hashed from their raw data because pickling a tensor embeds a non-deterministic storage id.
    Numpy arrays are handled similarly. Lists and tuples are encoded element by element, and every other value falls
    back to pickling (deterministic for the ints/strings that make up the rest of the RNG states).
    """
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return (
            b"torch.Tensor:"
            + str(tensor.dtype).encode("ascii")
            + b":"
            + str(tuple(tensor.shape)).encode("ascii")
            + b":"
            + tensor.numpy().tobytes()
        )
    if isinstance(value, np.ndarray):
        return (
            b"np.ndarray:"
            + str(value.dtype).encode("ascii")
            + b":"
            + str(value.shape).encode("ascii")
            + b":"
            + value.tobytes()
        )
    if isinstance(value, (list, tuple)):
        parts = [_rng_state_to_bytes(element) for element in value]
        return type(value).__name__.encode("ascii") + b":" + b",".join(parts)
    return b"pickle:" + pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def compute_rng_state_digest(states: dict, process_index: int) -> dict:
    """
    Compute a validation summary for the RNG states saved by `process_index`.

    The summary records the rank of the process that produced the states and a SHA-256 digest of every individual
    generator state. It is stored alongside the states and can be used on restore to detect rank mismatches (a
    checkpoint saved by another rank) or generator mismatches / corruption.

    Args:
        states (`dict`):
            The RNG states about to be saved. The digest key itself, if present, is ignored.
        process_index (`int`):
            The rank of the process saving the states.

    Returns:
        `dict`: The validation summary to store under `RNG_STATE_DIGEST_KEY`.
    """
    summary = {
        "process_index": process_index,
        "states": {
            key: hashlib.sha256(_rng_state_to_bytes(value)).hexdigest()
            for key, value in states.items()
            if key != RNG_STATE_DIGEST_KEY
        },
    }
    return summary


def validate_rng_state_digest(states: dict, process_index: int) -> bool:
    """
    Validate the RNG states loaded from a checkpoint against the digest stored at save time.

    Args:
        states (`dict`):
            The RNG states loaded from the checkpoint file.
        process_index (`int`):
            The rank of the process restoring the states.

    Returns:
        `bool`: Whether a validation summary was present and validated. `False` means the checkpoint was saved in the
        old format without a digest and was not validated (backward compatible).

    Raises:
        `RNGStateValidationError`: If the states were saved by a different process, a generator state is missing, or a
            generator state is corrupted / does not match. No RNG state has been applied yet when this is raised.
    """
    summary = states.get(RNG_STATE_DIGEST_KEY)
    if summary is None:
        # Old checkpoint format: no validation summary, skip validation for backward compatibility.
        return False

    if not isinstance(summary, dict):
        raise RNGStateValidationError(
            f"RNG state validation summary in the checkpoint restored by process {process_index} is malformed "
            f"(expected a dict, got {type(summary).__name__}). No RNG state has been overwritten."
        )

    saved_process_index = summary.get("process_index")
    if saved_process_index != process_index:
        raise RNGStateValidationError(
            f"RNG state mismatch: the checkpoint was saved by process {saved_process_index} but is being restored by "
            f"process {process_index}. Each process must restore its own RNG state file "
            f"(`{RNG_STATE_NAME}_{process_index}.pkl`). Restoring another rank's state would corrupt the random number "
            "sequence. No RNG state has been overwritten."
        )

    saved_digests = summary.get("states")
    if not isinstance(saved_digests, dict):
        raise RNGStateValidationError(
            f"RNG state validation summary in the checkpoint restored by process {process_index} is malformed "
            "(missing the per-state digests). No RNG state has been overwritten."
        )

    for key, saved_digest in saved_digests.items():
        if key not in states:
            raise RNGStateValidationError(
                f"RNG state mismatch: the generator state '{key}' is missing from the checkpoint being restored by "
                f"process {process_index}. No RNG state has been overwritten."
            )
        current_digest = hashlib.sha256(_rng_state_to_bytes(states[key])).hexdigest()
        if current_digest != saved_digest:
            raise RNGStateValidationError(
                f"RNG state mismatch: the generator state '{key}' restored by process {process_index} does not match "
                "the state saved in the checkpoint. The checkpoint may be corrupted or was saved by a different "
                "generator. No RNG state has been overwritten."
            )
    return True


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


def synchronize_rng_state(rng_type: Optional[RNGType] = None, generator: Optional[torch.Generator] = None):
    # Get the proper rng state
    if rng_type == RNGType.TORCH:
        rng_state = torch.get_rng_state()
    elif rng_type == RNGType.CUDA:
        rng_state = torch.cuda.get_rng_state()
    elif rng_type == RNGType.XLA:
        assert is_torch_xla_available(), "Can't synchronize XLA seeds as torch_xla is unavailable."
        rng_state = torch.tensor(xm.get_rng_state())
    elif rng_type == RNGType.NPU:
        assert is_npu_available(), "Can't synchronize NPU seeds on an environment without NPUs."
        rng_state = torch.npu.get_rng_state()
    elif rng_type == RNGType.MLU:
        assert is_mlu_available(), "Can't synchronize MLU seeds on an environment without MLUs."
        rng_state = torch.mlu.get_rng_state()
    elif rng_type == RNGType.SDAA:
        assert is_sdaa_available(), "Can't synchronize SDAA seeds on an environment without SDAAs."
        rng_state = torch.sdaa.get_rng_state()
    elif rng_type == RNGType.MUSA:
        assert is_musa_available(), "Can't synchronize MUSA seeds on an environment without MUSAs."
        rng_state = torch.musa.get_rng_state()
    elif rng_type == RNGType.XPU:
        assert is_xpu_available(), "Can't synchronize XPU seeds on an environment without XPUs."
        rng_state = torch.xpu.get_rng_state()
    elif rng_type == RNGType.HPU:
        assert is_hpu_available(), "Can't synchronize HPU seeds on an environment without HPUs."
        rng_state = torch.hpu.get_rng_state()
    elif rng_type == RNGType.NEURON:
        assert is_neuron_available(), "Can't synchronize Neuron seeds on an environment without Neuron Cores."
        rng_state = torch.neuron.get_rng_state()
    elif rng_type == RNGType.GENERATOR:
        assert generator is not None, "Need a generator to synchronize its seed."
        rng_state = generator.get_state()

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


def synchronize_rng_states(rng_types: list[Union[str, RNGType]], generator: Optional[torch.Generator] = None):
    for rng_type in rng_types:
        synchronize_rng_state(RNGType(rng_type), generator=generator)

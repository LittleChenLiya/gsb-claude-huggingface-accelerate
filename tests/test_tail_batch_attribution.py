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

"""
Local two-process CPU fixture for the distributed tail batch attribution report.

This script spawns two CPU processes with torch.multiprocessing (gloo backend) and iterates through two
datasets (five samples and an empty set) with a sharded DataLoader, verifying the tail batch attribution
report, which lists each rank's valid and padding sample counts. The report is purely informational:
computing it does not consume or alter the shard's iterator, so the iteration order is left unchanged.

Run locally with:

    python tests/test_tail_batch_attribution.py
"""

import os
import tempfile

import torch
from torch.distributed import all_gather_object
from torch.utils.data import DataLoader, TensorDataset

from accelerate.data_loader import prepare_data_loader
from accelerate.utils import patch_environment


def build_dataloader(length, batch_size=2):
    """Build a non-shuffled DataLoader for a dataset of the given length."""
    if length > 0:
        dataset = TensorDataset(torch.arange(length))
    else:
        dataset = TensorDataset(torch.empty(0, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def verify_attribution(length, batch_size, expected_attribution, process_index, num_processes):
    """
    Iterate through a sharded DataLoader and verify the tail batch attribution report.
    """
    dataloader = build_dataloader(length, batch_size)
    prepared = prepare_data_loader(dataloader, num_processes=num_processes, process_index=process_index)

    attribution = prepared.tail_batch_attribution()
    assert attribution is not None, f"Rank {process_index}: tail batch attribution is None for length {length}"
    assert len(attribution) == num_processes, (
        f"Rank {process_index}: expected {num_processes} entries, got {len(attribution)}"
    )

    for rank in range(num_processes):
        actual = attribution[rank]
        expected = expected_attribution[rank]
        assert actual == expected, (
            f"Rank {process_index}: attribution mismatch for rank {rank} (length={length}). "
            f"Expected {expected}, got {actual}"
        )

    batches_before = [batch[0].tolist() for batch in prepared]
    _ = prepared.tail_batch_attribution()
    batches_after = [batch[0].tolist() for batch in prepared]
    assert batches_before == batches_after, (
        f"Rank {process_index}: iteration order changed after computing tail batch attribution"
    )

    if length > 0 and expected_attribution[process_index]["valid"] + expected_attribution[process_index]["padding"] > 0:
        last_batch = batches_before[-1]
        assert len(last_batch) == batch_size, (
            f"Rank {process_index}: expected tail batch of size {batch_size}, got {len(last_batch)}"
        )
        valid = expected_attribution[process_index]["valid"]
        padding = expected_attribution[process_index]["padding"]
        assert valid + padding == batch_size, (
            f"Rank {process_index}: valid ({valid}) + padding ({padding}) != batch_size ({batch_size})"
        )

    return attribution, batches_before


def _test_tail_batch_attribution(process_index, num_processes):
    """
    The per-process test function, launched in each of the two CPU processes.
    """
    world_size = int(os.environ["WORLD_SIZE"])
    rdv_file = os.environ["ACCELERATE_DEBUG_RDV_FILE"]
    torch.distributed.init_process_group(
        "gloo",
        rank=process_index,
        store=torch.distributed.FileStore(rdv_file, world_size),
        world_size=world_size,
    )

    batch_size = 2

    # Case 1: five samples. With batch_size=2 and num_processes=2, the total batch size is 4, so the last
    # batch has 1 valid sample and 3 padding samples. Distributed across the two ranks:
    #   rank 0: 1 valid + 1 padding
    #   rank 1: 0 valid + 2 padding
    expected_five = [
        {"valid": 1, "padding": 1},
        {"valid": 0, "padding": 2},
    ]
    attribution_five, batches_five = verify_attribution(
        length=5,
        batch_size=batch_size,
        expected_attribution=expected_five,
        process_index=process_index,
        num_processes=num_processes,
    )

    # Case 2: empty dataset. No samples, no tail batch, so every rank reports 0 valid and 0 padding.
    expected_empty = [
        {"valid": 0, "padding": 0},
        {"valid": 0, "padding": 0},
    ]
    attribution_empty, batches_empty = verify_attribution(
        length=0,
        batch_size=batch_size,
        expected_attribution=expected_empty,
        process_index=process_index,
        num_processes=num_processes,
    )

    # Gather the batches from both ranks so we can print the full picture.
    gathered_batches_five = [None] * num_processes
    all_gather_object(gathered_batches_five, batches_five)
    gathered_batches_empty = [None] * num_processes
    all_gather_object(gathered_batches_empty, batches_empty)

    if process_index == 0:
        print("=" * 70)
        print("Distributed tail batch attribution report - verification passed")
        print("=" * 70)
        print(f"num_processes: {num_processes}, batch_size: {batch_size}")
        print()
        print("Case 1: five samples (non-divisible batch size)")
        print(f"  Expected attribution: {expected_five}")
        print(f"  Actual attribution:   {attribution_five}")
        for rank in range(num_processes):
            print(f"  Rank {rank} batches: {gathered_batches_five[rank]}")
        print()
        print("Case 2: empty dataset")
        print(f"  Expected attribution: {expected_empty}")
        print(f"  Actual attribution:   {attribution_empty}")
        for rank in range(num_processes):
            print(f"  Rank {rank} batches: {gathered_batches_empty[rank]}")
        print("=" * 70)
        print("All checks passed.")

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def main():
    """Spawn two CPU processes and run the tail batch attribution test."""
    num_processes = 2
    # Use a file inside a temporary directory for the FileStore rendezvous. On Windows, NamedTemporaryFile
    # keeps the file open, which causes a permission error when FileStore tries to open it, so we create a
    # plain file inside a TemporaryDirectory instead.
    with tempfile.TemporaryDirectory() as tmp_dir:
        rdv_file = os.path.join(tmp_dir, "rdv")
        with patch_environment(
            world_size=num_processes,
            master_addr="127.0.0.1",
            master_port="29500",
            accelerate_mixed_precision="no",
            accelerate_debug_rdv_file=rdv_file,
            accelerate_use_cpu="yes",
        ):
            torch.multiprocessing.start_processes(
                _test_tail_batch_attribution,
                args=(num_processes,),
                nprocs=num_processes,
                start_method="spawn",
            )


if __name__ == "__main__":
    main()

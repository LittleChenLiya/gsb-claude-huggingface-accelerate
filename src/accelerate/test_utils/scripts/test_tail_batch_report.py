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
Fixture script for the distributed tail-batch attribution report (`DataLoaderShard.tail_batch_report`).

It runs a local two-process CPU fixture: each rank prepares a sharded `DataLoader` over a five-sample dataset (batch
size 2, so the tail batch does not divide evenly across the two ranks) and over an empty dataset, iterates both, and
cross-checks the report against what was actually yielded. Per-rank results are dumped as JSON into the directory
pointed to by the `ACCELERATE_TAIL_BATCH_REPORT_TMPDIR` environment variable so the parent test process can verify
them.

Run it directly (it spawns the two CPU processes itself, using a file-store rendezvous so it also works on platforms
without `fork` or libuv-backed TCP stores):

```bash
ACCELERATE_TAIL_BATCH_REPORT_TMPDIR=$(pwd)/tail_batch_report_output \
    python src/accelerate/test_utils/scripts/test_tail_batch_report.py
```

It can also be launched externally, e.g. with `torchrun --nproc_per_node=2`, in which case the rendezvous is left to
the launcher (`env://`).
"""

import json
import os
import tempfile

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from accelerate import Accelerator


WORLD_SIZE = 2


# With 5 samples, a batch size of 2 and 2 processes (even_batches=True, split_batches=False), the sharded batches are:
#   rank 0: [0, 1], [4, 0]  -> 3 real samples, 1 padding sample
#   rank 1: [2, 3], [1, 2]  -> 2 real samples, 2 padding samples
# The empty dataset deterministically attributes zero real and zero padding samples to every rank.
EXPECTED_REPORTS = {
    "five_samples": [
        {"rank": 0, "real_samples": 3, "padding_samples": 1},
        {"rank": 1, "real_samples": 2, "padding_samples": 2},
    ],
    "empty": [
        {"rank": 0, "real_samples": 0, "padding_samples": 0},
        {"rank": 1, "real_samples": 0, "padding_samples": 0},
    ],
}

DATASETS = {
    "five_samples": list(range(5)),
    "empty": [],
}


def replay_attribution(streams, world_size):
    """
    Recomputes the per-rank real/padding attribution from the actual yielded batches: batches are replayed in global
    round-robin order (batch `j` of rank `r` is global batch `j * world_size + r`) and a sample counts as real the
    first time its value is seen, and as padding on any later occurrence.
    """
    seen = set()
    actual = [{"real_samples": 0, "padding_samples": 0} for _ in range(world_size)]
    num_batches = max((len(stream) for stream in streams), default=0)
    for batch_index in range(num_batches):
        for rank in range(world_size):
            if batch_index < len(streams[rank]):
                for value in streams[rank][batch_index]:
                    if value in seen:
                        actual[rank]["padding_samples"] += 1
                    else:
                        seen.add(value)
                        actual[rank]["real_samples"] += 1
    return actual


def main():
    if not torch.distributed.is_initialized():
        # Launched externally (e.g. torchrun): use the launcher-provided rendezvous. Spawned locally: use the
        # file-store rendezvous prepared by the parent process.
        init_method = os.environ.get("ACCELERATE_TAIL_BATCH_REPORT_RDV_FILE")
        if init_method is not None:
            torch.distributed.init_process_group(
                "gloo",
                init_method=f"file://{init_method}",
                rank=int(os.environ["RANK"]),
                world_size=int(os.environ["WORLD_SIZE"]),
            )
        else:
            torch.distributed.init_process_group("gloo")
    accelerator = Accelerator()
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    assert world_size == WORLD_SIZE, f"This fixture expects exactly {WORLD_SIZE} processes, got {world_size}."
    output_dir = os.environ["ACCELERATE_TAIL_BATCH_REPORT_TMPDIR"]

    results = {}
    for name, values in DATASETS.items():
        dataloader = DataLoader(values, batch_size=2, shuffle=False)
        dataloader = accelerator.prepare(dataloader)

        # The report is available before iterating and must not depend on (or consume) the iterator.
        report_before = dataloader.tail_batch_report()
        first_pass = [batch.tolist() for batch in dataloader]
        report_after = dataloader.tail_batch_report()
        second_pass = [batch.tolist() for batch in dataloader]

        # Computing the report must not change the iteration order.
        assert first_pass == second_pass, f"[{name}] iteration order changed after calling `tail_batch_report`."
        assert report_before == report_after, f"[{name}] report changed after iterating."

        expected = EXPECTED_REPORTS[name]
        assert report_before == expected, f"[{name}] unexpected report on rank {rank}: {report_before} != {expected}"
        # Each rank only yields its own shard: `real + padding` samples of its own entry.
        yielded = sum(len(batch) for batch in first_pass)
        own = expected[rank]
        assert yielded == own["real_samples"] + own["padding_samples"], (
            f"[{name}] rank {rank} yielded {yielded} samples but the report attributes "
            f"{own['real_samples'] + own['padding_samples']}."
        )
        results[name] = {"report": report_before, "stream": first_pass}

    with open(os.path.join(output_dir, f"rank{rank}.json"), "w") as f:
        json.dump(results, f)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        # Cross-check the reports against what was actually yielded on every rank.
        all_results = []
        for r in range(world_size):
            with open(os.path.join(output_dir, f"rank{r}.json")) as f:
                all_results.append(json.load(f))
        for name, values in DATASETS.items():
            streams = [all_results[r][name]["stream"] for r in range(world_size)]
            actual = replay_attribution(streams, world_size)
            expected = EXPECTED_REPORTS[name]
            for r in range(world_size):
                assert actual[r] == {k: expected[r][k] for k in ("real_samples", "padding_samples")}, (
                    f"[{name}] replayed attribution on rank {r} ({actual[r]}) does not match the report "
                    f"({expected[r]})."
                )
                assert all_results[r][name]["report"] == expected, (
                    f"[{name}] report dumped by rank {r} does not match the expected report."
                )
            total_real = sum(entry["real_samples"] for entry in expected)
            assert total_real == len(values), f"[{name}] real samples {total_real} != dataset length {len(values)}."
        print("Tail batch attribution report checks passed on 2 CPU processes.", flush=True)


def _spawn_entry(rank, world_size, rdv_file):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    os.environ["ACCELERATE_USE_CPU"] = "yes"
    os.environ["ACCELERATE_TAIL_BATCH_REPORT_RDV_FILE"] = rdv_file
    main()


if __name__ == "__main__":
    if os.environ.get("RANK") is not None:
        # Launched by an external launcher (torchrun, accelerate launch, ...): the rendezvous is env://.
        main()
    else:
        # Local two-process CPU fixture: spawn the ranks ourselves with a file-store rendezvous.
        rdv_dir = tempfile.mkdtemp(prefix="tail_batch_report_rdv_")
        mp.spawn(_spawn_entry, args=(WORLD_SIZE, os.path.join(rdv_dir, "rdv")), nprocs=WORLD_SIZE, join=True)

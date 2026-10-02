"""End-to-end acceptance script for the checkpoint file manifest feature.

Scenario (CPU only, tiny model):
  1. Train a small model with `Accelerator` and save a checkpoint.
  2. Inspect the generated "checkpoint file manifest" (full file listing + size summary).
  3. Restore from the intact checkpoint into a fresh model and verify the weights match.
  4. Make three damaged copies of the checkpoint (deleted file / truncated file /
     same-size byte tampering): each restore attempt must fail *before* any state is
     loaded, leaving the current model weights untouched.
  5. Remove the manifest (simulating an old checkpoint): the checkpoint must still load.

Run:  python repro_checkpoint_manifest.py
"""

import json
import os
import shutil
import tempfile

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from accelerate import Accelerator
from accelerate.checkpointing import CheckpointIntegrityError
from accelerate.utils import CHECKPOINT_MANIFEST_NAME, set_seed


class TinyModel(nn.Module):
    """y = a * x + b"""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return x * self.a + self.b


def make_dataloader():
    x = torch.randn(64, 1)
    y = 2.0 * x + 3.0 + 0.1 * torch.randn(64, 1)
    return DataLoader(TensorDataset(x, y), batch_size=16, shuffle=True)


def train_steps(model, dataloader, optimizer, accelerator, steps=4):
    model.train()
    for _ in range(steps):
        for x, y in dataloader:
            loss = torch.nn.functional.mse_loss(model(x), y)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()


def fresh_model_and_accelerator():
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    dataloader = make_dataloader()
    accelerator = Accelerator()
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    return accelerator, model, optimizer, dataloader


def weights(model):
    return model.a.detach().cpu().clone(), model.b.detach().cpu().clone()


def main():
    results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1-2. Train and save, then inspect the manifest.
        set_seed(42)
        accelerator, model, optimizer, dataloader = fresh_model_and_accelerator()
        train_steps(model, dataloader, optimizer, accelerator)
        checkpoint = os.path.join(tmpdir, "ckpt")
        accelerator.save_state(checkpoint)
        accelerator.wait_for_everyone()
        trained_a, trained_b = weights(model)

        with open(os.path.join(checkpoint, CHECKPOINT_MANIFEST_NAME), encoding="utf-8") as f:
            manifest = json.load(f)
        print("== checkpoint_manifest.json ==")
        print(f"format_version : {manifest['format_version']}")
        print(f"total_files    : {manifest['total_files']}")
        print(f"total_size     : {manifest['total_size']} bytes")
        for name, entry in manifest["files"].items():
            print(f"  - {name:28s} size={entry['size']:>8d}  sha256={entry['sha256'][:16]}...")
        on_disk = {n for n in os.listdir(checkpoint) if n != CHECKPOINT_MANIFEST_NAME}
        results.append(("manifest lists every checkpoint file", set(manifest["files"]) == on_disk))

        # 3. Intact checkpoint loads and restores the trained weights.
        set_seed(0)
        acc2, model2, opt2, dl2 = fresh_model_and_accelerator()
        acc2.load_state(checkpoint)
        a2, b2 = weights(model2)
        ok = torch.equal(a2, trained_a) and torch.equal(b2, trained_b)
        results.append(("intact checkpoint restores trained weights", ok))

        def attempt_bad_load(label, mutate):
            damaged = os.path.join(tmpdir, label)
            shutil.copytree(checkpoint, damaged)
            mutate(damaged)
            set_seed(0)
            acc3, model3, _, _ = fresh_model_and_accelerator()
            before_a, before_b = weights(model3)
            try:
                acc3.load_state(damaged)
            except CheckpointIntegrityError as e:
                after_a, after_b = weights(model3)
                untouched = torch.equal(after_a, before_a) and torch.equal(before_b, after_b)
                print(f"\n[{label}] load rejected as expected: {type(e).__name__}")
                print(f"  reason: {str(e).splitlines()[-1].strip()}")
                print(f"  existing model weights left untouched: {untouched}")
                results.append((f"{label}: load rejected & weights untouched", untouched))
            else:
                results.append((f"{label}: load rejected", False))

        # 4a. Missing file.
        attempt_bad_load("missing_file", lambda d: os.remove(os.path.join(d, "optimizer.bin")))

        # 4b. Truncated (size-changing) model weights file.
        def truncate_weights(d):
            wf = [n for n in os.listdir(d) if n.endswith(".safetensors") or n.endswith(".bin")][0]
            path = os.path.join(d, wf)
            with open(path, "r+b") as f:
                f.truncate(os.path.getsize(path) // 2)

        attempt_bad_load("truncated_file", truncate_weights)

        # 4c. Same-size tampering -> only the checksum can detect it.
        def flip_one_byte(d):
            wf = [n for n in os.listdir(d) if n.endswith(".safetensors") or n.endswith(".bin")][0]
            path = os.path.join(d, wf)
            size = os.path.getsize(path)
            with open(path, "r+b") as f:
                f.seek(size // 2)
                byte = f.read(1)
                f.seek(size // 2)
                f.write(bytes([byte[0] ^ 0xFF]))

        attempt_bad_load("tampered_same_size", flip_one_byte)

        # 5. Legacy checkpoint (no manifest) must still load.
        legacy = os.path.join(tmpdir, "legacy")
        shutil.copytree(checkpoint, legacy)
        os.remove(os.path.join(legacy, CHECKPOINT_MANIFEST_NAME))
        set_seed(0)
        acc4, model4, _, _ = fresh_model_and_accelerator()
        acc4.load_state(legacy)  # must not raise
        a4, b4 = weights(model4)
        ok = torch.equal(a4, trained_a) and torch.equal(b4, trained_b)
        results.append(("legacy checkpoint (no manifest) still loads", ok))

    print("\n== acceptance results ==")
    all_passed = True
    for name, ok in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        all_passed &= ok
    print("\nALL CHECKS PASSED" if all_passed else "\nSOME CHECKS FAILED")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

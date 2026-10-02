# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import json
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from accelerate import Accelerator
from accelerate.checkpointing import (
    CHECKPOINT_MANIFEST_NAME,
    CheckpointManifestError,
    verify_checkpoint_manifest,
)
from accelerate.utils import set_seed


class SmallModel(nn.Module):
    "A tiny CPU model so checkpoint save/load is fast and deterministic"

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


def build_accelerated_state(seed: int = 42):
    "Build a fresh (model, optimizer, accelerator) tuple with reproducible weights"
    set_seed(seed)
    model = SmallModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    accelerator = Accelerator()
    model, optimizer = accelerator.prepare(model, optimizer)
    return accelerator, model, optimizer


def clone_weights(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def assert_weights_equal(weights_a, model_b):
    weights_b = model_b.state_dict()
    assert set(weights_a.keys()) == set(weights_b.keys()), "state dict keys differ"
    for key in weights_a:
        assert torch.equal(weights_a[key], weights_b[key]), f"weights for {key} were overwritten or differ"


def assert_weights_close(weights_a, model_b, atol=1e-6):
    weights_b = model_b.state_dict()
    assert set(weights_a.keys()) == set(weights_b.keys()), "state dict keys differ"
    for key in weights_a:
        assert torch.allclose(weights_a[key], weights_b[key], atol=atol), f"weights for {key} differ"


def test_manifest_is_written_on_save():
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoint"
        accelerator, _, _ = build_accelerated_state()
        accelerator.save_state(checkpoint_dir)

        manifest_path = checkpoint_dir / CHECKPOINT_MANIFEST_NAME
        assert manifest_path.is_file(), "checkpoint manifest was not written on save"

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["manifest_version"] >= 1
        entries = manifest["files"]
        names = [entry["name"] for entry in entries]
        # Standard checkpoint files must all be listed
        assert "model.safetensors" in names
        assert "optimizer.bin" in names
        assert f"random_states_{accelerator.process_index}.pkl" in names
        # The manifest must not list itself
        assert CHECKPOINT_MANIFEST_NAME not in names
        # Every entry has a size and a sha256 checksum
        for entry in entries:
            assert entry["size_bytes"] > 0
            assert len(entry["sha256"]) == 64 and all(c in "0123456789abcdef" for c in entry["sha256"])
        # The size summary must match the listed files
        assert manifest["total_size_bytes"] == sum(entry["size_bytes"] for entry in entries)

        # Verification passes on an intact checkpoint
        verify_checkpoint_manifest(checkpoint_dir)


def test_manifest_with_pytorch_serialization():
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoint"
        accelerator, model, _ = build_accelerated_state()
        accelerator.save_state(checkpoint_dir, safe_serialization=False)

        manifest = json.loads((checkpoint_dir / CHECKPOINT_MANIFEST_NAME).read_text(encoding="utf-8"))
        names = [entry["name"] for entry in manifest["files"]]
        assert "pytorch_model.bin" in names
        assert "model.safetensors" not in names

        verify_checkpoint_manifest(checkpoint_dir)

        # And the checkpoint still loads correctly
        fresh_accelerator, fresh_model, _ = build_accelerated_state(seed=123)
        fresh_accelerator.load_state(checkpoint_dir)
        assert_weights_close(clone_weights(model), fresh_model)


def test_load_restores_weights_after_verification():
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoint"
        accelerator, model, _ = build_accelerated_state()
        saved_weights = clone_weights(model)
        accelerator.save_state(checkpoint_dir)

        # Load into a fresh model with different weights
        fresh_accelerator, fresh_model, _ = build_accelerated_state(seed=999)
        assert not torch.allclose(saved_weights["linear.weight"], fresh_model.state_dict()["linear.weight"])
        fresh_accelerator.load_state(checkpoint_dir)
        assert_weights_close(saved_weights, fresh_model)


def test_missing_file_raises_and_does_not_overwrite_model_state():
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoint"
        accelerator, _, _ = build_accelerated_state()
        accelerator.save_state(checkpoint_dir)

        # A fresh model with different weights that must NOT be overwritten
        fresh_accelerator, fresh_model, _ = build_accelerated_state(seed=999)
        current_weights = clone_weights(fresh_model)

        # Remove a checkpoint file
        (checkpoint_dir / "optimizer.bin").unlink()

        with pytest.raises(CheckpointManifestError, match="missing file: optimizer.bin"):
            fresh_accelerator.load_state(checkpoint_dir)

        # The current model state must be untouched
        assert_weights_equal(current_weights, fresh_model)


def test_corrupted_file_raises_and_does_not_overwrite_model_state():
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoint"
        accelerator, _, _ = build_accelerated_state()
        accelerator.save_state(checkpoint_dir)

        fresh_accelerator, fresh_model, _ = build_accelerated_state(seed=999)
        current_weights = clone_weights(fresh_model)

        # Corrupt the model weights in place while keeping the same file size (flip a byte)
        model_file = checkpoint_dir / "model.safetensors"
        data = bytearray(model_file.read_bytes())
        data[len(data) // 2] ^= 0xFF
        model_file.write_bytes(bytes(data))

        with pytest.raises(CheckpointManifestError, match="checksum mismatch for model.safetensors"):
            fresh_accelerator.load_state(checkpoint_dir)

        assert_weights_equal(current_weights, fresh_model)


def test_truncated_file_raises_on_size_mismatch():
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoint"
        accelerator, _, _ = build_accelerated_state()
        accelerator.save_state(checkpoint_dir)

        fresh_accelerator, fresh_model, _ = build_accelerated_state(seed=999)
        current_weights = clone_weights(fresh_model)

        # Truncate the model weights file
        model_file = checkpoint_dir / "model.safetensors"
        data = model_file.read_bytes()
        model_file.write_bytes(data[:-10])

        with pytest.raises(CheckpointManifestError, match="size mismatch for model.safetensors"):
            fresh_accelerator.load_state(checkpoint_dir)

        assert_weights_equal(current_weights, fresh_model)


def test_legacy_checkpoint_without_manifest_still_loads():
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoint"
        accelerator, model, _ = build_accelerated_state()
        saved_weights = clone_weights(model)
        accelerator.save_state(checkpoint_dir)

        # Simulate an old checkpoint: remove the manifest
        (checkpoint_dir / CHECKPOINT_MANIFEST_NAME).unlink()
        verify_checkpoint_manifest(checkpoint_dir)  # must not raise

        fresh_accelerator, fresh_model, _ = build_accelerated_state(seed=999)
        fresh_accelerator.load_state(checkpoint_dir)
        assert_weights_close(saved_weights, fresh_model)


def test_manifest_includes_custom_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoint"
        accelerator, _, _ = build_accelerated_state()
        custom_obj = nn.Linear(3, 3)
        accelerator.register_for_checkpointing(custom_obj)
        accelerator.save_state(checkpoint_dir)

        manifest = json.loads((checkpoint_dir / CHECKPOINT_MANIFEST_NAME).read_text(encoding="utf-8"))
        names = [entry["name"] for entry in manifest["files"]]
        assert "custom_checkpoint_0.pkl" in names

        # The custom checkpoint is also verified on load
        verify_checkpoint_manifest(checkpoint_dir)

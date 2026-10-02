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

import inspect
import json
import logging
import os
import random
import shutil
import tempfile
import uuid
from contextlib import contextmanager

import pytest
import torch
from parameterized import parameterized_class
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from accelerate import Accelerator
from accelerate.checkpointing import CHECKPOINT_MANIFEST_VERSION, CheckpointIntegrityError
from accelerate.test_utils import (
    DEFAULT_LAUNCH_COMMAND,
    execute_subprocess_async,
    require_non_cpu,
    require_non_torch_xla,
    run_first,
)
from accelerate.test_utils.testing import AccelerateTestCase
from accelerate.utils import (
    CHECKPOINT_MANIFEST_NAME,
    DistributedType,
    ProjectConfiguration,
    patch_environment,
    set_seed,
)


logger = logging.getLogger(__name__)


def dummy_dataloaders(a=2, b=3, batch_size=16, n_train_batches: int = 10, n_valid_batches: int = 2):
    "Generates a tuple of dummy DataLoaders to test with"

    def get_dataset(n_batches):
        x = torch.randn(batch_size * n_batches, 1)
        return TensorDataset(x, a * x + b + 0.1 * torch.randn(batch_size * n_batches, 1))

    train_dataset = get_dataset(n_train_batches)
    valid_dataset = get_dataset(n_valid_batches)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size, num_workers=4)
    return (train_dataloader, valid_dataloader)


def train(num_epochs, model, dataloader, optimizer, accelerator, scheduler=None):
    "Trains for `num_epochs`"
    rands = []
    for epoch in range(num_epochs):
        # Train quickly
        model.train()
        for batch in dataloader:
            x, y = batch
            outputs = model(x)
            loss = torch.nn.functional.mse_loss(outputs, y)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        rands.append(random.random())  # Introduce some randomness
        if scheduler is not None:
            scheduler.step()
    return rands


class DummyModel(nn.Module):
    "Simple model to do y=mx+b"

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return x * self.a + self.b


class DummyStatefulObject:
    "Minimal object with `state_dict`/`load_state_dict` that can be registered for checkpointing."

    def __init__(self):
        self.value = torch.arange(4, dtype=torch.float32)

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state_dict):
        self.value = state_dict["value"]


def parameterized_custom_name_func(func, param_num, param):
    # customize the test name generator function as we want both params to appear in the sub-test
    # name, as by default it shows only the first param
    param_based_name = "use_safetensors" if param["use_safetensors"] is True else "use_pytorch"
    return f"{func.__name__}_{param_based_name}"


@parameterized_class(("use_safetensors",), [[True], [False]], class_name_func=parameterized_custom_name_func)
class CheckpointTest(AccelerateTestCase):
    def check_adam_state(self, state1, state2, distributed_type):
        # For DistributedType.XLA, the `accelerator.save_state` function calls `xm._maybe_convert_to_cpu` before saving.
        # As a result, all tuple values are converted to lists. Therefore, we need to convert them back here.
        # Remove this code once Torch XLA fixes this issue.
        if distributed_type == DistributedType.XLA:
            state1["param_groups"][0]["betas"] = tuple(state1["param_groups"][0]["betas"])
            state2["param_groups"][0]["betas"] = tuple(state2["param_groups"][0]["betas"])
        assert state1 == state2

    def test_with_save_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(42)
            model = DummyModel()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
            train_dataloader, valid_dataloader = dummy_dataloaders()
            project_config = ProjectConfiguration(total_limit=1, project_dir=tmpdir, automatic_checkpoint_naming=True)
            # Train baseline
            accelerator = Accelerator(project_config=project_config)
            model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, valid_dataloader
            )
            # Save initial
            accelerator.save_state(safe_serialization=self.use_safetensors)

            # Save second state
            accelerator.save_state(safe_serialization=self.use_safetensors)
            assert len(os.listdir(accelerator.project_dir)) == 1

    def test_can_resume_training_with_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(42)
            model = DummyModel()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
            train_dataloader, valid_dataloader = dummy_dataloaders()
            # Train baseline
            accelerator = Accelerator()
            model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, valid_dataloader
            )
            # Save initial
            initial = os.path.join(tmpdir, "initial")
            accelerator.save_state(initial, safe_serialization=self.use_safetensors)
            (a, b) = model.a.item(), model.b.item()
            opt_state = optimizer.state_dict()
            ground_truth_rands = train(3, model, train_dataloader, optimizer, accelerator)
            (a1, b1) = model.a.item(), model.b.item()
            opt_state1 = optimizer.state_dict()

            # Train partially
            set_seed(42)
            model = DummyModel()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
            train_dataloader, valid_dataloader = dummy_dataloaders()
            accelerator = Accelerator()
            model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, valid_dataloader
            )
            accelerator.load_state(initial)
            (a2, b2) = model.a.item(), model.b.item()
            opt_state2 = optimizer.state_dict()
            self.assertEqual(a, a2)
            self.assertEqual(b, b2)
            assert a == a2
            assert b == b2
            self.check_adam_state(opt_state, opt_state2, accelerator.distributed_type)

            test_rands = train(2, model, train_dataloader, optimizer, accelerator)
            # Save everything
            checkpoint = os.path.join(tmpdir, "checkpoint")
            accelerator.save_state(checkpoint, safe_serialization=self.use_safetensors)

            # Load everything back in and make sure all states work
            accelerator.load_state(checkpoint)
            test_rands += train(1, model, train_dataloader, optimizer, accelerator)
            (a3, b3) = model.a.item(), model.b.item()
            opt_state3 = optimizer.state_dict()
            assert a1 == a3
            assert b1 == b3
            self.check_adam_state(opt_state1, opt_state3, accelerator.distributed_type)
            assert ground_truth_rands == test_rands

    def test_can_resume_training(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(42)
            model = DummyModel()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
            train_dataloader, valid_dataloader = dummy_dataloaders()
            project_config = ProjectConfiguration(automatic_checkpoint_naming=True)

            # Train baseline
            accelerator = Accelerator(project_dir=tmpdir, project_config=project_config)
            model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, valid_dataloader
            )
            # Save initial
            accelerator.save_state(safe_serialization=self.use_safetensors)
            (a, b) = model.a.item(), model.b.item()
            opt_state = optimizer.state_dict()
            ground_truth_rands = train(3, model, train_dataloader, optimizer, accelerator)
            (a1, b1) = model.a.item(), model.b.item()
            opt_state1 = optimizer.state_dict()

            # Train partially
            set_seed(42)
            model = DummyModel()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
            train_dataloader, valid_dataloader = dummy_dataloaders()
            project_config = ProjectConfiguration(iteration=1, automatic_checkpoint_naming=True)
            accelerator = Accelerator(project_dir=tmpdir, project_config=project_config)
            model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, valid_dataloader
            )
            accelerator.load_state(os.path.join(tmpdir, "checkpoints", "checkpoint_0"))
            (a2, b2) = model.a.item(), model.b.item()
            opt_state2 = optimizer.state_dict()
            assert a == a2
            assert b == b2
            self.check_adam_state(opt_state, opt_state2, accelerator.distributed_type)

            test_rands = train(2, model, train_dataloader, optimizer, accelerator)
            # Save everything
            accelerator.save_state(safe_serialization=self.use_safetensors)

            # Load everything back in and make sure all states work
            accelerator.load_state(os.path.join(tmpdir, "checkpoints", "checkpoint_1"))
            test_rands += train(1, model, train_dataloader, optimizer, accelerator)
            (a3, b3) = model.a.item(), model.b.item()
            opt_state3 = optimizer.state_dict()
            assert a1 == a3
            assert b1 == b3
            self.check_adam_state(opt_state1, opt_state3, accelerator.distributed_type)
            assert ground_truth_rands == test_rands

    def test_can_resume_training_checkpoints_relative_path(self):
        # See #1983
        # This test is like test_can_resume_training but uses a relative path for the checkpoint and automatically
        # infers the checkpoint path when loading.
        @contextmanager
        def temporary_relative_directory():
            # This is equivalent to tempfile.TemporaryDirectory() except that it returns a relative path
            rand_dir = f"test_path_{uuid.uuid4()}"
            os.mkdir(rand_dir)
            try:
                yield rand_dir
            finally:
                shutil.rmtree(rand_dir)

        with temporary_relative_directory() as tmpdir:
            set_seed(42)
            model = DummyModel()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
            train_dataloader, valid_dataloader = dummy_dataloaders()
            project_config = ProjectConfiguration(automatic_checkpoint_naming=True)

            # Train baseline
            accelerator = Accelerator(project_dir=tmpdir, project_config=project_config)
            model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, valid_dataloader
            )
            # Save initial
            accelerator.save_state(safe_serialization=self.use_safetensors)
            (a, b) = model.a.item(), model.b.item()
            opt_state = optimizer.state_dict()
            ground_truth_rands = train(3, model, train_dataloader, optimizer, accelerator)
            (a1, b1) = model.a.item(), model.b.item()
            opt_state1 = optimizer.state_dict()

            # Train partially
            set_seed(42)
            model = DummyModel()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
            train_dataloader, valid_dataloader = dummy_dataloaders()
            project_config = ProjectConfiguration(iteration=1, automatic_checkpoint_naming=True)
            accelerator = Accelerator(project_dir=tmpdir, project_config=project_config)
            model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, valid_dataloader
            )
            accelerator.load_state()  # <= infer the directory automatically
            (a2, b2) = model.a.item(), model.b.item()
            opt_state2 = optimizer.state_dict()
            assert a == a2
            assert b == b2
            self.check_adam_state(opt_state, opt_state2, accelerator.distributed_type)
            assert opt_state == opt_state2

            test_rands = train(2, model, train_dataloader, optimizer, accelerator)
            # Save everything
            accelerator.save_state(safe_serialization=self.use_safetensors)

            # Load everything back in and make sure all states work
            accelerator.load_state(os.path.join(tmpdir, "checkpoints", "checkpoint_1"))
            test_rands += train(1, model, train_dataloader, optimizer, accelerator)
            (a3, b3) = model.a.item(), model.b.item()
            opt_state3 = optimizer.state_dict()
            assert a1 == a3
            assert b1 == b3
            self.check_adam_state(opt_state1, opt_state3, accelerator.distributed_type)
            assert ground_truth_rands == test_rands

    def test_invalid_registration(self):
        t = torch.tensor([1, 2, 3])
        t1 = torch.tensor([2, 3, 4])
        net = DummyModel()
        opt = torch.optim.Adam(net.parameters())
        accelerator = Accelerator()
        with self.assertRaises(ValueError) as ve:
            accelerator.register_for_checkpointing(t, t1, net, opt)
        message = str(ve.exception)
        assert "Item at index 0" in message
        assert "Item at index 1" in message
        assert "Item at index 2" not in message
        assert "Item at index 3" not in message

    def test_with_scheduler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(42)
            model = DummyModel()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
            train_dataloader, valid_dataloader = dummy_dataloaders()
            project_config = ProjectConfiguration(automatic_checkpoint_naming=True)
            # Train baseline
            accelerator = Accelerator(project_dir=tmpdir, project_config=project_config)
            model, optimizer, train_dataloader, valid_dataloader, scheduler = accelerator.prepare(
                model, optimizer, train_dataloader, valid_dataloader, scheduler
            )
            # Save initial
            accelerator.save_state(safe_serialization=self.use_safetensors)
            scheduler_state = scheduler.state_dict()
            train(3, model, train_dataloader, optimizer, accelerator, scheduler)
            assert scheduler_state != scheduler.state_dict()

            # Load everything back in and make sure all states work
            accelerator.load_state(os.path.join(tmpdir, "checkpoints", "checkpoint_0"))
            assert scheduler_state == scheduler.state_dict()

    def test_automatic_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(42)
            model = DummyModel()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
            train_dataloader, valid_dataloader = dummy_dataloaders()
            project_config = ProjectConfiguration(automatic_checkpoint_naming=True)
            # Train baseline
            accelerator = Accelerator(project_dir=tmpdir, project_config=project_config)
            model, optimizer, train_dataloader, valid_dataloader, scheduler = accelerator.prepare(
                model, optimizer, train_dataloader, valid_dataloader, scheduler
            )
            # Save initial
            accelerator.save_state(safe_serialization=self.use_safetensors)
            train(2, model, train_dataloader, optimizer, accelerator, scheduler)
            (a2, b2) = model.a.item(), model.b.item()
            # Save a first time
            accelerator.save_state(safe_serialization=self.use_safetensors)
            train(1, model, train_dataloader, optimizer, accelerator, scheduler)
            (a3, b3) = model.a.item(), model.b.item()

            # Load back in the last saved checkpoint, should point to a2, b2
            accelerator.load_state()
            assert a3 != model.a.item()
            assert b3 != model.b.item()
            assert a2 == model.a.item()
            assert b2 == model.b.item()

    def test_checkpoint_deletion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(42)
            model = DummyModel()
            project_config = ProjectConfiguration(automatic_checkpoint_naming=True, total_limit=2)
            # Train baseline
            accelerator = Accelerator(project_dir=tmpdir, project_config=project_config)
            model = accelerator.prepare(model)
            # Save 3 states:
            for _ in range(11):
                accelerator.save_state(safe_serialization=self.use_safetensors)
            assert not os.path.exists(os.path.join(tmpdir, "checkpoints", "checkpoint_0"))
            assert os.path.exists(os.path.join(tmpdir, "checkpoints", "checkpoint_9"))
            assert os.path.exists(os.path.join(tmpdir, "checkpoints", "checkpoint_10"))

    def _make_pretrained_accelerator(self, tmpdir, name="ckpt", train_steps=2):
        """Prepares a model/optimizer/dataloader, trains for a few steps and saves a checkpoint."""
        set_seed(42)
        model = DummyModel()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
        train_dataloader, valid_dataloader = dummy_dataloaders()
        accelerator = Accelerator()
        model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, valid_dataloader
        )
        train(train_steps, model, train_dataloader, optimizer, accelerator)
        checkpoint = os.path.join(tmpdir, name)
        accelerator.save_state(checkpoint, safe_serialization=self.use_safetensors)
        accelerator.wait_for_everyone()
        return accelerator, model, optimizer, checkpoint

    @staticmethod
    def _model_weight_file(checkpoint):
        weight_suffixes = (".safetensors", ".bin")
        excluded = ("optimizer", "scheduler", "sampler")
        for candidate in os.listdir(checkpoint):
            if candidate.endswith(weight_suffixes) and not any(name in candidate for name in excluded):
                return os.path.join(checkpoint, candidate)
        raise FileNotFoundError(f"No model weights file found in {checkpoint}")

    def test_manifest_is_saved_with_file_listing_and_size_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, _, checkpoint = self._make_pretrained_accelerator(tmpdir)
            manifest_path = os.path.join(checkpoint, CHECKPOINT_MANIFEST_NAME)
            assert os.path.exists(manifest_path)
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)

            assert manifest["format_version"] == CHECKPOINT_MANIFEST_VERSION
            listed_files = manifest["files"]
            # Every non-manifest file in the directory must be listed and only those.
            on_disk = {name for name in os.listdir(checkpoint) if name != CHECKPOINT_MANIFEST_NAME}
            assert set(listed_files) == on_disk
            for name, entry in listed_files.items():
                assert entry["size"] == os.path.getsize(os.path.join(checkpoint, name))
                assert len(entry["sha256"]) == 64  # SHA256 hex digest
            assert manifest["total_files"] == len(listed_files)
            assert manifest["total_size"] == sum(entry["size"] for entry in listed_files.values())

    def test_manifest_verifies_on_clean_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, trained_model, _, checkpoint = self._make_pretrained_accelerator(tmpdir)
            expected_a, expected_b = trained_model.a.item(), trained_model.b.item()

            # A fresh model loads the verified checkpoint and recovers the weights.
            set_seed(0)
            model = DummyModel()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
            train_dataloader, valid_dataloader = dummy_dataloaders()
            accelerator = Accelerator()
            model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, valid_dataloader
            )
            accelerator.load_state(checkpoint)
            assert model.a.item() == expected_a
            assert model.b.item() == expected_b

    def _assert_load_aborts_and_weights_untouched(self, checkpoint):
        """Attempts to load a damaged checkpoint and asserts current weights survive."""
        set_seed(0)
        model = DummyModel()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
        train_dataloader, valid_dataloader = dummy_dataloaders()
        accelerator = Accelerator()
        model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, valid_dataloader
        )
        untrained_a, untrained_b = model.a.detach().clone(), model.b.detach().clone()
        with pytest.raises(CheckpointIntegrityError):
            accelerator.load_state(checkpoint)
        # The existing model state must not have been overwritten by the bad checkpoint.
        torch.testing.assert_close(model.a.detach().cpu(), untrained_a)
        torch.testing.assert_close(model.b.detach().cpu(), untrained_b)

    def test_missing_checkpoint_file_does_not_overwrite_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, _, checkpoint = self._make_pretrained_accelerator(tmpdir)
            # Delete the optimizer state (any single missing file must abort the load).
            os.remove(os.path.join(checkpoint, "optimizer.bin"))
            self._assert_load_aborts_and_weights_untouched(checkpoint)

    def test_truncated_checkpoint_file_does_not_overwrite_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, _, checkpoint = self._make_pretrained_accelerator(tmpdir)
            # Corrupt (truncate) the model weights file: size check must catch it.
            weight_file = self._model_weight_file(checkpoint)
            with open(weight_file, "r+b") as f:
                f.truncate(max(1, os.path.getsize(weight_file) // 2))
            self._assert_load_aborts_and_weights_untouched(checkpoint)

    def test_tampered_same_size_checkpoint_file_does_not_overwrite_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, _, checkpoint = self._make_pretrained_accelerator(tmpdir)
            # Modify a byte in the middle without changing the file size: the SHA256
            # checksum must detect the tampering even though sizes still match.
            weight_file = self._model_weight_file(checkpoint)
            size = os.path.getsize(weight_file)
            with open(weight_file, "r+b") as f:
                f.seek(size // 2)
                original = f.read(1)
                f.seek(size // 2)
                f.write(bytes([original[0] ^ 0xFF]))
            assert os.path.getsize(weight_file) == size
            self._assert_load_aborts_and_weights_untouched(checkpoint)

    def test_corrupted_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, _, checkpoint = self._make_pretrained_accelerator(tmpdir)
            with open(os.path.join(checkpoint, CHECKPOINT_MANIFEST_NAME), "w") as f:
                f.write("{not valid json")
            self._assert_load_aborts_and_weights_untouched(checkpoint)

    def test_legacy_checkpoint_without_manifest_still_loads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _, trained_model, _, checkpoint = self._make_pretrained_accelerator(tmpdir)
            expected_a, expected_b = trained_model.a.item(), trained_model.b.item()
            # Simulate a checkpoint written by an older version of Accelerate.
            os.remove(os.path.join(checkpoint, CHECKPOINT_MANIFEST_NAME))

            set_seed(0)
            model = DummyModel()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
            train_dataloader, valid_dataloader = dummy_dataloaders()
            accelerator = Accelerator()
            model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, valid_dataloader
            )
            # Must not raise despite the missing manifest.
            accelerator.load_state(checkpoint)
            assert model.a.item() == expected_a
            assert model.b.item() == expected_b

    def test_custom_registered_object_is_listed_in_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(42)
            model = DummyModel()
            optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
            train_dataloader, valid_dataloader = dummy_dataloaders()
            accelerator = Accelerator()
            model, optimizer, train_dataloader, valid_dataloader = accelerator.prepare(
                model, optimizer, train_dataloader, valid_dataloader
            )
            tensor = DummyStatefulObject()
            accelerator.register_for_checkpointing(tensor)
            checkpoint = os.path.join(tmpdir, "ckpt")
            accelerator.save_state(checkpoint, safe_serialization=self.use_safetensors)
            accelerator.wait_for_everyone()
            with open(os.path.join(checkpoint, CHECKPOINT_MANIFEST_NAME), encoding="utf-8") as f:
                manifest = json.load(f)
            assert "custom_checkpoint_0.pkl" in manifest["files"]
            # And the manifest check passes on load.
            accelerator.load_state(checkpoint)

    @run_first
    @require_non_cpu
    @require_non_torch_xla
    def test_map_location(self):
        cmd = DEFAULT_LAUNCH_COMMAND + [inspect.getfile(self.__class__)]

        env_kwargs = dict(use_safe_tensors=str(self.use_safetensors), omp_num_threads="1")
        with patch_environment(**env_kwargs):
            execute_subprocess_async(cmd)


if __name__ == "__main__":
    use_safetensors = os.environ.get("USE_SAFETENSORS", "False") == "True"
    savedir = "/tmp/accelerate/state_checkpointing"
    model = DummyModel()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    train_dataloader, valid_dataloader = dummy_dataloaders()
    project_config = ProjectConfiguration(automatic_checkpoint_naming=True)
    # Train baseline
    accelerator = Accelerator(project_dir=savedir, project_config=project_config, mixed_precision="no")
    if accelerator.process_index == 0:
        if os.path.exists(savedir):
            shutil.rmtree(savedir)
        os.makedirs(savedir)
    model, optimizer, train_dataloader, valid_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, scheduler
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    train(3, model, train_dataloader, optimizer, accelerator, scheduler)
    # Check that the initial optimizer is loaded on the GPU
    for group in optimizer.param_groups:
        param_device = group["params"][0].device
        break
    assert param_device.type == accelerator.device.type
    model = model.cpu()
    accelerator.wait_for_everyone()
    accelerator.save_state(safe_serialization=use_safetensors)
    accelerator.wait_for_everyone()

    # Check CPU state
    accelerator.load_state(os.path.join(savedir, "checkpoints", "checkpoint_0"), map_location="cpu")
    for group in optimizer.param_groups:
        param_device = group["params"][0].device
        break
    assert param_device.type == torch.device("cpu").type, (
        f"Loaded optimizer states did not match, expected to be loaded on the CPU but got {param_device}"
    )

    # Check device state
    model.to(accelerator.device)
    accelerator.load_state(os.path.join(savedir, "checkpoints", "checkpoint_0"), map_location="on_device")
    for group in optimizer.param_groups:
        param_device = group["params"][0].device
        break
    assert param_device.type == accelerator.device.type, (
        f"Loaded optimizer states did not match, expected to be loaded on {accelerator.device} but got {param_device}"
    )

    # Check error
    with pytest.raises(TypeError, match="Unsupported optimizer map location passed"):
        accelerator.load_state(os.path.join(savedir, "checkpoints", "checkpoint_0"), map_location="invalid")
    accelerator.wait_for_everyone()
    if accelerator.process_index == 0:
        shutil.rmtree(savedir)
    accelerator.wait_for_everyone()

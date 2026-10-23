#!/usr/bin/env python3
"""Run one fixed GDTS asset stage with epoch-boundary RNG-complete resumes.

The legacy implementation remains read-only.  This adapter adds RNG fields to
its ordinary epoch-boundary ``last`` checkpoint and restores them on resume;
it does not change model, loss, sampling, optimizer, or data semantics.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml


def rng_state(loader) -> dict:
    numpy_state = np.random.get_state()
    generator = getattr(loader, "generator", None)
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "train_loader_generator": (
            generator.get_state() if generator is not None else None
        ),
    }


def cpu_byte_rng_state(value, name: str) -> torch.Tensor:
    """Return the canonical state tensor required by PyTorch RNG APIs.

    Legacy checkpoint loading uses ``map_location=self.device``.  On CUDA this
    also moves serialized RNG tensors to the GPU, while ``set_rng_state`` and
    ``Generator.set_state`` require CPU ByteTensors.  Normalizing the storage
    device here preserves the serialized bytes without changing the state.
    """
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"{name} RNG state is not a tensor")
    if value.dtype != torch.uint8:
        raise RuntimeError(f"{name} RNG state is not uint8")
    return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def restore_rng(state: dict, loader) -> None:
    required = {
        "python", "numpy", "torch_cpu", "torch_cuda_all",
        "train_loader_generator",
    }
    if not isinstance(state, dict) or set(state) != required:
        raise RuntimeError("asset checkpoint has no complete runtime RNG state")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state["bit_generator"], numpy_state["keys"].cpu().numpy(),
        int(numpy_state["position"]), int(numpy_state["has_gauss"]),
        float(numpy_state["cached_gaussian"]),
    ))
    torch.set_rng_state(cpu_byte_rng_state(state["torch_cpu"], "torch_cpu"))
    if state["torch_cuda_all"] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("cannot restore CUDA RNG without CUDA")
        torch.cuda.set_rng_state_all([
            cpu_byte_rng_state(value, f"torch_cuda_all[{index}]")
            for index, value in enumerate(state["torch_cuda_all"])
        ])
    generator = getattr(loader, "generator", None)
    if state["train_loader_generator"] is not None:
        if generator is None:
            raise RuntimeError("checkpoint contains loader RNG but loader has none")
        generator.set_state(cpu_byte_rng_state(
            state["train_loader_generator"], "train_loader_generator"))


def atomic_torch_save(payload: dict, destination: Path) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def patch_goal_pretrainer(goal_pretrainer) -> None:
    original_payload = goal_pretrainer._checkpoint_payload
    original_load = goal_pretrainer._load_checkpoint

    def checkpoint_payload(self, epoch):
        payload = original_payload(self, epoch)
        payload.update({
            "runtime_rng_state": rng_state(self.data_loaders["train"]),
            "full_runtime_resume_contract": True,
            "full_runtime_resume_provider": "paper_adapter_v1",
        })
        return payload

    def load_checkpoint(self, load_checkpoint):
        epoch = original_load(self, load_checkpoint)
        if load_checkpoint is not None:
            restore_rng(
                self._pending_training_state.get("runtime_rng_state"),
                self.data_loaders["train"],
            )
        return epoch

    goal_pretrainer._checkpoint_payload = checkpoint_payload
    goal_pretrainer._load_checkpoint = load_checkpoint


def patch_baseline_trainer(trainer) -> None:
    original_save = trainer._save_checkpoint
    original_restart = trainer._load_or_restart

    def save_checkpoint(self, epoch, best_epoch=False, last_epoch=False):
        original_save(self, epoch, best_epoch=best_epoch, last_epoch=last_epoch)
        if best_epoch:
            return
        name = "last_model.pt" if last_epoch else f"epoch_{int(epoch):03d}.pt"
        path = Path(self.args.model_dir) / "saved_models" / name
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload.update({
            "runtime_rng_state": rng_state(self.data_loaders["train"]),
            "full_runtime_resume_contract": True,
            "full_runtime_resume_provider": "paper_adapter_v1",
        })
        atomic_torch_save(payload, path)

    def load_or_restart(self):
        start_epoch = original_restart(self)
        if self.args.load_checkpoint is not None:
            restore_rng(
                self._pending_training_state.get("runtime_rng_state"),
                self.data_loaders["train"],
            )
        return start_epoch

    trainer._save_checkpoint = save_checkpoint
    trainer._load_or_restart = load_or_restart


def fixed_arguments(recipe: dict, stage: str, goal_checkpoint: str | None) -> list[str]:
    common = [
        "--dataset", "eth5", "--test_set", "zara1",
        "--reproducibility", "True", "--seed", str(recipe["seed"]),
        "--batch_size", "64", "--learning_rate", "0.001",
        "--optimizer", "Adam", "--scheduler", "ExponentialLR",
        "--device", "cuda:0", "--clip", "1", "--down_factor", "8",
        "--skip_ts_window", "1", "--num_samples", "20",
        "--ddpm_step", "100", "--ddim_step", "20",
        "--trunk_stage_step", "30", "--goal_model_type", "independent",
        "--use_scene_latent", "False", "--use_dynamic_relation", "False",
        "--use_joint_energy", "False", "--use_dependency_corrector", "False",
        "--training_stage", "baseline", "--amp_enabled", "False",
        "--amp_dtype", "fp32", "--early_stopping_patience", "0",
        "--shuffle_test_batches", "False", "--use_wandb", "False",
        "--save_every", "10",
    ]
    if stage == "goal":
        section = recipe["goal_pretrain"]
        return common + [
            "--phase", "goal_pretrain", "--run_name", section["run_name"],
            "--num_epochs", str(section["epochs"]), "--num_workers", "0",
            "--data_augmentation", "False", "--shuffle_train_batches", "False",
            "--start_validation", "1", "--validate_every", "1",
        ]
    if stage == "baseline":
        if not goal_checkpoint:
            raise ValueError("--goal-checkpoint is required for baseline")
        section = recipe["gdts_baseline"]
        return common + [
            "--phase", "train", "--run_name", section["run_name"],
            "--num_epochs", str(section["epochs"]), "--num_workers", "2",
            "--data_augmentation", "True", "--shuffle_train_batches", "True",
            "--start_validation", "1", "--validate_every", "10",
            "--goal_pretrain_checkpoint", str(Path(goal_checkpoint).resolve()),
        ]
    if stage == "scene_cache":
        return [
            "--dataset", "eth5", "--test_set", "zara1",
            "--phase", "pre-process", "--batch_size", "64",
            "--device", "cpu", "--num_workers", "0", "--down_factor", "8",
            "--skip_ts_window", "1", "--goal_model_type", "social",
            "--training_stage", "finetune", "--data_augmentation", "False",
            "--shuffle_train_batches", "False", "--shuffle_test_batches", "False",
            "--reproducibility", "True", "--seed", str(recipe["seed"]),
            "--use_wandb", "False",
        ]
    raise ValueError(stage)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--stage", choices=("goal", "baseline", "scene_cache"), required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--batch-cache-root", required=True)
    parser.add_argument("--goal-checkpoint")
    args = parser.parse_args()
    recipe = yaml.safe_load(Path(args.recipe).read_text(encoding="utf-8"))
    legacy = Path(recipe["legacy_gdts"]["root"]).resolve()
    runtime = Path(args.runtime_root).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    data_link = runtime / "data"
    if not data_link.exists() and not data_link.is_symlink():
        data_link.symlink_to(Path(recipe["source_data_root"]).resolve(), target_is_directory=True)
    if data_link.resolve() != Path(recipe["source_data_root"]).resolve():
        raise RuntimeError("runtime data link points to an unexpected target")
    sys.path.insert(0, str(legacy))
    os.chdir(runtime)
    from src.models.goal_pretrain import goal_pretrainer
    from src.trainer import trainer
    patch_goal_pretrainer(goal_pretrainer)
    patch_baseline_trainer(trainer)
    argv = fixed_arguments(recipe, args.stage, args.goal_checkpoint)
    argv += ["--batch_cache_root", str(Path(args.batch_cache_root).resolve())]
    if args.stage == "goal":
        last = runtime / "output" / "zara1" / "runs" / recipe["goal_pretrain"]["run_name"] / "saved_models" / "goal_pretrain_last_model.pt"
        if last.is_file():
            argv += ["--load_checkpoint", "last"]
    elif args.stage == "baseline":
        last = runtime / "output" / "zara1" / "runs" / recipe["gdts_baseline"]["run_name"] / "saved_models" / "last_model.pt"
        if last.is_file():
            argv += ["--load_checkpoint", "last"]
    sys.argv = [str(legacy / "main.py"), *argv]
    from main import main as legacy_main
    legacy_main()


if __name__ == "__main__":
    main()

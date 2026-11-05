#!/usr/bin/env python3
"""Run one fixed GDTS asset stage with epoch-boundary RNG-complete resumes.

The legacy implementation remains read-only.  This adapter adds RNG fields to
its ordinary epoch-boundary ``last`` checkpoint and restores them on resume;
it does not change model, loss, sampling, optimizer, or data semantics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

from ebjd.early_stopping import RelativeEarlyStop


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


def atomic_json(payload: dict, destination: Path) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def patch_baseline_trainer(trainer, contract: dict) -> None:
    original_save = trainer._save_checkpoint
    original_restart = trainer._load_or_restart
    original_evaluate = trainer._evaluate_epoch

    required = {
        "monitor", "earliest_stop_epoch", "patience_validations",
        "significant_relative_improvement", "validation_frequency_epochs",
    }
    if not required.issubset(contract):
        raise RuntimeError("incomplete matched-GDTS early-stop contract")
    if str(contract["monitor"]) != "valid_ADE":
        raise RuntimeError("matched-GDTS selector monitor must be valid_ADE")
    if int(contract["validation_frequency_epochs"]) != 10:
        raise RuntimeError("matched-GDTS validation frequency must remain 10")
    if int(contract["earliest_stop_epoch"]) < int(
            contract.get("minimum_epochs", 1)):
        raise RuntimeError("earliest stop precedes minimum epochs")

    def fresh_state():
        return RelativeEarlyStop(
            monitor=str(contract["monitor"]),
            earliest_stop_epoch=int(contract["earliest_stop_epoch"]),
            patience=int(contract["patience_validations"]),
            relative_improvement=float(
                contract["significant_relative_improvement"]),
        )

    def rebuild_state(self, through_epoch):
        state = fresh_state()
        if not os.path.isfile(self.log_curve_file):
            return state, "no_history_fresh_state"
        with open(self.log_curve_file, newline="", encoding="utf-8") as handle:
            rows = csv.DictReader(handle)
            for row in rows:
                epoch = int(float(row["epoch"]))
                if epoch > through_epoch:
                    break
                value = row.get(str(contract["monitor"]))
                if value not in (None, ""):
                    state.observe(float(value), epoch)
        return state, "rebuilt_from_log_curve"

    def evaluate_epoch(self, epoch, mode="valid", *args, **kwargs):
        metrics = original_evaluate(
            self, epoch, mode=mode, *args, **kwargs)
        if mode == "valid":
            monitor = str(contract["monitor"])
            if monitor not in metrics:
                raise RuntimeError(
                    f"matched-GDTS early-stop monitor absent: {monitor}")
            self._paper_early_stop_epoch = int(epoch)
            self._paper_early_stop_score = float(metrics[monitor])
        return metrics

    def record_validation_outcome(self, _selection_improved):
        if not hasattr(self, "_paper_early_stop_state"):
            self._paper_early_stop_state = fresh_state()
            self._paper_early_stop_restore = "fresh_run"
        epoch = int(self._paper_early_stop_epoch)
        result = self._paper_early_stop_state.observe(
            self._paper_early_stop_score, epoch)
        self._paper_early_stop_decision = result
        self._validations_without_improvement = int(
            self._paper_early_stop_state.stale_validations)
        print(
            "[GDTS early-stop] "
            f"E{epoch:03d} monitor={contract['monitor']} "
            f"score={self._paper_early_stop_score:.9g} "
            f"best={self._paper_early_stop_state.best_score:.9g} "
            f"significant_best="
            f"{self._paper_early_stop_state.significant_best:.9g} "
            f"stale={self._paper_early_stop_state.stale_validations}/"
            f"{self._paper_early_stop_state.patience} "
            f"earliest={self._paper_early_stop_state.earliest_stop_epoch}",
            flush=True,
        )
        # The legacy loop owns the stop branch. Preserve accumulated patience
        # before minimum_epochs, but hide it from that branch until eligible.
        return (
            self._paper_early_stop_state.stale_validations
            if epoch >= self._paper_early_stop_state.earliest_stop_epoch else 0
        )

    def save_checkpoint(self, epoch, best_epoch=False, last_epoch=False):
        original_save(self, epoch, best_epoch=best_epoch, last_epoch=last_epoch)
        name = (
            "best_model.pt" if best_epoch else
            "last_model.pt" if last_epoch else f"epoch_{int(epoch):03d}.pt")
        path = Path(self.args.model_dir) / "saved_models" / name
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload.update({
            "paper_early_stop_state": (
                self._paper_early_stop_state.state_dict()
                if hasattr(self, "_paper_early_stop_state") else
                fresh_state().state_dict()),
            "paper_early_stop_contract": dict(contract),
            "paper_early_stop_restore": getattr(
                self, "_paper_early_stop_restore", "fresh_run"),
        })
        if not best_epoch:
            payload.update({
                "runtime_rng_state": rng_state(self.data_loaders["train"]),
                "full_runtime_resume_contract": True,
                "full_runtime_resume_provider": "paper_adapter_v1",
            })
        atomic_torch_save(payload, path)

    def load_or_restart(self):
        start_epoch = original_restart(self)
        if self.args.load_checkpoint is not None:
            path = Path(self.args.model_dir) / "saved_models/last_model.pt"
            payload = torch.load(path, map_location="cpu", weights_only=False)
            restore_rng(
                self._pending_training_state.get("runtime_rng_state"),
                self.data_loaders["train"],
            )
            saved = payload.get("paper_early_stop_state")
            if saved is None:
                state, source = rebuild_state(self, start_epoch - 1)
                self._paper_early_stop_state = state
                self._paper_early_stop_restore = source
            else:
                state = RelativeEarlyStop.from_state_dict(saved)
                expected = fresh_state().state_dict()
                actual = state.state_dict()
                identity_fields = (
                    "monitor", "direction", "earliest_stop_epoch",
                    "patience", "relative_improvement",
                )
                if any(actual[name] != expected[name]
                       for name in identity_fields):
                    state, source = rebuild_state(self, start_epoch - 1)
                    self._paper_early_stop_state = state
                    self._paper_early_stop_restore = (
                        "contract_changed_" + source)
                else:
                    self._paper_early_stop_state = state
                    self._paper_early_stop_restore = "checkpoint"
            self._validations_without_improvement = int(
                self._paper_early_stop_state.stale_validations)
        else:
            self._paper_early_stop_state = fresh_state()
            self._paper_early_stop_restore = "fresh_run"
        return start_epoch

    trainer._evaluate_epoch = evaluate_epoch
    trainer._record_validation_outcome = record_validation_outcome
    trainer._save_checkpoint = save_checkpoint
    trainer._load_or_restart = load_or_restart


def fixed_arguments(recipe: dict, stage: str, goal_checkpoint: str | None) -> list[str]:
    fold = str(recipe["fold"])
    dataset = str(recipe.get("legacy_dataset", "eth5"))
    test_set = fold if dataset == "eth5" else dataset
    common = [
        "--dataset", dataset, "--test_set", test_set,
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
        "--amp_dtype", "fp32",
        "--shuffle_test_batches", "False", "--use_wandb", "False",
        "--save_every", "10",
        "--compress_batch_cache", "True",
    ]
    if stage == "goal":
        section = recipe["goal_pretrain"]
        return common + [
            "--phase", "goal_pretrain", "--run_name", section["run_name"],
            "--num_epochs", str(section["epochs"]), "--num_workers", "0",
            "--data_augmentation", "False", "--shuffle_train_batches", "False",
            "--start_validation", "1", "--validate_every", "1",
            "--early_stopping_patience", "0",
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
            "--early_stopping_patience", str(
                section["early_stopping"]["patience_validations"]),
            "--goal_pretrain_checkpoint", str(Path(goal_checkpoint).resolve()),
        ]
    if stage in ("scene_cache", "scene_cache_independent"):
        return [
            "--dataset", dataset, "--test_set", test_set,
            "--phase", "pre-process", "--batch_size", "64",
            "--device", "cpu", "--num_workers", "0", "--down_factor", "8",
            "--skip_ts_window", "1", "--goal_model_type",
            "independent" if stage == "scene_cache_independent" else "social",
            "--training_stage", "finetune", "--data_augmentation", "False",
            "--shuffle_train_batches", "False", "--shuffle_test_batches", "False",
            "--reproducibility", "True", "--seed", str(recipe["seed"]),
            "--use_wandb", "False",
            "--compress_batch_cache", "True",
        ]
    raise ValueError(stage)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument(
        "--stage",
        choices=("goal", "baseline", "scene_cache", "scene_cache_independent"),
        required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--batch-cache-root", required=True)
    parser.add_argument("--goal-checkpoint")
    args = parser.parse_args()
    if args.stage in ("scene_cache", "scene_cache_independent"):
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        os.nice(10)
    recipe = yaml.safe_load(Path(args.recipe).read_text(encoding="utf-8"))
    fold = str(recipe["fold"])
    output_key = (
        str(recipe.get("legacy_dataset"))
        if str(recipe.get("legacy_dataset", "eth5")) != "eth5" else fold)
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
    from src.data_src import data_utils
    source_data_root = str(Path(recipe["source_data_root"]).resolve())
    data_utils.get_data_location = lambda: source_data_root
    if str(recipe.get("legacy_dataset", "eth5")) == "sdd":
        from src.data_src.dataset_src.dataset_sdd import Dataset_sdd
        from src.data_src.scene_src.scene_sdd import Scene_sdd
        original_dataset_init = Dataset_sdd.__init__

        def exact_jmm_init(self, verbose=False, load_visual_data=True):
            original_dataset_init(
                self, verbose=verbose, load_visual_data=load_visual_data)
            self.train_scenes.discard("hyang_7")
            self.train_scenes.add("nexus_2")
            self.scenes.pop("hyang_7", None)
            self.scenes["nexus_2"] = Scene_sdd(
                "nexus_2", verbose, load_visual_data=load_visual_data)

        Dataset_sdd.__init__ = exact_jmm_init
    from src.models.goal_pretrain import goal_pretrainer
    from src.trainer import trainer
    patch_goal_pretrainer(goal_pretrainer)
    patch_baseline_trainer(
        trainer, recipe["gdts_baseline"]["early_stopping"])
    argv = fixed_arguments(recipe, args.stage, args.goal_checkpoint)
    argv += ["--batch_cache_root", str(Path(args.batch_cache_root).resolve())]
    if args.stage == "goal":
        last = runtime / "output" / output_key / "runs" / recipe["goal_pretrain"]["run_name"] / "saved_models" / "goal_pretrain_last_model.pt"
        if last.is_file():
            argv += ["--load_checkpoint", "last"]
    elif args.stage == "baseline":
        last = runtime / "output" / output_key / "runs" / recipe["gdts_baseline"]["run_name"] / "saved_models" / "last_model.pt"
        if last.is_file():
            argv += ["--load_checkpoint", "last"]
    sys.argv = [str(legacy / "main.py"), *argv]
    from main import main as legacy_main
    legacy_main()
    if args.stage == "baseline":
        section = recipe["gdts_baseline"]
        model_dir = (
            runtime / "output" / output_key / "runs" /
            section["run_name"] / "saved_models")
        best = model_dir / "best_model.pt"
        last = model_dir / "last_model.pt"
        if not best.is_file() or not last.is_file():
            raise RuntimeError("matched-GDTS completed without best/last checkpoint")
        last_payload = torch.load(last, map_location="cpu", weights_only=False)
        early_state = last_payload.get("paper_early_stop_state")
        if early_state is None:
            raise RuntimeError("matched-GDTS last checkpoint lacks early-stop state")
        completed_epoch = int(last_payload["epoch"])
        contract = section["early_stopping"]
        early_stopped = (
            completed_epoch >= int(contract["earliest_stop_epoch"])
            and int(early_state["stale_validations"]) >=
            int(contract["patience_validations"])
        )
        if completed_epoch < int(section["epochs"]) and not early_stopped:
            raise RuntimeError(
                "matched-GDTS returned before max epochs without a valid "
                "early-stop decision")
        receipt = Path(recipe["pipeline_root"]) / (
            "04_matched_gdts/MATCHED_GDTS_COMPLETION.json")
        atomic_json({
            "schema": "ebjd-matched-gdts-completion-v1",
            "stage_status": "COMPLETED",
            "completion_reason": (
                "EARLY_STOP_PATIENCE" if early_stopped else "MAX_EPOCHS"),
            "planned_epochs": int(section["epochs"]),
            "completed_epochs": completed_epoch,
            "selected_checkpoint": str(best.resolve()),
            "selected_checkpoint_sha256": sha256(best),
            "last_checkpoint": str(last.resolve()),
            "last_checkpoint_sha256": sha256(last),
            "monitor": str(contract["monitor"]),
            "unit": str(contract["unit"]),
            "early_stop_state": early_state,
            "early_stop_contract": contract,
            "goal_pretrain_initialization": str(
                Path(args.goal_checkpoint).resolve()),
            "goal_pretrain_initialization_sha256": sha256(
                Path(args.goal_checkpoint)),
        }, receipt)


if __name__ == "__main__":
    main()

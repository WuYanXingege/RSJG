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
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ebjd.early_stopping import RelativeEarlyStop
from ebjd.legacy_sdd_input_pipeline import MODE as OPTIMIZED_INPUT_MODE
from ebjd.legacy_sdd_input_pipeline import PREALLOC_MODE
from ebjd.legacy_sdd_input_pipeline import install as install_sdd_input_pipeline
from ebjd.compact_sdd_heatmap_pipeline import MODE as COMPACT_HEATMAP_MODE
from ebjd.compact_sdd_heatmap_pipeline import AFFINE_MODE as COMPACT_AFFINE_MODE
from ebjd.compact_sdd_heatmap_pipeline import install as install_compact_heatmaps
from ebjd.compact_sdd_heatmap_pipeline import install_fixed_legacy_augmentation
from ebjd.compact_sdd_heatmap_pipeline import (
    append_epoch_statistics,
    pipeline_statistics_delta,
    pipeline_statistics_snapshot,
)


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


def patch_baseline_trainer(
        trainer, contract: dict, *, input_pipeline_mode: str = "legacy",
        first_epoch_receipt: Path | None = None,
        pipeline_statistics_receipt: Path | None = None) -> None:
    original_save = trainer._save_checkpoint
    original_restart = trainer._load_or_restart
    original_evaluate = trainer._evaluate_epoch
    original_train_epoch = trainer._train_epoch

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
        if (last_epoch and first_epoch_receipt is not None and
                not first_epoch_receipt.exists()):
            atomic_json({
                "schema": "ebjd-sdd-input-pipeline-first-epoch-v1",
                "epoch": int(epoch),
                "optimizer_updates": int(len(self.data_loaders["train"])),
                "input_pipeline_mode": input_pipeline_mode,
                "train_seconds": float(getattr(
                    self, "_paper_train_epoch_seconds", float("nan"))),
                "recorded_at_unix": time.time(),
            }, first_epoch_receipt)

    def train_epoch(self, epoch):
        statistics_before = (
            pipeline_statistics_snapshot()
            if input_pipeline_mode in {
                COMPACT_HEATMAP_MODE, COMPACT_AFFINE_MODE} else None)
        started = time.perf_counter()
        result = original_train_epoch(self, epoch)
        self._paper_train_epoch_seconds = time.perf_counter() - started
        if (statistics_before is not None and
                pipeline_statistics_receipt is not None):
            statistics = pipeline_statistics_delta(
                statistics_before, pipeline_statistics_snapshot())
            statistics["train_seconds"] = self._paper_train_epoch_seconds
            statistics["input_pipeline_mode"] = input_pipeline_mode
            append_epoch_statistics(
                pipeline_statistics_receipt, epoch, statistics)
            print(
                "[SDD input routes] "
                f"E{int(epoch):03d} fast={statistics['gpu_fast_batches']} "
                f"fallback={statistics['cpu_fallback_batches']} "
                f"fallback_reasons={statistics['fallback_reasons']}",
                flush=True)
        return result

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
    trainer._train_epoch = train_epoch


class _TimedLimitedLoader:
    """Expose a fixed prefix while measuring blocking iterator time."""

    def __init__(self, source, count: int):
        self.source = source
        self.count = int(count)
        self.data_wait_seconds = 0.0
        self.batch_identities = []

    def __len__(self):
        return self.count

    def __iter__(self):
        iterator = iter(self.source)
        for _ in range(self.count):
            started = time.perf_counter()
            value = next(iterator)
            self.data_wait_seconds += time.perf_counter() - started
            try:
                identity = value[1]
                def integer_list(field):
                    raw = identity[field]
                    if torch.is_tensor(raw):
                        return [int(v) for v in raw.reshape(-1).tolist()]
                    return [
                        int(v.item()) if torch.is_tensor(v) else int(v)
                        for v in raw]
                self.batch_identities.append({
                    "scene_name": str(identity["scene_name"][0]),
                    "starting_frames": integer_list("starting_frames"),
                    "agent_ids": integer_list("agent_ids"),
                })
            except Exception:
                self.batch_identities.append({"unavailable": True})
            yield value


def patch_runtime_benchmark(
        trainer, *, output: Path, warmup: int, measured: int,
        input_pipeline_mode: str,
        state_output: Path | None = None) -> None:
    """Bound the normal restored trainer to a short, timed update window."""

    def parameter_summary(self):
        total = torch.zeros((), dtype=torch.float64, device=self.device)
        squared = torch.zeros((), dtype=torch.float64, device=self.device)
        finite = True
        for parameter in self.net.parameters():
            detached = parameter.detach()
            finite = finite and bool(torch.isfinite(detached).all().item())
            total += detached.double().sum()
            squared += detached.double().square().sum()
        return {
            "finite": finite,
            "sum": float(total.item()),
            "l2": float(squared.sqrt().item()),
        }

    def benchmark_loop(self, start_epoch, end_epoch):
        del end_epoch
        source = self.data_loaders["train"]
        gradient_norms = []
        original_clip = torch.nn.utils.clip_grad_norm_

        def recorded_clip(parameters, *args, **kwargs):
            value = original_clip(parameters, *args, **kwargs)
            gradient_norms.append(float(value.detach().cpu()))
            return value

        try:
            torch.nn.utils.clip_grad_norm_ = recorded_clip
            warmup_loader = _TimedLimitedLoader(source, warmup)
            self.data_loaders["train"] = warmup_loader
            self._train_epoch(start_epoch)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
            measured_loader = _TimedLimitedLoader(source, measured)
            self.data_loaders["train"] = measured_loader
            gradient_norms.clear()
            started = time.perf_counter()
            losses = self._train_epoch(start_epoch)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - started
            try:
                import psutil
                process = psutil.Process()
                rss = process.memory_info().rss + sum(
                    child.memory_info().rss for child in process.children())
            except Exception:
                rss = None
            atomic_json({
                "schema": "ebjd-sdd-gdts-runtime-benchmark-v2",
                "input_pipeline_mode": input_pipeline_mode,
                "warmup_updates": warmup,
                "measured_updates": measured,
                "elapsed_seconds": elapsed,
                "seconds_per_update": elapsed / measured,
                "data_wait_seconds": measured_loader.data_wait_seconds,
                "batch_identities": measured_loader.batch_identities,
                "data_wait_seconds_per_update": (
                    measured_loader.data_wait_seconds / measured),
                "non_data_seconds": elapsed - measured_loader.data_wait_seconds,
                "losses": {name: float(value) for name, value in losses.items()},
                "gradient_norms": gradient_norms,
                "parameters": parameter_summary(self),
                "peak_cuda_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated(self.device))
                    if self.device.type == "cuda" else 0),
                "peak_cuda_reserved_bytes": (
                    int(torch.cuda.max_memory_reserved(self.device))
                    if self.device.type == "cuda" else 0),
                "process_tree_rss_bytes": rss,
                "main_process_peak_rss_bytes": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                "worker_processes_peak_rss_bytes": int(
                    resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss *
                    1024),
            }, output)
            if state_output is not None:
                state_output.parent.mkdir(parents=True, exist_ok=True)
                atomic_torch_save({
                    "model_state_dict": {
                        name: value.detach().cpu()
                        for name, value in self.net.state_dict().items()
                    },
                    "optimizer_state_dict": self.optimizer.state_dict(),
                }, state_output)
        finally:
            torch.nn.utils.clip_grad_norm_ = original_clip
            self.data_loaders["train"] = source

    trainer._train_loop = benchmark_loop


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
            "--num_epochs", str(section["epochs"]), "--num_workers", str(
                section.get("data_loader_workers", 2)),
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
    parser.add_argument(
        "--input-pipeline-mode",
        choices=("legacy", PREALLOC_MODE, OPTIMIZED_INPUT_MODE,
                 COMPACT_HEATMAP_MODE, COMPACT_AFFINE_MODE))
    parser.add_argument("--compact-cache-root")
    parser.add_argument("--runtime-benchmark-output")
    parser.add_argument("--runtime-benchmark-checkpoint")
    parser.add_argument("--runtime-benchmark-name")
    parser.add_argument("--runtime-benchmark-warmup", type=int, default=10)
    parser.add_argument("--runtime-benchmark-batches", type=int, default=100)
    parser.add_argument("--runtime-benchmark-state-output")
    parser.add_argument("--data-loader-workers", type=int)
    parser.add_argument(
        "--runtime-benchmark-disable-augmentation", action="store_true")
    parser.add_argument("--runtime-benchmark-fixed-augmentation-seed", type=int)
    args = parser.parse_args()
    if args.stage in ("scene_cache", "scene_cache_independent"):
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        os.nice(10)
    recipe = yaml.safe_load(Path(args.recipe).read_text(encoding="utf-8"))
    if args.runtime_benchmark_fixed_augmentation_seed is not None:
        if not args.runtime_benchmark_output:
            raise RuntimeError("fixed augmentation is benchmark-only")
        os.environ["EBJD_SDD_FIXED_AUGMENTATION_SEED"] = str(
            args.runtime_benchmark_fixed_augmentation_seed)
    input_pipeline_mode = (
        args.input_pipeline_mode or
        str(recipe.get("gdts_baseline", {}).get(
            "input_pipeline_mode", "legacy")))
    if input_pipeline_mode not in {
            "legacy", PREALLOC_MODE, OPTIMIZED_INPUT_MODE,
            COMPACT_HEATMAP_MODE, COMPACT_AFFINE_MODE}:
        raise RuntimeError(
            f"unsupported input pipeline mode: {input_pipeline_mode}")
    if (input_pipeline_mode not in {
            "legacy", COMPACT_HEATMAP_MODE, COMPACT_AFFINE_MODE}
            and not args.runtime_benchmark_output):
        raise RuntimeError(
            "rejected SDD input-pipeline candidates are benchmark-only")
    if (input_pipeline_mode == COMPACT_AFFINE_MODE and
            not args.runtime_benchmark_output):
        raise RuntimeError(
            "compact GPU affine replay did not pass the formal parameter "
            "update gate and remains benchmark-only")
    if args.runtime_benchmark_output and args.stage != "baseline":
        raise RuntimeError("runtime benchmark is baseline-only")
    if bool(args.runtime_benchmark_output) != bool(
            args.runtime_benchmark_checkpoint):
        raise RuntimeError(
            "runtime benchmark output and checkpoint are both required")
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
    if (args.stage == "baseline" and input_pipeline_mode not in
            {"legacy", COMPACT_HEATMAP_MODE, COMPACT_AFFINE_MODE}):
        from src import batch_cache_io, data_loader
        install_sdd_input_pipeline(
            data_loader, batch_cache_io, mode=input_pipeline_mode)
        print(
            f"[SDD input pipeline] mode={input_pipeline_mode}",
            flush=True,
        )
    elif (args.stage == "baseline" and input_pipeline_mode == "legacy" and
          args.runtime_benchmark_fixed_augmentation_seed is not None):
        from src import data_loader
        install_fixed_legacy_augmentation(data_loader)
    from src.models.goal_pretrain import goal_pretrainer
    if (args.stage == "baseline" and input_pipeline_mode in {
            COMPACT_HEATMAP_MODE, COMPACT_AFFINE_MODE}):
        from src import data_loader
        from src.models.model import GDTS
        compact_root = Path(
            args.compact_cache_root or
            recipe["gdts_baseline"]["compact_cache_root"]).resolve()
        receipt = compact_root / "COMPACT_CACHE_RECEIPT.json"
        if not receipt.is_file():
            raise RuntimeError(f"compact cache receipt missing: {receipt}")
        install_compact_heatmaps(
            data_loader, GDTS, compact_root, mode=input_pipeline_mode)
        print(
            f"[SDD input pipeline] mode={input_pipeline_mode} "
            f"cache={compact_root}", flush=True)
    from src.trainer import trainer
    patch_goal_pretrainer(goal_pretrainer)
    effective_workers = (
        int(args.data_loader_workers) if args.data_loader_workers is not None
        else int(recipe.get("gdts_baseline", {}).get(
            "data_loader_workers", 2)))
    first_epoch_receipt = (
        Path(recipe["pipeline_root"]) / "receipts" /
        f"INPUT_PIPELINE_FIRST_FULL_EPOCH_{input_pipeline_mode}_"
        f"w{effective_workers}.json")
    patch_baseline_trainer(
        trainer, recipe["gdts_baseline"]["early_stopping"],
        input_pipeline_mode=input_pipeline_mode,
        first_epoch_receipt=first_epoch_receipt,
        pipeline_statistics_receipt=(
            Path(args.runtime_benchmark_output).resolve().with_suffix(
                ".pipeline_stats.json")
            if args.runtime_benchmark_output else
            Path(recipe["pipeline_root"]) /
            "receipts/INPUT_PIPELINE_EPOCH_STATISTICS.json"))
    if args.runtime_benchmark_output:
        patch_runtime_benchmark(
            trainer,
            output=Path(args.runtime_benchmark_output).resolve(),
            warmup=args.runtime_benchmark_warmup,
            measured=args.runtime_benchmark_batches,
            input_pipeline_mode=input_pipeline_mode,
            state_output=(
                Path(args.runtime_benchmark_state_output).resolve()
                if args.runtime_benchmark_state_output else None))
    argv = fixed_arguments(recipe, args.stage, args.goal_checkpoint)
    if args.data_loader_workers is not None:
        if args.data_loader_workers not in {2, 4}:
            raise RuntimeError("this comparison only permits 2 or 4 workers")
        index = argv.index("--num_workers")
        argv[index + 1] = str(args.data_loader_workers)
    if args.runtime_benchmark_disable_augmentation:
        index = argv.index("--data_augmentation")
        argv[index + 1] = "False"
    if args.runtime_benchmark_output:
        benchmark_name = (
            args.runtime_benchmark_name or
            f"sdd_gdts_input_benchmark_{input_pipeline_mode}")
        index = argv.index("--run_name")
        argv[index + 1] = benchmark_name
    argv += ["--batch_cache_root", str(Path(args.batch_cache_root).resolve())]
    if args.stage == "goal":
        last = runtime / "output" / output_key / "runs" / recipe["goal_pretrain"]["run_name"] / "saved_models" / "goal_pretrain_last_model.pt"
        if last.is_file():
            argv += ["--load_checkpoint", "last"]
    elif args.stage == "baseline":
        run_name = (
            args.runtime_benchmark_name or
            f"sdd_gdts_input_benchmark_{input_pipeline_mode}"
            if args.runtime_benchmark_output else
            recipe["gdts_baseline"]["run_name"])
        last = (
            runtime / "output" / output_key / "runs" / run_name /
            "saved_models/last_model.pt")
        if args.runtime_benchmark_output:
            source = Path(args.runtime_benchmark_checkpoint).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            last.parent.mkdir(parents=True, exist_ok=True)
            if last.exists() or last.is_symlink():
                last.unlink()
            last.symlink_to(source)
        if last.is_file():
            argv += ["--load_checkpoint", "last"]
    sys.argv = [str(legacy / "main.py"), *argv]
    from main import main as legacy_main
    legacy_main()
    if args.stage == "baseline" and not args.runtime_benchmark_output:
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

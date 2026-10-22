#!/usr/bin/env python3
"""Low-overhead one-update rollout timing using existing record ranges."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
import time
from pathlib import Path

import torch

import ebjd.denoiser as denoiser_module
import ebjd.sampling as sampling_module
import ebjd.trainer as trainer_module

from benchmark_training_speed import (
    atomic_json, baseline_config, collect_microbatches, load_config, _loader,
)
from check_baseline_repeatability import update_ledger
from ebjd.config import build_model, trainer_options
from ebjd.trainer import EBJDTrainer


class CudaRangeRecorder:
    """Record only named CUDA-event boundaries; never enumerate operators."""

    def __init__(self, wanted: set[str]) -> None:
        self.wanted = wanted
        self.events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = (
            defaultdict(list))

    def range(self, name: str):
        recorder = self

        class Range:
            def __enter__(self):
                if name in recorder.wanted:
                    self.start = torch.cuda.Event(enable_timing=True)
                    self.end = torch.cuda.Event(enable_timing=True)
                    self.start.record()
                return self

            def __exit__(self, exc_type, exc, traceback):
                if name in recorder.wanted:
                    self.end.record()
                    recorder.events[name].append((self.start, self.end))
                return False

        return Range()

    def rows(self) -> list[dict]:
        rows = []
        for name, events in self.events.items():
            rows.append({
                "region": name,
                "calls": len(events),
                "cuda_total_ms": sum(
                    start.elapsed_time(end) for start, end in events),
            })
        return sorted(rows, key=lambda item: item["cuda_total_ms"], reverse=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/resolved/hotel.json")
    parser.add_argument(
        "--checkpoint", default="assets/checkpoints/hotel/epoch_045_full.pt")
    parser.add_argument("--epoch", type=int, default=46)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-state", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    state_path = Path(args.task_state)
    started_wall = time.time()
    started = time.perf_counter()
    active = {
        "name": "single_rollout_update_low_overhead_profile",
        "started_unix": started_wall,
    }
    update_ledger(state_path, status="GPU_RUNNING", active_call=active)
    outcome = "FAILED"
    failure = None
    try:
        if not torch.cuda.is_available() or not args.device.startswith("cuda"):
            raise RuntimeError("this profile requires an available CUDA device")
        device = torch.device(args.device)
        source = load_config(args.config)
        config = baseline_config(source)
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        generator = torch.Generator()
        model = build_model(config).to(device)
        trainer = EBJDTrainer(model, trainer_options(config), config, {})
        trainer.restore_checkpoint_payload(
            payload, loader_generator=generator, require_identity=False)
        model.train()
        loader = _loader(config, "train", False, generator)
        iterator = iter(loader)
        accumulation = int(config["training"]["gradient_accumulation"])

        # Exact real updates advance to the next rollout boundary.  They count
        # toward GPU cost but are outside the profiled timing table.
        while trainer.update_index % trainer.options.rollout_every:
            batches, _, _, _ = collect_microbatches(
                iterator, generator, accumulation)
            trainer.train_step(batches, args.epoch)
            del batches
        batches, data_seconds, audit_seconds, batch_sha = collect_microbatches(
            iterator, generator, accumulation)
        wanted = {
            "ebjd.context_encoder",
            "ebjd.base_diffusion_and_losses",
            "ebjd.rollout_forward",
            "ebjd.grad_marginal_ade",
            "ebjd.grad_marginal_fde",
            "ebjd.grad_total",
            "ebjd.adamw_proposal",
            "ebjd.fp64_qp_projection",
            "ebjd.optimizer_apply",
            "ebjd.ddim_step_checkpoint",
            "ebjd.denoiser_block_checkpoint",
        }
        recorder = CudaRangeRecorder(wanted)
        original_ranges = (
            trainer_module.record_function,
            sampling_module.record_function,
            denoiser_module.record_function,
        )
        trainer_module.record_function = recorder.range
        sampling_module.record_function = recorder.range
        denoiser_module.record_function = recorder.range
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        profile_start = time.perf_counter()
        try:
            record = trainer.train_step(batches, args.epoch)
            torch.cuda.synchronize(device)
        finally:
            (
                trainer_module.record_function,
                sampling_module.record_function,
                denoiser_module.record_function,
            ) = original_ranges
        profiled_wall = time.perf_counter() - profile_start
        rows = recorder.rows()
        result = {
            "schema": "ebjd-rollout-low-overhead-profile-v1",
            "formal_training_artifact": False,
            "rollout_update": bool(record["rollout_update"]),
            "source_update_index": trainer.update_index - 1,
            "batch_sha256": batch_sha,
            "data_seconds_excluded": data_seconds,
            "audit_hash_seconds_excluded": audit_seconds,
            "profiled_wall_seconds_high_overhead_excluded_from_speed_claims": profiled_wall,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "regions": rows,
            "interpretation": {
                "ddim_step_checkpoint_active": any(
                    row["region"] == "ebjd.ddim_step_checkpoint" for row in rows),
                "denoiser_block_checkpoint_event_present": any(
                    row["region"] == "ebjd.denoiser_block_checkpoint" for row in rows),
                "rollout_block_checkpointing": bool(
                    trainer.options.rollout_block_checkpointing),
                "nested_step_and_block_checkpoint": bool(
                    trainer.options.rollout_block_checkpointing),
                "block_event_scope": (
                    "base diffusion denoiser only when rollout_block_checkpointing=false"),
            },
        }
        atomic_json(output_path, result)
        outcome = "PASS"
        print(json.dumps({
            "output": str(output_path.resolve()),
            "profiled_wall_seconds": profiled_wall,
            "regions": rows,
        }, sort_keys=True), flush=True)
        del trainer, model, loader, iterator, batches, recorder
        gc.collect()
        torch.cuda.empty_cache()
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        update_ledger(state_path, status="CPU_ANALYSIS", call={
            "name": active["name"],
            "started_unix": started_wall,
            "wall_seconds": time.perf_counter() - started,
            "outcome": outcome,
            "failure": failure,
            "output": str(output_path.resolve()),
        })


if __name__ == "__main__":
    main()

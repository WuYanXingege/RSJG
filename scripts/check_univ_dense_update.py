#!/usr/bin/env python3
"""One temporary dense UNIV rollout update for memory-safety validation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from benchmark_rollout_candidate import candidate_config
from benchmark_training_speed import atomic_json, load_config
from check_baseline_repeatability import update_ledger
from ebjd.config import build_model, trainer_options
from ebjd.data import SceneManifestDataset, augment_batch, collate_scenes
from ebjd.trainer import EBJDTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/resolved/univ.json")
    parser.add_argument(
        "--checkpoint", default="assets/checkpoints/univ/epoch_073_selected_formal.pt")
    parser.add_argument("--epoch", type=int, default=74)
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
    active = {"name": "univ_dense_real_update", "started_unix": started_wall}
    update_ledger(state_path, status="GPU_RUNNING", active_call=active)
    outcome = "FAILED"
    failure = None
    try:
        if not torch.cuda.is_available() or not args.device.startswith("cuda"):
            raise RuntimeError("this check requires an available CUDA device")
        source = load_config(args.config)
        config = candidate_config(source, "small-group")
        manifest_path = Path(config["paths"]["train_manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        counts = []
        for index, descriptor in enumerate(manifest["shards"]):
            with np.load(
                manifest_path.parent / descriptor["path"], allow_pickle=True,
            ) as archive:
                count = int(np.asarray(archive["observed"][0]).shape[0])
            counts.append((count, index))
        selected = [index for _, index in sorted(counts, reverse=True)[:32]]
        dataset = SceneManifestDataset(manifest_path, verify_hashes=False)
        generator = torch.Generator()
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        device = torch.device(args.device)
        model = build_model(config).to(device)
        trainer = EBJDTrainer(model, trainer_options(config), config, {})
        trainer.restore_checkpoint_payload(
            payload, loader_generator=generator, require_identity=False)
        model.train()

        def make_update(indices: list[int]):
            batches = []
            for offset in range(0, 16, 4):
                batch = collate_scenes([
                    dataset[item] for item in indices[offset:offset + 4]])
                batches.append(augment_batch(batch, generator))
            return batches

        # Use the lower half only to align the checkpoint to the rollout
        # boundary; reserve the densest real scenes for the measured ON update.
        first = make_update(selected[16:32])
        # epoch73/update18615 is one OFF update before a rollout boundary.
        if trainer.update_index % trainer.options.rollout_every == 0:
            raise ValueError("expected the reference checkpoint to precede rollout by one update")
        alignment = trainer.train_step(first, args.epoch)
        if alignment["rollout_update"]:
            raise ValueError("alignment update unexpectedly enabled rollout")
        second = make_update(selected[:16])
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        update_start = time.perf_counter()
        record = trainer.train_step(second, args.epoch)
        torch.cuda.synchronize(device)
        update_seconds = time.perf_counter() - update_start
        if not record["rollout_update"]:
            raise ValueError("dense validation update did not enable rollout")
        result = {
            "schema": "ebjd-univ-dense-rollout-memory-check-v1",
            "formal_training_artifact": False,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_epoch": int(payload["epoch"]),
            "source_update": int(payload["update_index"]),
            "tested_update": trainer.update_index - 1,
            "selected_real_scene_indices": selected,
            "microbatch_agent_counts": [
                batch.valid.sum(dim=1).tolist() for batch in second],
            "rollout_update_seconds": update_seconds,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "rollout_step_checkpointed_group_fraction": record[
                "rollout_step_checkpointed_group_fraction"],
            "maximum_physical_agent_slots": record[
                "maximum_physical_agent_slots"],
            "loss": record["loss"],
            "finite": all(np.isfinite(value) for value in (
                record["loss"], record["rollout"],
                record["candidate_decrement_norm"],
                record["applied_decrement_norm"])),
        }
        # Record the actual selected counts without relying on manifest order.
        count_by_index = {index: count for count, index in counts}
        result["selected_agent_counts"] = [count_by_index[index] for index in selected]
        atomic_json(output_path, result)
        outcome = "PASS"
        print(json.dumps({
            "output": str(output_path.resolve()),
            "update_seconds": update_seconds,
            "peak_allocated_bytes": result["peak_allocated_bytes"],
            "peak_reserved_bytes": result["peak_reserved_bytes"],
            "checkpointed_fraction": result[
                "rollout_step_checkpointed_group_fraction"],
        }, sort_keys=True), flush=True)
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

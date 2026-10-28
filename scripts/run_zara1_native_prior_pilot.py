#!/usr/bin/env python3
"""Strictly paired short ZARA1 native-prior conditional-bridge pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ebjd.config import build_model, load_config, trainer_options
from ebjd.data import (
    SceneManifestDataset, augment_batch_with_points, collate_scenes,
)
from ebjd.metrics import MetricAccumulator
from ebjd.proposals import NativeProposalBank
from ebjd.sampling import predict
from ebjd.trainer import EBJDTrainer, ProposalTrainingCondition


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loader(config: dict, split: str, generator: torch.Generator) -> DataLoader:
    dataset = SceneManifestDataset(config["paths"][f"{split}_manifest"])
    workers = int(config["training"].get("data_loader_workers", 0))
    arguments = dict(
        dataset=dataset,
        # The frozen endpoint attribution used one complete scene per
        # validation batch.  Preserve that RNG-to-scene mapping exactly;
        # training retains the formal four-scene logical microbatch.
        batch_size=(int(config["training"]["batch_scenes"])
                    if split == "train" else 1),
        shuffle=split == "train", collate_fn=collate_scenes,
        num_workers=workers, generator=generator)
    if workers:
        arguments.update(
            prefetch_factor=int(config["training"].get(
                "data_loader_prefetch_factor", 2)), persistent_workers=True)
    return DataLoader(**arguments)


def convert_parent(model, checkpoint: Path) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    parent = payload["model"]
    target = model.state_dict()
    additions = [name for name in target if name not in parent]
    expected_additions = [
        "denoiser.proposal_embedding.0.weight",
        "denoiser.proposal_embedding.0.bias",
        "denoiser.proposal_embedding.2.weight",
        "denoiser.proposal_embedding.2.bias",
    ]
    if additions != expected_additions or any(name not in target for name in parent):
        raise ValueError(
            f"parent conversion key-space mismatch: additions={additions}")
    for name, value in parent.items():
        if target[name].shape != value.shape:
            raise ValueError(f"parent conversion shape mismatch: {name}")
        target[name].copy_(value)
    model.load_state_dict(target, strict=True)
    inherited_exact = all(
        torch.equal(model.state_dict()[name].cpu(), value.cpu())
        for name, value in parent.items())
    final_zero = (
        torch.count_nonzero(model.denoiser.proposal_embedding[-1].weight) == 0
        and torch.count_nonzero(model.denoiser.proposal_embedding[-1].bias) == 0)
    if not inherited_exact or not final_zero:
        raise RuntimeError("proposal-model parent conversion audit failed")
    return {
        "origin_epoch": int(payload["epoch"]),
        "origin_update": int(payload["update_index"]),
        "inherited_tensor_count": len(parent),
        "inherited_exact": inherited_exact,
        "added_keys": additions, "proposal_embedding_final_zero": bool(final_zero),
    }


@torch.no_grad()
def evaluate(
    model, validation_loader, bank: NativeProposalBank, device: torch.device,
    eta: float, seed: int, worlds: int, steps: int,
) -> dict:
    model.eval()
    overall = MetricAccumulator()
    single = MetricAccumulator()
    multi = MetricAccumulator()
    generator = torch.Generator(device=device).manual_seed(seed)
    recall_hits = recall_count = 0
    corrections = []
    endpoint_max_abs = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for cpu_batch in validation_loader:
        proposals = bank.batch(cpu_batch).to(device)
        batch = cpu_batch.to(device)
        prediction, goals = predict(
            model, batch.observed, batch.semantic_maps, batch.valid,
            worlds=worlds, steps=steps, generator=generator,
            proposal_goals=proposals, proposal_eta=eta)
        overall.update(prediction, batch.future, batch.valid)
        counts = batch.valid.sum(1)
        for selector, accumulator in ((counts == 1, single), (counts >= 2, multi)):
            if selector.any():
                accumulator.update(
                    prediction[selector], batch.future[selector], batch.valid[selector])
        endpoint_error = torch.linalg.vector_norm(
            goals - batch.future[:, None, :, -1, :], dim=-1)
        best = endpoint_error.min(1).values
        recall_hits += int(((best <= 0.5) & batch.valid).sum())
        recall_count += int(batch.valid.sum())
        valid_world = batch.valid[:, None].expand_as(endpoint_error)
        corrections.append(torch.linalg.vector_norm(
            goals - proposals, dim=-1)[valid_world].float().cpu())
        if eta == 0.0:
            endpoint_max_abs = max(
                endpoint_max_abs, float((goals - proposals).abs().max().cpu()))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    correction = torch.cat(corrections)
    result = {
        "seed": seed, "eta": eta, "metrics": overall.as_dict(),
        "N_eq_1": single.as_dict(), "N_ge_2": multi.as_dict(),
        "recall_at_0_5m": recall_hits / recall_count,
        "goal_correction_p50_m": float(correction.quantile(0.50)),
        "goal_correction_p95_m": float(correction.quantile(0.95)),
        "eta0_endpoint_max_abs_m": endpoint_max_abs,
        "elapsed_seconds": elapsed,
        "peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else 0),
    }
    if eta == 0.0 and endpoint_max_abs != 0.0:
        raise RuntimeError("eta=0 endpoint is not the literal proposal")
    return result


def build_trainer(config: dict, device: torch.device, run_identity: dict):
    model = build_model(config, proposal_conditioned=True).to(device)
    options = trainer_options(config)
    options.epochs = 9
    options.new_module_lr = 2e-5
    options.unet_lr = 2e-6
    options.constant_learning_rate = True
    options.use_rollout_loss = False
    options.use_actual_step_constraint = False
    trainer = EBJDTrainer(model, options, config, run_identity)
    return model, trainer, options


def train_epochs(
    label: str, model, trainer: EBJDTrainer, options,
    config: dict, train_bank: NativeProposalBank,
    validation_bank: NativeProposalBank, validation_loader,
    loader_generator: torch.Generator, start_epoch: int, count: int,
    eta: float, trust_weight: float, output: Path, device: torch.device,
) -> list[dict]:
    records = []
    for local_epoch in range(1, count + 1):
        epoch = start_epoch + local_epoch - 1
        model.train()
        pending_batches = []
        pending_conditions = []
        losses = []
        started = time.perf_counter()
        world_histogram = torch.zeros(20, dtype=torch.long)
        for raw_batch in loader(config, "train", loader_generator):
            proposals = train_bank.batch(raw_batch)
            augmented, augmented_proposals = augment_batch_with_points(
                raw_batch, proposals, loader_generator)
            selected, indices = NativeProposalBank.select_training_world(
                augmented_proposals, augmented.future, augmented.valid)
            world_histogram += torch.bincount(indices, minlength=20)
            pending_batches.append(augmented)
            pending_conditions.append(ProposalTrainingCondition(
                selected, eta=eta, trust_weight=trust_weight))
            if len(pending_batches) == options.accumulation:
                log = trainer.train_step(
                    pending_batches, epoch, pending_conditions)
                losses.append(log["loss"])
                with (output / "train.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        **log, "stage": label, "stage_epoch": local_epoch,
                        "epoch_equivalent": epoch, "update": trainer.update_index,
                    }, sort_keys=True) + "\n")
                pending_batches, pending_conditions = [], []
        if pending_batches:
            log = trainer.train_step(pending_batches, epoch, pending_conditions)
            losses.append(log["loss"])
            with (output / "train.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    **log, "stage": label, "stage_epoch": local_epoch,
                    "epoch_equivalent": epoch, "update": trainer.update_index,
                }, sort_keys=True) + "\n")
        metrics = evaluate(
            model, validation_loader, validation_bank, device, eta,
            seed=2035, worlds=20, steps=20)
        elapsed = time.perf_counter() - started
        record = {
            "stage": label, "stage_epoch": local_epoch,
            "epoch_equivalent": epoch, "update": trainer.update_index,
            "loss": statistics.fmean(losses), "validation": metrics,
            "elapsed_seconds": elapsed,
            "training_world_histogram": world_histogram.tolist(),
        }
        records.append(record)
        atomic_json(output / "epoch_state.json", record)
        print(
            f"[{label} {local_epoch}/{count}] loss={record['loss']:.6f} "
            f"minADE={metrics['metrics']['minADE']:.6f} "
            f"minFDE={metrics['metrics']['minFDE']:.6f} "
            f"JADE={metrics['metrics']['JADE']:.6f} "
            f"JFDE={metrics['metrics']['JFDE']:.6f} "
            f"time={elapsed/60:.2f}min", flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--pilot-config", required=True)
    parser.add_argument("--train-bank", required=True)
    parser.add_argument("--validation-bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_config(args.model_config)
    pilot = yaml.safe_load(Path(args.pilot_config).read_text(encoding="utf-8"))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if int(config["denoiser"]["hidden_dim"]) != 192 or int(config["denoiser"]["ff_dim"]) != 768:
        raise ValueError("pilot requires the frozen 192/4/768 parent architecture")
    if sha256(pilot["parent_checkpoint"]) != pilot["parent_checkpoint_sha256"]:
        raise ValueError("parent checkpoint SHA256 mismatch")
    train_manifest_sha = sha256(pilot["train_manifest"])
    train_bank = NativeProposalBank(
        args.train_bank,
        expected_checkpoint_sha256=pilot["gdts_checkpoint_sha256"],
        expected_manifest_sha256=train_manifest_sha, expected_split="train")
    validation_bank = NativeProposalBank(
        args.validation_bank,
        expected_checkpoint_sha256=pilot["gdts_checkpoint_sha256"],
        expected_manifest_sha256=pilot["validation_manifest_sha256"],
        expected_split="validation")
    seed_all(int(pilot["training_seed"]))
    loader_generator = torch.Generator().manual_seed(int(pilot["training_seed"]))
    identity = {
        "schema": pilot["schema"],
        "fold": str(config["experiment"]["fold"]),
        "seed": int(pilot["training_seed"]),
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True).stdout.strip(),
        "parent_sha256": pilot["parent_checkpoint_sha256"],
        "train_proposal_bank_sha256": train_bank.sha256,
        "validation_proposal_bank_sha256": validation_bank.sha256,
    }
    model, trainer, options = build_trainer(config, device, identity)
    conversion = convert_parent(model, Path(pilot["parent_checkpoint"]))
    validation_generator = torch.Generator().manual_seed(99173)
    validation_loader = loader(config, "validation", validation_generator)
    atomic_json(output / "RUN_IDENTITY.json", {
        **identity, "parent_conversion": conversion,
        "pilot_config_sha256": sha256(args.pilot_config),
        "model_config_sha256": sha256(args.model_config),
        "rollout_updates": 0, "projection_updates": 0,
    })
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    pilot_started = time.perf_counter()
    stage_a_dir = output / "stage_a"
    stage_a_dir.mkdir(exist_ok=True)
    stage_a_records = train_epochs(
        "StageA", model, trainer, options, config, train_bank,
        validation_bank, validation_loader, loader_generator,
        1, int(pilot["stage_a_epochs"]), 0.0, 0.0, stage_a_dir, device)
    trainer.save_checkpoint(
        stage_a_dir / "last.pt", 3, {"fixed_stage": "StageA"},
        loader_generator, {"records": stage_a_records})
    stage_payload = torch.load(
        stage_a_dir / "last.pt", map_location="cpu", weights_only=False)
    arm_definitions = {
        "Control": (0.0, 0.0),
        "Candidate": (1.0, float(pilot["candidate_trust_weight"])),
    }
    requested_arms = pilot.get("arms", ["Control", "Candidate"])
    if not requested_arms or any(name not in arm_definitions for name in requested_arms):
        raise ValueError(f"unsupported pilot arms: {requested_arms}")
    arm_results = {}
    for label in requested_arms:
        eta, trust = arm_definitions[label]
        arm_dir = output / label.lower()
        arm_dir.mkdir(exist_ok=True)
        arm_generator = torch.Generator().manual_seed(0)
        arm_model, arm_trainer, arm_options = build_trainer(config, device, identity)
        arm_trainer.restore_checkpoint_payload(
            stage_payload, arm_generator, require_identity=True)
        records = train_epochs(
            label, arm_model, arm_trainer, arm_options, config, train_bank,
            validation_bank, validation_loader, arm_generator,
            4, int(pilot["arm_epochs"]), eta, trust, arm_dir, device)
        arm_trainer.save_checkpoint(
            arm_dir / "fixed_eplus3.pt", 6,
            {"fixed_stage": label, "not_best_selected": True}, arm_generator,
            {"records": records, "eta": eta, "trust_weight": trust})
        arm_results[label] = {
            "records": records,
            "checkpoint": str((arm_dir / "fixed_eplus3.pt").resolve()),
            "checkpoint_sha256": sha256(arm_dir / "fixed_eplus3.pt"),
            "final_seed2035": records[-1]["validation"],
            "update_index": arm_trainer.update_index,
        }
        del arm_trainer, arm_model
        torch.cuda.empty_cache()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    gpu_elapsed = time.perf_counter() - pilot_started
    result = {
        "schema": "ebjd-native-prior-conditional-bridge-pilot-result-v1",
        "status": "SEED2035_COMPLETE", "identity": identity,
        "parent_conversion": conversion,
        "stage_a": {"records": stage_a_records,
                    "checkpoint": str((stage_a_dir / "last.pt").resolve()),
                    "checkpoint_sha256": sha256(stage_a_dir / "last.pt"),
                    "update_index": trainer.update_index},
        "arms": arm_results, "optimizer_updates": {
            "stage_a": trainer.update_index,
            "control_total_including_stage_a": arm_results["Control"]["update_index"],
            "candidate_total_including_stage_a": arm_results["Candidate"]["update_index"],
        },
        "gpu_elapsed_seconds_excluding_proposal_export": gpu_elapsed,
        "peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else 0),
    }
    atomic_json(output / "SEED2035_RESULT.json", result)
    print(json.dumps({
        "status": result["status"], "gpu_seconds": gpu_elapsed,
        "arms": {
            name: value["final_seed2035"]["metrics"]
            for name, value in arm_results.items()
        },
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Three-epoch paired native-vs-MPJC conditional bridge pilot."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch

from ebjd.config import build_model, load_config, trainer_options
from ebjd.coupling import MPJCEndpointBank, gather_worlds
from ebjd.data import augment_batch_with_points
from ebjd.metrics import MetricAccumulator
from ebjd.proposals import NativeProposalBank, file_sha256
from ebjd.sampling import predict
from ebjd.trainer import EBJDTrainer, ProposalTrainingCondition
from run_zara1_native_prior_pilot import atomic_json, loader, seed_all


def build_fresh_arm(config: dict, checkpoint: Path, expected_sha: str, device, identity):
    if file_sha256(checkpoint) != expected_sha:
        raise ValueError("Candidate checkpoint SHA256 mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(config, proposal_conditioned=True).to(device)
    model.load_state_dict(payload["model"], strict=True)
    options = trainer_options(config)
    options.epochs = int(payload["epoch"]) + 3
    options.new_module_lr = 2e-5
    options.unet_lr = 2e-6
    options.constant_learning_rate = True
    options.use_rollout_loss = False
    options.use_actual_step_constraint = False
    trainer = EBJDTrainer(model, options, config, identity)
    if trainer.update_index != 0 or trainer.optimizer.state:
        raise RuntimeError("MPJC arm optimizer is not fresh")
    return model, trainer, options, payload


@torch.no_grad()
def evaluate(
    model, config: dict, bank: MPJCEndpointBank, mode: str,
    device: torch.device, seed: int,
) -> dict:
    model.eval()
    overall, single, multi = MetricAccumulator(), MetricAccumulator(), MetricAccumulator()
    generator = torch.Generator(device=device).manual_seed(seed)
    recall_hits = recall_count = 0
    endpoint_max_abs = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for cpu_batch in loader(config, "validation", torch.Generator().manual_seed(99173)):
        proposals, permutation = bank.batch(cpu_batch, mode)
        batch = cpu_batch.to(device)
        proposals = proposals.to(device)
        permutation = permutation.to(device)
        base_noise = torch.randn(
            batch.observed.shape[0], 20, batch.observed.shape[1], 12, 2,
            device=device, dtype=torch.float32, generator=generator)
        noise = base_noise if mode == "native" else gather_worlds(base_noise, permutation)
        prediction, goals = predict(
            model, batch.observed, batch.semantic_maps, batch.valid,
            worlds=20, steps=20, proposal_goals=proposals, proposal_eta=0.0,
            initial_noise=noise)
        overall.update(prediction, batch.future, batch.valid)
        counts = batch.valid.sum(1)
        for selector, accumulator in ((counts == 1, single), (counts >= 2, multi)):
            if selector.any():
                accumulator.update(
                    prediction[selector], batch.future[selector], batch.valid[selector])
        endpoint_error = torch.linalg.vector_norm(
            goals - batch.future[:, None, :, -1], dim=-1)
        best = endpoint_error.min(1).values
        recall_hits += int(((best <= 0.5) & batch.valid).sum())
        recall_count += int(batch.valid.sum())
        endpoint_max_abs = max(
            endpoint_max_abs, float((goals - proposals).abs().nan_to_num().max().cpu()))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if endpoint_max_abs != 0.0:
        raise RuntimeError("eta=0 bridge did not preserve its fixed endpoints")
    return {
        "seed": seed, "mode": mode, "metrics": overall.as_dict(),
        "N_eq_1": single.as_dict(), "N_ge_2": multi.as_dict(),
        "recall_at_0_5m": recall_hits / recall_count,
        "endpoint_max_abs_m": endpoint_max_abs,
        "elapsed_seconds": time.perf_counter() - started,
    }


def train_arm(
    label: str, mode: str, config: dict, checkpoint: Path,
    candidate_sha: str, train_bank: MPJCEndpointBank,
    validation_bank: MPJCEndpointBank, output: Path, device: torch.device,
    identity: dict,
) -> dict:
    seed_all(3101)
    loader_generator = torch.Generator().manual_seed(3101)
    model, trainer, options, parent = build_fresh_arm(
        config, checkpoint, candidate_sha, device, identity)
    parent_epoch = int(parent["epoch"])
    records = []
    arm_dir = output / label.lower()
    arm_dir.mkdir(parents=True, exist_ok=True)
    for local_epoch in range(1, 4):
        epoch = parent_epoch + local_epoch
        model.train()
        pending_batches, pending_conditions, losses = [], [], []
        world_histogram = torch.zeros(20, dtype=torch.long)
        started = time.perf_counter()
        for raw_batch in loader(config, "train", loader_generator):
            proposals, _ = train_bank.batch(raw_batch, mode)
            augmented, augmented_proposals = augment_batch_with_points(
                raw_batch, proposals, loader_generator)
            selected, indices = NativeProposalBank.select_training_world(
                augmented_proposals, augmented.future, augmented.valid)
            world_histogram += torch.bincount(indices, minlength=20)
            pending_batches.append(augmented)
            pending_conditions.append(ProposalTrainingCondition(
                selected, eta=0.0, trust_weight=0.0))
            if len(pending_batches) == options.accumulation:
                log = trainer.train_step(pending_batches, epoch, pending_conditions)
                losses.append(log["loss"])
                with (arm_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        **log, "arm": label, "arm_epoch": local_epoch,
                        "epoch_equivalent": epoch,
                    }, sort_keys=True) + "\n")
                pending_batches, pending_conditions = [], []
        if pending_batches:
            log = trainer.train_step(pending_batches, epoch, pending_conditions)
            losses.append(log["loss"])
            with (arm_dir / "train.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    **log, "arm": label, "arm_epoch": local_epoch,
                    "epoch_equivalent": epoch,
                }, sort_keys=True) + "\n")
        metrics = evaluate(model, config, validation_bank, mode, device, 2035)
        record = {
            "arm": label, "mode": mode, "arm_epoch": local_epoch,
            "epoch_equivalent": epoch, "update": trainer.update_index,
            "loss": statistics.fmean(losses), "validation": metrics,
            "elapsed_seconds": time.perf_counter() - started,
            "training_world_histogram": world_histogram.tolist(),
        }
        records.append(record)
        atomic_json(arm_dir / "epoch_state.json", record)
        print(
            f"[{label} {local_epoch}/3] loss={record['loss']:.6f} "
            f"minADE={metrics['metrics']['minADE']:.6f} "
            f"minFDE={metrics['metrics']['minFDE']:.6f} "
            f"JADE={metrics['metrics']['JADE']:.6f} "
            f"JFDE={metrics['metrics']['JFDE']:.6f} "
            f"time={record['elapsed_seconds']/60:.2f}min", flush=True)
    checkpoint_path = arm_dir / "fixed_eplus3.pt"
    trainer.save_checkpoint(
        checkpoint_path, parent_epoch + 3,
        {"fixed_stage": label, "not_best_selected": True}, loader_generator,
        {"records": records, "proposal_eta": 0.0, "trust_weight": 0.0})
    return {
        "label": label, "mode": mode, "records": records,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "fixed_seed2035": records[-1]["validation"],
        "optimizer_updates": trainer.update_index,
        "parent_epoch": parent_epoch,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--train-mpjc-bank", required=True)
    parser.add_argument("--validation-mpjc-bank", required=True)
    parser.add_argument("--train-manifest-sha256", required=True)
    parser.add_argument("--validation-manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_config(args.model_config)
    device = torch.device(args.device)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_bank = MPJCEndpointBank(
        args.train_mpjc_bank,
        expected_candidate_sha256=args.candidate_sha256,
        expected_manifest_sha256=args.train_manifest_sha256,
        expected_split="train")
    validation_bank = MPJCEndpointBank(
        args.validation_mpjc_bank,
        expected_candidate_sha256=args.candidate_sha256,
        expected_manifest_sha256=args.validation_manifest_sha256,
        expected_split="validation")
    identity = {
        "schema": "ebjd-zara1-mpjc-bridge-pilot-v1", "fold": "zara1",
        "seed": 3101, "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True).stdout.strip(),
        "candidate_parent_sha256": args.candidate_sha256,
        "train_mpjc_bank_sha256": train_bank.sha256,
        "validation_mpjc_bank_sha256": validation_bank.sha256,
    }
    atomic_json(output / "RUN_IDENTITY.json", identity)
    started = time.perf_counter()
    results = {}
    for label, mode in (("BRIDGE_NATIVE", "native"), ("BRIDGE_COUPLED", "coupled")):
        results[label] = train_arm(
            label, mode, config, Path(args.candidate_checkpoint),
            args.candidate_sha256, train_bank, validation_bank, output,
            device, identity)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = {
        "schema": "ebjd-zara1-mpjc-bridge-pilot-result-v1",
        "status": "SEED2035_COMPLETE", "identity": identity,
        "arms": results, "gpu_elapsed_seconds": time.perf_counter() - started,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda" else 0,
    }
    atomic_json(output / "SEED2035_RESULT.json", result)
    print(json.dumps({
        "status": result["status"], "gpu_seconds": result["gpu_elapsed_seconds"],
        "native": results["BRIDGE_NATIVE"]["fixed_seed2035"]["metrics"],
        "coupled": results["BRIDGE_COUPLED"]["fixed_seed2035"]["metrics"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

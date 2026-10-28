#!/usr/bin/env python3
"""Train the fixed three-epoch GT-conditioned bridge adaptation."""

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
from ebjd.data import augment_batch_with_gt_endpoint_condition
from ebjd.metrics import MetricAccumulator, per_world_errors
from ebjd.proposals import file_sha256
from ebjd.sampling import predict
from ebjd.trainer import EBJDTrainer, ProposalTrainingCondition
from run_zara1_native_prior_pilot import atomic_json, loader, seed_all


class PrefixAccumulator:
    """Formal agent minADE and scene JADE restricted to frames 1--11."""

    def __init__(self) -> None:
        self.agent_sum = 0.0
        self.agent_count = 0
        self.scene_sum = 0.0
        self.scene_count = 0

    def update(self, prediction, target, valid) -> None:
        distance = torch.linalg.vector_norm(
            prediction[..., :11, :].float() - target[:, None, :, :11].float(), dim=-1)
        ade = distance.mean(-1).permute(0, 2, 1)
        counts = valid.sum(1).clamp_min(1).float()
        self.agent_sum += float(ade.min(2).values[valid].double().sum())
        self.agent_count += int(valid.sum())
        scene = ((ade * valid[..., None]).sum(1) / counts[:, None]).min(1).values
        self.scene_sum += float(scene.double().sum())
        self.scene_count += int(scene.numel())

    def as_dict(self) -> dict:
        return {
            "prefix_minADE_11": self.agent_sum / self.agent_count,
            "prefix_JADE_11": self.scene_sum / self.scene_count,
            "agent_count": self.agent_count, "scene_count": self.scene_count,
        }


def build_fresh(config, checkpoint: Path, expected_sha: str, device, identity):
    if file_sha256(checkpoint) != expected_sha:
        raise ValueError("Candidate checkpoint SHA256 mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(config, proposal_conditioned=True).to(device)
    model.load_state_dict(payload["model"], strict=True)
    options = trainer_options(config)
    options.epochs = int(payload["epoch"]) + 3
    options.new_module_lr, options.unet_lr = 2e-5, 2e-6
    options.constant_learning_rate = True
    options.use_rollout_loss = False
    options.use_actual_step_constraint = False
    trainer = EBJDTrainer(model, options, config, identity)
    if trainer.update_index != 0 or trainer.optimizer.state:
        raise RuntimeError("GT-conditioned bridge optimizer is not fresh")
    return model, trainer, options, payload


@torch.no_grad()
def evaluate_mode(model, config, bank, mode: str, device, seed: int) -> dict:
    """Frozen inference reads only observed/map/valid plus G/H endpoint bank."""
    model.eval()
    overall, single, multi = MetricAccumulator(), MetricAccumulator(), MetricAccumulator()
    prefix, prefix_single, prefix_multi = PrefixAccumulator(), PrefixAccumulator(), PrefixAccumulator()
    generator = torch.Generator(device=device).manual_seed(seed)
    recall_hits = recall_count = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for cpu_batch in loader(config, "validation", torch.Generator().manual_seed(99173)):
        goals, permutation = bank.batch(cpu_batch, mode)
        batch = cpu_batch.to(device)
        goals, permutation = goals.to(device), permutation.to(device)
        base_noise = torch.randn(
            batch.observed.shape[0], 20, batch.observed.shape[1], 12, 2,
            device=device, dtype=torch.float32, generator=generator)
        noise = base_noise if mode == "native" else gather_worlds(base_noise, permutation)
        prediction, decoded_goals = predict(
            model, batch.observed, batch.semantic_maps, batch.valid,
            worlds=20, steps=20, proposal_goals=goals, proposal_eta=0.0,
            initial_noise=noise)
        if float((decoded_goals - goals).abs().nan_to_num().max().cpu()) != 0.0:
            raise RuntimeError("eta=0 inference changed the frozen endpoint bank")
        overall.update(prediction, batch.future, batch.valid)
        prefix.update(prediction, batch.future, batch.valid)
        counts = batch.valid.sum(1)
        for selector, accumulator, prefix_accumulator in (
            (counts == 1, single, prefix_single), (counts >= 2, multi, prefix_multi),
        ):
            if selector.any():
                accumulator.update(
                    prediction[selector], batch.future[selector], batch.valid[selector])
                prefix_accumulator.update(
                    prediction[selector], batch.future[selector], batch.valid[selector])
        endpoint = torch.linalg.vector_norm(
            decoded_goals - batch.future[:, None, :, -1], dim=-1).min(1).values
        recall_hits += int(((endpoint <= 0.5) & batch.valid).sum())
        recall_count += int(batch.valid.sum())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {
        "seed": seed, "mode": mode, "metrics": overall.as_dict(),
        "prefix": prefix.as_dict(), "N_eq_1": single.as_dict(),
        "N_eq_1_prefix": prefix_single.as_dict(), "N_ge_2": multi.as_dict(),
        "N_ge_2_prefix": prefix_multi.as_dict(),
        "recall_at_0_5m": recall_hits / recall_count,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda" else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--validation-bank", required=True)
    parser.add_argument("--validation-manifest-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_config(args.model_config)
    device = torch.device(args.device)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    bank = MPJCEndpointBank(
        args.validation_bank, expected_candidate_sha256=args.candidate_sha256,
        expected_manifest_sha256=args.validation_manifest_sha256,
        expected_split="validation")
    identity = {
        "schema": "ebjd-gt-conditioned-bridge-e3-v1",
        "fold": str(config["experiment"]["fold"]), "seed": 3101,
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True).stdout.strip(),
        "candidate_parent_sha256": args.candidate_sha256,
        "validation_mpjc_bank_sha256": bank.sha256,
        "training_condition": "augmented_ground_truth_endpoint",
        "inference_condition": "frozen_G_or_H",
    }
    atomic_json(output / "RUN_IDENTITY.json", identity)
    seed_all(3101)
    loader_generator = torch.Generator().manual_seed(3101)
    model, trainer, options, parent = build_fresh(
        config, Path(args.candidate_checkpoint), args.candidate_sha256,
        device, identity)
    records = []
    started = time.perf_counter()
    for local_epoch in range(1, 4):
        epoch = int(parent["epoch"]) + local_epoch
        model.train()
        pending_batches, pending_conditions, losses = [], [], []
        epoch_started = time.perf_counter()
        for raw_batch in loader(config, "train", loader_generator):
            augmented, gt_goal = augment_batch_with_gt_endpoint_condition(
                raw_batch, loader_generator)
            pending_batches.append(augmented)
            pending_conditions.append(ProposalTrainingCondition(
                gt_goal, eta=0.0, trust_weight=0.0))
            if len(pending_batches) == options.accumulation:
                log = trainer.train_step(pending_batches, epoch, pending_conditions)
                losses.append(log["loss"])
                with (output / "train.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        **log, "stage": "GT_CONDITIONED_BRIDGE_E3",
                        "stage_epoch": local_epoch, "epoch_equivalent": epoch,
                    }, sort_keys=True) + "\n")
                pending_batches, pending_conditions = [], []
        if pending_batches:
            log = trainer.train_step(pending_batches, epoch, pending_conditions)
            losses.append(log["loss"])
            with (output / "train.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    **log, "stage": "GT_CONDITIONED_BRIDGE_E3",
                    "stage_epoch": local_epoch, "epoch_equivalent": epoch,
                }, sort_keys=True) + "\n")
        validation = evaluate_mode(model, config, bank, "coupled", device, 2035)
        record = {
            "stage_epoch": local_epoch, "epoch_equivalent": epoch,
            "update": trainer.update_index, "loss": statistics.fmean(losses),
            "validation_H_seed2035": validation,
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        records.append(record)
        atomic_json(output / "epoch_state.json", record)
        metrics = validation["metrics"]
        print(
            f"[GT_CONDITIONED_BRIDGE_E3 {local_epoch}/3] "
            f"loss={record['loss']:.6f} minADE={metrics['minADE']:.6f} "
            f"minFDE={metrics['minFDE']:.6f} JADE={metrics['JADE']:.6f} "
            f"JFDE={metrics['JFDE']:.6f} "
            f"time={record['elapsed_seconds']/60:.2f}min "
            f"ETA={(3-local_epoch)*record['elapsed_seconds']/60:.2f}min", flush=True)
    checkpoint = output / "fixed_eplus3.pt"
    trainer.save_checkpoint(
        checkpoint, int(parent["epoch"]) + 3,
        {"fixed_stage": "GT_CONDITIONED_BRIDGE_E3", "not_best_selected": True},
        loader_generator, {"records": records, "training_condition": "gt_endpoint"})
    result = {
        "schema": "ebjd-gt-conditioned-bridge-e3-training-v1",
        "status": "COMPLETE", "identity": identity, "records": records,
        "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
        "optimizer_updates": trainer.update_index,
        "gpu_elapsed_seconds": time.perf_counter() - started,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda" else 0,
    }
    atomic_json(output / "TRAINING_RESULT.json", result)
    print(json.dumps({
        "status": "COMPLETE", "checkpoint_sha256": result["checkpoint_sha256"],
        "optimizer_updates": result["optimizer_updates"],
        "gpu_seconds": result["gpu_elapsed_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

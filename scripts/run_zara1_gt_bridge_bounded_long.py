#!/usr/bin/env python3
"""Bounded warm-start continuation for the frozen-H conditional bridge.

The entrypoint restores the complete E3 optimizer/RNG state, trains only with
ground-truth endpoint conditions, validates against a frozen H bank, and uses
the pre-registered eligible-lowest-JADE selector.  It is deliberately separate
from the short research pilots so the 30-epoch schedule cannot be confused
with a fresh or constant-learning-rate run.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

import torch
import yaml

from ebjd.config import build_model, config_hash, load_config, trainer_options
from ebjd.coupling import MPJCEndpointBank
from ebjd.data import augment_batch_with_gt_endpoint_condition
from ebjd.proposals import file_sha256
from ebjd.trainer import EBJDTrainer, ProposalTrainingCondition
from run_zara1_gt_conditioned_bridge import evaluate_mode
from run_zara1_native_prior_pilot import atomic_json, loader


METRICS = ("minADE", "minFDE", "JADE", "JFDE")


def cosine_factor(index: int, count: int, final_fraction: float) -> float:
    """Return an endpoint-inclusive cosine factor for ``count`` epochs."""
    if count < 2 or not 0 <= index < count:
        raise ValueError("cosine schedule index/count is invalid")
    if not 0.0 < final_fraction <= 1.0:
        raise ValueError("cosine final fraction must be in (0, 1]")
    progress = index / (count - 1)
    return final_fraction + (1.0 - final_fraction) * 0.5 * (
        1.0 + math.cos(math.pi * progress))


def eligible(metrics: dict, baseline: dict) -> bool:
    return (
        metrics["minADE"] <= baseline["minADE"]
        and metrics["minFDE"] <= baseline["minFDE"]
        and metrics["JADE"] < baseline["JADE"]
        and metrics["JFDE"] < baseline["JFDE"]
    )


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
        text=True).stdout.strip()


def load_plan(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        plan = yaml.safe_load(handle)
    if plan.get("schema") != "ebjd-zara1-gt-bridge-h-bounded-long-v1":
        raise ValueError("unsupported bounded-long plan schema")
    return plan


def validate_contract(plan: dict, config: dict) -> None:
    model = plan["model_contract"]
    if (
        config["denoiser"]["hidden_dim"] != model["hidden_dim"]
        or config["denoiser"]["heads"] != model["heads"]
        or config["denoiser"]["ff_dim"] != model["ff_dim"]
        or config["denoiser"]["coarse_blocks"]
        + config["denoiser"]["refinement_blocks"] != model["fjab_blocks"]
        or config["denoiser"].get("ctf_feedback_basis", "legacy") != "legacy"
    ):
        raise ValueError("model config differs from the frozen 192/4/768/6 FULL contract")
    continuation = plan["continuation"]
    if (
        continuation["maximum_additional_epochs"] != 30
        or continuation["rollout_training_enabled"] is not False
        or continuation["paired_actual_step_projection_enabled"] is not False
    ):
        raise ValueError("bounded continuation contract was changed")
    frozen = plan["frozen_inference"]
    if frozen["worlds"] != 20 or frozen["ddim_steps"] != 20:
        raise ValueError("formal inference must remain 20 worlds / 20 DDIM steps")


def restore_trainer(plan: dict, config: dict, device: torch.device, identity: dict,
                    loader_generator: torch.Generator, resume: Path | None):
    model = build_model(config, proposal_conditioned=True).to(device)
    options = trainer_options(config)
    continuation = plan["continuation"]
    options.epochs = int(continuation["maximum_additional_epochs"])
    options.warmup_epochs = 0
    options.lr_final_fraction = float(continuation["cosine_final_fraction"])
    options.constant_learning_rate = True
    options.use_rollout_loss = False
    options.use_actual_step_constraint = False
    options.new_module_lr = float(continuation["main_initial_lr"])
    options.unet_lr = float(continuation["unet_initial_lr"])
    trainer = EBJDTrainer(model, options, config, identity)
    if resume is None:
        parent_path = Path(plan["parent"]["checkpoint"])
        if file_sha256(parent_path) != plan["parent"]["checkpoint_sha256"]:
            raise ValueError("E3 parent checkpoint SHA256 mismatch")
        payload = torch.load(parent_path, map_location="cpu", weights_only=False)
        if payload.get("epoch") != plan["parent"]["epoch"] or payload.get(
                "update_index") != plan["parent"]["update"]:
            raise ValueError("E3 parent epoch/update mismatch")
        trainer.restore_checkpoint_payload(
            payload, loader_generator=loader_generator, require_identity=False)
        trainer.run_identity = identity
    else:
        trainer.load_checkpoint(resume, loader_generator=loader_generator)
        payload = torch.load(resume, map_location="cpu", weights_only=False)
    return model, trainer, options, payload


def aggregate_h(records: list[dict]) -> dict:
    return {
        metric: {
            "mean": statistics.fmean(row["metrics"][metric] for row in records),
            "sample_standard_deviation": statistics.stdev(
                row["metrics"][metric] for row in records),
        }
        for metric in METRICS
    }


def formal_evaluation(model, plan: dict, config: dict, selected_sha: str,
                      output: Path, device: torch.device) -> dict:
    target = output / "FORMAL_FIVE_SEED.json"
    if target.exists():
        result = json.loads(target.read_text(encoding="utf-8"))
        if result.get("selected_checkpoint_sha256") != selected_sha:
            raise ValueError("existing formal result belongs to another checkpoint")
        return result
    frozen = plan["frozen_inference"]
    records = []
    for seed in frozen["formal_seeds"]:
        bank_path = Path(
            frozen["h_bank_seed2035"] if seed == 2035
            else frozen["h_bank_seed_pattern"].format(seed=seed))
        bank = MPJCEndpointBank(
            str(bank_path),
            expected_candidate_sha256=frozen["candidate_reference_sha256"],
            expected_manifest_sha256=frozen["validation_manifest_sha256"],
            expected_split="validation")
        records.append(evaluate_mode(model, config, bank, "coupled", device, seed))
    result = {
        "schema": "ebjd-zara1-bounded-long-five-seed-v1",
        "status": "COMPLETE", "selected_checkpoint_sha256": selected_sha,
        "records": records, "summary": aggregate_h(records),
    }
    atomic_json(target, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    plan_path = Path(args.plan_config).resolve()
    plan = load_plan(plan_path)
    if git_head() != args.source_commit:
        raise ValueError("working-tree HEAD differs from --source-commit")
    model_config_path = Path(plan["model_config"])
    if not model_config_path.is_absolute():
        model_config_path = (Path.cwd() / model_config_path).resolve()
    config = load_config(model_config_path)
    validate_contract(plan, config)
    frozen = plan["frozen_inference"]
    validation_bank = MPJCEndpointBank(
        frozen["h_bank_seed2035"],
        expected_candidate_sha256=frozen["candidate_reference_sha256"],
        expected_manifest_sha256=frozen["validation_manifest_sha256"],
        expected_split="validation")
    identity = {
        "schema": plan["schema"], "fold": "zara1", "training_seed": 3101,
        "source_commit": args.source_commit, "plan_config_sha256": file_sha256(plan_path),
        "model_config_sha256": config_hash(config),
        "parent_checkpoint_sha256": plan["parent"]["checkpoint_sha256"],
        "validation_h_bank_sha256": validation_bank.sha256,
        "validation_manifest_sha256": frozen["validation_manifest_sha256"],
    }
    device = torch.device(args.device)
    output = Path(args.output).resolve()
    last = output / "last.pt"
    loader_generator = torch.Generator().manual_seed(plan["training_seed"])
    model, trainer, options, payload = restore_trainer(
        plan, config, device, identity, loader_generator,
        last if last.exists() else None)
    if args.preflight_only:
        print(json.dumps({
            "status": "PREFLIGHT_OK", "optimizer_updates_performed": 0,
            "parent_epoch": plan["parent"]["epoch"],
            "parent_update": plan["parent"]["update"],
            "restored_optimizer_state_count": len(trainer.optimizer.state),
            "restored_rng": True, "rollout": options.use_rollout_loss,
            "projection": options.use_actual_step_constraint,
            "validation_h_bank_sha256": validation_bank.sha256,
        }, sort_keys=True))
        return
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "RUN_IDENTITY.json", identity)
    state = payload.get("extra", {}).get("bounded_long_state")
    if state is None:
        initial = plan["selection"]["initial_parent_h_seed2035"]
        baseline = plan["selection"]["matched_gdts_seed2035"]
        if not eligible(initial, baseline):
            raise ValueError("pre-registered initial E3 checkpoint is not eligible")
        state = {
            "completed_additional_epochs": 0, "patience": 0,
            "patience_reference_jade": initial["JADE"],
            "best_jade": initial["JADE"],
            "best_checkpoint": plan["parent"]["checkpoint"],
            "best_checkpoint_sha256": plan["parent"]["checkpoint_sha256"],
            "gpu_elapsed_seconds": 0.0, "records": [],
        }
    continuation, selection = plan["continuation"], plan["selection"]
    maximum = int(continuation["maximum_additional_epochs"])
    run_started = time.perf_counter()
    for local_epoch in range(state["completed_additional_epochs"] + 1, maximum + 1):
        elapsed = state["gpu_elapsed_seconds"] + time.perf_counter() - run_started
        if elapsed >= continuation["gpu_wallclock_limit_seconds"]:
            break
        factor = cosine_factor(
            local_epoch - 1, maximum, continuation["cosine_final_fraction"])
        options.new_module_lr = continuation["main_initial_lr"] * factor
        options.unet_lr = continuation["unet_initial_lr"] * factor
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
                log = trainer.train_step(
                    pending_batches, local_epoch, pending_conditions)
                losses.append(log["loss"])
                with (output / "train.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        **log, "additional_epoch": local_epoch,
                        "epoch_equivalent": plan["parent"]["epoch"] + local_epoch,
                    }, sort_keys=True) + "\n")
                pending_batches, pending_conditions = [], []
        if pending_batches:
            log = trainer.train_step(pending_batches, local_epoch, pending_conditions)
            losses.append(log["loss"])
            with (output / "train.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    **log, "additional_epoch": local_epoch,
                    "epoch_equivalent": plan["parent"]["epoch"] + local_epoch,
                }, sort_keys=True) + "\n")
        validation = evaluate_mode(
            model, config, validation_bank, "coupled", device,
            frozen["selection_seed"])
        metrics = validation["metrics"]
        is_eligible = eligible(metrics, selection["matched_gdts_seed2035"])
        record = {
            "additional_epoch": local_epoch,
            "epoch_equivalent": plan["parent"]["epoch"] + local_epoch,
            "update": trainer.update_index, "loss": statistics.fmean(losses),
            "learning_rate_factor": factor, "eligible": is_eligible,
            "validation_H_seed2035": validation,
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        state["records"].append(record)
        state["completed_additional_epochs"] = local_epoch
        if is_eligible and metrics["JADE"] < state["best_jade"]:
            state["best_jade"] = metrics["JADE"]
            best = output / "best.pt"
            trainer.save_checkpoint(best, record["epoch_equivalent"], state,
                                    loader_generator, {"bounded_long_state": state})
            state["best_checkpoint"] = str(best)
            state["best_checkpoint_sha256"] = file_sha256(best)
        reset = is_eligible and metrics["JADE"] <= (
            state["patience_reference_jade"]
            - selection["patience_reset_minimum_improvement_m"])
        if reset:
            state["patience_reference_jade"] = metrics["JADE"]
            state["patience"] = 0
        else:
            state["patience"] += 1
        state["gpu_elapsed_seconds"] += record["elapsed_seconds"]
        trainer.save_checkpoint(
            last, record["epoch_equivalent"], state, loader_generator,
            {"bounded_long_state": state})
        atomic_json(output / "STATE.json", state)
        print(
            f"[BOUNDED {local_epoch:02d}/{maximum}] loss={record['loss']:.6f} "
            f"minADE={metrics['minADE']:.6f} minFDE={metrics['minFDE']:.6f} "
            f"JADE={metrics['JADE']:.6f} JFDE={metrics['JFDE']:.6f} "
            f"eligible={is_eligible} patience={state['patience']}", flush=True)
        if state["patience"] >= selection["patience_epochs"]:
            break
    budget_hit = state["gpu_elapsed_seconds"] >= continuation["gpu_wallclock_limit_seconds"]
    selected = Path(state["best_checkpoint"])
    if file_sha256(selected) != state["best_checkpoint_sha256"]:
        raise ValueError("selected checkpoint changed before formal evaluation")
    result = {
        "status": "GPU_BUDGET_EXHAUSTED" if budget_hit else "TRAINING_COMPLETE",
        "selection": state, "formal_five_seed": None,
    }
    if not budget_hit:
        selected_payload = torch.load(selected, map_location="cpu", weights_only=False)
        model.load_state_dict(selected_payload["model"], strict=True)
        result["formal_five_seed"] = formal_evaluation(
            model, plan, config, state["best_checkpoint_sha256"], output, device)
        result["status"] = "COMPLETE"
    atomic_json(output / "RESULT.json", result)


if __name__ == "__main__":
    main()


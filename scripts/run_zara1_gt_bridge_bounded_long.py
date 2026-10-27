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
import csv
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


def append_epoch_csv(path: Path, record: dict, state: dict, options,
                     total_elapsed: float, budget: float) -> None:
    metrics = record["validation_H_seed2035"]["metrics"]
    row = {
        "additional_epoch": record["additional_epoch"],
        "epoch_equivalent": record["epoch_equivalent"],
        "global_update": record["update"], "loss": record["loss"],
        **{metric: metrics[metric] for metric in METRICS},
        "eligible": record["eligible"], "best_JADE": state["best_jade"],
        "patience": state["patience"], "main_lr": options.new_module_lr,
        "unet_lr": options.unet_lr,
        "epoch_seconds": record["elapsed_seconds"],
        "total_elapsed_seconds": total_elapsed,
        "remaining_budget_seconds": max(budget - total_elapsed, 0.0),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def formal_evaluation(model, plan: dict, config: dict, selected_sha: str,
                      output: Path, device: torch.device, total_elapsed,
                      budget_seconds: float) -> dict:
    target = output / "FORMAL_FIVE_SEED.json"
    if target.exists():
        result = json.loads(target.read_text(encoding="utf-8"))
        if result.get("selected_checkpoint_sha256") != selected_sha:
            raise ValueError("existing formal result belongs to another checkpoint")
        if result.get("status") == "COMPLETE":
            return result
    frozen = plan["frozen_inference"]
    records = []
    per_seed = output / "formal_per_seed"
    per_seed.mkdir(parents=True, exist_ok=True)
    for seed in frozen["formal_seeds"]:
        seed_target = per_seed / f"seed_{seed}.json"
        if seed_target.exists():
            record = json.loads(seed_target.read_text(encoding="utf-8"))
            if (record.get("seed") != seed
                    or record.get("selected_checkpoint_sha256") != selected_sha):
                raise ValueError(f"formal seed receipt identity mismatch: {seed_target}")
            records.append(record["result"])
            continue
        completed_times = [row["elapsed_seconds"] for row in records]
        estimated = statistics.fmean(completed_times) if completed_times else 240.0
        if budget_seconds - total_elapsed() < estimated:
            break
        bank_path = Path(
            frozen["h_bank_seed2035"] if seed == 2035
            else frozen["h_bank_seed_pattern"].format(seed=seed))
        bank = MPJCEndpointBank(
            str(bank_path),
            expected_candidate_sha256=frozen["candidate_reference_sha256"],
            expected_manifest_sha256=frozen["validation_manifest_sha256"],
            expected_split="validation")
        record = evaluate_mode(model, config, bank, "coupled", device, seed)
        atomic_json(seed_target, {
            "status": "COMPLETE", "seed": seed,
            "selected_checkpoint_sha256": selected_sha,
            "bank_sha256": bank.sha256, "result": record,
            "total_elapsed_seconds": total_elapsed(),
        })
        records.append(record)
    result = {
        "schema": "ebjd-zara1-bounded-long-five-seed-v1",
        "status": "COMPLETE" if len(records) == len(frozen["formal_seeds"])
        else "PARTIAL_GPU_BUDGET_EXHAUSTED",
        "selected_checkpoint_sha256": selected_sha,
        "completed_seeds": [row["seed"] for row in records],
        "records": records,
        "summary": aggregate_h(records) if len(records) == 5 else None,
        "total_elapsed_seconds": total_elapsed(),
    }
    atomic_json(target, result)
    return result


def write_final_report(plan: dict, formal: dict, state: dict, output: Path,
                       stop_reason: str) -> dict:
    frozen = plan["frozen_inference"]
    if file_sha256(frozen["e3_five_seed_result"]) != frozen[
            "e3_five_seed_result_sha256"]:
        raise ValueError("E3 five-seed result SHA256 mismatch")
    if file_sha256(frozen["gdts_five_seed_result"]) != frozen[
            "gdts_five_seed_result_sha256"]:
        raise ValueError("GDTS five-seed result SHA256 mismatch")
    e3 = json.loads(Path(frozen["e3_five_seed_result"]).read_text(encoding="utf-8"))
    gdts = json.loads(Path(frozen["gdts_five_seed_result"]).read_text(encoding="utf-8"))
    comparison = {
        "schema": "ebjd-zara1-bounded-long-final-comparison-v1",
        "status": formal["status"], "stop_reason": stop_reason,
        "selected_additional_epoch": state.get("best_additional_epoch", 0),
        "selected_checkpoint": state["best_checkpoint"],
        "selected_checkpoint_sha256": state["best_checkpoint_sha256"],
        "optimizer_updates": state.get("final_update"),
        "gpu_elapsed_seconds": state["gpu_elapsed_seconds"],
        "formal": formal, "E3": e3["summary"]["H"], "GDTS": gdts["summary"],
    }
    if formal["status"] == "COMPLETE":
        final_by_seed = {row["seed"]: row["metrics"] for row in formal["records"]}
        e3_by_seed = {row["seed"]: row["H"]["metrics"] for row in e3["records"]}
        gdts_by_seed = {row["seed"]: row["metrics"] for row in gdts["runs"]}
        comparison["paired_deltas"] = {
            label: {
                str(seed): {
                    metric: final_by_seed[seed][metric] - reference[seed][metric]
                    for metric in METRICS}
                for seed in frozen["formal_seeds"]}
            for label, reference in (("final_minus_E3", e3_by_seed),
                                     ("final_minus_GDTS", gdts_by_seed))
        }
        means = {metric: formal["summary"][metric]["mean"] for metric in METRICS}
        gdts_means = {metric: gdts["summary"][metric]["mean"] for metric in METRICS}
        comparison["performance_contract_pass"] = eligible(means, gdts_means)
    atomic_json(output / "FINAL_COMPARISON.json", comparison)
    report = [
        "# ZARA1 新桥＋H 有界续训最终报告", "",
        f"- 状态：`{formal['status']}`", f"- 停止原因：`{stop_reason}`",
        f"- 选中追加轮：`{comparison['selected_additional_epoch']}`（0 表示保留 E3）",
        f"- selected SHA256：`{state['best_checkpoint_sha256']}`",
        f"- 累计 GPU 秒：`{state['gpu_elapsed_seconds']:.3f}`", "",
    ]
    if formal["status"] == "COMPLETE":
        report += ["| 指标 | 最终均值 | 样本标准差 |", "|---|---:|---:|"]
        for metric in METRICS:
            row = formal["summary"][metric]
            report.append(
                f"| {metric} | {row['mean']:.6f} | {row['sample_standard_deviation']:.6f} |")
        report += ["", f"相对 matched GDTS 原合同：**{'PASS' if comparison['performance_contract_pass'] else 'FAIL'}**。"]
    else:
        report += ["正式五种子未齐，不给出完整合同结论。",
                   f"已完成 seeds：`{formal['completed_seeds']}`。"]
    report += ["", "该结果是单训练 seed 的 warm-start 续训；provider/reference 全程冻结。", ""]
    (output.parent / "reports").mkdir(parents=True, exist_ok=True)
    (output.parent / "reports" / "FINAL_REPORT_ZH.md").write_text(
        "\n".join(report), encoding="utf-8")
    return comparison


def main() -> None:
    process_started = time.perf_counter()
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
    state_path = output / "STATE.json"
    if last.exists() and state_path.exists():
        disk_state = json.loads(state_path.read_text(encoding="utf-8"))
        if disk_state.get("completed_additional_epochs") != state.get(
                "completed_additional_epochs"):
            raise ValueError("STATE.json and last.pt epoch differ")
        state = disk_state
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
            "best_additional_epoch": 0,
            "gpu_elapsed_seconds": 0.0, "records": [],
        }
    continuation, selection = plan["continuation"], plan["selection"]
    maximum = int(continuation["maximum_additional_epochs"])
    spent_before_process = float(state.get("gpu_elapsed_seconds", 0.0))

    def total_elapsed() -> float:
        return spent_before_process + time.perf_counter() - process_started

    if state["completed_additional_epochs"] >= maximum:
        stop_reason = "MAXIMUM_30_EPOCHS"
        epoch_range = range(0)
    elif state["patience"] >= selection["patience_epochs"]:
        stop_reason = "EARLY_STOP_PATIENCE_8"
        epoch_range = range(0)
    else:
        stop_reason = None
        epoch_range = range(state["completed_additional_epochs"] + 1, maximum + 1)
    for local_epoch in epoch_range:
        remaining = continuation["gpu_wallclock_limit_seconds"] - total_elapsed()
        prior_times = [row["elapsed_seconds"] for row in state["records"]]
        estimated_epoch = statistics.fmean(prior_times[-3:]) if prior_times else float(
            continuation["expected_epoch_seconds"])
        if remaining < estimated_epoch + continuation["final_evaluation_reserve_seconds"]:
            stop_reason = "GPU_BUDGET_SAFE_BOUNDARY"
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
            state["best_additional_epoch"] = local_epoch
        reset = is_eligible and metrics["JADE"] <= (
            state["patience_reference_jade"]
            - selection["patience_reset_minimum_improvement_m"])
        if reset:
            state["patience_reference_jade"] = metrics["JADE"]
            state["patience"] = 0
        else:
            state["patience"] += 1
        state["gpu_elapsed_seconds"] = total_elapsed()
        trainer.save_checkpoint(
            last, record["epoch_equivalent"], state, loader_generator,
            {"bounded_long_state": state})
        state["gpu_elapsed_seconds"] = total_elapsed()
        state["final_update"] = trainer.update_index
        atomic_json(state_path, state)
        append_epoch_csv(
            output.parent / "logs" / "epoch_metrics.csv", record, state,
            options, state["gpu_elapsed_seconds"],
            continuation["gpu_wallclock_limit_seconds"])
        eta_seconds = max(maximum - local_epoch, 0) * record["elapsed_seconds"]
        print(
            f"[H-LONG {local_epoch:02d}/{maximum}] update={trainer.update_index} "
            f"loss={record['loss']:.6f} ADE={metrics['minADE']:.6f} "
            f"FDE={metrics['minFDE']:.6f} "
            f"JADE={metrics['JADE']:.6f} JFDE={metrics['JFDE']:.6f} "
            f"eligible={'Y' if is_eligible else 'N'} best_JADE={state['best_jade']:.6f} "
            f"patience={state['patience']}/{selection['patience_epochs']} "
            f"lr={options.new_module_lr:.3e} time={record['elapsed_seconds']/60:.2f}min "
            f"ETA={eta_seconds/3600:.2f}h", flush=True)
        if state["patience"] >= selection["patience_epochs"]:
            stop_reason = "EARLY_STOP_PATIENCE_8"
            break
    if stop_reason is None:
        stop_reason = (
            "MAXIMUM_30_EPOCHS" if state["completed_additional_epochs"] >= maximum
            else "TRAINING_ALREADY_COMPLETE")
    selected = Path(state["best_checkpoint"])
    if file_sha256(selected) != state["best_checkpoint_sha256"]:
        raise ValueError("selected checkpoint changed before formal evaluation")
    selected_payload = torch.load(selected, map_location="cpu", weights_only=False)
    model.load_state_dict(selected_payload["model"], strict=True)
    state["final_update"] = trainer.update_index
    formal = formal_evaluation(
        model, plan, config, state["best_checkpoint_sha256"], output, device,
        total_elapsed, continuation["gpu_wallclock_limit_seconds"])
    state["gpu_elapsed_seconds"] = total_elapsed()
    atomic_json(state_path, state)
    comparison = write_final_report(plan, formal, state, output, stop_reason)
    checkpoint_index = {
        "schema": "ebjd-zara1-bounded-long-checkpoint-index-v1",
        "E3": {"path": plan["parent"]["checkpoint"],
               "sha256": plan["parent"]["checkpoint_sha256"]},
        "last": {"path": str(last), "sha256": file_sha256(last) if last.exists() else None},
        "best": {"path": state["best_checkpoint"],
                 "sha256": state["best_checkpoint_sha256"]},
        "selected": {"path": state["best_checkpoint"],
                     "sha256": state["best_checkpoint_sha256"]},
    }
    atomic_json(output.parent / "checkpoints" / "CHECKPOINT_INDEX.json", checkpoint_index)
    result = {
        "status": "COMPLETE" if formal["status"] == "COMPLETE" else formal["status"],
        "stop_reason": stop_reason, "selection": state,
        "formal_five_seed": formal, "comparison": comparison,
    }
    atomic_json(output / "RESULT.json", result)


if __name__ == "__main__":
    main()


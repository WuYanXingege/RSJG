#!/usr/bin/env python3
"""Read-only CPU audit for the bounded HOTEL A0/A1 queue.

This program never imports the model, creates a dataloader, or executes a
sampler.  It reads atomic receipts, completed CSV/JSONL records, and trusted
local checkpoints with ``weights_only=True`` and ``map_location='cpu'``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml


PARENT_SHA = "5c101c2474ebb1a3cb3ecf882f0e9db2fbb2489c741c58068b8183b0b663b97b"
TRAIN_SEEDS = (3101, 3102, 3103)
EVAL_SEEDS = (2035, 2036, 2037, 2038, 2039)
METRICS = ("ADE_world", "FDE_world", "JADE", "JFDE", "Collision_Rate")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(block)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, hash_content: bool = True) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path.resolve()), "exists": path.exists()}
    if not path.exists():
        return record
    before = path.stat()
    record.update({"size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns})
    if hash_content:
        digest = sha256_file(path)
        after = path.stat()
        record["stable_during_hash"] = (
            before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
        )
        if record["stable_during_hash"]:
            record["sha256"] = digest
    return record


def read_json(path: Path, retries: int = 3) -> Any:
    error: Exception | None = None
    for _ in range(retries):
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            error = exc
            time.sleep(0.05)
    raise RuntimeError(f"could not read stable JSON {path}: {error}")


def read_jsonl_complete(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # An actively appended final line is not an audit failure.
                break
    return rows


def read_curve(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                parsed = {key: float(value) for key, value in row.items() if value != ""}
                parsed["epoch"] = int(parsed["epoch"])
                rows.append(parsed)
            except (TypeError, ValueError):
                break
    return rows


def checkpoint_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = payload.get("model_state_dict", payload)
    if not isinstance(state, dict):
        raise TypeError("checkpoint has no state dictionary")
    return state


def tensor_state_sha(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def prefix(name: str) -> str:
    return name.split(".", 1)[0]


def json_safe(value: Any) -> Any:
    """Preserve checkpoint metadata without emitting large RNG tensors."""
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
        if tensor.numel() == 1:
            return tensor.item()
        return {"tensor_shape": list(tensor.shape), "dtype": str(tensor.dtype),
                "numel": tensor.numel(), "sha256": digest}
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def inspect_completed_checkpoint(
    initial_path: Path, best_path: Path, last_path: Path, parent_state: dict[str, torch.Tensor]
) -> dict[str, Any]:
    initial_payload = torch.load(initial_path, map_location="cpu", weights_only=True)
    best_payload = torch.load(best_path, map_location="cpu", weights_only=True)
    last_payload = torch.load(last_path, map_location="cpu", weights_only=True)
    initial = checkpoint_state(initial_payload)
    best = checkpoint_state(best_payload)

    frozen_mismatches = []
    for name, reference in parent_state.items():
        if name not in best or not torch.equal(best[name], reference):
            frozen_mismatches.append(name)

    zero_names = (
        "jdv2_corrector.output.2.weight",
        "jdv2_corrector.output.2.bias",
    )
    zero_output = {
        name: bool(name in best and torch.count_nonzero(best[name]).item() == 0)
        for name in zero_names
    }

    changes: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"tensor_count": 0, "changed_tensor_count": 0, "parameter_count": 0,
                 "squared_l2_delta": 0.0}
    )
    for name, value in best.items():
        if name not in initial:
            continue
        group = prefix(name)
        item = changes[group]
        item["tensor_count"] += 1
        item["parameter_count"] += value.numel()
        delta = value.detach().cpu().double() - initial[name].detach().cpu().double()
        sq = float(delta.square().sum())
        item["squared_l2_delta"] += sq
        item["changed_tensor_count"] += int(sq != 0.0)
    for item in changes.values():
        item["l2_delta"] = math.sqrt(item.pop("squared_l2_delta"))

    scheduler = last_payload.get("scheduler_state_dict", {})
    mc = last_payload.get("jdv2_goal_objective_state")
    result = {
        "best_epoch": int(best_payload.get("epoch", -1)),
        "best_selection": best_payload.get("best_selection"),
        "best_checkpoint_role": best_payload.get("checkpoint_role"),
        "last_epoch": int(last_payload.get("epoch", -1)),
        "last_checkpoint_role": last_payload.get("checkpoint_role"),
        "resume_safe": bool(last_payload.get("resume_safe", False)),
        "optimizer_attempts": int(last_payload.get("optimizer_attempts", 0)),
        "successful_updates": int(last_payload.get("stage_optimizer_steps_completed", 0)),
        "failed_updates": int(last_payload.get("failed_optimizer_updates", 0)),
        "skipped_updates": int(last_payload.get("skipped_optimizer_updates", 0)),
        "planned_total_steps": int(last_payload.get("planned_total_steps", 0)),
        "scheduler": {
            "gamma": scheduler.get("gamma"), "last_epoch": scheduler.get("last_epoch"),
            "step_count": scheduler.get("_step_count"), "last_lr": scheduler.get("_last_lr"),
        },
        "parent_tensor_count": len(parent_state),
        "frozen_parent_exact": not frozen_mismatches,
        "frozen_parent_mismatches": frozen_mismatches,
        "frozen_state_sha256": last_payload.get("frozen_state_sha256"),
        "corrector_final_layer_exact_zero": zero_output,
        "initial_state_file_sha256": sha256_file(initial_path),
        "initial_state_declared_sha256": initial_payload.get("state_sha256"),
        "initial_state_recomputed_sha256": tensor_state_sha(initial),
        "parameter_delta_by_module": dict(changes),
        "mc_rng_state": json_safe(mc),
        "architecture_config": last_payload.get("architecture_config"),
        "inference_sampler_config": last_payload.get("inference_sampler_config"),
    }
    del initial_payload, best_payload, last_payload, initial, best
    return result


def extract_formal_rows(name: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scope = "E_joint" if name.startswith("E_joint") else "mechanism"
    arm = ("baseline" if name == "E_joint_baseline" else
           name.replace("E_joint_", "") if scope == "E_joint" else name)
    for index, run in enumerate(payload.get("runs", [])):
        if "metrics" in run:
            metrics = run["metrics"]
            eval_seed = run.get("evaluation_seed", EVAL_SEEDS[index] if index < 5 else "")
        else:
            metrics = run
            eval_seed = (payload.get("evaluation_seeds") or EVAL_SEEDS)[index]
        row = {"scope": scope, "artifact": name, "arm": arm,
               "training_seed": ("" if arm == "baseline" or scope == "mechanism"
                                 else arm.split("_")[-1]),
               "evaluation_seed": eval_seed, "status": "COMPLETED", "units": "m; collision=proportion"}
        row.update({metric: metrics.get(metric, "") for metric in METRICS})
        rows.append(row)
    return rows


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-repo", type=Path, required=True)
    parser.add_argument("--run-worktree", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
    started = time.monotonic()

    main_repo = args.main_repo.resolve()
    run_worktree = args.run_worktree.resolve()
    out = args.review_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    experiment = main_repo / "outputs/joint_dependency_v2/hotel_a0_a1_5b6e4e4"
    launch = experiment / "formal_launch"
    manifest_path = launch / "QUEUE_MANIFEST.json"
    manifest = read_json(manifest_path)
    progress = read_json(launch / "QUEUE_PROGRESS.json") if (launch / "QUEUE_PROGRESS.json").exists() else {}
    parent_path = main_repo.parent / "GDTS_official_297d508_HOTEL/output/hotel/saved_models/best_model.pt"
    parent_file_sha = sha256_file(parent_path)
    if parent_file_sha != PARENT_SHA:
        raise RuntimeError("frozen parent SHA256 mismatch")
    parent_payload = torch.load(parent_path, map_location="cpu", weights_only=True)
    parent_state = checkpoint_state(parent_payload)

    provenance: dict[str, Any] = {
        "schema": "hotel-a0-a1-cpu-audit-inputs-v1", "snapshot_utc": now_utc(),
        "main_repo": str(main_repo), "run_worktree": str(run_worktree),
        "run_source_commit": manifest.get("run_source_commit"),
        "cache_source_commit": manifest.get("cache_source_commit"),
        "parent": file_record(parent_path), "queue_manifest": file_record(manifest_path),
        "queue_progress": file_record(launch / "QUEUE_PROGRESS.json"),
        "files": [],
    }

    arm_results: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    checkpoint_loads = 1
    for item in manifest["arms"]:
        arm, seed = item["arm"], int(item["seed"])
        label = f"{arm}_{seed}"
        run_dir = Path(item["model_dir"])
        config_path = Path(item["config"])
        receipt_path = launch / "receipts" / f"{label}.json"
        curve_path = run_dir / "log_curve.txt"
        diag_path = run_dir / "joint_diagnostics.jsonl"
        receipt = read_json(receipt_path) if receipt_path.exists() else None
        config = yaml.safe_load(config_path.read_text())
        curves = read_curve(curve_path)
        diagnostics = {int(row["epoch"]): row for row in read_jsonl_complete(diag_path)}
        for row in curves:
            epoch = int(row["epoch"])
            diag = diagnostics.get(epoch, {}).get("train_joint_diagnostics", {})
            merged = {"arm": arm, "training_seed": seed, **row}
            for key in ("relation_entropy", "dynamic_relation_entropy", "teacher_relation_entropy",
                        "energy_std", "energy_mean", "mean_KL_r"):
                merged[key] = diag.get(key, "")
            usage = [diag.get(f"predicted_relation_usage_{idx}") for idx in range(4)]
            valid_usage = [float(x) for x in usage if x is not None]
            merged["dynamic_mode_mass_sum"] = sum(valid_usage) if valid_usage else ""
            merged["dynamic_dominant_mode_fraction"] = (
                max(valid_usage) / sum(valid_usage) if valid_usage and sum(valid_usage) else ""
            )
            training_rows.append(merged)
        best = min(curves, key=lambda row: (row["valid_JADE"], row["valid_JFDE"], row["epoch"])) if curves else None
        record: dict[str, Any] = {
            "arm": arm, "seed": seed, "label": label,
            "status": receipt.get("status") if receipt else "RUNNING_OR_PENDING",
            "objective": config.get("jdv2_goal_objective"),
            "run_source_commit": manifest.get("run_source_commit"),
            "config_sha256_manifest": item.get("config_sha256"),
            "config_sha256_actual": sha256_file(config_path),
            "parent_sha256": config.get("load_checkpoint_sha256", config.get("stage_a_parent_checkpoint_sha256", PARENT_SHA)),
            "cache_manifest_hash": config.get("trajectory_bank_manifest_hash", manifest.get("cache_manifest_hash")),
            "completed_epochs_visible": len(curves),
            "selected_from_curve": ({"epoch": best["epoch"], "JADE": best["valid_JADE"],
                                     "JFDE": best["valid_JFDE"], "ADE": best["valid_ADE_world"],
                                     "FDE": best["valid_FDE_world"],
                                     "CRmean": best["valid_Collision_Rate"]} if best else None),
            "receipt": receipt,
        }
        provenance["files"].extend([
            file_record(config_path), file_record(receipt_path), file_record(curve_path), file_record(diag_path)
        ])
        if receipt and receipt.get("status") == "COMPLETED":
            initial = experiment / "initial_states" / f"jdv2_initial_seed{seed}.pt"
            best_path = run_dir / "saved_models/best_model.pt"
            last_path = run_dir / "saved_models/last_model.pt"
            record["checkpoint"] = inspect_completed_checkpoint(
                initial, best_path, last_path, parent_state
            )
            checkpoint_loads += 3
            record["best_checkpoint"] = file_record(best_path)
            record["last_checkpoint"] = file_record(last_path)
            provenance["files"].extend([
                file_record(initial), file_record(best_path), file_record(last_path)
            ])
        arm_results.append(record)
    del parent_payload, parent_state

    # Formal E_joint rows: completed records are read; all others stay explicit PENDING.
    final_rows: list[dict[str, Any]] = []
    formal_status: dict[str, Any] = {}
    for task in manifest.get("post_tasks", []):
        path = Path(task["result"])
        if path.exists():
            payload = read_json(path)
            formal_status[task["name"]] = {"status": payload.get("status", "COMPLETED"), "path": str(path)}
            final_rows.extend(extract_formal_rows(task["name"], payload))
            provenance["files"].append(file_record(path))
        else:
            formal_status[task["name"]] = {"status": "PENDING", "path": str(path)}
            pending_arm = ("baseline" if task["name"] == "E_joint_baseline"
                           else task["name"].replace("E_joint_", ""))
            final_rows.append({
                "scope": "E_joint" if task["name"].startswith("E_joint") else "mechanism",
                "artifact": task["name"], "arm": pending_arm,
                "training_seed": (pending_arm.split("_")[-1]
                                  if pending_arm[:2] in ("A0", "A1") else ""),
                "evaluation_seed": "", "status": "PENDING", "units": "",
                **{metric: "" for metric in METRICS},
            })

    # Keep E_official separate and explicitly non-comparable to E_joint.
    official_path = main_repo / "docs/joint_dependency_v2/reviews/2026-10-06_764ccb5_official_gdts_hotel_validation/VALIDATION_RECEIPT.json"
    official = read_json(official_path)
    for item in official["independent_validation"]["runs"]:
        final_rows.append({
            "scope": "E_official", "artifact": "official_GDTS_independent_validation",
            "arm": "baseline", "training_seed": "", "evaluation_seed": f"legacy_run_{item['run']}",
            "status": "REFERENCE_ONLY_NOT_E_JOINT", "units": "m; collision=not available",
            "ADE_world": item["ade20_m"], "FDE_world": item["fde20_m"],
            "JADE": "", "JFDE": "", "Collision_Rate": "",
        })
    provenance["files"].append(file_record(official_path))

    final_fields = ["scope", "artifact", "arm", "training_seed", "evaluation_seed", "status", "units", *METRICS]
    with (out / "FINAL_METRICS.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=final_fields)
        writer.writeheader(); writer.writerows(final_rows)

    # Formal means are computed within an arm only after all five inference
    # seeds exist.  They remain separate from the E_official reference rows.
    formal_arm_means: dict[str, dict[str, float]] = {}
    completed_joint = [row for row in final_rows
                       if row["scope"] == "E_joint" and row["status"] == "COMPLETED"]
    for arm in sorted({row["arm"] for row in completed_joint}):
        rows = [row for row in completed_joint if row["arm"] == arm]
        if len(rows) == len(EVAL_SEEDS):
            formal_arm_means[arm] = {
                metric: mean([float(row[metric]) for row in rows]) for metric in METRICS
            }
    baseline_effects: dict[str, Any] = {}
    if "baseline" in formal_arm_means:
        for arm, values in formal_arm_means.items():
            if arm == "baseline":
                continue
            baseline_effects[arm] = {
                metric: {"method_minus_baseline": values[metric] - formal_arm_means["baseline"][metric],
                         "relative_reduction": ((formal_arm_means["baseline"][metric] - values[metric]) /
                                                formal_arm_means["baseline"][metric]
                                                if formal_arm_means["baseline"][metric] else None)}
                for metric in METRICS
            }
    mechanism_means: dict[str, dict[str, float]] = {}
    completed_mechanism = [row for row in final_rows
                           if row["scope"] == "mechanism" and row["status"] == "COMPLETED"]
    for arm in sorted({row["arm"] for row in completed_mechanism}):
        rows = [row for row in completed_mechanism if row["arm"] == arm]
        if len(rows) == len(EVAL_SEEDS):
            mechanism_means[arm] = {
                metric: mean([float(row[metric]) for row in rows]) for metric in METRICS
            }

    # Validation-only A1-A0 pair effects. These are never labeled formal.
    paired_rows: list[dict[str, Any]] = []
    by_label = {row["label"]: row for row in arm_results}
    completed_pair_deltas: list[dict[str, float]] = []
    for seed in TRAIN_SEEDS:
        a0, a1 = by_label[f"A0_{seed}"], by_label[f"A1_{seed}"]
        row: dict[str, Any] = {"scope": "selection_validation_seed2035", "training_seed": seed,
                               "status": "COMPLETE_PAIR" if a0["receipt"] and a1["receipt"] else "PENDING_PAIR",
                               "units": "m; collision delta=absolute proportion"}
        if row["status"] == "COMPLETE_PAIR":
            deltas: dict[str, float] = {}
            for metric, key in (("ADE", "ADE"), ("FDE", "FDE"), ("JADE", "JADE"),
                                ("JFDE", "JFDE"), ("CRmean", "CRmean")):
                ref, method = a0["selected_from_curve"][key], a1["selected_from_curve"][key]
                row[f"A0_{metric}"] = ref; row[f"A1_{metric}"] = method
                row[f"delta_A1_minus_A0_{metric}"] = method - ref
                row[f"relative_reduction_{metric}"] = (ref - method) / ref if ref else ""
                deltas[metric] = method - ref
            completed_pair_deltas.append(deltas)
        paired_rows.append(row)
    if completed_pair_deltas:
        summary = {"scope": "selection_validation_seed2035", "training_seed": "mean_complete_pairs",
                   "status": f"PARTIAL_{len(completed_pair_deltas)}_OF_3_PAIRS",
                   "units": "m; collision delta=absolute proportion"}
        for metric in ("ADE", "FDE", "JADE", "JFDE", "CRmean"):
            summary[f"delta_A1_minus_A0_{metric}"] = mean([x[metric] for x in completed_pair_deltas])
            refs = [by_label[f"A0_{seed}"]["selected_from_curve"][metric]
                    for seed in TRAIN_SEEDS if by_label[f"A0_{seed}"]["receipt"] and by_label[f"A1_{seed}"]["receipt"]]
            summary[f"relative_reduction_{metric}"] = -summary[f"delta_A1_minus_A0_{metric}"] / mean(refs)
        paired_rows.append(summary)

    formal_pair_deltas: list[dict[str, float]] = []
    for seed in TRAIN_SEEDS:
        a0_label, a1_label = f"A0_{seed}", f"A1_{seed}"
        row = {"scope": "E_joint_formal_five_seed_mean", "training_seed": seed,
               "status": "COMPLETE_PAIR" if a0_label in formal_arm_means and a1_label in formal_arm_means
               else "PENDING_PAIR", "units": "m; collision delta=absolute proportion"}
        if row["status"] == "COMPLETE_PAIR":
            deltas = {}
            for metric, source in (("ADE", "ADE_world"), ("FDE", "FDE_world"),
                                   ("JADE", "JADE"), ("JFDE", "JFDE"),
                                   ("CRmean", "Collision_Rate")):
                ref = formal_arm_means[a0_label][source]
                method = formal_arm_means[a1_label][source]
                row[f"A0_{metric}"] = ref; row[f"A1_{metric}"] = method
                row[f"delta_A1_minus_A0_{metric}"] = method - ref
                row[f"relative_reduction_{metric}"] = (ref - method) / ref if ref else ""
                deltas[metric] = method - ref
            formal_pair_deltas.append(deltas)
        paired_rows.append(row)

    gate_status: dict[str, Any] = {
        "status": "NOT_EVALUATED_FORMAL_RESULTS_PENDING",
        "requirements": {"JADE_absolute_reduction_m": 0.01, "JADE_relative_reduction": 0.02,
                         "same_direction_pairs_min": 2, "ADE_harm_m_max": 0.003,
                         "ADE_harm_relative_max": 0.01, "FDE_harm_m_max": 0.005,
                         "FDE_harm_relative_max": 0.01, "JFDE_harm_m_max": 0.01,
                         "CRmean_harm_absolute_max": 0.005},
    }
    if len(formal_pair_deltas) == 3:
        deltas = {metric: mean([row[metric] for row in formal_pair_deltas])
                  for metric in ("ADE", "FDE", "JADE", "JFDE", "CRmean")}
        a0_means = {metric: mean([formal_arm_means[f"A0_{seed}"][source]
                                  for seed in TRAIN_SEEDS])
                    for metric, source in (("ADE", "ADE_world"), ("FDE", "FDE_world"),
                                           ("JADE", "JADE"), ("JFDE", "JFDE"),
                                           ("CRmean", "Collision_Rate"))}
        relative = {metric: -deltas[metric] / a0_means[metric]
                    for metric in deltas}
        direction = sum(row["JADE"] < 0 for row in formal_pair_deltas)
        checks = {
            "JADE_absolute": -deltas["JADE"] >= 0.01,
            "JADE_relative": relative["JADE"] >= 0.02,
            "JADE_direction": direction >= 2,
            "ADE_guardrail": deltas["ADE"] <= 0.003 and -relative["ADE"] <= 0.01,
            "FDE_guardrail": deltas["FDE"] <= 0.005 and -relative["FDE"] <= 0.01,
            "JFDE_guardrail": deltas["JFDE"] <= 0.01,
            "CRmean_guardrail": deltas["CRmean"] <= 0.005,
        }
        gate_status.update({"status": "PASS" if all(checks.values()) else "FAIL",
                            "mean_delta_A1_minus_A0": deltas,
                            "relative_reduction": relative,
                            "same_direction_pairs": direction, "checks": checks})
    paired_fields = sorted({key for row in paired_rows for key in row})
    with (out / "PAIRED_EFFECTS.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=paired_fields)
        writer.writeheader(); writer.writerows(paired_rows)

    curve_fields = sorted({key for row in training_rows for key in row}, key=lambda x: (x not in ("arm", "training_seed", "epoch"), x))
    with (out / "TRAINING_CURVES.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve_fields)
        writer.writeheader(); writer.writerows(training_rows)

    complete_arms = sum(row["status"] == "COMPLETED" for row in arm_results)
    queue_running = complete_arms < 6
    preliminary = next((row for row in paired_rows if row["training_seed"] == "mean_complete_pairs"), None)
    results = {
        "schema": "hotel-a0-a1-final-cpu-diagnostics-v1",
        "status": "PARTIAL_QUEUE_RUNNING" if queue_running else (
            "COMPLETE" if all(x["status"] != "PENDING" for x in formal_status.values()) else "TRAINING_COMPLETE_EVALUATION_PENDING"),
        "snapshot_utc": provenance["snapshot_utc"],
        "queue": {"complete_arms": complete_arms, "total_arms": 6,
                  "deadline_utc": manifest.get("deadline_utc"), "progress_receipt": progress},
        "identity": {"run_source_commit": manifest.get("run_source_commit"),
                     "cache_source_commit": manifest.get("cache_source_commit"),
                     "parent_sha256": parent_file_sha, "evaluation_seeds": manifest.get("evaluation_seeds")},
        "arms": arm_results,
        "formal_results": formal_status,
        "formal_arm_five_seed_means": formal_arm_means,
        "formal_effects_vs_E_joint_baseline": baseline_effects,
        "mechanism_five_seed_means": mechanism_means,
        "formal_primary_comparison_status": "PENDING" if any(
            value["status"] == "PENDING" for name, value in formal_status.items() if name.startswith("E_joint")) else "AVAILABLE",
        "preliminary_selection_validation_only": preliminary,
        "a1_gate": gate_status,
        "deep_native_tensor_diagnostics": {
            "status": "NOT_RUN_MISSING_NATIVE_PAYLOAD",
            "missing": ["edge-level relation probabilities", "deployable prior pair-cost matrices",
                        "fixed q/u/logits with historical draw identities", "decision objective payload"],
            "consequence": "H_agg, I/a/b, Jensen gap, and assignment margin cannot be reconstructed from aggregate logs",
        },
        "call_accounting": {
            "new_gpu_calls": 0, "new_model_forward_calls": 0, "new_training_calls": 0,
            "new_diffusion_calls": 0, "new_sampler_calls": 0, "new_cache_exports": 0,
            "cpu_checkpoint_loads": checkpoint_loads, "cpu_solver_calls": 0,
            "synthetic_math_reference_cases": 4,
        },
        "budget": {"thread_limit": 4, "rss_limit_gib": 8, "wall_clock_limit_hours": 2,
                   "elapsed_seconds": time.monotonic() - started},
    }
    (out / "RESULTS.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    (out / "INPUT_PROVENANCE.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

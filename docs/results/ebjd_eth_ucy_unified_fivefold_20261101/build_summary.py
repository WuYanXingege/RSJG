#!/usr/bin/env python3
"""Recompute the formal EBJD unified five-fold summary from raw seed records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path("/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG")
ASSETS = ROOT / "RSJG_EBJD_assets"
PAPER = ROOT / "RSJG_EBJD_Paper"
SEEDS = [2035, 2036, 2037, 2038, 2039]
FOLDS = ["eth", "hotel", "univ", "zara1", "zara2"]
METRICS = ["minADE", "minFDE", "JADE", "JFDE"]
METHODS = ["GDTS", "G", "H"]

EXPERIMENTS = {
    "zara1": ASSETS / "experiments/zara1_unified_coupling_bridge_v1_20261028_214026",
    "zara2": ASSETS / "experiments/zara2_unified_coupling_bridge_v1_20261029_094715",
    "eth": ASSETS / "experiments/eth_unified_coupling_bridge_v1_20261030_104912",
    "hotel": ASSETS / "experiments/hotel_unified_coupling_bridge_v1_20261030_211601",
    "univ": ASSETS / "experiments/univ_unified_coupling_bridge_v1_20261031_111541",
}
RESOLVED = {
    "zara1": EXPERIMENTS["zara1"] / "configs/model_config.json",
    "zara2": EXPERIMENTS["zara2"] / "configs/RESOLVED_ZARA2_UNIFIED.json",
    "eth": EXPERIMENTS["eth"] / "configs/RESOLVED_ETH_UNIFIED.json",
    "hotel": EXPERIMENTS["hotel"] / "configs/RESOLVED_HOTEL_UNIFIED.json",
    "univ": EXPERIMENTS["univ"] / "configs/RESOLVED_UNIV_UNIFIED.json",
}
SELECTED_EPOCH = {"eth": 19, "hotel": 21, "univ": 30, "zara1": 30, "zara2": 23}
CONFIG_NOTES = {
    "zara1": (
        "model_config.json retains the pre-unified 250-epoch/0.01 base fields; "
        "configs/training_contract.yaml, RUN_IDENTITY, and the completed 30-epoch log bind "
        "the actual unified run (30 epochs, final LR fraction 0.1)."
    ),
    "zara2": "Fold adaptation plus scene-compaction/runtime engineering; unified method fields match.",
    "eth": "Fold/provider adaptation; unified method fields match.",
    "hotel": "Fold/provider adaptation; unified method fields match.",
    "univ": "Fold/provider adaptation; unified method fields match.",
}
SET_RECEIPTS = {
    "zara1": "explicit_formal_report_plus_one_to_one_pairing",
    "zara2": "explicit_formal_report_plus_one_to_one_pairing",
    "eth": "implementation_invariant_and_exact_minFDE_only; no standalone full-bank set receipt found",
    "hotel": "implementation_invariant_and_exact_minFDE_only; no standalone full-bank set receipt found",
    "univ": "implementation_invariant_and_exact_minFDE_only; no standalone full-bank set receipt found",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def file_record(path: Path) -> dict:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256(path) if path.exists() else None,
    }


def resolve_zara1_manifest(value: str) -> Path:
    value = value.removeprefix("assets/")
    return ASSETS / value


def load_seed_metrics() -> tuple[dict, dict]:
    data: dict[str, dict[int, dict[str, dict[str, float]]]] = {}
    metadata: dict[str, dict] = {}
    for fold in FOLDS:
        root = EXPERIMENTS[fold]
        status = read_json(root / "training/STATUS.json")
        config = read_json(RESOLVED[fold])
        by_seed = {}
        if fold == "zara1":
            for run in status["runs"]:
                seed = int(run["seed"])
                by_seed[seed] = {
                    method: {metric: float(run[method][metric]) for metric in METRICS}
                    for method in ["G", "H"]
                }
            gdts_path = ASSETS / "experiments/zara1_main_readiness_20261027/gdts_five_seed/results.json"
            gdts = read_json(gdts_path)
            for run in gdts["runs"]:
                by_seed[int(run["seed"])]["GDTS"] = {
                    metric: float(run["metrics"][metric]) for metric in METRICS
                }
            raw_source = root / "training/STATUS.json"
            paths = config["paths"]
            manifests = {
                split: resolve_zara1_manifest(paths[f"{split}_manifest"])
                for split in ["train", "validation", "test"]
            }
            gdts_raw = gdts_path
            formal_summary = PAPER / "docs/validation/ebjd_zara1_unified_coupling_bridge_20261028/FINAL_RESULT_SUMMARY.json"
            contract_file = root / "configs/training_contract.yaml"
        else:
            raw_source = root / "reports/PER_SEED_METRICS.csv"
            for row in read_csv(raw_source):
                seed = int(row["seed"])
                by_seed[seed] = {
                    method: {metric: float(row[f"{method}_{metric}"]) for metric in METRICS}
                    for method in METHODS
                }
            paths = config["paths"]
            manifests = {split: Path(paths[f"{split}_manifest"]) for split in ["train", "validation", "test"]}
            gdts_raw = root / "assets_manifest/MATCHED_GDTS_RESULTS.json"
            formal_summary = root / "reports/FINAL_RESULT_SUMMARY.json"
            contract_file = RESOLVED[fold]

        if sorted(by_seed) != SEEDS:
            raise RuntimeError(f"{fold}: incomplete seeds {sorted(by_seed)}")
        for seed in SEEDS:
            if set(by_seed[seed]) != set(METHODS):
                raise RuntimeError(f"{fold}/{seed}: incomplete methods")
        data[fold] = by_seed

        identity = read_json(root / "training/RUN_IDENTITY.json")
        selected = Path(status["selected_checkpoint"])
        actual_sha = sha256(selected)
        if actual_sha != status["selected_checkpoint_sha256"]:
            raise RuntimeError(f"{fold}: selected checkpoint hash mismatch")
        epoch_rows = read_csv(root / "training/epoch_metrics.csv")
        evaluation = config["evaluation"]
        denoiser = config["denoiser"]
        training = config["training"]
        matched = evaluation["matched_gdts_validation"]
        if training["epochs"] != 30 and fold != "zara1":
            raise RuntimeError(f"{fold}: unexpected epochs")
        metadata[fold] = {
            "experiment_root": str(root),
            "status": status["status"],
            "source_commit": identity["source_commit"],
            "training_seed": identity["training_seed"],
            "selected_checkpoint": file_record(selected),
            "selected_epoch": SELECTED_EPOCH[fold],
            "selection_identity": status["selection_identity"].removesuffix(".pt"),
            "completed_epochs": int(epoch_rows[-1]["epoch"]),
            "optimizer_updates": int(status["optimizer_updates"]),
            "resolved_config": file_record(RESOLVED[fold]),
            "actual_contract": file_record(contract_file),
            "configuration_note": CONFIG_NOTES[fold],
            "method_signature": {
                "hidden_dim": denoiser["hidden_dim"],
                "heads": denoiser["heads"],
                "ff_dim": denoiser["ff_dim"],
                "fjab_blocks": denoiser["coarse_blocks"] + denoiser["refinement_blocks"],
                "full_raw": True,
                "worlds": evaluation["worlds"],
                "ddim_steps": evaluation["steps"],
                "rollout_training_enabled": training["rollout_training_enabled"],
                "paired_projection_enabled": training["paired_marginal_actual_step_projection_enabled"],
            },
            "manifests": {split: file_record(path) for split, path in manifests.items()},
            "validation_test_identity": evaluation["validation_test_identity"],
            "raw_G_H_seed_results": file_record(raw_source),
            "raw_matched_gdts_results": file_record(gdts_raw),
            "formal_summary": file_record(formal_summary),
            "epoch_metrics": file_record(root / "training/epoch_metrics.csv"),
            "checkpoint_index": file_record(root / "training/checkpoint_index.csv"),
            "run_identity": file_record(root / "training/RUN_IDENTITY.json"),
            "gpu_budget": file_record(root / "GPU_BUDGET.json") if (root / "GPU_BUDGET.json").exists() else None,
            "matched_gdts_checkpoint_sha256": matched["checkpoint_sha256"],
            "matched_manifest_sha256": matched["manifest_sha256"],
            "coordinate_unit": matched["unit"],
            "scene_count": matched["scene_count"],
            "agent_occurrence_count": matched["agent_occurrence_count"],
            "candidate_set_preservation_evidence": SET_RECEIPTS[fold],
        }
        if metadata[fold]["manifests"]["validation"]["sha256"] != matched["manifest_sha256"]:
            raise RuntimeError(f"{fold}: matched manifest hash mismatch")
    return data, metadata


def stats(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def calculate(data: dict) -> tuple[list[dict], list[dict], dict]:
    per_seed_rows = []
    summary_rows = []
    summary = {fold: {} for fold in FOLDS}
    for fold in FOLDS:
        for seed in SEEDS:
            for method in METHODS:
                row = {"fold": fold.upper(), "seed": seed, "method": method}
                row.update(data[fold][seed][method])
                per_seed_rows.append(row)
        for method in METHODS:
            summary[fold][method] = {}
            for metric in METRICS:
                values = [data[fold][seed][method][metric] for seed in SEEDS]
                mean, sd = stats(values)
                summary[fold][method][metric] = {"mean": mean, "sd_ddof1": sd}
                summary_rows.append({
                    "scope": fold.upper(), "method": method, "metric": metric,
                    "mean": mean, "sd_ddof1": sd,
                })

    macro = {method: {} for method in METHODS}
    macro_seed = {seed: {method: {} for method in METHODS} for seed in SEEDS}
    for seed in SEEDS:
        for method in METHODS:
            for metric in METRICS:
                macro_seed[seed][method][metric] = statistics.mean(
                    data[fold][seed][method][metric] for fold in FOLDS
                )
    for method in METHODS:
        for metric in METRICS:
            values = [macro_seed[seed][method][metric] for seed in SEEDS]
            mean, sd = stats(values)
            macro[method][metric] = {"mean": mean, "sd_ddof1": sd}
            summary_rows.append({
                "scope": "MACRO_5FOLD", "method": method, "metric": metric,
                "mean": mean, "sd_ddof1": sd,
            })
    summary["macro_5fold"] = macro
    summary["macro_by_inference_seed"] = macro_seed
    return per_seed_rows, summary_rows, summary


def count_directions(values: list[float], tol: float = 1e-12) -> dict:
    return {
        "improved": sum(v < -tol for v in values),
        "equal": sum(abs(v) <= tol for v in values),
        "regressed": sum(v > tol for v in values),
    }


def contract(data: dict, summary: dict) -> dict:
    result = {"folds": {}, "macro_5fold": {}, "comparison_tolerance": 1e-12}
    for fold in FOLDS:
        perf = {}
        pairing = {}
        for metric in METRICS:
            h = summary[fold]["H"][metric]["mean"]
            g = summary[fold]["G"][metric]["mean"]
            gdts = summary[fold]["GDTS"][metric]["mean"]
            h_g_values = [data[fold][s]["H"][metric] - data[fold][s]["G"][metric] for s in SEEDS]
            h_b_values = [data[fold][s]["H"][metric] - data[fold][s]["GDTS"][metric] for s in SEEDS]
            perf[metric] = {
                "H_minus_GDTS": h - gdts,
                "improvement_percent": 100.0 * (gdts - h) / gdts,
                "pass_non_degradation": h <= gdts + 1e-12,
                "paired_seed_directions": count_directions(h_b_values),
            }
            pairing[metric] = {
                "H_minus_G": h - g,
                "improvement_percent": 100.0 * (g - h) / g,
                "paired_seed_directions": count_directions(h_g_values),
            }
        result["folds"][fold] = {
            "performance_by_metric": perf,
            "four_metric_contract_pass": all(v["pass_non_degradation"] for v in perf.values()),
            "pairing": pairing,
            "pairing_joint_gain": (
                pairing["JADE"]["H_minus_G"] < 0 and pairing["JFDE"]["H_minus_G"] < 0
            ),
            "candidate_set_evidence": SET_RECEIPTS[fold],
        }
    for metric in METRICS:
        h = summary["macro_5fold"]["H"][metric]["mean"]
        g = summary["macro_5fold"]["G"][metric]["mean"]
        b = summary["macro_5fold"]["GDTS"][metric]["mean"]
        result["macro_5fold"][metric] = {
            "H_minus_G": h - g,
            "H_minus_GDTS": h - b,
            "H_vs_G_improvement_percent": 100.0 * (g - h) / g,
            "H_vs_GDTS_improvement_percent": 100.0 * (b - h) / b,
            "pass_non_degradation_vs_GDTS": h <= b + 1e-12,
        }
    result["macro_5fold"]["four_metric_contract_pass"] = all(
        result["macro_5fold"][m]["pass_non_degradation_vs_GDTS"] for m in METRICS
    )
    result["folds_passing_four_metric_contract"] = [
        fold for fold in FOLDS if result["folds"][fold]["four_metric_contract_pass"]
    ]
    return result


def costs(metadata: dict) -> list[dict]:
    rows = []
    for fold in FOLDS:
        root = EXPERIMENTS[fold]
        epoch_rows = read_csv(root / "training/epoch_metrics.csv")
        train_val = sum(float(r["training_seconds"]) + float(r["validation_seconds"]) for r in epoch_rows)
        status = read_json(root / "training/STATUS.json")
        final_eval = sum(float(r.get("elapsed_seconds", 0.0)) for r in status.get("runs", []))
        if (root / "GPU_BUDGET.json").exists():
            budget = read_json(root / "GPU_BUDGET.json")
            total = float(budget["consumed_seconds"])
            provider = sum(
                float(s.get("seconds", 0.0)) for s in budget.get("stages", [])
                if "PROVIDER" in str(s.get("stage", "")) or "EXPORT" in str(s.get("stage", ""))
            )
            provider_note = "Recorded provider/parity/export stages, including recorded failed or carried attempts."
        else:
            total = float(status["gpu_wall_seconds"])
            provider = ""
            provider_note = "Not separately recorded in the unified ZARA1 receipt."
        rows.append({
            "fold": fold.upper(),
            "provider_and_candidate_generation_seconds": provider,
            "provider_cost_note": provider_note,
            "main_training_plus_epoch_validation_seconds": train_val,
            "final_five_seed_evaluation_seconds": final_eval,
            "recorded_pipeline_gpu_wall_seconds": total,
            "selected_epoch": metadata[fold]["selected_epoch"],
            "total_training_epochs": metadata[fold]["completed_epochs"],
            "optimizer_updates": metadata[fold]["optimizer_updates"],
            "historical_prior_training_cost": "unknown/not included in current unified receipts",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, metadata = load_seed_metrics()
    per_seed, summary_rows, summary = calculate(data)
    contracts = contract(data, summary)
    cost_rows = costs(metadata)

    write_csv(args.output_dir / "PER_SEED_METRICS.csv", per_seed,
              ["fold", "seed", "method", *METRICS])
    write_csv(args.output_dir / "FIVEFOLD_METRICS.csv", summary_rows,
              ["scope", "method", "metric", "mean", "sd_ddof1"])
    write_csv(args.output_dir / "COMPUTE_COSTS.csv", cost_rows, list(cost_rows[0]))
    (args.output_dir / "SOURCE_MANIFEST.json").write_text(
        json.dumps({
            "schema": "ebjd_unified_fivefold_source_manifest_v1",
            "folds_complete": FOLDS,
            "inference_seeds": SEEDS,
            "methods": {
                "GDTS": "fold-bound matched GDTS full prediction",
                "G": "native endpoints plus current conditional bridge full trajectory",
                "H": "learned endpoint reassignment plus the same conditional bridge full trajectory",
            },
            "fold_sources": metadata,
            "comparability": {
                "coordinate_unit": "m",
                "worlds": 20,
                "ddim_steps": 20,
                "validation_test_relation": "mirrored official-package internal validation/test members for all five folds",
                "training_seeds": 1,
                "inference_repetitions_per_fold": 5,
                "source_commit_difference_assessment": (
                    "Source commits differ because each fold was launched sequentially with fold/provider/runtime adaptations. "
                    "Resolved method signatures match 192/4/768, 6 FJAB, FULL/Raw, 20x20, rollout/projection OFF. "
                    "No source evidence of a method-semantic change was found in the bound configs."
                ),
            },
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "CONTRACT_SUMMARY.json").write_text(
        json.dumps(contracts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

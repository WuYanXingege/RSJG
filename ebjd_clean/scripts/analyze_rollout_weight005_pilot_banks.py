#!/usr/bin/env python3
"""CPU-only frozen four-model analysis for the rollout-weight 0.05 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np

from ebjd.diagnostic_bank import atomic_json, sha256_file
from analyze_goal_weight_pilot_banks import (
    HIT_THRESHOLDS, KS, METHODS, METRICS, SEEDS, Accumulator,
    heavy_tail_tables, k_tables, load_and_reduce, metric_tables,
    permutation_tables, sample_std, stratified_tables, write_csv,
)


def random_subset_tables(errors):
    rows = []
    for method in METHODS:
        for seed in SEEDS:
            for repeat in range(100):
                for k in KS:
                    accumulator = Accumulator()
                    for identity, (ade, fde) in errors[method][seed].items():
                        seed_value = int.from_bytes(hashlib.sha256(
                            f"rollout005-subset\x1f{seed}\x1f{identity}\x1f{repeat}".encode()
                        ).digest()[:8], "big")
                        indices = np.random.default_rng(seed_value).permutation(20)[:k]
                        accumulator.update(ade, fde, indices)
                    rows.append({
                        "selection": "shared_random_world_subset",
                        "model": method, "seed": seed, "repeat": repeat,
                        "K": k, **accumulator.result(),
                    })
    summary = []
    for method in METHODS:
        for k in KS:
            selected = [
                row for row in rows if row["model"] == method and row["K"] == k]
            item = {"model": method, "K": k, "records": len(selected)}
            for metric in METRICS:
                values = [float(row[metric]) for row in selected]
                item[f"{metric}_mean"] = float(np.mean(values))
                item[f"{metric}_sample_std"] = sample_std(values)
            summary.append(item)
    return rows, summary


def coverage_tables(strata):
    summary = [
        row for row in strata
        if row["group_type"] == "overall" and row["category"] == "all"]
    by_model = {row["model"]: row for row in summary}
    control = by_model["control_epoch50"]
    candidate = by_model["candidate_epoch50"]
    fields = (
        "best_ADE", "best_FDE", "median_candidate_FDE", "p90_candidate_FDE",
        "endpoint_center_offset", "endpoint_spread_rms", "endpoint_covariance_trace",
        *(f"hit_{threshold:g}m" for threshold in HIT_THRESHOLDS),
    )
    contrast = {
        "contrast": "candidate-control",
        **{f"delta_{field}": float(candidate[field]) - float(control[field])
           for field in fields},
    }
    hit_improvements = sum(
        contrast[f"delta_hit_{threshold:g}m"] > 0 for threshold in HIT_THRESHOLDS)
    clear_improvement = (
        contrast["delta_best_FDE"] < 0
        and contrast["delta_median_candidate_FDE"] < 0
        and contrast["delta_p90_candidate_FDE"] < 0
        and hit_improvements >= 3
    )
    return summary, contrast, {
        "status": "IMPROVED" if clear_improvement else "MIXED_OR_NOT_IMPROVED",
        "definition": (
            "best, median and p90 FDE all improve and at least three of four "
            "fixed-radius hit rates improve; spread alone is never sufficient"),
        "hit_thresholds_improved": hit_improvements,
    }


def verdicts(contrasts, coverage_verdict):
    candidate_control = next(
        row for row in contrasts if row["contrast"] == "candidate-control")
    candidate_epoch45 = next(
        row for row in contrasts if row["contrast"] == "candidate-epoch45")
    candidate_gdts = next(
        row for row in contrasts if row["contrast"] == "candidate-GDTS")
    marginal = all(
        candidate_control[f"delta_{metric}_mean"] < 0
        for metric in ("minADE", "minFDE"))
    joint = all(
        candidate_control[f"delta_{metric}_mean"] <= 0
        for metric in ("JADE", "JFDE"))
    better_epoch45 = all(
        candidate_epoch45[f"delta_{metric}_mean"] < 0 for metric in METRICS)
    original = (
        candidate_gdts["delta_minADE_mean"] <= 0
        and candidate_gdts["delta_minFDE_mean"] <= 0
        and candidate_gdts["delta_JADE_mean"] < 0
        and candidate_gdts["delta_JFDE_mean"] < 0)
    return {
        "short_marginal_effect": {
            "status": "PASSED" if marginal else "NOT_PASSED",
            "definition": "candidate improves both minADE and minFDE versus control",
        },
        "joint_preservation": {
            "status": "PASSED" if joint else "NOT_PASSED",
            "definition": "candidate does not degrade either JADE or JFDE versus control",
        },
        "short_target_effect": {
            "status": "PASSED" if marginal and joint else "NOT_PASSED",
            "definition": "both marginal metrics improve and neither joint metric degrades",
        },
        "coverage_effect": coverage_verdict,
        "candidate_better_than_epoch45": {
            "status": "PASSED" if better_epoch45 else "NOT_PASSED",
            "definition": "candidate improves all four means versus original epoch45",
        },
        "original_research_contract": {
            "status": "PASSED" if original else "NOT_PASSED",
            "definition": (
                "candidate marginal non-degradation and both joint improvements versus GDTS"),
        },
        "continue_rollout_weight_search": {
            "status": "SUPPORTED" if marginal and joint and better_epoch45 else "NOT_SUPPORTED",
            "definition": (
                "no automatic follow-up unless the sole-factor arm improves the short target "
                "and exceeds the retained epoch45 reference"),
        },
    }


def markdown_report(result: dict) -> str:
    means = {row["model"]: row for row in result["aggregate_metrics"]}
    contrasts = {row["contrast"]: row for row in result["contrasts"]}
    cc = contrasts["candidate-control"]
    ce = contrasts["candidate-epoch45"]
    cg = contrasts["candidate-GDTS"]
    coverage = result["coverage_contrast"]
    lines = [
        "---", "experiment_id: EBJD-B-261012-002", f"date: {result['date']}",
        "system: EBJD", "type: rollout_weight005_paired_pilot",
        "status: COMPLETED", "independent_test: false", "---", "",
        "# EBJD epoch45 rollout-weight 0.05 paired pilot", "",
        "## Frozen epoch50 results", "",
        "| model | minADE | minFDE | JADE | JFDE |", "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = means[method]
        lines.append(
            f"| {method} | {row['minADE_mean']:.6f} ± {row['minADE_sample_std']:.6f} | "
            f"{row['minFDE_mean']:.6f} ± {row['minFDE_sample_std']:.6f} | "
            f"{row['JADE_mean']:.6f} ± {row['JADE_sample_std']:.6f} | "
            f"{row['JFDE_mean']:.6f} ± {row['JFDE_sample_std']:.6f} |")
    lines.extend([
        "", "Five inference-seed means ± sample SD in metres; training seed is 3101.", "",
        "## Frozen conclusions", "",
        f"- Engineering contract: **{result['engineering_contract']['status']}**; "
        f"reused control: **{result['control_reuse']['status']}**.",
        f"- Candidate−control: ΔminADE `{cc['delta_minADE_mean']:+.6f}`, "
        f"ΔminFDE `{cc['delta_minFDE_mean']:+.6f}`, "
        f"ΔJADE `{cc['delta_JADE_mean']:+.6f}`, "
        f"ΔJFDE `{cc['delta_JFDE_mean']:+.6f}` m.",
        f"- Marginal effect: **{result['verdicts']['short_marginal_effect']['status']}**; "
        f"joint preservation: **{result['verdicts']['joint_preservation']['status']}**; "
        f"combined short target: **{result['verdicts']['short_target_effect']['status']}**.",
        f"- Coverage: **{result['verdicts']['coverage_effect']['status']}**; "
        f"Δmedian-candidate-FDE `{coverage['delta_median_candidate_FDE']:+.6f}`, "
        f"Δp90-candidate-FDE `{coverage['delta_p90_candidate_FDE']:+.6f}`, "
        f"Δspread-RMS `{coverage['delta_endpoint_spread_rms']:+.6f}` m.",
        f"- Candidate−epoch45: ΔminADE `{ce['delta_minADE_mean']:+.6f}`, "
        f"ΔminFDE `{ce['delta_minFDE_mean']:+.6f}`, "
        f"ΔJADE `{ce['delta_JADE_mean']:+.6f}`, "
        f"ΔJFDE `{ce['delta_JFDE_mean']:+.6f}` m; all-four verdict: "
        f"**{result['verdicts']['candidate_better_than_epoch45']['status']}**.",
        f"- Candidate−GDTS: ΔminADE `{cg['delta_minADE_mean']:+.6f}`, "
        f"ΔminFDE `{cg['delta_minFDE_mean']:+.6f}`, "
        f"ΔJADE `{cg['delta_JADE_mean']:+.6f}`, "
        f"ΔJFDE `{cg['delta_JFDE_mean']:+.6f}` m; original contract: "
        f"**{result['verdicts']['original_research_contract']['status']}**.",
        f"- Further rollout-weight search: "
        f"**{result['verdicts']['continue_rollout_weight_search']['status']}**.",
        "", "Spread is reported together with centre error and fixed-radius hit rates; "
        "greater dispersion alone is not treated as improved coverage.",
        "The HOTEL test mirror is not an independent held-out test (`independent_test=false`). "
        "No cross-training-seed significance claim is made.", "",
        "## Bound artifacts", "",
        "Detailed per-seed, prefix-K, shared-random-subset, strata, event, track and "
        "world-permutation tables are stored beside the structured JSON.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdts-bank", required=True)
    parser.add_argument("--epoch45-bank", required=True)
    parser.add_argument("--control-bank", required=True)
    parser.add_argument("--candidate-bank", required=True)
    parser.add_argument("--control-reuse-receipt", required=True)
    parser.add_argument("--training-receipt", required=True)
    parser.add_argument("--preflight-receipt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--archive-dir")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    roots = {
        "GDTS": Path(args.gdts_bank).resolve(),
        "EBJD_epoch45": Path(args.epoch45_bank).resolve(),
        "control_epoch50": Path(args.control_bank).resolve(),
        "candidate_epoch50": Path(args.candidate_bank).resolve(),
    }
    summaries, errors, endpoints, scenes, receipts, identity, rng = load_and_reduce(roots)
    per_seed, aggregate, contrasts = metric_tables(summaries)
    strata, paired_strata, _ = stratified_tables(endpoints, scenes)
    k_rows, k_summary = k_tables(errors)
    random_rows, random_summary = random_subset_tables(errors)
    permutation_rows, permutation_summary, permutation_invariants = permutation_tables(errors)
    occurrences, events, tracks, sensitivity, continuity, tail_mean = heavy_tail_tables(
        endpoints, scenes)
    coverage_summary, coverage_contrast, coverage_verdict = coverage_tables(strata)
    verdict = verdicts(contrasts, coverage_verdict)
    control_reuse = json.loads(Path(args.control_reuse_receipt).read_text())
    training = json.loads(Path(args.training_receipt).read_text())
    preflight = json.loads(Path(args.preflight_receipt).read_text())
    engineering_pass = (
        control_reuse.get("status") == "PASSED"
        and training.get("status") == "PASSED"
        and preflight.get("status") == "PASSED"
        and identity.get("status") == "PASSED"
        and rng.get("status") == "PASSED")
    tables = {
        "per_seed_metrics.csv": per_seed,
        "aggregate_metrics.csv": aggregate,
        "paired_contrasts.csv": contrasts,
        "endpoint_coverage_summary.csv": coverage_summary,
        "endpoint_coverage_contrast.csv": [coverage_contrast],
        "marginal_endpoint_strata.csv": strata,
        "candidate_control_paired_strata.csv": paired_strata,
        "k_prefix_per_seed.csv": k_rows,
        "k_curve_summary.csv": k_summary,
        "k_random_subset_per_seed_repeat.csv": random_rows,
        "k_random_subset_summary.csv": random_summary,
        "world_permutation_replicates.csv": permutation_rows,
        "world_permutation_summary.csv": permutation_summary,
        "candidate_control_occurrences.csv": occurrences,
        "candidate_control_deceleration_events.csv": events,
        "candidate_control_tracks.csv": tracks,
        "candidate_control_event_sensitivity.csv": sensitivity,
        "candidate_control_track_continuity.csv": continuity,
    }
    for name, rows in tables.items():
        write_csv(output / name, rows)
    atomic_json({
        "schema": "ebjd-rollout-weight005-bank-receipts-v1", "banks": receipts,
    }, output / "BANK_RECEIPTS.json")
    result = {
        "schema": "ebjd-epoch45-rollout-weight005-pilot-v1",
        "status": "COMPLETED", "date": str(date.today()),
        "training_seed": 3101, "inference_seeds": list(SEEDS),
        "independent_test": False,
        "counts": {"scenes_per_seed": int(receipts[0]["scene_count"]),
                   "agent_occurrences_per_seed": int(receipts[0]["agent_occurrences"]),
                   "worlds": 20, "steps": 20},
        "sole_factor": {"training.rollout_weight_end": {"control": 0.20,
                                                           "candidate": 0.05},
                        "goal_v_weight": 1.0,
                        "rollout_forward_backward_frequency_unchanged": True},
        "identity_matching": identity, "evaluation_rng_pairing": rng,
        "aggregate_metrics": aggregate, "contrasts": contrasts,
        "coverage_contrast": coverage_contrast,
        "engineering_contract": {"status": "PASSED" if engineering_pass else "FAILED"},
        "control_reuse": {"status": control_reuse.get("status"),
                          "receipt_sha256": sha256_file(args.control_reuse_receipt)},
        "training_pairing_receipt_sha256": sha256_file(args.training_receipt),
        "preflight_receipt_sha256": sha256_file(args.preflight_receipt),
        "verdicts": verdict,
        "heavy_tail": {"overall_mean_delta_minFDE_m": tail_mean,
                       "official_metric_exclusions": 0,
                       "top_deceleration_events": events[:10],
                       "top_tracks": tracks[:10], "sensitivity": sensitivity,
                       "continuity": continuity,
                       "event_independence_limitation": (
                           "temporal groups are heuristic, not independent samples")},
        "world_permutation_invariants": permutation_invariants,
        "world_permutation_summary": permutation_summary,
        "files": {},
    }
    for path in output.iterdir():
        if path.is_file() and path.suffix in {".csv", ".json"}:
            result["files"][path.name] = sha256_file(path)
    result_path = output / "EBJD_ROLLOUT_WEIGHT005_PILOT_2026-10-12.json"
    atomic_json(result, result_path)
    report_path = output / "EBJD_ROLLOUT_WEIGHT005_PILOT_2026-10-12.md"
    report_path.write_text(markdown_report(result), encoding="utf-8")
    receipt = {
        "status": "COMPLETED", "result": str(result_path),
        "result_sha256": sha256_file(result_path), "report": str(report_path),
        "report_sha256": sha256_file(report_path),
    }
    atomic_json(receipt, output / "FINAL_RESULT_RECEIPT.json")
    if args.archive_dir:
        archive = Path(args.archive_dir).resolve()
        archive.mkdir(parents=True, exist_ok=True)
        for path in output.iterdir():
            if path.is_file():
                shutil.copy2(path, archive / path.name)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

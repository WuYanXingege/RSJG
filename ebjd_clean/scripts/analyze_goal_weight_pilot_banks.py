#!/usr/bin/env python3
"""CPU-only four-model analysis for the preregistered goal-weight pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

from ebjd.diagnostic_bank import (
    SeedBank, assert_observation_parity, atomic_json, reorder_like, sha256_file,
)
from aggregate_epoch45_tail_events import (
    build_events, continuity_audit, motion, track_rows,
)
from diagnose_epoch45_banks import (
    Accumulator, endpoint_arrays, error_arrays, motion_metadata, quantiles,
    scene_categories,
)


SEEDS = (2035, 2036, 2037, 2038, 2039)
KS = (4, 8, 12, 16, 20)
METHODS = ("GDTS", "EBJD_epoch45", "control_epoch50", "candidate_epoch50")
METRICS = ("minADE", "minFDE", "JADE", "JFDE")
HIT_THRESHOLDS = (0.1, 0.2, 0.5, 1.0)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sample_std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def bank_path(root: Path, seed: int) -> Path:
    return root / f"seed_{seed}" / "manifest.json"


def load_and_reduce(roots: dict[str, Path]):
    banks = {
        method: {seed: SeedBank(bank_path(root, seed)) for seed in SEEDS}
        for method, root in roots.items()
    }
    errors = {method: {seed: {} for seed in SEEDS} for method in METHODS}
    endpoints = {method: {seed: {} for seed in SEEDS} for method in METHODS}
    summaries = {method: {} for method in METHODS}
    reference_scenes = {}
    receipts = []
    maximum_history = maximum_future = 0.0
    control_candidate_rng_equal = True
    rng_rows = []

    for method in METHODS:
        for seed in SEEDS:
            payload = banks[method][seed].payload
            if payload.get("status") != "COMPLETED":
                raise RuntimeError(f"incomplete bank: {method} seed {seed}")
            receipts.append({
                "model": method, "seed": seed,
                "manifest": str(banks[method][seed].path),
                "manifest_sha256": sha256_file(banks[method][seed].path),
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "config_sha256": payload["config_sha256"],
                "source_manifest_sha256": payload["source_manifest_sha256"],
                "scene_count": payload["scene_count"],
                "agent_occurrences": payload["agent_occurrences"],
                "replay_metrics": payload["replay_metrics"],
            })

    for seed in SEEDS:
        identity_sets = {
            method: set(banks[method][seed].identities()) for method in METHODS}
        reference_set = identity_sets["EBJD_epoch45"]
        if any(values != reference_set for values in identity_sets.values()):
            raise RuntimeError(f"bank window identities differ at seed {seed}")
        reference_order = banks["EBJD_epoch45"][seed].identities()
        accumulators = {method: Accumulator() for method in METHODS}
        for identity in reference_order:
            reference = banks["EBJD_epoch45"][seed].load(identity)
            if seed == SEEDS[0]:
                reference_scenes[identity] = reference
            for method in METHODS:
                scene = reference if method == "EBJD_epoch45" else reorder_like(
                    reference, banks[method][seed].load(identity))
                differences = assert_observation_parity(reference, scene, 1e-5)
                maximum_history = max(
                    maximum_history, differences["history_max_abs_m"])
                maximum_future = max(
                    maximum_future, differences["future_max_abs_m"])
                ade, fde = error_arrays(scene)
                errors[method][seed][identity] = (ade, fde)
                endpoints[method][seed][identity] = endpoint_arrays(scene, ade, fde)
                accumulators[method].update(ade, fde, np.arange(20))
        for method in METHODS:
            summaries[method][seed] = accumulators[method].result()

        control = banks["control_epoch50"][seed].payload
        candidate = banks["candidate_epoch50"][seed].payload
        rng_fields_present = all(
            item.get("rng_state_sha256_before") is not None
            and item.get("rng_state_sha256_after") is not None
            for payload in (control, candidate) for item in payload["files"])
        per_file_equal = rng_fields_present and all(
            left.get("rng_state_sha256_before") == right.get("rng_state_sha256_before")
            and left.get("rng_state_sha256_after") == right.get("rng_state_sha256_after")
            for left, right in zip(control["files"], candidate["files"], strict=True))
        chain_equal = (
            control.get("rng_state_chain_sha256") is not None
            and control.get("rng_state_chain_sha256")
            == candidate.get("rng_state_chain_sha256")
        )
        control_candidate_rng_equal &= per_file_equal and chain_equal
        rng_rows.append({
            "seed": seed, "rng_fields_present": rng_fields_present,
            "scene_rng_boundaries_equal": per_file_equal,
            "rng_state_chain_equal": chain_equal,
            "control_rng_state_chain_sha256": control.get("rng_state_chain_sha256"),
            "candidate_rng_state_chain_sha256": candidate.get("rng_state_chain_sha256"),
        })

    identity = {
        "history_max_abs_m": maximum_history,
        "future_max_abs_m": maximum_future,
        "tolerance_m": 1e-5,
        "status": "PASSED" if max(maximum_history, maximum_future) <= 1e-5 else "FAILED",
    }
    rng = {
        "status": "PASSED" if control_candidate_rng_equal else "FAILED",
        "comparison": "generator state before/after every scene; no draws added",
        "per_seed": rng_rows,
    }
    if identity["status"] != "PASSED" or rng["status"] != "PASSED":
        raise RuntimeError(f"bank pairing failed: identity={identity}, rng={rng}")
    return summaries, errors, endpoints, reference_scenes, receipts, identity, rng


def metric_tables(summaries):
    per_seed = []
    aggregate = []
    for method in METHODS:
        for seed in SEEDS:
            per_seed.append({"model": method, "seed": seed, **summaries[method][seed]})
        row = {"model": method, "inference_seeds": len(SEEDS)}
        for metric in (*METRICS, "scene_weighted_minADE", "scene_weighted_minFDE"):
            values = [float(summaries[method][seed][metric]) for seed in SEEDS]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_sample_std"] = sample_std(values)
        aggregate.append(row)
    contrasts = []
    pairs = (
        ("candidate-control", "candidate_epoch50", "control_epoch50"),
        ("control-epoch45", "control_epoch50", "EBJD_epoch45"),
        ("candidate-epoch45", "candidate_epoch50", "EBJD_epoch45"),
        ("epoch45-GDTS", "EBJD_epoch45", "GDTS"),
        ("control-GDTS", "control_epoch50", "GDTS"),
        ("candidate-GDTS", "candidate_epoch50", "GDTS"),
    )
    for label, left, right in pairs:
        row = {"contrast": label, "left": left, "right": right}
        for metric in METRICS:
            values = [
                float(summaries[left][seed][metric])
                - float(summaries[right][seed][metric]) for seed in SEEDS]
            row[f"delta_{metric}_mean"] = float(np.mean(values))
            row[f"delta_{metric}_sample_std"] = sample_std(values)
            row[f"delta_{metric}_improved_seed_count"] = sum(value < 0 for value in values)
            row[f"delta_{metric}_per_seed_json"] = json.dumps(values, separators=(",", ":"))
        contrasts.append(row)
    return per_seed, aggregate, contrasts


def stratified_tables(endpoints, reference_scenes):
    aggregate = defaultdict(lambda: defaultdict(list))
    identifiers = defaultdict(lambda: {"scenes": set(), "agents": set()})
    paired = defaultdict(lambda: defaultdict(list))
    metadata = {
        identity: motion_metadata(scene) for identity, scene in reference_scenes.items()}
    for seed in SEEDS:
        for identity, scene in reference_scenes.items():
            categories = scene_categories(scene, metadata[identity])
            for method in METHODS:
                values = endpoints[method][seed][identity]
                for group_type, category, mask in categories:
                    if not mask.any():
                        continue
                    key = (method, group_type, category)
                    for name, data in values.items():
                        aggregate[key][name].extend(data[mask].tolist())
                    identifiers[key]["scenes"].add(identity)
                    for index in np.flatnonzero(mask):
                        identifiers[key]["agents"].add((identity, str(scene.agent_ids[index])))
            control = endpoints["control_epoch50"][seed][identity]
            candidate = endpoints["candidate_epoch50"][seed][identity]
            for group_type, category, mask in categories:
                if not mask.any():
                    continue
                key = (group_type, category)
                for name in ("best_ADE", "best_FDE", "endpoint_center_offset",
                             "endpoint_spread_rms", "hit_0.2m", "hit_0.5m"):
                    paired[key][name].extend((candidate[name] - control[name])[mask].tolist())
    rows = []
    for key in sorted(aggregate):
        method, group_type, category = key
        row = {
            "model": method, "group_type": group_type, "category": category,
            "unique_scenes": len(identifiers[key]["scenes"]),
            "unique_agent_occurrences": len(identifiers[key]["agents"]),
        }
        for name, values in aggregate[key].items():
            row[name] = float(np.mean(values))
        rows.append(row)
    paired_rows = []
    for (group_type, category), values_by_name in sorted(paired.items()):
        row = {"contrast": "candidate-control", "group_type": group_type,
               "category": category}
        for name, values in values_by_name.items():
            for label, value in quantiles(values).items():
                row[f"delta_{name}_{label}"] = value
        paired_rows.append(row)
    return rows, paired_rows, metadata


def k_tables(errors):
    rows = []
    for method in METHODS:
        for seed in SEEDS:
            previous = None
            for k in KS:
                accumulator = Accumulator()
                for ade, fde in errors[method][seed].values():
                    accumulator.update(ade, fde, np.arange(k))
                result = accumulator.result()
                if previous is not None:
                    for metric in METRICS:
                        if result[metric] > previous[metric] + 1e-12:
                            raise RuntimeError("nested K metric is not monotone")
                previous = result
                rows.append({"selection": "shared_nested_prefix", "model": method,
                             "seed": seed, "K": k, **result})
    summary = []
    for method in METHODS:
        for k in KS:
            selected = [row for row in rows if row["model"] == method and row["K"] == k]
            item = {"model": method, "K": k, "seed_count": len(selected)}
            for metric in METRICS:
                values = [float(row[metric]) for row in selected]
                item[f"{metric}_mean"] = float(np.mean(values))
                item[f"{metric}_sample_std"] = sample_std(values)
            summary.append(item)
    return rows, summary


def permutation_tables(errors):
    rows = []
    invariant_marginal_max = 0.0
    for method in METHODS:
        for seed in SEEDS:
            original = Accumulator()
            for ade, fde in errors[method][seed].values():
                original.update(ade, fde, np.arange(20))
            original_result = original.result()
            for repeat in range(100):
                shuffled = Accumulator()
                for identity, (ade, fde) in errors[method][seed].items():
                    shuffled_ade = np.empty_like(ade)
                    shuffled_fde = np.empty_like(fde)
                    for agent in range(ade.shape[1]):
                        seed_value = int.from_bytes(
                            __import__("hashlib").sha256(
                                f"goal-weight-permutation\x1f{seed}\x1f{identity}\x1f{repeat}\x1f{agent}".encode()
                            ).digest()[:8], "big")
                        permutation = np.random.default_rng(seed_value).permutation(20)
                        shuffled_ade[:, agent] = ade[permutation, agent]
                        shuffled_fde[:, agent] = fde[permutation, agent]
                    invariant_marginal_max = max(
                        invariant_marginal_max,
                        float(np.max(np.abs(shuffled_ade.min(0) - ade.min(0)))),
                        float(np.max(np.abs(shuffled_fde.min(0) - fde.min(0)))),
                    )
                    shuffled.update(shuffled_ade, shuffled_fde, np.arange(20))
                result = shuffled.result()
                rows.append({
                    "model": method, "seed": seed, "repeat": repeat,
                    "original_JADE": original_result["JADE"],
                    "original_JFDE": original_result["JFDE"],
                    "shuffled_JADE": result["JADE"],
                    "shuffled_JFDE": result["JFDE"],
                    "delta_JADE": result["JADE"] - original_result["JADE"],
                    "delta_JFDE": result["JFDE"] - original_result["JFDE"],
                })
    summary = []
    for method in METHODS:
        selected = [row for row in rows if row["model"] == method]
        item = {"model": method, "records": len(selected)}
        for name in ("original_JADE", "original_JFDE", "shuffled_JADE",
                     "shuffled_JFDE", "delta_JADE", "delta_JFDE"):
            for label, value in quantiles([row[name] for row in selected]).items():
                item[f"{name}_{label}"] = value
        summary.append(item)
    invariants = {
        "marginal_max_abs_m": invariant_marginal_max,
        "status": "PASSED" if invariant_marginal_max <= 1e-12 else "FAILED",
    }
    if invariants["status"] != "PASSED":
        raise RuntimeError(f"permutation invariant failed: {invariants}")
    return rows, summary, invariants


def heavy_tail_tables(endpoints, reference_scenes):
    occurrences = []
    for identity, scene in reference_scenes.items():
        for index, agent in enumerate(scene.agent_ids.tolist()):
            row = {
                "occurrence_key": f"{identity}|agent={agent}",
                "source_sequence": scene.source_sequence,
                "window_key": identity, "frame_start": int(scene.frame_ids[0]),
                "frame_end": int(scene.frame_ids[-1]), "agent_id": str(agent),
                "inference_seed_count": len(SEEDS),
                **motion(scene.history[index].astype(np.float64),
                         scene.future[index].astype(np.float64), float(scene.dt_seconds)),
            }
            means = {}
            for method in METHODS:
                means[method] = {
                    name: float(np.mean([
                        endpoints[method][seed][identity][name][index] for seed in SEEDS]))
                    for name in endpoints[method][SEEDS[0]][identity]
                }
                for name, value in means[method].items():
                    row[f"{method}_{name}"] = value
            row["delta_minADE_m"] = (
                means["candidate_epoch50"]["best_ADE"]
                - means["control_epoch50"]["best_ADE"])
            row["delta_minFDE_m"] = (
                means["candidate_epoch50"]["best_FDE"]
                - means["control_epoch50"]["best_FDE"])
            occurrences.append(row)
    occurrences.sort(key=lambda item: item["delta_minFDE_m"], reverse=True)
    events = build_events(occurrences)
    tracks = track_rows(occurrences)
    total = len(occurrences)
    overall = float(np.mean([row["delta_minFDE_m"] for row in occurrences]))
    for collection in (events, tracks):
        for row in collection:
            contribution = float(row["sum_delta_minFDE_m"]) / total
            row["signed_contribution_to_overall_minFDE_m"] = contribution
            row["signed_contribution_percent_of_net_delta"] = (
                100 * contribution / overall if abs(overall) > 1e-15 else math.nan)
    sensitivity = []
    all_keys = {row["occurrence_key"] for row in occurrences}
    for count in (1, 3, 5):
        removed = {
            key for event in events[:count]
            for key in json.loads(event["occurrence_keys_json"])}
        kept = [row["delta_minFDE_m"] for row in occurrences
                if row["occurrence_key"] in all_keys - removed]
        sensitivity.append({
            "excluded_top_deceleration_events": count,
            "excluded_unique_occurrences": len(removed),
            "remaining_occurrences": len(kept),
            "sensitivity_mean_delta_minFDE_m": float(np.mean(kept)),
        })
    top_tracks = {(row["source_sequence"], row["agent_id"]) for row in tracks[:10]}
    continuity = continuity_audit(reference_scenes, top_tracks)
    return occurrences, events, tracks, sensitivity, continuity, overall


def markdown_report(result: dict) -> str:
    means = {row["model"]: row for row in result["aggregate_metrics"]}
    contrast = next(row for row in result["contrasts"] if row["contrast"] == "candidate-control")
    candidate_gdts = next(row for row in result["contrasts"] if row["contrast"] == "candidate-GDTS")
    lines = [
        "---", "experiment_id: EBJD-B-261012-001", "date: 2026-10-12",
        "system: EBJD", "type: goal_weight_paired_pilot", "status: COMPLETED",
        "independent_test: false", "---", "",
        "# EBJD epoch 45 goal-weight paired pilot", "",
        "## Frozen epoch-50 results", "",
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
        "", "Values are five inference-seed means ± sample SD in metres; training seed is 3101.", "",
        "## Preregistered conclusions", "",
        f"- Engineering contract: **{result['engineering_contract']['status']}**.",
        f"- Candidate−control ΔminADE: `{contrast['delta_minADE_mean']:+.6f} m`; "
        f"ΔminFDE: `{contrast['delta_minFDE_mean']:+.6f} m`.",
        f"- Candidate−control ΔJADE: `{contrast['delta_JADE_mean']:+.6f} m`; "
        f"ΔJFDE: `{contrast['delta_JFDE_mean']:+.6f} m`.",
        f"- Short target-effect verdict: **{result['short_target_effect']['status']}**.",
        f"- Candidate relative to GDTS contract: **{result['original_research_contract']['status']}** "
        f"(ΔminADE `{candidate_gdts['delta_minADE_mean']:+.6f}`, "
        f"ΔminFDE `{candidate_gdts['delta_minFDE_mean']:+.6f}`, "
        f"ΔJADE `{candidate_gdts['delta_JADE_mean']:+.6f}`, "
        f"ΔJFDE `{candidate_gdts['delta_JFDE_mean']:+.6f}` m).",
        f"- Candidate−control heavy-tail mean ΔminFDE: "
        f"`{result['heavy_tail']['overall_mean_delta_minFDE_m']:+.6f} m`.",
        "", "The HOTEL test mirror is not an independent held-out test (`independent_test=false`). "
        "No cross-training-seed significance claim is made.", "",
        "## Bound artifacts", "",
        f"- Result JSON SHA256: `{result['self_sha256_placeholder']}` (replaced in receipt after write).",
        "- Detailed CSV tables and bank receipts are stored beside the structured result.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdts-bank", required=True)
    parser.add_argument("--epoch45-bank", required=True)
    parser.add_argument("--control-bank", required=True)
    parser.add_argument("--candidate-bank", required=True)
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
    summaries, errors, endpoints, reference_scenes, receipts, identity, rng = (
        load_and_reduce(roots))
    per_seed, aggregate, contrasts = metric_tables(summaries)
    strata, paired_strata, _ = stratified_tables(endpoints, reference_scenes)
    k_rows, k_summary = k_tables(errors)
    permutation_rows, permutation_summary, permutation_invariants = (
        permutation_tables(errors))
    occurrences, events, tracks, sensitivity, continuity, tail_mean = (
        heavy_tail_tables(endpoints, reference_scenes))

    tables = {
        "per_seed_metrics.csv": per_seed,
        "aggregate_metrics.csv": aggregate,
        "paired_contrasts.csv": contrasts,
        "marginal_endpoint_strata.csv": strata,
        "candidate_control_paired_strata.csv": paired_strata,
        "k_prefix_per_seed.csv": k_rows,
        "k_curve_summary.csv": k_summary,
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
    atomic_json({"schema": "ebjd-goal-weight-pilot-bank-receipts-v1",
                 "banks": receipts}, output / "BANK_RECEIPTS.json")

    candidate_control = next(
        row for row in contrasts if row["contrast"] == "candidate-control")
    candidate_gdts = next(
        row for row in contrasts if row["contrast"] == "candidate-GDTS")
    short_pass = all(candidate_control[f"delta_{metric}_mean"] < 0 for metric in METRICS)
    original_pass = (
        candidate_gdts["delta_minADE_mean"] <= 0
        and candidate_gdts["delta_minFDE_mean"] <= 0
        and candidate_gdts["delta_JADE_mean"] < 0
        and candidate_gdts["delta_JFDE_mean"] < 0
    )
    result = {
        "schema": "ebjd-epoch45-goal-weight-paired-pilot-v1",
        "status": "COMPLETED", "date": str(date.today()),
        "training_seed": 3101, "inference_seeds": list(SEEDS),
        "independent_test": False,
        "counts": {
            "scenes_per_seed": int(receipts[0]["scene_count"]),
            "agent_occurrences_per_seed": int(receipts[0]["agent_occurrences"]),
                   "worlds": 20, "steps": 20},
        "identity_matching": identity, "evaluation_rng_pairing": rng,
        "aggregate_metrics": aggregate, "contrasts": contrasts,
        "engineering_contract": {"status": "PASSED"},
        "short_target_effect": {
            "status": "PASSED" if short_pass else "NOT_PASSED",
            "definition": "candidate improves all four mean metrics versus control",
        },
        "original_research_contract": {
            "status": "PASSED" if original_pass else "NOT_PASSED",
            "definition": "candidate marginal non-degradation and both joint improvements vs GDTS",
        },
        "heavy_tail": {
            "overall_mean_delta_minFDE_m": tail_mean,
            "event_independence_limitation": "temporal groups are heuristic, not independent samples",
            "official_metric_exclusions": 0,
            "top_deceleration_events": events[:10], "top_tracks": tracks[:10],
            "sensitivity": sensitivity, "continuity": continuity,
        },
        "world_permutation_invariants": permutation_invariants,
        "world_permutation_summary": permutation_summary,
        "files": {}, "self_sha256_placeholder": "see FINAL_RESULT_RECEIPT.json",
    }
    for path in output.iterdir():
        if path.is_file() and path.suffix in {".csv", ".json"}:
            result["files"][path.name] = sha256_file(path)
    result_path = output / "EBJD_EPOCH045_PAIRED_GOAL_WEIGHT_PILOT_2026-10-12.json"
    atomic_json(result, result_path)
    report_path = output / "EBJD_EPOCH045_PAIRED_GOAL_WEIGHT_PILOT_2026-10-12.md"
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

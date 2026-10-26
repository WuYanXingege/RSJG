#!/usr/bin/env python3
"""CPU-only frozen endpoint attribution for ZARA1 A/B/C banks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


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


def load_bank(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as payload:
        bank = {name: payload[name] for name in payload.files}
    metadata = json.loads(str(bank.pop("metadata_json")))
    expected = {
        "schema": "ebjd-frozen-endpoint-bank-v1", "seed": 2035,
        "worlds": 20, "ddim_steps": 20, "coordinate_unit": "m",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"{path}: metadata {key} mismatch")
    bank["metadata"] = metadata
    bank["path"] = str(path.resolve())
    bank["sha256"] = sha256(path)
    return bank


def validate_alignment(banks: dict[str, dict]) -> None:
    reference = banks["A"]
    for label, bank in banks.items():
        for key in ("scene_ids", "agent_ids", "valid", "truth"):
            if not np.array_equal(bank[key], reference[key], equal_nan=True):
                raise ValueError(f"bank {label} identity/truth mismatch at {key}")
        if bank["endpoints"].shape != reference["endpoints"].shape:
            raise ValueError(f"bank {label} endpoint shape mismatch")
        valid = np.broadcast_to(bank["valid"][:, None, :, None], bank["endpoints"].shape)
        if not np.isfinite(bank["endpoints"][valid]).all():
            raise ValueError(f"bank {label} has non-finite valid endpoints")
    a_id = banks["A"]["metadata"].get("shared_forward_capture_id")
    b_id = banks["B"]["metadata"].get("shared_forward_capture_id")
    if not a_id or a_id != b_id:
        raise ValueError("GDTS final/proposal banks do not share one forward capture")


def per_agent_errors(bank: dict, indices: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    endpoints = bank["endpoints"]
    if indices is not None:
        endpoints = endpoints[:, indices]
    truth = bank["truth"][:, None]
    distances = np.linalg.norm(endpoints.astype(np.float64) - truth.astype(np.float64), axis=-1)
    distances = np.where(bank["valid"][:, None], distances, np.inf)
    best = distances.min(axis=1)
    return best[bank["valid"]], distances


def summarize(bank: dict, scene_selector: np.ndarray | None = None) -> dict:
    selector = np.ones(len(bank["scene_ids"]), dtype=bool) if scene_selector is None else scene_selector
    valid = bank["valid"] & selector[:, None]
    distances = np.linalg.norm(
        bank["endpoints"].astype(np.float64) - bank["truth"][:, None].astype(np.float64),
        axis=-1)
    distances = np.where(valid[:, None], distances, np.inf)
    best = distances.min(axis=1)[valid]
    scene_joint = []
    for scene in np.flatnonzero(selector):
        mask = bank["valid"][scene]
        scene_joint.append(distances[scene, :, mask].mean(axis=1).min())
    centers = bank["endpoints"].astype(np.float64).mean(axis=1)
    centered = bank["endpoints"].astype(np.float64) - centers[:, None]
    rms = np.sqrt(np.square(centered).sum(axis=-1).mean(axis=1))[valid]
    center_error = np.linalg.norm(centers - bank["truth"].astype(np.float64), axis=-1)[valid]
    order = np.sort(best)[::-1]
    top_count = max(1, int(np.ceil(0.10 * len(order))))
    return {
        "minFDE": float(best.mean()), "JFDE": float(np.mean(scene_joint)),
        "Recall@0.25m": float(np.mean(best <= 0.25)),
        "Recall@0.5m": float(np.mean(best <= 0.5)),
        "Recall@1.0m": float(np.mean(best <= 1.0)),
        "P50": float(np.quantile(best, 0.50)),
        "P90": float(np.quantile(best, 0.90)),
        "P95": float(np.quantile(best, 0.95)),
        "candidate_rms_radius": float(rms.mean()),
        "candidate_center_to_gt": float(center_error.mean()),
        "top10_error_contribution": float(order[:top_count].sum() / order.sum()),
        "agent_count": int(valid.sum()), "scene_count": int(selector.sum()),
    }


def paired(left: np.ndarray, right: np.ndarray) -> dict:
    delta = left - right
    return {
        "mean_delta": float(delta.mean()), "median_delta": float(np.median(delta)),
        "left_lower_fraction": float(np.mean(delta < 0)),
        "right_lower_fraction": float(np.mean(delta > 0)),
        "equal_fraction": float(np.mean(delta == 0)),
    }


def k_curves(banks: dict[str, dict], repeats: int = 20, seed: int = 26026) -> dict:
    generator = np.random.default_rng(seed)
    sizes = (4, 8, 12, 20)
    values = {label: {size: [] for size in sizes} for label in banks}
    for _ in range(repeats):
        permutation = generator.permutation(20)
        for size in sizes:
            indices = permutation[:size]
            for label, bank in banks.items():
                errors, _ = per_agent_errors(bank, indices)
                values[label][size].append(float(errors.mean()))
    return {
        label: {
            str(size): {
                "mean_minFDE": float(np.mean(samples)),
                "sample_standard_deviation": float(np.std(samples, ddof=1)),
            } for size, samples in by_size.items()
        } for label, by_size in values.items()
    } | {"protocol": {
        "name": "fixed_bank_nested_random_subsets", "repeats": repeats,
        "permutation_seed": seed, "shared_indices_across_A_B_C": True,
    }}


def write_csv(path: Path, summaries: dict) -> None:
    fields = [
        "path", "group", "minFDE", "JFDE", "Recall@0.25m", "Recall@0.5m",
        "Recall@1.0m", "P50", "P90", "P95", "candidate_rms_radius",
        "candidate_center_to_gt", "top10_error_contribution", "agent_count",
        "scene_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, groups in summaries.items():
            for group, values in groups.items():
                writer.writerow({"path": label, "group": group, **values})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-a", required=True)
    parser.add_argument("--bank-b", required=True)
    parser.add_argument("--bank-c", required=True)
    parser.add_argument("--historical-gdts-result", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    banks = {
        "A": load_bank(Path(args.bank_a)),
        "B": load_bank(Path(args.bank_b)),
        "C": load_bank(Path(args.bank_c)),
    }
    validate_alignment(banks)
    counts = banks["A"]["valid"].sum(axis=1)
    groups = {"all": np.ones(len(counts), dtype=bool), "N=1": counts == 1, "N>=2": counts >= 2}
    summaries = {
        label: {name: summarize(bank, selector) for name, selector in groups.items()}
        for label, bank in banks.items()
    }
    historical = json.loads(Path(args.historical_gdts_result).read_text(encoding="utf-8"))
    matching_runs = [
        run for run in historical.get("runs", []) if int(run.get("seed", -1)) == 2035]
    if len(matching_runs) != 1:
        raise ValueError("historical GDTS result must contain exactly one seed-2035 run")
    historical_minfde = float(matching_runs[0]["metrics"]["minFDE"])
    reproduction_delta = summaries["A"]["all"]["minFDE"] - historical_minfde
    if abs(reproduction_delta) > 1e-6:
        raise RuntimeError(
            f"GDTS endpoint bank does not reproduce bound minFDE: delta={reproduction_delta}")
    errors = {label: per_agent_errors(bank)[0] for label, bank in banks.items()}
    paired_results = {
        "B-A": paired(errors["B"], errors["A"]),
        "C-B": paired(errors["C"], errors["B"]),
        "C-A": paired(errors["C"], errors["A"]),
    }
    f_a = summaries["A"]["all"]["minFDE"]
    f_b = summaries["B"]["all"]["minFDE"]
    f_c = summaries["C"]["all"]["minFDE"]
    recovery = ((f_c - f_b) / (f_c - f_a)) if f_c != f_a else None
    native_supported = bool(
        f_c > f_a and recovery is not None and recovery >= 0.70
        and f_c - f_b >= 0.01
        and summaries["B"]["all"]["Recall@0.5m"] >= summaries["C"]["all"]["Recall@0.5m"])
    refinement_priority = bool(
        f_c - f_a >= 0.01
        and (recovery is None or recovery <= 0.30 or f_b >= f_c))
    if native_supported:
        classification = "NATIVE_PRIOR_SUPPORTED"
        priority = "可靠终点先验＋条件联合桥，并以受约束终点残差保持可修正性"
    elif refinement_priority:
        classification = "DIFFUSION_REFINEMENT_PRIORITY"
        priority = "先研究GDTS完整轨迹去噪如何修正终点，再决定复用或蒸馏路径"
    else:
        classification = "PARTIAL_SUPPORT"
        priority = (
            "只做一次原生goal条件注入EBJD联合桥的冻结适配验证"
            if f_b < f_c else
            "只做一次GDTS去噪终点修正路径的冻结机制验证")
    result = {
        "schema": "ebjd-zara1-frozen-endpoint-attribution-v1",
        "status": "COMPLETE", "optimizer_updates": 0,
        "banks": {label: {
            "path": bank["path"], "sha256": bank["sha256"],
            "metadata": bank["metadata"],
        } for label, bank in banks.items()},
        "summaries": summaries, "paired_best_endpoint_error": paired_results,
        "k_subsampling": k_curves(banks),
        "historical_gdts_reproduction": {
            "expected_minFDE": historical_minfde,
            "bank_minFDE": f_a, "difference": reproduction_delta,
        },
        "decision": {
            "F_A": f_a, "F_B": f_b, "F_C": f_c,
            "native_proposal_recovery_R": recovery,
            "classification": classification,
            "single_priority_direction": priority,
            "thresholds": {
                "native_R_min": 0.70, "native_B_over_C_minFDE_gain_min_m": 0.01,
                "native_recall_0.5_not_below_C": True,
                "refinement_R_max": 0.30,
                "clearly_better_A_over_C_minFDE_m": 0.01,
            },
        },
        "evidence_boundary": (
            "one training checkpoint per method and one inference seed; "
            "K curves are CPU fixed-bank nested subsamples, not sampler reruns"),
    }
    output = Path(args.output_dir)
    atomic_json(output / "SUMMARY.json", result)
    write_csv(output / "METRICS.csv", summaries)
    print(json.dumps(result["decision"], sort_keys=True))


if __name__ == "__main__":
    main()

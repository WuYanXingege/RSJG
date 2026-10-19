"""CLI for real fixed-bank endpoint-spread calibration audits."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from .diagnostic_bank import SeedBank, sha256_file
from .endpoint_calibration import occurrence_changes, summarize_factor


FORMAL_METRICS = ("minADE", "minFDE", "JADE", "JFDE")
GDTS_REFERENCE = {
    "minADE": 0.417052859, "minFDE": 0.580032304,
    "JADE": 0.489906767, "JFDE": 0.743411792,
}


def _verify_external_bindings(payload: dict) -> dict:
    checks = {}
    for name, hash_name in (
        ("source_manifest", "source_manifest_sha256"),
        ("checkpoint", "checkpoint_sha256"),
        ("config", "config_sha256"),
    ):
        path_value = payload.get(name)
        expected = payload.get(hash_name)
        path = Path(path_value).resolve() if path_value else None
        actual = sha256_file(path) if path is not None and path.is_file() else None
        checks[name] = {
            "path": str(path) if path is not None else None,
            "exists": bool(path is not None and path.is_file()),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "sha256_matches": bool(expected and actual == expected),
        }
    checks["all_sha256_match"] = all(
        checks[name]["sha256_matches"]
        for name in ("source_manifest", "checkpoint", "config"))
    return checks


def _json_ready(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _select_factor(results: list[dict]) -> float:
    baseline = results[0]["metrics"]
    all_better = [
        row for row in results[1:]
        if all(row["metrics"][key] < baseline[key] for key in FORMAL_METRICS)
    ]
    candidates = all_better or results[1:]
    return float(min(
        candidates,
        key=lambda row: np.mean([
            (row["metrics"][key] - baseline[key]) / baseline[key]
            for key in FORMAL_METRICS
        ]),
    )["factor"])


def _case_rows(scenes, factor: float) -> list[dict]:
    rows = occurrence_changes(scenes, factor)
    ordered = sorted(rows, key=lambda row: row["delta_minFDE_m"])
    selected = [
        ("largest_improvement", ordered[0]),
        ("largest_regression", ordered[-1]),
        ("largest_center_bias", max(
            rows, key=lambda row: row["endpoint_center_bias_m"])),
        ("largest_expansion", max(
            rows, key=lambda row: row["max_endpoint_radius_m"])),
    ]
    output = []
    seen = set()
    for rule, row in selected:
        identity = (row["scene_index"], row["agent_index"])
        if identity in seen:
            continue
        seen.add(identity)
        output.append({"selection_rule": rule, **row})
    return output


def _plot_cases(scenes, factors, rows: list[dict], output_dir: Path) -> list[str]:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/rsjg-matplotlib")
    import matplotlib.pyplot as plt

    from .endpoint_calibration import calibrate_endpoint_spread

    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    for case_index, row in enumerate(rows):
        scene = scenes[row["scene_index"]]
        agent = row["agent_index"]
        figure, axes = plt.subplots(1, len(factors), figsize=(4 * len(factors), 4))
        axes = np.atleast_1d(axes)
        for axis, factor in zip(axes, factors):
            prediction = calibrate_endpoint_spread(scene.predictions, factor)
            axis.plot(scene.history[agent, :, 0], scene.history[agent, :, 1],
                      "k.-", linewidth=2, label="observed")
            axis.plot(scene.future[agent, :, 0], scene.future[agent, :, 1],
                      "g.-", linewidth=2, label="GT")
            for world in range(prediction.shape[0]):
                axis.plot(prediction[world, agent, :, 0],
                          prediction[world, agent, :, 1],
                          color="#d95f02", alpha=0.25, linewidth=0.8)
            axis.scatter(prediction[:, agent, -1, 0], prediction[:, agent, -1, 1],
                         s=9, color="#d95f02", alpha=0.7)
            axis.set_title(f"gamma={factor:g}")
            axis.set_aspect("equal", adjustable="datalim")
            axis.grid(alpha=0.2)
        axes[0].legend(fontsize=8)
        figure.suptitle(
            f"{row['selection_rule']} | {row['window_key']} | agent {row['agent_id']}\n"
            f"delta minFDE={row['delta_minFDE_m']:+.3f} m",
            fontsize=10)
        figure.tight_layout()
        stem = f"case_{case_index:02d}_{row['selection_rule']}"
        for suffix in ("png", "pdf"):
            path = output_dir / f"{stem}.{suffix}"
            figure.savefig(path, dpi=180, bbox_inches="tight")
            files.append(str(path.resolve()))
        plt.close(figure)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit endpoint spread on one real fixed trajectory bank")
    parser.add_argument("--bank", required=True,
                        help="seed manifest.json (bank.json alias accepted)")
    parser.add_argument("--factors", type=float, nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-output-dir")
    args = parser.parse_args()
    start = time.time()
    bank_path = Path(args.bank).resolve()
    if bank_path.name == "bank.json" and not bank_path.exists():
        bank_path = bank_path.with_name("manifest.json")
    factors = [float(value) for value in args.factors]
    if factors != [1.0, 1.25, 1.5, 2.0]:
        raise ValueError("formal audit factors must be exactly 1, 1.25, 1.5, 2")
    bank = SeedBank(bank_path, verify_hashes=True)
    scenes = [bank.load(identity) for identity in bank.identities()]
    actual_agent_occurrences = sum(scene.agents for scene in scenes)
    expected_contract = {
        "scene_count": 253, "agent_occurrences": 364, "worlds": 20,
        "steps": 20, "inference_seed": 2035,
        "source_manifest_sha256": (
            "03f9b6bef89d43137b31782bbf0eb5b16d19f8ddfe378e2c7c62e700e128397b"),
    }
    actual_contract = {
        "scene_count": len(scenes),
        "agent_occurrences": actual_agent_occurrences,
        "worlds": int(scenes[0].predictions.shape[0]),
        "steps": int(bank.payload.get("steps", -1)),
        "inference_seed": int(bank.payload.get("inference_seed", -1)),
        "source_manifest_sha256": bank.payload.get("source_manifest_sha256"),
    }
    if actual_contract != expected_contract:
        raise ValueError(
            f"bank does not satisfy fixed ETH audit contract: {actual_contract}")
    binding_verification = _verify_external_bindings(bank.payload)
    if not binding_verification["all_sha256_match"]:
        raise ValueError("external manifest/config/checkpoint SHA256 verification failed")
    results = [summarize_factor(scenes, factor) for factor in factors]

    baseline = results[0]
    for row in results:
        expected = float(row["factor"])
        observed = (row["coverage"]["endpoint_rms_radius_m"] /
                    baseline["coverage"]["endpoint_rms_radius_m"])
        row["invariants"]["endpoint_rms_radius_ratio"] = observed
        row["invariants"]["endpoint_rms_radius_expected_ratio"] = expected
        row["invariants"]["endpoint_rms_radius_ratio_abs_error"] = abs(
            observed - expected)
        row["comparison"] = {
            "delta_vs_original_m": {
                key: row["metrics"][key] - baseline["metrics"][key]
                for key in FORMAL_METRICS
            },
            "delta_vs_matched_GDTS_m": {
                key: row["metrics"][key] - GDTS_REFERENCE[key]
                for key in FORMAL_METRICS
            },
            "all_four_improve_vs_original": all(
                row["metrics"][key] < baseline["metrics"][key]
                for key in FORMAL_METRICS),
            "all_four_reach_matched_GDTS": all(
                row["metrics"][key] <= GDTS_REFERENCE[key]
                for key in FORMAL_METRICS),
        }

    replay = bank.payload.get("replay_metrics", {})
    replay_delta = {
        key: baseline["metrics"][key] - float(replay[key])
        for key in FORMAL_METRICS
    }
    chosen = _select_factor(results)
    cases = _case_rows(scenes, chosen)
    case_files: list[str] = []
    if args.case_output_dir:
        case_files = _plot_cases(
            scenes, factors, cases, Path(args.case_output_dir).resolve())
    any_consistent = any(
        row["comparison"]["all_four_improve_vs_original"] for row in results[1:])
    any_reaches = any(
        row["comparison"]["all_four_reach_matched_GDTS"] for row in results[1:])
    status = ("RECOVERED_TO_MATCHED_GDTS" if any_reaches else
              "PARTIAL_RECOVERY" if any_consistent else
              "NO_CONSISTENT_RECOVERY")
    payload = {
        "schema": "ebjd-endpoint-calibration-audit-v1",
        "status": status,
        "optimizer_updates": 0,
        "bank_manifest": str(bank_path),
        "bank_manifest_sha256": sha256_file(bank_path),
        "bank_binding": {
            key: bank.payload.get(key) for key in (
                "bank_schema", "status", "scene_count", "agent_occurrences",
                "worlds", "steps", "inference_seed", "source_manifest",
                "source_manifest_sha256", "checkpoint", "checkpoint_sha256",
                "config", "config_sha256", "execution_commit",
                "coordinate_conversion", "dtype", "autocast")
        },
        "fixed_contract": {
            "expected": expected_contract, "actual": actual_contract,
            "matches": actual_contract == expected_contract,
        },
        "external_binding_verification": binding_verification,
        "formula": (
            "Y_new[k,i,t]=Y[k,i,t]+(t/12)*(gamma-1)*"
            "(g[k,i]-mean_worlds(g[:,i]))"),
        "factors": factors,
        "matched_GDTS_reference": GDTS_REFERENCE,
        "gamma1_replay": {
            "bank_replay_metrics": {key: replay.get(key) for key in FORMAL_METRICS},
            "absolute_differences_m": replay_delta,
            "maximum_absolute_difference_m": max(abs(x) for x in replay_delta.values()),
        },
        "results": results,
        "case_selection_factor": chosen,
        "case_selection": cases,
        "case_plot_files": case_files,
        "runtime": {
            "elapsed_seconds": time.time() - start,
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "device": "cpu",
        },
        "limitations": [
            "Validation and test manifests are mirrored; this is exploratory evidence.",
            "One inference seed is not an independent test or multiple training seeds.",
            "The transform does not read ground truth, move candidate centers, add NFE, or retrain.",
        ],
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(json.dumps({
        "status": status, "output": str(output),
        "elapsed_seconds": payload["runtime"]["elapsed_seconds"],
        "selected_factor_for_cases": chosen,
    }, sort_keys=True))


if __name__ == "__main__":
    main()

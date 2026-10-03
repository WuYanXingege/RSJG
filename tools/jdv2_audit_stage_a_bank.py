#!/usr/bin/env python3
"""Authenticate a complete A bank, gate CPU parity, then audit only on CPU."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import sys
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.jdv2_stage_a_bank import (
    BankError, require, certify_bank, load_kernel, write_json, write_npz, sha256,
    MASTERS, SEEDS, KERNEL_SHA, graph_summary, aggregate_scores,
)

def audit_one(scene, kernel, jmm):
    before = {k: scene[k].tobytes() for k in ("Y", "GT", "metric_mask", "future_mask", "edge_index")}
    original = kernel.score(scene, np, jmm)
    names = list(original["metrics"])
    rows = {"common": [], "independent": []}
    tie_counts = {"common": [], "independent": []}
    errors, tie_changes = [], 0
    for master in MASTERS:
        for arm in rows:
            shuffled = kernel.gather(scene, kernel.permutations(scene, master, arm == "common", np), np)
            value = kernel.score(shuffled, np, jmm)
            for metric in ("agent_minADE", "agent_minFDE"):
                np.testing.assert_allclose(value[metric], original[metric], atol=1e-12, rtol=1e-12)
            for name in names:
                if arm == "common" or original["valid_agents"] == 1:
                    if name == "JMM_CRJADE_first_index_diagnostic" and original["tie_count"] > 1:
                        if arm == "common":
                            tie_changes += int(value["metrics"][name] != original["metrics"][name])
                        continue
                    error = abs(value["metrics"][name] - original["metrics"][name])
                    np.testing.assert_allclose(value["metrics"][name], original["metrics"][name],
                                               atol=1e-12, rtol=1e-12)
                    errors.append(error)
            rows[arm].append([value["metrics"][name] for name in names])
            tie_counts[arm].append(value["tie_count"])
    require(all(scene[k].tobytes() == value for k, value in before.items()), "Input bytes mutated")
    arrays = {arm: np.asarray(value) for arm, value in rows.items()}
    arrays.update({arm + "_tie_counts": np.asarray(value) for arm, value in tie_counts.items()})
    arrays["original"] = np.asarray([original["metrics"][name] for name in names])
    arrays["master_seeds"] = np.asarray(MASTERS, dtype=np.int64)
    return names, arrays, {"negative_controls": "PASS", "exact_inverse_bytes": "PASS",
                          "common_or_N1_max_abs_error": max(errors, default=0.),
                          "common_first_index_tie_changes": tie_changes,
                          "original_exact_tie_count": original["tie_count"],
                          "independent_tie_rate": float(np.mean(arrays["independent_tie_counts"] > 1)),
                          "rme_omitted_reason": original["rme_omitted_reason"]}

def summarize(rows):
    # Each row is one scene/inference-seed. Permutations are a null distribution,
    # NOT independent observations or training replicates.
    names = sorted(set.intersection(*(set(r["names"]) for r in rows)))
    result = {}
    for name in names:
        weights = np.asarray([r["valid_agents"] if name in ("minADE", "minFDE") else 1 for r in rows])
        baseline = np.average([r["arrays"]["original"][r["names"].index(name)] for r in rows], weights=weights)
        arms = {}
        for arm in ("common", "independent"):
            null = np.average(np.stack([r["arrays"][arm][:, r["names"].index(name)] for r in rows]),
                              axis=0, weights=weights)
            delta = null - baseline
            arms[arm] = {"replicate_metrics": null.tolist(), "replicate_deltas": delta.tolist(),
                         "mean_shuffled_minus_original": float(delta.mean()),
                         "delta_quantiles_05_50_95": np.quantile(delta, [.05, .5, .95]).tolist(),
                         "original_empirical_cdf_le": float(np.mean(null <= baseline))}
        result[name] = {"original": float(baseline), **arms}
    return {"scene_seed_records": len(rows), "valid_agent_instances": sum(r["valid_agents"] for r in rows),
            "metrics": result, "rme_partial_mask_policy": "Only reported when every row in group has full metric mask"}

def analyze(manifest_path, expected_sha, output):
    manifest, loaded, parity = certify_bank(manifest_path, expected_sha)
    output = Path(output).resolve()
    bank_root = Path(manifest_path).resolve().parent
    require(not output.is_relative_to(bank_root) and not bank_root.is_relative_to(output),
            "Audit output must be separate from bank")
    require(not output.exists(), "Refusing existing audit output")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "INCOMPLETE.json", {"status": "INCOMPLETE", "bank_sha256": expected_sha,
                                           "CPU_parity": parity, "master_seeds": MASTERS})
    kernel = load_kernel()
    _, jmm = kernel.modules()
    state_py, state_np = random.getstate(), np.random.get_state()
    rows, details = [], []
    try:
        for record, scene, raw, baseline in loaded:
            names, arrays, controls = audit_one(scene, kernel, jmm)
            degree, labels = graph_summary(len(scene["agent_ids"]), scene["edge_index"])
            sizes = np.bincount(labels).tolist()
            tags = {"N_valid": str(baseline["valid_agents"]),
                    "E": "E=0" if scene["edge_index"].shape[1] == 0 else "E>0",
                    "max_degree": str(int(degree.max())),
                    "component_sizes": ",".join(map(str, sorted(sizes))),
                    "activity": "singleton" if len(degree) == 1 else
                        ("E=0" if not degree.any() else "all-active" if degree.all() else "mixed")}
            row = {"seed": record["inference_seed"], "source_id": record["source_id"],
                   "window_index": record["window_index"], "valid_agents": baseline["valid_agents"],
                   "names": names, "arrays": arrays, "tags": tags}
            rows.append(row)
            relative = f"details/seed{row['seed']}_window{row['window_index']:06d}.npz"
            write_npz(output / relative, arrays)
            details.append({k: v for k, v in row.items() if k != "arrays"} |
                           {"path": relative, "sha256": sha256(output / relative), "controls": controls,
                            "N": len(degree), "E": scene["edge_index"].shape[1],
                            "degree": degree.tolist(), "component_sizes": sizes,
                            "original_agent_minADE": baseline["agent_minADE"].tolist(),
                            "original_agent_minFDE": baseline["agent_minFDE"].tolist()})
        now = np.random.get_state()
        require(random.getstate() == state_py and state_np[0] == now[0] and
                np.array_equal(state_np[1], now[1]) and state_np[2:] == now[2:], "Audit advanced global RNG")
        sources = sorted(set(r["source_id"] for r in rows))
        strata = {}
        for dimension in rows[0]["tags"]:
            strata[dimension] = {value: summarize([r for r in rows if r["tags"][dimension] == value])
                                 for value in sorted(set(r["tags"][dimension] for r in rows))}
        result = {"status": "COMPLETE", "bank_manifest_sha256": expected_sha,
                  "bank_export_source_commit": manifest["provenance"]["code"]["export_source_commit"],
                  "kernel_sha256": KERNEL_SHA, "CPU_parity": parity,
                  "coverage": {"unique_windows": 139, "windows_per_seed": 139,
                               "scene_seed_records": len(rows), "inference_seeds": SEEDS},
                  "master_seeds": MASTERS, "epsilon_joint": None,
                  "negative_controls": "PASS", "global_rng_unchanged": True,
                  "per_seed": {str(s): summarize([r for r in rows if r["seed"] == s]) for s in SEEDS},
                  "pooled_descriptive": summarize(rows), "strata": strata,
                  "independent_sequence_summaries": {s: summarize([r for r in rows if r["source_id"] == s]) for s in sources},
                  "uncertainty": {"unit": "raw source sequence, fixed before scoring",
                                  "sequence_count": len(sources), "confidence_interval": None,
                                  "reason": "Descriptive per-sequence effects only; no independence claim for overlapping windows or five inference seeds."},
                  "details": details,
                  "claim_boundary": "Current bank relative to specified random pairing null only; not pair-energy causality, held-out generalization, or five training repeats."}
        write_json(output / "RESULTS.json", result)
        return {"status": "COMPLETE", "results": str(output / "RESULTS.json"),
                "sha256": sha256(output / "RESULTS.json"), "CPU_parity": parity}
    except BaseException as exc:
        write_json(output / "FAILED.json", {"status": getattr(exc, "status", "FAILED_CPU_AUDIT"),
                                            "error": str(exc), "completed_scene_seed_records": len(rows)})
        raise

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(analyze(args.manifest, args.manifest_sha256, args.output), indent=2))
    except BankError as exc:
        print(json.dumps({"status": exc.status, "error": str(exc)}), file=sys.stderr)
        return 2
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED_BANK_CERTIFICATION",
                          "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

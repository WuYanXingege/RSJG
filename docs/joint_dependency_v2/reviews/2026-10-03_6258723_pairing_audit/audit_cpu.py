#!/usr/bin/env python3
"""Read-only discovery and CPU pairing kernel/self-tests. Never runs a model.

Commands emit JSON to stdout and never write source, data, or result files.
A real-bank loader/baseline certification is intentionally not claimed complete:
this revision has no certified post-fix A bank to test that integration against.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[4]
SOURCE = "6258723915f3842ee974aea6f8744cd685d5f047"
PRIOR_SOURCE = "1f07e7a5b81374033377c5c057d5c6ab3673b02a"
IDENTITY_SOURCE = "d1a1382ae324b9096ac3a6d8de812ffbbb8d9780"
AUDIT = "outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/canonical_stage_a_route_certification/"
OLD_REVIEW = "docs/joint_dependency_v2/reviews/2026-10-03_1f07e7a/"
EXPECTED = {
    OLD_REVIEW + "REVIEW.md": "046e17ac45676c2a6e9f2510a69416a4e37a92437763792bdfa4b1fb39916201",
    OLD_REVIEW + "checks.json": "fc255a4d5d72aaf79256f617931ffee2d6e5e0a838fdca75979c471029a8bb91",
    AUDIT + "postfix_bf16_results.json": "3df9d0b8592c8fc448594b05cf7dee2ee534de66e8b4bf9b27ecebeda4625bb9",
    AUDIT + "postfix_fp32_results.json": "8796e60061e5c913763e0f3aedbbab94cbb992240eadc99f59337d2d0c1f0aec",
    AUDIT + "postfix_paired_five_seed_results.json": "4c1103cfd2bd923c93368d881e16de83c308cd9693087a25b1f5cc85014c39f6",
}
ATOL = RTOL = 1e-12
NUMERICAL_BASELINE_ATOL = NUMERICAL_BASELINE_RTOL = 1e-6


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(*args, cwd=REPO):
    p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True)
    return p.stdout


def stable_seed(*parts):
    # Stable typed JSON + SHA256, NOT Python hash or global RNG.
    payload = json.dumps(["rsjg-pairing-v1", *parts], ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def discover():
    import re
    workspace = REPO.parent
    exclusions = [
        "!**/.git/**", "!**/data/**", "!**/__pycache__/**",
        "!**/baseline_source_batches/**", "!**/output/sdd/**",
        "!**/outputs/**/sdd/**",
    ]
    command = ["rg", "--files", "--hidden", "--no-ignore", "GDTS", REPO.name]
    for pattern in exclusions:
        command += ["-g", pattern]
    files = sorted(run(*command, cwd=workspace).splitlines())
    manifest_paths = [p for p in files if re.search(r"manifest[^/]*\.json$", p, re.I)]
    array_paths = [p for p in files if re.search(r"\.(npz|npy|h5|hdf5)$", p, re.I)]
    archive_paths = [p for p in files if re.search(r"\.(tar|tar\.gz|tgz|zip|7z|zst)$", p, re.I)]
    text_samples = [p for p in files if re.search(r"/sample_\d+\.txt$", p)]
    groups = {}
    for path in text_samples:
        root = path.split("/trajectories/")[0] if "/trajectories/" in path else str(Path(path).parent.parent)
        groups[root] = groups.get(root, 0) + 1
    binary_paths = [p for p in files if re.search(r"\.(pt|pth|pkl|pickle|bin)$", p, re.I)]
    non_bank_tokens = ("/saved_models/", "/checkpoints/", "/data_batches",
                       "/source_batches/", "/cache/eth_full_stage_a/",
                       "/cache/univ_full_stage_a/", "/jmm_official/cache/",
                       "/prototype_bank/", "/set_selector", "/motion_modes/", "/training/")
    other_binary = [p for p in binary_paths if not any(t in p for t in non_bank_tokens)]
    external = []
    for root in [*sorted(Path("/tmp").glob("rsjg*")), *sorted(Path("/tmp").glob("jdv2*")),
                 Path("/home/ee615/Downloads"), Path("/home/ee615/Desktop"),
                 Path("/home/ee615/Documents")]:
        item = {"root": str(root), "exists": root.exists(), "method": "related filenames only"}
        if root.is_dir():
            proc = subprocess.run(["rg", "--files", "--hidden", "--max-depth", "5", str(root)],
                                  capture_output=True, text=True)
            if proc.returncode not in (0, 1):
                item["scan_error"] = proc.stderr[:500]
            item["related_candidates"] = [
                p for p in proc.stdout.splitlines()
                if re.search(r"(rsjg|jdv2|trajectory|pairing|bank)", p, re.I)
                and re.search(r"\.(json|npz|npy|pt|pth|pkl|tar|gz|zip|h5|hdf5)$", p, re.I)]
        external.append(item)
    evidence = []
    parsed = {}
    for relative, expected in EXPECTED.items():
        path = REPO / relative
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"Historical fingerprint mismatch: {relative}")
        row = {"path": relative, "sha256": actual, "expected_sha256_match": True}
        if path.suffix == ".json":
            obj = json.loads(path.read_text())
            parsed[relative] = obj
            row["top_level_keys"] = sorted(obj)
            if relative.startswith(AUDIT):
                assert obj["provenance"]["source_commit"] == IDENTITY_SOURCE
                row.update(status=obj["status"], generating_source_commit=IDENTITY_SOURCE)
                if "per_seed" in obj:
                    row["inference_seeds"] = [s["seed"] for s in obj["per_seed"]]
                    row["counts_by_seed"] = [{"seed": s["seed"], **s["counts"]} for s in obj["per_seed"]]
                row["full_prediction_array_at_top_level"] = False
        evidence.append(row)
    references = [
        "docs/joint_dependency_v2/JDV2_CANONICAL_STAGE_A_IDENTITY_FIX_VALIDATION.md",
        "src/metrics.py", "src/jmm_protocol.py", "src/models/model.py",
        "tools/jdv2_postfix_paired_five_seed_evaluation.py",
        "tools/jdv2_stage_b_v1_failure_mechanism_audit.py",
        "src/models/joint_dependency_v2/joint_sampler.py",
        "src/models/joint_dependency_v2/dynamic_relation.py",
        "configs/joint_dependency_v2/jdv2_stage_a_frozen_eth.yaml",
    ]
    evidence += [{"path": p, "sha256": sha256(REPO / p)} for p in references]
    inspected_manifests = []
    for relative in manifest_paths:
        if ("/jmm_official/stage_b_v2a_epoch005_seed2035" in relative
                or relative.endswith("/cache/eth_full_stage_a/manifest.json")
                or relative.endswith("/stage_a_freeze/manifest.json")):
            path = workspace / relative
            obj = json.loads(path.read_text())
            row = {"path": relative, "sha256": sha256(path),
                   "source_commit": obj.get("source_commit", obj.get("freeze_source_commit")),
                   "status": obj.get("status"), "checkpoint_sha256": obj.get("checkpoint_sha256", obj.get("checkpoint", {}).get("sha256") if isinstance(obj.get("checkpoint"), dict) else None)}
            if "/jmm_official/" in relative:
                row.update(classification="REJECTED_OLD_D_WRONG_PROTOCOL",
                           correction_route="D; established by prior report/config, not an explicit manifest flag",
                           completed_scenes=obj.get("completed_scenes"),
                           seed=obj.get("seed"), protocol=obj.get("protocol"))
            elif "/cache/" in relative:
                row.update(classification="REJECTED_GOAL_CACHE_NOT_FULL_PREDICTIONS",
                           schema_version=obj.get("schema_version"))
            else:
                row["classification"] = "CHECKPOINT_FREEZE_METADATA_ONLY"
            inspected_manifests.append(row)
    return {
        "status": "BLOCKED_MISSING_TRAJECTORY_BANK",
        "searched_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_commit": run("git", "rev-parse", "HEAD").strip(),
        "branch": run("git", "branch", "--show-current").strip(),
        "tracked_worktree_status": run("git", "status", "--porcelain", "--untracked-files=no"),
        "changes_since_6258723": run("git", "diff", "--stat", SOURCE, "HEAD"),
        "model_source_changes_since_1f07e7a": run("git", "diff", "--stat", PRIOR_SOURCE, "HEAD", "--", "src", "configs", "tools"),
        "discovery": {
            "workspace": str(workspace), "command": command, "excluded_globs": exclusions,
            "enumerated_file_count": len(files), "manifest_paths": manifest_paths,
            "array_artifact_paths": array_paths, "archive_paths": archive_paths,
            "binary_file_count": len(binary_paths), "other_binary_candidates": other_binary,
            "full_text_sample_groups": groups, "inspected_manifests": inspected_manifests,
            "external_scans": external,
            "accepted_banks": [], "available_inference_seeds": [],
            "missing_inference_seeds": list(range(2035, 2040)),
            "limitations": [
                "No unspecified external/cloud/archive location was supplied.",
                "No model/checkpoint/pickle deserialization, archive extraction or SDD cache access.",
                "Absence is limited to these searched roots; not proof of global nonexistence.",
                "Filename-filtered binaries cannot establish provenance; no uncertified artifact is accepted."
            ],
        },
        "evidence": evidence,
        "prior_checks": {"read_and_hashed": True, "old_63_summaries_2940_gauges_60_scenes_rerun": False},
    }


def modules():
    import numpy as np
    spec = importlib.util.spec_from_file_location("pairing_jmm", REPO / "src/jmm_protocol.py")
    jmm = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = jmm
    spec.loader.exec_module(jmm)
    return np, jmm


def validate_scene(scene, np):
    y, gt = scene["Y"], scene["GT"]
    if y.ndim != 4 or y.shape[-1] != 2 or min(y.shape[:3]) < 1:
        raise ValueError("Y requires nonempty [N,P,T,2]")
    n, p, t, _ = y.shape
    if gt.shape != (n, t, 2) or y.dtype.kind != "f" or gt.dtype.kind != "f":
        raise ValueError("GT/float dtype contract")
    if not np.isfinite(y).all() or not np.isfinite(gt).all():
        raise ValueError("Nonfinite trajectory/GT")
    mask, future = scene["metric_mask"], scene["future_mask"]
    if mask.shape != (n,) or mask.dtype != np.bool_ or not mask.any():
        raise ValueError("Nonempty boolean metric_mask required")
    if future.shape != (n, t) or future.dtype != np.bool_ or not future[mask].all():
        raise ValueError("Scored agents must have complete futures")
    ids = scene["agent_ids"]
    if len(ids) != n or len(set(ids)) != n or not all(isinstance(i, str) for i in ids):
        raise ValueError("Unique stable string agent IDs required")
    edge = scene["edge_index"]
    if edge.ndim != 2 or edge.shape[0] != 2 or edge.dtype.kind not in "iu":
        raise ValueError("Canonical integer edge_index[2,E] required")
    if edge.size and (edge.min() < 0 or edge.max() >= n or not (edge[0] < edge[1]).all()):
        raise ValueError("Invalid/cross-scene/noncanonical edge")
    if len({tuple(e) for e in edge.T}) != edge.shape[1]:
        raise ValueError("Duplicate canonical edge")
    if scene["units"] != "world_metres":
        raise ValueError("This metric adapter only accepts world_metres")
    if (scene["sample_valid"].shape != (n, p) or scene["sample_valid"].dtype != np.bool_
            or not scene["sample_valid"].all()):
        raise ValueError("Only complete valid sample banks supported")
    if (scene["sample_weights"].shape != (n, p) or scene["sample_weights"].dtype.kind != "f"
            or not np.all(scene["sample_weights"] == 1.0 / p)):
        raise ValueError("Only equal-weight banks supported; never reweight input")
    for name, value in scene.get("sample_aux", {}).items():
        if value.shape[:2] != (n, p):
            raise ValueError(f"Sample auxiliary axis mismatch: {name}")


def score(scene, np, jmm):
    validate_scene(scene, np)
    y, gt, mask = scene["Y"], scene["GT"], scene["metric_mask"]
    # Float64 diagnostic adapter, canonical definitions; real artifact parity is a separate gate.
    yv, gtv = y[mask].astype(np.float64), gt[mask].astype(np.float64)
    distances = np.linalg.norm(yv - gtv[:, None], axis=-1)
    ade, fde = distances.mean(-1), distances[:, :, -1]
    jade_by_slot, jfde_by_slot = ade.mean(0), fde.mean(0)
    cr = jmm.collision_rates_per_sample(yv.transpose(1, 2, 0, 3))
    ties = np.flatnonzero(jade_by_slot == jade_by_slot.min())  # exact ties, no epsilon
    values = {
        "minADE": float(ade.min(1).mean()), "minFDE": float(fde.min(1).mean()),
        "JADE": float(jade_by_slot.min()), "JFDE": float(jfde_by_slot.min()),
        "JMM_CRmean_diagnostic": float(cr.mean()),
        "JMM_CRJADE_first_index_diagnostic": float(cr[int(np.argmin(jade_by_slot))]),
        "JMM_CRJADE_exact_ties_mean": float(cr[ties].mean()),
        "JMM_CRJADE_exact_ties_min": float(cr[ties].min()),
        "JMM_CRJADE_exact_ties_max": float(cr[ties].max()),
    }
    if mask.all():
        src, dst = scene["edge_index"]
        if src.size:
            rel = y.astype(np.float64)[dst] - y.astype(np.float64)[src]
            target = gt.astype(np.float64)[dst] - gt.astype(np.float64)[src]
            values["RME_legacy_full_mask"] = float(np.linalg.norm(rel - target[:, None], axis=-1).mean((0, 2)).min())
        else:
            values["RME_legacy_full_mask"] = 0.0
    # Do not silently "fix" production's mask-ignoring RME for partial masks.
    return {"metrics": values, "agent_minADE": ade.min(1), "agent_minFDE": fde.min(1),
            "tie_count": int(ties.size), "valid_agents": int(mask.sum()),
            "rme_omitted_reason": None if mask.all() else "partial_mask_legacy_semantics_not_certified"}


def permutations(scene, master_seed, common, np):
    n, p = scene["Y"].shape[:2]
    identity = [scene["source_id"], scene["window_id"], scene["scene_id"], scene["inference_seed"]]
    if common:
        seed = stable_seed(master_seed, *identity, "common")
        row = np.random.Generator(np.random.PCG64(seed)).permutation(p)
        return np.broadcast_to(row, (n, p)).copy()
    return np.stack([np.random.Generator(np.random.PCG64(
        stable_seed(master_seed, *identity, "independent", agent))).permutation(p)
        for agent in scene["agent_ids"]])


def gather(scene, pi, np):
    n, p = scene["Y"].shape[:2]
    if pi.shape != (n, p) or pi.dtype.kind not in "iu" or not np.all(np.sort(pi, axis=1) == np.arange(p)):
        raise ValueError("Permutation must be a per-agent bijection")
    rows = np.arange(n)[:, None]
    out = dict(scene)
    for key in ("Y", "sample_valid", "sample_weights"):
        out[key] = scene[key][rows, pi]
        restored = out[key][rows, np.argsort(pi, axis=1)]
        if restored.dtype != scene[key].dtype or restored.tobytes() != scene[key].tobytes():
            raise AssertionError(f"Inverse byte recovery failed: {key}")
    out["sample_aux"] = {}
    for key, value in scene.get("sample_aux", {}).items():
        changed = value[rows, pi]
        if changed[rows, np.argsort(pi, axis=1)].tobytes() != value.tobytes():
            raise AssertionError("Auxiliary inverse recovery failed")
        out["sample_aux"][key] = changed
    return out


def audit_scene(scene, seeds, np, jmm):
    validate_scene(scene, np)
    before = {k: scene[k].tobytes() for k in ("Y", "GT", "metric_mask", "future_mask", "edge_index")}
    original = score(scene, np, jmm)
    rows, tie_changes, independent_ties = [], 0, []
    max_control_error = 0.0
    for seed in seeds:
        common = score(gather(scene, permutations(scene, seed, True, np), np), np, jmm)
        independent = score(gather(scene, permutations(scene, seed, False, np), np), np, jmm)
        for changed in (common, independent):
            for name in ("agent_minADE", "agent_minFDE"):
                np.testing.assert_allclose(changed[name], original[name], atol=ATOL, rtol=RTOL)
        for name, value in original["metrics"].items():
            changed = common["metrics"][name]
            if name == "JMM_CRJADE_first_index_diagnostic" and original["tie_count"] > 1:
                tie_changes += int(changed != value)
                continue
            np.testing.assert_allclose(changed, value, atol=ATOL, rtol=RTOL)
            max_control_error = max(max_control_error, abs(changed - value))
            if original["valid_agents"] == 1:
                np.testing.assert_allclose(independent["metrics"][name], value, atol=ATOL, rtol=RTOL)
        rows.append(independent["metrics"])
        independent_ties.append(independent["tie_count"] > 1)
    for key, value in before.items():
        assert scene[key].tobytes() == value, f"Input mutated: {key}"
    delta = {}
    for name, value in original["metrics"].items():
        null = np.asarray([row[name] for row in rows])
        differences = null - value
        delta[name] = {
            "mean_shuffled_minus_original": float(differences.mean()),
            "delta_quantiles_05_50_95": np.quantile(differences, [.05, .5, .95]).tolist(),
            "original_empirical_cdf_le": float(np.mean(null <= value)),
            "interpretation": "descriptive permutation distribution, not a significance p value",
        }
    return {"original": original["metrics"], "original_exact_jade_tie_count": original["tie_count"],
            "original_has_exact_jade_tie": original["tie_count"] > 1,
            "independent_tie_rate": float(np.mean(independent_ties)),
            "controls_passed": True, "input_bytes_unchanged": True,
            "common_max_abs_error": max_control_error, "official_common_tie_changes": tie_changes,
            "replicates": len(seeds), "descriptive_deltas": delta}


def self_test(manifest):
    for record in manifest["evidence_files"]:
        assert sha256(REPO / record["path"]) == record["sha256"], record["path"]
    np, jmm = modules()
    seeds = manifest["permutations"]["master_seeds"]
    assert len(seeds) == len(set(seeds)) == 100
    local = np.random.Generator(np.random.PCG64(9026258723))
    def fixture(n, p, t, name, edges):
        return {
            "Y": local.normal(size=(n, p, t, 2)), "GT": local.normal(size=(n, t, 2)),
            "metric_mask": np.ones(n, dtype=bool), "future_mask": np.ones((n, t), dtype=bool),
            "edge_index": np.asarray(edges, dtype=np.int64).reshape(2, -1),
            "agent_ids": [f"agent-{i}" for i in range(n)],
            "source_id": "SYNTHETIC_ONLY", "window_id": name, "scene_id": "scene-0", "inference_seed": 2035,
            "units": "world_metres", "sample_valid": np.ones((n, p), dtype=bool),
            "sample_weights": np.full((n, p), 1.0 / p),
            "sample_aux": {"ids": np.broadcast_to(np.arange(p), (n, p)).copy()},
        }
    cases = [
        fixture(1, 7, 5, "singleton", [[], []]),
        fixture(3, 7, 5, "multiagent_E0", [[], []]),
        fixture(4, 9, 3, "mixed", [[0, 1], [1, 2]]),
        fixture(3, 5, 4, "all_active", [[0, 0, 1], [1, 2, 2]]),
        fixture(3, 7, 6, "partial_agent_mask", [[0, 1], [1, 2]]),
        fixture(2, 20, 12, "exact_CRJADE_tie", [[0], [1]]),
        fixture(2, 2, 3, "non_tie_joint_change", [[], []]),
    ]
    cases[4]["metric_mask"][-1] = False
    cases[4]["future_mask"][-1, -2:] = False
    tie = cases[5]
    tie["GT"].fill(0)
    tie["GT"][0, :, 0], tie["GT"][1, :, 0] = -1, 1
    tie["Y"].fill(0)
    tie["Y"][0, 1:, :, 0], tie["Y"][1, 1:, :, 0] = -2, 2
    last = cases[6]
    last["GT"].fill(0); last["Y"].fill(0); last["Y"][:, 1, :, 0] = 10
    py_state, np_state = random.getstate(), np.random.get_state()
    results = {s["window_id"]: audit_scene(s, seeds, np, jmm) for s in cases}
    assert random.getstate() == py_state
    now = np.random.get_state()
    assert np_state[0] == now[0] and np.array_equal(np_state[1], now[1]) and np_state[2:] == now[2:]
    swap = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    assert score(gather(last, swap, np), np, jmm)["metrics"]["JADE"] == 5.0
    assert results["exact_CRJADE_tie"]["official_common_tie_changes"] > 0
    masked = cases[4]
    perturbed = dict(masked, Y=masked["Y"].copy(), GT=masked["GT"].copy())
    perturbed["Y"][-1] += 1000; perturbed["GT"][-1] -= 1000
    assert score(masked, np, jmm)["metrics"] == score(perturbed, np, jmm)["metrics"]
    # Stable agent identity, rather than row number, determines independent streams.
    order = np.asarray([2, 0, 1])
    base = cases[3]
    perm0 = permutations(base, seeds[0], False, np)
    reordered = dict(base, Y=base["Y"][order], agent_ids=[base["agent_ids"][i] for i in order])
    assert np.array_equal(permutations(reordered, seeds[0], False, np), perm0[order])
    # Independent new processes with different randomized Python hashes agree.
    probe = []
    for value in ("1", "999"):
        env = dict(os.environ, PYTHONHASHSEED=value)
        probe.append(subprocess.check_output([sys.executable, "-B", __file__, "seed-probe"],
                                            env=env, text=True).strip())
    assert probe[0] == probe[1] == str(stable_seed(625872300, "s", "w", 2035, "independent", "a"))
    # Read-only reuse of actual CPU metric implementations; no model/CUDA imports/calls.
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    spec = importlib.util.spec_from_file_location("pairing_metrics", REPO / "src/metrics.py")
    metric_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric_module)
    max_reference_error = 0.0
    for scene in cases:
        score0 = score(scene, np, jmm)
        pred = torch.from_numpy(scene["Y"].transpose(1, 2, 0, 3).copy())
        gt = torch.from_numpy(scene["GT"].transpose(1, 0, 2).copy())
        mask = torch.from_numpy(scene["metric_mask"])
        reference = {
            "minADE": np.mean(metric_module.minADE_at_K(pred, gt, mask, obs_length=0)),
            "minFDE": np.mean(metric_module.minFDE_at_K(pred, gt, mask, obs_length=0)),
            "JADE": metric_module.JADE(pred, gt, mask, obs_length=0)[0],
            "JFDE": metric_module.JFDE(pred, gt, mask, obs_length=0)[0],
        }
        for name, value in reference.items():
            np.testing.assert_allclose(score0["metrics"][name], value, atol=ATOL, rtol=RTOL)
            max_reference_error = max(max_reference_error, abs(value - score0["metrics"][name]))
    # Fail-closed input-contract tests (not production patches).
    negative_passed = []
    def reject(name, scene):
        try:
            validate_scene(scene, np)
        except ValueError:
            negative_passed.append(name)
            return
        raise AssertionError(f"Invalid fixture was accepted: {name}")
    s = cases[3]
    reject("duplicate_agent_ID", dict(s, agent_ids=["a"] * 3))
    reject("empty_valid_agents", dict(s, metric_mask=np.zeros(3, dtype=bool)))
    bad = s["Y"].copy(); bad[0, 0, 0, 0] = np.nan
    reject("nonfinite_trajectory", dict(s, Y=bad))
    bad = s["future_mask"].copy(); bad[0, -1] = False
    reject("incomplete_scored_future", dict(s, future_mask=bad))
    bad = s["sample_valid"].copy(); bad[0, 0] = False
    reject("missing_sample", dict(s, sample_valid=bad))
    bad = s["sample_weights"].copy(); bad[0, 0] = .9
    reject("nonuniform_weights", dict(s, sample_weights=bad))
    reject("wrong_units", dict(s, units="pixel"))
    reject("cross_scene_edge", dict(s, edge_index=np.asarray([[0], [9]], dtype=np.int64)))
    reject("duplicate_edge", dict(s, edge_index=np.asarray([[0, 0], [1, 1]], dtype=np.int64)))
    reject("aux_axis_mismatch", dict(s, sample_aux={"bad": np.zeros((3, 2))}))
    try:
        gather(s, np.zeros(s["Y"].shape[:2], dtype=np.int64), np)
    except ValueError:
        negative_passed.append("nonbijective_permutation")
    else:
        raise AssertionError("Nonbijection accepted")
    return {
        "scope": "NEW_KERNEL_SYNTHETIC_CPU_ONLY",
        "model_execution": False, "cuda_calls": False, "real_A_scenes_audited": 0,
        "status": "PASS", "fixture_count": len(cases), "replicates_per_fixture": len(seeds),
        "fixtures": results, "negative_contract_tests_passed": negative_passed,
        "canonical_CPU_metric_max_abs_error": max_reference_error,
        "python_numpy_global_rng_unchanged": True,
        "agent_identity_stream_transport": True, "python_hash_seed_independence": True,
        "atol": ATOL, "rtol": RTOL,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__},
        "resource_environment": {k: os.environ.get(k) for k in (
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "limitations": ["No certified real bank: real-bank CPU baseline parity NOT_RUN.",
                       "No real-bank file loader/provenance gate has been certified.",
                       "RME for partial metric masks omitted; no production metric change.",
                       "No effect-size/significance/generalization claim from synthetic results."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["discover", "self-test", "seed-probe"])
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.command == "seed-probe":
        print(stable_seed(625872300, "s", "w", 2035, "independent", "a"))
        return
    if args.command == "discover":
        result = discover()
    else:
        if args.manifest is None:
            parser.error("self-test requires pre-registered --manifest")
        result = self_test(json.loads(args.manifest.read_text()))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

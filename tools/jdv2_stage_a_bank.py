#!/usr/bin/env python3
"""Shared, fail-closed Stage-A bank I/O. No model or CUDA import."""
from __future__ import annotations
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[1]
PRIOR = "5a0ae411f401fe77b9ec4f114d5c4bfa45109a4e"
MODEL_SOURCE = "d1a1382ae324b9096ac3a6d8de812ffbbb8d9780"
CACHE_PIN = "4b75110a571e6fd200938ff951a3a218ca4eb1e8"
CONFIG = "configs/joint_dependency_v2/jdv2_stage_a_frozen_eth.yaml"
CONFIG_SHA = "bf5d4a692b0a9e841de8523f2780013147ec3e9bcb78877cc3c352571bc63e62"
CHECKPOINT = "outputs/joint_dependency_v2/eth/joint_dependency_v2/runs/jdv2_stage_a_no_z_full_seed2035/saved_models/best_model.pt"
CHECKPOINT_SHA = "699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb"
CACHE = "outputs/joint_dependency_v2/cache/eth_full_stage_a"
CACHE_SHA = "eadae55391b45224a3d7920e6128764db9a56298ffe70d723030ec00193786dd"
SOURCE_BATCHES = "outputs/joint_dependency_v2/cache/source_batches/eth5/eth/data_batches_jdv2_v2"
KERNEL = "docs/joint_dependency_v2/reviews/2026-10-03_6258723_pairing_audit/audit_cpu.py"
KERNEL_SHA = "80aeae22f7d0a488779e5920a9093519ce05329d82e9361b72ae6f080e4d3225"
SCRIPT_PATHS = ["tools/jdv2_stage_a_bank.py", "tools/jdv2_export_stage_a_bank.py",
                "tools/jdv2_audit_stage_a_bank.py", "tools/jdv2_stage_a_bank_cpu_checks.py"]
SEEDS = list(range(2035, 2040))
MASTERS = list(range(625872300, 625872400))
METRICS = {"minADE": "minADE@K", "minFDE": "minFDE@K",
           "JADE": "JADE", "JFDE": "JFDE",
           "RME_legacy_full_mask": "Relative_Motion_Error"}
PROTOCOL = {"route": "A", "split": "ETH validation", "windows_per_seed": 139,
            "inference_seeds": SEEDS, "P": 20, "T": 12, "K": 21,
            "refinement_steps": 2, "precision": "CUDA/BF16",
            "units": "world_metres", "dt": 0.4,
            "coordinate_transform": "velocities_to_predictions once; down_factor=8; per-sample scene.make_world_coord_torch; future [8:20]; transpose N,P,T,2"}
BASE_KEYS = {"Y", "GT", "metric_mask", "future_mask", "sample_valid", "sample_weights", "edge_index"}
AUX_KEYS = ("sample_indices", "selected_candidate_ids", "joint_goals_world", "joint_goals_map")

class BankError(RuntimeError):
    def __init__(self, message, status="BLOCKED_BANK_CERTIFICATION"):
        super().__init__(message)
        self.status = status

def require(condition, message, status="BLOCKED_BANK_CERTIFICATION"):
    if not condition:
        raise BankError(message, status)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()

def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + "\n").encode()

def run(*args):
    return subprocess.check_output(args, cwd=REPO, text=True).strip()

def git_bytes(commit, path):
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=REPO)

def code_provenance(require_committed=True):
    head = run("git", "rev-parse", "HEAD")
    require(not run("git", "diff", MODEL_SOURCE, "--", "src", "configs"),
            "Model/config content differs from identity source", "BLOCKED_EXPORT_INPUT")
    # Tools reused by exporter must also retain their certified content.
    helpers = ["tools/audit_jdv2_stage_a.py",
               "tools/jdv2_stage_b_v1_failure_mechanism_audit.py",
               "tools/jdv2_stage_b_identity_contract_audit.py"]
    for path in helpers:
        require((REPO / path).read_bytes() == git_bytes(MODEL_SOURCE, path),
                f"Historical helper changed: {path}", "BLOCKED_EXPORT_INPUT")
    require(sha256(REPO / KERNEL) == KERNEL_SHA, "Archived kernel changed")
    scripts = {p: sha256(REPO / p) for p in SCRIPT_PATHS}
    if require_committed:
        for path, digest in scripts.items():
            require(digest_bytes(git_bytes(head, path)) == digest,
                    f"Exporter not committed/clean: {path}", "BLOCKED_EXPORT_INPUT")
    source_files = run("git", "ls-files", "src", "configs").splitlines() + helpers
    hashes = {p: sha256(REPO / p) for p in source_files}
    return {"export_source_commit": head, "model_source_commit": MODEL_SOURCE,
            "model_config_source_tree_sha256": digest_bytes(json_bytes(hashes)),
            "source_files": hashes, "scripts": scripts, "kernel_sha256": KERNEL_SHA,
            "relevant_worktree_clean": True, "scripts_committed": require_committed}

def atomic_new(path, payload):
    """Publish complete bytes without replacement, even under a creation race."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"Refusing overwrite: {path}")
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".incomplete", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # atomic no-clobber publication on same filesystem
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)

def write_json(path, value):
    atomic_new(path, json_bytes(value))

def write_npz(path, arrays):
    require(all(v.dtype.kind in "biuf" for v in arrays.values()),
            "Only numeric/bool NPZ fields; object/string arrays forbidden")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    atomic_new(path, buffer.getvalue())

def load_kernel():
    require(sha256(REPO / KERNEL) == KERNEL_SHA, "Archived kernel hash mismatch")
    spec = importlib.util.spec_from_file_location("stage_a_original_pairing_kernel", REPO / KERNEL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def graph_summary(n, edges):
    degree = np.bincount(edges.reshape(-1), minlength=n).astype(np.int64)
    labels = np.arange(n, dtype=np.int64)
    for a, b in edges.T:
        old, new = max(labels[a], labels[b]), min(labels[a], labels[b])
        labels[labels == old] = new
    _, labels = np.unique(labels, return_inverse=True)
    return degree, labels.astype(np.int64)

def array_contract(arrays):
    return {k: {"shape": list(v.shape), "dtype": str(v.dtype),
                "sha256": digest_bytes(v.tobytes())} for k, v in sorted(arrays.items())}

def safe_child(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts,
            f"Unsafe artifact path: {name}")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), f"Artifact escapes bank: {name}")
    return path

def load_record(root, record, kernel):
    path = safe_child(root, record["path"])
    require(sha256(path) == record["sha256"], f"NPZ hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), "Duplicate NPZ keys")
        arrays = {name: archive[name] for name in archive.files}
    require(all(v.dtype.kind in "biuf" for v in arrays.values()), "Non-numeric NPZ")
    require(array_contract(arrays) == record["arrays"], "Array shape/dtype/byte contract mismatch")
    require(all(np.isfinite(v).all() for v in arrays.values()), "Nonfinite bank auxiliary")
    require(BASE_KEYS | set(AUX_KEYS) | {"degree", "component_index", "frame_ids"} <= arrays.keys(),
            "Missing required arrays")
    scene = {k: arrays[k] for k in BASE_KEYS}
    for key in ("source_id", "window_id", "scene_id", "inference_seed", "agent_ids", "units"):
        scene[key] = record[key]
    scene["sample_aux"] = {k: arrays[k] for k in AUX_KEYS}
    kernel.validate_scene(scene, np)
    n, p, t, _ = scene["Y"].shape
    require((p, t) == (20, 12), "Bank budget is not P20/T12")
    require(record["dt"] == .4 and record["units"] == "world_metres", "Units/dt mismatch")
    require(record["rng_capture_serialization_unchanged"] is True, "Capture changed RNG")
    require(record["graph_provenance"] == "same_forward_history_only_frozen_cache", "Graph source mismatch")
    require(arrays["frame_ids"].shape == (20, n) and arrays["frame_ids"].dtype.kind in "iu",
            "frame_ids must be integer [20,N]")
    require(np.all(arrays["frame_ids"] == arrays["frame_ids"][:, :1]) and
            np.all(np.diff(arrays["frame_ids"][:, 0]) == 6), "Noncanonical ETH frame sequence")
    degree, components = graph_summary(n, scene["edge_index"])
    require(np.array_equal(degree, arrays["degree"]) and
            np.array_equal(components, arrays["component_index"]), "Graph component metadata mismatch")
    require(np.array_equal(arrays["sample_indices"], np.broadcast_to(np.arange(p), (n, p))),
            "Missing/duplicate original sample indices")
    candidates = arrays["selected_candidate_ids"]
    require(candidates.dtype.kind in "iu" and candidates.shape == (n, p) and
            (candidates >= 0).all() and (candidates < 21).all() and
            all(len(set(row)) == p for row in candidates), "Invalid 20-of-21 candidate coverage")
    for key in ("joint_goals_world", "joint_goals_map"):
        require(arrays[key].shape == (n, p, 2) and np.isfinite(arrays[key]).all(), f"Invalid {key}")
    required_context = {"goal_candidates_world_K", "goal_candidates_map_K", "candidate_log_prior_K",
                        "trunk_goal_world", "trunk_candidate_id", "homography_H", "edge_weight",
                        "edge_relation_embedding_original_world"}
    require(required_context <= arrays.keys(), "Missing K/trunk/graph context")
    for key in ("goal_candidates_world_K", "goal_candidates_map_K"):
        require(arrays[key].shape == (n, 21, 2), f"Invalid K axis: {key}")
    require(arrays["candidate_log_prior_K"].shape == (n, 21), "Invalid K prior")
    selected_world = arrays["goal_candidates_world_K"][np.arange(n)[:, None], candidates]
    require(np.array_equal(selected_world, arrays["joint_goals_world"]), "Selected world goals do not match K IDs")
    require(arrays["trunk_goal_world"].shape == (n, 1, 2) and
            arrays["trunk_candidate_id"].shape == (n, 1), "Trunk mixed into P")
    require(arrays["edge_weight"].shape == (scene["edge_index"].shape[1],) and
            arrays["edge_relation_embedding_original_world"].shape[:2] ==
            (scene["edge_index"].shape[1], p), "Invalid graph/context axes")
    require(arrays["homography_H"].shape == (3, 3), "Invalid homography")
    for key in ("Y", "GT"):
        declaration = record["storage"][key]
        require(declaration["storage_dtype"] == str(arrays[key].dtype), "Storage dtype mismatch")
        original = declaration["original_dtype"]
        expected = "float32" if original == "torch.bfloat16" else original.removeprefix("torch.")
        require(expected == str(arrays[key].dtype), "Lossy/undeclared conversion")
    return scene, arrays

def compare(actual, expected, location, errors):
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
            f"Invalid evaluator baseline: {location}", "FAILED_CPU_PARITY")
    difference = np.abs(a - b)
    if not np.allclose(a, b, atol=1e-6, rtol=1e-6):
        first = np.argwhere(difference > 1e-6 + 1e-6 * np.abs(b))[0].tolist()
        raise BankError(f"First parity mismatch {location}, index={first}, "
                        f"CPU={a.tolist()}, evaluator={b.tolist()}", "FAILED_CPU_PARITY")
    errors.append(float(difference.max()) if difference.size else 0.)

def aggregate_scores(scores):
    return {key: float(np.average([s["metrics"][key] for s in scores],
                    weights=[s["valid_agents"] if key in ("minADE", "minFDE") else 1
                             for s in scores]))
            for key in set.intersection(*(set(s["metrics"]) for s in scores))}

def authenticate_code(code):
    """Validate recorded source against immutable Git objects, not current HEAD."""
    require(code["scripts_committed"] is True and code["relevant_worktree_clean"] is True and
            code["model_source_commit"] == MODEL_SOURCE, "Uncommitted or wrong model source")
    commit = code["export_source_commit"]
    require(len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), "Invalid commit")
    require(not run("git", "diff", MODEL_SOURCE, commit, "--", "src", "configs"), "Changed model/config")
    require(set(code["scripts"]) == set(SCRIPT_PATHS), "Missing script provenance")
    expected_files = set(run("git", "ls-tree", "-r", "--name-only", commit, "src", "configs").splitlines())
    expected_files.update(("tools/audit_jdv2_stage_a.py",
                           "tools/jdv2_stage_b_v1_failure_mechanism_audit.py",
                           "tools/jdv2_stage_b_identity_contract_audit.py"))
    require(set(code["source_files"]) == expected_files, "Incomplete source fingerprint set")
    for path in expected_files:
        require(digest_bytes(git_bytes(MODEL_SOURCE, path)) == code["source_files"][path],
                f"Source/helper not identity-version equivalent: {path}")
    for path, digest in {**code["source_files"], **code["scripts"]}.items():
        require(digest_bytes(git_bytes(commit, path)) == digest, f"Commit fingerprint mismatch: {path}")
    require(digest_bytes(json_bytes(code["source_files"])) == code["model_config_source_tree_sha256"],
            "Source tree digest mismatch")
    require(code["kernel_sha256"] == KERNEL_SHA, "Wrong kernel provenance")

def validate_seed_rows(rows, seed, filenames, complete=False):
    require([r["window_index"] for r in rows] == list(range(len(rows))),
            "Interrupted seed is not a contiguous prefix")
    require(len(rows) <= 139 and (not complete or len(rows) == 139), "Incomplete recovered seed")
    for index, row in enumerate(rows):
        require(row["inference_seed"] == seed and row["window_id"] == f"valid/{filenames[index]}" and
                row["path"] == f"seed{seed}/window{index:06d}.npz", "Recovered record identity mismatch")

def authenticate_recovery(manifest, root):
    """Per-seed origin and counts must not disguise reruns as one uninterrupted pass."""
    recovery = manifest["provenance"].get("seed_boundary_recovery")
    if recovery is None:
        require(manifest["forwards_completed"] == 695, "Wrong fresh forward count")
        return
    source_path = root / "RECOVERY_SOURCE_INCOMPLETE.json"
    require(sha256(source_path) == recovery["source_marker_sha256"], "Recovery source marker changed")
    source = json.loads(source_path.read_text())
    require(source["status"] == "INCOMPLETE" and source["protocol"] == PROTOCOL,
            "Wrong recovery source protocol")
    authenticate_code(source["provenance"]["code"])
    old_commit = source["provenance"]["code"]["export_source_commit"]
    require(old_commit == "e4c36036be986d04f9a14a43ff73582ba85ac564",
            "Recovery proof supports only the explicitly reviewed interrupted exporter")
    require(sha256(root / "RECOVERY_SOURCE_RESOLVED_ARGS.json") == recovery["source_resolved_args_sha256"],
            "Recovery resolved arguments changed")
    for key in ("checkpoint_sha256", "config_sha256", "cache_manifest_sha256", "cache_pin",
                "source_batch_manifest_sha256", "physical_validation_filenames",
                "dataset_hash", "split_hash", "cache_manifest_hash"):
        require(source["provenance"][key] == manifest["provenance"][key], "Recovery input mismatch: " + key)
    old_args = json.loads((root / "RECOVERY_SOURCE_RESOLVED_ARGS.json").read_text())
    new_args_path = root / "RESOLVED_ARGS.json"
    require(sha256(new_args_path) == manifest["provenance"]["resolved_args_sha256"],
            "Current resolved arguments changed")
    new_args = json.loads(new_args_path.read_text())
    require({k: v for k, v in old_args.items() if k not in ("model_dir", "save_dir")} ==
            {k: v for k, v in new_args.items() if k not in ("model_dir", "save_dir")},
            "Recovered and fresh runtime arguments differ beyond output directories")
    reused = recovery["reused_seeds"]
    require(reused and len(set(reused)) == len(reused) and set(reused) < set(SEEDS),
            "Invalid recovered seed set")
    require(recovery["reused_records"] == 139 * len(reused) and
            manifest["forwards_completed"] == 139 * (len(SEEDS) - len(reused)) and
            recovery["reused_records"] + manifest["forwards_completed"] == len(manifest["records"]),
            "Recovered/fresh record accounting mismatch")
    new_commit = manifest["provenance"]["code"]["export_source_commit"]
    require(manifest["generation_source_commit_by_seed"] ==
            {str(seed): old_commit if seed in reused else new_commit for seed in SEEDS},
            "Per-seed source attribution mismatch")
    expected_paths = {f"seed{seed}/window{i:06d}.json" for seed in reused for i in range(139)}
    require(set(recovery["reused_sidecar_sha256"]) == expected_paths, "Missing recovered sidecars")
    for row in manifest["records"]:
        sidecar = row["path"].removesuffix(".npz") + ".json"
        path = safe_child(root, sidecar)
        require(json.loads(path.read_text()) == row, "Sidecar/manifest row mismatch")
        if row["inference_seed"] in reused:
            require(sha256(path) == recovery["reused_sidecar_sha256"][sidecar],
                    "Recovered sidecar bytes changed")

def certify_bank(manifest_path, expected_sha256):
    """Authenticate every row and baseline before returning any data for shuffling."""
    manifest_path = Path(manifest_path).resolve()
    require(sha256(manifest_path) == expected_sha256, "Manifest fingerprint mismatch")
    manifest = json.loads(manifest_path.read_text())
    require(manifest["schema"] == "rsjg-canonical-a-bank-v1" and manifest["status"] == "COMPLETE",
            "Incomplete or unsupported bank")
    require(manifest["protocol"] == PROTOCOL, "Wrong route/protocol/precision/seeds")
    provenance = manifest["provenance"]
    require(provenance["checkpoint_sha256"] == CHECKPOINT_SHA and
            provenance["config_sha256"] == CONFIG_SHA and
            provenance["cache_manifest_sha256"] == CACHE_SHA and
            provenance["cache_pin"] == CACHE_PIN, "Frozen input provenance mismatch")
    require(provenance["validated_cache_manifest"] is True and
            provenance["strict_checkpoint_load"] is True and
            provenance["exact_zero_eval_head"] is True, "Missing runtime input gates")
    authenticate_code(provenance["code"])
    authenticate_recovery(manifest, manifest_path.parent)
    for path, expected in ((CONFIG, CONFIG_SHA), (CHECKPOINT, CHECKPOINT_SHA),
                           (CACHE + "/manifest.json", CACHE_SHA)):
        require(sha256(REPO / path) == expected, f"Local frozen input changed: {path}")
    records = manifest["records"]
    require(len(records) == 139 * 5,
            "Incomplete 139-window x 5-seed coverage")
    require([(r["inference_seed"], r["window_index"]) for r in records] ==
            [(seed, i) for seed in SEEDS for i in range(139)], "Data order/coverage mismatch")
    kernel = load_kernel()
    _, jmm = kernel.modules()
    seen, by_seed, errors, loaded, identities = set(), {}, [], [], {}
    checked_inputs = {}
    for row in records:
        seed, index = row["inference_seed"], row["window_index"]
        require(seed in SEEDS and type(index) is int and 0 <= index < 139, "Invalid seed/index")
        require((seed, index) not in seen, "Duplicate scene-seed row")
        seen.add((seed, index))
        scene, arrays = load_record(manifest_path.parent, row, kernel)
        filename = provenance["physical_validation_filenames"][index]
        input_paths = {"raw_source_sha256": safe_child((REPO / "data").resolve(), row["source_id"]),
                       "source_batch_sha256": safe_child(REPO / SOURCE_BATCHES / "valid_batches", filename),
                       "deployment_cache_sha256": REPO / CACHE / "valid" / f"{index:06d}.pt"}
        for key, path in input_paths.items():
            if str(path) not in checked_inputs:
                checked_inputs[str(path)] = sha256(path)
            require(checked_inputs[str(path)] == row["input_fingerprints"][key], f"Input bytes changed: {path}")
        identity = (row["source_id"], row["window_id"], row["scene_id"])
        require(all(isinstance(v, str) and v for v in identity), "Missing stable identity")
        fixed = (identity, tuple(row["agent_ids"]),
                 tuple((k, digest_bytes(arrays[k].tobytes())) for k in
                       ("GT", "metric_mask", "future_mask", "edge_index", "frame_ids")))
        if index in identities:
            require(identities[index] == fixed, "Cross-seed scene/order/GT/mask/graph mismatch")
        identities[index] = fixed
        score = kernel.score(scene, np, jmm)
        baseline = row["baseline"]
        for name in ("agent_minADE", "agent_minFDE"):
            compare(score[name], baseline[name], f"seed={seed}/window={index}/{name}", errors)
        for name in METRICS:
            if name in score["metrics"]:
                compare(score["metrics"][name], baseline["metrics"][name],
                        f"seed={seed}/window={index}/{name}", errors)
        by_seed.setdefault(seed, []).append(score)
        loaded.append((row, scene, arrays, score))
    require(len({v[0] for v in identities.values()}) == 139, "Duplicate physical window identity")
    for seed in SEEDS:
        group = [(r, a) for r, _, a, _ in loaded if r["inference_seed"] == seed]
        counts = {"E=0": sum(not a["degree"].any() for _, a in group),
                  "mixed": sum(a["degree"].any() and not a["degree"].all() for _, a in group),
                  "all-active": sum(a["degree"].all() for _, a in group)}
        require(counts == {"E=0": 47, "mixed": 30, "all-active": 62} and
                sum(a["Y"].shape[0] for _, a in group) == 368, "Historical graph/agent coverage mismatch")
        summary = aggregate_scores(by_seed[seed])
        for name in METRICS:
            if name in summary:
                compare(summary[name], manifest["baseline_by_seed"][str(seed)][name],
                        f"aggregate/seed={seed}/{name}", errors)
    return manifest, loaded, {"status": "PASS", "atol": 1e-6, "rtol": 1e-6,
                              "max_abs_error": max(errors), "checks": len(errors)}

#!/usr/bin/env python3
"""Prepare and audit the bounded official-HOTEL A0/A1 experiment."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from src.jdv2_objective_state import atomic_save
from src.joint_dependency_v2_cache import (
    load_cache_record, sha256_file, stable_json_hash,
)
from src.models.model import GDTS
from src.p2_checkpoint import state_hash
from src.utils import set_seed


PARENT_SHA256 = (
    "5c101c2474ebb1a3cb3ecf882f0e9db2fbb2489c741c58068b8183b0b663b97b")
TRAIN_SEEDS = (3101, 3102, 3103)
EVAL_SEEDS = (2035, 2036, 2037, 2038, 2039)
ORDER = (
    ("A0", 3101), ("A1", 3101), ("A1", 3102),
    ("A0", 3102), ("A0", 3103), ("A1", 3103),
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def verify_parent(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or len(state) != 112:
        raise RuntimeError("Official HOTEL parent must contain 112 state tensors")
    digest = sha256_file(path)
    if digest != PARENT_SHA256 or payload.get("epoch") != 110:
        raise RuntimeError("Official HOTEL parent byte/epoch identity mismatch")
    return {
        "path": str(path.resolve()), "sha256": digest,
        "epoch": int(payload["epoch"]), "state_tensor_count": len(state),
        "size_bytes": path.stat().st_size,
    }


def load_config(path: Path) -> dict:
    with path.open() as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError("Resolved config must be a mapping")
    return value


def model_from_parent(config: dict, parent: Path, seed: int) -> GDTS:
    args = SimpleNamespace(**copy.deepcopy(config))
    args.seed = int(seed)
    args.phase = "train"
    args.jdv2_initial_state_path = None
    args.jdv2_initial_state_sha256 = None
    args.jdv2_active = True
    set_seed(seed, use_cuda=False)
    model = GDTS(args, torch.device("cpu"))
    payload = torch.load(parent, map_location="cpu", weights_only=False)
    source = payload["model_state_dict"]
    incompatible = model.load_state_dict(source, strict=False)
    allowed = (
        "interaction_graph.", "social_encoder.", "relation_inference.",
        "jdv2_",
    )
    invalid_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith(allowed)]
    if invalid_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Parent mapping mismatch: missing={invalid_missing}, "
            f"unexpected={incompatible.unexpected_keys}")
    loaded = model.state_dict()
    for name, tensor in source.items():
        if name not in loaded or not torch.equal(loaded[name].cpu(), tensor.cpu()):
            raise RuntimeError(f"Official parent tensor mismatch after load: {name}")
    output = model.jdv2_corrector.output[-1]
    if (torch.count_nonzero(output.weight).item() or
            torch.count_nonzero(output.bias).item()):
        raise RuntimeError("Fresh dependency-corrector output is not exact zero")
    return model


def make_initials(args) -> None:
    config = load_config(Path(args.config))
    parent = Path(args.parent).resolve()
    verify_parent(parent)
    with Path(args.cache_manifest).open() as handle:
        cache_manifest = json.load(handle)
    cache_hash = stable_json_hash(cache_manifest)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in TRAIN_SEEDS:
        model = model_from_parent(config, parent, seed)
        model_state = model.state_dict()
        tensor_hash = state_hash(model_state)
        target = output / f"jdv2_initial_seed{seed}.pt"
        atomic_save({
            "artifact_role": "hotel_a0_a1_shared_initial",
            "format_version": 1,
            "seed": seed,
            "optimizer_updates": 0,
            "parent_checkpoint_sha256": PARENT_SHA256,
            "cache_manifest_hash": cache_hash,
            "state_sha256": tensor_hash,
            "model_state_dict": model_state,
        }, str(target))
        check = torch.load(target, map_location="cpu", weights_only=True)
        if state_hash(check["model_state_dict"]) != tensor_hash:
            raise RuntimeError("Initial-state atomic readback mismatch")
        records.append({
            "seed": seed, "path": str(target),
            "file_sha256": sha256_file(target),
            "state_sha256": tensor_hash,
            "size_bytes": target.stat().st_size,
        })
    write_json(output / "INITIAL_STATE_SUMMARY.json", {
        "status": "PASS", "parent_sha256": PARENT_SHA256,
        "cache_manifest_hash": cache_hash, "initial_states": records,
    })


def audit_cache(args) -> None:
    root = Path(args.cache_root).resolve()
    with (root / "manifest.json").open() as handle:
        manifest = json.load(handle)
    if manifest["goal_checkpoint_hash"] != PARENT_SHA256:
        raise RuntimeError("Cache is not bound to the official HOTEL parent")
    counts = {}
    agents = {}
    edges = {}
    edge_buckets = {"E=0": 0, "E>0": 0}
    for split in ("train", "valid", "test"):
        files = sorted((root / split).glob("[0-9]*.pt"))
        counts[split] = len(files)
        agents[split] = 0
        edges[split] = 0
        for path in files:
            record = load_cache_record(
                str(path), allow_future_supervision=False)
            candidates = record["goal_candidates_world"]
            prior = record["candidate_log_prior"]
            edge_index = record["edge_index"].long()
            if candidates.ndim != 3 or candidates.shape[1:] != (21, 2):
                raise RuntimeError(f"Invalid candidate shape: {path}")
            if prior.shape != candidates.shape[:2] or not torch.isfinite(
                    candidates).all() or not torch.isfinite(prior).all():
                raise RuntimeError(f"Invalid candidate values: {path}")
            scene_index = record["scene_index"].long()
            if edge_index.numel() and not torch.equal(
                    scene_index[edge_index[0]], scene_index[edge_index[1]]):
                raise RuntimeError(f"Cross-scene graph edge: {path}")
            if edge_index.numel() and (edge_index[0] >= edge_index[1]).any():
                raise RuntimeError(f"Non-canonical graph edge: {path}")
            agents[split] += int(candidates.shape[0])
            edges[split] += int(edge_index.shape[1])
            edge_buckets["E>0" if edge_index.shape[1] else "E=0"] += 1
            teacher = path.with_name(path.stem + ".teacher.pt")
            if not teacher.is_file():
                raise RuntimeError(f"Missing teacher sidecar: {teacher}")
            teacher_record = load_cache_record(
                str(teacher), allow_future_supervision=True)
            if teacher_record["cache_id"] != record["cache_id"]:
                raise RuntimeError(f"Teacher/deployment identity mismatch: {path}")
    receipt = {
        "status": "PASS", "cache_root": str(root),
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "manifest_hash": stable_json_hash(manifest),
        "record_counts": counts, "agent_occurrences": agents,
        "edge_occurrences": edges, "edge_window_buckets": edge_buckets,
        "audit_scope": "all deployment records and teacher identity sidecars",
    }
    write_json(Path(args.receipt), receipt)


def freeze_configs(args) -> None:
    base = load_config(Path(args.base_config))
    summary = json.load(open(Path(args.initial_summary)))
    initials = {item["seed"]: item for item in summary["initial_states"]}
    archive = Path(args.archive_dir).resolve()
    archive.mkdir(parents=True, exist_ok=True)
    root = Path(args.repository).resolve()
    rows = []
    configs = {}
    for arm, seed in ORDER:
        run_name = f"hotel_stage_a_{arm.lower()}_seed{seed}_bounded_20261006"
        model_dir = root / "outputs" / "joint_dependency_v2" / "hotel" / \
            "joint_dependency_v2" / "runs" / run_name
        config_path = model_dir / "config.yaml"
        config = copy.deepcopy(base)
        config.update({
            "run_name": run_name,
            "save_dir": str(model_dir), "model_dir": str(model_dir),
            "config": str(config_path), "phase": "train",
            "seed": seed, "validation_seed": 2035,
            "num_epochs": 40, "save_every": 1,
            "start_validation": 1, "validate_every": 1,
            "early_stopping_patience": 0,
            "learning_rate": 1e-4, "optimizer": "Adam",
            "scheduler": "ExponentialLR", "clip": 1.0,
            "num_workers": 0, "data_augmentation": False,
            "shuffle_train_batches": True,
            "amp_enabled": True, "amp_dtype": "bf16",
            "collision_threshold_meter": 0.2,
            "best_metric": "JADE", "use_wandb": False,
            "jdv2_goal_objective": (
                "mean_energy" if arm == "A0" else
                "expected_conditional_mc"),
            "jdv2_neighbor_draws": 4,
            "jdv2_mc_backend": "cuda_fp32_bf16_v1",
            "jdv2_neighbor_seed": seed + 100000,
            "jdv2_cache_seed": 2025,
            "jdv2_initial_state_path": initials[seed]["path"],
            "jdv2_initial_state_sha256": initials[seed]["file_sha256"],
            "jdv2_eval_regenerate_candidates": True,
            "jdv2_evaluation_seeds": list(EVAL_SEEDS),
            "jdv2_gpu_reserved_limit_bytes": 12 * 1024 ** 3,
            "jdv2_arm_time_limit_seconds": 6 * 3600,
            "jdv2_queue_deadline_utc": args.deadline_utc,
            "jdv2_numbered_resume": True,
            "jdv2_full_resume_state": True,
            "jdv2_step_cap": 0,
            "jdv2_pair_cost_intervention": "full",
            "load_checkpoint": None, "pretrain_path": None,
            "jdv2_source_checkpoint_hash": PARENT_SHA256,
            "stage_a_parent_checkpoint_sha256": PARENT_SHA256,
        })
        model_dir.mkdir(parents=True, exist_ok=True)
        with config_path.open("w") as handle:
            yaml.safe_dump(config, handle, sort_keys=True)
        archive_config = archive / f"{arm}_seed{seed}_RESOLVED.yaml"
        with archive_config.open("w") as handle:
            yaml.safe_dump(config, handle, sort_keys=True)
        rows.append({
            "arm": arm, "seed": seed, "run_name": run_name,
            "model_dir": str(model_dir), "config": str(config_path),
            "config_sha256": sha256_file(config_path),
        })
        configs[(arm, seed)] = config
    diffs = []
    permitted = {
        "jdv2_goal_objective", "run_name", "save_dir", "model_dir",
        "config",
    }
    for seed in TRAIN_SEEDS:
        left, right = configs[("A0", seed)], configs[("A1", seed)]
        changed = sorted(key for key in set(left) | set(right)
                         if left.get(key) != right.get(key))
        disallowed = sorted(set(changed) - permitted)
        if disallowed:
            raise RuntimeError(f"Unapproved paired config difference: {disallowed}")
        diffs.append({"seed": seed, "changed_fields": changed,
                      "permitted_fields": sorted(permitted)})
    write_json(archive / "PAIRED_CONFIG_DIFF.json", {
        "status": "PASS", "pairs": diffs})
    manifest_path = Path(args.output_manifest).resolve()
    run_dir = manifest_path.parent
    baseline_output = run_dir / "results" / "E_joint_baseline.json"
    first_config = configs[("A0", 3101)]["config"]
    post_tasks = [{
        "name": "E_joint_baseline",
        "command": [str(Path(args.python).resolve()), "-u",
                    "tools/hotel_a0_a1_baseline_eval.py",
                    "--config", first_config,
                    "--parent", str(Path(args.parent).resolve()),
                    "--output", str(baseline_output)],
        "result": str(baseline_output),
    }]
    for arm, seed in ORDER:
        run_name = configs[(arm, seed)]["run_name"]
        model_dir = Path(configs[(arm, seed)]["model_dir"])
        target = run_dir / "results" / f"E_joint_{arm}_{seed}.json"
        post_tasks.append({
            "name": f"E_joint_{arm}_{seed}",
            "command": [str(Path(args.python).resolve()), "-u", "main.py",
                        "--dataset", "eth5", "--test_set", "hotel",
                        "--goal_model_type", "joint_dependency_v2",
                        "--run_name", run_name, "--phase", "test",
                        "--load_checkpoint", "best"],
            "result_source": str(model_dir / "final_test_results.json"),
            "result": str(target),
        })
    a1_run = configs[("A1", 3101)]["run_name"]
    a1_dir = Path(configs[("A1", 3101)]["model_dir"])
    for intervention in ("off", "physical", "pure_interaction_off"):
        target = run_dir / "results" / f"mechanism_{intervention}.json"
        post_tasks.append({
            "name": f"mechanism_{intervention}",
            "command": [str(Path(args.python).resolve()), "-u", "main.py",
                        "--dataset", "eth5", "--test_set", "hotel",
                        "--goal_model_type", "joint_dependency_v2",
                        "--run_name", a1_run, "--phase", "test",
                        "--load_checkpoint", "best",
                        "--jdv2_pair_cost_intervention", intervention],
            "result_source": str(a1_dir / "final_test_results.json"),
            "result": str(target),
        })
    write_json(manifest_path, {
        "schema": "hotel-a0-a1-queue-v1",
        "status": "FROZEN_NOT_LAUNCHED", "deadline_utc": args.deadline_utc,
        "python": str(Path(args.python).resolve()),
        "source_dir": str(Path(args.source_dir).resolve()),
        "repository": str(root), "arms": rows,
        "training_order": [f"{arm}_{seed}" for arm, seed in ORDER],
        "evaluation_seeds": list(EVAL_SEEDS),
        "parent_sha256": PARENT_SHA256,
        "post_tasks": post_tasks,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    parent = sub.add_parser("verify-parent")
    parent.add_argument("--parent", required=True)
    parent.add_argument("--receipt", required=True)
    initial = sub.add_parser("make-initials")
    initial.add_argument("--config", required=True)
    initial.add_argument("--parent", required=True)
    initial.add_argument("--cache-manifest", required=True)
    initial.add_argument("--output-dir", required=True)
    audit = sub.add_parser("audit-cache")
    audit.add_argument("--cache-root", required=True)
    audit.add_argument("--receipt", required=True)
    freeze = sub.add_parser("freeze-configs")
    freeze.add_argument("--base-config", required=True)
    freeze.add_argument("--initial-summary", required=True)
    freeze.add_argument("--archive-dir", required=True)
    freeze.add_argument("--repository", required=True)
    freeze.add_argument("--source-dir", required=True)
    freeze.add_argument("--python", required=True)
    freeze.add_argument("--parent", required=True)
    freeze.add_argument("--deadline-utc", required=True)
    freeze.add_argument("--output-manifest", required=True)
    args = parser.parse_args()
    if args.command == "verify-parent":
        write_json(Path(args.receipt), verify_parent(Path(args.parent)))
    elif args.command == "make-initials":
        make_initials(args)
    elif args.command == "audit-cache":
        audit_cache(args)
    else:
        freeze_configs(args)


if __name__ == "__main__":
    main()

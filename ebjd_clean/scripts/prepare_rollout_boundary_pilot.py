#!/usr/bin/env python3
"""Freeze the ETH native-goal rollout-boundary pilot before formal updates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path

import torch


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-epoch10", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()

    checkpoint = Path(args.candidate_epoch10).resolve()
    actual_sha = sha256(checkpoint)
    if actual_sha != args.expected_sha256:
        raise ValueError(
            f"candidate epoch-10 SHA256 mismatch: {actual_sha}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("schema") != "ebjd-checkpoint-v2" or \
            payload.get("epoch_boundary_only") is not True:
        raise ValueError("candidate must be a complete epoch-boundary v2 checkpoint")
    if int(payload["epoch"]) != 10 or int(payload["update_index"]) != 2570:
        raise ValueError("candidate is not the frozen epoch-10/update-2570 boundary")
    required = {"model", "optimizer", "rng", "selection_state", "resolved_config"}
    if not required.issubset(payload):
        raise ValueError(f"candidate checkpoint is incomplete: {sorted(required-set(payload))}")

    source_root = Path(__file__).resolve().parents[2]
    source_commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain", "--", "ebjd_clean"],
        text=True).strip()
    if dirty:
        raise ValueError("refusing to freeze protocol from a dirty ebjd_clean tree")

    base = deepcopy(payload["resolved_config"])
    if not base.get("native_goal_conditioning", {}).get("enabled"):
        raise ValueError("candidate checkpoint lacks enabled native-goal conditioning")
    if int(base["training"]["rollout_start_epoch"]) != 21:
        raise ValueError("rollout start boundary is not epoch 21")
    run_root = Path(args.run_root).resolve()
    configs = {}
    for name, directory in (
        ("shared", "shared_runs"), ("on", "on_runs"), ("off", "off_runs")
    ):
        config = deepcopy(base)
        config["experiment"]["source_commit"] = source_commit
        config["paths"]["output_root"] = str(run_root / directory)
        configs[name] = config

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config_paths = {}
    for name, config in configs.items():
        path = output / f"{name.upper()}_CONFIG.json"
        atomic_json(path, config)
        config_paths[name] = path

    validation_manifest = Path(base["paths"]["validation_manifest"])
    expected_manifest = (
        "03f9b6bef89d43137b31782bbf0eb5b16d19f8ddfe378e2c7c62e700e128397b")
    if sha256(validation_manifest) != expected_manifest:
        raise ValueError("held-out validation manifest SHA256 mismatch")
    train_cache = Path(base["native_goal_conditioning"]["train_cache_manifest"])
    if sha256(train_cache) != base["native_goal_conditioning"]["train_cache_sha256"]:
        raise ValueError("train native-goal bank manifest SHA256 mismatch")
    validation_banks = {
        seed: sha256(path) for seed, path in
        base["native_goal_conditioning"]["validation_cache_by_seed"].items()}

    plan = {
        "schema": "ebjd-eth-rollout-boundary-pilot-plan-v1",
        "status": "FROZEN_BEFORE_FORMAL_UPDATES",
        "implementation_commit": source_commit,
        "origin": {
            "path": str(checkpoint), "sha256": actual_sha,
            "epoch": 10, "update_index": 2570,
            "complete_state": True,
        },
        "assets": {
            "heldout_manifest_sha256": expected_manifest,
            "train_goal_bank_manifest_sha256": sha256(train_cache),
            "validation_goal_bank_manifest_sha256_by_seed": validation_banks,
            "provider_checkpoint_sha256": base["native_goal_conditioning"][
                "provider_checkpoint_sha256"],
            "provider_code_commit": base["native_goal_conditioning"][
                "provider_code_commit"],
        },
        "configs": {name: {
            "path": str(path), "sha256": sha256(path)}
            for name, path in config_paths.items()},
        "stages": {
            "A_shared": {"epochs": list(range(11, 21)), "rollout": False,
                         "fixed_checkpoint": "epoch_020.pt"},
            "B_on": {"epochs": list(range(21, 26)),
                     "rollout_loss": True, "marginal_projection": True},
            "B_off": {"epochs": list(range(21, 26)),
                      "ablation": "rollout_mechanism_off",
                      "rollout_loss": False, "marginal_projection": False},
        },
        "sole_stage_B_intervention": (
            "rollout loss plus paired marginal actual-step projection ON versus OFF"),
        "full_state_pairing": (
            "model+optimizer+LR schedule+global update+base RNG+loader RNG+"
            "isolated rollout RNG restored from the same epoch-20 checkpoint"),
        "evaluation": {
            "fixed_checkpoints": ["shared_epoch20", "off_epoch25", "on_epoch25"],
            "reused_references": ["candidate_epoch10", "matched_gdts"],
            "seeds": list(range(2035, 2040)), "worlds": 20, "steps": 20,
            "eta": 0.0, "precision": "fp32", "gamma": 1.0,
            "per_agent_reordering": False, "independent_test": False,
        },
        "runtime_estimate_hours": {
            "shared_11_20": 2.2, "off_21_25": 1.1,
            "on_21_25": 3.4, "three_five_seed_evaluations": 0.35,
            "total": 7.05,
        },
        "scope": {"training_seed": 3101, "lr_schedule_epochs": 100,
                  "maximum_formal_epoch": 25, "weight_scan": False,
                  "other_datasets": False},
    }
    atomic_json(output / "PLAN.json", plan)
    commands = f"""# ETH rollout-boundary executable command

```bash
cd {source_root / 'ebjd_clean'}
bash scripts/run_eth_rollout_boundary_pipeline.sh
```

Frozen protocol: `{output}`  
Local-only run root: `{run_root}`  
Origin checkpoint: `{checkpoint}`  
Origin SHA256: `{actual_sha}`
"""
    (output / "COMMANDS.md").write_text(commands, encoding="utf-8")
    print(json.dumps(plan, sort_keys=True))


if __name__ == "__main__":
    main()

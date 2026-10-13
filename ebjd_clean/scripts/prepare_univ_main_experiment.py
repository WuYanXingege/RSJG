#!/usr/bin/env python3
"""Materialize and audit the single-seed formal UNIV EBJD experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path

import torch
import yaml

from ebjd.config import config_differences, load_config, validate_supported_config
from ebjd.train import validate_execution_migration_settings


EXECUTION_PROFILE = {
    "execution_strategy": "small_group_uncheckpoint_v1",
    "scene_forward_chunk": 1,
    "scene_compaction": True,
    "physical_group_max_scenes": 4,
    "physical_group_max_agent_slots": 96,
    "physical_group_max_padding_ratio": 1.5,
    "rollout_block_checkpointing": False,
    "rollout_uncheckpointed_max_agent_slots": 32,
}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
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


def atomic_yaml(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def manifest_receipt(path: Path) -> tuple[dict, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    statistics = payload["statistics"]
    receipt = {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "metadata": payload["metadata"],
        "statistics": statistics,
        "source_files": payload["source_files"],
        "map_sources": payload["map_sources"],
        "shard_count": len(payload["shards"]),
        "all_shards_present_and_bound": all(
            (path.parent / descriptor["path"]).is_file()
            and sha256(path.parent / descriptor["path"]) == descriptor["sha256"]
            for descriptor in payload["shards"]),
    }
    return receipt, payload


def verify_goal_checkpoint(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict")
    if payload.get("checkpoint_type") != "gdts_goal_pretrain":
        raise RuntimeError("UNIV initialization is not a GDTS goal-pretrain artifact")
    if payload.get("test_set") != "univ":
        raise RuntimeError("goal-pretrain initialization is not fold=univ")
    if int(payload.get("training_epochs_planned", -1)) != 150:
        raise RuntimeError("goal-pretrain artifact was not trained on the 150-epoch protocol")
    if not isinstance(state, dict) or len(state) != 46:
        raise RuntimeError("goal-pretrain artifact must contain exactly 46 U-Net tensors")
    if any(not name.startswith("goal_module.") for name in state):
        raise RuntimeError("goal-pretrain artifact contains non-goal model parameters")
    config_path = path.parent.parent / "config.yaml"
    source_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected = {
        "dataset": "eth5", "test_set": "univ", "phase": "goal_pretrain",
        "seed": 3101, "num_epochs": 150, "batch_size": 64,
        "learning_rate": 0.001, "optimizer": "Adam",
        "scheduler": "ExponentialLR", "data_augmentation": False,
        "shuffle_train_batches": False, "shuffle_test_batches": False,
        "amp_enabled": False, "amp_dtype": "fp32",
    }
    mismatched = {
        key: {"expected": value, "actual": source_config.get(key)}
        for key, value in expected.items() if source_config.get(key) != value}
    if mismatched:
        raise RuntimeError(f"goal-pretrain config differs: {mismatched}")
    nonfinite = sum(
        int((~torch.isfinite(value)).sum())
        for value in state.values() if torch.is_tensor(value) and value.is_floating_point())
    if nonfinite:
        raise RuntimeError("goal-pretrain checkpoint contains non-finite parameters")
    return {
        "path": str(path.resolve()), "sha256": sha256(path),
        "checkpoint_type": payload["checkpoint_type"],
        "format_version": payload.get("format_version"),
        "epoch": int(payload["epoch"]), "test_set": payload["test_set"],
        "training_epochs_planned": int(payload["training_epochs_planned"]),
        "selection_metric": payload.get("best_metric_name"),
        "best_metrics": payload.get("best_metrics"),
        "best_metrics_epochs": payload.get("best_metrics_epochs"),
        "state_tensor_count": len(state), "non_goal_tensor_count": 0,
        "source_config": str(config_path.resolve()),
        "source_config_sha256": sha256(config_path),
        "source_config_contract": expected,
    }


def require_same_protocol(hotel: dict, univ: dict) -> None:
    exact_sections = (
        "data", "map_encoder", "history_encoder", "representation",
        "denoiser", "diffusion", "objectives", "ablations",
    )
    mismatched = {
        section: config_differences(hotel[section], univ[section])
        for section in exact_sections if hotel[section] != univ[section]}
    hotel_training = deepcopy(hotel["training"])
    univ_training = deepcopy(univ["training"])
    for key in EXECUTION_PROFILE:
        hotel_training.pop(key, None)
        univ_training.pop(key, None)
    if hotel_training != univ_training:
        mismatched["training_without_execution_profile"] = config_differences(
            hotel_training, univ_training)
    evaluation_fields = (
        "worlds", "steps", "selection_seed", "inference_seeds",
        "minimum_training_seeds", "metrics", "per_agent_reordering",
        "final_test_used_for_checkpoint_selection", "validation_test_identity",
        "marginal_tolerance",
    )
    evaluation_diff = {
        key: {"hotel": hotel["evaluation"].get(key),
              "univ": univ["evaluation"].get(key)}
        for key in evaluation_fields
        if hotel["evaluation"].get(key) != univ["evaluation"].get(key)}
    if evaluation_diff:
        mismatched["evaluation_protocol"] = evaluation_diff
    if mismatched:
        raise RuntimeError(f"UNIV changes model/optimizer semantics: {mismatched}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-univ-config", required=True)
    parser.add_argument("--hotel-checkpoint", required=True)
    parser.add_argument("--goal-checkpoint", required=True)
    parser.add_argument("--experiment-output-root", required=True)
    parser.add_argument("--archive-dir", required=True)
    parser.add_argument("--output-config", required=True)
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "ebjd_clean"],
        cwd=repository, text=True).strip()
    if dirty:
        raise RuntimeError("refuse formal materialization from a dirty ebjd_clean tree")

    base = load_config(args.base_univ_config)
    hotel_payload = torch.load(
        args.hotel_checkpoint, map_location="cpu", weights_only=False)
    if int(hotel_payload.get("epoch", -1)) != 45:
        raise RuntimeError("HOTEL reference is not epoch 45")
    if sha256(args.hotel_checkpoint) != (
        "425988097793b639eb868fcc9ff28e41a3022c08598d13f396383da89268e1b1"
    ):
        raise RuntimeError("HOTEL epoch-45 reference SHA256 mismatch")
    hotel = hotel_payload["resolved_config"]
    initialization = verify_goal_checkpoint(Path(args.goal_checkpoint).resolve())

    config = deepcopy(base)
    config["experiment"]["source_commit"] = commit
    config["paths"]["output_root"] = str(Path(args.experiment_output_root).resolve())
    config["paths"]["accepted_fold_matched_gdts_unet"] = initialization["path"]
    config["paths"]["accepted_fold_matched_gdts_unet_sha256"] = initialization["sha256"]
    config["evaluation"]["matched_gdts_validation"][
        "ebjd_initialization_sha256"] = initialization["sha256"]
    config["evaluation"]["matched_gdts_validation"]["reference_limitation"] = (
        "comparator_full_checkpoint_differs_from_fold_matched_goal_pretrain_initialization")
    config["training"].update(EXECUTION_PROFILE)
    validate_supported_config(config)
    validate_execution_migration_settings(config)
    require_same_protocol(hotel, config)

    train_path = Path(config["paths"]["train_manifest"]).resolve()
    validation_path = Path(config["paths"]["validation_manifest"]).resolve()
    test_path = Path(config["paths"]["test_manifest"]).resolve()
    train_receipt, train = manifest_receipt(train_path)
    validation_receipt, validation = manifest_receipt(validation_path)
    test_receipt, test = manifest_receipt(test_path)
    train_keys = set(train["window_keys"])
    validation_keys = set(validation["window_keys"])
    test_keys = set(test["window_keys"])
    data_receipt = {
        "schema": "ebjd-univ-data-receipt-v1", "fold": "univ",
        "train": train_receipt, "validation": validation_receipt,
        "test": test_receipt,
        "window_identity_intersections": {
            "train_validation": len(train_keys & validation_keys),
            "train_test": len(train_keys & test_keys),
            "validation_test": len(validation_keys & test_keys),
        },
        "validation_test_mirror": (
            validation_keys == test_keys
            and [item["sha256"] for item in validation["source_files"]]
            == [item["sha256"] for item in test["source_files"]]),
        "independent_test": False,
    }
    if not all(item["all_shards_present_and_bound"] for item in (
            train_receipt, validation_receipt, test_receipt)):
        raise RuntimeError("a manifest shard is missing or has a SHA256 mismatch")
    if data_receipt["window_identity_intersections"]["train_validation"] != 0:
        raise RuntimeError("UNIV train and held-out validation window identities overlap")
    if data_receipt["window_identity_intersections"]["train_test"] != 0:
        raise RuntimeError("UNIV train and held-out test window identities overlap")
    if not data_receipt["validation_test_mirror"]:
        raise RuntimeError("expected official UNIV validation/test mirror is not exact")

    archive = Path(args.archive_dir).resolve()
    config_path = Path(args.output_config).resolve()
    atomic_yaml(config, config_path)
    if load_config(config_path) != config:
        raise RuntimeError("materialized YAML changed values or scalar types")
    differences = config_differences(hotel, config)
    difference_receipt = {
        "schema": "ebjd-hotel-epoch45-to-univ-difference-v1",
        "hotel_checkpoint": str(Path(args.hotel_checkpoint).resolve()),
        "hotel_checkpoint_sha256": sha256(args.hotel_checkpoint),
        "univ_config": str(config_path), "source_commit": commit,
        "differences": differences,
        "semantic_and_optimizer_protocol_exact": True,
        "categorized_change_prefixes": [
            "experiment", "paths", "evaluation.matched_gdts_validation",
            "training.execution", "training.physical_group",
            "training.rollout_uncheckpointed_max_agent_slots",
        ],
    }
    atomic_json(data_receipt, archive / "UNIV_DATA_RECEIPT.json")
    atomic_json(initialization, archive / "UNIV_INITIALIZATION_RECEIPT.json")
    atomic_json(difference_receipt, archive / "HOTEL_TO_UNIV_DIFFERENCES.json")
    result = {
        "schema": "ebjd-univ-main-materialization-v1", "status": "READY",
        "source_commit": commit, "config": str(config_path),
        "config_sha256": sha256(config_path),
        "execution_profile": EXECUTION_PROFILE,
        "expected_updates_per_epoch": 255,
        "expected_terminal_update_index": 25500,
        "minimum_training_seeds_requirement": 3,
        "training_seeds_executed_by_this_experiment": [3101],
        "data_receipt": str(archive / "UNIV_DATA_RECEIPT.json"),
        "initialization_receipt": str(archive / "UNIV_INITIALIZATION_RECEIPT.json"),
        "difference_receipt": str(archive / "HOTEL_TO_UNIV_DIFFERENCES.json"),
    }
    atomic_json(result, archive / "MATERIALIZATION_RECEIPT.json")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

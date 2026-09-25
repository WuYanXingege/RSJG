"""Fail-closed contracts for the JDV2 clean ETH reproduction protocol."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from src import clean_split_protocol as protocol
from src.data_loader import get_dataloader
from src.joint_dependency_v2_cache import (
    load_cache_record, sha256_file, stable_json_hash,
)
from src.models.joint_dependency_v2 import DependencyCorrector
from src.parser import check_and_add_additional_args, get_parser
from src.trainer import _requested_data_splits, trainer as Trainer


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / (
    "outputs/joint_dependency_v2/eth/joint_dependency_v2/"
    "clean_split_protocol/manifest.json")
PREFLIGHT = MANIFEST.with_name("preflight_results.json")
STAGE_A_CONFIG = ROOT / (
    "configs/joint_dependency_v2/jdv2_stage_a_clean_eth.yaml")
STAGE_B_CONFIG = ROOT / (
    "configs/joint_dependency_v2/jdv2_stage_b_v2a_clean_eth.yaml")
OLD_STAGE_B_CONFIG = ROOT / (
    "configs/joint_dependency_v2/jdv2_stage_b_v2a_eth.yaml")
HISTORICAL_STAGE_A_CONFIG = ROOT / (
    "outputs/joint_dependency_v2/eth/joint_dependency_v2/runs/"
    "jdv2_stage_a_no_z_full_seed2035/config.yaml")


def _yaml(path):
    return yaml.safe_load(path.read_text())


def _resolved_config(path):
    values = vars(get_parser().parse_args([]))
    values.update(_yaml(path))
    return check_and_add_additional_args(argparse.Namespace(**values))


def test_manifest_exact_membership_sources_overlap_and_determinism():
    raw = json.loads(MANIFEST.read_text())
    manifest = protocol.load_clean_split_manifest(
        MANIFEST, raw["manifest_hash"])
    assert {key: value["window_count"]
            for key, value in manifest["splits"].items()} == {
                "train": 3790, "internal_valid": 320, "final_test": 139}
    assert [item["physical_cache_index"] for item in
            manifest["splits"]["train"]["records"]] == list(range(3790))
    assert [item["physical_cache_index"] for item in
            manifest["splits"]["internal_valid"]["records"]] == \
        list(range(3790, 4110))
    assert {item["raw_source_path"] for item in
            manifest["splits"]["internal_valid"]["records"]} == {
                "data/eth5/eth/train/uni_examples.txt"}
    assert {item["raw_source_path"] for item in
            manifest["splits"]["final_test"]["records"]} == {
                "data/eth5/eth/val/biwi_eth.txt"}
    assert all(value == 0 for value in manifest["overlap_counts"].values())
    assert stable_json_hash({key: value for key, value in raw.items()
                             if key != "manifest_hash"}) == \
        raw["manifest_hash"]
    preflight = json.loads(PREFLIGHT.read_text())
    assert preflight["repeated_construction_identical"] is True
    assert preflight["internal_valid_jdv2_alignment"] == {
        "passed": 320, "total": 320}


def test_all_320_manifest_members_address_original_train_jdv2_records():
    manifest = json.loads(MANIFEST.read_text())
    root = ROOT / "outputs/joint_dependency_v2/cache/eth_full_stage_a/train"
    for record in manifest["splits"]["internal_valid"]["records"]:
        index = record["physical_cache_index"]
        assert record["cache_filename"] == f"train_batch_{index + 1:04d}.pkl"
        cache = load_cache_record(
            root / record["jdv2_cache_filename"],
            allow_future_supervision=False)
        assert cache["cache_id"] == f"train-{index:06d}"
        frame_ids = cache["frame_ids"]
        assert frame_ids[:, 0].tolist() == record["frame_ids"]
        assert torch.equal(frame_ids, frame_ids[:, :1].expand_as(frame_ids))
        assert cache["goal_candidates_world"].shape[1] == 21
        assert cache["edge_index"].shape[0] == 2


def test_real_logical_valid_permissions_and_original_indices():
    args = _resolved_config(STAGE_A_CONFIG)
    valid = get_dataloader(args, "valid")
    assert valid.dataset.physical_indices == list(range(3790, 4110))
    assert valid.dataset.dataset.physical_set_name == "train"
    assert valid.dataset.dataset.logical_set_name == "valid"
    assert valid.dataset.dataset.data_augmentation is False
    batch, _ = valid.dataset[0]
    assert "jdv2_cache" in batch
    assert batch["jdv2_cache"]["cache_id"] == "train-003790"
    assert "jdv2_teacher_cache" not in batch


def test_real_logical_train_retains_teacher_sidecar():
    args = _resolved_config(STAGE_A_CONFIG)
    train = get_dataloader(args, "train")
    batch, _ = train.dataset[0]
    assert train.dataset.dataset.logical_set_name == "train"
    assert batch["jdv2_cache"]["cache_id"] == "train-000000"
    assert batch["jdv2_teacher_cache"]["cache_id"] == "train-000000"


def test_split_construction_is_phase_scoped_and_legacy_compatible():
    assert _requested_data_splits(SimpleNamespace(
        clean_split_protocol=True, phase="train")) == ["train", "valid"]
    assert _requested_data_splits(SimpleNamespace(
        clean_split_protocol=True, phase="test")) == ["test"]
    assert _requested_data_splits(SimpleNamespace(
        clean_split_protocol=False, phase="train")) == [
            "train", "valid", "test"]


def test_final_test_fails_before_any_loader_construction():
    args = SimpleNamespace(
        clean_split_protocol=True, final_test_access="blocked")
    with pytest.raises(RuntimeError, match="locked"):
        get_dataloader(args, "test")


def _write_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "_current_commit", lambda: "a" * 40)
    model_dir = tmp_path / "target"
    best = model_dir / "saved_models" / "best_model.pt"
    best.parent.mkdir(parents=True)
    best.write_bytes(b"stage-a")
    stage_b = tmp_path / "stage_b.pt"
    stage_b.write_bytes(b"stage-b")
    config_a, config_b = tmp_path / "a.yaml", tmp_path / "b.yaml"
    config_a.write_text("a: 1\n")
    config_b.write_text("b: 2\n")
    manifest_hash = json.loads(MANIFEST.read_text())["manifest_hash"]
    lock = {
        "schema": protocol.FINAL_TEST_LOCK_SCHEMA,
        "clean_split_manifest_hash": manifest_hash,
        "source_commit": "a" * 40,
        "test_seeds": protocol.CLEAN_TEST_SEEDS,
        "configs": {
            "stage_a": {"path": str(config_a),
                        "sha256": sha256_file(config_a)},
            "stage_b": {"path": str(config_b),
                        "sha256": sha256_file(config_b)},
        },
        "checkpoints": {
            "stage_a": {"path": str(best), "sha256": sha256_file(best),
                        "selection_epoch": 1, "selection_metric": "JFDE"},
            "stage_b": {"path": str(stage_b),
                        "sha256": sha256_file(stage_b),
                        "selection_epoch": 1, "selection_metric": "JADE"},
        },
    }
    lock["lock_hash"] = stable_json_hash(lock)
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock))
    args = SimpleNamespace(
        clean_split_protocol=True,
        final_test_access="authorized",
        final_test_lock_path=str(lock_path),
        clean_split_manifest_path=str(MANIFEST),
        clean_split_manifest_hash=manifest_hash,
        clean_evaluation_target="stage_a",
        model_dir=str(model_dir),
    )
    return args, lock_path, best


def test_final_test_lock_authenticates_all_artifacts(tmp_path, monkeypatch):
    args, lock_path, best = _write_lock(tmp_path, monkeypatch)
    lock = protocol.validate_final_test_lock(args)
    assert lock["test_seeds"] == [2035, 2036, 2037, 2038, 2039]
    missing_hash = dict(lock)
    missing_hash.pop("lock_hash")
    lock_path.write_text(json.dumps(missing_hash))
    with pytest.raises(RuntimeError, match="hash is missing"):
        protocol.validate_final_test_lock(args)
    lock_path.write_text(json.dumps(lock))
    best.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        protocol.validate_final_test_lock(args)
    assert lock_path.exists()


def test_clean_parser_rejects_combined_or_unlocked_phases():
    base = vars(get_parser().parse_args([]))
    base.update(_yaml(STAGE_A_CONFIG))
    base["phase"] = "train_test"
    with pytest.raises(ValueError, match="supports only"):
        check_and_add_additional_args(argparse.Namespace(**base))
    base["phase"] = "test"
    base["final_test_access"] = "blocked"
    with pytest.raises(ValueError, match="authorized"):
        check_and_add_additional_args(argparse.Namespace(**base))


def test_clean_stage_a_scientific_training_contract_is_frozen():
    clean = _yaml(STAGE_A_CONFIG)
    historical = _yaml(HISTORICAL_STAGE_A_CONFIG)
    for key in (
            "jdv2_latent_objective", "use_scene_latent",
            "use_dynamic_relation", "use_joint_energy",
            "use_social_encoder", "jdv2_relation_modes",
            "num_goal_candidates", "num_samples", "jdv2_energy_rank",
            "num_refinement_steps", "graph_radius", "ttc_threshold",
            "learning_rate", "optimizer", "scheduler", "clip",
            "num_epochs", "start_validation", "validate_every",
            "early_stopping_patience", "save_every", "amp_enabled",
            "amp_dtype", "seed", "validation_seed"):
        assert clean[key] == historical[key], key
    assert clean["jdv2_refinement_policy"] == \
        "exact_lexicographic_persistent_tie"
    assert clean["model_selection_split"] == "internal_train"
    assert clean["final_test_access"] == "blocked"


def test_clean_stage_b_scientific_contract_is_frozen():
    clean, historical = _yaml(STAGE_B_CONFIG), _yaml(OLD_STAGE_B_CONFIG)
    protocol_keys = {
        "run_name", "model_selection_split", "internal_validation_strategy",
        "internal_validation_fraction", "internal_validation_seed",
        "clean_split_protocol", "clean_split_manifest_path",
        "clean_split_manifest_hash", "final_test_access",
        "final_test_lock_path", "clean_evaluation_target", "pretrain_path",
        "stage_a_parent_checkpoint_sha256", "shuffle_train_batches",
    }
    for key in set(historical) & set(clean) - protocol_keys:
        assert clean[key] == historical[key], key
    assert clean["pretrain_path"].endswith(
        "jdv2_stage_a_clean_eth_seed2035/saved_models/best_model.pt")
    assert clean["stage_a_parent_checkpoint_sha256"] is None
    assert sum(parameter.numel() for parameter in
               DependencyCorrector().parameters()) == 30851


def test_clean_checkpoint_manifest_provenance_rejects_mismatch(tmp_path):
    class TinyNet(torch.nn.Module):
        jdv2_active = True

        def __init__(self):
            super().__init__()
            self.jdv2_weight = torch.nn.Parameter(torch.ones(()))

    shell = Trainer.__new__(Trainer)
    shell.device = torch.device("cpu")
    shell.net = TinyNet()
    shell.args = SimpleNamespace(
        jdv2_latent_objective="strict_no_z",
        jdv2_cache_manifest_hash="cache",
        jdv2_source_checkpoint_hash="source",
        clean_split_protocol=True,
        clean_split_manifest_hash="expected",
        training_stage="joint_goal",
        stage_b_architecture_version=None,
    )
    shell._jdv2_architecture_config = lambda: {"variant": "strict_no_z"}
    shell._jdv2_ablation_config = lambda: {"scene": False}
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({
        "model_state_dict": shell.net.state_dict(),
        "latent_objective": "strict_no_z",
        "architecture_config": {"variant": "strict_no_z"},
        "ablation_config": {"scene": False},
        "cache_manifest_hash": "cache",
        "source_checkpoint_hash": "source",
        "clean_split_manifest_hash": "wrong",
        "training_stage": "joint_goal",
    }, checkpoint)
    with pytest.raises(RuntimeError, match="clean split manifest mismatch"):
        shell._load_state_file(checkpoint)

from copy import deepcopy

import torch
import pytest

from ebjd.config import config_hash
from ebjd.train import file_sha256, validate_objective_fork_payload


def fixture_payload(tmp_path):
    origin_config = {
        "experiment": {"source_commit": "old"},
        "paths": {"output_root": "/old"},
        "objectives": {"goal_v_weight": 1.0, "bridge_v_weight": 1.0},
        "training": {
            "epochs": 100, "scene_forward_chunk": 1,
            "rollout_weight_end": 0.20,
        },
    }
    fixed = {
        "implementation_base": "base", "fold": "hotel", "seed": 3101,
        "data": {"test": "same"}, "initialization": {"sha": "same"},
        "validation_comparator": {"sha": "same"}, "ablation": "none",
    }
    origin_identity = {
        **fixed, "config_hash": config_hash(origin_config),
        "source_commit": "old", "run_id": "origin",
    }
    payload = {
        "schema": "ebjd-checkpoint-v2", "epoch_boundary_only": True,
        "resolved_config": origin_config, "run_identity": origin_identity,
        "selection_state": {"candidates": []},
        "rng": {"python": (), "numpy": (), "torch_cpu": torch.zeros(1),
                "torch_cuda": [], "loader_generator": torch.zeros(1)},
        "model": {"weight": torch.ones(1)},
        "optimizer": {"state": torch.ones(1)},
        "epoch": 45, "update_index": 11970,
    }
    path = tmp_path / "epoch_045.pt"
    torch.save(payload, path)
    new_config = deepcopy(origin_config)
    new_config["experiment"]["source_commit"] = "new"
    new_config["paths"]["output_root"] = "/new"
    new_config["objectives"]["goal_v_weight"] = 2.0
    execution_identity = {
        **fixed, "config_hash": config_hash(new_config),
        "source_commit": "new", "run_id": "fork",
    }
    return payload, path, new_config, execution_identity


def test_goal_weight_fork_is_exactly_whitelisted(tmp_path):
    payload, path, config, identity = fixture_payload(tmp_path)
    receipt = validate_objective_fork_payload(
        payload, path, file_sha256(path), config, identity, "none",
        "goal_v_weight_2")
    assert receipt["origin_epoch"] == 45
    assert receipt["resume_start_epoch"] == 46
    assert receipt["checked_config_diff"]["objectives.goal_v_weight"] == {
        "old": 1.0, "new": 2.0}


def test_goal_weight_fork_rejects_any_second_semantic_change(tmp_path):
    payload, path, config, identity = fixture_payload(tmp_path)
    config["objectives"]["bridge_v_weight"] = 0.5
    identity["config_hash"] = config_hash(config)
    with pytest.raises(ValueError, match="frozen contract"):
        validate_objective_fork_payload(
            payload, path, file_sha256(path), config, identity, "none",
            "goal_v_weight_2")


def test_rollout_weight_fork_is_exactly_whitelisted(tmp_path):
    payload, path, config, identity = fixture_payload(tmp_path)
    config["objectives"]["goal_v_weight"] = 1.0
    config["training"]["rollout_weight_end"] = 0.05
    identity["config_hash"] = config_hash(config)
    receipt = validate_objective_fork_payload(
        payload, path, file_sha256(path), config, identity, "none",
        "rollout_weight_end_005")
    assert receipt["origin_epoch"] == 45
    assert receipt["resume_start_epoch"] == 46
    assert receipt["checked_config_diff"]["training.rollout_weight_end"] == {
        "old": 0.2, "new": 0.05}


def test_rollout_weight_fork_rejects_a_second_semantic_change(tmp_path):
    payload, path, config, identity = fixture_payload(tmp_path)
    config["objectives"]["goal_v_weight"] = 1.0
    config["training"]["rollout_weight_end"] = 0.05
    config["objectives"]["bridge_v_weight"] = 0.5
    identity["config_hash"] = config_hash(config)
    with pytest.raises(ValueError, match="frozen contract"):
        validate_objective_fork_payload(
            payload, path, file_sha256(path), config, identity, "none",
            "rollout_weight_end_005")


def test_rollout_mechanism_off_fork_changes_only_explicit_ablation(tmp_path):
    payload, path, config, identity = fixture_payload(tmp_path)
    config["objectives"]["goal_v_weight"] = 1.0
    identity["config_hash"] = config_hash(config)
    identity["ablation"] = "rollout_mechanism_off"
    receipt = validate_objective_fork_payload(
        payload, path, file_sha256(path), config, identity,
        "rollout_mechanism_off", "rollout_mechanism_off")
    assert receipt["checked_config_diff"] == {
        "experiment.source_commit": {"old": "old", "new": "new"},
        "paths.output_root": {"old": "/old", "new": "/new"},
    }
    assert "actual-step projection" in receipt["intervention"]


@pytest.mark.parametrize("weight", [0.0, 0.10])
def test_fixed_anchor_path_fork_accepts_only_epoch25_parent_and_weight(tmp_path, weight):
    payload, path, config, identity = fixture_payload(tmp_path)
    payload["epoch"] = 25
    payload["update_index"] = 6425
    torch.save(payload, path)
    config["objectives"]["goal_v_weight"] = 1.0
    config["objectives"]["fixed_anchor_path_weight"] = weight
    identity["config_hash"] = config_hash(config)
    receipt = validate_objective_fork_payload(
        payload, path, file_sha256(path), config, identity, "none",
        "fixed_anchor_path_weight")
    assert receipt["origin_epoch"] == 25
    assert receipt["resume_start_epoch"] == 26
    assert receipt["checked_config_diff"]["objectives.fixed_anchor_path_weight"] == {
        "old": "<MISSING>", "new": weight}


def test_fixed_anchor_path_fork_rejects_wrong_weight_or_parent(tmp_path):
    payload, path, config, identity = fixture_payload(tmp_path)
    config["objectives"]["goal_v_weight"] = 1.0
    config["objectives"]["fixed_anchor_path_weight"] = 0.2
    identity["config_hash"] = config_hash(config)
    with pytest.raises(ValueError, match="epoch25/update6425"):
        validate_objective_fork_payload(
            payload, path, file_sha256(path), config, identity, "none",
            "fixed_anchor_path_weight")
    payload["epoch"], payload["update_index"] = 25, 6425
    torch.save(payload, path)
    with pytest.raises(ValueError, match="missing->0"):
        validate_objective_fork_payload(
            payload, path, file_sha256(path), config, identity, "none",
            "fixed_anchor_path_weight")


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("training", "new_module_lr", 5e-5),
        ("objectives", "rollout_joint_ade_weight", 0.5),
        ("evaluation", "worlds", 8),
        ("diffusion", "training_noise_bins", 50),
    ],
)
def test_goal_weight_fork_rejects_lr_objective_world_or_step_changes(
    tmp_path, section, key, value,
):
    payload, path, config, identity = fixture_payload(tmp_path)
    config.setdefault(section, {})[key] = value
    identity["config_hash"] = config_hash(config)
    with pytest.raises(ValueError, match="frozen contract"):
        validate_objective_fork_payload(
            payload, path, file_sha256(path), config, identity, "none",
            "goal_v_weight_2")

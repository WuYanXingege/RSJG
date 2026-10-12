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
        "training": {"epochs": 100},
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

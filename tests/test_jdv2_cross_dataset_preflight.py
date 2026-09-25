"""Cross-dataset JDV2 preflight and provenance contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from src.models.joint_dependency_v2.dependency_corrector import (
    DependencyCorrector,
)
from src.parser import check_and_add_additional_args, get_parser


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("hotel", "univ", "zara1", "zara2")
RESULTS = ROOT / (
    "outputs/joint_dependency_v2/cross_dataset_preflight/results.json")
ETH_STAGE_A = ROOT / "configs/joint_dependency_v2/jdv2_stage_a_frozen_eth.yaml"
ETH_STAGE_B = ROOT / "configs/joint_dependency_v2/jdv2_stage_b_v2a_eth.yaml"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_config(path: Path):
    saved = _load_yaml(path)
    parser = get_parser()
    known = set(vars(parser.parse_args([])))
    assert not (set(saved) - known)
    parser.set_defaults(**saved)
    args = parser.parse_args([])
    args.device = "cpu"
    args.data_augmentation = False
    return check_and_add_additional_args(args)


@pytest.mark.parametrize("target", TARGETS)
def test_stage_a_templates_preserve_frozen_scientific_contract(target):
    args = _parse_config(
        ROOT / f"configs/joint_dependency_v2/jdv2_stage_a_{target}.yaml")
    frozen = _load_yaml(ETH_STAGE_A)

    assert args.cross_dataset_protocol_target == target
    assert args.dataset == "eth5" and args.test_set == target
    assert args.training_stage == "joint_goal"
    assert args.jdv2_latent_objective == "strict_no_z"
    assert args.use_scene_latent is False
    assert (args.num_goal_candidates, args.num_samples) == (21, 20)
    assert (args.jdv2_relation_modes, args.jdv2_energy_rank) == (4, 8)
    assert args.graph_type == "radius_ttc"
    assert (args.graph_radius, args.ttc_threshold, args.trajectory_dt) == (
        6.0, 8.0, 0.4)
    assert args.jdv2_refinement_policy == \
        "exact_lexicographic_persistent_tie"
    assert args.num_refinement_steps == 2
    assert args.amp_enabled and args.amp_dtype == "bf16"
    assert (args.seed, args.validation_seed, args.num_test_runs) == (
        2035, 2035, 5)

    immutable = {
        "num_goal_candidates",
        "num_samples", "jdv2_relation_modes", "jdv2_energy_rank",
        "graph_type", "graph_radius", "ttc_threshold", "trajectory_dt",
        "jdv2_refinement_policy", "num_refinement_steps", "amp_enabled",
        "amp_dtype",
    }
    config = _load_yaml(
        ROOT / f"configs/joint_dependency_v2/jdv2_stage_a_{target}.yaml")
    assert {key: config[key] for key in immutable} == {
        key: frozen[key] for key in immutable}
    # The freeze config is evaluation-only. These values are pinned from the
    # authoritative ETH strict-no-z formal-training command/config instead.
    assert {
        key: config[key] for key in (
            "learning_rate", "optimizer", "scheduler", "num_epochs",
            "start_validation", "validate_every",
            "early_stopping_patience", "lambda_diff",
            "lambda_relative")
    } == {
        "learning_rate": 1e-4,
        "optimizer": "Adam",
        "scheduler": "ExponentialLR",
        "num_epochs": 300,
        "start_validation": 1,
        "validate_every": 1,
        "early_stopping_patience": 12,
        "lambda_diff": 1.0,
        "lambda_relative": 0.05,
    }


@pytest.mark.parametrize("target", TARGETS)
def test_stage_b_templates_preserve_v2a_contract_and_fail_closed(target):
    path = ROOT / f"configs/joint_dependency_v2/jdv2_stage_b_v2a_{target}.yaml"
    config = _load_yaml(path)
    eth = _load_yaml(ETH_STAGE_B)

    assert config["training_stage"] == "joint_trajectory"
    assert config["stage_b_architecture_version"] == "jdv2-stage-b-v2a"
    assert config["jdv2_residual_projection"] == "component_zero_mean"
    assert config["stage_a_parent_checkpoint_sha256"] is None
    assert config["stage_a_freeze_manifest_sha256"] is None
    assert config["stage_a_freeze_source_commit"] is None
    assert target in config["stage_a_freeze_manifest_path"]
    assert target in config["pretrain_path"]
    with pytest.raises(ValueError, match="Stage-B remains unresolved"):
        _parse_config(path)

    immutable = {
        "learning_rate", "optimizer", "scheduler", "num_epochs",
        "start_validation", "validate_every", "early_stopping_patience",
        "lambda_diff", "lambda_relative", "num_goal_candidates",
        "num_samples", "jdv2_relation_modes", "jdv2_energy_rank",
        "graph_type", "graph_radius", "ttc_threshold", "trajectory_dt",
        "jdv2_refinement_policy", "num_refinement_steps", "amp_enabled",
        "amp_dtype", "stage_b_architecture_version",
        "jdv2_residual_projection",
    }
    assert {key: config[key] for key in immutable} == {
        key: eth[key] for key in immutable}


@pytest.mark.parametrize("target", TARGETS)
def test_cross_dataset_paths_are_target_isolated(target):
    config = _load_yaml(
        ROOT / f"configs/joint_dependency_v2/jdv2_stage_a_{target}.yaml")
    parser = get_parser()
    defaults = vars(parser.parse_args([]))
    defaults.update(config)
    defaults["jdv2_cache_root"] = (
        "outputs/joint_dependency_v2/cache/eth_full_stage_a")
    with pytest.raises(ValueError, match="target-isolated"):
        check_and_add_additional_args(argparse.Namespace(**defaults))


def test_cross_dataset_target_identity_rejects_swapped_protocol():
    config = _load_yaml(
        ROOT / "configs/joint_dependency_v2/jdv2_stage_a_univ.yaml")
    parser = get_parser()
    defaults = vars(parser.parse_args([]))
    defaults.update(config)
    defaults["test_set"] = "hotel"
    with pytest.raises(ValueError, match="must equal"):
        check_and_add_additional_args(argparse.Namespace(**defaults))


def test_results_pin_explicit_protocol_and_provenance_contracts():
    results = json.loads(RESULTS.read_text())
    assert results["status"] == "CROSS_DATASET_BENCHMARK_BLOCKED"
    assert results["next_state"] is None
    assert set(results["targets"]) == set(TARGETS)
    assert results["evaluation_seeds"] == [2035, 2036, 2037, 2038, 2039]
    assert results["scientific_contract"]["stage_a"] == {
        "K": 21,
        "M": 4,
        "P": 20,
        "dt_s": 0.4,
        "energy_rank": 8,
        "graph_type": "radius_ttc",
        "radius_m": 6.0,
        "refinement_policy": "exact_lexicographic_persistent_tie",
        "refinement_rounds": 2,
        "round0": "weighted_gumbel_top_p_without_replacement",
        "ttc_s": 8.0,
        "variant": "strict_no_z",
    }
    for target, entry in results["targets"].items():
        assert entry["protocol_classification"] == \
            "ORIGINAL_GDTS_MIRRORED_VAL_TEST"
        assert entry["raw_valid_test_byte_identical"] is True
        assert entry["ready_for_stage_a"] is False
        assert entry["configs"]["stage_a"]["sha256"] == _sha256(
            ROOT / entry["configs"]["stage_a"]["path"])
        assert entry["configs"]["stage_b_v2a"]["sha256"] == _sha256(
            ROOT / entry["configs"]["stage_b_v2a"]["path"])
        checkpoint = entry["gdts_checkpoint"].get("path")
        if checkpoint is not None:
            assert "/eth/" not in checkpoint.lower()
            assert target in checkpoint.lower()


def test_corrector_parameter_count_and_protected_eth_hashes_are_pinned():
    corrector = DependencyCorrector()
    assert sum(parameter.numel() for parameter in corrector.parameters()) == \
        30851
    results = json.loads(RESULTS.read_text())
    assert results["protected_eth_hashes"] == {
        "component_projection_source":
            "355de2a9ace3265efe1d5fc1ed4b5a6c054a621fc452a7ad95d70b6a2c3b5630",
        "dependency_corrector_source":
            "0ff28eb7107e49a0ad6f2bf339a65d8847f83c4a7cf315faa5a80cac4dc00522",
        "stage_a":
            "699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb",
        "stage_b_v1":
            "e24a1cbfc5760db57bcb5cacd64c66a4afd1aed0f11bb1efefa8549933d21c1a",
        "stage_b_v2a":
            "e4c114c729ba8d75ac72fc7d41f0f05e790cf2aec7b5cade563ba792930c7b73",
    }

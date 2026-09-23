"""Regression contract for the frozen JDV2 Stage-A method."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import yaml

import src.models.joint_dependency_v2.exact_lexicographic_assignment as exact
from src.models.joint_dependency_v2.joint_sampler import (
    REFINEMENT_POLICIES,
    ParallelConditionalSampler,
    _categorical,
    structured_gumbel_assignment,
)
from src.parser import check_and_add_additional_args, get_parser


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/joint_dependency_v2/jdv2_stage_a_frozen_eth.yaml"
MANIFEST = ROOT / (
    "outputs/joint_dependency_v2/eth/joint_dependency_v2/"
    "stage_a_freeze/manifest.json")
ADOPTION = ROOT / (
    "outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/"
    "exact_persistent_refinement_production_validation/results.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_frozen_args():
    saved = yaml.safe_load(CONFIG.read_text())
    parser = get_parser()
    known = set(vars(check_and_add_additional_args(parser.parse_args([
        "--device", "cpu", "--data_augmentation", "False"]))))
    assert not (set(saved) - known)
    parser.set_defaults(**saved)
    args = parser.parse_args([])
    args.device = "cpu"
    return check_and_add_additional_args(args)


def test_frozen_config_loads_and_locks_canonical_contract():
    args = _load_frozen_args()
    assert args.goal_model_type == "joint_dependency_v2"
    assert args.training_stage == "joint_goal"
    assert args.jdv2_latent_objective == "strict_no_z"
    assert args.use_scene_latent is False
    assert args.use_dynamic_relation and args.use_joint_energy
    assert args.jdv2_refinement_policy == \
        "exact_lexicographic_persistent_tie"
    assert (args.num_goal_candidates, args.num_samples) == (21, 20)
    assert (args.jdv2_relation_modes, args.jdv2_energy_rank) == (4, 8)
    assert args.num_refinement_steps == 2
    assert args.best_metric == "JFDE"
    assert args.validation_seed == 2035 and args.num_test_runs == 5


def test_manifest_hashes_and_authoritative_evidence_are_pinned():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["status"] == "STAGE_A_FROZEN"
    assert manifest["freeze_source_commit"] == \
        "ad35a440695a4b4eb23ad3e900eb27af9b3b8909"
    assert _sha256(CONFIG) == manifest["canonical_config"]["sha256"]
    assert _sha256(ADOPTION) == \
        manifest["production_adoption_result"]["sha256"]
    adoption = json.loads(ADOPTION.read_text())
    assert adoption["status"] == "ADOPTION_VALIDATION_PASSED"
    assert adoption["adoption_gate"]["status"] == \
        "ADOPTION_VALIDATION_PASSED"
    assert adoption["adoption_gate"]["production_means"] == {
        "JADE": 0.4131992939350416,
        "JFDE": 0.7021253722431741,
        "Joint_Goal_Compatibility": 0.48730886015210223,
        "Joint_Goal_Endpoint_Error": 0.689827065159091,
        "Relative_Motion_Error": 0.30181329901055465,
        "minADE@K": 0.27551551555769277,
        "minFDE@K": 0.38264789698802376,
    }


def test_checkpoint_hash_and_strict_no_z_state_contract_when_available():
    manifest = json.loads(MANIFEST.read_text())
    reference = manifest["checkpoint"]
    assert reference["epoch"] == 13
    assert reference["sha256"] == \
        "699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb"
    checkpoint_path = ROOT / reference["path"]
    if not checkpoint_path.exists():
        return  # Large checkpoint is an external artifact, not committed.
    assert _sha256(checkpoint_path) == reference["sha256"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["epoch"] == 13
    architecture = checkpoint["architecture_config"]
    assert architecture["architecture_variant"] == "strict_no_z"
    assert architecture["scene_modes"] == 0
    assert architecture["relation_modes"] == 4
    assert architecture["goal_candidates"] == 21
    assert architecture["joint_samples"] == 20
    assert architecture["energy_rank"] == 8
    keys = tuple(checkpoint["model_state_dict"])
    assert not any("jdv2_scene_prior" in key for key in keys)
    assert not any("jdv2_joint_energy.scene_embedding" in key for key in keys)


def test_global_default_and_alternative_paths_remain_available():
    assert get_parser().parse_args([]).jdv2_refinement_policy == "categorical"
    assert set(REFINEMENT_POLICIES) == {
        "categorical", "structured_gumbel_assignment",
        "exact_lexicographic_persistent_tie"}
    assert ParallelConditionalSampler(
        strict_no_z=True).refinement_policy == "categorical"

    score = torch.tensor([[[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]]])
    mask = torch.ones_like(score, dtype=torch.bool)
    categorical = _categorical(
        score, mask, 1.0, "map", generator=None)
    assert categorical.tolist() == [[0, 2]]
    cpsr = structured_gumbel_assignment(
        score, mask, 1.0, torch.Generator().manual_seed(7))
    assert cpsr.shape == (1, 2)
    assert torch.unique(cpsr).numel() == 2


def test_degree_zero_identity_and_exact_solver_are_callable(monkeypatch):
    score = torch.tensor([[[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]]])
    mask = torch.ones_like(score, dtype=torch.bool)
    previous = torch.tensor([[0, 2]])
    goals = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    called = []
    monkeypatch.setattr(
        exact, "solve_exact_persistent_tie",
        lambda *args, **kwargs: called.append(True))
    selected = exact.exact_persistent_refinement(
        score, mask, previous, goals, torch.tensor([0]),
        evaluation_seed=2035, window_index=0)
    assert torch.equal(selected, previous)
    assert called == []

    assignment, objective = exact.exact_max_weight_assignment(
        ((3, 0, 1), (0, 4, 1)),
        ((True, True, True), (True, True, True)))
    assert assignment == (0, 1)
    assert objective == 7


def test_all_off_configuration_preserves_legacy_baseline_semantics():
    parser = get_parser()
    args = parser.parse_args([
        "--device", "cpu", "--goal_model_type", "joint_dependency_v2",
        "--training_stage", "baseline", "--use_scene_latent", "False",
        "--use_dynamic_relation", "False", "--use_joint_energy", "False",
        "--use_dependency_corrector", "False"])
    args = check_and_add_additional_args(args)
    assert args.jdv2_active is False
    assert args.joint_goal_enabled is False
    assert args.training_stage == "baseline"

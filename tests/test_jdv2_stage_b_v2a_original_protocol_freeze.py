"""Regression contract for the original-GDTS-protocol V2-A freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import yaml

from src.joint_dependency_v2_cache import stable_json_hash
from src.models.joint_dependency_v2 import (
    DependencyCorrector,
    build_component_metadata,
    component_zero_mean_projection,
)
from src.models.model import jdv2_active_corrector_timesteps


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / (
    "outputs/joint_dependency_v2/eth/joint_dependency_v2/"
    "stage_b_v2a_original_protocol_freeze/manifest.json")
MANIFEST_SHA = MANIFEST.with_suffix(".sha256")
STAGE_A_CONFIG = ROOT / (
    "configs/joint_dependency_v2/jdv2_stage_a_frozen_eth.yaml")
V2A_CONFIG = ROOT / (
    "configs/joint_dependency_v2/jdv2_stage_b_v2a_eth.yaml")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def test_manifest_integrity_status_and_protocol_are_pinned():
    manifest = _manifest()
    assert manifest["schema"] == \
        "jdv2-stage-b-v2a-original-protocol-freeze-v1"
    assert manifest["status"] == \
        "STAGE_B_V2A_ORIGINAL_GDTS_PROTOCOL_FROZEN"
    assert manifest["evaluation_protocol"] == \
        "original_gdts_eth_ucy_mirrored_val_test"
    assert manifest["scientific_evidence_commit"] == \
        "817cc7551bd7f3422422230821acf8b3dd5cdc6c"
    assert manifest["adoption_base_commit"] == \
        "fb767ee87f01342a073198a8c8c1959460d496a7"
    payload = {key: value for key, value in manifest.items()
               if key != "manifest_hash"}
    assert stable_json_hash(payload) == manifest["manifest_hash"]
    assert MANIFEST_SHA.read_text().strip().split()[0] == _sha256(MANIFEST)
    assert manifest["v2b_authorized"] is False
    assert manifest["clean_heldout_result_available"] is False
    assert manifest["protocol_claims"] == {
        "fair_relative_comparison": True,
        "strict_heldout_generalization_established": False,
    }


def test_protected_checkpoint_hashes_are_pinned_and_verified_when_available():
    expected = {
        "stage_a": "699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb",
        "stage_b_v1": "e24a1cbfc5760db57bcb5cacd64c66a4afd1aed0f11bb1efefa8549933d21c1a",
        "stage_b_v2a": "e4c114c729ba8d75ac72fc7d41f0f05e790cf2aec7b5cade563ba792930c7b73",
    }
    manifest = _manifest()
    for name, digest in expected.items():
        artifact = manifest["checkpoints"][name]
        assert artifact["sha256"] == digest
        path = ROOT / artifact["path"]
        if path.exists():  # Large checkpoints are external to source control.
            assert _sha256(path) == digest


def test_source_parameter_and_projection_contracts_are_pinned():
    manifest = _manifest()
    sources = manifest["source_hashes"]
    for record in sources.values():
        assert _sha256(ROOT / record["path"]) == record["sha256"]
    assert sources["dependency_corrector"]["sha256"] == \
        "0ff28eb7107e49a0ad6f2bf339a65d8847f83c4a7cf315faa5a80cac4dc00522"
    assert sources["component_projection"]["sha256"] == \
        "355de2a9ace3265efe1d5fc1ed4b5a6c054a621fc452a7ad95d70b6a2c3b5630"
    assert sum(parameter.numel()
               for parameter in DependencyCorrector().parameters()) == 30851
    assert manifest["method"]["projection"]["parameters"] == 0


def test_canonical_configs_pin_original_protocol_and_frozen_method():
    manifest = _manifest()
    stage_a = yaml.safe_load(STAGE_A_CONFIG.read_text())
    v2a = yaml.safe_load(V2A_CONFIG.read_text())
    assert _sha256(STAGE_A_CONFIG) == \
        manifest["canonical_configs"]["stage_a"]["sha256"]
    assert _sha256(V2A_CONFIG) == \
        manifest["canonical_configs"]["stage_b_v2a"]["sha256"]
    assert v2a["model_selection_split"] == "dataset_valid"
    assert v2a["stage_b_architecture_version"] == "jdv2-stage-b-v2a"
    assert v2a["jdv2_residual_projection"] == "component_zero_mean"
    assert stage_a["jdv2_latent_objective"] == "strict_no_z"
    assert stage_a["use_scene_latent"] is False
    for config in (stage_a, v2a):
        assert config["jdv2_refinement_policy"] == \
            "exact_lexicographic_persistent_tie"
        assert (config["num_goal_candidates"], config["num_samples"],
                config["num_relation_modes"], config["goal_pair_rank"]) == \
            (21, 20, 4, 8)
        assert config["graph_type"] == "radius_ttc"
        assert (config["graph_radius"], config["ttc_threshold"],
                config["trajectory_dt"]) == (6.0, 8.0, 0.4)
    assert (v2a["lambda_diff"], v2a["lambda_relative"]) == (1.0, 0.05)
    assert (v2a["optimizer"], v2a["learning_rate"],
            v2a["amp_dtype"]) == ("Adam", 0.0001, "bf16")
    assert jdv2_active_corrector_timesteps(100, 20, 14) == \
        (30, 25, 20, 15, 10, 5)


def test_e0_and_mixed_degree_zero_projection_identity_contract():
    raw = torch.tensor([
        [[1.0, -2.0]], [[3.0, 4.0]], [[7.0, -5.0]]])
    scene = torch.zeros(3, dtype=torch.long)

    e0 = build_component_metadata(
        torch.empty(2, 0, dtype=torch.long), 3, scene)
    assert torch.equal(
        component_zero_mean_projection(raw, e0), torch.zeros_like(raw))

    mixed = build_component_metadata(
        torch.tensor([[0], [1]], dtype=torch.long), 3, scene)
    projected = component_zero_mean_projection(raw, mixed)
    assert torch.equal(projected[2], torch.zeros_like(projected[2]))
    assert torch.equal(projected[:2].sum(dim=0),
                       torch.zeros_like(projected[0]))
    assert mixed.active_mask.tolist() == [True, True, False]


def test_adopted_metrics_and_non_authoritative_clean_run_are_pinned():
    manifest = _manifest()
    comparisons = manifest["metrics"]
    assert comparisons["relative_percent_v2a_vs_v1"] == {
        "JADE": -0.267,
        "JFDE": -0.783,
        "Relative_Motion_Error": -0.146,
        "minADE": -0.936,
        "minFDE": -0.986,
    }
    assert comparisons["v1_gap_recovery_percent"] == {
        "minADE": 59.92, "minFDE": 79.38}
    clean = manifest["interrupted_clean_stage_a"]
    assert clean["status"] == "ABORTED_NON_AUTHORITATIVE"
    assert clean["authoritative"] is False
    assert clean["completed_epochs"] == 1

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RESULTS = (
    ROOT
    / "outputs/joint_dependency_v2/univ/joint_dependency_v2/"
    "stage_a_readiness/results.json"
)
CROSS_RESULTS = (
    ROOT / "outputs/joint_dependency_v2/cross_dataset_preflight/results.json"
)
SOURCE_COMMIT = "db47b913820371cca25876d08c624709830ae1e8"
CHECKPOINT_SHA256 = (
    "ebfbae9de25463497c7bcc20981022f0638bc0349c7eeeb38bfc235dfaf61a3e"
)
CACHE_HASH = (
    "7d4983630a2665f6856afeba7265a822e5bd87cc3f1c856485d644ca6a718008"
)


def _load_results():
    return json.loads(RESULTS.read_text())


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_univ_readiness_state_is_narrow_and_training_free():
    results = _load_results()
    assert results["status"] == "UNIV_READY_FOR_STAGE_A"
    assert results["next_state"] == "eligible_for_univ_stage_a_reproduction"
    assert results["target"] == "univ"
    assert results["source_commit"] == SOURCE_COMMIT
    assert results["training_performed"] is False
    assert results["benchmark_evaluation_performed"] is False
    assert all(results["gates"].values())


def test_univ_checkpoint_and_caches_are_authenticated():
    results = _load_results()
    checkpoint = results["gdts_checkpoint"]
    assert checkpoint["sha256"] == CHECKPOINT_SHA256
    assert checkpoint["architecture_compatible"] is True
    assert checkpoint["epoch"] == 100

    source = results["source_cache"]
    jdv2 = results["jdv2_cache"]
    expected_counts = {"train": 3302, "valid": 947, "test": 947}
    assert {
        split: source["splits"][split]["count"] for split in expected_counts
    } == expected_counts
    assert {
        split: jdv2["splits"][split]["count"] for split in expected_counts
    } == expected_counts
    assert jdv2["manifest_stable_hash"] == CACHE_HASH
    assert jdv2["manifest"]["goal_checkpoint_hash"] == CHECKPOINT_SHA256
    assert jdv2["manifest"]["source_commit"] == SOURCE_COMMIT
    assert jdv2["manifest"]["K"] == 21
    assert jdv2["manifest"]["dt"] == 0.4
    assert jdv2["manifest"]["coordinate_units"] == {
        "position": "world_m",
        "velocity": "world_m_per_s",
    }
    assert jdv2["manifest"]["graph_config"] == {
        "adaptive": False,
        "radius": 6.0,
        "ttc_threshold": 8.0,
        "type": "radius_ttc",
    }
    for split, count in expected_counts.items():
        entry = jdv2["splits"][split]
        assert entry["teacher_sidecar_count"] == count
        assert entry["deployment_records_future_free"] is True
        assert entry["teacher_sidecars_separate"] is True
        assert entry["all_records_finite"] is True
        assert entry["candidate_shapes_valid"] is True
        assert entry["canonical_edges"] is True
        assert entry["no_cross_scene_edges"] is True


def test_univ_cuda_bf16_smoke_covers_e0_and_interacting_paths():
    smoke = _load_results()["cuda_bf16_no_optimizer_smoke"]
    assert smoke["status"] == "PASS"
    assert smoke["optimizer_constructed"] is False
    assert smoke["baseline_gdts_frozen"] is True
    assert smoke["baseline_state_unchanged"] is True
    assert smoke["dependency_corrector_trainable"] is False
    assert smoke["dependency_corrector_state_unchanged"] is True
    assert smoke["strict_no_z"] is True
    assert smoke["refinement_policy"] == \
        "exact_lexicographic_persistent_tie"
    assert smoke["windows"]["e0"]["edge_count"] == 0
    assert smoke["windows"]["e_gt0"]["edge_count"] > 0
    for window in smoke["windows"].values():
        assert window["finite"] is True
        assert window["no_cross_scene_edges"] is True
        assert set(window["coverage_unique_per_agent"]) == {20}
        assert window["prediction_shape"][0:2] == [20, 20]


def test_univ_configs_pin_cache_and_keep_stage_b_fail_closed():
    results = _load_results()
    for stage in ("stage_a", "stage_b_v2a"):
        entry = results["config"][stage]
        path = ROOT / entry["path"]
        config = yaml.safe_load(path.read_text())
        assert _sha256(path) == entry["sha256"]
        assert config["jdv2_cache_source_commit"] == SOURCE_COMMIT
        assert config["jdv2_cache_manifest_hash"] == CACHE_HASH

    stage_b = yaml.safe_load(
        (ROOT / results["config"]["stage_b_v2a"]["path"]).read_text()
    )
    assert stage_b["stage_a_parent_checkpoint_sha256"] is None
    assert stage_b["stage_a_freeze_source_commit"] is None
    assert stage_b["stage_a_freeze_manifest_sha256"] is None


def test_cross_dataset_result_promotes_only_univ():
    cross = json.loads(CROSS_RESULTS.read_text())
    assert cross["status"] == "CROSS_DATASET_BENCHMARK_PARTIALLY_READY"
    assert cross["next_state"] == "eligible_for_univ_stage_a_reproduction"
    for target, entry in cross["targets"].items():
        if target == "univ":
            assert entry["status"] == "TARGET_READY_FOR_STAGE_A"
            assert entry["ready_for_stage_a"] is True
            assert entry["jdv2_stage_a_cache"]["manifest_hash"] == CACHE_HASH
            assert entry["source_batch_cache"]["window_counts"] == {
                "train": 3302,
                "valid": 947,
                "test": 947,
            }
        else:
            assert entry["status"] == \
                "TARGET_BLOCKED_MISSING_GDTS_CHECKPOINT"
            assert entry["ready_for_stage_a"] is False

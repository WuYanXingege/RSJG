import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.jmm_protocol import (
    JMMWindow,
    JMM_ETH_EXPECTED_AGENT_INSTANCES,
    JMM_ETH_EXPECTED_WINDOWS,
    JMM_NUM_SAMPLES,
    build_jmm_eth_windows,
    score_standardized_trajectories,
    summarize_windows,
    write_standardized_window,
)
from tools.evaluate_jmm_official import (
    _collate_cache_record,
    _load_validated_jdv2_cache,
    _record_set_hash,
    _validate_jdv2_cache_record,
    build_parser,
)



def _agentformer_row(frame, agent, x, y):
    row = ["-1"] * 17
    row[0] = str(frame)
    row[1] = str(agent)
    row[2] = "Pedestrian"
    row[13] = str(x)
    row[15] = str(y)
    return " ".join(row)


def test_window_builder_matches_agentformer_complete_track_rule(tmp_path):
    source = tmp_path / "eth.txt"
    rows = []
    # Agent 1 exists for both possible 20-step windows. Agent 2 enters at
    # frame 1 and is therefore included only in the second window.
    for frame in range(21):
        rows.append(_agentformer_row(frame, 1, frame, 0.0))
        if frame >= 1:
            rows.append(_agentformer_row(frame, 2, frame, 2.0))
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")

    windows = build_jmm_eth_windows(source, strict_official=False)

    assert len(windows) == 2
    assert [window.last_observation_frame for window in windows] == [70, 80]
    assert windows[0].agent_ids.tolist() == [1]
    assert windows[1].agent_ids.tolist() == [1, 2]
    assert windows[1].world_coordinates.shape == (20, 2, 2)
    assert summarize_windows(windows) == {
        "num_scenes": 2,
        "num_agent_instances": 3,
        "mean_pedestrians_per_scene": 1.5,
        "pedestrian_count_distribution": {"1": 1, "2": 1},
    }


def test_strict_builder_rejects_a_dataset_that_is_not_official(tmp_path):
    source = tmp_path / "eth.txt"
    source.write_text(
        "\n".join(_agentformer_row(frame, 1, frame, 0) for frame in range(20)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not the official JMM ETH"):
        build_jmm_eth_windows(source, strict_official=True)


def test_standard_export_and_score_keep_one_joint_sample(tmp_path):
    frames = np.arange(20, dtype=np.int64) * 10
    agents = np.array([1, 2], dtype=np.int64)
    ground_truth = np.zeros((20, 2, 2), dtype=np.float64)
    window = JMMWindow("biwi_eth", frames, agents, ground_truth)

    predictions = np.zeros((JMM_NUM_SAMPLES, 12, 2, 2), dtype=np.float64)
    # No common sample is perfect: sample 0 misses agent 2, while sample 1
    # misses agent 1. Marginal minADE/minFDE may mix them and becomes zero;
    # joint metrics must use one sample and therefore equal five metres.
    predictions[0, :, 1, 0] = 10.0
    predictions[1, :, 0, 0] = 10.0
    predictions[2:] = predictions[0]
    write_standardized_window(tmp_path, window, predictions)

    metrics = score_standardized_trajectories(
        tmp_path, strict_official=False
    )

    assert metrics["minJADE@20"] == pytest.approx(5.0)
    assert metrics["minJFDE@20"] == pytest.approx(5.0)
    assert metrics["minADE@20"] == pytest.approx(0.0)
    assert metrics["minFDE@20"] == pytest.approx(0.0)
    assert metrics["num_scenes"] == 1
    assert metrics["num_agent_instances"] == 2


def test_checked_in_official_data_has_paper_protocol_counts():
    source = (
        Path(__file__).resolve().parents[1]
        / "data" / "jmm_official" / "eth" / "biwi_eth_agentformer.txt"
    )
    if not source.is_file():
        pytest.skip("official JMM data has not been prepared")

    windows = build_jmm_eth_windows(source, strict_official=True)
    summary = summarize_windows(windows)
    assert summary["num_scenes"] == JMM_ETH_EXPECTED_WINDOWS
    assert summary["num_agent_instances"] == JMM_ETH_EXPECTED_AGENT_INSTANCES


def _deployment_record(window, scene_index=0, num_candidates=21):
    num_agents = window.num_agents
    return {
        "cache_id": f"jmm-eth-{scene_index:06d}",
        "window_directory": window.directory_name,
        "num_goal_candidates": num_candidates,
        "scene_index": torch.zeros(num_agents, dtype=torch.long),
        "scene_ptr": torch.tensor([0, num_agents], dtype=torch.long),
        "frame_ids": torch.from_numpy(
            np.repeat(window.frame_ids[:, None], num_agents, axis=1)
        ).long(),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
        "edge_feat": torch.empty((0, 8)),
        "edge_weight": torch.empty((0,)),
        "goal_candidates_map": torch.zeros(num_agents, num_candidates, 2),
        "goal_candidates_world": torch.zeros(num_agents, num_candidates, 2),
        "candidate_log_prior": torch.zeros(num_agents, num_candidates),
        "contains_future_supervision": False,
    }


def test_jmm_jdv2_deployment_record_contract_and_single_agent_collation():
    window = JMMWindow(
        "biwi_eth",
        np.arange(20, dtype=np.int64) * 10,
        np.array([7], dtype=np.int64),
        np.zeros((20, 1, 2), dtype=np.float64),
    )
    record = _deployment_record(window)
    _validate_jdv2_cache_record(record, window, 0)

    collated = _collate_cache_record(record)
    assert collated["goal_candidates_world"].shape == (1, 1, 21, 2)
    assert collated["scene_index"].shape == (1, 1)
    assert collated["edge_index"].shape == (1, 2, 0)
    assert collated["contains_future_supervision"] is False


def test_jmm_jdv2_record_rejects_future_or_window_mismatch():
    window = JMMWindow(
        "biwi_eth",
        np.arange(20, dtype=np.int64) * 10,
        np.array([1, 2], dtype=np.int64),
        np.zeros((20, 2, 2), dtype=np.float64),
    )
    future_record = _deployment_record(window)
    future_record["contains_future_supervision"] = True
    with pytest.raises(RuntimeError, match="future supervision"):
        _validate_jdv2_cache_record(future_record, window, 0)

    wrong_window = _deployment_record(window)
    wrong_window["frame_ids"][0, 0] += 10
    with pytest.raises(RuntimeError, match="frame_ids mismatch"):
        _validate_jdv2_cache_record(wrong_window, window, 0)


def test_jmm_jdv2_manifest_and_record_hash_are_fail_closed(tmp_path):
    cache_dir = tmp_path / "cache"
    records = cache_dir / "records"
    records.mkdir(parents=True)
    record_path = records / "000000.pt"
    torch.save({"cache_id": "jmm-eth-000000"}, record_path)
    identity = {
        "schema_version": "jmm-jdv2-deployment-cache-v1",
        "data_sha256": "data",
        "candidate_seed": 2035,
    }
    manifest = {
        **identity,
        "status": "complete",
        "record_set_sha256": _record_set_hash([record_path]),
    }
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    loaded = _load_validated_jdv2_cache(cache_dir, [object()], identity)
    assert loaded["record_set_sha256"] == manifest["record_set_sha256"]
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        _load_validated_jdv2_cache(
            cache_dir, [object()], {**identity, "candidate_seed": 2036}
        )
    record_path.write_bytes(record_path.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="record-set hash mismatch"):
        _load_validated_jdv2_cache(cache_dir, [object()], identity)


def test_jmm_cli_exposes_explicit_jdv2_cache_contract():
    parser = build_parser()
    build = parser.parse_args([
        "build-jdv2-cache", "--run-dir", "run", "--cache-dir", "cache"
    ])
    assert build.seed == 2035
    infer = parser.parse_args([
        "infer", "--run-dir", "run", "--checkpoint", "best",
        "--jdv2-cache-dir", "cache",
    ])
    assert infer.jdv2_cache_seed == 2035

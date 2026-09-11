from pathlib import Path

import numpy as np
import pytest

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

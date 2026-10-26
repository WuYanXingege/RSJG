import importlib.util
from pathlib import Path

import numpy as np


_SPEC = importlib.util.spec_from_file_location(
    "endpoint_bank_analysis",
    Path(__file__).parents[1] / "scripts" / "analyze_zara1_endpoint_banks.py")
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
paired = _MODULE.paired
summarize = _MODULE.summarize


def _bank(offsets):
    endpoints = np.zeros((2, 20, 2, 2), dtype=np.float32)
    truth = np.zeros((2, 2, 2), dtype=np.float32)
    valid = np.asarray([[True, False], [True, True]])
    endpoints[:, :, :, 0] = np.asarray(offsets)[:, None, :]
    endpoints[:, :, 1, 0] += np.linspace(0, 1, 20)[None]
    return {"endpoints": endpoints, "truth": truth, "valid": valid,
            "scene_ids": np.asarray(["a", "b"]),
            "agent_ids": np.asarray([["1", ""], ["2", "3"]])}


def test_summary_masks_padding_and_reports_tail() -> None:
    result = summarize(_bank([[0.1, 100.0], [0.2, 0.3]]))
    assert np.isclose(result["minFDE"], 0.2)
    assert result["agent_count"] == 3
    assert result["scene_count"] == 2
    assert result["Recall@0.25m"] == 2 / 3
    assert result["P95"] >= result["P90"] >= result["P50"]
    assert np.isclose(result["JFDE"], 0.175)


def test_paired_delta_sign_is_explicit() -> None:
    result = paired(np.asarray([1.0, 2.0]), np.asarray([2.0, 1.0]))
    assert result["mean_delta"] == 0.0
    assert result["left_lower_fraction"] == 0.5
    assert result["right_lower_fraction"] == 0.5

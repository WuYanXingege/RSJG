import torch

from tools.jdv2_canonical_route_certification import (
    _empty_parity,
    _update_parity,
    classify_status,
    tensor_comparison,
)


def _parity():
    return {
        pair: {scope: _empty_parity() for scope in (
            "all_agents", "active_agents", "inactive_agents")}
        for pair in ("A_vs_B", "B_vs_C", "A_vs_C")
    }


def test_tensor_comparison_detects_dtype_only_divergence():
    left = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    right = left.float()

    result = tensor_comparison(left, right)

    assert result["shape_equal"]
    assert not result["dtype_equal"]
    assert not result["tensor_equal"]
    assert result["max_abs_difference"] == 0.0
    assert result["different_elements"] == 0


def test_update_parity_accumulates_value_differences():
    parity = _empty_parity()
    _update_parity(
        parity, torch.tensor([1.0, 2.0]), torch.tensor([1.0, 3.0]))
    _update_parity(
        parity, torch.tensor([4.0]), torch.tensor([4.0]))

    assert parity == {
        "max_abs_difference": 1.0,
        "different_elements": 1,
        "compared_elements": 3,
        "nonidentical_windows": 1,
    }


def test_status_requires_all_pairing_invariants():
    parity = _parity()
    invariants = {"goals": True, "noise": False}

    assert classify_status(invariants, parity) == \
        "PAIRING_INPUT_CONTRACT_FAILED"


def test_status_requires_stage_b_reference_identity():
    parity = _parity()
    parity["B_vs_C"]["all_agents"]["different_elements"] = 1

    assert classify_status({"paired": True}, parity) == \
        "STAGE_B_REFERENCE_PATH_MISMATCH"


def test_status_distinguishes_zero_add_drift_from_literal_identity():
    parity = _parity()
    assert classify_status({"paired": True}, parity) == \
        "CANONICAL_STAGE_A_LITERAL_IDENTITY_CONFIRMED"

    parity["A_vs_B"]["all_agents"]["different_elements"] = 1
    assert classify_status({"paired": True}, parity) == \
        "CANONICAL_STAGE_A_ZERO_ADD_DRIFT_CONFIRMED"

from ebjd.evaluate import sample_stdev, valid_seed_receipt


def expected():
    return {
        "checkpoint_sha256": "checkpoint",
        "config_sha256": "config",
        "manifest_sha256": "manifest",
        "evaluator_sha256": "evaluator",
        "future_social_mode": "self_only",
        "seed": 2035,
        "worlds": 20,
        "steps": 20,
        "ddim_eta": 0.0,
        "scene_count": 947,
        "agent_occurrence_count": 24334,
    }


def receipt(fixed=False):
    return {
        "schema": "ebjd-evaluation-seed-receipt-v2",
        "status": "COMPLETE",
        **expected(),
        "raw_record": {"seed": 2035, "metrics": {}},
        "fixed_record": ({
            "seed": 2035,
            "metrics": {},
            "derived_from_same_forward_as_raw": True,
        } if fixed else None),
    }


def test_single_seed_standard_deviation_is_neutral():
    assert sample_stdev([1.25]) == 0.0


def test_multiple_seed_standard_deviation_uses_ddof_one():
    assert sample_stdev([1.0, 3.0]) == 2 ** 0.5


def test_raw_receipt_requires_exact_identity():
    value = receipt()
    assert valid_seed_receipt(value, expected(), require_fixed=False)
    value["checkpoint_sha256"] = "wrong"
    assert not valid_seed_receipt(value, expected(), require_fixed=False)


def test_fixed_receipt_requires_same_forward_marker():
    value = receipt(fixed=True)
    assert valid_seed_receipt(value, expected(), require_fixed=True)
    value["fixed_record"]["derived_from_same_forward_as_raw"] = False
    assert not valid_seed_receipt(value, expected(), require_fixed=True)


def test_legacy_receipt_identity_remains_readable():
    old_expected = expected()
    old_expected.pop("future_social_mode")
    value = receipt()
    value.pop("future_social_mode")
    value["schema"] = "ebjd-evaluation-seed-receipt-v1"
    assert valid_seed_receipt(value, old_expected, require_fixed=False)

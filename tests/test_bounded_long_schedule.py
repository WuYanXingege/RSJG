import importlib.util
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
_SPEC = importlib.util.spec_from_file_location(
    "run_zara1_gt_bridge_bounded_long",
    Path(__file__).parents[1] / "scripts" / "run_zara1_gt_bridge_bounded_long.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
cosine_factor = _MODULE.cosine_factor
eligible = _MODULE.eligible


def test_cosine_schedule_has_exact_registered_endpoints():
    assert cosine_factor(0, 30, 0.1) == 1.0
    assert cosine_factor(29, 30, 0.1) == 0.1
    assert cosine_factor(14, 30, 0.1) > cosine_factor(15, 30, 0.1)


def test_eligible_uses_nonstrict_marginal_and_strict_joint_contract():
    baseline = {"minADE": 1.0, "minFDE": 2.0, "JADE": 3.0, "JFDE": 4.0}
    assert eligible(
        {"minADE": 1.0, "minFDE": 2.0, "JADE": 2.9, "JFDE": 3.9}, baseline)
    assert not eligible(
        {"minADE": 1.0, "minFDE": 2.0, "JADE": 3.0, "JFDE": 3.9}, baseline)

"""Bounded synthetic loss certification runner; stdout JSON, no artifact writes."""
import contextlib
import io
import json
import os
import pathlib
import platform
import sys
import time

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("Run with CUDA_VISIBLE_DEVICES=''")
ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import pytest
import torch

torch.set_num_threads(1)
EVENTS = []
FORBIDDEN = {"module_forward_attempts": 0, "cuda_initialization_attempts": 0,
             "checkpoint_load_attempts": 0}


class Recorder:
    def pytest_runtest_logreport(self, report):
        EVENTS.append({"nodeid": report.nodeid, "phase": report.when,
                       "outcome": report.outcome, "seconds": report.duration,
                       "failure": str(report.longrepr) if report.failed else None})


def forbid_module(*args, **kwargs):
    FORBIDDEN["module_forward_attempts"] += 1
    raise AssertionError("No nn.Module forward is permitted in this CPU audit")


def forbid_cuda(*args, **kwargs):
    FORBIDDEN["cuda_initialization_attempts"] += 1
    raise AssertionError("No CUDA initialization is permitted")


def forbid_checkpoint(*args, **kwargs):
    FORBIDDEN["checkpoint_load_attempts"] += 1
    raise AssertionError("No checkpoint reads are permitted")


TESTS = [
    "tests/test_jdv2_expected_conditional.py",
    "tests/test_joint_dependency_v2.py::test_pseudo_likelihood_no_edges_reduces_to_unary_scene_balanced",
    "tests/test_joint_dependency_v2.py::test_scene_mode_score_and_mixture_match_manual_enumeration",
    "tests/test_joint_dependency_v2.py::test_mixture_and_posterior_distillation_gradient_ownership",
    "tests/test_joint_dependency_v2.py::test_relation_kl_and_scene_kl_are_finite_with_empty_edge",
    "tests/test_joint_dependency_v2.py::test_relation_kl_detaches_scene_responsibility_weight",
    "tests/test_joint_dependency_v2.py::test_kl_warmup",
    "tests/test_joint_dependency_v2.py::test_teacher_curriculum",
]
original = torch.nn.Module._call_impl, torch.cuda._lazy_init, torch.load
torch.nn.Module._call_impl, torch.cuda._lazy_init, torch.load = (
    forbid_module, forbid_cuda, forbid_checkpoint)
capture = io.StringIO()
started = time.monotonic()
try:
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        code = pytest.main(["-q", "-p", "no:cacheprovider", *TESTS], plugins=[Recorder()])
finally:
    torch.nn.Module._call_impl, torch.cuda._lazy_init, torch.load = original
elapsed = time.monotonic() - started
module = next((m for name, m in sys.modules.items()
               if name.endswith("test_jdv2_expected_conditional")), None)
calls = [e for e in EVENTS if e["phase"] == "call"]
result = {
    "status": "PASS" if code == 0 and not any(FORBIDDEN.values()) else "FAILED",
    "pytest_exit_code": int(code), "wall_seconds": elapsed,
    "environment": {"python": platform.python_version(), "torch": torch.__version__,
                    "pytest": pytest.__version__, "threads": torch.get_num_threads(),
                    "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
                    "torch_build_cuda": torch.version.cuda,
                    "note": "CUDA build version is metadata, not a runtime query"},
    "test_selectors": TESTS,
    "tests": {"attempted": len(calls), "completed": sum(e["outcome"] == "passed" for e in calls),
              "failed": sum(e["outcome"] == "failed" for e in EVENTS), "unknown": 0},
    "synthetic_loss_audit": None if module is None else module.AUDIT,
    "forbidden_operation_attempts": FORBIDDEN, "events": EVENTS,
    "pytest_stdout": capture.getvalue(),
}
print(json.dumps(result, indent=2, allow_nan=False))
sys.exit(int(code) if code else int(any(FORBIDDEN.values())))

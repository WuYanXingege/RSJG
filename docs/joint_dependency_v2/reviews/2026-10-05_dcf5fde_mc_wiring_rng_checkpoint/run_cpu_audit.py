"""Bounded CPU audit; emits JSON only, never opens real artifacts."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT/'tests')]
if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise RuntimeError('CUDA_VISIBLE_DEVICES must be empty')
import pytest
import torch
torch.set_num_threads(1)
from src.jdv2_objective_state import ObjectiveRNG
from src.models.model import GDTS

COUNTS = dict(module_forward=0, cuda_initialization=0, toy_load=0, toy_save=0,
              objective_draw_attempts=0, objective_draws=0, sampled_agent_draws=0,
              actual_no_z_dispatch_calls=0, actual_mc_method_calls=0,
              adam_steps=0, real_artifact_attempts=0)
EVENTS = []


def forbidden(name):
    def call(*a, **kw):
        COUNTS[name] += 1
        raise AssertionError('Forbidden in synthetic CPU audit: '+name)
    return call


def check_path(value):
    path = value if isinstance(value, (str, os.PathLike)) else getattr(value, 'name', value)
    if isinstance(path, int):  # fdopen from the actual atomic helper
        return
    if not str(Path(path).resolve()).startswith('/tmp/'):
        COUNTS['real_artifact_attempts'] += 1
        raise AssertionError('Only temporary toy checkpoints authorized')


original_save, original_load = torch.save, torch.load
def save(payload, path, *a, **kw):
    check_path(path); COUNTS['toy_save'] += 1
    return original_save(payload, path, *a, **kw)
def load(path, *a, **kw):
    check_path(path); COUNTS['toy_load'] += 1
    return original_load(path, *a, **kw)

original_draw = ObjectiveRNG.draw
def draw(self, *a, **kw):
    COUNTS['objective_draw_attempts'] += 1
    value = original_draw(self, *a, **kw)
    COUNTS['objective_draws'] += 1
    COUNTS['sampled_agent_draws'] += value.numel()
    return value


def counted(original, key):
    def call(*a, **kw):
        COUNTS[key] += 1
        return original(*a, **kw)
    return call


torch.save, torch.load = save, load
torch.cuda._lazy_init = forbidden('cuda_initialization')
torch.nn.Module._call_impl = forbidden('module_forward')
ObjectiveRNG.draw = draw
GDTS._jdv2_no_z_goal_losses = counted(GDTS._jdv2_no_z_goal_losses, 'actual_no_z_dispatch_calls')
GDTS._jdv2_mc_goal_losses = counted(GDTS._jdv2_mc_goal_losses, 'actual_mc_method_calls')
torch.optim.Adam.step = counted(torch.optim.Adam.step, 'adam_steps')


class Recorder:
    def pytest_runtest_logreport(self, report):
        EVENTS.append(dict(nodeid=report.nodeid, phase=report.when, outcome=report.outcome,
                           seconds=report.duration,
                           failure=str(report.longrepr)[-1800:] if report.failed else None))


selectors = sys.argv[1:] or [
    'tests/test_jdv2_mc_wiring.py',
    'tests/test_jdv2_resume_stopping_state.py',
    'tests/test_joint_dependency_v2_integration.py::test_validation_seed_defaults_to_training_seed',
    'tests/test_joint_dependency_v2_integration.py::test_cross_stage_load_resets_epoch_and_progress_while_same_stage_resumes',
]
output = io.StringIO()
started = time.monotonic()
with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
    code = pytest.main(['-q', '--tb=short', '-p', 'no:cacheprovider', *selectors], plugins=[Recorder()])
import mc_wiring_fixture
test = sys.modules.get('test_jdv2_mc_wiring')
print(json.dumps(dict(status='PASS' if code == 0 else 'FAILED', exit_code=int(code),
    wall_seconds=time.monotonic()-started, selectors=selectors, events=EVENTS,
    counts=COUNTS, fixture_counts=mc_wiring_fixture.COUNTS,
    measurements={} if test is None else test.MEASUREMENTS, stdout=output.getvalue()), indent=2))
sys.exit(code)

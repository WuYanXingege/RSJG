from __future__ import annotations

import random
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_zara1_unified_coupling_bridge.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("unified_runner", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
seed_all = MODULE.seed_all


def snapshot(seed: int) -> tuple[float, float, torch.Tensor]:
    seed_all(seed)
    return random.random(), float(np.random.rand()), torch.rand(4)


def test_training_seed_controls_all_primary_random_streams() -> None:
    a, b, c = snapshot(3102), snapshot(3102), snapshot(3103)
    assert a[0] == b[0] and a[1] == b[1] and torch.equal(a[2], b[2])
    assert a[0] != c[0] and a[1] != c[1] and not torch.equal(a[2], c[2])

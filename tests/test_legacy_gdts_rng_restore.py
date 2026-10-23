from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np
import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "run_legacy_gdts_stage.py"
SPEC = importlib.util.spec_from_file_location("run_legacy_gdts_stage", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Loader:
    def __init__(self) -> None:
        self.generator = torch.Generator().manual_seed(17)


def test_restore_rng_round_trip_uses_canonical_cpu_byte_tensors() -> None:
    loader = Loader()
    random.seed(11)
    np.random.seed(13)
    torch.manual_seed(19)
    saved = MODULE.rng_state(loader)

    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(3)
    expected_loader = torch.rand(3, generator=loader.generator)

    random.seed(101)
    np.random.seed(103)
    torch.manual_seed(107)
    loader.generator.manual_seed(109)
    MODULE.restore_rng(saved, loader)

    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    assert torch.equal(torch.rand(3), expected_torch)
    assert torch.equal(torch.rand(3, generator=loader.generator), expected_loader)


def test_rng_state_rejects_non_byte_tensor() -> None:
    with pytest.raises(RuntimeError, match="not uint8"):
        MODULE.cpu_byte_rng_state(torch.zeros(4, dtype=torch.int64), "bad")

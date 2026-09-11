from pathlib import Path

import pytest

from tools.generate_v4_report import (
    PROJECT_ROOT,
    _display_path,
    _resolve_validation_diagnostics,
)


def test_display_path_removes_machine_specific_project_root():
    checkpoint = PROJECT_ROOT / 'output' / 'eth' / 'saved_models' / 'best_model.pt'

    assert _display_path(checkpoint) == 'output/eth/saved_models/best_model.pt'


def test_resolve_validation_diagnostics_prefers_legacy_name(tmp_path: Path):
    diagnostics = tmp_path / 'diagnostics'
    diagnostics.mkdir()
    legacy = diagnostics / 'valid_epoch_003.json'
    seed_zero = diagnostics / 'valid_epoch_003_seed_00.json'
    legacy.touch()
    seed_zero.touch()

    assert _resolve_validation_diagnostics(tmp_path, 3) == legacy


def test_resolve_validation_diagnostics_supports_packed_seed_name(
        tmp_path: Path):
    diagnostics = tmp_path / 'diagnostics'
    diagnostics.mkdir()
    seed_one = diagnostics / 'valid_epoch_003_seed_01.json'
    seed_zero = diagnostics / 'valid_epoch_003_seed_00.json'
    seed_one.touch()
    seed_zero.touch()

    assert _resolve_validation_diagnostics(tmp_path, 3) == seed_zero


def test_resolve_validation_diagnostics_reports_missing_epoch(tmp_path: Path):
    (tmp_path / 'diagnostics').mkdir()

    with pytest.raises(FileNotFoundError, match='epoch 3'):
        _resolve_validation_diagnostics(tmp_path, 3)

import csv

import pytest

from ebjd.train import append_epoch_metrics, format_epoch_summary
from ebjd.config import validate_supported_config

from conftest import load_paper_config


def _row(epoch):
    return {
        "epoch": epoch,
        "train_loss": 1.25 / epoch,
        "minADE": 0.21,
        "minFDE": 0.39,
        "JADE": 0.31,
        "JFDE": 0.61,
        "epoch_minutes": 3.5,
    }


def test_epoch_summary_is_one_compact_line():
    line = format_epoch_summary(1, 100, _row(1), 3.5)
    assert line == (
        "Epoch 001/100 | minADE=0.210000 | minFDE=0.390000 | "
        "JADE=0.310000 | JFDE=0.610000 | time=3.50min")
    assert "\n" not in line


def test_epoch_csv_is_atomic_append_with_one_row_per_epoch(tmp_path):
    path = tmp_path / "epoch_metrics.csv"
    append_epoch_metrics(path, _row(1))
    append_epoch_metrics(path, _row(2))
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(item["epoch"]) for item in rows] == [1, 2]
    assert set(rows[0]) == {
        "epoch", "train_loss", "minADE", "minFDE", "JADE", "JFDE",
        "epoch_minutes",
    }
    assert not path.with_suffix(".csv.tmp").exists()


def test_epoch_csv_refuses_duplicate_or_reordered_epoch(tmp_path):
    path = tmp_path / "epoch_metrics.csv"
    append_epoch_metrics(path, _row(2))
    with pytest.raises(ValueError, match="strictly increasing"):
        append_epoch_metrics(path, _row(2))
    with pytest.raises(ValueError, match="strictly increasing"):
        append_epoch_metrics(path, _row(1))


def test_zara1_config_is_explicitly_data_ready_but_formal_training_blocked():
    config = load_paper_config("zara1")
    validate_supported_config(config)
    assert config["experiment"]["fold"] == "zara1"
    assert config["paths"]["train_manifest"].endswith("train_manifest.json")
    assert config["paths"]["accepted_fold_matched_gdts_unet"] is None
    assert config["evaluation"]["matched_gdts_validation"]["status"] == (
        "missing_fold_matched_assets")

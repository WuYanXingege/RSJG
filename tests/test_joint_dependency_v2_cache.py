import json
from types import SimpleNamespace

import pytest
import torch

from src.joint_dependency_v2_cache import (
    JDV2_CACHE_SCHEMA_VERSION,
    atomic_json_save,
    atomic_torch_save,
    build_manifest,
    load_cache_record,
    sha256_file,
    stable_json_hash,
    validate_manifest,
)


def _args():
    return SimpleNamespace(
        dataset="eth5", test_set="eth", obs_length=8, pred_length=12,
        down_factor=8, skip_ts_window=1, graph_type="radius_ttc",
        graph_radius=6.0, ttc_threshold=8.0, trajectory_dt=0.4,
        num_goal_candidates=21, seed=2035,
    )


def test_checkpoint_hash_uses_file_bytes(tmp_path):
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    first.write_bytes(b"same bytes")
    second.write_bytes(b"same bytes")
    assert sha256_file(first) == sha256_file(second)
    second.write_bytes(b"different bytes")
    assert sha256_file(first) != sha256_file(second)


def test_manifest_has_required_fields_and_mismatch_is_rejected(tmp_path):
    source = tmp_path / "source.pkl"
    source.write_bytes(b"source")
    checkpoint = tmp_path / "goal.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = build_manifest(
        _args(), {"train": [source], "valid": [source], "test": [source]},
        str(checkpoint), completed_splits=("train", "valid", "test"))
    required = {
        "schema_version", "source_commit", "dataset_hash", "split_hash",
        "preprocess_config", "goal_checkpoint_hash", "graph_config",
        "coordinate_units", "dt", "K", "dtype", "creation_time",
    }
    assert required.issubset(manifest)
    assert manifest["schema_version"] == JDV2_CACHE_SCHEMA_VERSION
    assert validate_manifest(manifest, manifest) == stable_json_hash(manifest)
    changed = dict(manifest, K=20)
    with pytest.raises(RuntimeError, match="manifest mismatch"):
        validate_manifest(changed, manifest)


def test_atomic_records_forbid_learned_outputs_and_future_leakage(tmp_path):
    path = tmp_path / "record.pt"
    with pytest.raises(ValueError, match="trainable-network"):
        atomic_torch_save({"p_z": torch.ones(1, 4)}, str(path))
    atomic_torch_save({
        "goal_candidates_world": torch.zeros(1, 21, 2),
        "future_pair_descriptor": torch.zeros(0, 6),
    }, str(path))
    assert load_cache_record(
        str(path), allow_future_supervision=True)[
            "goal_candidates_world"].shape == (1, 21, 2)
    with pytest.raises(RuntimeError, match="Future-supervision"):
        load_cache_record(str(path), allow_future_supervision=False)


def test_atomic_json_write_is_complete(tmp_path):
    path = tmp_path / "manifest.json"
    atomic_json_save({"complete": True}, str(path))
    assert json.loads(path.read_text()) == {"complete": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_loader_accepts_only_explicitly_pinned_cache_source_commit(
        tmp_path, monkeypatch):
    from src import data_loader

    old_commit = "1" * 40
    new_commit = "2" * 40
    actual = {
        "source_commit": old_commit,
        "invariant": "same",
        "completed_splits": ["test", "train", "valid"],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(actual))
    checkpoint = tmp_path / "goal.pt"
    checkpoint.write_bytes(b"goal")
    args = SimpleNamespace(
        jdv2_source_checkpoint=str(checkpoint), pretrain_path=None,
        jdv2_cache_source_commit=old_commit)
    monkeypatch.setattr(data_loader, "jdv2_cache_root", lambda _args: str(tmp_path))
    monkeypatch.setattr(data_loader, "batch_cache_path", lambda _args: str(tmp_path))
    monkeypatch.setattr(data_loader, "batch_cache_files", lambda _path: [])
    monkeypatch.setattr(data_loader, "build_manifest", lambda *args, **kwargs: {
        "source_commit": new_commit,
        "invariant": "same",
        "completed_splits": ["test", "train", "valid"],
    })
    data_loader._JDV2_MANIFEST_CACHE.clear()
    loaded = data_loader._validated_jdv2_manifest(args)
    assert loaded["source_commit"] == old_commit
    assert loaded["manifest_hash"] == stable_json_hash(actual)

    args.jdv2_cache_source_commit = "3" * 40
    with pytest.raises(RuntimeError, match="source_commit"):
        data_loader._validated_jdv2_manifest(args)

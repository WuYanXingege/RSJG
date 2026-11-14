import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_gated_release_background import atomic_json, progress, resume_command
from run_gated_release_background import finish_bounded, digest


def test_progress_tolerates_a_partially_written_tail(tmp_path):
    (tmp_path / "train.jsonl").write_text(json.dumps({"epoch": 2, "update": 300, "loss": 1.2}) + '\n{"epoch":')
    assert progress(tmp_path)["update"] == 300


def test_resume_requires_full_checkpoint_and_preserves_command(tmp_path):
    command = ["python", "train.py"]
    assert resume_command(command, tmp_path) == command
    (tmp_path / "train.jsonl").write_text("partial")
    with pytest.raises(RuntimeError, match="complete last"):
        resume_command(command, tmp_path)
    (tmp_path / "last.pt").write_bytes(b"stand-in for existence check; child strict-loads")
    assert resume_command(command, tmp_path) == command + ["--resume"]
    assert command == ["python", "train.py"]


def test_atomic_status_replacement(tmp_path):
    path = tmp_path / "STATE.json"
    atomic_json(path, {"status": "STARTING"})
    atomic_json(path, {"status": "RUNNING", "update": 1})
    assert json.loads(path.read_text()) == {"status": "RUNNING", "update": 1}
    assert list(tmp_path.iterdir()) == [path]


def bounded_fixture(root):
    train=root/'training';(train/'validation').mkdir(parents=True)
    (train/'last.pt').write_bytes(b'test checkpoint identity; trainer performs strict-load')
    atomic_json(train/'STATUS.json',dict(status='BOUNDED_STOP',completed_epoch=5,update=1285))
    (train/'checkpoint_index.jsonl').write_text(json.dumps(dict(epoch=5,update=1285,sha256=digest(train/'last.pt')))+'\n')
    atomic_json(train/'validation/epoch_005.json',dict(seed=2035,H=dict(minADE=.5,minFDE=.9,JADE=.6,JFDE=1.)))
    return train


def test_bounded_run_freezes_only_complete_epoch_and_is_idempotent(tmp_path):
    train=bounded_fixture(tmp_path)
    result=finish_bounded(tmp_path,train,5,0)
    assert result['status']=='BOUNDED_TRAINING_COMPLETE'
    assert not result['automatic_extension'] and not result['old_model_control_trained']
    assert digest(train/'fixed_epoch005.pt')==digest(train/'last.pt')
    assert finish_bounded(tmp_path,train,5,0)==result


def test_bounded_run_rejects_incomplete_or_changed_checkpoint(tmp_path):
    train=bounded_fixture(tmp_path)
    with pytest.raises(RuntimeError,match='complete epoch'):finish_bounded(tmp_path,train,5,1)
    with pytest.raises(RuntimeError,match='complete epoch'):finish_bounded(tmp_path,train,4,0)
    (train/'last.pt').write_bytes(b'wrong state')
    with pytest.raises(RuntimeError,match='mismatch'):finish_bounded(tmp_path,train,5,0)

import json

import pytest

from scripts.run_gated_release_background import atomic_json, progress, resume_command


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

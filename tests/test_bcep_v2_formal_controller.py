"""CPU control-plane checks only. These tests perform zero optimizer updates."""
import signal

import pytest
import torch

from bcep.contracts import canonical_hash, digest
from bcep_formal import control
from bcep_joint.common import write
from bcep_v2.formal_manifest import dev_ranks, freeze_all, load_rank
from bcep_v2.formal_run import Controller, locked


@pytest.mark.parametrize("exit_code,stop", [(75, False), (0, True), (2, False)])
def test_worker_stop_does_not_advance(tmp_path, monkeypatch, exit_code, stop):
    (tmp_path / "receipts").mkdir()
    monkeypatch.setattr(control, "_PAUSE", False)
    fake_child = tmp_path / "fake_child.py"
    trigger = "os.kill(os.getppid(), signal.SIGTERM)" if stop else "pass"
    fake_child.write_text("import os,signal,time\nsignal.signal(signal.SIGTERM, lambda *_: None)\n"
                          + trigger + "\ntime.sleep(0.1)\nraise SystemExit(" + str(exit_code) + ")\n")
    # Exercise the controller's actual Popen, signal forwarding, wait and status transitions.
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    calls = []
    with locked(tmp_path) as lock:
        c = Controller.__new__(Controller)
        c.root, c.lock, c.fold, c.stage, c.child = tmp_path, lock, "zara2", "density", None
        c.run, c.stop, c.started = {"commit": "fixture"}, False, 0
        def fake_fold():
            c.invoke(["fake_child"])
            calls.append("next stage")
        monkeypatch.setattr(c, "training_fold", fake_fold)
        old_handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            rc = c.execute()
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
        assert rc == (75 if stop or exit_code == 75 else 1)
        assert c.state["status"] == ("PAUSED_RESUMABLE" if rc == 75 else "FAILED")
        assert c.state["substep"] == "density" and not calls
        with pytest.raises(BlockingIOError), locked(tmp_path):
            pass


def test_topfive_dev_only_ties_keep_earlier_epochs(tmp_path):
    history = [dict(epoch=e, validation=v, test=100-e, training_loss=e) for e, v in
               [(7, 1.), (2, 1.), (5, 0.5), (4, 2.), (3, 0.2), (1, 1.), (6, 0.1)]]
    assert [r["epoch"] for r in dev_ranks(history)] == [6, 3, 5, 1, 2]
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        dev_ranks(history + [history[1]])
    write(tmp_path / "FORMAL_RUN.json", {"commit": "fixture"})
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        freeze_all(tmp_path, {"zara2": {}})
    assert not (tmp_path / "ALL_FOLDS_FROZEN.json").exists()


def test_readonly_nonrank1_strict_load_preserves_official(tmp_path):
    model = torch.nn.Linear(2, 1)
    ident = dict(stage="joint", fold="fixture", base="d", head="h", train_bank="a", dev_bank="b")
    cp = tmp_path / "epoch_005.pt"
    saved = dict(identity=ident, completed_epoch=5, update=10, cursor=0, order=None,
                 history=[dict(validation=1.25)], status="EPOCH_COMPLETE", model=model.state_dict(),
                 optimizer={}, scheduler={}, rng={}, epoch_sums={}, stale=1)
    torch.save(saved, cp)
    row = dict(dev_rank=2, epoch=5, update=10, dev_value=1.25, file=cp.name,
               identity=canonical_hash(ident), checkpoint_sha256=digest(cp))
    for name in ("best.pt", "FROZEN.json", "TOP5.json", "official/COMPLETE.json"):
        write(tmp_path / name, {"rank": 1})
    def snapshot():
        return {p: (digest(p), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    before = snapshot()
    target = torch.nn.Linear(2, 1)
    load_rank(tmp_path, row, ident, target)
    assert torch.equal(target.weight, model.weight) and not target.training and not target.weight.requires_grad
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        load_rank(tmp_path, row, dict(ident, head="different"), target)
    with pytest.raises(RuntimeError):
        load_rank(tmp_path, row, ident, torch.nn.Linear(3, 1))
    assert before == snapshot()

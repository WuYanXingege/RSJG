"""Engineering checks on TEST_ONLY synthetic windows. No optimizer steps here."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F

from bcep_joint.common import read, seed_all, write
from bcep_joint.inference import keyed_noise
from bcep_v2 import bank, joint, stages
from bcep_v2.provider import GEOMETRY, Provider, spatial_encoding, head_sums
from bcep_v2.runtime import bilinear_zeros, pool_map
from ebjd.data import SyntheticSceneDataset, collate_scenes
from ebjd.unified import gather_candidates


def fixture_identity():
    split = dict(
        fold="TEST_ONLY",
        train_indices=[0, 1],
        members=[dict(scene_id=f"fixture{i}", agent_ids=["a"], frame_ids=list(range(20))) for i in range(2)],
        scales=[[1, 1]] * 12,
    )
    ident = bank.bank_identity(
        split, "train", 2035, {}, "TEST_ONLY", dict(stage="density", trained=False), test_only=True
    )
    return ident, split


def fixture_goals(i):
    return torch.full((20, 1, 2), float(i))


def test_export_cli_signal_and_strict_resume(tmp_path):
    # A real signal during predict, and the same exit handler as python -m bcep_v2.workflow.
    code = """
from pathlib import Path
import os, signal, sys
from unittest.mock import patch
from bcep_formal.control import install
from bcep_v2 import workflow as wf, bank
from test_bcep_v2_readiness import fixture_identity, fixture_goals
root = Path(sys.argv[1])
ident, split = fixture_identity()
def predict(i):
    os.kill(os.getpid(), signal.SIGTERM)
    return fixture_goals(i)
def command():
    bank.export(root/'first', ident, split, predict)
    (root/'next-bank-started').touch()
    return {}
install()
with patch.object(wf, 'main', lambda: wf.run_command(command, root, 'base-cache')):
    wf.cli()
"""
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
        ),
    )
    result = subprocess.run(
        [
            "sh",
            "-c",
            '"$1" -c "$2" "$3" && touch "$3/next-stage-started"',
            "TEST_ONLY",
            sys.executable,
            code,
            str(tmp_path),
        ],
        env=env,
        timeout=45,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert result.returncode == 75, result.stdout
    assert read(tmp_path / "EXPORT_PAUSED.json")["next_index"] == 1
    assert not (tmp_path / "next-bank-started").exists()
    assert not (tmp_path / "next-stage-started").exists()
    ident, split = fixture_identity()
    partial = tmp_path / "first"
    assert sorted(p.name for p in partial.iterdir()) == ["0.json", "0.npy", "IDENTITY.json"]
    with pytest.raises(FileNotFoundError):
        bank.Bank(partial, ident, split)
    calls = []

    def predict(i):
        calls.append(i)
        return fixture_goals(i)

    with pytest.raises(ValueError):
        bank.export(partial, dict(ident, seed=3101), split, predict)
    row = read(partial / "0.json")
    write(partial / "0.json", dict(row, member=split["members"][1]))
    with pytest.raises(ValueError):
        bank.export(partial, ident, split, predict)
    write(partial / "0.json", row)
    bank.export(partial, ident, split, predict)
    assert calls == [1]
    other = tmp_path / "continuous"
    bank.export(other, ident, split, fixture_goals)
    assert bank.Bank(partial, ident, split).hash == bank.Bank(other, ident, split).hash
    assert all((partial / f"{i}.npy").read_bytes() == (other / f"{i}.npy").read_bytes() for i in (0, 1))
    with (partial / "0.npy").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError):
        bank.Bank(partial, ident, split)


def test_pause_at_last_window_and_between_banks(tmp_path, monkeypatch):
    ident, split = fixture_identity()
    paused = False
    monkeypatch.setattr(bank, "pause_requested", lambda: paused)

    def predict(i):
        nonlocal paused
        paused = i == 1
        return fixture_goals(i)

    with pytest.raises(bank.ExportPaused):
        bank.export(tmp_path / "a", ident, split, predict)
    assert (tmp_path / "a" / "1.json").is_file() and not (tmp_path / "a" / "COMPLETE.json").exists()
    with pytest.raises(bank.ExportPaused):
        bank.export(tmp_path / "b", ident, split, predict)
    assert not (tmp_path / "b").exists()
    paused = False
    # Simulate interruption between data and receipt renames.
    (tmp_path / "a" / "1.json").unlink()
    with pytest.raises(ValueError, match="uncommitted"):
        bank.export(tmp_path / "a", ident, split, lambda i: fixture_goals(i + 1))
    bank.export(tmp_path / "a", ident, split, fixture_goals)
    bank.Bank(tmp_path / "a", ident, split)


def test_five_seed_validation_reuses_only_current_frozen_encoding():
    seed_all(245)
    dataset = SyntheticSceneDataset(scenes=2, agents=2, pixels=256)
    for item in dataset.items:
        item["semantic_maps"][:, 1, :, ::2] = 1

    class Data:
        reads = 0
        split = dict(
            members=[dict(agent_ids=x["agent_ids"]) for x in dataset.items], selection_dev_indices=[0, 1]
        )

        def batch(self, ids, purpose):
            self.reads += 1
            return collate_scenes([dataset[i] for i in ids])

    class Goals:
        def __init__(self, seed):
            self.seed, self.calls = seed, 0

        def batch(self, ids, x):
            self.calls += 1
            gen = torch.Generator().manual_seed(100 * self.seed + ids[0])
            return x.observed[:, None, :, -1] + torch.randn(1, 20, x.valid.shape[1], 2, generator=gen)

    banks = [Goals(i) for i in range(2035, 2040)]
    p, data = Provider().eval(), Data()
    geo = dict(layout=GEOMETRY)
    with torch.no_grad():
        p.head.readout[-1].weight.normal_(std=0.03)
        old = {}
        for b in banks:
            total, count = 0.0, 0
            for idx in (0, 1):
                x = data.batch([idx], "selection_dev")
                o, m, v = x.observed[0], x.semantic_maps[0], x.valid[0]
                enc, feat = spatial_encoding(p.base, o, m, v)
                g, shift = p.head(p.base, o, m, v, b.batch([idx], x)[0], enc, feat, geo)
                total += float(head_sums(g, shift, x.future[0, :, -1], v)[1])
                count += int(v.sum())
            old[b.seed] = (total, count)
    assert data.reads == 10
    data.reads = 0
    _, validate = stages.head_callbacks(p, data, "cpu", geo, banks[:2], banks)
    with (
        patch.object(stages, "spatial_encoding", wraps=spatial_encoding) as enc,
        patch.object(p.head, "forward", wraps=p.head.forward) as head,
    ):
        selected = validate()
    assert enc.call_count == data.reads == 2 and head.call_count == 10
    assert [b.calls for b in banks] == [4] * 5 and validate.totals == old
    assert selected == sum(s / n for s, n in old.values()) / 5
    assert all(not q.requires_grad and q.grad is None for q in p.base.parameters())


def assignment_gap(scores):
    a = scores.detach().double().numpy()
    r, c = linear_sum_assignment(-a)
    best = a[r, c].sum()
    alternatives = []
    for i, j in zip(r, c):
        modified = -a.copy()
        modified[i, j] = np.inf
        rr, cc = linear_sum_assignment(modified)
        alternatives.append(a[rr, cc].sum())
    return best - max(alternatives)


def test_live_nonzero_feedback_permutations_and_both_soft_gradients(record_property):
    seed_all(801)
    model = joint.FeedbackCoupling()
    with torch.no_grad():
        model.compatibility[-1].weight.normal_(std=0.015)
        model.world_queries.mul_(100.0)  # distinguished synthetic world queries
    v = torch.tensor([[True, True, True, False]])
    o, h = torch.randn(1, 4, 8, 2), torch.randn(1, 4, 192)
    g = torch.randn(1, 20, 4, 2)
    z = keyed_noise(["TEST_ONLY"], [["a", "b", "c"]], v, 2035)
    target = torch.randn(1, 4, 2)

    def call(o, h, v, g):
        return model.assign(
            SimpleNamespace(observed=o, social_h=h, valid=v), g, torch.tensor(1.0), straight_through=True
        )

    trace = []
    original = joint.assignment

    def capture(s, valid, training):
        result = original(s, valid, training)
        trace.append((s, *result))
        return result

    with patch.object(joint, "assignment", capture):
        goals, idx, scores = call(o, h, v, g)
    initial_scores, _, p0, _ = trace[0]
    _, _, _, soft = trace[1]
    loss, _ = joint.pair_loss(goals, target, v)
    d0, dfinal = torch.autograd.grad(loss, (p0, soft), retain_graph=True)
    via_initial = torch.autograd.grad(p0, initial_scores, d0, retain_graph=True)[0]
    via_final = torch.autograd.grad(soft, scores, dfinal, retain_graph=True)[0]
    loss.backward()
    for grad in (
        via_initial,
        via_final,
        model.compatibility[-1].weight.grad,
        model.project.weight.grad,
        model.score[-1].weight.grad,
    ):
        assert grad.isfinite().all() and grad.abs().max() > 0
    for name, values in (("initial", initial_scores), ("refined", scores)):
        gap = min(assignment_gap(values[0, i]) for i in range(3))
        assert gap > 1e-8
        record_property(name + "_min_assignment_gap", gap)
    assert torch.equal(idx.sort(1).values, torch.arange(20)[None, :, None].expand_as(idx))
    torch.testing.assert_close(
        (goals - target[:, None]).norm(dim=-1).amin(1)[v],
        (g - target[:, None]).norm(dim=-1).amin(1)[v],
        atol=0,
        rtol=0,
    )
    agents = [2, 0, 3, 1]
    perms = torch.stack([torch.randperm(20) for _ in agents], -1)[None]
    # The padded agent uses the contractual identity, not a physical candidate.
    perms[:, :, 3] = torch.arange(20)
    shuffled = gather_candidates(g, perms)[:, :, agents]
    result, ix, s = call(o[:, agents], h[:, agents], v[:, agents], shuffled)
    columns = perms.permute(0, 2, 1)[:, :, None, :].expand_as(scores)
    torch.testing.assert_close(s, scores.gather(-1, columns)[:, agents], rtol=2e-5, atol=2e-6)
    soft2 = original(s, v[:, agents], True)[2]
    torch.testing.assert_close(soft2, soft.gather(-1, columns)[:, agents], rtol=2e-5, atol=2e-6)
    assert torch.equal(goals[:, :, agents], result)
    assert torch.equal(
        gather_candidates(z, idx)[:, :, agents],
        gather_candidates(gather_candidates(z, perms)[:, :, agents], ix),
    )
    single = call(o[:, :1], h[:, :1], v[:, :1], g[:, :, :1])[1]
    assert torch.equal(single[0, :, 0], torch.arange(20))
    tied = torch.ones_like(scores)
    tied[:, :, 0, :] = 2
    pi = original(tied, v, False)[0]
    assert torch.equal(pi.sort(1).values, torch.arange(20)[None, :, None].expand_as(pi))
    assert torch.all(tied.gather(-1, pi.transpose(1, 2).unsqueeze(-1)).sum(2) == 21)


def test_v2_bilinear_values_and_both_gradients():
    seed_all(73)
    x = torch.randn(2, 3, 5, 8, requires_grad=True)
    grid = (torch.rand(2, 11, 2, 2) * 3 - 1.5).requires_grad_()
    weight = torch.randn(2, 3, 11, 2)
    args = dict(mode="bilinear", padding_mode="zeros", align_corners=True)
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        reference = F.grid_sample(x, grid, **args)
        actual = bilinear_zeros(x, grid, **args)
        for a, b in zip(
            (actual, *torch.autograd.grad(actual, (x, grid), weight)),
            (reference, *torch.autograd.grad(reference, (x, grid), weight)),
        ):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=2e-6)
    finally:
        torch.use_deterministic_algorithms(previous)


def test_divisible_map_pool_windows_and_gradient():
    seed_all(21)
    x = torch.randn(2, 3, 64, 32, requires_grad=True)
    w = torch.randn(2, 3, 8, 8)
    original = F.adaptive_avg_pool2d(x, (8, 8))
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        actual = pool_map(x, (8, 8))
        torch.testing.assert_close(actual, original, atol=1e-6, rtol=1e-6)
        assert torch.equal(torch.autograd.grad(actual, x, w)[0], torch.autograd.grad(original, x, w)[0])
    finally:
        torch.use_deterministic_algorithms(previous)

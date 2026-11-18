"""CPU fixtures only. The sole optimizer test uses exactly eight synthetic updates."""

from types import SimpleNamespace
import math
import random
import numpy as np
import pytest
import torch
import yaml
from bcep.contracts import digest
from bcep_joint.checks import equal, synthetic_batch
from bcep_joint.common import seed_all, read
from bcep_joint.inference import generate
from ebjd.unified import gather_candidates, LearnableJointCouplingHead
from bcep_v2.bank import bank_identity, Bank, export
from bcep_v2.joint import FeedbackCoupling, assignment, neighbors, build_joint, clean_path_loss, pair_loss
from bcep_v2.provider import GEOMETRY, Provider, query_map, spatial_encoding, head_sums
from bcep_v2.objective import bridge_and_pair_loss
from bcep_v2.stages import augment
from bcep_v2.state import run_stage
from bcep_v2.workflow import CONFIG


def config():
    return dict(yaml.safe_load(CONFIG.read_text())["joint"], scales=dict(goal=1.0, bridge=0.25))


def test_feature_centers_missing_and_outside():
    geo = dict(layout=GEOMETRY)
    y, x = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    f = (2 * x + y).float().expand(1, 128, 32, 32)
    maps = torch.ones(1, 6, 256, 256)
    pixel = torch.tensor([[[76.0, 88.0], [248.0, 248.0], [255.0, 255.0], [-1.0, 0.0], [0.0, 0.0]]])
    goal = pixel * 32 / 255 - 16
    maps[:, :, 0, 0] = 0
    val, ok = query_map(f, maps, goal, torch.zeros(1, 2), geo)
    torch.testing.assert_close(val[0, :2, 0], torch.tensor([30.0, 93.0]))
    assert ok.tolist() == [[True, True, False, False, False]]
    with pytest.raises(ValueError, match="verified geometry"):
        query_map(f, maps, goal, torch.zeros(1, 2), {})


def test_head_set_equivariance_bounds_gradient_and_padding():
    seed_all(91)
    p = Provider()
    p.base.eval().requires_grad_(False)
    x = synthetic_batch(3)
    valid = torch.tensor([True, False, True])
    o = x.observed[0]
    o[1] = float("nan")
    h, f = spatial_encoding(p.base, o, None, valid)
    goals = torch.randn(20, 3, 2, dtype=torch.float64)
    result, shift = p.head(p.base, o, None, valid, goals, h, f, None)
    assert result.dtype == goals.dtype and torch.equal(result[:, valid], goals[:, valid])
    assert not result[:, ~valid].any()
    total, _ = head_sums(result, shift, x.future[0, :, -1], valid)
    total.backward()
    assert p.head.readout[-1].weight.grad.abs().sum() > 0 and all(
        q.grad is None for q in p.base.parameters()
    )
    with torch.no_grad():
        p.head.readout[-1].weight.normal_()
        perm = torch.randperm(20)
        result, d = p.head(p.base, o, None, valid, goals, h, f, None)
        shuffled, _ = p.head(p.base, o, None, valid, goals[perm], h, f, None)
    torch.testing.assert_close(shuffled, result[perm], rtol=1e-6, atol=1e-7)
    assert d.norm(dim=-1).max() < 0.5


def test_augmentation_and_refreshed_features():
    seed_all(25)
    x = synthetic_batch(2)
    x.semantic_maps[:] = 0
    x.semantic_maps[:, :, 0, 105:150, 145:169] = 1
    g = x.future[:, :, -1][:, None].expand(-1, 20, -1, -1).clone()
    from ebjd.data import augment_batch_with_points

    p = Provider()
    for seed in (0, 1):  # reflection and rotation-only
        a, r, receipt = augment(x, g, torch.Generator().manual_seed(seed))
        original, points = augment_batch_with_points(x, g, torch.Generator().manual_seed(seed))
        assert torch.equal(a.semantic_maps, original.semantic_maps) and torch.equal(r, points)
        torch.testing.assert_close(a.future[:, :, -1], r[:, 0])
        centered = (g[0] - receipt["center"][0]) @ receipt["rotation"][0].T + receipt["center"][0]
        torch.testing.assert_close(r[0], centered, atol=1e-6, rtol=1e-6)
        with torch.no_grad():
            h, f = spatial_encoding(p.base, a.observed[0], a.semantic_maps[0], a.valid[0])
            assert torch.equal(h, p.base.encoder(a.observed[0], a.semantic_maps[0], a.valid[0], 0.4))
            assert torch.equal(f, p.base.encoder.map[:-1](a.semantic_maps[0]))


def test_micro_weighting_and_physical_units():
    p = Provider()
    from bcep_v2.stages import density_callbacks

    class Data:
        split = dict(
            members=[dict(agent_ids=["a"]), dict(agent_ids=["a", "b", "c"])], selection_dev_indices=[0, 1]
        )

        def item(self, idx, purpose):
            x = synthetic_batch(1 if idx == 0 else 3, pixels=16)
            return dict(
                observed=x.observed[0], semantic_maps=x.semantic_maps[0], future=x.future[0] + idx
            )

    d = Data()
    _, validate = density_callbacks(p, d, "cpu")
    sums = []
    for i in (0, 1):
        x = d.item(i, "selection_dev")
        valid = torch.ones(len(x["observed"]), dtype=torch.bool)
        sums.append(
            p.base.loss_components(x["observed"], x["semantic_maps"], valid, x["future"])["nll_final_sum"]
        )
    assert math.isclose(validate(), float(sum(sums).detach() / 4), rel_tol=1e-6)
    g = torch.zeros(20, 3, 2)
    g[:, 0, 0] = 1
    g[:, 1, 0] = 3
    total, dist = head_sums(
        g, torch.zeros(3, 20, 2), torch.zeros(3, 2), torch.tensor([True, True, False])
    )
    assert total == dist == 4
    future = torch.zeros(2, 3, 12, 2)
    path = torch.zeros(2, 1, 3, 12, 2)
    path[0, ..., 0] = 1.0
    path[1, ..., 0] = 2.0
    path[..., -1, :] = 200.0
    path.requires_grad_()
    valid = torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.bool)
    value = clean_path_loss(path, future, valid, torch.tensor([0.2, 0.8]).view(2, 1, 1, 1, 1))
    torch.testing.assert_close(value, torch.tensor((0.2 * 0.75 + 0.8 * 1.75) / 2))
    value.backward()
    assert path.grad[..., -1, :].abs().sum() == 0 and path.grad[0, :, 1:].abs().sum() == 0


def test_expected_pair_messages_and_ties():
    torch.manual_seed(7)
    m = FeedbackCoupling()
    m.compatibility[-1].weight.data.normal_()
    o = torch.zeros(1, 11, 8, 2)
    v = torch.ones(1, 11, dtype=torch.bool)
    scene, i, j = neighbors(o, v)
    assert len(i) == 110 and (i != j).all()
    f = torch.randn(1, 2, 20, 192)
    g = torch.randn(1, 20, 2, 2)
    p0 = torch.eye(20).expand(1, 2, -1, -1).clone().requires_grad_()
    a = m.messages(f, o[:, :2], g, v[:, :2], p0)
    b = m.messages(f, o[:, :2], g, v[:, :2], p0.flip(2))
    assert not torch.equal(a, b)
    a.sum().backward()
    assert p0.grad.isfinite().all() and p0.grad.abs().sum() > 0
    assert not m.messages(f[:, :1], o[:, :1], g[:, :, :1], v[:, :1], p0[:, :1]).any()
    with pytest.raises(ValueError):
        neighbors(o, ~v)


def test_assignment_and_metric_invariants():
    seed_all(310)
    m = FeedbackCoupling()
    v = torch.tensor([[True, True, False], [True, False, False]])
    o = torch.randn(2, 3, 8, 2)
    h = torch.randn(2, 3, 192)
    ctx = SimpleNamespace(valid=v, observed=o, social_h=h)
    g = torch.randn(2, 20, 3, 2, dtype=torch.float64, requires_grad=True)
    coupled, pi, s = m.assign(ctx, g, torch.tensor(1.0), straight_through=True)
    assert torch.equal(coupled, gather_candidates(g, pi))
    assert torch.equal(pi.sort(1).values, torch.arange(20)[None, :, None].expand_as(pi))
    assert torch.equal(pi[1, :, 0], torch.arange(20))
    _, _, soft = assignment(s, v, True)
    assert torch.equal(soft[1, 0], torch.eye(20)) and torch.equal(soft[0, -1], torch.eye(20))
    old_scores = LearnableJointCouplingHead.forward(
        m, ctx, torch.where(v[:, None, :, None], g, 0.0), torch.tensor(1.0)
    )
    assert torch.equal(s, old_scores)
    truth = torch.randn(2, 3, 2)
    loss, _ = pair_loss(coupled, truth, v)
    loss.backward()
    for param in [m.score[-1].weight, m.compatibility[-1].weight]:
        assert param.grad.isfinite().all() and param.grad.abs().sum() > 0
    assert g.grad is None
    assert torch.equal(
        (g - truth[:, None]).norm(dim=-1).amin(1), (coupled - truth[:, None]).norm(dim=-1).amin(1)
    )
    # Unique diagonal optimum is equivariant to agent and candidate indexing.
    scores = torch.eye(20)[None, None].expand(1, 2, -1, -1) * 10.0
    shuffle = torch.randperm(20)
    indices, _, _ = assignment(scores[..., shuffle], v[:1, :2], False)
    assert torch.equal(shuffle[indices], torch.arange(20)[None, :, None].expand_as(indices))
    order = torch.tensor([1, 0, 2])
    c2 = SimpleNamespace(valid=v[:, order], observed=o[:, order], social_h=h[:, order])
    _, _, s2 = m.assign(c2, g[:, :, order], torch.tensor(1.0), straight_through=False)
    torch.testing.assert_close(s2, s[:, order], atol=1e-5, rtol=1e-5)


def test_bank_to_bridge(tmp_path):
    seed_all(2)
    x = synthetic_batch(2, 32)
    member = dict(scene_id=x.scene_ids[0], agent_ids=x.agent_ids[0], frame_ids=x.frame_ids[0])
    split = dict(fold="fixture", members=[member], train_indices=[0], scales=[[1.0, 1.0]] * 12)
    base = dict(stage="density", trained=False, frozen=False)
    head = dict(stage="head", trained=False, frozen=False)
    geo = dict(layout=GEOMETRY)
    with pytest.raises(ValueError):
        bank_identity(split, "train", 2035, geo, "test", base, head)
    ident = bank_identity(split, "train", 2035, geo, "test", base, head, test_only=True)
    path = tmp_path / "synthetic_test_only"
    g = torch.randn(20, 2, 2)
    export(path, ident, split, lambda idx: g)
    bank = Bank(path, ident, split)
    for key in ("split", "base", "head", "geometry", "seed", "schema"):
        with pytest.raises(ValueError):
            Bank(path, dict(ident, **{key: "wrong"}), split)
    model = build_joint(config())
    goals = bank.batch([0], x)
    called = []
    hook = model.bridge.denoiser.register_forward_hook(lambda *args: called.append(1))
    total, *_ = bridge_and_pair_loss(model, x, goals, time_draw=torch.tensor([0.5]))
    hook.remove()
    assert len(called) == 1 and total.isfinite()
    total.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.bridge.denoiser.parameters())
    with torch.no_grad():
        gpath, hpath, pi, receipt = generate(
            model, x.observed, x.semantic_maps, x.valid, x.scene_ids, x.agent_ids, goals, 2035
        )
        torch.testing.assert_close(gpath[..., -1, :], goals)
        torch.testing.assert_close(hpath[..., -1, :], gather_candidates(goals, pi))
        assert receipt["exact_noise_gather"]


def test_synthetic_resume_eight_steps(tmp_path):
    recipe = dict(
        seed=87,
        lr=0.001,
        weight_decay=0.0001,
        clip=1.0,
        logical_scenes=2,
        max_epochs=2,
        minimum=2,
        patience=5,
        scheduler_patience=2,
    )
    ident = dict(stage="head", synthetic_test_only=True)
    traces = {}

    def run(folder, limit, resume=False):
        seed_all(3)
        model = torch.nn.Linear(2, 1)

        def backward(ids, order, views):
            view = int(torch.randint(2, (), generator=views))
            traces.setdefault(folder, []).append((ids, view))
            x = torch.rand(len(ids), 2) + random.random() + np.random.rand()
            y = torch.rand(len(ids), 1) + view / 2
            loss = (model(x) - y).square().mean()
            loss.backward()
            return dict(
                loss=loss.detach(),
                valid_agent_count=len(ids),
                nll_final_sum=sum(ids),
                h4_nll_sum=sum(ids) * 2,
                h8_nll_sum=sum(ids) * 3,
            )

        return run_stage(
            tmp_path / folder,
            ident,
            recipe,
            model,
            [1, 2, 3, 4, 5],
            backward,
            lambda: 0.23,
            limit,
            resume=resume,
        )

    a = run("continuous", 4)
    paused = run("resumed", 2)
    assert paused["update"] == 2 and paused["cursor"] == 4
    assert paused["epoch_counts"] == {"valid_agent_count": 4}
    assert run("resumed", 2, True)["update"] == 2
    b = run("resumed", 4, True)
    assert len(traces["continuous"]) + len(traces["resumed"]) == 8
    assert traces["continuous"] == traces["resumed"]
    for key in (
        "model",
        "optimizer",
        "scheduler",
        "order",
        "cursor",
        "rng",
        "history",
        "stale",
        "completed_epoch",
        "update",
        "epoch_sums",
        "epoch_raw_sums",
        "epoch_counts",
        "epoch_steps",
    ):
        equal(a[key], b[key])
    assert b["history"][0]["train_nll_final_agent_micro"] == 3
    assert b["history"][0]["train_h4_nll_agent_micro"] == 6
    assert b["history"][0]["train_h8_nll_agent_micro"] == 9
    assert b["epoch_counts"] == {"valid_agent_count": 2}
    path = tmp_path / "resumed" / "last.pt"
    obsolete = torch.load(path, weights_only=False)
    obsolete["schema"] = "bcep-ebjd-integrated-state-v2"
    torch.save(obsolete, path)
    with pytest.raises(ValueError, match="schema"):
        run("resumed", 4, True)
    assert read(tmp_path / "resumed" / "TOP5.json")[0]["sha256"] == digest(tmp_path / "resumed" / "best.pt")


def test_stage_objectives_on_synthetic_logical_group():
    from bcep_v2.stages import density_callbacks, head_callbacks, joint_callbacks
    from ebjd.data import SyntheticSceneDataset, collate_scenes

    seed_all(310)
    dataset = SyntheticSceneDataset(scenes=2, agents=2, pixels=256)

    class Data:
        split = dict(
            members=[dict(agent_ids=item["agent_ids"]) for item in dataset.items],
            selection_dev_indices=[0, 1],
        )

        def item(self, idx, purpose):
            return dataset[idx]

        def batch(self, ids, purpose):
            return collate_scenes([dataset[i] for i in ids])

    class FixtureBank:
        def batch(self, ids, b):
            shifts = torch.linspace(-0.4, 0.4, 20)[None, :, None, None]
            return b.observed[:, None, :, -1] + shifts

    data = Data()
    p = Provider()
    rng = torch.Generator().manual_seed(10)
    b = FixtureBank()
    backward, _ = density_callbacks(p, data, "cpu")
    parts = []
    for item in dataset.items:
        n = len(item["agent_ids"])
        parts.append(
            n
            * p.base.loss_components(
                item["observed"], item["semantic_maps"], torch.ones(n, dtype=torch.bool), item["future"]
            )["loss_total"]
        )
    reference = sum(parts) / 3
    grad_ref = torch.autograd.grad(reference, p.base.flow.nets[0][-1].weight)[0]
    measured = backward([0, 1], rng, rng)["loss"]
    assert measured == pytest.approx(float(reference.detach()), rel=1e-6)
    torch.testing.assert_close(p.base.flow.nets[0][-1].weight.grad, grad_ref, atol=1e-5, rtol=1e-5)
    p.base.zero_grad(set_to_none=True)
    backward, val = head_callbacks(p, data, "cpu", dict(layout=GEOMETRY), [b, b], [b] * 5)
    head_metrics = backward([0, 1], rng, rng)
    assert head_metrics["valid_agent_count"] == 3
    assert head_metrics["head_loss_sum"] == pytest.approx(
        head_metrics["head_fde_sum"] + 0.1 * head_metrics["head_move_sum"]
    )
    assert head_metrics["loss"] == pytest.approx(head_metrics["head_loss_sum"] / 3)
    assert all(param.grad is None for param in p.base.parameters())
    with torch.no_grad():
        assert math.isfinite(val()) and len(val.metrics) == 5
    c = config()
    model = build_joint(c)
    backward, _ = joint_callbacks(model, data, "cpu", c, b, b)
    metrics = backward([0, 1], rng, rng)
    assert all(math.isfinite(v) for v in metrics.values())
    assert metrics["pair_weight"] == 1 and metrics["effective_edges"] == 2
    assert metrics["valid_scene_count"] == 2
    assert metrics["loss"] == pytest.approx(
        sum(
            metrics[k]
            for k in ("diffusion_term", "map_term", "geometry_term", "pair_term", "clean_path_term")
        ),
        abs=1e-6,
    )
    assert metrics["clean_path_term"] == pytest.approx(
        0.05 * metrics["clean_path_signal_sum"] / 2 / c["gradient_accumulation"], abs=1e-7
    )


def test_forward_without_future_or_candidate_indices():
    seed_all(718)
    p = Provider().eval()
    x = synthetic_batch(2, pixels=32)

    def sample():
        return p.candidates(x.observed[0], None, x.valid[0], x.scene_ids[0], x.agent_ids[0], 2035, None)

    a = sample()
    x.future.fill_(float("nan"))
    assert torch.equal(a, sample())
    # Offsetting both endpoints in opposite worlds defeats private agent winners.
    g = torch.zeros(1, 20, 2, 2)
    g[0, :10, 0, 0] = 2
    g[0, 10:, 1, 0] = 2
    val, diag = pair_loss(g, torch.zeros(1, 2, 2), torch.ones(1, 2, dtype=torch.bool))
    assert val.item() == pytest.approx(1.0) and diag["pair_exact_hard_jfde_m"] == 1.0

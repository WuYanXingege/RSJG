import tempfile
from pathlib import Path
import random
import numpy as np
import torch
from bcep.contracts import digest
from bcep_joint.common import read, seed_all
from bcep_v2.state import run_stage, optimizer
from ebjd.metrics import hard_metric_statistics
from ebjd.unified import gather_candidates
from bcep_joint.inference import keyed_noise, check_permutation
from scene_pair_selector.model import SceneScorer, choose_initial


def fixture(n=3):
    torch.manual_seed(13)
    return (torch.randn(1, n, 8, 2), torch.randn(1, n, 192), torch.randn(1, 20, n, 2), torch.ones(1, n, dtype=torch.bool))


def test_scorer_pooling_and_pair_sensitivity():
    obs, ctx, goals, valid = fixture()
    model = SceneScorer().eval()
    value = model(obs, ctx, goals, valid)
    # Simultaneous relabelling of all 20 worlds leaves the scene score invariant.
    permutation = torch.randperm(20)
    torch.testing.assert_close(value, model(obs, ctx, goals[:, permutation], valid), atol=1e-7, rtol=1e-5)
    # Invalid padding must have no effect, including through max pooling.
    po = torch.cat((obs, torch.randn(1, 2, 8, 2)*1000), 1)
    pc = torch.cat((ctx, torch.randn(1, 2, 192)*1000), 1)
    pg = torch.cat((goals, torch.randn(1, 20, 2, 2)*1000), 2)
    pv = torch.tensor([[True, True, True, False, False]])
    torch.testing.assert_close(value, model(po, pc, pg, pv), atol=1e-7, rtol=1e-5)
    other = goals.clone()
    other[:, :, 1] = goals[:, permutation, 1]
    delta = model.difference(obs, ctx, other, goals, valid)
    assert float(delta.detach().abs().max()) > 1e-7
    delta.square().sum().backward()
    assert sum(float(p.grad.abs().sum()) for p in model.edge.parameters()) > 0
    assert torch.equal(model.difference(obs, ctx, goals, goals, valid), torch.zeros(1, 2))
    singleton = fixture(1)
    assert torch.isfinite(model(*singleton)).all()


def test_fixed_gate():
    valid = torch.ones(6, 2, dtype=torch.bool)
    ip = torch.arange(20)[None, :, None].expand(6, -1, 2).clone()
    hp = ip.clone()
    hp[:, :, 1] = hp[:, :, 1].flip(1)
    delta = torch.tensor([[0., -1.], [-1., 0.], [1., -1.], [-1., -1.], [-1., -1.], [-1., -1.]])
    hp[4] = ip[4]
    valid[5, 1] = False
    assert choose_initial(delta, ip, hp, valid).tolist() == [True, False, False, True, False, False]


def test_noise_bank_occurrence_identity():
    valid = torch.ones(1, 3, dtype=torch.bool)
    noise = keyed_noise(['scene'], [['a', 'b', 'c']], valid, 2035)
    pi = torch.stack([torch.randperm(20) for _ in range(3)], 1)[None]
    check_permutation(pi, valid)
    assert torch.equal(gather_candidates(gather_candidates(noise, pi), pi.argsort(1)), noise)
    assert not torch.equal(noise, gather_candidates(noise, pi))


def test_native_joint_metric_order():
    prediction = torch.zeros(1, 20, 2, 12, 2)
    prediction[0, :, 0, :, 0] = 10
    prediction[0, 0, 0] = 0
    prediction[0, 0, 1, :, 0] = 8
    truth = torch.zeros(1, 2, 12, 2)
    stats = hard_metric_statistics(prediction, truth, torch.ones(1, 2, dtype=torch.bool))
    assert float(stats.scene_jade) == float(stats.scene_jfde) == 4
    assert float(stats.scene_marginal_ade) == 0


def test_actual_writer_retention_and_nonzero_resume(tmp_path):
    recipe = dict(seed=3201, lr=1e-4, weight_decay=1e-4, clip=5., logical_scenes=2,
                  max_epochs=7, minimum=7, patience=15, scheduler_horizon=100)
    identity = dict(stage='joint', fold='synthetic_test')
    validations = [0.8, 0.4, 0.5, 0.4, 0.7, 0.2, 0.6]

    def execute(folder, until, resume):
        seed_all(3201)
        model = torch.nn.Linear(2, 2)
        epoch = [0 if not resume else torch.load(folder/'last.pt', weights_only=False)['completed_epoch']]
        def backward(group, order_rng, view_rng):
            x = torch.randn(len(group), 2)+random.random()+np.random.rand()
            y = torch.randn(len(group), 2, generator=view_rng)
            loss = (model(x)-y).square().mean()
            loss.backward()
            return dict(loss=float(loss.detach()))
        def validate():
            result = validations[epoch[0]]
            epoch[0] += 1
            return result
        return run_stage(folder, identity, recipe, model, list(range(5)), backward, validate,
                         until, resume=resume, record_progress=False)

    full = execute(tmp_path/'full', 21, False)
    partial = execute(tmp_path/'resume', 5, False)
    assert partial['completed_epoch'] == 1 and partial['cursor'] == 4
    assert len(list((tmp_path/'resume').glob('epoch_*.pt'))) == 1
    resumed = execute(tmp_path/'resume', 21, True)
    for name, tensor in full['model'].items():
        torch.testing.assert_close(tensor, resumed['model'][name], rtol=0, atol=0)
    assert full['update'] == resumed['update'] == 21
    assert resumed['completed_epoch'] == 7 and resumed['status'] == 'COMPLETE'
    top = read(tmp_path/'resume/TOP5.json')
    assert [r['epoch'] for r in top] == [6, 2, 4, 3, 7]
    assert len(list((tmp_path/'resume').glob('epoch_*.pt'))) == 7
    assert digest(tmp_path/'resume/best.pt') == top[0]['sha256']
    assert digest(tmp_path/'resume/last.pt') == digest(tmp_path/'resume/epoch_007.pt')
    for row in top:
        path = tmp_path/'resume'/row['file']
        assert digest(path) == row['sha256']
        state = torch.load(path, weights_only=False)
        assert {'model', 'optimizer', 'scheduler', 'rng', 'completed_epoch', 'update', 'order', 'cursor', 'history', 'stale'} <= state.keys()
        assert {'cpu', 'cuda', 'python', 'numpy', 'order', 'view'} <= state['rng'].keys()

"""CPU synthetic wiring/state contracts, not real-model certification."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

import mc_wiring_fixture as fixture
from mc_wiring_fixture import harness, losses, run_steps, save, load
from test_jdv2_expected_conditional import explicit_conditional
from src import parser
from src.jdv2_objective_state import (
    MC, FIELD, ObjectiveRNG, atomic_save, CheckpointDurabilityError,
    validate_mode, semantic_fingerprint,
)
from src.joint_goal_loss import build_soft_goal_target

MEASUREMENTS = {}


def equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def close(a, b, name):
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    MEASUREMENTS[name] = float((a-b).detach().abs().max())


def test_legacy_actual_method_against_pinned_source(tmp_path):
    import src.models.model as module
    source = subprocess.check_output(['git', 'show',
        'dcf5fde1aa77f62a25a9209fd91e4f854a21510a:src/models/model.py'], text=True)
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'GDTS')
    node = next(n for n in cls.body if getattr(n, 'name', '') == '_jdv2_no_z_goal_losses')
    baseline = textwrap.dedent('\n'.join(source.splitlines()[node.lineno-1:node.end_lineno]))
    namespace = dict(vars(module))
    exec(compile(baseline, 'pinned_legacy_method', 'exec'), namespace)
    a = harness(tmp_path/'a', 'mean_energy')
    b = harness(tmp_path/'b', 'mean_energy')
    del a.args.jdv2_goal_objective  # missing old args field
    actual = losses(a)
    expected = namespace['_jdv2_no_z_goal_losses'](b.net, b.inputs)
    equal(actual, expected)
    equal(a.net.set_losses_coeffs(), b.net.set_losses_coeffs())
    sum(actual.values()).backward(); sum(expected.values()).backward()
    fixture.COUNTS['backward_calls'] += 2
    for pa, pb in zip(a.net.parameters(), b.net.parameters()):
        equal(pa.grad, pb.grad)
    assert not hasattr(a, 'jdv2_objective_rng')
    assert not hasattr(a.net, 'jdv2_objective_rng')
    before = {k: v.shape for k, v in a.net.state_dict().items()}
    mc = harness(tmp_path/'mc')
    assert before == {k: v.shape for k, v in mc.net.state_dict().items()}
    MEASUREMENTS['legacy_pinned_value_gradient_keys_coefficients_exact'] = True


@pytest.mark.parametrize('n,edges', [(5, True), (4, True), (5, False), (1, False)])
def test_actual_wiring_reference_and_draw_chunk_counts(tmp_path, monkeypatch, n, edges):
    import src.models.model as module
    seen = []
    pure = module.expected_conditional_composite
    def spy(u, q, scene, edge, chunks, ids, mask):
        seen.append((ids, [(a, b) for a, b, _ in chunks]))
        return pure(u, q, scene, edge, chunks, ids, mask)
    monkeypatch.setattr(module, 'expected_conditional_composite', spy)
    item = harness(tmp_path, n=n, edges=edges)
    before = copy.deepcopy(fixture.COUNTS)
    actual = losses(item)
    assert len(seen) == 2 and seen[0][0] is seen[1][0]
    assert seen[0][1] == seen[1][1]
    structured = item.net.fixture_structured
    q = build_soft_goal_target(structured['goal_candidates_world'], item.inputs['world_coord'][-1],
                               1., item.mask)
    assert not q.requires_grad
    u = structured['unary_score']
    edge = item.edge
    m = item.net
    lp = torch.log_softmax(m.jdv2_deploy, -1)
    lq = torch.log_softmax(m.jdv2_teacher, -1)
    left = m.jdv2_left+.1*m.jdv2_social[edge[0], None, :, None]
    energy = -torch.einsum('emkr,emlr->eklm', left, m.jdv2_right)/2**.5
    cp = -torch.logsumexp(lq[:, None, None, :]-energy, -1)
    cr = -torch.logsumexp(lp-energy, -1)
    ref = [torch.stack([explicit_conditional(u, q, item.inputs['scene_index'],
                        edge, c, row, item.mask) for row in item.last_ids]).mean() for c in (cp, cr)]
    kl = u.sum()*0
    for e in range(edge.shape[1]):
        for k in range(3):
            for j in range(3):
                kl = kl + q[edge[0, e], k]*q[edge[1, e], j]*(
                    lq[e].exp()*(lq[e]-lp[e, k, j])).sum()/edge.shape[1]
    for name, reference in zip(actual, [*ref, kl]):
        close(actual[name], reference, f'{n}_{edges}_{name}')
    total = .5*actual['jdv2_pl_post']+.5*actual['jdv2_pl_prior']+.07*actual['jdv2_relation_kl']
    reference = .5*ref[0]+.5*ref[1]+.07*kl
    ga = torch.autograd.grad(total, tuple(m.parameters()), allow_unused=True, retain_graph=True)
    gr = torch.autograd.grad(reference, tuple(m.parameters()), allow_unused=True)
    fixture.COUNTS['backward_calls'] += 2
    for index, (a, b) in enumerate(zip(ga, gr)):
        if a is None:
            assert b is None or b.numel() == 0
        else:
            close(a, b, f'{n}_{edges}_gradient_{index}')
            if edges:
                assert a.abs().max() > 0
    chunks = (edge.shape[1]+1)//2
    assert fixture.COUNTS['synthetic_encode_calls']-before['synthetic_encode_calls'] == 1
    assert fixture.COUNTS['draws']-before['draws'] == 1
    assert fixture.COUNTS['relation_chunks']-before['relation_chunks'] == chunks
    assert fixture.COUNTS['energy_factor_chunks']-before['energy_factor_chunks'] == chunks
    assert fixture.COUNTS['energy_mixtures']-before['energy_mixtures'] == 2*chunks


def draw_inputs():
    q = torch.tensor([[.2, .3, .5], [.4, .2, .4]])
    return torch.zeros_like(q), q, torch.tensor([-7, -7]), torch.tensor([[0], [1]]), torch.ones_like(q, dtype=torch.bool)


def draw(owner):
    z = owner.draw(*draw_inputs())
    owner.finish_update(False)
    return z


def test_rng_isolation_seed_snapshot_restore(tmp_path):
    args = fixture.args(tmp_path)
    args.jdv2_neighbor_seed = None
    a, b = ObjectiveRNG(args), ObjectiveRNG(args)
    evaluation = torch.Generator().manual_seed(881)
    states = (torch.random.get_rng_state(), random.getstate(), np.random.get_state(), evaluation.get_state())
    for _ in range(3):
        equal(draw(a), draw(b))
    snapshot = a.snapshot(9, 'resume_last')
    saved_bytes = snapshot['generator_state'].clone()
    expected = [draw(a) for _ in range(4)]
    equal(saved_bytes, snapshot['generator_state'])
    b.restore(snapshot, 9)
    for z in expected:
        equal(z, draw(b))
    changed = copy.copy(args); changed.jdv2_neighbor_seed = 123
    c = ObjectiveRNG(changed)
    assert any(not torch.equal(x, draw(c)) for x in expected)
    equal(states[0], torch.random.get_rng_state())
    assert states[1] == random.getstate()
    assert np.array_equal(states[2][1], np.random.get_state()[1])
    equal(states[3], evaluation.get_state())
    changed.jdv2_cache_source_commit = 'f'*40
    assert semantic_fingerprint(changed) == semantic_fingerprint(args)


@pytest.mark.parametrize('kind', ['negative', 'mass', 'trainable', 'mask', 'dtype', 'device', 'scene', 'edge'])
def test_invalid_request_does_not_consume(tmp_path, kind):
    owner = ObjectiveRNG(fixture.args(tmp_path))
    values = list(draw_inputs())
    if kind == 'negative': values[1][0, 0] = -1
    if kind == 'mass': values[1] *= 2
    if kind == 'trainable': values[1].requires_grad_()
    if kind == 'mask': values[4][0] = False
    if kind == 'dtype': values[0] = values[0].half()
    if kind == 'device': values[0] = torch.empty((2, 3), device='meta')
    if kind == 'scene': values[2][1] = 100
    if kind == 'edge': values[3][1] = 0
    before = owner.snapshot(9, 'resume_last')
    with pytest.raises((ValueError, TypeError)):
        owner.draw(*values)
    equal(before, owner.snapshot(9, 'resume_last'))


@pytest.mark.parametrize('key,value', [
    ('device', 'cuda:0'), ('amp_enabled', True), ('amp_dtype', 'bf16'),
    ('jdv2_neighbor_draws', 8), ('jdv2_goal_objective', 'unknown'),
    ('training_stage', 'joint_trajectory'), ('training_stage', 'joint_finetune'),
    ('jdv2_latent_objective', 'v2_marginal_responsibility'),
    ('goal_model_type', 'independent'), ('dataset', 'sdd'),
    ('trajectory_coupling', 'multiway_v4'), ('jdv2_neighbor_seed', -1)])
def test_mode_gate_before_constructor(tmp_path, monkeypatch, key, value):
    from src.trainer import trainer
    import src.trainer as module
    args = fixture.args(tmp_path)
    setattr(args, key, value)
    def forbidden(*a, **kw):
        pytest.fail('device/data/network reached before mode rejection')
    monkeypatch.setattr(module, 'get_dataloader', forbidden)
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    with pytest.raises(ValueError):
        trainer(args)


def test_parser_and_eval_reject_before_draw(tmp_path):
    raw = parser.get_parser().parse_args([
        '--dataset', 'eth5', '--test_set', 'eth', '--data_augmentation', 'False',
        '--device', 'cpu', '--goal_model_type', 'jdv2',
        '--training_stage', 'joint_goal', '--use_scene_latent', 'False',
        '--jdv2_latent_objective', 'strict_no_z', '--jdv2_goal_objective', MC])
    checked = parser.check_and_add_additional_args(raw)
    assert checked.jdv2_neighbor_draws == 4
    item = harness(tmp_path)
    item.net.eval()
    before = item.jdv2_objective_rng.snapshot(9, 'resume_last')
    with pytest.raises(RuntimeError, match='eval mode'):
        losses(item)
    equal(before, item.jdv2_objective_rng.snapshot(9, 'resume_last'))


def checkpoint(item, path):
    run_steps(item, 0, 3)
    save(item, last_epoch=True)
    return torch.load(path, weights_only=False)


@pytest.mark.parametrize('mutation', [
    'missing_state', 'format_version', 'neighbor_draws', 'generator_device',
    'sampling_semantics_version', 'objective_semantic_fingerprint',
    'generator_state', 'missing_rng', 'draw_calls', 'sampled_agent_draws',
    'successful_optimizer_updates', 'stage_progress', 'pending_backward',
    'checkpoint_role', 'stage_total_optimizer_steps', 'torch_version',
    'top_progress', 'scheduler_epoch', 'optimizer', 'scheduler', 'architecture', 'provenance',
    'optimizer_step', 'optimizer_shape', 'scheduler_progress', 'scheduler_lr'])
def test_checkpoint_matrix_transactional(tmp_path, mutation):
    source = harness(tmp_path/'source')
    path = tmp_path/'source'/'saved_models'/'last_model.pt'
    payload = checkpoint(source, path)
    state = payload[FIELD]
    if mutation == 'missing_state': del payload[FIELD]
    elif mutation == 'missing_rng': del state['generator_state']
    elif mutation == 'generator_state': state[mutation] = torch.tensor([1], dtype=torch.uint8)
    elif mutation in ('draw_calls', 'sampled_agent_draws', 'successful_optimizer_updates'): state[mutation] = -1
    elif mutation in ('format_version', 'neighbor_draws', 'stage_total_optimizer_steps'): state[mutation] += 1
    elif mutation == 'stage_progress': state[mutation] = .99
    elif mutation == 'pending_backward': state[mutation] = True
    elif mutation == 'top_progress': payload['stage_progress'] = .1
    elif mutation == 'scheduler_epoch': payload['mc_scheduler_epoch'] = 0
    elif mutation == 'optimizer': payload['optimizer_state_dict'] = None
    elif mutation == 'scheduler': payload['scheduler_state_dict'] = {}
    elif mutation == 'optimizer_step': next(iter(payload['optimizer_state_dict']['state'].values()))['step'] = torch.tensor(-1.)
    elif mutation == 'optimizer_shape': next(iter(payload['optimizer_state_dict']['state'].values()))['exp_avg'] = torch.ones(7)
    elif mutation == 'scheduler_progress': payload['scheduler_state_dict']['last_epoch'] = 0
    elif mutation == 'scheduler_lr': payload['scheduler_state_dict']['_last_lr'] = [99.]
    elif mutation == 'architecture': payload['architecture_config'] = {}
    elif mutation == 'provenance': payload['cache_manifest_hash'] = 'wrong'
    else: state[mutation] = 'wrong'
    bad = tmp_path/'corrupt.pt'; torch.save(payload, bad)
    target = harness(tmp_path/'target')
    before = copy.deepcopy((target.net.state_dict(), target.optimizer.state_dict(),
                            target.scheduler.state_dict(), target.jdv2_objective_rng.snapshot(9, 'resume_last')))
    with pytest.raises((ValueError, RuntimeError)):
        load(target, bad, baseline_initialization=True)
    equal(before, (target.net.state_dict(), target.optimizer.state_dict(),
                  target.scheduler.state_dict(), target.jdv2_objective_rng.snapshot(9, 'resume_last')))
    assert target._pending_training_state is None


def test_objective_roles_eval_and_legacy_matrix(tmp_path):
    item = harness(tmp_path/'source')
    path = tmp_path/'source'/'saved_models'/'last_model.pt'
    payload = checkpoint(item, path)
    legacy = harness(tmp_path/'legacy', 'mean_energy')
    with pytest.raises(ValueError, match='mean_energy'):
        load(legacy, path)
    load(legacy, path, weights_only=True)
    assert not hasattr(legacy, 'jdv2_objective_rng')
    del payload[FIELD]
    old = tmp_path/'old.pt'; torch.save(payload, old)
    load(legacy, old)
    with pytest.raises(ValueError, match='Missing MC'):
        load(harness(tmp_path/'other'), old, baseline_initialization=True)
    for flags, name in [({'best_epoch': True}, 'best_model.pt'), ({}, 'epoch_001.pt')]:
        save(item, **flags)
        snapshot = path.with_name(name)
        with pytest.raises(ValueError, match='checkpoint_role'):
            load(harness(tmp_path/'target'), snapshot)
        before = item.jdv2_objective_rng.snapshot(9, 'resume_last')
        load(item, snapshot, weights_only=True)
        equal(before, item.jdv2_objective_rng.snapshot(9, 'resume_last'))
    save(legacy, last_epoch=True)
    assert FIELD not in torch.load(tmp_path/'legacy'/'saved_models'/'last_model.pt', weights_only=False)


def test_error_after_draw_pending_and_skipped_update(tmp_path):
    item = harness(tmp_path)
    original = item.net.jdv2_joint_energy.effective_energy
    item.net.jdv2_joint_energy.effective_energy = lambda *a: (_ for _ in ()).throw(ValueError('cost failure'))
    with pytest.raises(ValueError, match='cost failure'): losses(item)
    assert item.jdv2_objective_rng.draw_calls == 1
    assert item._stage_optimizer_steps_completed == 0
    with pytest.raises(RuntimeError, match='pending'): save(item)
    with pytest.raises(RuntimeError, match='unfinished'): losses(item)
    item = harness(tmp_path/'skip')
    losses(item)
    item._mc_optimizer_step(skip=True)
    assert item.jdv2_objective_rng.draw_calls == 1
    assert item._stage_optimizer_steps_completed == 0
    assert item.net.set_losses_coeffs()['jdv2_relation_kl'] == 0
    item = harness(tmp_path/'nan')
    values = losses(item)
    sum(values.values()).backward(); fixture.COUNTS['backward_calls'] += 1
    next(item.net.parameters()).grad.fill_(torch.nan)
    with pytest.raises(FloatingPointError): item._mc_optimizer_step()
    assert item.jdv2_objective_rng.pending_backward
    assert item._stage_optimizer_steps_completed == 0
    item = harness(tmp_path/'boundary')
    with pytest.raises(RuntimeError, match='scheduler'): save(item, last_epoch=True)


@pytest.mark.parametrize('failure', ['serialize', 'flush', 'file_fsync', 'replace', 'directory_fsync'])
def test_atomic_failure_policy(tmp_path, monkeypatch, failure):
    path = tmp_path/'checkpoint.pt'
    atomic_save({'version': 1}, path)
    old = path.read_bytes()
    def fail(*a, **kw): raise OSError('injected')
    if failure == 'serialize': monkeypatch.setattr(torch, 'save', fail)
    if failure == 'replace': monkeypatch.setattr(os, 'replace', fail)
    if failure == 'flush':
        original_fdopen = os.fdopen
        class FlushFailure:
            def __init__(self, handle): self.handle = handle
            def __enter__(self): return self
            def __exit__(self, *args): self.handle.close()
            def __getattr__(self, name): return getattr(self.handle, name)
            def flush(self): fail()
        monkeypatch.setattr(os, 'fdopen', lambda *a, **kw: FlushFailure(original_fdopen(*a, **kw)))
    if failure in ('file_fsync', 'directory_fsync'):
        original = os.fsync
        calls = [0]
        def fsync(fd):
            calls[0] += 1
            if calls[0] == (1 if failure == 'file_fsync' else 2): fail()
            return original(fd)
        monkeypatch.setattr(os, 'fsync', fsync)
    with pytest.raises(CheckpointDurabilityError if failure == 'directory_fsync' else OSError) as error:
        atomic_save({'version': 2}, path)
    if failure == 'directory_fsync':
        assert error.value.committed
        assert torch.load(path, weights_only=False)['version'] == 2
    else:
        assert path.read_bytes() == old
    assert not list(tmp_path.glob('.mc-checkpoint-*'))


def test_separate_process_exact_resume(tmp_path):
    full = harness(tmp_path/'full')
    continuous = run_steps(full, 0, 9)
    partial = harness(tmp_path/'partial')
    prefix = run_steps(partial, 0, 3)
    save(partial, last_epoch=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(Path.cwd())+os.pathsep+str(Path.cwd()/'tests'))
    child = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('mc_wiring_fixture.py')),
                            str(tmp_path/'partial')], env=env, text=True, capture_output=True, check=True)
    result = json.loads(child.stdout)
    assert continuous == prefix+result['trace']
    MEASUREMENTS['resume'] = dict(exact=True, continuous_updates=9, prefix_updates=3,
        child_updates=6, independent_resume_processes=1, child_counts=result['counts'],
        compared_fields=list(continuous[0]),
        trace_sha256=hashlib.sha256(json.dumps(continuous, sort_keys=True).encode()).hexdigest())


def test_fresh_baseline_initialization_resets_mc_training_state(tmp_path):
    item = harness(tmp_path/'fresh')
    run_steps(item, 0, 3)
    # Fixture has only jdv2_* raw parameters: an empty legacy state represents
    # their allowed absence. No actual baseline/checkpoint is read.
    path = tmp_path/'base.pt'
    torch.save({'model_state_dict': {}, 'training_stage': 'baseline', 'epoch': 87}, path)
    load(item, path, baseline_initialization=True)
    assert item._stage_optimizer_steps_completed == 0
    assert item.args.jdv2_stage_progress == 0
    assert item.jdv2_objective_rng.draw_calls == 0
    assert not item.optimizer.state
    assert item.scheduler.last_epoch == 0
    equal(item.jdv2_objective_rng.generator.get_state(),
          ObjectiveRNG(item.args).generator.get_state())


def test_direct_invalid_rng_restore_is_transactional_and_fp64_draw(tmp_path):
    owner = ObjectiveRNG(fixture.args(tmp_path))
    values = list(draw_inputs()); values[0] = values[0].double()
    values[1] = torch.tensor([[.25, .25, .5], [.5, .25, .25]], dtype=torch.float64)
    owner.draw(*values); owner.finish_update(False)
    before = owner.snapshot(9, 'resume_last')
    bad = copy.deepcopy(before); bad['generator_state'] = torch.zeros(3, dtype=torch.uint8)
    with pytest.raises(ValueError): owner.restore(bad, 9)
    equal(before, owner.snapshot(9, 'resume_last'))


@pytest.mark.parametrize('kind', ['mask', 'scene', 'unary_dtype', 'amp'])
def test_actual_wiring_invalid_before_draw(tmp_path, kind):
    item = harness(tmp_path)
    before = item.jdv2_objective_rng.snapshot(9, 'resume_last')
    if kind == 'mask': item.mask[0] = False
    if kind == 'scene': item.inputs['scene_index'][1] = 999
    if kind == 'unary_dtype':
        original = item.net.encode
        def encode(*a, **kw):
            _, values = original(*a, **kw)
            values['unary_score'] = values['unary_score'].half()
            return None, values
        item.net.encode = encode
    if kind == 'amp':
        with torch.autocast('cpu', dtype=torch.bfloat16):
            with pytest.raises(ValueError): losses(item)
    else:
        with pytest.raises((ValueError, TypeError)): losses(item)
    equal(before, item.jdv2_objective_rng.snapshot(9, 'resume_last'))


def test_deferred_production_restore_and_semantic_mismatch(tmp_path):
    source = harness(tmp_path/'source')
    path = tmp_path/'source'/'saved_models'/'last_model.pt'
    checkpoint(source, path)
    target = harness(tmp_path/'target')
    del target.optimizer, target.scheduler
    load(target, path)
    # Actual constructor loads before train() creates the optimizer.
    assert target.jdv2_objective_rng.draw_calls == 0
    target.optimizer = target._set_optimizer(target.net.parameters())
    target.scheduler = target._set_scheduler(target.optimizer)
    target._restore_mc_training_state(target._pending_training_state)
    equal(target.jdv2_objective_rng.snapshot(9, 'resume_last'),
          source.jdv2_objective_rng.snapshot(9, 'resume_last'))
    equal(target.optimizer.state_dict(), source.optimizer.state_dict())
    equal(target.scheduler.state_dict(), source.scheduler.state_dict())
    for name, value in [('goal_soft_sigma', 2.), ('jdv2_cache_manifest_hash', 'new'),
                        ('jdv2_source_checkpoint_hash', 'new'), ('clean_split_manifest_hash', 'new')]:
        other = harness(tmp_path/'other')
        setattr(other.args, name, value)
        other._initialize_mc_objective()
        with pytest.raises(ValueError, match='fingerprint'): load(other, path)


@pytest.mark.parametrize('kind', ['nan_cost', 'optimizer_error'])
def test_post_draw_failure_never_reseeds_or_counts_update(tmp_path, kind):
    item = harness(tmp_path)
    initial = item.jdv2_objective_rng.generator.get_state().clone()
    if kind == 'nan_cost':
        original = item.net.jdv2_joint_energy.effective_energy
        item.net.jdv2_joint_energy.effective_energy = lambda *a: original(*a)*torch.nan
        with pytest.raises(ValueError, match='finite'): losses(item)
    else:
        sum(losses(item).values()).backward()
        fixture.COUNTS['backward_calls'] += 1
        def failed_step(): raise RuntimeError('optimizer injected failure')
        item.optimizer.step = failed_step
        with pytest.raises(RuntimeError, match='injected'): item._mc_optimizer_step()
    assert not torch.equal(initial, item.jdv2_objective_rng.generator.get_state())
    assert item.jdv2_objective_rng.draw_calls == 1
    assert item._stage_optimizer_steps_completed == 0
    assert item.jdv2_objective_rng.pending_backward
    with pytest.raises(RuntimeError, match='pending'): save(item)

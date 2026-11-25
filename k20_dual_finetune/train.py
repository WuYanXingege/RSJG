"""Fixed eight-epoch K20 fine-tuning, restricted to official training windows."""
from copy import deepcopy
import hashlib
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from bcep.contracts import canonical_hash, digest
from bcep.motion_frame import frame, to_local
from bcep_formal.control import pause_requested
from bcep_joint.common import read, restore_rng, rng_state, save, seed_all, write
from joint_flow_dependency.inference import encode_observed
from joint_flow_dependency.model import delta_nll, undirected_edges
from joint_flow_dependency.sampling import base_noise
from joint_flow_fourfold.common import immutable
from joint_flow_fourfold.inputs import state_hash
from .forward import check_initial_replay, endpoint_forward


RECIPE = dict(seed=3201, train_noise_seed=2035, epochs=8, logical_windows=16,
              dependence_lr=1e-4, head_lr=2e-5, weight_decay=1e-4,
              clip_grad_norm=5., delta_nll_weight=.1, M=256, K=20)
TRAINABLE_PROVIDER = ('head.readout.2.weight', 'head.readout.2.bias')
CHECKPOINT_INTERVAL = 10


def _stop_requested(should_yield):
    return pause_requested() or (should_yield is not None and should_yield())


def _progress(output, phase, **details):
    result = dict(phase=phase, unix=time.time(), **details)
    write(output/'PROGRESS.json', result)
    print(result, flush=True)


def _tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _provider_frozen_hash(provider):
    return canonical_hash({name: _tensor_hash(value) for name, value in provider.state_dict().items()
                           if name not in TRAINABLE_PROVIDER})


def _atomic_alias(path, target):
    temp = path.with_suffix('.link')
    temp.unlink(missing_ok=True)
    os.link(target, temp)
    temp.replace(path)


def _optimizer(dependence, provider):
    groups = []
    modules = (
        ('dependence', dependence, RECIPE['dependence_lr']),
        ('readout', provider.head.readout[-1], RECIPE['head_lr']),
    )
    for prefix, module, learning_rate in modules:
        for use_decay in (True, False):
            selected = [(name, param) for name, param in module.named_parameters()
                        if (param.ndim > 1 and not name.endswith('bias')
                            and 'norm' not in name.lower()) == use_decay]
            groups.append(dict(params=[param for _, param in selected],
                               names=[f'{prefix}.{name}' for name, _ in selected],
                               lr=learning_rate,
                               weight_decay=RECIPE['weight_decay'] if use_decay else 0.))
    return torch.optim.AdamW(groups)


def _endpoint_terms(goals, truth):
    error = torch.linalg.vector_norm(goals.float() - truth[None].float(), dim=-1)
    return error.amin(0), error.mean(1).amin()


def _record_token(inputs, index, cache_key):
    member = inputs.split['members'][index]
    return canonical_hash(dict(cache_key=cache_key, index=index,
                               member=member, npz_sha256=inputs.split['records'][index]['sha256']))


@torch.no_grad()
def _make_record(inputs, index, token, teacher_provider, teacher_dependence, device):
    # Every tensor here is from the official train partition. The target is
    # stored separately from all inputs passed into the observed-only forward.
    batch = inputs.data.batch([index], 'train', truth=True).to(device)
    assert len(batch.observed) == 1 and batch.valid.all()
    observed = batch.observed[0]
    maps = batch.semantic_maps[0]
    valid = batch.valid[0]
    truth = batch.future[0, :, -1]
    context, features = encode_observed(teacher_provider, batch)
    edges = undirected_edges(observed)
    n = len(observed)
    assert context.shape == (n, 128) and features.shape == (n, 128, 32, 32)
    coverage = (maps.float().sum(1, keepdim=True) > 0).cpu().numpy().reshape(n, -1)
    coverage_bits = torch.from_numpy(np.packbits(coverage, axis=1))
    scene = batch.scene_ids[0]
    agents = batch.agent_ids[0]
    noise_seed = RECIPE['train_noise_seed']
    epsilon, xi = base_noise(scene, agents, edges, seed=noise_seed, count=RECIPE['M'])
    goals = endpoint_forward(teacher_provider, teacher_dependence, observed, maps,
                             valid, context, features, edges, epsilon.to(device), xi.to(device))
    minima, teacher_jfde = _endpoint_terms(goals['corrected'], truth)
    base = teacher_provider.base
    origin, rotation, scale = frame(observed, 12, base.dt, base.scales[-1])
    local = to_local(truth, origin, rotation, scale)
    latent_truth, _ = base.flow.transform(local, context, 12*base.dt, inverse=True)
    assert torch.isfinite(latent_truth).all()
    noise_key = dict(scene=scene, agents=list(agents), seed=noise_seed,
                     edges=edges.cpu().tolist(), streams=['personal', 'undirected-edge'])
    return dict(index=index, token=token, agents=n,
                observed=observed.cpu().contiguous(), context=context.cpu().contiguous(),
                features=features.cpu().contiguous(), coverage_bits=coverage_bits,
                edges=edges.cpu(), epsilon=epsilon.contiguous(), xi=xi.contiguous(),
                noise_key_sha256=canonical_hash(noise_key),
                epsilon_sha256=_tensor_hash(epsilon), xi_sha256=_tensor_hash(xi),
                truth=truth.cpu().contiguous(), latent_truth=latent_truth.cpu(),
                teacher_minfde_sum_m=float(minima.double().sum()),
                teacher_jfde_m=float(teacher_jfde))


def _cache(inputs, teacher_provider, teacher_dependence, output, identity, device, should_yield):
    cache = output/'train_cache'
    cache.mkdir(exist_ok=True)
    cache_key = canonical_hash(dict(identity=identity, purpose='train',
                                    sampler='joint-flow-dependence-v1',
                                    noise_seed=RECIPE['train_noise_seed'], M=RECIPE['M'], K=RECIPE['K']))
    indices = inputs.split['train_indices']
    manifest = cache/'MANIFEST.json'
    prior = read(manifest) if manifest.exists() else None
    if prior is not None:
        assert prior['cache_key'] == cache_key and len(prior['rows']) == len(indices)
    rows = []
    started = time.monotonic()
    for position, index in enumerate(indices):
        path = cache/f'{position:04d}.pt'
        side = path.with_suffix('.json')
        token = _record_token(inputs, index, cache_key)
        if not path.exists():
            assert prior is None and not side.exists()
            record = _make_record(inputs, index, token, teacher_provider, teacher_dependence, device)
            save(record, path)
        else:
            record = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        assert record['index'] == index and record['token'] == token
        n = len(inputs.split['members'][index]['agent_ids'])
        assert record['agents'] == n
        row = {name: record[name] for name in ('index', 'token', 'agents', 'teacher_minfde_sum_m',
                                               'teacher_jfde_m', 'noise_key_sha256',
                                               'epsilon_sha256', 'xi_sha256')}
        row.update(file=path.name, sha256=digest(path))
        immutable(side, row)
        if prior is not None:
            assert prior['rows'][position] == row
        rows.append(row)
        complete = position + 1
        if complete % 25 == 0 or complete == len(indices):
            _progress(output, 'TRAIN_TEACHER_CACHE', fold=inputs.fold, windows=complete,
                      total=len(indices), wall_seconds=time.monotonic()-started)
            if prior is None and _stop_requested(should_yield):
                return None
    m0 = math.fsum(row['teacher_minfde_sum_m'] for row in rows)/sum(row['agents'] for row in rows)
    j0 = math.fsum(row['teacher_jfde_m'] for row in rows)/len(rows)
    assert math.isfinite(m0) and math.isfinite(j0) and m0 > 0 and j0 > 0
    immutable(manifest, dict(cache_key=cache_key, rows=rows, M0_m=m0, J0_m=j0))
    write(output/'TEACHER_REFERENCES.json', dict(M0_m=m0, J0_m=j0,
          train_windows=len(rows), train_agents=sum(row['agents'] for row in rows),
          teacher_cache_sha256=digest(manifest), noise_seed=RECIPE['train_noise_seed']))
    return cache, rows, m0, j0


def _preflight(inputs, teacher_provider, teacher_dependence, student_provider,
               student_dependence, output, identity, device):
    path = output/'PREFLIGHT.json'
    identity_sha = canonical_hash(identity)
    if path.exists():
        receipt = read(path)
        assert receipt['passed'] and receipt['identity_sha256'] == identity_sha
        return
    train_indices = inputs.split['train_indices']
    position, index = next((pos, i) for pos, i in enumerate(train_indices)
                           if 3 <= len(inputs.split['members'][i]['agent_ids']) <= 12)
    batch = inputs.data.batch([index], 'train', truth=True).to(device)
    context, features = encode_observed(teacher_provider, batch)
    receipt = check_initial_replay(teacher_provider, teacher_dependence, student_provider,
                                   student_dependence, batch, context, features,
                                   RECIPE['train_noise_seed'])
    assert state_hash(student_dependence) == identity['teacher']['dependence']
    assert state_hash(student_provider) == identity['teacher']['provider']
    assert not any(p.grad is not None for p in student_provider.parameters())
    assert not any(p.grad is not None for p in student_dependence.parameters())
    receipt.update(window=f'train_{position+1:04d}', identity_sha256=identity_sha,
                   parameters_unchanged=True, gradients_left_in_optimizer=False)
    immutable(path, receipt)
    _progress(output, 'PREFLIGHT_PASSED', fold=inputs.fold, receipt=receipt)


def _checkpoint_index(folder, history):
    # This experiment selects a fixed epoch and has no intermediate validation.
    # Train loss is logged, never used to fabricate a validation ranking.
    entries = [dict(epoch=row['epoch'], path=row['file'], sha256=row['sha256'],
                    validation_metric=None) for row in history]
    write(folder/'TOP5.json', dict(status='not_ranked_fixed_epoch_8', requested_rank_count=5,
                                   selected_epoch=8 if len(history) == 8 else None,
                                   retained_completed_epochs=entries, ranked_epochs=[]))
    write(folder/'EPOCH_METRICS.json', history)


def _load_latest(folder, run_identity):
    last = folder/'last.pt'
    snapshots = sorted(folder.glob('epoch_*.pt'))
    if not (last.exists() or snapshots):
        return None
    state = torch.load(last if last.exists() else snapshots[-1], map_location='cpu', weights_only=False)
    if snapshots and int(snapshots[-1].stem.rsplit('_', 1)[1]) > state['completed_epoch']:
        # Recover a snapshot written immediately before an interrupted last-link update.
        state = torch.load(snapshots[-1], map_location='cpu', weights_only=False)
    assert state['identity'] == run_identity and state['recipe'] == RECIPE
    for row in state['history']:
        row['sha256'] = digest(folder/row['file'])
    return state


def _student_window(cache, row, provider, dependence, device):
    record = torch.load(cache/row['file'], map_location='cpu', weights_only=True, mmap=True)
    assert record['token'] == row['token'] and record['index'] == row['index']
    n = record['agents']
    bits = record['coverage_bits'].numpy()
    coverage = torch.from_numpy(np.unpackbits(bits, axis=1).reshape(n, 1, 256, 256)).to(device)
    tensors = {key: record[key].to(device) for key in ('observed', 'context', 'features',
               'edges', 'epsilon', 'xi', 'truth', 'latent_truth')}
    result = endpoint_forward(provider, dependence, tensors['observed'], coverage,
        torch.ones(n, device=device, dtype=torch.bool), tensors['context'], tensors['features'],
        tensors['edges'], tensors['epsilon'], tensors['xi'])
    agent_minima, jfde = _endpoint_terms(result['corrected'], tensors['truth'])
    penalty = (jfde - row['teacher_jfde_m']).relu()
    nll = delta_nll(result['R'], tensors['latent_truth'])
    return agent_minima.sum(), jfde, penalty, nll


def train_fold(inputs, model, teacher_provider, teacher_dependence, output, device='cuda',
               *, should_yield=None):
    """Return epoch-8 identity, or None after yielding the GPU at a saved boundary."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    for network in (model, teacher_provider, teacher_dependence):
        network.eval().requires_grad_(False)
    teacher_hashes = dict(backbone=state_hash(model), provider=state_hash(teacher_provider),
                          dependence=state_hash(teacher_dependence))
    identity = dict(fold=inputs.fold, upstream=inputs.identity, teacher=teacher_hashes,
                    recipe=RECIPE, train_members_sha256=canonical_hash(
                        [inputs.split['members'][i] for i in inputs.split['train_indices']]))
    immutable(output/'TRAIN_IDENTITY.json', identity)
    result_path = output/'TRAINING_RESULT.json'
    if result_path.exists():
        result = read(result_path)
        assert result['identity'] == identity and result['completed_epochs'] == RECIPE['epochs']
        assert digest(result['final_checkpoint']) == result['final_checkpoint_sha256']
        return result
    student_provider = deepcopy(teacher_provider)
    student_dependence = deepcopy(teacher_dependence).requires_grad_(True)
    student_provider.head.readout[-1].requires_grad_(True)
    assert state_hash(student_provider) == teacher_hashes['provider']
    assert state_hash(student_dependence) == teacher_hashes['dependence']
    assert tuple(name for name, param in student_provider.named_parameters()
                 if param.requires_grad) == TRAINABLE_PROVIDER
    assert sum(p.numel() for p in student_dependence.parameters()) == 16964
    assert sum(p.numel() for p in student_provider.head.readout[-1].parameters()) == 130
    frozen_student = _provider_frozen_hash(student_provider)
    seed_all(RECIPE['seed'])
    generator = torch.Generator().manual_seed(RECIPE['seed'])
    _preflight(inputs, teacher_provider, teacher_dependence, student_provider,
               student_dependence, output, identity, device)
    cached = _cache(inputs, teacher_provider, teacher_dependence, output, identity, device, should_yield)
    if cached is None or _stop_requested(should_yield):
        return None
    cache, rows, m0, j0 = cached
    run_identity = dict(identity, teacher_reference_manifest_sha256=digest(cache/'MANIFEST.json'))
    opt = _optimizer(student_dependence, student_provider)
    params = [p for group in opt.param_groups for p in group['params']]
    checkpoint_dir = output/'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    state = _load_latest(checkpoint_dir, run_identity)
    if state is not None:
        student_dependence.load_state_dict(state['dependence'], strict=True)
        student_provider.head.readout[-1].load_state_dict(state['readout'], strict=True)
        opt.load_state_dict(state['optimizer'])
        restore_rng(state['rng'], generator)
    else:
        state = dict(identity=run_identity, recipe=RECIPE, update=0, completed_epoch=0,
                     order=None, cursor=0, history=[], sums={}, scheduler=None,
                     epoch_train_seconds=0., status='initialized')

    def persist(path):
        state.update(dependence=student_dependence.state_dict(),
                     readout=student_provider.head.readout[-1].state_dict(),
                     optimizer=opt.state_dict(), rng=rng_state(generator))
        save(state, path)

    while state['completed_epoch'] < RECIPE['epochs']:
        epoch = state['completed_epoch'] + 1
        if state['order'] is None:
            state['order'] = torch.randperm(len(rows), generator=generator).tolist()
            state['sums'] = dict(agent_minfde_sum=0., jfde_sum=0., degradation_sum=0.,
                                 delta_nll_sum=0., objective_sum=0., windows=0, agents=0, batches=0)
        while state['cursor'] < len(rows):
            start = time.monotonic()
            cursor = state['cursor']
            batch_rows = [rows[i] for i in state['order'][cursor:cursor+RECIPE['logical_windows']]]
            batch_n = sum(row['agents'] for row in batch_rows)
            batch_w = len(batch_rows)
            weights = (1/(batch_n*m0), 1/(batch_w*j0), 1/(batch_w*j0),
                       RECIPE['delta_nll_weight']/batch_w)
            opt.zero_grad(set_to_none=True)
            batch_sums = torch.zeros(5, device=device, dtype=torch.float64)
            for row in batch_rows:
                terms = _student_window(cache, row, student_provider, student_dependence, device)
                loss = sum(term*weight for term, weight in zip(terms, weights))
                loss.backward()
                batch_sums += torch.stack((*[term.detach() for term in terms], loss.detach())).double()
            grad = torch.nn.utils.clip_grad_norm_(params, RECIPE['clip_grad_norm'], error_if_nonfinite=True)
            opt.step()
            if not torch.cat([param.detach().flatten() for param in params]).isfinite().all():
                raise FloatingPointError('nonfinite updated parameters')
            state['cursor'] += batch_w
            state['update'] += 1
            counts = dict(windows=batch_w, agents=batch_n, batches=1)
            for key, value in zip(('agent_minfde_sum', 'jfde_sum', 'degradation_sum',
                                   'delta_nll_sum', 'objective_sum'), batch_sums.tolist()):
                state['sums'][key] += value
            for key, value in counts.items():
                state['sums'][key] += value
            state['epoch_train_seconds'] += time.monotonic()-start
            is_boundary = state['update'] == 1 or state['update'] % CHECKPOINT_INTERVAL == 0
            if is_boundary:
                state['status'] = 'update_complete'
                persist(checkpoint_dir/'last.pt')
                _progress(output, 'TRAIN_UPDATE', fold=inputs.fold, epoch=epoch, epochs=RECIPE['epochs'],
                          update=state['update'], windows=state['cursor'], total=len(rows),
                          batch_objective=float(batch_sums[-1]), grad_norm=float(grad),
                          epoch_train_seconds=state['epoch_train_seconds'])
                if _stop_requested(should_yield):
                    return None
        totals = state['sums']
        assert totals['windows'] == len(rows) and totals['agents'] == sum(row['agents'] for row in rows)
        filename = f'epoch_{epoch:03d}.pt'
        row = dict(epoch=epoch, update=state['update'], file=filename,
                   train_seconds=state['epoch_train_seconds'],
                   minFDE_m=totals['agent_minfde_sum']/totals['agents'],
                   JFDE_m=totals['jfde_sum']/totals['windows'],
                   JFDE_degradation_m=totals['degradation_sum']/totals['windows'],
                   delta_nll_nat_per_person=totals['delta_nll_sum']/totals['windows'],
                   objective_mean_per_update=totals['objective_sum']/totals['batches'],
                   windows=totals['windows'], agents=totals['agents'])
        assert _provider_frozen_hash(student_provider) == frozen_student
        state['history'].append(row)
        state.update(completed_epoch=epoch, cursor=0, order=None, sums={},
                     epoch_train_seconds=0., status='complete' if epoch == RECIPE['epochs'] else 'epoch_complete')
        target = checkpoint_dir/filename
        persist(target)
        row['sha256'] = digest(target)
        _atomic_alias(checkpoint_dir/'last.pt', target)
        _checkpoint_index(checkpoint_dir, state['history'])
        _progress(output, 'TRAIN_EPOCH_COMPLETE', fold=inputs.fold, **row)
        if epoch < RECIPE['epochs'] and _stop_requested(should_yield):
            return None
    assert len(state['history']) == RECIPE['epochs']
    for row in state['history']:
        path = checkpoint_dir/row['file']
        assert digest(path) == row['sha256']
        snapshot = torch.load(path, map_location='cpu', weights_only=False)
        assert snapshot['completed_epoch'] == row['epoch'] and snapshot['cursor'] == 0
        assert all(key in snapshot for key in ('dependence', 'readout', 'optimizer', 'scheduler',
                   'rng', 'completed_epoch', 'update', 'order', 'identity', 'recipe'))
    target = checkpoint_dir/state['history'][-1]['file']
    _atomic_alias(checkpoint_dir/'final.pt', target)
    _atomic_alias(checkpoint_dir/'selected.pt', target)
    _atomic_alias(checkpoint_dir/'last.pt', target)
    _checkpoint_index(checkpoint_dir, state['history'])
    assert _provider_frozen_hash(student_provider) == frozen_student
    assert teacher_hashes == dict(backbone=state_hash(model), provider=state_hash(teacher_provider),
                                  dependence=state_hash(teacher_dependence))
    result = dict(status='COMPLETE', fold=inputs.fold, final_checkpoint=str(target.resolve()),
                  final_checkpoint_sha256=digest(target), completed_epochs=RECIPE['epochs'],
                  updates=state['update'], identity=identity, recipe=RECIPE,
                  checkpoint_selection='fixed epoch 8', M0_m=m0, J0_m=j0,
                  train_windows=len(rows), train_agents=sum(row['agents'] for row in rows),
                  DDIM_calls=0, test_access=False, frozen_parameters_and_statistics_unchanged=True,
                  history=state['history'], optimizer_groups=[{key: group[key] for key in
                      ('names', 'lr', 'weight_decay')} for group in opt.param_groups])
    write(result_path, result)
    return result

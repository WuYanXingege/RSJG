"""Fit only the group-slot matcher with hard JFDE and a surrogate gradient."""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import time

import numpy as np
import torch

from bcep.contracts import canonical_hash, digest, require
from bcep_joint.common import read, restore_rng, rng_state, save, seed_all, write
from direct_k20.model import K
from direct_k20.train import _atomic_alias, _progress
from joint_flow_fourfold.common import immutable

from .matcher import Matcher, straight_through_assignment, window_features


RECIPE = dict(
    seed=3201, epochs=12, logical_windows=16, optimizer='AdamW',
    learning_rate=1e-3, weight_decay=1e-4, precision='FP32', K=K,
    hungarian='maximum-score bijection', sinkhorn_iterations=20, sinkhorn_temperature=.5,
    softmin_temperature_m=.1, window_weighting='one equal weight per window',
    selection='only completed epoch 12 is evaluated',
    retention='every complete epoch plus final/selected and last',
    recovery='atomic last after each complete logical batch',
)
METER_ATOL = 1e-6
METRICS = ('JADE', 'JFDE', 'minADE', 'minFDE')


@dataclass(frozen=True)
class TrainingWindow:
    row: torch.Tensor
    column: torch.Tensor
    error: torch.Tensor
    people: int

    @property
    def groups(self):
        return self.error.shape[0]


def _load_window(cache, index):
    sample = cache.load(index)
    observed = np.asarray(sample['observed'], dtype=np.float32)
    goals = np.asarray(sample['g0'], dtype=np.float32)
    features = window_features(observed, goals)
    groups = np.asarray(sample['groups'])
    error = torch.as_tensor(sample['E'], dtype=torch.float32)
    row = torch.from_numpy(features['row_features'])
    column = torch.from_numpy(features['column_features'])
    require(np.array_equal(groups, features['group_index']), 'cached loss groups are not in matcher order')
    require(error.shape == (len(row), K) and len(row) == len(column) > 0
            and bool(torch.isfinite(error).all()) and not bool((error < 0).any()),
            'invalid train-only group endpoint errors')
    return TrainingWindow(row, column, error, len(observed))


def _pack(windows, device):
    count = len(windows)
    max_groups = max(window.groups for window in windows)
    row = torch.zeros((count, max_groups, *windows[0].row.shape[1:]), dtype=torch.float32)
    column = torch.zeros((count, max_groups, *windows[0].column.shape[1:]), dtype=torch.float32)
    error = torch.zeros((count, max_groups, K), dtype=torch.float32)
    group_valid = torch.zeros((count, max_groups), dtype=torch.bool)
    people = torch.empty(count, dtype=torch.float32)
    for i, window in enumerate(windows):
        row[i, :window.groups] = window.row
        column[i, :window.groups] = window.column
        error[i, :window.groups] = window.error
        group_valid[i, :window.groups] = True
        people[i] = window.people
    return tuple(x.to(device) for x in (row, column, error, group_valid, people))


def hard_jfde_loss(model, queries, row, column, error, group_valid, people):
    """Keep the exact Hungarian/min forward values and the prescribed surrogate gradient."""
    scores = model(row, column, queries, group_valid)
    assignment, permutation = straight_through_assignment(scores, group_valid)
    costs = (assignment * error.unsqueeze(-2)).sum(dim=(1, 3)) / people[:, None]
    hard = costs.detach().min(dim=-1).values
    temperature = RECIPE['softmin_temperature_m']
    softmin = -temperature * torch.logsumexp(-costs/temperature, dim=-1)
    loss = (hard + (softmin-softmin.detach())).mean()
    return loss, hard, permutation


def change_flags(permutation, group_valid):
    """Identify changes to co-occurrence independently of the output slot names."""
    slots = torch.arange(K, device=permutation.device)
    changed_name = ((permutation != slots) & group_valid[..., None]).any(dim=(1, 2))
    relative = ((permutation != permutation[:, :1]) & group_valid[..., None]).any(dim=(1, 2))
    return relative, changed_name & ~relative


def _preflight(model, queries, examples, output, identity, device):
    path = output / 'PREFLIGHT.json'
    if path.exists():
        receipt = read(path)
        require(receipt['passed'] and receipt['identity_sha256'] == canonical_hash(identity),
                'mismatched training preflight')
        return receipt
    order = torch.randperm(len(examples), generator=torch.Generator().manual_seed(RECIPE['seed'])).tolist()
    windows = [examples[i] for i in order[:RECIPE['logical_windows']]]
    require(any(window.groups > 1 for window in windows), 'first logical batch has no multi-group windows')
    row, column, error, valid, people = _pack(windows, device)
    loss, hard, permutation = hard_jfde_loss(model, queries, row, column, error, valid, people)
    slots = torch.arange(K, device=device).expand_as(permutation)
    require(torch.equal(permutation, slots), 'initial hard matching is not exactly the identity')
    require(torch.equal(permutation.sort(dim=-1).values, slots), 'preflight produced a nonbijective assignment')
    original = (error.sum(dim=1)/people[:, None]).min(dim=-1).values
    require(torch.allclose(hard, original, atol=METER_ATOL, rtol=0)
            and torch.equal(loss.detach(), hard.mean()), 'forward loss differs from native hard JFDE')
    gradients = torch.autograd.grad(loss, tuple(model.parameters()))
    require(all(bool(torch.isfinite(value).all()) for value in gradients), 'nonfinite preflight gradients')
    maximum = max(float(value.abs().max()) for value in gradients)
    require(maximum > 0 and all(p.grad is None for p in model.parameters()),
            'surrogate gradient is zero or the preflight wrote optimizer gradients')
    receipt = dict(passed=True, identity_sha256=canonical_hash(identity), windows=len(windows),
        multi_group_windows=sum(window.groups > 1 for window in windows), identity_permutation=True,
        hard_assignments_bijective=True, hard_jfde_max_abs_replay_m=float((hard-original).abs().max()),
        surrogate_gradient_max_abs=maximum, parameter_updates=0)
    immutable(path, receipt)
    return receipt


def _query_sha256(queries):
    return hashlib.sha256(queries.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _checkpoint_index(output, history):
    write(output / 'EPOCH_METRICS.json', history)
    write(output / 'checkpoints/TOP5.json', dict(
        selector=RECIPE['selection'], ranked_epoch_count=0, entries=[],
        snapshots=[{k: row[k] for k in ('epoch', 'file', 'sha256')} for row in history]))


def record_final_metrics(output, metrics):
    """Record only the requested epoch-12 development result."""
    checkpoints = Path(output) / 'checkpoints'
    path = checkpoints / 'TOP5.json'
    index = read(path)
    require(len(index['snapshots']) == RECIPE['epochs'], 'incomplete checkpoint inventory')
    selected = index['snapshots'][-1]
    require(selected['epoch'] == RECIPE['epochs']
            and digest(checkpoints / selected['file']) == selected['sha256'], 'epoch-12 snapshot changed')
    values = {key: float(metrics[key]) for key in METRICS}
    require(all(math.isfinite(x) for x in values.values()), 'final development metrics must be finite')
    index.update(ranked_epoch_count=1, entries=[dict(selected, rank=1, metrics=values, purpose='selection_dev')])
    write(path, index)
    _atomic_alias(checkpoints / 'best.pt', checkpoints / selected['file'])
    return index


def _resume(checkpoints, identity):
    last = checkpoints / 'last.pt'
    epochs = sorted(checkpoints.glob('epoch_*.pt'))
    if not last.exists() and not epochs:
        return None
    state = torch.load(last if last.exists() else epochs[-1], weights_only=False, map_location='cpu')
    if epochs and int(epochs[-1].stem.split('_')[1]) > state['completed_epoch']:
        state = torch.load(epochs[-1], weights_only=False, map_location='cpu')
    require(state['identity'] == identity and state['recipe'] == RECIPE, 'resume identity changed')
    require(len(epochs) == len(state['history']) == state['completed_epoch'], 'epoch inventory is inconsistent')
    for row in state['history']:
        sha = digest(checkpoints / row['file'])
        require(row.get('sha256', sha) == sha, 'complete epoch hash changed since recovery')
        row['sha256'] = sha
    return state


def _max_allocated(device):
    return torch.cuda.max_memory_allocated(device)/2**20 if device.type == 'cuda' else 0.


def train(cache, queries, output, config, *, device='cuda', should_yield=None):
    """Train the same matcher separately on one fold's locked train cache."""
    output = Path(output)
    checkpoints = output / 'checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=True)
    indices = tuple(cache.indices)
    require(len(indices) == len(set(indices)) > 0, 'train population must be unique')
    device = torch.device(device)
    queries = torch.as_tensor(queries, device=device, dtype=torch.float32).detach()
    require(queries.shape == (K, 32) and bool(torch.isfinite(queries).all()), 'invalid frozen student slot queries')
    identity = dict(recipe=RECIPE, config_sha256=canonical_hash(config), cache=cache.identity,
                    S0_sha256=config['S0_sha256'], ordered_train_indices_sha256=canonical_hash(list(indices)),
                    frozen_queries_sha256=_query_sha256(queries))
    immutable(output / 'TRAIN_IDENTITY.json', identity)
    result_path = output / 'TRAINING_RESULT.json'
    if result_path.exists():
        result = read(result_path)
        require(result['identity'] == identity and result['completed_epochs'] == RECIPE['epochs']
                and digest(result['final_checkpoint']) == result['final_sha256'], 'final train result changed')
        return result
    tick = time.monotonic()
    examples = [_load_window(cache, index) for index in indices]
    feature_seconds = time.monotonic()-tick
    multi_groups = sum(window.groups > 1 for window in examples)
    seed_all(RECIPE['seed'])
    model = Matcher().to(device=device, dtype=torch.float32).train()
    loader_rng = torch.Generator().manual_seed(RECIPE['seed'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=RECIPE['learning_rate'],
                                  weight_decay=RECIPE['weight_decay'])
    recovery = _resume(checkpoints, identity)
    preflight = _preflight(model, queries, examples, output, identity, device)
    if recovery is None:
        state = dict(identity=identity, recipe=RECIPE, completed_epoch=0, update=0, cursor=0,
                     order=None, sums={}, history=[], scheduler=None, first_update=None,
                     fixed_candidate=RECIPE['epochs'], feature_seconds=feature_seconds,
                     peak_gpu_allocated_mib=_max_allocated(device))
    else:
        state = recovery
        model.load_state_dict(state['matcher'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng'], loader_rng)
        state['feature_seconds'] += feature_seconds
        _checkpoint_index(output, state['history'])
        if state['first_update'] is not None:
            immutable(output / 'FIRST_UPDATE.json', state['first_update'])

    def persist(path):
        state.update(matcher=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state(loader_rng))
        save(state, path)

    if recovery is None:
        persist(checkpoints / 'last.pt')
    fold = config.get('fold', cache.identity.get('fold', ''))
    while state['completed_epoch'] < RECIPE['epochs']:
        epoch = state['completed_epoch'] + 1
        if state['order'] is None:
            state['order'] = torch.randperm(len(examples), generator=loader_rng).tolist()
            state['sums'] = dict(windows=0, hard_jfde_sum_m=0., multi_group_windows=0,
                                 relative_windows=0, global_relabel_windows=0, updates=0, skipped=0,
                                 gradient_norm_sum=0., gradient_norm_max=0., train_seconds=0.)
        while state['cursor'] < len(examples):
            if should_yield is not None and should_yield():
                _progress(output, 'YIELD_TO_COMPLETE_BATCH', fold=fold, epoch=epoch,
                          update=state['update'], windows=state['cursor'], total=len(examples))
                return None
            tick = time.monotonic()
            offset = state['cursor']
            batch = [examples[i] for i in state['order'][offset:offset+RECIPE['logical_windows']]]
            eligible = sum(window.groups > 1 for window in batch)
            row, column, error, valid, people = _pack(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, hard, permutation = hard_jfde_loss(model, queries, row, column, error, valid, people)
            relative, relabel = change_flags(permutation, valid)
            values = dict(windows=len(batch), hard_jfde_sum_m=float(hard.double().sum()),
                          multi_group_windows=eligible, relative_windows=int(relative.sum()),
                          global_relabel_windows=int(relabel.sum()))
            require(math.isfinite(values['hard_jfde_sum_m']), 'nonfinite hard training JFDE')
            gradient = 0.
            if eligible:
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                require(gradients, 'missing trainable matcher gradients')
                gradient = float(torch.linalg.vector_norm(torch.stack([
                    torch.linalg.vector_norm(value.detach()) for value in gradients])))
                require(math.isfinite(gradient), 'nonfinite matcher gradient')
                optimizer.step()
                state['update'] += 1
            values.update(updates=int(eligible > 0), skipped=int(eligible == 0),
                          gradient_norm_sum=gradient, train_seconds=time.monotonic()-tick)
            sums = state['sums']
            for key, value in values.items():
                sums[key] += value
            sums['gradient_norm_max'] = max(sums['gradient_norm_max'], gradient)
            state['cursor'] += len(batch)
            state['peak_gpu_allocated_mib'] = max(state['peak_gpu_allocated_mib'], _max_allocated(device))
            info = dict(fold=fold, epoch=epoch, update=state['update'], windows=state['cursor'], total=len(examples),
                        hard_jfde_m=values['hard_jfde_sum_m']/len(batch), gradient_norm=gradient,
                        relative_windows=values['relative_windows'], global_relabel_windows=values['global_relabel_windows'])
            first = state['first_update'] is None and eligible > 0
            if first:
                state['first_update'] = info
            persist(checkpoints / 'last.pt')
            if first:
                immutable(output / 'FIRST_UPDATE.json', info)
            if first or (eligible and state['update'] % 50 == 0):
                _progress(output, 'TRAIN_UPDATE', **info)
        sums = state['sums']
        require(sums['windows'] == len(examples) and sums['multi_group_windows'] == multi_groups,
                'attempted to checkpoint an incomplete epoch')
        row = dict(epoch=epoch, file=f'epoch_{epoch:03d}.pt', **sums,
                   hard_jfde_m=sums['hard_jfde_sum_m']/sums['windows'],
                   relative_fraction_all=sums['relative_windows']/sums['windows'],
                   relative_fraction_eligible=sums['relative_windows']/multi_groups,
                   mean_gradient_norm=sums['gradient_norm_sum']/max(sums['updates'], 1))
        history = state['history']
        history.append(row)
        state.update(completed_epoch=epoch, cursor=0, order=None, sums={})
        target = checkpoints / row['file']
        persist(target)
        row['sha256'] = digest(target)
        _atomic_alias(checkpoints / 'last.pt', target)
        _checkpoint_index(output, history)
        _progress(output, 'TRAIN_EPOCH_COMPLETE', fold=fold, **row)

    history = state['history']
    require([row['epoch'] for row in history] == list(range(1, RECIPE['epochs']+1)),
            '12 complete snapshots required')
    for row in history:
        require(digest(checkpoints / row['file']) == row['sha256'], 'retained epoch checksum changed')
    final = checkpoints / history[-1]['file']
    for alias in ('final.pt', 'selected.pt'):
        _atomic_alias(checkpoints / alias, final)
    require(_query_sha256(queries) == identity['frozen_queries_sha256'] and not queries.requires_grad,
            'frozen slot queries changed')
    result = dict(status='COMPLETE', identity=identity, fold=fold, completed_epochs=RECIPE['epochs'],
                  final_checkpoint=str(final.resolve()), final_sha256=digest(final),
                  parameter_updates=state['update'], S0_updates=0, A_updates=0, B_updates=0,
                  train_windows=len(examples), train_multi_group_windows=multi_groups,
                  train_seconds=math.fsum(row['train_seconds'] for row in history),
                  feature_seconds=state['feature_seconds'],
                  peak_gpu_allocated_mib=state['peak_gpu_allocated_mib'],
                  epoch_rows=history, preflight=preflight, first_update=state['first_update'],
                  retained_epochs=[row['epoch'] for row in history], dev_or_test_reads=0, ddim_calls=0)
    immutable(result_path, result)
    return result

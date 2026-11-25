"""Continue the frozen HOTEL K20 training run from its complete epoch eight."""
from copy import deepcopy
import math
from pathlib import Path
import time

import torch

from bcep.contracts import canonical_hash, digest
from bcep_joint.common import read, restore_rng, rng_state, save, write
from joint_flow_fourfold.common import immutable
from joint_flow_fourfold.inputs import state_hash
from k20_dual_finetune.train import (
    CHECKPOINT_INTERVAL,
    RECIPE,
    TRAINABLE_PROVIDER,
    _atomic_alias,
    _load_latest,
    _optimizer,
    _progress,
    _provider_frozen_hash,
    _record_token,
    _stop_requested,
    _student_window,
)


PARENT_EPOCH = 8
TARGET_EPOCH = 16


def _teacher_hashes(model, provider, dependence):
    return dict(backbone=state_hash(model), provider=state_hash(provider),
                dependence=state_hash(dependence))


def _frozen_parents(inputs, model, teacher_provider, teacher_dependence, source, parent_sha256):
    assert inputs.fold == 'hotel'
    hashes = _teacher_hashes(model, teacher_provider, teacher_dependence)
    indices = inputs.split['train_indices']
    identity = dict(fold=inputs.fold, upstream=inputs.identity, teacher=hashes,
                    recipe=RECIPE, train_members_sha256=canonical_hash(
                        [inputs.split['members'][index] for index in indices]))
    assert identity == read(source / 'TRAIN_IDENTITY.json')
    original = read(source / 'TRAINING_RESULT.json')
    assert original['identity'] == identity and original['recipe'] == RECIPE
    assert original['status'] == 'COMPLETE' and original['completed_epochs'] == PARENT_EPOCH
    parent_path = source / 'checkpoints' / 'epoch_008.pt'
    assert digest(parent_path) == original['final_checkpoint_sha256'] == parent_sha256
    parent = torch.load(parent_path, map_location='cpu', weights_only=False)
    cache = source / 'train_cache'
    manifest_path = cache / 'MANIFEST.json'
    manifest_sha256 = digest(manifest_path)
    assert parent['identity'] == dict(identity, teacher_reference_manifest_sha256=manifest_sha256)
    assert parent['recipe'] == RECIPE and parent['completed_epoch'] == PARENT_EPOCH
    updates_per_epoch = math.ceil(len(indices) / RECIPE['logical_windows'])
    assert parent['update'] == original['updates'] == PARENT_EPOCH * updates_per_epoch
    assert parent['order'] is None and parent['cursor'] == 0 and parent['scheduler'] is None
    assert [row['epoch'] for row in parent['history']] == list(range(1, PARENT_EPOCH + 1))
    assert {'python', 'numpy', 'cpu', 'cuda', 'loader'} <= parent['rng'].keys()
    manifest = read(manifest_path)
    assert manifest['cache_key'] == canonical_hash(dict(
        identity=identity, purpose='train', sampler='joint-flow-dependence-v1',
        noise_seed=RECIPE['train_noise_seed'], M=RECIPE['M'], K=RECIPE['K']))
    rows = manifest['rows']
    assert len(indices) == len(rows)
    m0, j0 = manifest['M0_m'], manifest['J0_m']
    assert math.isfinite(m0) and math.isfinite(j0) and m0 > 0 and j0 > 0
    references = read(source / 'TEACHER_REFERENCES.json')
    assert references['teacher_cache_sha256'] == manifest_sha256
    assert (m0, j0) == (references['M0_m'], references['J0_m']) == (original['M0_m'], original['J0_m'])
    continuation = dict(parent['identity'], parent_checkpoint_sha256=parent_sha256,
                        first_new_epoch=PARENT_EPOCH + 1, target_completed_epoch=TARGET_EPOCH)
    return parent, cache, manifest, continuation, updates_per_epoch


def _verify_cache(inputs, cache, manifest, identity, output, should_yield):
    """Verify every read-only teacher shard and its fixed noise identity."""
    start = time.monotonic()
    rows = manifest['rows']
    for position, (index, row) in enumerate(zip(inputs.split['train_indices'], rows)):
        path = cache / f'{position:04d}.pt'
        assert row['index'] == index and row['file'] == path.name
        assert row['token'] == _record_token(inputs, index, manifest['cache_key'])
        assert row['agents'] == len(inputs.split['members'][index]['agent_ids'])
        assert read(path.with_suffix('.json')) == row
        assert digest(path) == row['sha256']
        completed = position + 1
        if completed % 200 == 0 or completed == len(rows):
            _progress(output, 'VERIFY_PARENT_TEACHER_CACHE', fold='hotel', windows=completed,
                      total=len(rows), wall_seconds=time.monotonic() - start)
            if _stop_requested(should_yield):
                return False
    write(output / 'CACHE_VERIFICATION.json', dict(
        identity=identity, files_verified=len(rows), cache_reused=True, read_only=True,
        noise_keys_unchanged=True, wall_seconds=time.monotonic() - start))
    return True


def _index(folder, history):
    entries = [dict(epoch=row['epoch'], path=row['file'], sha256=row['sha256'],
                    validation_metric=None) for row in history]
    selected = TARGET_EPOCH if history and history[-1]['epoch'] == TARGET_EPOCH else None
    write(folder / 'TOP5.json', dict(
        status='not_ranked_fixed_epoch_16', requested_rank_count=5, selected_epoch=selected,
        retained_completed_epochs=entries, ranked_epochs=[]))
    write(folder / 'EPOCH_METRICS.json', history)


def _assert_optimizer(optimizer, snapshot):
    state = optimizer.state_dict()
    assert state['param_groups'] == snapshot['optimizer']['param_groups']
    assert state['state'].keys() == snapshot['optimizer']['state'].keys()
    for key, saved in snapshot['optimizer']['state'].items():
        restored = state['state'][key]
        assert restored.keys() == saved.keys()
        for name, value in saved.items():
            if torch.is_tensor(value):
                assert torch.equal(restored[name].detach().cpu(), value)
            else:
                assert restored[name] == value
        assert int(restored['step']) == snapshot['update']


def continue_fold(inputs, model, teacher_provider, teacher_dependence, source_dir, output,
                  parent_sha256, *, device='cuda', should_yield=None):
    """Continue the original AdamW trajectory for epochs 9 through 16."""
    source, output = Path(source_dir).resolve(), Path(output).resolve()
    assert source != output and source not in output.parents
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    for network in (model, teacher_provider, teacher_dependence):
        network.eval().requires_grad_(False)
    parent, cache, manifest, identity, updates_per_epoch = _frozen_parents(
        inputs, model, teacher_provider, teacher_dependence, source, parent_sha256)
    immutable(output / 'TRAIN_IDENTITY.json', identity)
    result_path = output / 'TRAINING_RESULT.json'
    new_updates = (TARGET_EPOCH - PARENT_EPOCH) * updates_per_epoch
    if result_path.exists():
        result = read(result_path)
        assert result['identity'] == identity and result['completed_epochs'] == TARGET_EPOCH
        assert result['updates'] == parent['update'] + new_updates and result['new_updates'] == new_updates
        assert digest(result['final_checkpoint']) == result['final_checkpoint_sha256']
        return result
    if _stop_requested(should_yield) or not _verify_cache(
            inputs, cache, manifest, identity, output, should_yield):
        return None
    rows = manifest['rows']
    m0, j0 = manifest['M0_m'], manifest['J0_m']
    student_provider = deepcopy(teacher_provider)
    student_dependence = deepcopy(teacher_dependence).requires_grad_(True)
    student_provider.head.readout[-1].requires_grad_(True)
    assert tuple(name for name, param in student_provider.named_parameters()
                 if param.requires_grad) == TRAINABLE_PROVIDER
    assert sum(p.numel() for p in student_dependence.parameters()) == 16964
    assert sum(p.numel() for p in student_provider.head.readout[-1].parameters()) == 130
    frozen_student = _provider_frozen_hash(student_provider)
    optimizer = _optimizer(student_dependence, student_provider)
    params = [p for group in optimizer.param_groups for p in group['params']]
    checkpoint_dir = output / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    state = _load_latest(checkpoint_dir, identity)
    if state is None:
        state = dict(parent, identity=identity, history=[], status='resume_epoch_008')
    assert PARENT_EPOCH <= state['completed_epoch'] <= TARGET_EPOCH
    assert [row['epoch'] for row in state['history']] == list(
        range(PARENT_EPOCH + 1, state['completed_epoch'] + 1))
    assert state['update'] == state['completed_epoch'] * updates_per_epoch + math.ceil(
        state['cursor'] / RECIPE['logical_windows'])
    if state['order'] is not None:
        assert sorted(state['order']) == list(range(len(rows)))
    else:
        assert state['cursor'] == 0
    student_dependence.load_state_dict(state['dependence'], strict=True)
    student_provider.head.readout[-1].load_state_dict(state['readout'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    _assert_optimizer(optimizer, state)
    generator = torch.Generator()
    restore_rng(state['rng'], generator)
    assert torch.equal(generator.get_state(), state['rng']['loader'])
    if 'first_update' in state:
        immutable(output / 'FIRST_UPDATE.json', state['first_update'])

    def persist(path):
        state.update(dependence=student_dependence.state_dict(),
                     readout=student_provider.head.readout[-1].state_dict(),
                     optimizer=optimizer.state_dict(), rng=rng_state(generator))
        save(state, path)

    while state['completed_epoch'] < TARGET_EPOCH:
        epoch = state['completed_epoch'] + 1
        if state['order'] is None:
            state['order'] = torch.randperm(len(rows), generator=generator).tolist()
            state['sums'] = dict(agent_minfde_sum=0., jfde_sum=0., degradation_sum=0.,
                                 delta_nll_sum=0., objective_sum=0., windows=0, agents=0, batches=0)
        while state['cursor'] < len(rows):
            start = time.monotonic()
            cursor = state['cursor']
            batch_rows = [rows[i] for i in state['order'][cursor:cursor + RECIPE['logical_windows']]]
            batch_n = sum(row['agents'] for row in batch_rows)
            batch_w = len(batch_rows)
            weights = (1/(batch_n*m0), 1/(batch_w*j0), 1/(batch_w*j0),
                       RECIPE['delta_nll_weight']/batch_w)
            optimizer.zero_grad(set_to_none=True)
            batch_sums = torch.zeros(5, device=device, dtype=torch.float64)
            for row in batch_rows:
                terms = _student_window(cache, row, student_provider, student_dependence, device)
                loss = sum(term * weight for term, weight in zip(terms, weights))
                loss.backward()
                batch_sums += torch.stack((*[term.detach() for term in terms], loss.detach())).double()
            grad = torch.nn.utils.clip_grad_norm_(params, RECIPE['clip_grad_norm'], error_if_nonfinite=True)
            optimizer.step()
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
            state['epoch_train_seconds'] += time.monotonic() - start
            first = state['update'] == parent['update'] + 1
            if first:
                assert epoch == PARENT_EPOCH + 1
                dependence_changed = any(not torch.equal(t.detach().cpu(), parent['dependence'][key])
                                         for key, t in student_dependence.state_dict().items())
                readout_changed = any(not torch.equal(t.detach().cpu(), parent['readout'][key])
                                      for key, t in student_provider.head.readout[-1].state_dict().items())
                assert dependence_changed and readout_changed
                state['first_update'] = dict(
                    parent_checkpoint_sha256=parent_sha256, epoch=epoch, previous_update=parent['update'],
                    update=state['update'], first_epoch_order_sha256=canonical_hash(state['order']),
                    restored_adamw_state=True, restored_rng_and_loader=True,
                    dependence_changed=dependence_changed, readout_changed=readout_changed)
            if first or state['update'] % CHECKPOINT_INTERVAL == 0:
                state['status'] = 'update_complete'
                persist(checkpoint_dir / 'last.pt')
                if first:
                    immutable(output / 'FIRST_UPDATE.json', state['first_update'])
                _progress(output, 'TRAIN_UPDATE', fold='hotel', epoch=epoch, epochs=TARGET_EPOCH,
                          update=state['update'], windows=state['cursor'], total=len(rows),
                          batch_objective=float(batch_sums[-1]), grad_norm=float(grad),
                          epoch_train_seconds=state['epoch_train_seconds'])
                if _stop_requested(should_yield):
                    return None

        totals = state['sums']
        assert totals['windows'] == len(rows) and totals['agents'] == sum(row['agents'] for row in rows)
        filename = f'epoch_{epoch:03d}.pt'
        record = dict(epoch=epoch, update=state['update'], file=filename,
                      train_seconds=state['epoch_train_seconds'],
                      minFDE_m=totals['agent_minfde_sum']/totals['agents'],
                      JFDE_m=totals['jfde_sum']/totals['windows'],
                      JFDE_degradation_m=totals['degradation_sum']/totals['windows'],
                      delta_nll_nat_per_person=totals['delta_nll_sum']/totals['windows'],
                      objective_mean_per_update=totals['objective_sum']/totals['batches'],
                      windows=totals['windows'], agents=totals['agents'])
        assert _provider_frozen_hash(student_provider) == frozen_student
        state['history'].append(record)
        state.update(completed_epoch=epoch, cursor=0, order=None, sums={}, epoch_train_seconds=0.,
                     status='complete' if epoch == TARGET_EPOCH else 'epoch_complete')
        target = checkpoint_dir / filename
        persist(target)
        record['sha256'] = digest(target)
        _atomic_alias(checkpoint_dir / 'last.pt', target)
        _index(checkpoint_dir, state['history'])
        _progress(output, 'TRAIN_EPOCH_COMPLETE', fold='hotel', **record)
        if epoch < TARGET_EPOCH and _stop_requested(should_yield):
            return None

    assert state['update'] == parent['update'] + new_updates
    assert len(state['history']) == TARGET_EPOCH - PARENT_EPOCH
    for row in state['history']:
        path = checkpoint_dir / row['file']
        assert digest(path) == row['sha256']
        saved = torch.load(path, map_location='cpu', weights_only=False)
        assert saved['identity'] == identity and saved['completed_epoch'] == row['epoch']
        assert saved['cursor'] == 0 and saved['order'] is None
        assert {'dependence', 'readout', 'optimizer', 'scheduler', 'rng', 'update'} <= saved.keys()
    target = checkpoint_dir / state['history'][-1]['file']
    for alias in ('final.pt', 'selected.pt', 'last.pt'):
        _atomic_alias(checkpoint_dir / alias, target)
    _index(checkpoint_dir, state['history'])
    assert _provider_frozen_hash(student_provider) == frozen_student
    assert _teacher_hashes(model, teacher_provider, teacher_dependence) == identity['teacher']
    result = dict(status='COMPLETE', fold='hotel', completed_epochs=TARGET_EPOCH,
                  first_new_epoch=PARENT_EPOCH + 1, parent_checkpoint_sha256=parent_sha256,
                  final_checkpoint=str(target), final_checkpoint_sha256=digest(target),
                  updates=state['update'], new_updates=new_updates, identity=identity, recipe=RECIPE,
                  checkpoint_selection='fixed epoch 16', M0_m=m0, J0_m=j0,
                  train_windows=len(rows), train_agents=sum(row['agents'] for row in rows),
                  DDIM_calls=0, test_access=False, frozen_parameters_and_statistics_unchanged=True,
                  history=state['history'], optimizer_groups=[{key: group[key] for key in
                      ('names', 'lr', 'weight_decay')} for group in optimizer.param_groups])
    write(result_path, result)
    return result

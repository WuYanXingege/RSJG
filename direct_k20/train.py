"""Fixed two-stage direct K20 training and original-development-split selection."""

import math
import os
from pathlib import Path
import time

import torch

from bcep.contracts import canonical_hash, digest, require
from bcep_formal.control import pause_requested
from bcep_joint.common import read, restore_rng, rng_state, save, seed_all, write
from ebjd.metrics import MetricAccumulator
from joint_flow_fourfold.common import immutable
from joint_flow_fourfold.inputs import state_hash
from joint_flow_fourfold.scoring import csvwrite
from .common import window_key
from .model import (
    Student,
    decode_student,
    encode_observed,
    endpoint_terms,
    preflight,
    student_endpoints,
    student_noise,
    teacher_terms,
    trajectory_terms,
)


METRICS = ('JADE', 'JFDE', 'minADE', 'minFDE')
EVAL_EPOCHS = (12, 16, 20, 24)
RECIPE = dict(
    seed=3201,
    train_bridge_seed=3201,
    development_seed=2035,
    epochs=24,
    logical_windows=16,
    end_only_epochs=12,
    trajectory_sample_fraction=1/8,
    trajectory_sample_count='max(1,(logical_batch_window_count+4)//8)',
    lr=3e-4,
    weight_decay=1e-4,
    clip_grad_norm=5.,
    teacher_weight=.25,
    trajectory_weight=.5,
    normalization_floor=1e-6,
    evaluation_epochs=list(EVAL_EPOCHS),
    K=20,
    DDIM_steps=20,
    proposal_eta=0.,
    checkpoint_steps=True,
    precision='FP32',
    ranking='minimize (max four metric ratios, mean four metric ratios, completed epoch)',
    retention='keep all completed epochs; rank the four evaluated epochs by the fixed selector',
)
CHECKPOINT_INTERVAL = 10


def _progress(output, phase, **kwargs):
    row = dict(phase=phase, unix=time.time(), **kwargs)
    write(output / 'PROGRESS.json', row)
    print(row, flush=True)


def _should_pause(should_yield):
    return pause_requested() or (should_yield is not None and should_yield())


def _atomic_alias(path, target):
    temporary = path.with_suffix('.link')
    temporary.unlink(missing_ok=True)
    os.link(target, temporary)
    temporary.replace(path)


def _check_frozen(assets, expected):
    require(dict(provider=state_hash(assets.provider), bridge=state_hash(assets.model)) == expected,
            'a frozen inference module changed during training')
    for module in (assets.provider, assets.model):
        require(all(not p.requires_grad and p.grad is None for p in module.parameters()),
                'a frozen inference parameter participated in the optimizer')
        require(not any(part.training for part in module.modules()),
                'frozen inference module left eval mode')


def _preflight(assets, student, cache, output, identity_hash):
    receipt_path = output / 'PREFLIGHT.json'
    if receipt_path.exists():
        receipt = read(receipt_path)
        require(receipt['identity_sha256'] == identity_hash and receipt['passed'],
                'preflight belongs to a different experiment')
        return receipt
    generator = torch.Generator().manual_seed(RECIPE['seed'])
    initial_order = torch.randperm(len(cache.indices), generator=generator).tolist()
    # Run the full-bridge check on the smallest interacting window of the
    # first logical batch. This also checks the edge MLP's derivative.
    first_batch = [cache.indices[i] for i in initial_order[:RECIPE['logical_windows']]]
    num_agents = lambda index: len(assets.split['members'][index]['agent_ids'])
    multi = [index for index in first_batch if num_agents(index) > 1]
    require(bool(multi), 'the first logical batch must include an interacting window')
    index = min(multi, key=num_agents)
    batch = assets.batch(index, 'train', truth=True)
    refs = cache.load(index, batch.observed.device)
    result = preflight(student, assets.model, assets.provider, batch, refs['A'], refs['B'],
                       window_key(assets.split, index), seed=RECIPE['train_bridge_seed'])
    require(result['passed'], 'real-window operator or gradient check failed')
    require(all(p.grad is None for p in student.parameters()), 'preflight left optimizer gradients')
    receipt = dict(result, identity_sha256=identity_hash, first_logical_batch=True,
                   window_sha256=window_key(assets.split, index), optimizer_updates=0)
    immutable(receipt_path, receipt)
    _progress(output, 'PREFLIGHT_PASSED', fold=assets.fold, receipt=receipt)
    return receipt


def _window_goals(assets, student, index, batch, *, input_grad):
    context, features = encode_observed(assets.provider, batch)
    goals = student_endpoints(student, assets.provider, batch.observed[0], batch.semantic_maps[0],
                              batch.valid[0], context, features, input_grad=input_grad)
    return goals, window_key(assets.split, index)


def _trajectory(assets, batch, goals, key, *, seed, input_grad):
    with torch.no_grad():
        context = assets.model.encode_context(batch.observed, batch.semantic_maps, batch.valid)
    noise = student_noise(key, batch.agent_ids[0], seed, device=goals.device)
    return decode_student(assets.model, context, goals, noise, input_grad=input_grad,
                          checkpoint_steps=bool(input_grad))


def _epoch_sample_count(batch_windows):
    return max(1, (batch_windows + 4) // 8)


def _train_microbatch(assets, student, cache, index, selected, weights):
    batch = assets.batch(index, 'train', truth=True)
    n = len(assets.split['members'][index]['agent_ids'])
    require(batch.valid.shape == (1, n) and bool(batch.valid.all()),
            'training member mask differs from the cache')
    teacher = cache.load(index, batch.observed.device)
    goals, key = _window_goals(assets, student, index, batch, input_grad=True)
    mf, jf = endpoint_terms(goals, batch.future[0, :, -1])
    ta, tb = teacher_terms(goals, teacher['A'], teacher['B'])
    end = weights['MF']*mf + weights['JF']*jf + weights['TA']*ta
    if n > 1:
        require(tb is not None, 'multi-agent window has no scene teacher term')
        end = end + weights['TB']*tb
    else:
        require(tb is None, 'singleton must not have a B teacher term')
        tb = goals.new_zeros(())
    total = end
    ma = ja = goals.new_zeros(())
    if selected:
        path = _trajectory(assets, batch, goals, key, seed=RECIPE['train_bridge_seed'],
                           input_grad=True)
        ma, ja = trajectory_terms(path, batch.future, batch.valid)
        total = total + weights['MA']*ma + weights['JA']*ja
    total.backward()
    return torch.stack([u.detach() for u in (mf, jf, ta, tb, ma, ja, end, total)]).double()


@torch.no_grad()
def _development(assets, student, output, epoch, should_yield):
    """Resume native validation from the last complete window, before selection."""
    result_path = output / f'DEV_EPOCH_{epoch:03d}.json'
    partial_path = output / f'DEV_EPOCH_{epoch:03d}_PARTIAL.json'
    indices = assets.split['selection_dev_indices']
    parameters_sha = state_hash(student)
    selector_identity = dict(fold=assets.fold, epoch=epoch, seed=RECIPE['development_seed'],
                             student_state_sha256=parameters_sha, upstream=assets.identity,
                             reference=assets.dev_B)
    if result_path.exists():
        result = read(result_path)
        require(result['identity'] == selector_identity and result['windows'] == len(indices),
                'completed dev score does not match the epoch weights or membership')
        return result
    acc = MetricAccumulator()
    cursor = 0
    elapsed = 0.
    if partial_path.exists():
        partial = read(partial_path)
        require(partial['identity'] == selector_identity, 'partial dev score identity differs')
        cursor = partial['cursor']
        elapsed = partial['seconds']
        require(cursor == partial['sums']['scene_count'] <= len(indices), 'invalid dev cursor')
        for name, value in partial['sums'].items():
            setattr(acc, name, value)
    started = time.monotonic()
    student.eval()
    for position in range(cursor, len(indices)):
        index = indices[position]
        batch = assets.batch(index, 'selection_dev', truth=True)
        goals, key = _window_goals(assets, student, index, batch, input_grad=False)
        trajectory = _trajectory(assets, batch, goals, key,
                                 seed=RECIPE['development_seed'], input_grad=False)
        acc.update(trajectory, batch.future, batch.valid)
        cursor = position + 1
        if cursor % 50 == 0 or cursor == len(indices) or _should_pause(should_yield):
            seconds = elapsed + time.monotonic()-started
            write(partial_path, dict(identity=selector_identity, cursor=cursor,
                                     sums=vars(acc), seconds=seconds))
            _progress(output, 'VALIDATING_TRUE_TRAJECTORY', fold=assets.fold, epoch=epoch,
                      windows=cursor, total=len(indices), wall_seconds=seconds)
            if cursor < len(indices) and _should_pause(should_yield):
                return None
    require(acc.scene_count == len(indices), 'native dev accumulator incomplete')
    metrics = acc.as_dict()
    ratios = {metric: metrics[metric]/float(assets.dev_B[metric]) for metric in METRICS}
    require(all(math.isfinite(value) for value in ratios.values()), 'nonfinite dev ratio')
    result = dict(identity=selector_identity, epoch=epoch, seed=RECIPE['development_seed'],
                  metrics=metrics, ratios=ratios,
                  max_ratio=max(ratios.values()), mean_ratio=math.fsum(ratios.values())/len(ratios),
                  windows=len(indices), ddim_calls=len(indices), seconds=elapsed+time.monotonic()-started)
    immutable(result_path, result)
    return result


def _rank(entry):
    return entry['dev']['max_ratio'], entry['dev']['mean_ratio'], entry['epoch']


def _record_history(output, checkpoints, history):
    write(output / 'EPOCH_METRICS.json', history)
    ranked = sorted((row for row in history if row['dev'] is not None), key=_rank)[:5]
    write(checkpoints / 'TOP5.json', dict(
        ranking='only prescribed dev evaluations, by maximum ratio, mean ratio, earlier epoch',
        ranked_epoch_count=len(ranked), all_completed_epochs_retained=len(history),
        entries=[dict(epoch=row['epoch'], file=row['file'], sha256=row['sha256'],
                      max_ratio=row['dev']['max_ratio'], mean_ratio=row['dev']['mean_ratio'],
                      ratios=row['dev']['ratios'], metrics=row['dev']['metrics']) for row in ranked]))
    if not ranked:
        return
    table = []
    for number, row in enumerate(ranked, 1):
        d = row['dev']
        table.append(dict(epoch=row['epoch'], rank=number, checkpoint_sha256=row['sha256'],
                          max_ratio=d['max_ratio'], mean_ratio=d['mean_ratio'],
                          **{f'{metric}_m': d['metrics'][metric] for metric in METRICS},
                          **{f'{metric}_div_B': d['ratios'][metric] for metric in METRICS}))
    csvwrite(output / 'DEV_SELECTION.csv', table)
    _atomic_alias(checkpoints / 'best.pt', checkpoints / ranked[0]['file'])


def _latest(checkpoints, identity):
    path = checkpoints / 'last.pt'
    epochs = sorted(checkpoints.glob('epoch_*.pt'))
    if not path.exists() and not epochs:
        return None
    state = torch.load(path if path.exists() else epochs[-1], weights_only=False, map_location='cpu')
    if epochs and int(epochs[-1].stem.split('_')[1]) > state['completed_epoch']:
        state = torch.load(epochs[-1], weights_only=False, map_location='cpu')
    require(state['identity'] == identity and state['recipe'] == RECIPE,
            'existing checkpoint belongs to another run')
    for row in state['history']:
        observed = digest(checkpoints/row['file'])
        require(row.get('sha256', observed) == observed, 'completed checkpoint changed')
        row['sha256'] = observed
    return state


def train_fold(assets, student, cache, output, *, device='cuda', should_yield=None):
    """Train exactly one fold, preserving each complete epoch and its resume state."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    indices = tuple(cache.indices)
    require(indices == tuple(assets.split['train_indices']), 'teacher cache membership mismatch')
    require(len(indices) > 0 and set(METRICS) <= set(cache.scales) and
            set(METRICS) <= set(assets.dev_B), 'missing baseline scales')
    scales = {metric: max(float(cache.scales[metric]), RECIPE['normalization_floor']) for metric in METRICS}
    references = {metric: float(assets.dev_B[metric]) for metric in METRICS}
    require(all(math.isfinite(value) and value > 0 for value in (*scales.values(), *references.values())),
            'invalid normalization scales or B development scores')
    for module in (assets.provider, assets.model):
        module.eval().requires_grad_(False)
    frozen = dict(provider=state_hash(assets.provider), bridge=state_hash(assets.model))
    identity = dict(fold=assets.fold, upstream=assets.identity, cache=cache.identity,
                    train_indices_sha256=canonical_hash(list(indices)), recipe=RECIPE,
                    frozen=frozen, scales_m=scales, development_B_m=references)
    immutable(output / 'TRAIN_IDENTITY.json', identity)
    result_path = output / 'TRAINING_RESULT.json'
    if result_path.exists():
        result = read(result_path)
        require(result['identity'] == identity and result['completed_epochs'] == RECIPE['epochs'] and
                digest(result['selected_checkpoint']) == result['selected_sha256'],
                'previous training result identity changed')
        return result
    seed_all(RECIPE['seed'])
    if student is None:
        student = Student()
    student = student.to(device=device, dtype=torch.float32)
    require(all(p.requires_grad and p.dtype == torch.float32 for p in student.parameters()),
            'only the FP32 query and MLP student should be trainable')
    loader = torch.Generator().manual_seed(RECIPE['seed'])
    sampler = torch.Generator().manual_seed(RECIPE['seed'] + 1)
    optimizer = torch.optim.AdamW(student.parameters(), lr=RECIPE['lr'], weight_decay=RECIPE['weight_decay'])
    state = _latest(checkpoints, identity)
    receipt = _preflight(assets, student, cache, output, canonical_hash(identity))

    if state is None:
        state = dict(identity=identity, recipe=RECIPE, completed_epoch=0, update=0,
                     order=None, cursor=0, sums={}, history=[], scheduler=None,
                     status='initialized', epoch_seconds=0.)
    else:
        student.load_state_dict(state['student'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng'], loader)
        sampler.set_state(state['trajectory_sampler_rng'])
        _record_history(output, checkpoints, state['history'])

    def persist(path):
        state.update(student=student.state_dict(), optimizer=optimizer.state_dict(),
                     rng=rng_state(loader), trajectory_sampler_rng=sampler.get_state())
        save(state, path)

    counts = {index: len(assets.split['members'][index]['agent_ids']) for index in indices}
    total_agents = sum(counts.values())
    total_multi = sum(n > 1 for n in counts.values())
    value_names = ('mf_sum', 'jf_sum', 'a_teacher_sum', 'b_teacher_sum',
                   'ma_drawn_sum', 'ja_drawn_sum', 'end_objective', 'objective')
    while state['completed_epoch'] < RECIPE['epochs']:
        epoch = state['completed_epoch'] + 1
        student.train()
        if state['order'] is None:
            state['order'] = torch.randperm(len(indices), generator=loader).tolist()
            state['sums'] = dict.fromkeys(value_names, 0.) | dict(
                windows=0, agents=0, multi=0, updates=0, ddim_calls=0,
                drawn_agents=0, weighted_ma=0., weighted_ja=0.)
        while state['cursor'] < len(indices):
            tick = time.monotonic()
            cursor = state['cursor']
            batch_indices = [indices[pos] for pos in state['order'][cursor:cursor+RECIPE['logical_windows']]]
            n_windows = len(batch_indices)
            n_agents = sum(counts[index] for index in batch_indices)
            n_multi = sum(counts[index] > 1 for index in batch_indices)
            sample_count = _epoch_sample_count(n_windows) if epoch > RECIPE['end_only_epochs'] else 0
            selected = set(torch.randperm(n_windows, generator=sampler)[:sample_count].tolist()) if sample_count else set()
            probability = sample_count / n_windows
            w = dict(
                MF=1/(n_agents*scales['minFDE']), JF=1/(n_windows*scales['JFDE']),
                TA=RECIPE['teacher_weight']/(20*n_agents*scales['minFDE']),
                TB=RECIPE['teacher_weight']/(20*n_multi*scales['JFDE']) if n_multi else 0.,
                MA=RECIPE['trajectory_weight']/(probability*n_agents*scales['minADE']) if sample_count else 0.,
                JA=RECIPE['trajectory_weight']/(probability*n_windows*scales['JADE']) if sample_count else 0.,
            )
            optimizer.zero_grad(set_to_none=True)
            batch_values = torch.zeros(len(value_names), device=device, dtype=torch.float64)
            for pos, index in enumerate(batch_indices):
                batch_values += _train_microbatch(assets, student, cache, index, pos in selected, w)
            gradient = torch.nn.utils.clip_grad_norm_(student.parameters(), RECIPE['clip_grad_norm'],
                                                      error_if_nonfinite=True)
            optimizer.step()
            values = batch_values.tolist()
            require(all(math.isfinite(v) for v in values), 'nonfinite microbatch objective')
            state['cursor'] += n_windows
            state['update'] += 1
            state['epoch_seconds'] += time.monotonic() - tick
            sums = state['sums']
            for name, value in zip(value_names, values):
                sums[name] += value
            additions = dict(windows=n_windows, agents=n_agents, multi=n_multi, updates=1,
                             ddim_calls=sample_count, drawn_agents=sum(counts[batch_indices[p]] for p in selected))
            if sample_count:
                additions.update(weighted_ma=values[4]/probability, weighted_ja=values[5]/probability)
            for name, value in additions.items():
                sums[name] += value
            pause = _should_pause(should_yield)
            if state['update'] == 1 or state['update'] % CHECKPOINT_INTERVAL == 0 or pause:
                state['status'] = 'update_complete'
                persist(checkpoints / 'last.pt')
                _progress(output, 'TRAIN_UPDATE', fold=assets.fold, epoch=epoch, update=state['update'],
                          windows=state['cursor'], total=len(indices), phase_ddim_calls=sums['ddim_calls'],
                          last_batch_draws=sample_count, last_batch_p=probability, grad_norm=float(gradient),
                          epoch_train_seconds=state['epoch_seconds'])
            if pause:
                return None
        totals = state['sums']
        require((totals['windows'], totals['agents'], totals['multi']) == (len(indices), total_agents, total_multi),
                'incomplete logical epoch')
        if epoch in EVAL_EPOCHS:
            state['status'] = 'awaiting_dev_score'
            persist(checkpoints / 'last.pt')
            dev = _development(assets, student, output, epoch, should_yield)
            if dev is None:
                return None
        else:
            dev = None
        _check_frozen(assets, frozen)
        row = dict(epoch=epoch, update=state['update'], file=f'epoch_{epoch:03d}.pt',
                   windows=totals['windows'], agents=totals['agents'], multi_windows=totals['multi'],
                   minFDE_m=totals['mf_sum']/total_agents, JFDE_m=totals['jf_sum']/len(indices),
                   TA_m=totals['a_teacher_sum']/(20*total_agents),
                   TB_m=totals['b_teacher_sum']/(20*total_multi) if total_multi else 0.,
                   sampled_minADE_HT_m=totals['weighted_ma']/total_agents if totals['ddim_calls'] else None,
                   sampled_JADE_HT_m=totals['weighted_ja']/len(indices) if totals['ddim_calls'] else None,
                   mean_end_objective_per_update=totals['end_objective']/totals['updates'],
                   mean_total_objective_per_update=totals['objective']/totals['updates'],
                   train_ddim_calls=totals['ddim_calls'], sampled_agents=totals['drawn_agents'],
                   train_seconds=state['epoch_seconds'], dev=dev)
        state['history'].append(row)
        state.update(completed_epoch=epoch, status='complete' if epoch == RECIPE['epochs'] else 'epoch_complete',
                     order=None, cursor=0, sums={}, epoch_seconds=0.)
        target = checkpoints / row['file']
        persist(target)
        row['sha256'] = digest(target)
        _atomic_alias(checkpoints / 'last.pt', target)
        _record_history(output, checkpoints, state['history'])
        _progress(output, 'TRAIN_EPOCH_COMPLETE', fold=assets.fold, epoch=epoch, update=state['update'],
                  train_seconds=row['train_seconds'], train_ddim_calls=row['train_ddim_calls'],
                  max_dev_ratio=dev['max_ratio'] if dev else None,
                  mean_dev_ratio=dev['mean_ratio'] if dev else None)
        if epoch < RECIPE['epochs'] and _should_pause(should_yield):
            return None

    history = state['history']
    require(len(history) == RECIPE['epochs'] and tuple(row['epoch'] for row in history if row['dev']) == EVAL_EPOCHS,
            'incomplete evaluated checkpoints')
    for row in history:
        path = checkpoints / row['file']
        snapshot = torch.load(path, weights_only=False, map_location='cpu')
        require(digest(path) == row['sha256'] and snapshot['completed_epoch'] == row['epoch']
                and snapshot['cursor'] == 0 and snapshot['identity'] == identity
                and all(key in snapshot for key in ('student','optimizer','scheduler','rng','update',
                                                   'trajectory_sampler_rng','order','history')),
                'completed checkpoint lacks exact recovery state')
    selected = min((row for row in history if row['dev']), key=_rank)
    path = checkpoints / selected['file']
    _atomic_alias(checkpoints / 'selected.pt', path)
    _atomic_alias(checkpoints / 'best.pt', path)
    _atomic_alias(checkpoints / 'final.pt', checkpoints / history[-1]['file'])
    _record_history(output, checkpoints, history)
    result = dict(status='COMPLETE', fold=assets.fold, identity=identity, recipe=RECIPE,
                  updates=state['update'], completed_epochs=RECIPE['epochs'],
                  selected_epoch=selected['epoch'], selected_checkpoint=str(path.resolve()),
                  selected_sha256=digest(path), final_checkpoint=str((checkpoints/'final.pt').resolve()),
                  final_sha256=history[-1]['sha256'],
                  dev_scores=[row['dev'] for row in history if row['dev']],
                  train_ddim_calls=sum(row['train_ddim_calls'] for row in history),
                  development_ddim_calls=sum(row['dev']['ddim_calls'] for row in history if row['dev']),
                  preflight=receipt, train_seconds=math.fsum(row['train_seconds'] for row in history),
                  development_seconds=math.fsum(row['dev']['seconds'] for row in history if row['dev']),
                  retained_epochs=[row['epoch'] for row in history],
                  test_access=False, epoch_metrics_sha256=digest(output/'EPOCH_METRICS.json'),
                  dev_selection_sha256=digest(output/'DEV_SELECTION.csv'))
    write(result_path, result)
    return result

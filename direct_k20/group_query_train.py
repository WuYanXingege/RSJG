"""Fixed twelve-epoch group-query warm start with native late-phase losses."""

import math
from pathlib import Path
import time

import torch

from bcep.contracts import canonical_hash, digest, require
from bcep_joint.common import read, restore_rng, rng_state, save, seed_all, write
from joint_flow_fourfold.common import immutable
from joint_flow_fourfold.inputs import state_hash
from .common import window_key
from .group_query import prepare_groups
from .interaction_train import _gpu_peaks, _sliced_trajectory
from .interaction_train import record_r12_metrics as record_final_metrics
from .model import (
    NATIVE_ATOL, NATIVE_RTOL, Student, encode_observed, endpoint_terms,
    student_endpoints, teacher_terms, trajectory_terms,
)
from .train import METRICS, RECIPE as NATIVE_RECIPE
from .train import _atomic_alias, _epoch_sample_count, _progress, _should_pause, _trajectory


RECIPE = NATIVE_RECIPE | dict(
    epochs=12,
    end_only_epochs=0,
    evaluation_epochs=[12],
    lr=3e-5,
    group_query_lr=3e-4,
    optimizer='AdamW',
    trajectory_sample_seed=3202,
    ranking='only complete epoch 12 is selected',
    retention='all 12 complete snapshots plus final/selected/best and last',
    recovery='atomically save last after every complete logical batch',
)
VALUE_NAMES = ('mf_sum', 'jf_sum', 'a_teacher_sum', 'b_teacher_sum',
               'ma_drawn_sum', 'ja_drawn_sum', 'end_objective', 'objective')


def _weights(n_agents, n_windows, n_multi, n_draws, scales):
    p = n_draws / n_windows
    return dict(
        MF=1/(n_agents*scales['minFDE']), JF=1/(n_windows*scales['JFDE']),
        TA=RECIPE['teacher_weight']/(20*n_agents*scales['minFDE']),
        TB=RECIPE['teacher_weight']/(20*n_multi*scales['JFDE']) if n_multi else 0.,
        MA=RECIPE['trajectory_weight']/(p*n_agents*scales['minADE']),
        JA=RECIPE['trajectory_weight']/(p*n_windows*scales['JADE']),
    )


def _goals(student, assets, batch, groups, *, input_grad):
    context, features = encode_observed(assets.provider, batch)
    return student_endpoints(
        student, assets.provider, batch.observed[0], batch.semantic_maps[0], batch.valid[0],
        context, features, groups=groups, input_grad=input_grad,
    )


def _microbatch(assets, student, index, n, teacher, groups, selected, weights, slot_chunk):
    batch = assets.batch(index, 'train', truth=True)
    require(batch.valid.shape == (1, n) and bool(batch.valid.all()), 'train mask differs from cache')
    if index not in groups:
        groups[index] = prepare_groups(batch.observed[0]).to(batch.observed.device)
    goals = _goals(student, assets, batch, groups[index], input_grad=True)
    mf, jf = endpoint_terms(goals, batch.future[0, :, -1])
    frozen_targets = teacher(index, goals.device)
    ta, tb = teacher_terms(goals, frozen_targets['A'], frozen_targets['B'])
    require((tb is None) == (n == 1), 'native singleton teacher rule changed')
    end = weights['MF']*mf + weights['JF']*jf + weights['TA']*ta
    if tb is None:
        tb = goals.new_zeros(())
    else:
        end = end + weights['TB']*tb
    total = end
    ma = ja = goals.new_zeros(())
    drift = None
    if selected:
        key = window_key(assets.split, index)
        if slot_chunk:
            path_loss, ma, ja, gradient, drift = _sliced_trajectory(
                assets, batch, goals, key, weights, slot_chunk,
            )
            torch.autograd.backward((end, goals), (None, gradient))
            total = end.detach() + path_loss
        else:
            path = _trajectory(assets, batch, goals, key, seed=RECIPE['train_bridge_seed'], input_grad=True)
            ma, ja = trajectory_terms(path, batch.future, batch.valid)
            total = total + weights['MA']*ma + weights['JA']*ja
    if total.requires_grad:
        total.backward()
    values = torch.stack([x.detach() for x in (mf, jf, ta, tb, ma, ja, end, total)]).double()
    return values, drift


def _readout_probe(student, observed, context, groups, gradients):
    """Trace the actual pre-personal query with a temporary nonzero readout."""
    readout = student.group_query[-1]
    prefix = f'group_query.{len(student.group_query)-1}.'
    probe = {
        prefix + name: p.detach() + 1e-3 * gradient.detach()/gradient.detach().abs().max().clamp_min(1e-8)
        for (name, p), gradient in zip(readout.named_parameters(), gradients)
    }
    trace = []
    hook = student.personal.register_forward_pre_hook(lambda module, args: trace.append(args[0][..., -32:]))
    try:
        with torch.no_grad():
            torch.func.functional_call(student, probe, (observed, context), {'groups': groups})
    finally:
        hook.remove()
    require(len(trace) == 1, 'native personal layer did not receive one query per agent and slot')
    local_query = trace[0]
    group_index = groups.group_index
    representative = groups.members[:, 0]
    expected = local_query[:, representative][:, group_index]
    torch.testing.assert_close(local_query, expected, rtol=0., atol=0.)
    require(not torch.equal(local_query, student.queries[:, None].expand_as(local_query)),
            'preflight sharing probe did not exercise nonzero offsets')


def _preflight(assets, student, s0_state, indices, counts, output, identity_sha):
    path = output / 'PREFLIGHT.json'
    if path.exists():
        receipt = read(path)
        require(receipt['identity_sha256'] == identity_sha and receipt['passed'],
                'preflight identity changed during recovery')
        return receipt
    generator = torch.Generator().manual_seed(RECIPE['seed'])
    order = torch.randperm(len(indices), generator=generator).tolist()
    first = [indices[position] for position in order[:RECIPE['logical_windows']]]
    candidates = sorted((i for i in first if counts[i] > 1), key=lambda i: (counts[i], i))
    require(bool(candidates), 'first logical batch lacks any interacting scene')
    for index in candidates:
        groups = prepare_groups(assets.batch(index, 'train', truth=False).observed[0])
        if len(groups.members) < counts[index]:
            break
    require(len(groups.members) < counts[index], 'first logical batch lacks a real paired group')
    batch = assets.batch(index, 'train', truth=True)
    observed, maps, valid = batch.observed[0], batch.semantic_maps[0], batch.valid[0]
    groups = groups.to(observed.device)
    require(valid.shape == (counts[index],) and bool(valid.all()), 'real preflight mask changed')
    before = assets.frozen_states()
    initial_student = state_hash(student)
    native = Student().to(device=observed.device, dtype=torch.float32).eval().requires_grad_(False)
    native.load_state_dict(s0_state, strict=True)
    with torch.no_grad():
        context, features = encode_observed(assets.provider, batch)
        g0 = student_endpoints(native, assets.provider, observed, maps, valid, context, features)
        zero_graph_goals = student_endpoints(student, assets.provider, observed, maps, valid,
                                             context, features, groups=groups)
        offsets = student.query_offsets(observed, context, groups)
    require(torch.equal(g0, zero_graph_goals) and not bool(offsets.count_nonzero()),
            'zero readout did not reproduce the native ordered endpoints')
    g1 = student_endpoints(student, assets.provider, observed, maps, valid, context, features,
                           groups=groups, input_grad=True)
    torch.testing.assert_close(g1.detach(), g0, rtol=NATIVE_RTOL, atol=NATIVE_ATOL)
    key = window_key(assets.split, index)
    path0 = _trajectory(assets, batch, g0, key, seed=RECIPE['train_bridge_seed'], input_grad=False)
    path1 = _trajectory(assets, batch, g1, key, seed=RECIPE['train_bridge_seed'], input_grad=True)
    torch.testing.assert_close(path1.detach(), path0, rtol=NATIVE_RTOL, atol=NATIVE_ATOL)
    interior = (path1[..., :-1, :] - batch.future[:, None, :, :-1, :]).norm(dim=-1).mean()
    gradients = torch.autograd.grad(interior, tuple(student.group_query[-1].parameters()))
    finite = all(bool(torch.isfinite(g).all()) for g in gradients)
    grad_max = {name: float(g.abs().max())
                for (name, _), g in zip(student.group_query[-1].named_parameters(), gradients)}
    require(finite and any(v > 0 for v in grad_max.values()), 'real DDIM gradient missed the new readout')
    _readout_probe(student, observed, context, groups, gradients)
    require(all(p.grad is None for p in student.parameters()) and state_hash(student) == initial_student
            and assets.frozen_states() == before, 'preflight changed native or trainable state')
    result = dict(
        passed=True, identity_sha256=identity_sha, window_sha256=key,
        agents=counts[index], groups=len(groups.members), optimizer_updates=0,
        exact_initial_endpoint_replay=True, nonzero_probe_shares_group_query=True,
        differentiable_endpoint_max_abs_diff_m=float((g0-g1.detach()).abs().max()),
        same_noise_trajectory_max_abs_diff_m=float((path0-path1.detach()).abs().max()),
        trajectory_interior_readout_gradient_max=grad_max, frozen_state_sha256=before,
        actual_ddim_calls=2,
    )
    immutable(path, result)
    _progress(output, 'PREFLIGHT_PASSED', fold=assets.fold, **result)
    return result


def _inventory(output, history):
    write(output / 'EPOCH_METRICS.json', history)
    write(output / 'checkpoints' / 'TOP5.json', dict(
        selector=RECIPE['ranking'], ranked_epoch_count=0, entries=[],
        snapshots=[{key: row[key] for key in ('epoch', 'file', 'sha256')} for row in history],
    ))


def _resume(checkpoints, identity):
    last = checkpoints / 'last.pt'
    epochs = sorted(checkpoints.glob('epoch_*.pt'))
    if not last.exists() and not epochs:
        return None
    state = torch.load(last if last.exists() else epochs[-1], weights_only=False, map_location='cpu')
    if epochs and int(epochs[-1].stem.split('_')[1]) > state['completed_epoch']:
        state = torch.load(epochs[-1], weights_only=False, map_location='cpu')
    require(state['identity'] == identity and state['recipe'] == RECIPE
            and len(epochs) == len(state['history']) == state['completed_epoch'],
            'incompatible or incomplete warm-start recovery state')
    for row in state['history']:
        sha = digest(checkpoints / row['file'])
        require(row.get('sha256', sha) == sha, 'completed snapshot changed')
        row['sha256'] = sha
    return state


def _epoch_row(epoch, state, total_agents, total_windows, total_multi):
    totals = state['sums']
    require((totals['windows'], totals['agents'], totals['multi']) ==
            (total_windows, total_agents, total_multi), 'incomplete logical train epoch')
    return dict(
        epoch=epoch, file=f'epoch_{epoch:03d}.pt', update=state['update'],
        windows=total_windows, agents=total_agents, multi_windows=total_multi,
        minFDE_m=totals['mf_sum']/total_agents, JFDE_m=totals['jf_sum']/total_windows,
        TA_m=totals['a_teacher_sum']/(20*total_agents),
        TB_m=totals['b_teacher_sum']/(20*total_multi) if total_multi else 0.,
        sampled_minADE_HT_m=totals['weighted_ma']/total_agents,
        sampled_JADE_HT_m=totals['weighted_ja']/total_windows,
        mean_end_objective_per_update=totals['end_objective']/totals['updates'],
        mean_total_objective_per_update=totals['objective']/totals['updates'],
        train_seconds=totals['train_seconds'], train_ddim_calls=totals['ddim_calls'],
        max_grad_norm=totals['max_grad_norm'], sampled_agents=totals['drawn_agents'],
        dense_replays=totals['dense_replays'], dense_max_replay_diff_m=totals['dense_max_replay_diff_m'],
    )


def train(assets, student, cache, output, config, *, device='cuda', slot_chunk=0, dense_agents=24,
          should_yield=None):
    """Fine-tune one sealed fold. Development scoring is a separate worker."""
    started_unix = time.time()
    require(assets.fold in ('hotel', 'univ') and config.get('fold', assets.fold) == assets.fold,
            'only HOTEL and UNIV can use this fixed warm start')
    require(dense_agents > 0 and slot_chunk in (0, 1, 2, 4, 5, 10), 'invalid DDIM memory profile')
    output = Path(output)
    checkpoints = output / 'checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=True)
    indices = tuple(cache.indices)
    require(indices == tuple(assets.split['train_indices']) and len(indices) == len(set(indices)) > 0,
            'original train cache membership differs')
    counts = {index: len(assets.split['members'][index]['agent_ids']) for index in indices}
    require(min(counts.values()) > 0, 'empty training window')
    baseline = Path(config['S0_checkpoint'])
    require(digest(baseline) == config['S0_sha256'], 'S0 checkpoint SHA256 changed')
    s0 = torch.load(baseline, map_location='cpu', weights_only=False)
    s0_state = s0['student']
    native_state = {k: v for k, v in student.state_dict().items() if not k.startswith('group_query.')}
    require(s0['completed_epoch'] == config['S0_epoch'] and s0['cursor'] == 0
            and s0['identity']['upstream'] == assets.identity
            and s0['identity']['cache'] == cache.identity
            and canonical_hash(list(indices)) == s0['identity']['train_indices_sha256']
            and set(native_state) == set(s0_state)
            and all(torch.equal(value.detach().cpu(), s0_state[key]) for key, value in native_state.items()),
            'initial native student, teachers or train population differs from S0')
    scales = {metric: max(float(cache.scales[metric]), RECIPE['normalization_floor']) for metric in METRICS}
    require(scales == s0['identity']['scales_m'] == config.get('scales_m', scales)
            and all(math.isfinite(value) and value > 0 for value in scales.values()),
            'warm start must use the original train-only scales')
    require(all(not bool(p.count_nonzero()) for p in student.group_query[-1].parameters()),
            'group-query readout must be exactly zero at initialization')
    device = torch.device(device)
    student.to(device=device, dtype=torch.float32).train().requires_grad_(True)
    frozen = assets.frozen_states()
    require(frozen['provider'] == s0['identity']['frozen']['provider']
            and frozen['bridge_and_A'] == s0['identity']['frozen']['bridge'],
            'warm start uses different frozen native operators')
    names = dict(student.named_parameters())
    new = {f'group_query.{name}' for name, _ in student.group_query.named_parameters()}
    original = set(names) - new
    require(new and original and {'queries', 'personal', 'edge', 'latent'} ==
            {name.split('.')[0] for name in original}
            and all(p.dtype == torch.float32 and p.requires_grad for p in names.values()),
            'only expected original and new FP32 parameters may participate')
    optimizer_groups = [
        dict(name=tag, lr=rate, names=[name for name in names if name in selected],
             parameters=sum(p.numel() for name, p in names.items() if name in selected))
        for tag, rate, selected in (
            ('native_student', RECIPE['lr'], original),
            ('group_query_mlp', RECIPE['group_query_lr'], new),
        )
    ]
    identity = dict(
        config_sha256=canonical_hash(config), fold=assets.fold, recipe=RECIPE, scales_m=scales,
        frozen_sha256=frozen, cache=cache.identity, S0_sha256=config['S0_sha256'],
        initial_student_sha256=state_hash(student), dense_agent_cutoff=dense_agents,
        optimizer_groups=optimizer_groups,
    )
    immutable(output / 'TRAIN_IDENTITY.json', identity)
    result_path = output / 'TRAINING_RESULT.json'
    if result_path.exists():
        result = read(result_path)
        require(result['identity'] == identity and result['completed_epochs'] == RECIPE['epochs']
                and digest(result['final_checkpoint']) == result['final_sha256'], 'final model changed')
        return result
    seed_all(RECIPE['seed'])
    loader = torch.Generator().manual_seed(RECIPE['seed'])
    sampler = torch.Generator().manual_seed(RECIPE['trajectory_sample_seed'])
    optimizer = torch.optim.AdamW([
        dict(params=[names[name] for name in group['names']], lr=group['lr'])
        for group in optimizer_groups
    ], weight_decay=RECIPE['weight_decay'])
    state = _resume(checkpoints, identity)
    preflight = _preflight(assets, student, s0_state, indices, counts, output, canonical_hash(identity))
    if state is None:
        seed_all(RECIPE['seed'])
        state = dict(identity=identity, recipe=RECIPE, completed_epoch=0, update=0, cursor=0,
                     order=None, sums={}, history=[], scheduler=None, first_update=None,
                     fixed_candidate=RECIPE['epochs'], started_unix=started_unix, **_gpu_peaks(device))
    else:
        student.load_state_dict(state['student'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng'], loader)
        sampler.set_state(state['trajectory_sampler_rng'])
        _inventory(output, state['history'])
        if state['first_update'] is not None:
            immutable(output / 'FIRST_UPDATE.json', state['first_update'])

    def persist(path):
        state.update(student=student.state_dict(), optimizer=optimizer.state_dict(),
                     rng=rng_state(loader), trajectory_sampler_rng=sampler.get_state())
        save(state, path)

    profiles = state.setdefault('trajectory_memory_profiles', [])
    if not profiles or profiles[-1]['slot_chunk'] != slot_chunk:
        profiles.append(dict(slot_chunk=slot_chunk, dense_agents=dense_agents, after_update=state['update']))
        _progress(output, 'DDIM_MEMORY_PROFILE', fold=assets.fold, **profiles[-1])
    if state['order'] is None and state['completed_epoch'] == 0:
        persist(checkpoints / 'last.pt')
    groups, teachers = {}, {}

    def teacher(index, device):
        if index not in teachers:
            teachers[index] = cache.load(index)
        return {key: target.to(device=device) for key, target in teachers[index].items()}

    total_agents = sum(counts.values())
    total_multi = sum(n > 1 for n in counts.values())
    while state['completed_epoch'] < RECIPE['epochs']:
        epoch = state['completed_epoch'] + 1
        if state['order'] is None:
            state.update(order=torch.randperm(len(indices), generator=loader).tolist(),
                         sums=dict.fromkeys(VALUE_NAMES, 0.) | dict(
                             windows=0, agents=0, multi=0, updates=0, ddim_calls=0, drawn_agents=0,
                             weighted_ma=0., weighted_ja=0., train_seconds=0., max_grad_norm=0.,
                             dense_replays=0, dense_max_replay_diff_m=0.))
        while state['cursor'] < len(indices):
            tick = time.monotonic()
            cursor = state['cursor']
            batch = [indices[i] for i in state['order'][cursor:cursor+RECIPE['logical_windows']]]
            sizes = [counts[index] for index in batch]
            n_windows, n_agents = len(batch), sum(sizes)
            n_multi = sum(n > 1 for n in sizes)
            draws = _epoch_sample_count(n_windows)
            chosen = set(torch.randperm(n_windows, generator=sampler)[:draws].tolist())
            weights = _weights(n_agents, n_windows, n_multi, draws, scales)
            optimizer.zero_grad(set_to_none=True)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            values = torch.zeros(len(VALUE_NAMES), dtype=torch.float64, device=device)
            drifts = []
            try:
                for pos, (index, n) in enumerate(zip(batch, sizes)):
                    if _should_pause(should_yield):
                        _progress(output, 'YIELD_TO_COMPLETE_BATCH', fold=assets.fold, epoch=epoch,
                                  update=state['update'], windows=cursor, total=len(indices))
                        return None
                    terms, drift = _microbatch(assets, student, index, n, teacher, groups, pos in chosen,
                                                weights, slot_chunk if n >= dense_agents else 0)
                    values += terms
                    if drift is not None:
                        drifts.append(drift)
                values = values.tolist()
                require(all(math.isfinite(x) for x in values), 'nonfinite warm-start objective')
                grad = float(torch.nn.utils.clip_grad_norm_(
                    student.parameters(), RECIPE['clip_grad_norm'], error_if_nonfinite=True))
                optimizer.step()
            except torch.cuda.OutOfMemoryError:
                _progress(output, 'ROLLBACK_CUDA_OOM', fold=assets.fold, epoch=epoch,
                          update=state['update'], windows=cursor, total=len(indices),
                          agents=n, sampled_for_ddim=pos in chosen, slot_chunk=slot_chunk,
                          **_gpu_peaks(device))
                raise
            state['cursor'] += n_windows
            state['update'] += 1
            totals = state['sums']
            for name, value in zip(VALUE_NAMES, values):
                totals[name] += value
            p = draws / n_windows
            for name, value in dict(windows=n_windows, agents=n_agents, multi=n_multi,
                    updates=1, ddim_calls=draws, drawn_agents=sum(sizes[pos] for pos in chosen),
                    weighted_ma=values[4]/p, weighted_ja=values[5]/p,
                    train_seconds=time.monotonic()-tick, dense_replays=len(drifts)).items():
                totals[name] += value
            totals['max_grad_norm'] = max(totals['max_grad_norm'], grad)
            totals['dense_max_replay_diff_m'] = max(totals['dense_max_replay_diff_m'], max(drifts, default=0.))
            peaks = _gpu_peaks(device)
            for name, value in peaks.items():
                state[name] = max(state[name], value)
            progress = dict(fold=assets.fold, epoch=epoch, windows=state['cursor'], total=len(indices),
                            update=state['update'], grad_norm=grad, last_batch_draws=draws,
                            last_batch_p=p, phase_ddim_calls=totals['ddim_calls'],
                            epoch_train_seconds=totals['train_seconds'], slot_chunk=slot_chunk, **peaks)
            if state['update'] == 1:
                state['first_update'] = progress
            persist(checkpoints / 'last.pt')
            if state['update'] == 1:
                immutable(output / 'FIRST_UPDATE.json', progress)
            if state['update'] == 1 or state['update'] % 10 == 0:
                _progress(output, 'TRAIN_UPDATE', **progress)

        row = _epoch_row(epoch, state, total_agents, len(indices), total_multi)
        require(assets.frozen_states() == frozen, 'frozen weights changed during fine-tuning')
        state['history'].append(row)
        state.update(completed_epoch=epoch, cursor=0, order=None, sums={})
        snapshot = checkpoints / row['file']
        persist(snapshot)
        row['sha256'] = digest(snapshot)
        _atomic_alias(checkpoints / 'last.pt', snapshot)
        _inventory(output, state['history'])
        _progress(output, 'TRAIN_EPOCH_COMPLETE', fold=assets.fold, **row)

    history = state['history']
    require([r['epoch'] for r in history] == list(range(1, RECIPE['epochs']+1)),
            'not all scheduled epochs reached completion')
    for row in history:
        path = checkpoints / row['file']
        snapshot = torch.load(path, weights_only=False, map_location='cpu')
        require(digest(path) == row['sha256'] and snapshot['completed_epoch'] == row['epoch']
                and snapshot['cursor'] == 0 and snapshot['identity'] == identity
                and {'student', 'optimizer', 'scheduler', 'rng', 'trajectory_sampler_rng',
                     'update', 'order', 'history'} <= snapshot.keys(), 'incomplete recovery snapshot')
    final = checkpoints / history[-1]['file']
    for alias in ('final.pt', 'selected.pt'):
        _atomic_alias(checkpoints / alias, final)
    require(digest(baseline) == config['S0_sha256']
            and digest(cache.folder / 'MANIFEST.json') == cache.identity['teachers_manifest_sha256']
            and digest(cache.folder / 'TRAIN_B_SCALES.json') == cache.identity['scale_receipt_sha256']
            and assets.frozen_states() == frozen, 'read-only baseline, scales or teacher cache changed')
    result = dict(
        status='COMPLETE', fold=assets.fold, identity=identity, recipe=RECIPE,
        completed_epochs=RECIPE['epochs'], parameter_updates=state['update'],
        final_checkpoint=str(final.resolve()), final_sha256=history[-1]['sha256'],
        final_student_state_sha256=state_hash(student), optimizer_groups=optimizer_groups,
        frozen_before_sha256=frozen, frozen_after_sha256=assets.frozen_states(),
        train_seconds=math.fsum(row['train_seconds'] for row in history), wall_seconds=time.time()-state['started_unix'],
        peak_gpu_allocated_mib=state['peak_gpu_allocated_mib'],
        peak_gpu_reserved_mib=state['peak_gpu_reserved_mib'],
        train_ddim_calls=sum(row['train_ddim_calls'] for row in history),
        dense_replays=sum(row['dense_replays'] for row in history), trajectory_memory_profiles=profiles,
        first_update=state['first_update'], preflight=preflight, epoch_rows=history,
        retained_epochs=[row['epoch'] for row in history], development_reads=0, test_access=False,
    )
    immutable(result_path, result)
    return result

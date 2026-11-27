"""Fixed, resumable training of the endpoint interaction layer."""

from copy import deepcopy
import math
from pathlib import Path
import time

import torch

from bcep.contracts import canonical_hash, digest, require
from bcep_joint.common import read, restore_rng, rng_state, save, seed_all, write
from ebjd.sampling import differentiable_sample
from joint_flow_fourfold.common import immutable
from joint_flow_fourfold.inputs import state_hash
from .common import window_key
from .endpoint_interaction import interaction_goals
from .model import (
    K, NATIVE_ATOL, NATIVE_RTOL, _bridge_grid_sample, decode_student, encode_observed,
    student_endpoints, student_noise, trajectory_terms,
)
from .train import METRICS, _atomic_alias, _epoch_sample_count, _progress, _trajectory


RECIPE = dict(
    seed=3201, trajectory_sample_seed=3202, train_noise_seed=3201,
    optimizer='AdamW', learning_rate=3e-4, weight_decay=1e-4, gradient_clip=5.,
    logical_windows=16, epochs=12, K=20, ddim_steps=20, eta=0., precision='FP32',
    checkpoint_steps=True, trajectory_weight=.5, coverage_guard_weight=1.,
    trajectory_subsample='uniform without replacement, max(1,(batch_windows+4)//8)',
    ranking='only R12 receives development evaluation; no intermediate selection',
    retention='all 12 distinct completed epoch snapshots plus final/selected and last',
    recovery='atomic last after every complete batch; uncommitted batch rolled back',
)
VALUE_NAMES = ('minFDE_sum', 'JFDE_sum', 'guard_sum', 'MA_drawn_sum', 'JA_drawn_sum', 'loss')


def _endpoint_loss(g0, g1, target, weights):
    error = torch.linalg.vector_norm(g1.float() - target[None].float(), dim=-1)
    new_person = error.min(dim=0).values
    old_person = torch.linalg.vector_norm(g0.detach().float() - target[None].float(),
                                         dim=-1).min(dim=0).values
    marginal = new_person.sum()
    joint = error.mean(dim=1).min()
    guard = torch.relu(new_person - old_person).sum()
    loss = weights['MF'] * marginal + weights['JF'] * joint + weights['MF'] * guard
    return loss, marginal, joint, guard


def _weights(people, windows, draws, scales):
    p = draws / windows
    return dict(MF=1/(people*scales['minFDE']), JF=1/(windows*scales['JFDE']),
                MA=RECIPE['trajectory_weight']/(p*people*scales['minADE']),
                JA=RECIPE['trajectory_weight']/(p*windows*scales['JADE']))


def _sliced_trajectory(assets, batch, goals, key, weights, chunk):
    """Find full-K20 winners, then propagate their cotangent a few slots at a time."""
    with torch.no_grad():
        context = assets.model.encode_context(batch.observed, batch.semantic_maps, batch.valid)
        noise = student_noise(key, batch.agent_ids[0], RECIPE['train_noise_seed'], device=goals.device)
        full = decode_student(assets.model, context, goals.detach(), noise)
    full = full.detach().requires_grad_(True)
    ma, ja = trajectory_terms(full, batch.future, batch.valid)
    loss = weights['MA'] * ma + weights['JA'] * ja
    cotangent, = torch.autograd.grad(loss, full)
    require(bool(torch.isfinite(cotangent).all()), 'nonfinite full-K20 trajectory gradient')
    bridge = assets.model.bridge
    map_sampler = bridge.denoiser.map_sampler
    bridge.denoiser.map_sampler = _bridge_grid_sample
    gradients = []
    max_drift = 0.
    try:
        for start in range(0, K, chunk):
            selected_slots = slice(start, start + chunk)
            proposal = goals[selected_slots].detach().requires_grad_(True)
            replay, _, _ = differentiable_sample(
                bridge, context, noise[:, selected_slots], steps=20, checkpoint_steps=True,
                checkpoint_blocks=False, proposal_goals=proposal[None], proposal_eta=0.,
            )
            native_slots = full.detach()[:, selected_slots]
            torch.testing.assert_close(replay.detach(), native_slots, rtol=NATIVE_RTOL, atol=1e-5)
            torch.testing.assert_close(replay.detach()[..., -1, :], proposal.detach()[None],
                                       rtol=NATIVE_RTOL, atol=NATIVE_ATOL)
            max_drift = max(max_drift, float((replay.detach() - native_slots).abs().max()))
            gradient, = torch.autograd.grad(replay, proposal, cotangent[:, selected_slots])
            gradients.append(gradient)
    finally:
        bridge.denoiser.map_sampler = map_sampler
    goal_gradient = torch.cat(gradients)
    require(bool(torch.isfinite(goal_gradient).all()), 'nonfinite sliced endpoint gradient')
    return loss.detach(), ma.detach(), ja.detach(), goal_gradient, max_drift


def _microbatch(assets, student, module, index, n, sampled, weights, slot_chunk):
    batch = assets.batch(index, 'train', truth=True)
    require(batch.valid.shape == (1, n) and bool(batch.valid.all()), 'invalid packed training window')
    g0, g1 = interaction_goals(assets, student, module, batch)
    require(g1.requires_grad and not g0.requires_grad, 'frozen S0 or interaction graph is incorrect')
    loss, marginal, joint, guard = _endpoint_loss(g0, g1, batch.future[0, :, -1], weights)
    ma = ja = g1.new_zeros(())
    drift = None
    if sampled:
        key = window_key(assets.split, index)
        if slot_chunk:
            path_loss, ma, ja, gradient, drift = _sliced_trajectory(assets, batch, g1, key, weights, slot_chunk)
            torch.autograd.backward((loss, g1), (None, gradient))
            loss = loss.detach() + path_loss
        else:
            path = _trajectory(assets, batch, g1, key, seed=RECIPE['train_noise_seed'], input_grad=True)
            ma, ja = trajectory_terms(path, batch.future, batch.valid)
            loss = loss + weights['MA'] * ma + weights['JA'] * ja
    if loss.requires_grad:
        loss.backward()
    return torch.stack([x.detach() for x in (marginal, joint, guard, ma, ja, loss)]).double(), drift


def _native_context(assets, student, batch):
    with torch.no_grad():
        context, maps = encode_observed(assets.provider, batch)
        g0 = student_endpoints(student, assets.provider, batch.observed[0], batch.semantic_maps[0],
                               batch.valid[0], context, maps)
    return g0, context


def _preflight(assets, student, module, indices, counts, output, identity_sha):
    path = output / 'PREFLIGHT.json'
    if path.exists():
        receipt = read(path)
        require(receipt['identity_sha256'] == identity_sha and receipt['passed'],
                'training preflight identity changed')
        return receipt
    order = torch.randperm(len(indices), generator=torch.Generator().manual_seed(RECIPE['seed'])).tolist()
    first = [indices[pos] for pos in order[:RECIPE['logical_windows']]]
    multi = [index for index in first if counts[index] > 1]
    require(bool(multi), 'first logical batch has no interacting windows')
    index = min(multi, key=lambda member: (counts[member] < 6, counts[member]))
    batch = assets.batch(index, 'train', truth=True)
    g0, context = _native_context(assets, student, batch)
    dt = float(assets.split['dt'])
    g1 = module(g0, batch.observed[0], context, batch.valid[0], dt=dt)
    require(torch.equal(g0, g1), 'zero interaction readout did not reproduce S0 endpoints')
    key = window_key(assets.split, index)
    native = _trajectory(assets, batch, g0, key, seed=RECIPE['train_noise_seed'], input_grad=False)
    refined = _trajectory(assets, batch, g1, key, seed=RECIPE['train_noise_seed'], input_grad=True)
    torch.testing.assert_close(refined.detach(), native, rtol=NATIVE_RTOL, atol=NATIVE_ATOL)
    interior_error = (refined[..., :-1, :] - batch.future[:, None, :, :-1, :]).norm(dim=-1).mean()
    gradients = torch.autograd.grad(interior_error, tuple(module.parameters()), allow_unused=True)
    finite = all(grad is None or bool(torch.isfinite(grad).all()) for grad in gradients)
    grad_max = {name: float(grad.abs().max()) for (name, _), grad in
                zip(module.named_parameters(), gradients) if grad is not None}
    groups = {branch: max((value for name, value in grad_max.items() if name.startswith(prefix)),
                          default=0.)
              for branch, prefix in (('motion', 'motion.'), ('social', 'pair_displacement.'))}
    require(finite and min(groups.values()) > 0,
            'real DDIM interior gradient did not reach both displacement heads')
    require(all(p.grad is None for p in module.parameters()), 'preflight mutated optimizer gradients')
    # Check equivariance with a nonzero displacement, on a temporary model only.
    test_module = deepcopy(module)
    order = torch.arange(len(context) - 1, -1, -1, device=g0.device)
    with torch.no_grad():
        for parameter, gradient in zip(test_module.parameters(), gradients):
            if gradient is not None and bool(gradient.count_nonzero()):
                parameter.add_(gradient / gradient.abs().max(), alpha=1e-3)
        observed, valid = batch.observed[0], batch.valid[0]
        probe = test_module(g0, observed, context, valid, dt=dt)
        permuted = test_module(g0[:, order], observed[order], context[order], valid[order], dt=dt)
        require(not torch.equal(probe, g0), 'permutation check needs a nonzero displacement')
        torch.testing.assert_close(permuted, probe[:, order], rtol=NATIVE_RTOL, atol=NATIVE_ATOL)
        # Keeping only a real agent exercises the no-neighbor branch with the same nonzero readout.
        _, _, social = test_module(g0[:, :1], observed[:1], context[:1], valid[:1],
                                   dt=dt, return_components=True)
        require(not bool(social.count_nonzero()), 'single-agent social feature is not exactly zero')
    receipt = dict(
        passed=True, identity_sha256=identity_sha, agents=len(context), optimizer_updates=0,
        window_sha256=key, g0_identity=True, max_trajectory_abs_diff_m=float((refined.detach()-native).abs().max()),
        permutation_max_abs_diff_m=float((permuted - probe[:, order]).abs().max()),
        singleton_social_exact_zero=True, trajectory_interior_grad_max=max(grad_max.values()),
        max_grad_per_displacement_head=groups,
        actual_ddim_calls=2,
    )
    immutable(path, receipt)
    _progress(output, 'PREFLIGHT_PASSED', fold=assets.fold, **receipt)
    return receipt


def _gpu_peaks(device):
    if device.type == 'cuda':
        return dict(peak_gpu_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                    peak_gpu_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20)
    return dict(peak_gpu_allocated_mib=0., peak_gpu_reserved_mib=0.)


def _frozen(assets, student):
    require(all(not module.training for module in student.modules()) and
            all(not p.requires_grad and p.grad is None for p in student.parameters()),
            'S0 must be frozen during interaction training')
    return assets.frozen_states() | dict(S0=state_hash(student))


def _inventory(output, checkpoints, history):
    write(output / 'EPOCH_METRICS.json', history)
    write(checkpoints / 'TOP5.json', dict(
        selector=RECIPE['ranking'], ranked_epoch_count=0, entries=[],
        snapshots=[{key: row[key] for key in ('epoch', 'file', 'sha256')} for row in history],
    ))


def record_r12_metrics(output, metrics):
    """Attach only the prescribed final development score to the checkpoint index."""
    checkpoints = Path(output) / 'checkpoints'
    index = read(checkpoints / 'TOP5.json')
    require(len(index['snapshots']) == RECIPE['epochs'], 'cannot score a partial training run')
    selected = index['snapshots'][-1]
    require(selected['epoch'] == RECIPE['epochs'] and digest(checkpoints / selected['file']) == selected['sha256'],
            'R12 snapshot changed before score indexing')
    values = {key: float(metrics[key]) for key in METRICS}
    require(all(math.isfinite(value) for value in values.values()), 'nonfinite final development metric')
    index.update(ranked_epoch_count=1, entries=[dict(selected, rank=1, metrics=values, purpose='selection_dev')])
    write(checkpoints / 'TOP5.json', index)
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
    require(state['identity'] == identity and state['recipe'] == RECIPE, 'different run in last checkpoint')
    for row in state['history']:
        sha = digest(checkpoints / row['file'])
        require(row.get('sha256', sha) == sha, 'completed checkpoint changed since recovery')
        row['sha256'] = sha
    require(len(epochs) == len(state['history']) == state['completed_epoch'], 'completed epoch inventory differs')
    return state


def train(assets, student, module, output, config, *, dense_agents=24, slot_chunk=5,
          should_yield=None):
    """Train exactly one fold; emit the sole candidate only after the twelfth complete epoch."""
    require(assets.fold in ('hotel', 'univ'), 'only HOTEL and UNIV are in scope')
    require(dense_agents > 0 and slot_chunk in (0, 1, 2, 4, 5, 10), 'invalid DDIM slot replay settings')
    output = Path(output)
    checkpoints = output / 'checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=True)
    indices = tuple(assets.split['train_indices'])
    require(len(indices) == len(set(indices)) > 0, 'bad train membership')
    counts = {index: len(assets.split['members'][index]['agent_ids']) for index in indices}
    require(min(counts.values()) > 0, 'empty train window')
    baseline = Path(config['S0_checkpoint'])
    require(digest(baseline) == config['S0_sha256'], 'S0 checkpoint changed')
    s0 = torch.load(baseline, map_location='cpu', weights_only=False)
    require(s0['completed_epoch'] == config['S0_epoch'] and s0['cursor'] == 0 and
            s0['identity']['upstream'] == assets.identity and
            canonical_hash(list(indices)) == s0['identity']['train_indices_sha256'] and
            all(torch.equal(v.detach().cpu(), s0['student'][k]) for k, v in student.state_dict().items()),
            'S0 checkpoint, native upstream, or train indices do not agree')
    scales = {metric: float(config['scales_m'][metric]) for metric in METRICS}
    require(scales == s0['identity']['scales_m'] and
            all(math.isfinite(x) and x >= 1e-6 for x in scales.values()),
            'normalization differs from the original train scales')
    device = torch.device(assets.device)
    student.eval().requires_grad_(False)
    module.to(device=device, dtype=torch.float32).train().requires_grad_(True)
    frozen = _frozen(assets, student)
    identity = dict(config_sha256=canonical_hash(config), fold=assets.fold, frozen=frozen,
                    S0_sha256=config['S0_sha256'], recipe=RECIPE, scales_m=scales,
                    dense_agent_cutoff=dense_agents,
                    initial_interaction_sha256=state_hash(module))
    immutable(output / 'TRAIN_IDENTITY.json', identity)
    result_path = output / 'TRAINING_RESULT.json'
    if result_path.exists():
        result = read(result_path)
        require(result['identity'] == identity and result['completed_epochs'] == RECIPE['epochs'] and
                digest(result['R12_checkpoint']) == result['R12_sha256'], 'completed training identity changed')
        return result
    seed_all(RECIPE['seed'])
    loader = torch.Generator().manual_seed(RECIPE['seed'])
    sampler = torch.Generator().manual_seed(RECIPE['trajectory_sample_seed'])
    optimizer = torch.optim.AdamW(module.parameters(), lr=RECIPE['learning_rate'],
                                  weight_decay=RECIPE['weight_decay'])
    state = _resume(checkpoints, identity)
    preflight = _preflight(assets, student, module, indices, counts, output, canonical_hash(identity))
    if state is None:
        state = dict(identity=identity, recipe=RECIPE, completed_epoch=0, update=0, cursor=0,
                     order=None, sums={}, history=[], scheduler=None, first_update=None,
                     fixed_candidate=RECIPE['epochs'], **_gpu_peaks(torch.device('cpu')))
    else:
        module.load_state_dict(state['interaction'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng'], loader)
        sampler.set_state(state['trajectory_sampler_rng'])
        _inventory(output, checkpoints, state['history'])

    def persist(path):
        state.update(interaction=module.state_dict(), optimizer=optimizer.state_dict(),
                     rng=rng_state(loader), trajectory_sampler_rng=sampler.get_state())
        save(state, path)

    if state['first_update'] is not None:
        immutable(output / 'FIRST_UPDATE.json', state['first_update'])
    profiles = state.setdefault('trajectory_memory_profiles', [])
    if not profiles or profiles[-1]['slot_chunk'] != slot_chunk:
        profiles.append(dict(slot_chunk=slot_chunk, dense_agents=dense_agents, after_update=state['update']))
        _progress(output, 'DDIM_MEMORY_PROFILE', fold=assets.fold, **profiles[-1])
    if state['order'] is None and state['completed_epoch'] == 0:
        persist(checkpoints / 'last.pt')

    total_people = sum(counts.values())
    while state['completed_epoch'] < RECIPE['epochs']:
        epoch = state['completed_epoch'] + 1
        if state['order'] is None:
            state.update(order=torch.randperm(len(indices), generator=loader).tolist(),
                         sums=dict.fromkeys(VALUE_NAMES, 0.) | dict(
                             windows=0, agents=0, batches=0, ddim_calls=0, replay_windows=0,
                             replay_max_abs_m=0., train_seconds=0., weighted_MA=0., weighted_JA=0.))
        while state['cursor'] < len(indices):
            tick = time.monotonic()
            cursor = state['cursor']
            positions = state['order'][cursor:cursor+RECIPE['logical_windows']]
            batch = [indices[position] for position in positions]
            sizes = [counts[index] for index in batch]
            n_windows, n_people = len(batch), sum(sizes)
            draws = _epoch_sample_count(n_windows)
            sampled = set(torch.randperm(n_windows, generator=sampler)[:draws].tolist())
            weights = _weights(n_people, n_windows, draws, scales)
            optimizer.zero_grad(set_to_none=True)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            values = torch.zeros(len(VALUE_NAMES), dtype=torch.float64, device=device)
            drift = []
            try:
                for i, (index, n) in enumerate(zip(batch, sizes)):
                    if should_yield is not None and should_yield():
                        _progress(output, 'YIELD_TO_COMPLETE_BATCH', fold=assets.fold, epoch=epoch,
                                  update=state['update'], windows=cursor, total=len(indices))
                        return None
                    sums, replay_diff = _microbatch(assets, student, module, index, n, i in sampled,
                                                    weights, slot_chunk if n >= dense_agents else 0)
                    values += sums
                    if replay_diff is not None:
                        drift.append(replay_diff)
                values = values.tolist()
                require(all(math.isfinite(x) for x in values), 'nonfinite interaction objective')
                gradient = torch.nn.utils.clip_grad_norm_(module.parameters(), RECIPE['gradient_clip'],
                                                          error_if_nonfinite=True)
                optimizer.step()
            except torch.cuda.OutOfMemoryError:
                _progress(output, 'ROLLBACK_CUDA_OOM', fold=assets.fold, epoch=epoch,
                          update=state['update'], windows=cursor, total=len(indices), agents=n,
                          sampled_for_ddim=i in sampled, slot_chunk=slot_chunk, **_gpu_peaks(device))
                raise
            state['cursor'] += n_windows
            state['update'] += 1
            metrics = state['sums']
            for key, value in zip(VALUE_NAMES, values):
                metrics[key] += value
            probability = draws / n_windows
            for key, value in dict(windows=n_windows, agents=n_people, batches=1, ddim_calls=draws,
                    replay_windows=len(drift), train_seconds=time.monotonic()-tick,
                    weighted_MA=values[3]/probability, weighted_JA=values[4]/probability).items():
                metrics[key] += value
            metrics['replay_max_abs_m'] = max(metrics['replay_max_abs_m'], max(drift, default=0.))
            gpu = _gpu_peaks(device)
            for key, value in gpu.items():
                state[key] = max(state[key], value)
            info = dict(fold=assets.fold, epoch=epoch, update=state['update'], windows=state['cursor'],
                        total=len(indices), grad_norm=float(gradient), phase_ddim_calls=metrics['ddim_calls'],
                        last_batch_draws=draws, last_batch_p=probability, epoch_train_seconds=metrics['train_seconds'],
                        dense_replays_this_update=len(drift), slot_chunk=slot_chunk, **gpu)
            if state['update'] == 1:
                state['first_update'] = info
            persist(checkpoints / 'last.pt')
            if state['update'] == 1:
                immutable(output / 'FIRST_UPDATE.json', info)
            if state['update'] == 1 or state['update'] % 10 == 0:
                _progress(output, 'TRAIN_UPDATE', **info)

        sums = state['sums']
        require((sums['windows'], sums['agents']) == (len(indices), total_people), 'cannot save partial epoch')
        require(_frozen(assets, student) == frozen, 'a frozen weight changed')
        row = dict(epoch=epoch, update=state['update'], file=f'epoch_{epoch:03d}.pt', **sums)
        state['history'].append(row)
        state.update(completed_epoch=epoch, cursor=0, order=None, sums={})
        target = checkpoints / row['file']
        persist(target)
        row['sha256'] = digest(target)
        _atomic_alias(checkpoints / 'last.pt', target)
        _inventory(output, checkpoints, state['history'])
        _progress(output, 'TRAIN_EPOCH_COMPLETE', fold=assets.fold, epoch=epoch,
                  update=state['update'], train_seconds=row['train_seconds'], train_ddim_calls=row['ddim_calls'])

    history = state['history']
    require([row['epoch'] for row in history] == list(range(1, RECIPE['epochs']+1)),
            'not all scheduled epochs were completed')
    for row in history:
        require(digest(checkpoints / row['file']) == row['sha256'], 'a retained epoch weight changed')
    final = checkpoints / history[-1]['file']
    for alias in ('final.pt', 'selected.pt'):
        _atomic_alias(checkpoints / alias, final)
    result = dict(
        status='COMPLETE', identity=identity, fold=assets.fold, completed_epochs=RECIPE['epochs'],
        updates=state['update'], R12_checkpoint=str(final.resolve()), R12_sha256=digest(final),
        interaction_sha256=state_hash(module), frozen_before=frozen, frozen_after=_frozen(assets, student),
        train_seconds=math.fsum(row['train_seconds'] for row in history),
        train_ddim_calls=sum(row['ddim_calls'] for row in history),
        dense_replay_windows=sum(row['replay_windows'] for row in history),
        dense_replay_max_abs_m=max(row['replay_max_abs_m'] for row in history),
        trajectory_memory_profiles=profiles,
        peak_gpu_allocated_mib=state['peak_gpu_allocated_mib'],
        peak_gpu_reserved_mib=state['peak_gpu_reserved_mib'],
        retained_epochs=[row['epoch'] for row in history], epoch_rows=history,
        first_update=state['first_update'], preflight=preflight, test_access=False,
    )
    immutable(result_path, result)
    return result

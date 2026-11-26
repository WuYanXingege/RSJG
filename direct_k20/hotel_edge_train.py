"""Continue the HOTEL S24 student with a new edge-only optimizer."""
from contextlib import nullcontext
import hashlib
import math
import os
from pathlib import Path
import time

import torch

from bcep.contracts import canonical_hash, digest, require
from bcep_joint.common import read, restore_rng, rng_state, save, seed_all, write
from ebjd.sampling import differentiable_sample
from joint_flow_fourfold.common import immutable
from joint_flow_fourfold.inputs import state_hash
from .assets import TeacherCache
from .model import (
    K, NATIVE_ATOL, NATIVE_RTOL, _bridge_grid_sample, decode_student,
    endpoint_terms, student_noise, teacher_terms, trajectory_terms,
)
from .train import (
    METRICS, RECIPE as ORIGINAL_RECIPE, _atomic_alias, _epoch_sample_count,
    _progress, _trajectory, _window_goals,
)

FIRST_EPOCH = 25
FINAL_EPOCH = 28
RECIPE = ORIGINAL_RECIPE | dict(
    lr=1e-4, epochs=4, first_epoch=FIRST_EPOCH, final_epoch=FINAL_EPOCH,
    end_only_epochs=0, evaluation_epochs=[FINAL_EPOCH],
    optimizer='new AdamW for student.edge; other settings from the original recipe',
    extra_weight=1., extra_population=[2, 8],
    extra_reduction='all eligible windows in the logical batch, including zero penalties',
    extra_reference='native B cached post-head joint endpoint error',
    ranking='fixed final epoch and single S24/S28 development acceptance',
    retention='keep all four completed epochs, final/selected and last; no intermediate dev scores',
    recovery='atomic last after each full batch with RNG, order and cursor; roll back an uncommitted batch',
)


def _cache(assets, folder, original):
    folder = Path(folder)
    manifest = read(folder / 'MANIFEST.json')
    scales = read(folder / 'TRAIN_B_SCALES.json')
    cache = TeacherCache(folder, manifest, scales)
    members = assets.split['members']
    indices = tuple(assets.split['train_indices'])
    require(cache.indices == indices and cache.identity == original['cache']
            and assets.identity == original['upstream']
            and canonical_hash(list(indices)) == original['train_indices_sha256'],
            'HOTEL train cache or upstream identity differs from S24')
    require(manifest['identity']['purpose'] == scales['purpose'] == 'train'
            and manifest['identity']['seed'] == scales['seed'] == 2035
            and scales['teachers_manifest_sha256'] == cache.manifest_sha256
            and manifest['target_reads_during_export'] == 0,
            'train normalization receipt differs from the S24 teacher cache')
    for index in indices:
        row = cache.rows[index]
        require(row['n'] == len(members[index]['agent_ids'])
                and row['member_sha256'] == canonical_hash(members[index])
                and row['record_sha256'] == assets.split['records'][index]['sha256'],
                'teacher window does not match the original train member')
    return cache


def _frozen_states(assets, student):
    frozen_modules = dict(personal=student.personal, latent=student.latent)
    for module in frozen_modules.values():
        require(not any(part.training for part in module.modules())
                and all(not p.requires_grad and p.grad is None for p in module.parameters()),
                'a non-edge student module is no longer frozen')
    require(not student.queries.requires_grad and student.queries.grad is None,
            'student queries are no longer frozen')
    queries = student.queries.detach().cpu().contiguous().numpy().tobytes()
    return assets.frozen_states() | {n: state_hash(m) for n, m in frozen_modules.items()} | {
        'queries': hashlib.sha256(queries).hexdigest(),
    }


def _batch_weights(counts, draws, scales):
    windows, agents = len(counts), sum(counts)
    multi = sum(n > 1 for n in counts)
    eligible = sum(2 <= n <= 8 for n in counts)
    p = draws / windows
    return dict(
        MF=1/(agents*scales['minFDE']), JF=1/(windows*scales['JFDE']),
        TA=RECIPE['teacher_weight']/(20*agents*scales['minFDE']),
        TB=RECIPE['teacher_weight']/(20*multi*scales['JFDE']) if multi else 0.,
        MA=RECIPE['trajectory_weight']/(p*agents*scales['minADE']),
        JA=RECIPE['trajectory_weight']/(p*windows*scales['JADE']),
        extra=RECIPE['extra_weight']/(eligible*scales['JFDE']) if eligible else 0.,
    )


def _trajectory_slot_vjp(assets, batch, goals, key, w, slot_chunk):
    """Compute the whole K20 trajectory loss, then replay its VJP in slot slices."""
    with torch.no_grad():
        context = assets.model.encode_context(batch.observed, batch.semantic_maps, batch.valid)
        noise = student_noise(key, batch.agent_ids[0], RECIPE['train_bridge_seed'], device=goals.device)
        native = decode_student(assets.model, context, goals.detach(), noise)
    path = native.detach().requires_grad_(True)
    ma, ja = trajectory_terms(path, batch.future, batch.valid)
    loss = w['MA']*ma + w['JA']*ja
    cotangent, = torch.autograd.grad(loss, path)
    require(bool(torch.isfinite(cotangent).all()), 'nonfinite native K20 trajectory cotangent')

    # The bridge has no operations mixing proposal slots. The native forward
    # above still resolves both minima jointly across all twenty slots.
    bridge = assets.model.bridge
    original_map_sampler = bridge.denoiser.map_sampler
    bridge.denoiser.map_sampler = _bridge_grid_sample
    gradients = []
    max_replay_diff = 0.
    try:
        for first in range(0, K, slot_chunk):
            slots = slice(first, first + slot_chunk)
            proposal = goals[slots].detach().requires_grad_(True)
            replay, _, _ = differentiable_sample(
                bridge, context, noise[:, slots], steps=20, checkpoint_steps=True,
                checkpoint_blocks=False, proposal_goals=proposal[None], proposal_eta=0.,
            )
            torch.testing.assert_close(replay.detach()[..., -1, :], proposal.detach()[None],
                                       rtol=NATIVE_RTOL, atol=NATIVE_ATOL)
            reference = native[:, slots]
            torch.testing.assert_close(replay.detach(), reference, rtol=NATIVE_RTOL, atol=1e-5)
            max_replay_diff = max(max_replay_diff, float((replay.detach() - reference).abs().max()))
            gradient, = torch.autograd.grad(replay, proposal, cotangent[:, slots])
            gradients.append(gradient)
    finally:
        bridge.denoiser.map_sampler = original_map_sampler
    goal_gradient = torch.cat(gradients)
    require(bool(torch.isfinite(goal_gradient).all()), 'nonfinite sliced bridge cotangent')
    return loss.detach(), goal_gradient, max_replay_diff


def _microbatch(assets, student, cache, index, n, selected, w, slot_chunk=0, replays=None):
    batch = assets.batch(index, 'train', truth=True)
    require(batch.valid.shape == (1, n) and bool(batch.valid.all()),
            'training member mask differs from the cache')
    teacher = cache.load(index, batch.observed.device)
    goals, key = _window_goals(assets, student, index, batch, input_grad=n > 1)
    truth = batch.future[0, :, -1]
    mf, jf = endpoint_terms(goals, truth)
    ta, tb = teacher_terms(goals, teacher['A'], teacher['B'])
    total = w['MF']*mf + w['JF']*jf + w['TA']*ta
    require((tb is not None) == (n > 1), 'teacher B scene term was applied to the wrong population')
    if tb is not None:
        total = total + w['TB']*tb
    extra = goals.new_zeros(())
    if 2 <= n <= 8:
        _, ref_jf = endpoint_terms(teacher['B'].detach(), truth)
        extra = torch.relu(jf - ref_jf)
        total = total + w['extra']*extra
    trajectory_backward = False
    if selected and slot_chunk and n >= 24 and goals.requires_grad:
        trajectory_loss, goal_gradient, replay_diff = _trajectory_slot_vjp(
            assets, batch, goals, key, w, slot_chunk)
        torch.autograd.backward((total, goals), (None, goal_gradient))
        total = total.detach() + trajectory_loss
        trajectory_backward = True
        if replays is not None:
            replays.append(dict(agents=n, slot_chunk=slot_chunk, max_path_replay_abs_m=replay_diff))
    elif selected:
        path = _trajectory(assets, batch, goals, key, seed=RECIPE['train_bridge_seed'],
                           input_grad=goals.requires_grad)
        ma, ja = trajectory_terms(path, batch.future, batch.valid)
        total = total + w['MA']*ma + w['JA']*ja
        trajectory_backward = path.requires_grad
    if total.requires_grad:
        total.backward()
    return torch.stack((total.detach(), extra.detach())).double(), trajectory_backward


def _memory(device):
    if device.type == 'cuda':
        return dict(peak_gpu_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20,
                    peak_gpu_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20)
    return dict(peak_gpu_allocated_mib=0., peak_gpu_reserved_mib=0.)


def _manifest(output, checkpoints, history):
    write(output / 'EPOCH_METRICS.json', history)
    write(checkpoints / 'TOP5.json', dict(
        selector=RECIPE['ranking'],
        ranked_epoch_count=0,
        all_completed_epochs_retained=len(history),
        entries=[dict(epoch=row['epoch'], file=row['file'], sha256=row['sha256'],
                      metrics=None, rank=None) for row in history],
        note='Only S28 is evaluated after training; all four epochs are kept.',
    ))


def _latest(checkpoints, identity):
    last = checkpoints / 'last.pt'
    epochs = sorted(checkpoints.glob('epoch_*.pt'))
    if not last.exists() and not epochs:
        return None
    state = torch.load(last if last.exists() else epochs[-1], weights_only=False, map_location='cpu')
    if epochs and int(epochs[-1].stem.split('_')[1]) > state['completed_epoch']:
        state = torch.load(epochs[-1], weights_only=False, map_location='cpu')
    require(state['identity'] == identity and state['recipe'] == RECIPE,
            'recovery checkpoint belongs to another experiment')
    for row in state['history']:
        observed = digest(checkpoints / row['file'])
        require(row.get('sha256', observed) == observed, 'a completed checkpoint changed')
        row['sha256'] = observed
    return state


def train(assets, student, teacher_cache, run, config, should_yield=None, *,
          activation_offload=False, dense_slot_chunk=0):
    """Run the fixed four epochs, returning None when the auxiliary worker yields."""
    require(assets.fold == 'hotel', 'this continuation accepts HOTEL only')
    require(0 <= dense_slot_chunk <= K, 'invalid dense replay slot size')
    output = Path(run) / 'training'
    checkpoints = output / 'checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=True)
    s24 = Path(config['S24_checkpoint'])
    require(digest(s24) == config['S24_sha256'], 'HOTEL S24 checkpoint hash differs')
    origin = torch.load(s24, map_location='cpu', weights_only=False)
    require(origin['completed_epoch'] == FIRST_EPOCH - 1 and origin['cursor'] == 0
            and origin['recipe'] == ORIGINAL_RECIPE
            and all(torch.equal(tensor.detach().cpu(), origin['student'][name])
                    for name, tensor in student.state_dict().items()),
            'continuation must start from the complete S24 student')
    cache = _cache(assets, teacher_cache, origin['identity'])
    scales = {m: float(cache.scales[m]) for m in METRICS}
    require(scales == origin['identity']['scales_m']
            and all(math.isfinite(x) and x >= RECIPE['normalization_floor'] for x in scales.values()),
            'train scale differs from S24')
    device = next(student.parameters()).device
    student.eval().requires_grad_(False)
    student.edge.train().requires_grad_(True)
    params = list(student.edge.parameters())
    require(all(p.dtype == torch.float32 for p in student.parameters())
            and {id(p) for p in student.parameters() if p.requires_grad} == {id(p) for p in params},
            'only FP32 edge parameters may train')
    frozen = _frozen_states(assets, student)
    identity = dict(config_sha256=canonical_hash(config), recipe=RECIPE, cache=cache.identity,
                    frozen=frozen, S24_sha256=config['S24_sha256'])
    immutable(output / 'TRAIN_IDENTITY.json', identity)

    seed_all(RECIPE['seed'])
    loader = torch.Generator().manual_seed(RECIPE['seed'])
    sampler = torch.Generator().manual_seed(RECIPE['seed'] + 1)
    optimizer = torch.optim.AdamW(params, lr=RECIPE['lr'], weight_decay=RECIPE['weight_decay'])
    saved = _latest(checkpoints, identity)
    if saved is None:
        state = dict(identity=identity, recipe=RECIPE, completed_epoch=FIRST_EPOCH-1,
                     update=0, completed_batches=0, cursor=0, order=None, history=[],
                     sums={}, first_update=None, first_ddim_backward_update=None,
                     scheduler=None, student_selector=RECIPE['ranking'], **_memory(torch.device('cpu')))
    else:
        state = saved
        student.load_state_dict(state['student'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng'], loader)
        sampler.set_state(state['trajectory_sampler_rng'])
        _manifest(output, checkpoints, state['history'])
    state.setdefault('dense_slot_replays', dict(windows=0, max_path_replay_abs_m=0.))
    require(_frozen_states(assets, student) == frozen, 'resumed frozen weights differ from S24')

    def persist(path):
        state.update(student=student.state_dict(), optimizer=optimizer.state_dict(),
                     rng=rng_state(loader), trajectory_sampler_rng=sampler.get_state())
        save(state, path)

    def milestones():
        for field, filename in (('first_update', 'FIRST_UPDATE.json'),
                                ('first_ddim_backward_update', 'FIRST_DDIM_BACKWARD_UPDATE.json')):
            if state[field] is not None:
                immutable(output / filename, state[field])

    def rollback(reason, epoch, n=None, selected=None):
        info = dict(reason=reason, epoch=epoch, cursor=state['cursor'],
                    completed_batches=state['completed_batches'], updates=state['update'],
                    interrupted_window_agents=n, sampled_for_ddim=selected,
                    activation_offload=activation_offload, worker_pid=os.getpid(),
                    dense_slot_chunk=dense_slot_chunk,
                    **_memory(device))
        write(output / 'ROLLBACK.json', dict(info, unix=time.time()))
        _progress(output, 'ROLL_BACK_UNCOMMITTED_BATCH', **info)

    indices = cache.indices
    counts = {index: cache.rows[index]['n'] for index in indices}
    milestones()
    if saved is None:
        persist(checkpoints / 'last.pt')
    while state['completed_epoch'] < FINAL_EPOCH:
        epoch = state['completed_epoch'] + 1
        if state['order'] is None:
            state.update(order=torch.randperm(len(indices), generator=loader).tolist(),
                         sums=dict(windows=0, agents=0, updates=0, skipped_batches=0,
                                   eligible_windows=0, active_windows=0, train_ddim_calls=0,
                                   ddim_backward_windows=0, extra_sum_m=0., objective_sum=0.,
                                   train_seconds=0.))
        while state['cursor'] < len(indices):
            tick = time.monotonic()
            cursor = state['cursor']
            positions = state['order'][cursor:cursor+RECIPE['logical_windows']]
            batch_indices = [indices[pos] for pos in positions]
            batch_counts = [counts[i] for i in batch_indices]
            windows = len(batch_indices)
            draws = _epoch_sample_count(windows)
            selected = set(torch.randperm(windows, generator=sampler)[:draws].tolist())
            weights = _batch_weights(batch_counts, draws, scales)
            optimizer.zero_grad(set_to_none=True)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            objectives = torch.zeros((windows, 2), device=device, dtype=torch.float64)
            backward_counts = []
            replay_diagnostics = []
            n = None
            sampled = None
            try:
                for pos, (index, n) in enumerate(zip(batch_indices, batch_counts)):
                    sampled = pos in selected
                    if should_yield is not None and should_yield():
                        rollback('resource_yield', epoch, n, sampled)
                        return None
                    offload = (torch.autograd.graph.save_on_cpu(pin_memory=True)
                               if activation_offload and device.type == 'cuda' and n > 1
                               else nullcontext())
                    with offload:
                        objectives[pos], trajectory_backward = _microbatch(
                            assets, student, cache, index, n, sampled, weights,
                            dense_slot_chunk, replay_diagnostics)
                    if trajectory_backward:
                        backward_counts.append(n)
                values = objectives.sum(0).tolist()
                active = int((objectives[:, 1] > 0.).sum())
                require(all(math.isfinite(v) for v in values), 'nonfinite training objective')
                actual_update = any(p.grad is not None for p in params)
                grad_norm = (float(torch.nn.utils.clip_grad_norm_(
                    params, RECIPE['clip_grad_norm'], error_if_nonfinite=True)) if actual_update else 0.)
                if actual_update:
                    optimizer.step()
            except torch.cuda.OutOfMemoryError:
                rollback('cuda_oom', epoch, n, sampled)
                raise

            memory = _memory(device)
            for name, value in memory.items():
                state[name] = max(state[name], value)
            state['cursor'] += windows
            state['completed_batches'] += 1
            state['update'] += int(actual_update)
            for diagnostic in replay_diagnostics:
                replays = state['dense_slot_replays']
                replays['windows'] += 1
                replays['max_path_replay_abs_m'] = max(replays['max_path_replay_abs_m'],
                                                     diagnostic['max_path_replay_abs_m'])
            eligible = sum(2 <= n <= 8 for n in batch_counts)
            for name, value in dict(windows=windows, agents=sum(batch_counts),
                    updates=int(actual_update), skipped_batches=int(not actual_update),
                    eligible_windows=eligible, active_windows=active, train_ddim_calls=draws,
                    ddim_backward_windows=len(backward_counts), extra_sum_m=values[1],
                    objective_sum=values[0], train_seconds=time.monotonic()-tick).items():
                state['sums'][name] += value
            info = dict(epoch=epoch, update=state['update'], batch=state['completed_batches'],
                        cursor=state['cursor'], total=len(indices), gradient_norm=grad_norm,
                        batch_eligible=eligible, batch_triggered=active,
                        sampled_agents=[batch_counts[pos] for pos in sorted(selected)],
                        ddim_backward_agents=backward_counts, activation_offload=activation_offload,
                        dense_slot_replay=replay_diagnostics,
                        worker_pid=os.getpid(), **memory)
            first_update = actual_update and state['first_update'] is None
            first_ddim = actual_update and bool(backward_counts) and state['first_ddim_backward_update'] is None
            if first_update:
                state['first_update'] = info
            if first_ddim:
                state['first_ddim_backward_update'] = info
            persist(checkpoints / 'last.pt')
            if first_update or first_ddim:
                milestones()
            if replay_diagnostics or first_ddim or first_update or state['completed_batches'] % 10 == 0:
                _progress(output, 'TRAIN_UPDATE', **info)
            if should_yield is not None and should_yield():
                _progress(output, 'YIELD_AT_COMPLETE_BATCH', **info)
                return None

        require(state['sums']['windows'] == len(indices)
                and state['sums']['agents'] == sum(counts.values())
                and state['sums']['eligible_windows'] == sum(2 <= n <= 8 for n in counts.values()),
                'partial epoch cannot be finalized')
        require(_frozen_states(assets, student) == frozen, 'a frozen parameter changed during continuation')
        row = dict(epoch=epoch, file=f'epoch_{epoch:03d}.pt', **state['sums'])
        state['history'].append(row)
        state.update(completed_epoch=epoch, cursor=0, order=None, sums={})
        target = checkpoints / row['file']
        persist(target)
        row['sha256'] = digest(target)
        _atomic_alias(checkpoints / 'last.pt', target)
        _manifest(output, checkpoints, state['history'])
        _progress(output, 'EPOCH_COMPLETE', **row)

    require(state['first_ddim_backward_update'] is not None and
            [row['epoch'] for row in state['history']] == list(range(FIRST_EPOCH, FINAL_EPOCH+1)),
            'the four complete epochs must include real DDIM backward updates')
    history = state['history']
    for row in history:
        require(digest(checkpoints / row['file']) == row['sha256'], 'retained epoch hash differs')
    final = checkpoints / history[-1]['file']
    for alias in ('final.pt', 'selected.pt'):
        _atomic_alias(checkpoints / alias, final)
    totals = {name: sum(row[name] for row in history) for name in (
        'eligible_windows', 'active_windows', 'train_ddim_calls', 'ddim_backward_windows',
        'train_seconds', 'skipped_batches')}
    return dict(status='COMPLETE', completed_epoch=FINAL_EPOCH, updates=state['update'],
                completed_batches=state['completed_batches'],
                eligible_windows=totals['eligible_windows'], active_windows=totals['active_windows'],
                extra_trigger_fraction=totals['active_windows']/max(totals['eligible_windows'], 1),
                train_ddim_calls=totals['train_ddim_calls'],
                ddim_backward_windows=totals['ddim_backward_windows'],
                train_seconds=totals['train_seconds'], skipped_batches=totals['skipped_batches'],
                peak_gpu_allocated_mib=state['peak_gpu_allocated_mib'],
                peak_gpu_reserved_mib=state['peak_gpu_reserved_mib'],
                first_update=state['first_update'],
                first_ddim_backward_update=state['first_ddim_backward_update'],
                dense_slot_replays=state['dense_slot_replays'],
                frozen_before_sha256=frozen, frozen_after_sha256=_frozen_states(assets, student),
                edge_after_sha256=state_hash(student.edge),
                S28_checkpoint=str(final.resolve()), S28_sha256=digest(final),
                retained_epochs=[row['epoch'] for row in history], epoch_rows=history,
                test_access=False)

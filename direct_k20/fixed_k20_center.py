"""Deterministic observed-only center trajectory and gated UNIV training."""

import math
from pathlib import Path
import time

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from bcep.contracts import canonical_hash, digest, require
from bcep.motion_frame import frame
from bcep_joint.common import read, restore_rng, rng_state, save, seed_all, write
from joint_flow_fourfold.common import immutable
from .interaction_train import _gpu_peaks, record_r12_metrics as record_final_metrics
from .model import encode_observed
from .train import _atomic_alias, _progress


RECIPE = dict(
    fold='univ', epochs=12, seed=3201, optimizer='AdamW', learning_rate=1e-3,
    weight_decay=1e-4, logical_windows=32, gradient_clip=5., precision='FP32',
    hidden_width=64, maximum_neighbors=4, observed_frames=8, prediction_frames=12,
    anchor_frames=[0, 4, 8, 12], dt=.4,
    objective='mean_windows(mean_people(FDE/train_minFDE + ADE/train_minADE))',
    only_eligible_epoch=12, retention='all completed epochs plus selected/final and recovery last',
)


def constant_velocity(observed, dt=.4):
    """Use only the final three measured displacements for the CV trajectory."""
    require(observed.shape[-2:] == (8, 2) and dt > 0, 'invalid CV observation geometry')
    recent = (observed[..., -1, :] - observed[..., -4, :]) / (3 * dt)
    horizon = torch.arange(1, 13, dtype=observed.dtype, device=observed.device) * dt
    return observed[..., -1:, :] + recent.unsqueeze(-2) * horizon[:, None]


def _features(observed, context, dt):
    """Return receiver-frame observations and observed four-neighbor inputs."""
    n = len(observed)
    require(observed.shape == (n, 8, 2) and context.shape == (n, 128) and n > 0,
            'center requires packed observed agents and context')
    _, rotation, _ = frame(observed, 12, dt, observed.new_ones(()))
    origin = observed[:, -1]
    motion = (observed - origin[:, None]) @ rotation
    personal = torch.cat((context, motion.flatten(1)), dim=-1)
    relative_position = origin[None] - origin[:, None]
    distance_sq = relative_position.square().sum(-1)
    distance_sq.fill_diagonal_(torch.inf)
    count = min(n - 1, RECIPE['maximum_neighbors'])
    neighbors = distance_sq.argsort(dim=-1, stable=True)[:, :count]
    people = torch.arange(n, device=observed.device)[:, None]
    velocity = (origin - observed[:, -4]) / (3 * dt)
    relative_velocity = velocity[neighbors] - velocity[:, None]
    local_position = relative_position[people, neighbors] @ rotation
    local_velocity = relative_velocity @ rotation
    relation = observed.new_zeros((n, RECIPE['maximum_neighbors'], 4))
    relation[:, :count] = torch.cat((local_position, local_velocity), dim=-1)
    neighbor_valid = torch.arange(RECIPE['maximum_neighbors'], device=observed.device) < count
    return (personal, relation, neighbor_valid.expand(n, -1), rotation,
            constant_velocity(observed, dt))


def _interpolation_weights():
    # The zero correction at step 0 is implicit in the first ramp.
    t = torch.arange(1, 13, dtype=torch.float32)
    anchors = torch.tensor([4., 8., 12.])
    return (1 - (t[:, None] - anchors).abs() / 4).clamp_min(0)


class CenterPredictor(nn.Module):
    """Encode observed neighbors and predict local corrections at 4/8/12."""

    def __init__(self):
        super().__init__()
        width = RECIPE['hidden_width']
        self.neighbor = nn.Sequential(nn.Linear(4, width), nn.SiLU(), nn.Linear(width, width))
        self.correction = nn.Sequential(nn.Linear(128 + 16 + width, width), nn.SiLU(), nn.Linear(width, 6))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        self.register_buffer('interpolation', _interpolation_weights(), persistent=False)

    def from_features(self, personal, relation, neighbor_valid, rotation, cv):
        encoded = self.neighbor(relation).masked_fill(~neighbor_valid[..., None], 0.)
        neighbor_mean = encoded.sum(dim=-2) / neighbor_valid.sum(-1, keepdim=True).clamp_min(1)
        anchors = self.correction(torch.cat((personal, neighbor_mean), dim=-1)).unflatten(-1, (3, 2))
        local = self.interpolation @ anchors
        return cv + local @ rotation.transpose(-1, -2)

    def forward(self, observed, context, *, dt=.4):
        return self.from_features(*_features(observed, context, dt))


@torch.no_grad()
def predict_center(assets, model, batch):
    """Decode the learned candidate once directly in world meters."""
    require(batch.valid.shape[0] == 1 and bool(batch.valid.all()) and not bool(batch.future.count_nonzero()),
            'center inference requires an observed-only packed window')
    device = next(model.parameters()).device
    require(device == torch.device(assets.device), 'frozen context and center devices differ')
    batch = batch.to(device)
    context, _ = encode_observed(assets.provider, batch)
    return model(batch.observed[0], context, dt=float(assets.split['dt']))


def _window_identity(assets, indices, frozen):
    return dict(
        fold='univ', purpose='train',
        member_sha256=canonical_hash([assets.split['members'][i] for i in indices]),
        record_sha256=canonical_hash([assets.split['records'][i]['sha256'] for i in indices]),
        geometry=dict(dt=assets.split['dt'], observed_frames=8, prediction_frames=12),
        frozen=frozen, source_sha256=digest(__file__),
    )


@torch.no_grad()
def _train_data(assets, output, indices, frozen):
    """Materialize the train-only context and targets in resumable CPU shards."""
    folder = output / 'frozen_train'
    folder.mkdir(exist_ok=True)
    identity = _window_identity(assets, indices, frozen)
    immutable(folder / 'IDENTITY.json', identity)
    previous = folder / 'COMPLETE.json'
    frozen_receipt = read(previous) if previous.exists() else None
    start = time.monotonic()
    examples, shards = [], []
    for start_ord in range(0, len(indices), 128):
        ids = indices[start_ord:start_ord + 128]
        payload = folder / f'{start_ord:05d}.pt'
        metadata = payload.with_suffix('.json')
        expected = dict(start_ordinal=start_ord, windows=len(ids),
                        identity_sha256=canonical_hash(identity), file=payload.name)
        if metadata.exists():
            row = read(metadata)
            require(all(row[key] == value for key, value in expected.items())
                    and row['sha256'] == digest(payload), 'frozen feature shard changed')
            items = torch.load(payload, weights_only=True, map_location='cpu')
        else:
            require(frozen_receipt is None, 'completed frozen feature cache has missing shards')
            items = []
            for i in ids:
                n = len(assets.split['members'][i]['agent_ids'])
                observed_batch = assets.batch(i, 'train', truth=False)
                require(observed_batch.valid.shape == (1, n) and bool(observed_batch.valid.all())
                        and not bool(observed_batch.future.count_nonzero()),
                        'frozen context attempted to use future labels')
                context, _ = encode_observed(assets.provider, observed_batch)
                observed = observed_batch.observed[0].detach().cpu()
                features = _features(observed, context.detach().cpu(), float(assets.split['dt']))
                target_batch = assets.batch(i, 'train', truth=True, device='cpu')
                require(torch.equal(target_batch.observed[0], observed)
                        and target_batch.agent_ids == observed_batch.agent_ids,
                        'train labels differ from frozen context membership')
                items.append((features, target_batch.future[0].detach().clone()))
            save(items, payload)
            row = expected | dict(sha256=digest(payload))
            immutable(metadata, row)
            _progress(output, 'CENTER_TRAIN_CONTEXT_CACHE', fold='univ',
                      windows=start_ord + len(ids), total=len(indices), seconds=time.monotonic()-start)
        require(len(items) == len(ids), 'frozen feature shard membership differs')
        for i, (features, future) in zip(ids, items):
            n = len(assets.split['members'][i]['agent_ids'])
            shapes = ((n, 144), (n, 4, 4), (n, 4), (n, 2, 2), (n, 12, 2))
            require(tuple(x.shape for x in features) == shapes and future.shape == (n, 12, 2) and
                    all(x.dtype == torch.float32 and bool(torch.isfinite(x).all())
                        for x in (*features[:2], *features[3:], future)) and
                    features[2].dtype == torch.bool, 'invalid cached train example')
        examples.extend(items)
        shards.append(row)
    require(assets.frozen_states() == frozen, 'frozen S0 assets changed during context extraction')
    core = dict(identity=identity, windows=len(examples), shards=shards,
                training_targets_only=True, context_future_reads=0, parameter_updates=0, ddim_calls=0)
    if frozen_receipt is not None:
        require(all(frozen_receipt[k] == v for k, v in core.items()), 'completed frozen cache changed')
    else:
        immutable(previous, core | dict(seconds=time.monotonic()-start))
    return examples, read(previous) | {'seal_sha256': digest(previous)}


def _pack(examples, device):
    features = tuple(pad_sequence(list(xs), batch_first=True).to(device)
                     for xs in zip(*(x[0] for x in examples)))
    future = pad_sequence([x[1] for x in examples], batch_first=True).to(device)
    counts = torch.as_tensor([len(x[1]) for x in examples], device=device)
    valid = torch.arange(future.shape[1], device=device) < counts[:, None]
    return features, future, valid, counts


def _window_loss(prediction, future, valid, counts, scales):
    error = torch.linalg.vector_norm(prediction - future, dim=-1)
    fde = (error[..., -1] * valid).sum(-1) / counts
    ade = (error.mean(-1) * valid).sum(-1) / counts
    return (fde/scales['minFDE'] + ade/scales['minADE']).mean(), fde.detach(), ade.detach()


def _preflight(model, examples, output, identity_sha, device, scales):
    path = output / 'PREFLIGHT.json'
    if path.exists():
        receipt = read(path)
        require(receipt['identity_sha256'] == identity_sha and receipt['passed'], 'preflight identity changed')
        return receipt
    rng = torch.Generator().manual_seed(RECIPE['seed'])
    select = torch.randperm(len(examples), generator=rng)[:RECIPE['logical_windows']]
    features, future, valid, counts = _pack([examples[i] for i in select.tolist()], device)
    prediction = model.from_features(*features)
    require(torch.equal(prediction, features[-1]), 'zero readout did not exactly reproduce CV')
    loss, _, _ = _window_loss(prediction, future, valid, counts, scales)
    derivatives = torch.autograd.grad(loss, tuple(model.correction[-1].parameters()))
    require(all(bool(d.isfinite().all()) for d in derivatives) and
            any(bool(d.count_nonzero()) for d in derivatives), 'center readout received no finite training gradient')
    result = dict(passed=True, identity_sha256=identity_sha, windows=len(select),
                  exact_zero_readout_CV=True, actual_optimizer_updates=0,
                  initial_readout_max_gradient=max(float(d.abs().max()) for d in derivatives))
    immutable(path, result)
    _progress(output, 'CENTER_PREFLIGHT_PASSED', fold='univ', **result)
    return result


def _inventory(output, history):
    write(output / 'EPOCH_METRICS.json', history)
    write(output / 'checkpoints' / 'TOP5.json', dict(
        selector='fixed completed epoch 12 only', entries=[], ranked_epoch_count=0,
        snapshots=[{k: row[k] for k in ('epoch','file','sha256')} for row in history],
    ))


def _resume(checkpoints, identity):
    last = checkpoints / 'last.pt'
    epochs = sorted(checkpoints.glob('epoch_*.pt'))
    if not last.exists() and not epochs:
        return None
    state = torch.load(last if last.exists() else epochs[-1], weights_only=False, map_location='cpu')
    if epochs and int(epochs[-1].stem.split('_')[1]) > state['completed_epoch']:
        state = torch.load(epochs[-1], weights_only=False, map_location='cpu')
    require(state['identity'] == identity and state['recipe'] == RECIPE and
            len(epochs) == len(state['history']) == state['completed_epoch'], 'invalid recovery snapshot')
    for row in state['history']:
        sha = digest(checkpoints / row['file'])
        require(row.get('sha256', sha) == sha, 'completed epoch SHA changed')
        row['sha256'] = sha
    return state


def train_center(assets, output, identity, scales, *, device='cuda', should_yield=None):
    """Fit precisely one UNIV final candidate after the immutable CV failure."""
    require(assets.fold == 'univ' and identity['cv_passed'] is False
            and len(identity['cv_receipt_sha256']) == 64 and 0 <= identity['j_drop'] < 20,
            'center launch requires the fixed failed UNIV CV receipt')
    device = torch.device(device)
    require(device == torch.device(assets.device) and float(assets.split['dt']) == RECIPE['dt'],
            'center training assets differ in device or geometry')
    started = time.time()
    output = Path(output)
    checkpoints = output / 'checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=True)
    indices = tuple(assets.split['train_indices'])
    require(len(indices) == len(set(indices)) > 0, 'invalid train members')
    scales = {k: float(scales[k]) for k in ('minFDE','minADE')}
    require(all(math.isfinite(v) and v > 0 for v in scales.values()), 'invalid original train scales')
    frozen = assets.frozen_states()
    examples, features = _train_data(assets, output, indices, frozen)
    identity = dict(fold='univ', protocol=identity, scales_m=scales,
                    frozen_before_sha256=frozen, train_cache_sha256=features['seal_sha256'], recipe=RECIPE)
    immutable(output / 'TRAIN_IDENTITY.json', identity)
    result_path = output / 'TRAINING_RESULT.json'
    if result_path.exists():
        result = read(result_path)
        require(result['identity'] == identity and result['completed_epochs'] == RECIPE['epochs'] and
                digest(result['final_checkpoint']) == result['final_sha256'], 'final center checkpoint changed')
        return result

    seed_all(RECIPE['seed'])
    model = CenterPredictor().to(device=device, dtype=torch.float32)
    loader = torch.Generator().manual_seed(RECIPE['seed'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=RECIPE['learning_rate'], weight_decay=RECIPE['weight_decay'])
    state = _resume(checkpoints, identity)
    preflight = _preflight(model, examples, output, canonical_hash(identity), device, scales)
    if state is None:
        state = dict(identity=identity, recipe=RECIPE, completed_epoch=0, update=0, cursor=0, order=None,
                     history=[], sums={}, scheduler=None, started_unix=started, first_update=None,
                     fixed_candidate_epoch=RECIPE['epochs'], **_gpu_peaks(device))
    else:
        model.load_state_dict(state['center'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng'], loader)
        _inventory(output, state['history'])
        if state['first_update'] is not None:
            immutable(output / 'FIRST_UPDATE.json', state['first_update'])

    def persist(path):
        state.update(center=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state(loader))
        save(state, path)

    if not (checkpoints / 'last.pt').exists():
        persist(checkpoints / 'last.pt')
    while state['completed_epoch'] < RECIPE['epochs']:
        epoch = state['completed_epoch'] + 1
        if state['order'] is None:
            state.update(order=torch.randperm(len(indices), generator=loader).tolist(),
                         sums=dict(windows=0, fde=0., ade=0., objective=0., seconds=0., max_gradient=0.))
        while state['cursor'] < len(indices):
            if should_yield is not None and should_yield():
                return None
            tick = time.monotonic()
            chunk = state['order'][state['cursor']:state['cursor'] + RECIPE['logical_windows']]
            x, future, valid, counts = _pack([examples[i] for i in chunk], device)
            optimizer.zero_grad(set_to_none=True)
            loss, fde, ade = _window_loss(model.from_features(*x), future, valid, counts, scales)
            require(bool(loss.isfinite()), 'nonfinite center objective')
            loss.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE['gradient_clip'],
                                                       error_if_nonfinite=True))
            optimizer.step()
            sums = state['sums']
            sums['windows'] += len(chunk)
            sums['fde'] += float(fde.double().sum())
            sums['ade'] += float(ade.double().sum())
            sums['objective'] += float(loss.detach()) * len(chunk)
            sums['seconds'] += time.monotonic() - tick
            sums['max_gradient'] = max(sums['max_gradient'], grad)
            state['cursor'] += len(chunk)
            state['update'] += 1
            for name, value in _gpu_peaks(device).items():
                state[name] = max(state[name], value)
            if state['update'] == 1:
                state['first_update'] = dict(epoch=epoch, update=1, windows=state['cursor'], gradient_norm=grad,
                                             objective=float(loss.detach()), **_gpu_peaks(device))
            persist(checkpoints / 'last.pt')
            if state['update'] == 1:
                immutable(output / 'FIRST_UPDATE.json', state['first_update'])
                _progress(output, 'CENTER_FIRST_UPDATE', fold='univ', **state['first_update'])

        totals = state['sums']
        n = totals['windows']
        require(n == len(indices), 'center must train on every original train window once per epoch')
        row = dict(epoch=epoch, update=state['update'], windows=n, file=f'epoch_{epoch:03d}.pt',
                   FDE_m=totals['fde']/n, ADE_m=totals['ade']/n, objective=totals['objective']/n,
                   train_seconds=totals['seconds'], gradient_norm_max=totals['max_gradient'])
        state['history'].append(row)
        state.update(completed_epoch=epoch, order=None, cursor=0, sums={})
        path = checkpoints / row['file']
        persist(path)
        row['sha256'] = digest(path)
        _atomic_alias(checkpoints / 'last.pt', path)
        _inventory(output, state['history'])
        _progress(output, 'CENTER_EPOCH_COMPLETE', fold='univ', **row)

    history = state['history']
    require(len(history) == RECIPE['epochs'] and all(
        digest(checkpoints / row['file']) == row['sha256'] for row in history), 'center epoch inventory changed')
    final = checkpoints / history[-1]['file']
    for alias in ('final.pt', 'selected.pt'):
        _atomic_alias(checkpoints / alias, final)
    after = assets.frozen_states()
    require(frozen == after, 'frozen downstream assets changed')
    result = dict(status='COMPLETE', fold='univ', identity=identity, recipe=RECIPE, completed_epochs=12,
                  parameter_updates=state['update'], final_checkpoint=str(final.resolve()),
                  final_sha256=history[-1]['sha256'], frozen_before_sha256=frozen, frozen_after_sha256=after,
                  feature_cache={k:v for k,v in features.items() if k != 'shards'},
                  epoch_rows=history, retained_epochs=[row['epoch'] for row in history],
                  scope='original train only', S0_updates=0, A_updates=0, B_updates=0, ddim_calls=0,
                  preflight=preflight, first_update=state['first_update'],
                  train_seconds=math.fsum(row['train_seconds'] for row in history),
                  train_wall_seconds=time.time()-state['started_unix'],
                  peak_gpu_allocated_mib=state['peak_gpu_allocated_mib'],
                  peak_gpu_reserved_mib=state['peak_gpu_reserved_mib'])
    immutable(result_path, result)
    return result


def load_center(path, sha, device):
    """Load only the prescribed epoch-12 checkpoint for frozen evaluation."""
    require(digest(path) == sha, 'center checkpoint SHA changed')
    state = torch.load(path, weights_only=False, map_location='cpu')
    require(state['completed_epoch'] == RECIPE['epochs'] and state['recipe'] == RECIPE,
            'only the locked complete epoch 12 is evaluated')
    model = CenterPredictor()
    model.load_state_dict(state['center'], strict=True)
    return model.to(device).eval().requires_grad_(False)

"""One unchanged continuous-dependence fit for each explicitly frozen fold."""
import csv
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from bcep.contracts import canonical_hash, digest
from bcep_joint.common import read, write, seed_all
from bcep_v2.runtime import configure_cuda
from bcep_v2.state import run_stage
from ebjd.metrics import MetricAccumulator, hard_metric_statistics
from joint_flow_dependency.inference import generate, endpoint_minfde
from joint_flow_dependency.model import EdgeDependence, delta_nll, sample_latents
from joint_flow_dependency.sampling import bridge_noise
from .context_cache import build_cache
from .inputs import Inputs, SOURCE, state_hash

RECIPE = dict(seed=3201, lr=1e-4, weight_decay=1e-4, clip=5., logical_scenes=16,
              max_epochs=100, minimum=30, patience=15, scheduler_horizon=100)


def source_hashes():
    return {str(path.relative_to(SOURCE)): digest(path)
        for package in ('joint_flow_fourfold', 'joint_flow_dependency', 'bcep_v2', 'bcep_joint',
                        'bcep_formal', 'bcep', 'ebjd', 'selector_control', 'bcep_confirmation')
        for path in sorted((SOURCE/package).rglob('*.py'))}


def _immutable(path, value):
    if path.exists():
        assert read(path) == value, 'immutable training contract changed: '+path.name
    else:
        write(path, value)


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _assert_frozen(model, provider, expected):
    for network in (model, provider):
        assert not any(module.training for module in network.modules())
        assert not any(parameter.requires_grad or parameter.grad is not None for parameter in network.parameters())
    current = dict(backbone=state_hash(model), provider=state_hash(provider))
    assert current == expected, 'frozen upstream state changed during dependency fit'
    return current


def _preflight(inputs, model, provider, dependence, train_cache, device):
    """At most two train windows; no optimizer, parameter change or RNG change."""
    receipt_path = inputs.output/'PREFLIGHT.json'
    if receipt_path.exists():
        receipt = read(receipt_path)
        assert receipt['passed'] and receipt['identity'] == inputs.identity
        assert receipt['dependency_initial_state_sha256'] == state_hash(dependence)
        return receipt
    mapping = {record['index']: record for record in train_cache}
    indices = inputs.split['train_indices']
    count = lambda index: len(inputs.split['members'][index]['agent_ids'])
    # One smallest case (including a singleton when present), one largest case.
    selected = list(dict.fromkeys((min(indices, key=count), max(indices, key=count))))
    assert len(selected) <= 2 and max(map(count, selected)) > 1
    rng = dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(),
               cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
    modes = [(module, module.training) for module in dependence.modules()]
    gradients = [None if p.grad is None else p.grad.detach().clone() for p in dependence.parameters()]
    before = state_hash(dependence)
    checks = []
    try:
        dependence.eval()
        for index in selected:
            batch = inputs.data.batch([index], 'train', truth=False).to(device)
            with torch.no_grad():
                prediction, endpoints, detail = generate(model, provider, dependence, batch, seed=2035)
            record = mapping[index]
            assert torch.equal(record['inputs']['observed'], batch.observed[0].cpu())
            dependence.zero_grad(set_to_none=True)
            with torch.enable_grad():
                pack = dependence(**{key: value.to(device) for key, value in record['inputs'].items()})
                covariance = pack['R']
                torch.testing.assert_close(covariance.detach(), endpoints['R'], rtol=0, atol=0)
                n = count(index)
                diag = torch.stack([covariance[2*i:2*i+2, 2*i:2*i+2] for i in range(n)])
                eye = torch.eye(2, device=device).expand_as(diag)
                torch.testing.assert_close(diag.detach(), eye, rtol=0, atol=2e-6)
                chol = torch.linalg.cholesky(covariance)
                loss = delta_nll(covariance, record['latent_truth'].to(device))
                assert torch.isfinite(loss) and loss.requires_grad
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in dependence.parameters())
                first = float(dependence.net[0].weight.grad.norm())
                last = float(dependence.net[-1].weight.grad.norm())
                if n == 1:
                    assert float(loss.detach()) == first == last == 0.
                else:
                    assert first > 0 and last > 0
            with torch.no_grad():
                z = sample_latents(pack, endpoints['epsilon'], endpoints['xi'])
                u = endpoints['epsilon'].clone()
                for edge, (i, j) in enumerate(pack['edges'].tolist()):
                    u[:, i] += endpoints['xi'][:, edge] @ pack['B'][edge, 0].T
                    u[:, j] += endpoints['xi'][:, edge] @ pack['B'][edge, 1].T
                direct = torch.linalg.solve_triangular(pack['L'][None], u[..., None], upper=False).squeeze(-1)
                torch.testing.assert_close(z, direct, rtol=1e-5, atol=1e-6)
                raw_ids = endpoints['raw_indices']
                assert raw_ids.numel() == len(torch.unique(raw_ids)) == 20
                assert (raw_ids >= 0).all() and (raw_ids < 256).all()
                assert torch.equal(endpoints['selected'], endpoints['raw'][raw_ids])
                keyed = bridge_noise(batch.scene_ids[0], batch.agent_ids[0], raw_ids.cpu().tolist(), seed=2035)
                assert torch.equal(endpoints['noise'], keyed.to(device))
                assert detail['ddim_steps'] == 20 and detail['ddim_calls'] == detail['correction_calls'] == 1
                checks.append(dict(window=f'train_{indices.index(index)+1:04d}', agents=n,
                    R_diagonal_max_abs=float((diag.detach()-eye).abs().max()),
                    R_cholesky_diagonal_min=float(chol.detach().diagonal().min()),
                    first_layer_gradient_l2=first, last_layer_gradient_l2=last,
                    raw_joint_ids_unique=True, endpoint_and_noise_raw_ids_aligned=True,
                    sampler_vs_direct_max_abs=float((z-direct).abs().max()),
                    endpoint_path_max_abs_m=float((prediction[..., -1, :]-endpoints['corrected']).abs().max())))
    finally:
        for parameter, gradient in zip(dependence.parameters(), gradients):
            parameter.grad = gradient
        for module, training in modes:
            module.training = training
        random.setstate(rng['python'])
        np.random.set_state(rng['numpy'])
        torch.set_rng_state(rng['cpu'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(rng['cuda'])
    assert state_hash(dependence) == before
    receipt = dict(passed=True, identity=inputs.identity, windows=checks, optimizer_updates=0,
        dependency_initial_state_sha256=before, dependency_weights_unchanged=True,
        all_rng_restored=True, gradients_restored=True, test_access=False)
    write(receipt_path, receipt)
    return receipt


@torch.no_grad()
def evaluate_epoch(inputs, model, provider, dependence, cache_dev, device, epoch):
    """Actual full K20/20-DDIM dev predictions, with the original accumulator."""
    model.eval(); provider.eval(); dependence.eval()
    mapping = {record['index']: record for record in cache_dev}
    indices = inputs.split['selection_dev_indices']
    assert len(mapping) == len(indices) and set(mapping) == set(indices)
    accumulators = {'all': MetricAccumulator(), 'N_ge_9': MetricAccumulator()}
    rows = []
    started = time.monotonic()
    for position, index in enumerate(indices):
        batch = inputs.data.batch([index], 'selection_dev', truth=False).to(device)
        path, endpoints, detail = generate(model, provider, dependence, batch, seed=2035)
        assert detail['ddim_steps'] == 20 and detail['ddim_calls'] == detail['correction_calls'] == 1
        truth = inputs.data.batch([index], 'selection_dev', truth=True).future.to(device)
        statistics = hard_metric_statistics(path, truth, batch.valid)
        accumulators['all'].update_statistics(statistics)
        n = int(batch.valid.sum())
        if n >= 9:
            accumulators['N_ge_9'].update_statistics(statistics)
        record = mapping[index]
        assert torch.equal(record['inputs']['observed'], batch.observed[0].cpu())
        likelihood = float(delta_nll(endpoints['R'], record['latent_truth'].to(device)))
        assert math.isfinite(likelihood)
        row = dict(window=f'dev_{position+1:04d}', agents=n,
            JADE=float(statistics.scene_jade[0]), JFDE=float(statistics.scene_jfde[0]),
            agent_minADE_sum=float(statistics.agent_minade[statistics.agent_valid].double().sum()),
            agent_minFDE_sum=float(statistics.agent_minfde[statistics.agent_valid].double().sum()),
            scene_minADE=float(statistics.scene_marginal_ade[0]), scene_minFDE=float(statistics.scene_marginal_fde[0]),
            delta_nll_nat_per_person=likelihood)
        for stage, key in (('raw256', 'raw'), ('selected20', 'selected'), ('corrected20', 'corrected')):
            goals = endpoints[key][0] if key == 'corrected' else endpoints[key]
            error = endpoint_minfde(goals, truth[0, :, -1]).double()
            row[stage+'_minFDE_agent_sum'] = float(error.sum())
            row[stage+'_minFDE_scene_mean'] = float(error.mean())
        rows.append(row)
        if (position+1) % 25 == 0 or position+1 == len(indices):
            progress = dict(phase='EPOCH_VALIDATION_B', fold=inputs.fold, epoch=epoch,
                windows=position+1, total=len(indices), seconds=time.monotonic()-started)
            write(inputs.output/'PROGRESS.json', progress)
            print(progress, flush=True)
    result = {group: accumulator.as_dict() if accumulator.scene_count else None
              for group, accumulator in accumulators.items()}
    result.update(epoch=epoch, windows=len(indices), dense_windows=sum(row['agents'] >= 9 for row in rows),
        dev_delta_nll_nat_per_person=math.fsum(row['delta_nll_nat_per_person'] for row in rows)/len(rows),
        actual_ddim_calls=len(indices), correction_calls=len(indices), ddim_steps=20,
        seed=2035, seconds=time.monotonic()-started, test_access=False)
    result['coverage'] = {}
    result['delta_nll_nat_per_person'] = {}
    for group in accumulators:
        subset = [row for row in rows if group == 'all' or row['agents'] >= 9]
        if not subset:
            result['coverage'][group] = None
            result['delta_nll_nat_per_person'][group] = None
            continue
        count = sum(row['agents'] for row in subset)
        result['coverage'][group] = {stage: dict(
            agent_micro_minFDE=math.fsum(row[stage+'_minFDE_agent_sum'] for row in subset)/count,
            scene_macro_minFDE=math.fsum(row[stage+'_minFDE_scene_mean'] for row in subset)/len(subset),
            agent_count=count, scene_count=len(subset)) for stage in ('raw256', 'selected20', 'corrected20')}
        result['delta_nll_nat_per_person'][group] = math.fsum(row['delta_nll_nat_per_person'] for row in subset)/len(subset)
    folder = inputs.output/'dev'
    _csv(folder/f'epoch_{epoch:03d}_windows.csv', rows)
    write(folder/f'epoch_{epoch:03d}_groups.json', result)
    return result


def _retention_check(folder, state):
    expected = sorted(state['history'], key=lambda row: (row['validation'], row['epoch']))[:5]
    actual = read(folder/'TOP5.json')
    assert len(actual) == min(5, state['completed_epoch'])
    assert len({row['epoch'] for row in actual}) == len(actual)
    for wanted, row in zip(expected, actual):
        assert {key: row[key] for key in wanted} == wanted
        assert digest(folder/row['file']) == row['sha256']
    required = ('model', 'optimizer', 'scheduler', 'rng', 'completed_epoch', 'update',
                'cursor', 'order', 'stale', 'history', 'identity', 'recipe')
    for row in state['history']:
        snapshot = torch.load(folder/row['file'], map_location='cpu', weights_only=False)
        assert all(key in snapshot for key in required)
        assert snapshot['completed_epoch'] == row['epoch'] and snapshot['cursor'] == 0
    assert (folder/'last.pt').exists() and digest(folder/'best.pt') == actual[0]['sha256']
    return dict(all_completed_epoch_snapshots=state['completed_epoch'], top5_distinct_epochs=[row['epoch'] for row in actual],
                complete_resume_state_verified=True, aliases_do_not_count_as_distinct_epochs=True)


def run_training(frozen_manifest, output, device):
    """Run/resume exactly one fold and return its dev-selected complete checkpoint."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    torch.set_num_threads(1)
    if device.type == 'cuda':
        configure_cuda()
    inputs = Inputs(frozen_manifest, output, device)
    sources = source_hashes()
    config = dict(schema='continuous-joint-flow-frozen-fold-training-v1', fold=inputs.fold,
        identity=inputs.identity, recipe=RECIPE, source_sha256=sources,
        parameter_count=16964, model='unchanged shared Linear260->64, SiLU, Linear64->4 edge dependence',
        train_windows=inputs.counts['train'], dev_windows=inputs.counts['selection_dev'],
        train_seed=3201, dev_seed=2035, FP32=True, raw_scenarios=256, K=20, DDIM=20,
        optimizer='original AdamW; matrix weights decay1e-4, bias/norm vectors decay0',
        scheduler='original five-epoch linear warmup followed by cosine to 0.1 over horizon100',
        logical_batch='16 windows; per-window deltaNLL divided by current group size before backward',
        selector='minimum actual full-dev B JADE; exact ties earlier completed epoch',
        retention='ALL completed epoch snapshots + best/last + Top5 distinct complete epochs; no deletion',
        future_usage='train synchronous endpoint likelihood target; selection_dev offline scoring only',
        test_access=False, upstream_optimization=False)
    _immutable(output/'RUN_CONFIG.json', config)
    model, provider = inputs.load_frozen(device)
    seed_all(RECIPE['seed'])
    dependence = EdgeDependence().to(device=device, dtype=torch.float32)
    assert sum(parameter.numel() for parameter in dependence.parameters()) == 16964
    assert all(parameter.requires_grad for parameter in dependence.parameters())
    before = dict(backbone=state_hash(model), provider=state_hash(provider))
    frozen_path = output/'FROZEN_MODELS.json'
    if frozen_path.exists():
        assert read(frozen_path)['before'] == before
    else:
        write(frozen_path, dict(before=before, provider=inputs.provider_identity, upstream_identity=inputs.identity))
    _assert_frozen(model, provider, before)
    cache = build_cache(inputs, provider, device)
    _preflight(inputs, model, provider, dependence, cache['train'], device)
    _assert_frozen(model, provider, before)
    identity = dict(inputs.identity, config=canonical_hash(config), stage='joint',
        experiment='continuous_joint_flow_remaining_folds', selector=config['selector'],
        latent_caches={purpose: digest(output/'latent_cache'/f'{purpose}_MANIFEST.json') for purpose in cache})
    folder = output/'dependence'
    last = folder/'last.pt'
    train_records = {record['index']: record for record in cache['train']}
    completed = [torch.load(last, map_location='cpu', weights_only=False)['completed_epoch'] if last.exists() else 0]

    def backward(group, order_rng, view_rng):
        del order_rng, view_rng
        total = 0.
        for index in group:
            record = train_records[index]
            covariance = dependence(**{key: value.to(device) for key, value in record['inputs'].items()})['R']
            loss = delta_nll(covariance, record['latent_truth'].to(device))
            assert torch.isfinite(loss)
            (loss/len(group)).backward()
            total += float(loss.detach())
        return dict(delta_nll_nat_per_person=total/len(group),
                    delta_nll_window_sum=total, equally_weighted_windows=len(group))

    def validate():
        epoch = completed[0]+1
        metrics = evaluate_epoch(inputs, model, provider, dependence, cache['selection_dev'], device, epoch)
        _assert_frozen(model, provider, before)
        validate.metrics = metrics
        completed[0] = epoch
        return metrics['all']['JADE']

    updates = math.ceil(len(train_records)/RECIPE['logical_scenes'])*RECIPE['max_epochs']
    state = run_stage(folder, identity, RECIPE, dependence, inputs.split['train_indices'], backward, validate,
                      updates, resume=last.exists(), record_progress=True)
    if state['status'] != 'COMPLETE':
        raise RuntimeError('Continue this same fold from its last checkpoint: '+state['status'])
    rank = min(state['history'], key=lambda row: (row['validation'], row['epoch']))
    selected = folder/rank['file']
    saved = torch.load(selected, map_location='cpu', weights_only=False)
    assert saved['identity'] == identity and saved['completed_epoch'] == rank['epoch']
    dependence.load_state_dict(saved['model'], strict=True)
    dependence.eval()
    after = _assert_frozen(model, provider, before)
    assert source_hashes() == sources, 'source changed during fit'
    retention = _retention_check(folder, state)
    result = dict(fold=inputs.fold, rank1_checkpoint=str(selected.resolve()),
        rank1_epoch=rank['epoch'], rank1_sha256=digest(selected), rank1_metrics=rank['metrics'],
        completed_epochs=state['completed_epoch'], updates=state['update'], stop_reason=state['stop_reason'],
        recipe=RECIPE, optimizer_parameter_names=state['trainable'], identity=identity,
        frozen_unchanged=True, retention=retention, test_access=False)
    write(output/'TRAINING_RESULT.json', result)
    write(frozen_path, dict(before=before, after=after, exact_unchanged=True,
        provider=inputs.provider_identity, upstream_identity=inputs.identity))
    return result

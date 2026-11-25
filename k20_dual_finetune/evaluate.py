"""Fixed epoch-eight, seed-2035 exploratory evaluation of the former heldout split."""

import csv
import math
from pathlib import Path
import time

import numpy as np
import torch

from bcep.contracts import canonical_hash, digest, require
from bcep_formal.data import FormalData
from bcep_joint.common import read, write
from ebjd.metrics import hard_metric_statistics
from hotel_endpoint_diagnostic.math import endpoint_statistics
from joint_flow_dependency.inference import decode, encode_observed, sample_endpoints
from joint_flow_dependency.sampling import bridge_noise
from joint_flow_frozen_test.inputs import ObservedOnlyHeldoutData
from joint_flow_frozen_test.run import forbid_heldout_truth
from joint_flow_fourfold.common import immutable
from joint_flow_fourfold.inputs import state_hash
from joint_flow_fourfold.scoring import csvwrite


PURPOSE = 'heldout_confirmation'
SCOPE = '原测试集上的单种子探索性开发评估'
SEED = 2035
METRICS = ('JADE', 'JFDE', 'minADE', 'minFDE', 'L', 'G')
NATIVE_METRICS = METRICS[:4]
GROUPS = {'hotel': ('all', 'single', 'multi'), 'univ': ('all', 'N_ge_9')}
COUNTS = {'hotel': (445, 1197), 'univ': (947, 24334)}
SOURCE = Path(__file__).resolve().parents[1]


def _csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def _history(inputs, historical, members):
    """Verify the existing paired B values without replaying its bridge."""
    fold = inputs.fold
    seal_path = historical/'PREDICTIONS_FROZEN.json'
    seal = read(seal_path)
    receipt = read(historical/'SCORE_RECEIPT.json')
    require(receipt['status'] == 'COMPLETE' and receipt['all_predictions_verified_before_any_truth'],
            'historical B score did not finish the frozen protocol')
    require(receipt['prediction_manifest_sha256'] == digest(seal_path), 'historical B seal changed')
    require(seal['fold'] == fold and seal['identity']['upstream'] == inputs.identity,
            'historical B upstream identity differs from the training initialization')
    require(seal['identity']['heldout_members'] == members and SEED in seal['seeds'],
            'historical B test membership or seed differs')
    initial = read(historical.parent/'B_FROZEN.json')
    require(seal['identity']['B_sha256'] == initial['rank1_sha256'] and
            seal['identity']['B_epoch'] == initial['rank1_epoch'], 'historical B selection changed')
    archive = SOURCE/'docs/research/continuous_joint_flow'/historical.parents[1].name/fold
    hashes = {}
    for name in ('PER_SEED.csv', 'PER_WINDOW.csv'):
        hashes[name] = digest(historical/name)
        require(hashes[name] == receipt['output_sha256'][name] == digest(archive/name),
                f'historical B {name} differs from the archived full-precision result')
    seeds = {row['group']: row for row in _csv(historical/'PER_SEED.csv') if row['seed'] == str(SEED)}
    windows = [row for row in _csv(historical/'PER_WINDOW.csv') if row['seed'] == str(SEED)]
    require(all(row['fold'] == fold for row in windows) and len(windows) == COUNTS[fold][0],
            'historical window count or fold differs')
    reference = {name: {key: float(row['B_'+key]) for key in NATIVE_METRICS}
                 for name, row in seeds.items() if int(row['scene_count'])}
    identity = dict(B_initial_sha256=initial['rank1_sha256'], B_initial_epoch=initial['rank1_epoch'],
                    history_sha256=hashes, historical_predictions_seal_sha256=digest(seal_path))
    return reference, windows, identity


def _read_array(path, n, phase):
    with np.load(path, allow_pickle=False) as archive:
        if phase == 'endpoint':
            require(set(archive.files) == {'endpoints', 'raw_ids'}, 'endpoint shard contains unexpected arrays')
            goals, ids = archive['endpoints'], archive['raw_ids']
            require(goals.dtype == np.float32 and goals.shape == (20, n, 2) and np.isfinite(goals).all(),
                    'invalid endpoint values')
            require(ids.dtype == np.int64 and ids.shape == (20,) and len(np.unique(ids)) == 20
                    and np.all((ids >= 0) & (ids < 256)), 'invalid raw sample IDs')
            return goals, ids
        require(set(archive.files) == {'trajectories'}, 'trajectory shard contains unexpected arrays')
        paths = archive['trajectories']
        require(paths.dtype == np.float32 and paths.shape == (20, n, 12, 2) and np.isfinite(paths).all(),
                'invalid trajectory values')
        return paths


def _save_shard(path, arrays):
    if path.exists():
        # A crash may have saved the shard but not yet written its receipt.
        with np.load(path, allow_pickle=False) as prior:
            require(set(prior.files) == set(arrays) and all(np.array_equal(prior[k], v) for k, v in arrays.items()),
                    'resumed prediction differs')
        return
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _decoder_inputs(directory, seal_sha256):
    manifest = directory/'PREDICTIONS_FROZEN.json'
    require(digest(manifest) == seal_sha256, 'endpoint seal changed before bridge evaluation')
    return {row['window']: row for row in read(manifest)['rows']}


def _decoder_input(directory, row):
    path = directory/row['file']
    require(digest(path) == row['sha256'], 'endpoint shard changed before bridge evaluation')
    return _read_array(path, row['n'], 'endpoint')


@torch.no_grad()
def _export(inputs, model, provider, dependence, directory, identity, *, endpoint_directory=None, device):
    phase = 'endpoint' if endpoint_directory is None else 'trajectory'
    manifest_path = directory/'PREDICTIONS_FROZEN.json'
    if manifest_path.exists():
        return read(manifest_path)
    split = inputs.split
    indices = split[PURPOSE+'_indices']
    observed = ObservedOnlyHeldoutData(split)
    directory.mkdir(parents=True, exist_ok=True)
    sealed_endpoints = None if endpoint_directory is None else _decoder_inputs(
        endpoint_directory, identity['endpoint_seal_sha256'])
    rows = []
    identity_sha = canonical_hash(identity)
    started = time.monotonic()
    with forbid_heldout_truth():
        for ordinal, index in enumerate(indices, 1):
            window = f'test_{ordinal:04d}'
            n = len(split['members'][index]['agent_ids'])
            path = directory/f'{window}.npz'
            side = path.with_suffix('.json')
            expected = dict(window=window, n=n, file=path.name, identity_sha256=identity_sha)
            if side.exists():
                row = read(side)
                require(row == dict(expected, sha256=digest(path)), 'prediction receipt or shard changed')
                _read_array(path, n, phase)
            else:
                batch = observed.batch([index], PURPOSE, truth=False).to(device)
                require(bool(batch.valid.all()) and int(batch.valid.sum()) == n and not batch.future.count_nonzero(),
                        'observed-only window is inconsistent')
                if phase == 'endpoint':
                    context, features = encode_observed(provider, batch)
                    endpoints = sample_endpoints(provider, dependence, batch, context, features, seed=SEED)
                    arrays = dict(endpoints=endpoints['corrected'][0].cpu().numpy(),
                                  raw_ids=endpoints['raw_indices'].cpu().numpy())
                else:
                    goals, ids = _decoder_input(endpoint_directory, sealed_endpoints[window])
                    goals = torch.from_numpy(goals)[None].to(device)
                    noise = bridge_noise(batch.scene_ids[0], batch.agent_ids[0], ids, seed=SEED).to(device)
                    context = model.encode_context(batch.observed, batch.semantic_maps, batch.valid)
                    trajectories = decode(model, context, goals, noise)
                    torch.testing.assert_close(trajectories[..., -1, :], goals, rtol=1e-6, atol=1e-6)
                    arrays = dict(trajectories=trajectories[0].cpu().numpy())
                _save_shard(path, arrays)
                _read_array(path, n, phase)
                row = dict(expected, sha256=digest(path))
                write(side, row)
            rows.append(row)
            if ordinal % 50 == 0 or ordinal == len(indices):
                progress = dict(fold=inputs.fold, phase=phase, windows=ordinal, total=len(indices),
                                wall_seconds=time.monotonic()-started, unix=time.time())
                write(directory/'PROGRESS.json', progress)
                print(progress, flush=True)
    seal = dict(identity=identity, phase=phase, rows=rows, fold=inputs.fold, seed=SEED,
                seal_unix=time.time(), wall_seconds=time.monotonic()-started,
                exporter_truth_reads=0, ddim_calls=0 if phase == 'endpoint' else len(rows))
    immutable(manifest_path, seal)
    return seal


def _verify_seal(split, seal, directory, identity, phase):
    indices = split[PURPOSE+'_indices']
    require(seal['phase'] == phase and seal['identity'] == identity and seal['seed'] == SEED,
            'prediction seal identity differs')
    require(len(seal['rows']) == len(indices), 'prediction seal incomplete')
    identity_sha = canonical_hash(identity)
    for ordinal, (index, row) in enumerate(zip(indices, seal['rows']), 1):
        window = f'test_{ordinal:04d}'
        path = directory/f'{window}.npz'
        n = len(split['members'][index]['agent_ids'])
        require(row == dict(window=window, n=n, file=path.name, identity_sha256=identity_sha,
                            sha256=digest(path)), 'invalid sealed prediction identity')
        _read_array(path, n, phase)
        require(digest(split['records'][index]['path']) == split['records'][index]['sha256'],
                'scoring record changed')
    receipt = dict(files=len(indices), manifest_sha256=digest(directory/'PREDICTIONS_FROZEN.json'),
                   completed_unix=time.time(), seal_unix=seal['seal_unix'])
    require(receipt['completed_unix'] >= seal['seal_unix'], 'verification predates prediction seal')
    return receipt


def _window_row(fold, ordinal, n, historical, endpoint, native=None):
    require(historical['window'] == f'test_{ordinal:04d}' and int(historical['n']) == n,
            'historical window population differs')
    row = dict(fold=fold, window=historical['window'], seed=SEED, n=n)
    row.update({f'B_{key}': float(historical[f'B_{key}']) for key in NATIVE_METRICS})
    row['B_agent_minFDE_sum'] = float(historical['B_agent_minFDE_sum'])
    row['B_agent_minADE_sum'] = float(historical['B_agent_minADE_sum'])
    row['B_L'] = row['B_minFDE']
    row['B_G'] = row['B_JFDE'] - row['B_L']
    row.update(new_JADE=None, new_minADE=None, new_agent_minADE_sum=None,
               new_JFDE=endpoint['J'], new_minFDE=endpoint['L'], new_L=endpoint['L'], new_G=endpoint['G'],
               new_agent_minFDE_sum=endpoint['agent_min_sum'])
    if native is not None:
        ade_sum = float(native.agent_minade[native.agent_valid].double().sum())
        fde_sum = float(native.agent_minfde[native.agent_valid].double().sum())
        jfde = float(native.scene_jfde[0])
        require(math.isclose(fde_sum/n, endpoint['L'], rel_tol=1e-6, abs_tol=1e-6) and
                math.isclose(jfde, endpoint['J'], rel_tol=1e-6, abs_tol=1e-6),
                'decoded last frame does not reproduce the endpoint score')
        row.update(new_JADE=float(native.scene_jade[0]), new_minADE=ade_sum/n,
                   new_agent_minADE_sum=ade_sum, new_JFDE=jfde, new_minFDE=fde_sum/n,
                   new_agent_minFDE_sum=fde_sum, new_L=fde_sum/n, new_G=jfde-fde_sum/n)
    row.update({f'delta_{key}': None if row[f'new_{key}'] is None else row[f'new_{key}']-row[f'B_{key}']
                for key in METRICS})
    return row


@torch.no_grad()
def _score(inputs, directory, seal, identity, historical, *, endpoints=None):
    split = inputs.split
    phase = 'endpoint' if endpoints is None else 'trajectory'
    verified = _verify_seal(split, seal, directory, identity, phase)
    receipt_path = directory/'SCORE_RECEIPT.json'
    table = directory/'PER_WINDOW.csv'
    if receipt_path.exists():
        receipt = read(receipt_path)
        require(receipt['status'] == 'COMPLETE' and receipt['manifest_sha256'] == verified['manifest_sha256'] and
                receipt['table_sha256'] == digest(table), 'completed score changed')
        rows = _csv(table)
        for row in rows:
            for key, val in row.items():
                if key not in ('fold', 'window'):
                    row[key] = None if val == '' else int(val) if key in ('seed', 'n') else float(val)
        return rows, receipt
    write(directory/'PRE_SCORE_VERIFICATION.json', verified)
    reader = FormalData(split, phase='scoring', verify=False)
    sealed_endpoints = None if endpoints is None else _decoder_inputs(endpoints, identity['endpoint_seal_sha256'])
    start = time.monotonic()
    scored_rows = []
    first_truth = None
    for ordinal, (index, row, previous) in enumerate(
            zip(split[PURPOSE+'_indices'], seal['rows'], historical), 1):
        n = row['n']
        path = directory/row['file']
        require(digest(path) == row['sha256'], 'scoring prediction changed since verification')
        arrays = _read_array(path, n, phase)
        if phase != 'endpoint':
            goals, _ = _decoder_input(endpoints, sealed_endpoints[row['window']])
            require(np.allclose(arrays[..., -1, :], goals, rtol=1e-6, atol=1e-6),
                    'decoded endpoints differ from sealed endpoints')
        if first_truth is None:
            first_truth = time.time()
            require(first_truth >= verified['completed_unix'] >= seal['seal_unix'],
                    'truth cannot be read before verification and seal')
        batch = reader.batch([index], PURPOSE, truth=True)
        require(bool(batch.valid.all()) and int(batch.valid.sum()) == n, 'scoring member differs')
        if phase == 'endpoint':
            goals, _ = arrays
            native = None
        else:
            native = hard_metric_statistics(torch.from_numpy(arrays)[None], batch.future, batch.valid)
        stats = endpoint_statistics(goals, batch.future[0, :, -1], batch.valid[0])
        scored_rows.append(_window_row(inputs.fold, ordinal, n, previous, stats, native))
    require(len(scored_rows) == len(seal['rows']), 'scoring table incomplete')
    csvwrite(table, scored_rows)
    receipt = dict(**verified, status='COMPLETE', phase=phase, first_truth_read_unix=first_truth,
                   truth_reads=len(scored_rows), score_wall_seconds=time.monotonic()-start,
                   table_sha256=digest(table), seed=SEED,
                   first_exploratory_truth_read=phase == 'endpoint')
    write(receipt_path, receipt)
    return scored_rows, receipt


def _groups(fold, rows, reference):
    grouped = {'all': rows}
    if fold == 'hotel':
        grouped['single'] = [row for row in rows if row['n'] == 1]
        grouped['multi'] = [row for row in rows if row['n'] > 1]
    else:
        grouped['N_ge_9'] = [row for row in rows if row['n'] >= 9]
    result = {}
    for group, values in grouped.items():
        scenes = len(values)
        agents = sum(row['n'] for row in values)
        require(scenes > 0 and agents > 0, f'empty required evaluation group {group}')
        item = dict(scene_count=scenes, agent_count=agents)
        for branch in ('B', 'new'):
            metrics = {}
            for name in METRICS:
                if values[0][f'{branch}_{name}'] is None:
                    require(all(row[f'{branch}_{name}'] is None for row in values), 'incomplete metric')
                    metrics[name] = None
                elif name in ('minADE', 'minFDE'):
                    metrics[name] = math.fsum(row[f'{branch}_agent_{name}_sum'] for row in values)/agents
                else:
                    metrics[name] = math.fsum(row[f'{branch}_{name}'] for row in values)/scenes
            if branch == 'B' and group in reference:
                for name, value in reference[group].items():
                    require(math.isclose(metrics[name], value, rel_tol=1e-12, abs_tol=1e-12),
                            'archived B seed summary does not match its windows')
                    metrics[name] = value
            item[branch] = metrics
        item['delta'] = {name: None if item['new'][name] is None else item['new'][name]-item['B'][name]
                         for name in METRICS}
        result[group] = item
    return result


def _summary_csv(path, fold, groups):
    rows = []
    for group in GROUPS[fold]:
        entry = groups[group]
        for metric in METRICS:
            rows.append(dict(fold=fold, group=group, seed=SEED, metric=metric,
                             scene_count=entry['scene_count'], agent_count=entry['agent_count'],
                             B_m=entry['B'][metric], new_m=entry['new'][metric], delta_m=entry['delta'][metric],
                             delta_cm=None if entry['delta'][metric] is None else 100*entry['delta'][metric]))
    csvwrite(path, rows)


@torch.no_grad()
def evaluate(inputs, model, provider, dependence, fold_dir, checkpoint_sha256, historical, *, device='cuda'):
    """Seal the epoch-eight endpoints, score the gate, and decode only on a pass."""
    start = time.monotonic()
    directory = Path(fold_dir)/'evaluation'
    directory.mkdir(parents=True, exist_ok=True)
    fold = inputs.fold
    split = inputs.split
    indices = split[PURPOSE+'_indices']
    require((len(indices), sum(len(split['members'][i]['agent_ids']) for i in indices)) == COUNTS[fold],
            'exploratory evaluation population differs')
    members = canonical_hash([split['members'][i] for i in indices])
    reference, historical_windows, historical_identity = _history(inputs, Path(historical), members)
    modules = dict(model=model, provider=provider, dependence=dependence)
    for module in modules.values():
        module.eval()
    hashes = {name: state_hash(module) for name, module in modules.items()}
    identity = dict(fold=fold, seed=SEED, epoch=8, checkpoint_sha256=checkpoint_sha256,
                    purpose=PURPOSE, scope=SCOPE, split_members_sha256=members,
                    output_state_sha256=hashes, **historical_identity)
    immutable(directory/'EVALUATION_CONFIG.json', dict(identity=identity, M=256, K=20,
              gates=dict(delta_minFDE_at_most=-.020, delta_JFDE_strictly_below=0.0),
              bridge=dict(ddim_steps=20, eta=0., noise='original sample ID from sealed endpoints'),
              marginal_aggregation='agent-weighted', joint_L_G_aggregation='window-weighted'))
    ep_dir = directory/'endpoints'
    endpoints = _export(inputs, model, provider, dependence, ep_dir, identity, device=device)
    rows, endpoint_receipt = _score(inputs, ep_dir, endpoints, identity, historical_windows)
    endpoint_groups = _groups(fold, rows, reference)
    all_delta = endpoint_groups['all']['delta']
    pass_m = all_delta['minFDE'] <= -.020
    pass_j = all_delta['JFDE'] < 0.
    gate = pass_m and pass_j
    immutable(directory/'ENDPOINT_GATE.json', dict(endpoint_gate_passed=gate, pass_minFDE=pass_m,
              pass_JFDE=pass_j, all_delta_m=all_delta, groups=endpoint_groups,
              endpoint_score_receipt_sha256=digest(ep_dir/'SCORE_RECEIPT.json')))
    trajectory_receipt = None
    trajectory_wall_seconds = None
    groups = endpoint_groups
    if gate:
        traj_dir = directory/'trajectories'
        decoder_identity = dict(identity, endpoint_seal_sha256=endpoint_receipt['manifest_sha256'])
        trajectories = _export(inputs, model, provider, dependence, traj_dir, decoder_identity,
                               endpoint_directory=ep_dir, device=device)
        trajectory_wall_seconds = trajectories['wall_seconds']
        rows, trajectory_receipt = _score(inputs, traj_dir, trajectories, decoder_identity,
                                          historical_windows, endpoints=ep_dir)
        groups = _groups(fold, rows, reference)
    require(hashes == {name: state_hash(module) for name, module in modules.items()},
            'evaluation modified a model state')
    full_pass = bool(gate and all(groups['all']['delta'][key] < 0. for key in NATIVE_METRICS))
    summary = dict(status='COMPLETE', scope=SCOPE, fold=fold, seed=SEED, evaluated_epoch=8,
                   identity=identity, groups=groups, endpoint_groups=endpoint_groups,
                   endpoint_gate_passed=gate, pass_minFDE=pass_m, pass_JFDE=pass_j,
                   four_metrics_improved=full_pass, ddim_calls=len(indices) if gate else 0,
                   endpoint_receipt=endpoint_receipt, trajectory_receipt=trajectory_receipt,
                   endpoint_export_wall_seconds=endpoints['wall_seconds'],
                   trajectory_export_wall_seconds=trajectory_wall_seconds,
                   states_unchanged=True, wall_seconds_this_attempt=time.monotonic()-start)
    csvwrite(directory/'PER_WINDOW.csv', rows)
    _summary_csv(directory/'SUMMARY.csv', fold, groups)
    summary['output_sha256'] = {name: digest(directory/name) for name in ('PER_WINDOW.csv', 'SUMMARY.csv')}
    write(directory/'SUMMARY.json', summary)
    return summary

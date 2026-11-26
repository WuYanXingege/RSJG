"""Seal all student trajectories before scoring the five-fold comparison."""
import math
from pathlib import Path
import time

import numpy as np
import torch

from bcep.contracts import require
from bcep_formal.data import FormalData
from bcep_joint.inference import generate as generate_a
from ebjd.metrics import MetricAccumulator, hard_metric_statistics
from joint_flow_dependency.inference import encode_observed, generate as generate_b
from joint_flow_frozen_test.inputs import ObservedOnlyHeldoutData
from joint_flow_frozen_test.run import forbid_heldout_truth
from joint_flow_fourfold.scoring import _arrays, _window_metrics, csvwrite
from .common import (
    ATOL, RTOL, FOLDS, GROUPS, METRICS, MODELS, PURPOSE, RUN, SEEDS,
    canonical_hash, digest, groups_for, immutable, read, status, window_key, write,
)


def _read_prediction(path, n):
    with np.load(path, allow_pickle=False) as saved:
        require(saved.files == ['paths'], 'student prediction must contain paths only')
        values = saved['paths']
    require(values.dtype == np.float32 and values.shape == (20, n, 12, 2)
            and np.isfinite(values).all(), 'invalid K20 student prediction')
    return values


def _save(path, paths):
    temp = path.with_suffix('.tmp.npz')
    np.savez_compressed(temp, paths=paths)
    temp.replace(path)


@torch.no_grad()
def _predict(assets, student, batch, key, seed):
    from .model import student_endpoints, student_noise, decode_student

    context, features = encode_observed(assets.provider, batch)
    goals = student_endpoints(student, assets.provider, batch.observed[0],
        batch.semantic_maps[0], batch.valid[0], context, features)
    bridge_context = assets.model.encode_context(batch.observed, batch.semantic_maps, batch.valid)
    noise = student_noise(key, batch.agent_ids[0], seed, goals.device)
    path = decode_student(assets.model, bridge_context, goals, noise)
    torch.testing.assert_close(path[..., -1, :], goals[None], atol=ATOL, rtol=RTOL)
    return path


@torch.no_grad()
def _benchmark(assets, reader, identity, rows, output):
    """Measure the original paths on the same resident windows/device as student."""
    target = output / 'TIMING_SAMPLE.json'
    if target.exists():
        sample = read(target)
        require(sample['identity'] == identity, 'timing reference changed')
        return sample
    split = assets.split
    indices = split[PURPOSE+'_indices']
    sample_indices = sorted(indices, key=lambda index: canonical_hash(
        ['direct-k20-timing', window_key(split, index)]))[:8]
    lookup = {(row['seed'], row['index']): row for row in rows}
    measured = []
    for index in sample_indices:
        batch = reader.batch([index], PURPOSE, truth=False).to('cuda')
        row = lookup[SEEDS[0], index]
        torch.cuda.synchronize()
        started = time.perf_counter()
        raw = assets.provider.candidates(batch.observed[0], batch.semantic_maps[0],
            batch.valid[0], batch.scene_ids[0], batch.agent_ids[0], SEEDS[0],
            assets.geometry, refined=True)
        generate_a(assets.model, batch.observed, batch.semantic_maps, batch.valid,
            batch.scene_ids, batch.agent_ids, raw[None], SEEDS[0], mode='H')
        torch.cuda.synchronize()
        a_seconds = time.perf_counter() - started
        started = time.perf_counter()
        generate_b(assets.model, assets.provider, assets.dependence, batch, seed=SEEDS[0])
        torch.cuda.synchronize()
        b_seconds = time.perf_counter() - started
        measured.append(dict(window=row['window'], n=row['n'], seed=SEEDS[0],
            A_seconds=a_seconds, B_seconds=b_seconds, student_seconds=row['inference_seconds']))
    sample = dict(identity=identity, sample=measured, A_ddim_calls=len(measured),
        B_ddim_calls=len(measured), student_ddim_calls_additional=0,
        timing_scope='resident observations through complete trajectories; synchronize CUDA; excludes read/write')
    immutable(target, sample)
    return sample


@torch.no_grad()
def export_fold(assets, student, selected, barrier_path, output, *, progress_root=RUN,
                benchmark=True):
    """Only observation arrays are read at generation time."""
    output.mkdir(parents=True, exist_ok=True)
    split = assets.split
    indices = split[PURPOSE+'_indices']
    identity = dict(fold=assets.fold, selected_sha256=selected['selected_sha256'],
        selected_epoch=selected['selected_epoch'], freeze_barrier_sha256=digest(barrier_path),
        ordered_members_sha256=canonical_hash([split['members'][index] for index in indices]),
        seeds=list(SEEDS), K=20, ddim_steps=20, eta=0., prediction_frames=12)
    immutable(output / 'EXPORT_CONFIG.json', identity)
    reader = ObservedOnlyHeldoutData(split)
    student.eval()
    rows = []
    started = time.monotonic()
    with forbid_heldout_truth():
        for ordinal, index in enumerate(indices, 1):
            member = split['members'][index]
            n = len(member['agent_ids'])
            record = split['records'][index]
            require(digest(record['path']) == record['sha256'], 'test input changed')
            batch = None
            reference_endpoints = None
            for seed in SEEDS:
                folder = output / f'seed_{seed}'
                folder.mkdir(exist_ok=True)
                path = folder / f'{ordinal:04d}.npz'
                side = path.with_suffix('.json')
                expected = dict(window=f'test_{ordinal:04d}', seed=seed, index=index, n=n,
                    member_sha256=window_key(split, index), file=str(path.relative_to(output)),
                    export_identity_sha256=canonical_hash(identity))
                if side.exists():
                    row = read(side)
                    require(all(row[k] == v for k, v in expected.items()) and
                            row['sha256'] == digest(path), 'student prediction changed')
                    paths = _read_prediction(path, n)
                else:
                    if batch is None:
                        batch = reader.batch([index], PURPOSE, truth=False).to('cuda')
                        require(batch.valid.all() and int(batch.valid.sum()) == n and
                                not batch.future.count_nonzero(), 'test observation packing differs')
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    prediction = _predict(assets, student, batch, expected['member_sha256'], seed)
                    torch.cuda.synchronize()
                    inference_seconds = time.perf_counter() - tick
                    paths = prediction[0].cpu().numpy()
                    require(paths.shape == (20, n, 12, 2) and paths.dtype == np.float32 and
                            np.isfinite(paths).all(), 'invalid student trajectory')
                    if path.exists():
                        require(np.array_equal(paths, _read_prediction(path, n)), 'orphan shard changed on resume')
                    else:
                        _save(path, paths)
                    row = dict(expected, sha256=digest(path), inference_seconds=inference_seconds)
                    write(side, row)
                endpoints = paths[..., -1, :]
                if reference_endpoints is None:
                    reference_endpoints = endpoints.copy()
                require(np.allclose(endpoints, reference_endpoints, rtol=RTOL, atol=ATOL),
                        'student endpoints depend on evaluation seed')
                rows.append(row)
            if ordinal % 50 == 0 or ordinal == len(indices):
                status(progress_root, 'STUDENT_TEST_EXPORT', fold=assets.fold, windows=ordinal,
                    total=len(indices), elapsed_seconds=time.monotonic()-started)
        timing = _benchmark(assets, reader, identity, rows, output) if benchmark else dict(
            A_ddim_calls=0, B_ddim_calls=0)
    seal = output / 'PREDICTIONS_FROZEN.json'
    if not seal.exists():
        immutable(seal, dict(identity=identity, rows=rows, seal_unix=time.time(), truth_reads=0,
            ddim_calls=len(rows), A_B_timing_ddim_calls=timing['A_ddim_calls'] + timing['B_ddim_calls'],
            export_wall_seconds=time.monotonic()-started))
    else:
        prior = read(seal)
        require(prior['identity'] == identity and prior['rows'] == rows, 'student seal changed')
    return dict(fold=assets.fold, seal_sha256=digest(seal), windows=len(indices), ddim_calls=len(rows))


def verify_export(fold, reference, directory, barrier_path, selected):
    seal_path = directory / 'PREDICTIONS_FROZEN.json'
    seal = read(seal_path)
    identity = seal['identity']
    require(seal['truth_reads'] == 0 and identity['fold'] == fold and
        identity['freeze_barrier_sha256'] == digest(barrier_path) and
        identity['selected_sha256'] == selected['selected_sha256'] and
        identity['selected_epoch'] == selected['selected_epoch'] and
        identity['ordered_members_sha256'] == reference['metadata']['ordered_members_sha256'],
        'student seal does not match frozen model or ordered test members')
    rows = {}
    endpoints = {}
    max_seed_endpoint_displacement = 0.
    for row in seal['rows']:
        key = row['seed'], row['index']
        original = reference['rows'][key]
        require(key not in rows and row['window'] == original['window'] and row['n'] == original['n'] and
                row['member_sha256'] == window_key(reference['split'], row['index']) and
                row['export_identity_sha256'] == canonical_hash(identity),
                'student/source row alignment changed')
        path = directory / row['file']
        require(digest(path) == row['sha256'], 'student prediction shard changed')
        goal = _read_prediction(path, row['n'])[..., -1, :]
        if row['index'] not in endpoints:
            endpoints[row['index']] = goal.copy()
        first = endpoints[row['index']]
        require(np.allclose(goal, first, atol=ATOL, rtol=RTOL), 'seed changed a sealed endpoint')
        max_seed_endpoint_displacement = max(max_seed_endpoint_displacement,
                                             float(np.max(np.abs(goal-first))))
        rows[key] = row
    require(rows.keys() == reference['rows'].keys(), 'incomplete student freeze')
    return rows, dict(fold=fold, sha256=digest(seal_path), pairs=len(rows),
                      max_seed_endpoint_displacement_m=max_seed_endpoint_displacement)


@torch.no_grad()
def score_all(references, barrier_path, *, run=RUN, folds=FOLDS):
    receipt_path = run / 'SCORE_RECEIPT.json'
    if receipt_path.exists():
        receipt = read(receipt_path)
        require(receipt['global_prediction_seal_sha256'] == digest(run/'ALL_PREDICTIONS_SEALED.json')
                and all(digest(run / n) == s for n, s in receipt['output_sha256'].items()),
                'completed native scoring changed')
        return receipt
    barrier = read(barrier_path)
    started = time.monotonic()
    mappings = {}
    records = {}
    for fold in folds:
        directory = run / fold / 'test'
        mappings[fold], records[fold] = verify_export(fold, references[fold],
            directory, barrier_path, barrier['folds'][fold])
    global_path = run / 'ALL_PREDICTIONS_SEALED.json'
    core = dict(freeze_barrier_sha256=digest(barrier_path), folds=records,
                window_seed_pairs=sum(record['pairs'] for record in records.values()), truth_reads=0)
    if global_path.exists():
        whole = read(global_path)
        require(all(whole[k] == v for k, v in core.items()), 'global student seal changed')
    else:
        whole = dict(core, seal_unix=time.time())
        immutable(global_path, whole)
    first_truth = None
    windows, seed_rows = [], []
    split_counts = {}
    max_drift = {}
    max_baseline_replay = 0.
    for fold in folds:
        reference = references[fold]
        split = reference['split']
        reader = FormalData(split, 'scoring', verify=False)
        counts_per_group = {}
        accumulators = {seed: {group: {name: MetricAccumulator() for name in MODELS}
                               for group in GROUPS} for seed in SEEDS}
        directory = run / fold / 'test'
        fold_max = {metric: 0. for metric in ('JFDE', 'minFDE')}
        for ordinal, index in enumerate(reference['indices'], 1):
            paths = {}
            for seed in SEEDS:
                old = reference['rows'][seed, index]
                original_path = reference['directory'] / old['file']
                require(digest(original_path) == old['sha256'], 'baseline prediction changed')
                arrays = _arrays(original_path, old['n'])
                row = mappings[fold][seed, index]
                student_path = directory / row['file']
                require(digest(student_path) == row['sha256'], 'student prediction changed before truth')
                arrays['new'] = _read_prediction(student_path, row['n'])
                paths[seed] = arrays
            rec = split['records'][index]
            require(digest(rec['path']) == rec['sha256'], 'test truth record changed')
            if first_truth is None:
                first_truth = time.time()
                require(first_truth >= whole['seal_unix'], 'truth read before global seal')
            batch = reader.batch([index], PURPOSE, truth=True)
            n = int(batch.valid.sum())
            require(bool(batch.valid.all()) and n == mappings[fold][SEEDS[0], index]['n'],
                    'native valid mask differs')
            original_student_scores = None
            for seed in SEEDS:
                values = dict(fold=fold, seed=seed, window=f'test_{ordinal:04d}', n=n)
                for name, pred in paths[seed].items():
                    metric = hard_metric_statistics(torch.from_numpy(pred)[None], batch.future, batch.valid)
                    for group in groups_for(n):
                        accumulators[seed][group][name].update_statistics(metric)
                    values.update(_window_metrics(metric, name+'_'))
                if original_student_scores is None:
                    original_student_scores = values
                for name in ('JFDE','minFDE'):
                    previous = original_student_scores['new_'+name]
                    actual = values['new_'+name]
                    require(math.isclose(actual, previous, rel_tol=RTOL, abs_tol=ATOL),
                            'student endpoint metrics changed across seeds')
                    fold_max[name] = max(fold_max[name], abs(actual-previous))
                for name in METRICS:
                    for ref in ('A', 'B'):
                        values[f'new_minus_{ref}_{name}_cm'] = 100*(values['new_'+name] - values[ref+'_'+name])
                windows.append(values)
            if ordinal % 100 == 0 or ordinal == len(reference['indices']):
                status(run, 'SCORING_FROZEN_STUDENTS', fold=fold, windows=ordinal, total=len(reference['indices']))
        max_drift[fold] = fold_max
        for seed in SEEDS:
            for group in GROUPS:
                models = accumulators[seed][group]
                count = models['A'].scene_count
                counts = dict(scene_count=count, agent_count=models['A'].agent_count)
                if seed == SEEDS[0]:
                    counts_per_group[group] = counts
                require(counts_per_group[group] == counts, 'seed population differed')
                row = dict(fold=fold, seed=seed, group=group, **counts)
                for name in MODELS:
                    values = models[name].as_dict() if count else {}
                    row.update({f'{name}_{metric}': values.get(metric) for metric in METRICS})
                    if name != 'new' and count and group in ('all','N_ge_9'):
                        historic = reference['baseline'][seed, group]
                        for metric in METRICS:
                            expected = float(historic[f'{name}_{metric}'])
                            actual = row[f'{name}_{metric}']
                            require(math.isclose(actual, expected, rel_tol=RTOL, abs_tol=ATOL),
                                    'formal A/B scores did not reproduce')
                            max_baseline_replay = max(max_baseline_replay, abs(expected-actual))
                for metric in METRICS:
                    for name in ('A','B'):
                        row[f'new_minus_{name}_{metric}_cm'] = (
                            100*(row[f'new_{metric}']-row[f'{name}_{metric}']) if count else None)
                seed_rows.append(row)
        split_counts[fold] = counts_per_group
    csvwrite(run/'PER_SEED.csv', seed_rows)
    csvwrite(run/'PER_WINDOW.csv', windows)
    scope = '五折五种子探索性对照' if len(folds) == 5 else f'{folds[0].upper()} 五种子提前探索性对照'
    receipt = dict(status='COMPLETE', scope=scope, K_per_seed=20,
        windows=sum(len(references[f]['indices']) for f in folds), window_seed_pairs=len(windows),
        selection_seal_unix=whole['seal_unix'], first_truth_read_unix=first_truth,
        global_prediction_seal_sha256=digest(global_path), split_counts=split_counts,
        max_seed_endpoint_metric_drift_m=max_drift, original_AB_max_abs_replay_m=max_baseline_replay,
        scoring_wall_seconds=time.monotonic()-started, output_sha256={
            filename: digest(run/filename) for filename in ('PER_SEED.csv','PER_WINDOW.csv')})
    write(receipt_path, receipt)
    return receipt

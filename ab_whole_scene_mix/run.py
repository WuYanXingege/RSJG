"""Seal a fixed A/B mixture from the official predictions, then score it."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from bcep.contracts import canonical_hash, digest, require
from bcep_formal.data import FormalData
from bcep_joint.common import read, write
from ebjd.metrics import MetricAccumulator, hard_metric_statistics
from joint_flow_fourfold.common import immutable, status
from joint_flow_fourfold.scoring import _arrays, _window_metrics, csvwrite
from .assets import discover
from .select import gather, select

SOURCE = Path(__file__).resolve().parents[1]
RUN = SOURCE.parent / 'RSJG_EBJD_assets/experiments/ab_whole_scene_mix_20261126'
DOC = SOURCE / 'docs/research/continuous_joint_flow/ab_whole_scene_mix_20261126'
BASE = '3296fc5968463ed14bf43265c064b3856d55ed30'
BRANCH = 'experiment/ab-whole-scene-mix-20261126'
FOLDS = ('zara1', 'hotel', 'univ', 'eth', 'zara2')
SEEDS = tuple(range(2035, 2040))
PURPOSE = 'heldout_confirmation'
GROUPS = ('all', 'single', 'multi', 'N_ge_9')
MODELS = ('A', 'B', 'new')
METRICS = ('JADE', 'JFDE', 'minADE', 'minFDE')
BASELINE_TOLERANCE = 1e-6


def sources():
    dependencies = ['ebjd/metrics.py', 'joint_flow_fourfold/scoring.py', 'bcep_formal/data.py']
    paths = [*sorted((SOURCE / 'ab_whole_scene_mix').glob('*.py')),
             *(SOURCE / name for name in dependencies)]
    return {str(path.relative_to(SOURCE)): digest(path) for path in paths}


def source_arrays(asset, seed, index):
    row = asset['rows'][seed, index]
    path = asset['directory'] / row['file']
    require(path.is_file() and digest(path) == row['sha256'],
            'official source prediction is missing or has changed')
    return row, _arrays(path, row['n'])


def observed_members(split, index):
    member = split['members'][index]
    record = split['records'][index]
    path = Path(record['path'])
    require(digest(path) == record['sha256'], 'official observation record changed')
    with np.load(path, allow_pickle=False) as archive:
        observed = archive['observed']
    n = len(member['agent_ids'])
    require(observed.shape == (n, 8, 2) and n > 0 and bool(np.isfinite(observed).all()),
            'invalid packed observation or valid agent count')
    frames = member['frame_ids']
    require(frames == sorted(set(frames)) and len(member['agent_ids']) == len(set(member['agent_ids'])),
            'invalid observation membership or frame ordering')
    return n, canonical_hash(member)


def raw_hash(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast('B')).hexdigest()


def verify_slots(slots, n):
    require(len(slots) == 20 and len({tuple(pair) for pair in slots}) == 20, 'incorrect slot count')
    require(all(model in ('A', 'B') and type(slot) is int and 0 <= slot < 20
                for model, slot in slots), 'invalid original slot')
    if n == 1:
        require(slots == [['A', slot] for slot in range(20)], 'singleton must replay A exactly')
    else:
        require([model for model, _ in slots] == ['B']*16 + ['A']*4,
                'multiplayer mix must contain 16 B and 4 A worlds')
        require([slot for _, slot in slots[:16]] == sorted(slot for _, slot in slots[:16]),
                'original B survivor order changed')


def seal_fold(fold, asset, config_sha):
    path = RUN / f'{fold}_SLOTS_SEALED.json'
    if path.exists():
        seal = read(path)
        require(seal['config_sha256'] == config_sha and seal['fold'] == fold, 'selection identity changed')
        return seal
    started = time.monotonic()
    selector_seconds = 0.
    rows = []
    split = asset['split']
    for ordinal, index in enumerate(asset['indices'], 1):
        n, member_hash = observed_members(split, index)
        for seed in SEEDS:
            source, arrays = source_arrays(asset, seed, index)
            require(source['n'] == n and source['window'] == f'test_{ordinal:04d}', 'source membership differs')
            selector_start = time.process_time()
            choice = select(arrays['A'], arrays['B'])
            selector_seconds += time.process_time() - selector_start
            verify_slots(choice['slots'], n)
            rows.append(dict(window=source['window'], index=index, seed=seed, n=n,
                             member_sha256=member_hash, source_pair_sha256=source['sha256'],
                             source_A_tensor_sha256=raw_hash(arrays['A']),
                             source_B_tensor_sha256=raw_hash(arrays['B']), **choice))
        if ordinal % 100 == 0 or ordinal == len(asset['indices']):
            status(RUN, 'SELECTING_FROM_OBSERVED_PREDICTIONS', fold=fold, windows=ordinal,
                   total=len(asset['indices']), seconds=time.monotonic()-started)
    seal = dict(fold=fold, config_sha256=config_sha, rows=rows, seal_unix=time.time(),
                truth_reads=0, combination_wall_seconds=time.monotonic()-started,
                selector_cpu_seconds=selector_seconds)
    immutable(path, seal)
    return seal


def seal_all(assets, config_sha):
    seals = {fold: seal_fold(fold, assets[fold], config_sha) for fold in FOLDS}
    path = RUN / 'ALL_SLOTS_SEALED.json'
    csv_path = RUN / 'SELECTED_SLOTS.csv'
    fold_hashes = {fold: digest(RUN / f'{fold}_SLOTS_SEALED.json') for fold in FOLDS}
    if path.exists():
        final = read(path)
        require(final['fold_sha256'] == fold_hashes and final['config_sha256'] == config_sha and
                final['selected_slots_csv_sha256'] == digest(csv_path), 'selection seal changed')
        return seals, final
    public_rows = []
    for fold in FOLDS:
        for row in seals[fold]['rows']:
            public_rows.append(dict(fold=fold, seed=row['seed'], window=row['window'], n=row['n'],
                source_pair_sha256=row['source_pair_sha256'],
                source_A_tensor_sha256=row['source_A_tensor_sha256'],
                source_B_tensor_sha256=row['source_B_tensor_sha256'],
                selected_model_and_zero_based_slot=json.dumps(row['slots'], separators=(',', ':')),
                B_removed_zero_based_slots=json.dumps(row['removed_B'], separators=(',', ':')),
                Jproxy_after_B_deletions_m=row['Jproxy_after_deletions_m'],
                Mproxy_before_A_additions_m=row['Mproxy_before_additions_m'],
                Mproxy_after_A_additions_m=row['Mproxy_after_additions_m']))
    require(len(public_rows) == 16740, 'incorrect fixed population')
    csvwrite(csv_path, public_rows)
    final = dict(fold_sha256=fold_hashes, config_sha256=config_sha, seal_unix=time.time(),
                 windows=sum(len(asset['indices']) for asset in assets.values()),
                 window_seed_pairs=len(public_rows), selected_slots_csv_sha256=digest(csv_path),
                 combination_wall_seconds=sum(s['combination_wall_seconds'] for s in seals.values()),
                 selector_cpu_seconds=sum(s['selector_cpu_seconds'] for s in seals.values()), truth_reads=0)
    immutable(path, final)
    return seals, final


def groups_for(n):
    return ('all', 'single') if n == 1 else (('all', 'multi', 'N_ge_9') if n >= 9 else ('all', 'multi'))


def score(assets, seals, sealed):
    out = RUN / 'SCORE_RECEIPT.json'
    outputs = ('PER_WINDOW.csv', 'PER_SEED.csv', 'SELECTED_SLOTS.csv')
    manifest_sha = digest(RUN / 'ALL_SLOTS_SEALED.json')
    if out.exists():
        receipt = read(out)
        require(receipt['selection_seal_sha256'] == manifest_sha and
                receipt['output_sha256'] == {name: digest(RUN/name) for name in outputs},
                'completed scoring artifact changed')
        return receipt
    started = time.monotonic()
    mappings = {}
    for fold in FOLDS:
        asset = assets[fold]
        split = asset['split']
        members = {i: canonical_hash(split['members'][i]) for i in asset['indices']}
        mapping = {}
        for row in seals[fold]['rows']:
            key = row['seed'], row['index']
            orig = asset['rows'][key]
            require(key not in mapping and row['window'] == orig['window'] and row['n'] == orig['n'] and
                    row['member_sha256'] == members[row['index']] and row['source_pair_sha256'] == orig['sha256'],
                    'invalid sealed choice alignment')
            verify_slots(row['slots'], row['n'])
            mapping[key] = row
        require(set(mapping) == set(asset['rows']), 'incomplete selection membership')
        mappings[fold] = mapping
    verification = time.time()
    require(verification >= sealed['seal_unix'], 'verification precedes selection seal')

    first_truth = None
    windows = []
    seed_rows = []
    split_counts = {}
    singleton_exact = 0
    replay_errors = []
    for fold in FOLDS:
        asset = assets[fold]
        scoring = FormalData(asset['split'], 'scoring', verify=False)
        accumulators = {seed: {group: {model: MetricAccumulator() for model in MODELS}
                               for group in GROUPS} for seed in SEEDS}
        for ordinal, index in enumerate(asset['indices'], 1):
            prepared = {}
            for seed in SEEDS:
                selected = mappings[fold][seed, index]
                source, arrays = source_arrays(asset, seed, index)
                require(source['sha256'] == selected['source_pair_sha256'] and
                        all(raw_hash(arrays[model]) == selected[f'source_{model}_tensor_sha256']
                            for model in ('A', 'B')), 'source differs from global selection seal')
                arrays['new'] = gather(arrays['A'], arrays['B'], selected['slots'])
                if selected['n'] == 1:
                    require(arrays['A'].tobytes() == arrays['new'].tobytes(), 'singleton changed trajectory')
                prepared[seed] = arrays
            record = asset['split']['records'][index]
            require(digest(record['path']) == record['sha256'], 'scoring record changed')
            if first_truth is None:
                first_truth = time.time()
                require(first_truth >= verification, 'truth read before seal verification')
            batch = scoring.batch([index], PURPOSE, truth=True)
            n = int(batch.valid.sum())
            require(bool(batch.valid.all()) and batch.future.shape == (1, n, 12, 2), 'invalid native scoring batch')
            require(batch.future.dtype == torch.float32 and batch.future.device.type == 'cpu', 'scoring dtype/device')
            for seed, arrays in prepared.items():
                row = dict(fold=fold, window=f'test_{ordinal:04d}', seed=seed, n=n)
                require(mappings[fold][seed, index]['n'] == n, 'valid population changed at scoring')
                for model, value in arrays.items():
                    stats = hard_metric_statistics(torch.from_numpy(value)[None], batch.future, batch.valid)
                    for group in groups_for(n):
                        accumulators[seed][group][model].update_statistics(stats)
                    row.update(_window_metrics(stats, f'{model}_'))
                for metric in METRICS:
                    row[f'new_minus_B_{metric}_cm'] = 100*(row[f'new_{metric}'] - row[f'B_{metric}'])
                    row[f'new_minus_A_{metric}_cm'] = 100*(row[f'new_{metric}'] - row[f'A_{metric}'])
                    if n == 1:
                        require(row[f'new_{metric}'] == row[f'A_{metric}'], 'singleton native scores changed')
                singleton_exact += n == 1
                windows.append(row)
            if ordinal % 100 == 0 or ordinal == len(asset['indices']):
                status(RUN, 'SCORING_SEALED_SELECTION', fold=fold, windows=ordinal,
                       total=len(asset['indices']), seconds=time.monotonic()-started)
        split_counts[fold] = {}
        for seed in SEEDS:
            for group in GROUPS:
                acc = accumulators[seed][group]
                n_windows, n_agents = acc['A'].scene_count, acc['A'].agent_count
                counts = dict(scene_count=n_windows, agent_count=n_agents)
                row = dict(fold=fold, seed=seed, group=group, **counts)
                if seed == SEEDS[0]:
                    split_counts[fold][group] = counts
                require(split_counts[fold][group] == counts, 'population differs by seed')
                for model in MODELS:
                    require(acc[model].agent_count == n_agents and acc[model].scene_count == n_windows,
                            'native model population differs')
                    values = acc[model].as_dict() if n_windows else {}
                    row.update({f'{model}_{metric}': values.get(metric) for metric in METRICS})
                if group in ('all', 'N_ge_9'):
                    archived = asset['baseline'][seed, group]
                    require(all(int(archived[name]) == count for name, count in counts.items()),
                            'archived seed population differs')
                    for model in ('A', 'B'):
                        for metric in METRICS:
                            key = f'{model}_{metric}'
                            if n_windows:
                                drift = abs(row[key] - float(archived[key]))
                                require(drift <= BASELINE_TOLERANCE + BASELINE_TOLERANCE * abs(float(archived[key])),
                                        f'official {fold} {seed} {group} {key} replay mismatch')
                                replay_errors.append(drift)
                            else:
                                require(archived[key] in ('', None), 'archived empty group differs')
                for metric in METRICS:
                    for base in ('A', 'B'):
                        row[f'new_minus_{base}_{metric}_cm'] = (
                            100*(row[f'new_{metric}'] - row[f'{base}_{metric}']) if n_windows else None)
                seed_rows.append(row)
    require(len(windows) == sealed['window_seed_pairs'], 'missing native window scores')
    require(singleton_exact == len(SEEDS)*sum(x['single']['scene_count'] for x in split_counts.values()),
            'singleton equality count differs')
    csvwrite(RUN / 'PER_WINDOW.csv', windows)
    csvwrite(RUN / 'PER_SEED.csv', seed_rows)
    receipt = dict(status='COMPLETE', training_updates=0, supplementary_prediction_pairs=0,
        supplementary_model_forwards=0, selection_seal_sha256=manifest_sha, windows=sealed['windows'],
        window_seed_pairs=len(windows), singleton_exact_pairs=singleton_exact,
        original_AB_max_abs_replay_m=max(replay_errors), AB_replay_atol=BASELINE_TOLERANCE,
        AB_replay_rtol=BASELINE_TOLERANCE, split_counts=split_counts,
        selection_seal_unix=sealed['seal_unix'], verified_selected_slots_unix=verification,
        first_truth_read_unix=first_truth, truth_reads=sealed['windows'], K_per_seed=20,
        scoring_wall_seconds=time.monotonic()-started,
        output_sha256={name: digest(RUN/name) for name in outputs})
    write(out, receipt)
    return receipt


def archive():
    from .report import build_report
    build_report(RUN, DOC)
    assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == BRANCH
    assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], text=True).strip()
    subprocess.run(['git', 'add', str(DOC.relative_to(SOURCE))], check=True)
    subprocess.run(['git', 'diff', '--cached', '--check'], check=True)
    if subprocess.call(['git', 'diff', '--cached', '--quiet']):
        subprocess.run(['git', 'commit', '-m', 'Archive five-fold fixed K20 A/B trajectory mixture replay'], check=True)
    subprocess.run(['git', 'push', '-u', 'origin', BRANCH], check=True)
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/' + BRANCH],
                                     text=True).split()[0]
    require(remote == sha, 'remote branch differs from archived result')
    write(RUN / 'DELIVERY.json', dict(sha=sha, verified_remote_sha=remote, branch=BRANCH))


def main():
    torch.set_num_threads(1)
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assets = discover()
        require(set(assets) == set(FOLDS), 'missing official fold')
        config = dict(base_commit=BASE, scope='依照已有测试结果设计的五折五种子探索性回放',
            folds=list(FOLDS), seeds=list(SEEDS), K=20, prediction_frames=12, purpose=PURPOSE,
            rule='N=1 use bitwise A; N>=2 delete 4 B by Jproxy and add 4 A by Mproxy',
            distance='0.5 mean full-trajectory Euclidean distance + 0.5 terminal Euclidean distance; CPU FP64',
            slot_tie='smallest original zero-based slot; B survivors ascending then A in greedy admission order',
            scoring='native CPU FP32 hard_metric_statistics & MetricAccumulator; fold equal within seed',
            online_reference='N=1 one A forward of K20; N>=2 A and B each produce K20 before CPU selection',
            supplementary_prediction_pairs=0, training_updates=0,
            reproduction='OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m ab_whole_scene_mix.run',
            source_sha256=sources(), fold_inputs={f: assets[f]['metadata'] for f in FOLDS})
        immutable(RUN / 'RUN_CONFIG.json', config)
        config_sha = digest(RUN / 'RUN_CONFIG.json')
        if not (RUN / 'START.json').exists():
            write(RUN / 'START.json', dict(unix=time.time(),
                       source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                       command=config['reproduction']))
        seals, sealed = seal_all(assets, config_sha)
        receipt = score(assets, seals, sealed)
        require(config_sha == digest(RUN / 'RUN_CONFIG.json') and config['source_sha256'] == sources(),
                'runtime config or source changed')
        write(RUN / 'WALL_TIME.json', {name: sealed[name] for name in (
            'combination_wall_seconds', 'selector_cpu_seconds')} | dict(
            scoring_wall_seconds=receipt['scoring_wall_seconds'],
            wall_seconds=time.time() - read(RUN / 'START.json')['unix']))
        status(RUN, 'ARCHIVING')
        archive()
        status(RUN, 'COMPLETE', window_seed_pairs=sealed['window_seed_pairs'],
               training_updates=0, supplementary_prediction_pairs=0)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        RUN.mkdir(parents=True, exist_ok=True)
        status(RUN, 'INTERRUPTED_RESUME_SAME_REPLAY', traceback=traceback.format_exc())
        raise

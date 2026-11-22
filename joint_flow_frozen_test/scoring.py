"""Score immutable observed-only test predictions with the native CPU metrics.

All five seeds' complete files are checked before FormalData(scoring) is even
constructed. This module never constructs a model, creates an optimizer, joins
seeds into K100, or alters an already sealed prediction.
"""
import csv
import math
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from bcep.contracts import canonical_hash, digest, require
from bcep_formal.data import FormalData
from bcep_joint.common import read, write
from ebjd.metrics import MetricAccumulator, hard_metric_statistics
from .inputs import RUN

SEEDS = list(range(2035, 2040))
GROUPS = ('all', 'N_ge_9')
METRICS = ('JADE', 'JFDE', 'minADE', 'minFDE')
OUTPUTS = ('PER_WINDOW.csv', 'PER_SEED.csv', 'NATIVE_METRICS.json', 'SUMMARY.json', 'SUMMARY.csv')


def csvwrite(path, rows):
    require(bool(rows), 'empty scoring table')
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _arrays(path, n):
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == {'A', 'B'}, 'prediction archive must contain A/B only, with no truth')
        arrays = {branch: archive[branch].copy() for branch in ('A', 'B')}
    for value in arrays.values():
        require(value.dtype == np.dtype('float32'), 'prediction is not FP32')
        require(value.shape == (20, n, 12, 2), 'wrong K20 prediction shape')
        require(bool(np.isfinite(value).all()), 'nonfinite prediction')
    return arrays


def verify_seal(inputs, seal):
    """Read prediction files/metadata only; no scoring data access is allowed."""
    manifest = RUN/'PREDICTIONS_FROZEN.json'
    require(manifest.is_file() and read(manifest) == seal, 'seal must already be persisted, unchanged')
    require(seal['identity'] == inputs.identity, 'frozen prediction identity mismatch')
    require(seal['source_config_sha256'] == digest(RUN/'RUN_CONFIG.json'), 'source configuration changed')
    require(seal['windows'] == 998 and seal['seeds'] == SEEDS, 'wrong fixed test/seed protocol')
    require(math.isfinite(seal['seal_unix']) and seal['seal_unix'] > 0, 'invalid seal timestamp')
    indices = inputs.split['heldout_confirmation_indices']
    require(len(indices) == len(set(indices)) == 998, 'wrong heldout membership')
    require(len(seal['rows']) == len(indices)*len(SEEDS), 'incomplete five-seed seal')
    position = {index: ordinal for ordinal, index in enumerate(indices)}
    mapping = {}
    dense = agents = dense_agents = 0
    for index in indices:
        n = len(inputs.split['members'][index]['agent_ids'])
        require(n > 0, 'empty heldout window')
        agents += n
        if n >= 9:
            dense += 1; dense_agents += n
    for row in seal['rows']:
        seed, index, n = row['seed'], row['index'], row['n']
        require(seed in SEEDS and index in position, 'unexpected prediction member')
        require((seed, index) not in mapping, 'duplicate sealed prediction')
        require(n == len(inputs.split['members'][index]['agent_ids']), 'sealed agent count changed')
        require(row['window'] == f'test_{position[index]+1:04d}', 'noncanonical anonymous window ID')
        require(row['file'] == f'predictions/seed_{seed}/{index}.npz', 'unexpected prediction path')
        path = RUN/row['file']
        require(path.is_file() and digest(path) == row['sha256'], 'sealed prediction hash mismatch')
        _arrays(path, n)
        mapping[seed, index] = row
    require(set(mapping) == {(seed, index) for seed in SEEDS for index in indices}, 'missing sealed member')
    require(read(manifest) == seal, 'manifest changed during pre-score verification')
    verified = dict(status='ALL_PREDICTIONS_VERIFIED_BEFORE_TRUTH', manifest_sha256=digest(manifest),
        manifest_canonical_sha256=canonical_hash(seal), source_config_sha256=seal['source_config_sha256'],
        files_verified=len(mapping), windows=998, seeds=SEEDS, K_per_seed=20, prediction_frames=12,
        groups=dict(all=dict(scene_count=998, agent_count=agents),
                    N_ge_9=dict(scene_count=dense, agent_count=dense_agents)),
        seal_unix=seal['seal_unix'], verification_complete_unix=time.time(), truth_reads=0,
        all_prediction_npz_keys=['A', 'B'], all_prediction_dtype='float32')
    require(verified['verification_complete_unix'] > seal['seal_unix'], 'seal timestamp is not earlier than verification')
    return mapping, verified


def _window_metrics(stats, prefix):
    valid = stats.agent_valid
    return {prefix+'JADE': float(stats.scene_jade[0]),
            prefix+'JFDE': float(stats.scene_jfde[0]),
            prefix+'minADE': float(stats.agent_minade[valid].double().mean()),
            prefix+'minFDE': float(stats.agent_minfde[valid].double().mean()),
            prefix+'agent_minADE_sum': float(stats.agent_minade[valid].double().sum()),
            prefix+'agent_minFDE_sum': float(stats.agent_minfde[valid].double().sum()),
            prefix+'scene_minADE': float(stats.scene_marginal_ade[0]),
            prefix+'scene_minFDE': float(stats.scene_marginal_fde[0])}


def _mean_std(values):
    require(len(values) == 5 and all(math.isfinite(x) for x in values), 'invalid seed statistics')
    return dict(mean=math.fsum(values)/len(values), std=statistics.stdev(values))


@torch.no_grad()
def score(inputs, seal):
    """One future read per window, after all 4,990 predictions are verified."""
    mapping, verified = verify_seal(inputs, seal)
    previous = RUN/'SCORE_RECEIPT.json'
    if previous.exists():
        saved = read(previous)
        require(saved['status'] == 'COMPLETE' and saved['prediction_manifest_sha256'] == verified['manifest_sha256'],
                'previous completed score belongs to another frozen prediction set')
        require(saved['output_sha256'] == {name: digest(RUN/name) for name in OUTPUTS}, 'completed scores changed')
        return saved
    write(RUN/'PRE_SCORE_SEAL_VERIFICATION.json', verified)
    # No FormalData in scoring mode and no future tensor exists before this line.
    scoring = FormalData(inputs.split, 'scoring')
    accumulators = {seed: {group: {branch: MetricAccumulator() for branch in ('A', 'B')}
                           for group in GROUPS} for seed in SEEDS}
    window_rows = []
    first_truth_read_unix = None
    truth_reads = 0
    start = time.monotonic()
    for ordinal, index in enumerate(inputs.split['heldout_confirmation_indices']):
        # Reload/recheck each sealed file before this window's single truth read.
        paths = {}
        for seed in SEEDS:
            row = mapping[seed, index]
            path = RUN/row['file']
            require(digest(path) == row['sha256'], 'prediction changed after the global seal audit')
            paths[seed] = _arrays(path, row['n'])
        if first_truth_read_unix is None:
            first_truth_read_unix = time.time()
            require(first_truth_read_unix > verified['verification_complete_unix'] > seal['seal_unix'],
                    'future access must follow complete prediction sealing and verification')
        batch = scoring.batch([index], 'heldout_confirmation', truth=True)
        truth_reads += 1
        n = int(batch.valid.sum())
        require(n == mapping[SEEDS[0], index]['n'] and bool(batch.valid.all()), 'scoring population mismatch')
        require(batch.future.device.type == 'cpu' and batch.future.dtype == torch.float32, 'native scoring must use CPU FP32')
        for seed in SEEDS:
            row = dict(window=f'test_{ordinal+1:04d}', seed=seed, n=n)
            for branch in ('A', 'B'):
                prediction = torch.from_numpy(paths[seed][branch])[None]
                stats = hard_metric_statistics(prediction, batch.future, batch.valid)
                accumulators[seed]['all'][branch].update_statistics(stats)
                if n >= 9:
                    accumulators[seed]['N_ge_9'][branch].update_statistics(stats)
                row.update(_window_metrics(stats, branch+'_'))
            row.update({f'B_minus_A_{metric}': row['B_'+metric]-row['A_'+metric] for metric in METRICS})
            window_rows.append(row)
        if (ordinal+1) % 100 == 0 or ordinal+1 == 998:
            progress = dict(phase='SCORING_FROZEN_TEST', windows=ordinal+1, total=998,
                            future_reads=truth_reads, prediction_manifest_sha256=verified['manifest_sha256'],
                            seconds=time.monotonic()-start)
            write(RUN/'SCORE_PROGRESS.json', progress)
            print(progress, flush=True)
    require(truth_reads == 998, 'each heldout future must be loaded once per scoring pass')
    native = {str(seed): {group: {branch: acc.as_dict() for branch, acc in branches.items()}
                         for group, branches in groups.items()} for seed, groups in accumulators.items()}
    seed_rows = []
    for seed in SEEDS:
        for group in GROUPS:
            counts = verified['groups'][group]
            row = dict(seed=seed, group=group, **counts)
            for branch in ('A', 'B'):
                value = native[str(seed)][group][branch]
                require(all(value[k] == v for k, v in counts.items()), 'native population changed across seeds')
                row.update({branch+'_'+metric: value[metric] for metric in METRICS})
            row.update({f'B_minus_A_{metric}': row['B_'+metric]-row['A_'+metric] for metric in METRICS})
            seed_rows.append(row)
    summary = dict(seeds=SEEDS, seed_count=5, ddof=1, unit='m', K_per_seed=20,
        aggregation='native metrics within each seed; equally weighted five-seed mean and sample standard deviation',
        difference_std='sample standard deviation of the five paired B-minus-A seed differences',
        groups={})
    summary_rows = []
    for group in GROUPS:
        subset = [r for r in seed_rows if r['group'] == group]
        result = dict(**verified['groups'][group])
        for branch in ('A', 'B', 'B_minus_A'):
            result[branch] = {metric: _mean_std([row[branch+'_'+metric] for row in subset]) for metric in METRICS}
            row = dict(group=group, branch=branch, **verified['groups'][group], seed_count=5, ddof=1)
            for metric in METRICS:
                row[metric+'_mean'] = result[branch][metric]['mean']
                row[metric+'_std'] = result[branch][metric]['std']
            summary_rows.append(row)
        summary['groups'][group] = result
    summary['interpretation'] = dict(
        all_JADE_lower=summary['groups']['all']['B_minus_A']['JADE']['mean'] < 0,
        all_JFDE_lower=summary['groups']['all']['B_minus_A']['JFDE']['mean'] < 0,
        N_ge_9_JFDE_lower=summary['groups']['N_ge_9']['B_minus_A']['JFDE']['mean'] < 0,
        marginal_minFDE_change_cm={group: 100*summary['groups'][group]['B_minus_A']['minFDE']['mean'] for group in GROUPS},
        comparator='same heldout test windows and paired seeds, fixed old coordinate epoch5 A',
        development_thresholds_applied=False, prediction_or_model_selection=False)
    require(read(RUN/'PREDICTIONS_FROZEN.json') == seal, 'prediction manifest changed during scoring')
    csvwrite(RUN/'PER_WINDOW.csv', window_rows)
    csvwrite(RUN/'PER_SEED.csv', seed_rows)
    write(RUN/'NATIVE_METRICS.json', native)
    write(RUN/'SUMMARY.json', summary)
    csvwrite(RUN/'SUMMARY.csv', summary_rows)
    receipt = dict(status='COMPLETE', prediction_manifest_sha256=verified['manifest_sha256'],
        prediction_manifest_canonical_sha256=verified['manifest_canonical_sha256'],
        source_config_sha256=seal['source_config_sha256'],
        pre_score_seal_verification_sha256=digest(RUN/'PRE_SCORE_SEAL_VERIFICATION.json'),
        all_predictions_verified_before_any_truth=True, prediction_files_verified=4990,
        seal_unix=seal['seal_unix'], verification_complete_unix=verified['verification_complete_unix'],
        first_truth_read_unix=first_truth_read_unix, scoring_complete_unix=time.time(),
        windows=998, groups=verified['groups'], seeds=SEEDS, K_per_seed=20, prediction_frames=12,
        future_reads_this_complete_pass=truth_reads, future_read_once_per_window_for_all_five_seeds=True,
        metric_implementation='ebjd.metrics.hard_metric_statistics and MetricAccumulator',
        metric_precision='CPU FP32 per-world errors and native FP64 accumulation',
        seed_pooling=False, parameter_updates=0, selection_on_test=False, development_thresholds_applied=False,
        output_sha256={name: digest(RUN/name) for name in OUTPUTS}, seconds=time.monotonic()-start)
    write(previous, receipt)
    return receipt

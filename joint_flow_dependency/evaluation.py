"""Native full-trajectory evaluation of continuous dependence and fixed controls.

Every epoch evaluates actual K20 bridge predictions. Final D averages the five
native statistics within each window before adding that window once to the
original MetricAccumulator. Future labels are read only after prediction.
"""
from dataclasses import fields
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import time

import torch
from bcep.contracts import canonical_hash, digest
from bcep_joint.common import read, write
from ebjd.metrics import BatchMetricStatistics, MetricAccumulator, hard_metric_statistics
from ebjd.unified import gather_candidates
from .inputs import ROOT, RUN, SOURCE, ARCHIVE, ARCHIVE_DIR, LIMITS
from .inference import generate, sample_endpoints, decode, endpoint_minfde
from .model import delta_nll
from .sampling import shuffled_correspondence

GROUPS = ('all', 'N_ge_9')
STAGES = ('raw256', 'selected20', 'corrected20')


def csvwrite(path, rows):
    """Atomic UTF-8 CSV; public records contain anonymous ordinal IDs only."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def cpu_statistics(statistics):
    return BatchMetricStatistics(**{f.name: getattr(statistics, f.name).detach().cpu()
                                    for f in fields(BatchMetricStatistics)})


def average_statistics(statistics):
    """Average native metric values, never trajectories or best-world indices."""
    assert len(statistics) == 5
    valid = statistics[0].agent_valid
    assert all(torch.equal(item.agent_valid, valid) for item in statistics)
    averaged = {}
    for field in fields(BatchMetricStatistics):
        if field.name == 'agent_valid':
            averaged[field.name] = valid
        else:
            averaged[field.name] = torch.stack([getattr(item, field.name).double() for item in statistics]).mean(0)
    return BatchMetricStatistics(**averaged)


def new_accumulators():
    return {group: MetricAccumulator() for group in GROUPS}


def update_accumulators(accumulators, statistics):
    accumulators['all'].update_statistics(statistics)
    if int(statistics.agent_valid.sum()) >= 9:
        accumulators['N_ge_9'].update_statistics(statistics)


def summaries(accumulators):
    return {group: accumulator.as_dict() for group, accumulator in accumulators.items()}


def metric_row(statistics, prefix=''):
    valid = statistics.agent_valid
    return {prefix+'JADE': float(statistics.scene_jade[0]),
            prefix+'JFDE': float(statistics.scene_jfde[0]),
            prefix+'minFDE': float(statistics.agent_minfde[valid].double().mean()),
            prefix+'agent_minFDE_sum': float(statistics.agent_minfde[valid].double().sum()),
            prefix+'scene_minFDE': float(statistics.scene_marginal_fde[0]),
            prefix+'minADE': float(statistics.agent_minade[valid].double().mean()),
            prefix+'agent_minADE_sum': float(statistics.agent_minade[valid].double().sum()),
            prefix+'scene_minADE': float(statistics.scene_marginal_ade[0])}


def covariance_diagnostics(covariance, prefix=''):
    n = covariance.shape[0]//2
    indices = torch.arange(n, device=covariance.device)
    blocks = covariance.reshape(n, 2, n, 2)[indices, :, indices, :]
    error = (blocks-torch.eye(2, device=covariance.device)).abs().max()
    diagonal = torch.linalg.cholesky(covariance).diagonal().min()
    return {prefix+'R_diagonal_block_max_abs_error': float(error),
            prefix+'R_cholesky_diagonal_min': float(diagonal)}


def coverage_row(endpoints, endpoint_truth, prefix=''):
    result = {}
    for stage, key in zip(STAGES, ('raw', 'selected', 'corrected')):
        values = endpoints[key][0] if key == 'corrected' else endpoints[key]
        error = endpoint_minfde(values, endpoint_truth).double()
        result[prefix+stage+'_minFDE_agent_sum'] = float(error.sum())
        result[prefix+stage+'_minFDE_scene_mean'] = float(error.mean())
    return result


def coverage_summary(rows, prefix=''):
    groups = {}
    for group in GROUPS:
        subset = [row for row in rows if group == 'all' or row['agents'] >= 9]
        if not subset:
            continue
        count = sum(row['agents'] for row in subset)
        groups[group] = {stage: dict(
            agent_micro_minFDE=math.fsum(row[prefix+stage+'_minFDE_agent_sum'] for row in subset)/count,
            scene_macro_minFDE=math.fsum(row[prefix+stage+'_minFDE_scene_mean'] for row in subset)/len(subset),
            agent_count=count, scene_count=len(subset)) for stage in STAGES}
    return groups


def nll_summary(rows, key):
    return {group: math.fsum(row[key] for row in rows if group == 'all' or row['agents'] >= 9)/
                   sum(group == 'all' or row['agents'] >= 9 for row in rows) for group in GROUPS}


def _ordered_cache(inputs, cache_dev):
    if isinstance(cache_dev, dict):
        mapping = cache_dev
    else:
        mapping = {record['index']: record for record in cache_dev}
    indices = inputs.split['selection_dev_indices']
    assert len(indices) == len(mapping) == 590
    assert set(mapping) == set(indices)
    return mapping


def _progress(phase, position, start, **extra):
    if position % 25 == 0 or position == 590:
        row = dict(phase=phase, windows=position, total=590, seconds=time.monotonic()-start, **extra)
        write(RUN/'PROGRESS.json', row)
        print(row, flush=True)


@torch.no_grad()
def evaluate_epoch(inputs, model, provider, dependence, cache_dev, device, epoch):
    """All 590 B predictions are newly generated and actually DDIM-decoded."""
    model.eval(); provider.eval(); dependence.eval()
    cache = _ordered_cache(inputs, cache_dev)
    accumulators = new_accumulators()
    rows = []
    start = time.monotonic()
    for position, index in enumerate(inputs.split['selection_dev_indices']):
        batch = inputs.data.batch([index], 'selection_dev', truth=False).to(device)
        path, endpoints, details = generate(model, provider, dependence, batch)
        assert details['ddim_calls'] == details['correction_calls'] == 1
        # Scoring begins only after the complete observed-only prediction.
        truth = inputs.data.batch([index], 'selection_dev', truth=True).future.to(device)
        statistics = cpu_statistics(hard_metric_statistics(path, truth, batch.valid))
        update_accumulators(accumulators, statistics)
        record = cache[index]
        assert torch.equal(record['inputs']['observed'], batch.observed[0].cpu())
        likelihood = float(delta_nll(endpoints['R'], record['latent_truth'].to(device)))
        assert math.isfinite(likelihood)
        row = dict(window=f'dev_{position+1:04d}', agents=int(batch.valid.sum()),
                   delta_nll_nat_per_person=likelihood, **metric_row(statistics),
                   **coverage_row(endpoints, truth[0, :, -1]), **covariance_diagnostics(endpoints['R']),
                   endpoint_vs_path_agent_minFDE_max_abs_drift_m=float((
                       endpoint_minfde(endpoints['corrected'][0], truth[0, :, -1]).cpu()-
                       statistics.agent_minfde[0]).abs().max()))
        rows.append(row)
        _progress('EPOCH_VALIDATION_B', position+1, start, epoch=epoch)
    groups = summaries(accumulators)
    nll = nll_summary(rows, 'delta_nll_nat_per_person')
    result = dict(**groups, dev_delta_nll_nat_per_person=nll['all'],
                  delta_nll_nat_per_person=nll, coverage=coverage_summary(rows), epoch=epoch,
                  windows=590, dense_windows=127, actual_ddim_calls=590, ddim_steps=20,
                  correction_calls=590, seconds=time.monotonic()-start,
                  numerical_diagnostics=dict(
                      R_diagonal_block_max_abs_error=max(row['R_diagonal_block_max_abs_error'] for row in rows),
                      R_cholesky_diagonal_min=min(row['R_cholesky_diagonal_min'] for row in rows),
                      endpoint_vs_path_agent_minFDE_max_abs_drift_m=max(
                          row['endpoint_vs_path_agent_minFDE_max_abs_drift_m'] for row in rows)))
    folder = RUN/'dev'
    csvwrite(folder/f'epoch_{epoch:03d}_windows.csv', rows)
    write(folder/f'epoch_{epoch:03d}_groups.json', result)
    return result


def _archive_blob(name):
    return subprocess.check_output(['git', 'show', ARCHIVE+':'+ARCHIVE_DIR+name], cwd=SOURCE)


def load_archived_h0(inputs):
    """Reconstruct A with the already archived, identity-verified native H0 stats.

    The closed selection head is never loaded or executed. Its prior offline
    H0 branch cache supplies only sufficient native metric statistics; every
    window's JADE/JFDE must equal the immutable published H0 CSV, and all native
    aggregates must equal the immutable published always-H0 summaries.
    """
    run_config = json.loads(_archive_blob('RUN_CONFIG.json'))
    assert run_config['identity'] == inputs.identity
    raw = _archive_blob('DEV_WINDOWS.csv')
    published = list(csv.DictReader(io.StringIO(raw.decode())))
    assert len(published) == 590
    expected = json.loads(_archive_blob('FINAL_METRICS.json'))['H0']
    root = ROOT/'RSJG_EBJD_assets/experiments/zara2_scene_pair_selector_20261122/cache'
    identity = read(root/'IDENTITY.json')
    assert {key: identity[key] for key in inputs.identity} == inputs.identity
    assert identity['noise_seed'] == 2035 and identity['K'] == identity['DDIM'] == 20
    identity_hash = canonical_hash(identity)
    manifest = read(root/'selection_dev_MANIFEST.json')
    assert len(manifest) == 590
    records = []
    accumulators = new_accumulators()
    for position, (item, archive_row, index) in enumerate(zip(manifest, published, inputs.split['selection_dev_indices'])):
        assert item['position'] == position and archive_row['window'] == f'dev_{position+1:04d}'
        path = root/item['file']
        assert digest(path) == item['sha256']
        record = torch.load(path, map_location='cpu', weights_only=False)
        assert record['identity'] == identity_hash and record['index'] == index
        statistics = BatchMetricStatistics(**record['statistics']['H0'])
        assert int(statistics.agent_valid.sum()) == int(archive_row['agents'])
        assert float(statistics.scene_jade[0]) == float(archive_row['H0_JADE'])
        assert float(statistics.scene_jfde[0]) == float(archive_row['H0_JFDE'])
        update_accumulators(accumulators, statistics)
        records.append(statistics)
    actual = summaries(accumulators)
    assert actual == expected
    write(RUN/'A_REFERENCE.json', dict(archive_commit=ARCHIVE, checkpoint_sha256=inputs.identity['checkpoint_sha256'],
        archive_csv_sha256=hashlib.sha256(raw).hexdigest(), cache_manifest_sha256=digest(root/'selection_dev_MANIFEST.json'),
        native_summary_exact_match=True, archived_window_metrics_exact_match=True, windows=590,
        groups=actual, additional_full_A_replay=False, selection_head_executed=False))
    return records


def _paired_summary(rows, group):
    subset = [row for row in rows if group == 'all' or row['agents'] >= 9]
    result = dict(group=group, windows=len(subset))
    for other in ('A', 'C', 'D'):
        for metric in ('JADE', 'JFDE'):
            differences = [row[f'B_{metric}']-row[f'{other}_{metric}'] for row in subset]
            result[f'B_minus_{other}_{metric}'] = math.fsum(differences)/len(subset)
            if other in ('C', 'D'):
                result[f'B_vs_{other}_{metric}_improved'] = sum(value < 0 for value in differences)
                result[f'B_vs_{other}_{metric}_worsened'] = sum(value > 0 for value in differences)
                result[f'B_vs_{other}_{metric}_tied'] = sum(value == 0 for value in differences)
    result['B_minus_C_delta_nll_nat_per_person'] = math.fsum(row['B_delta_nll_nat_per_person'] for row in subset)/len(subset)
    return result


@torch.no_grad()
def evaluate_final(inputs, model, provider, dependence, cache_dev, device, rank1_epoch=None):
    """Actual same-rank1 B plus independent C and five transported-shuffle D controls."""
    model.eval(); provider.eval(); dependence.eval()
    cache = _ordered_cache(inputs, cache_dev)
    archived = load_archived_h0(inputs)
    accumulators = {branch: new_accumulators() for branch in ('A', 'B', 'C', 'D')}
    rows, shuffle_rows = [], []
    endpoint_sets_checks = endpoint_minfde_checks = 0
    shuffled_ddim_calls = 0
    path_minfde_max_drift = 0.
    start = time.monotonic()
    for position, index in enumerate(inputs.split['selection_dev_indices']):
        batch = inputs.data.batch([index], 'selection_dev', truth=False).to(device)
        n = int(batch.valid.sum())
        path_b, goals_b, details = generate(model, provider, dependence, batch)
        assert details['ddim_calls'] == details['correction_calls'] == 1
        # C shares B's entire base streams; only the factor dependence is bypassed.
        goals_c = sample_endpoints(provider, dependence, batch, details['flow_context'], details['map_features'],
                                   independent=True, shared_base=(goals_b['epsilon'], goals_b['xi']))
        assert torch.equal(goals_b['epsilon'], goals_c['epsilon'])
        assert torch.equal(goals_b['xi'], goals_c['xi'])
        assert goals_c['R'] is None and goals_c['correction_calls'] == 1
        path_c = decode(model, details['bridge_context'], goals_c['corrected'], goals_c['noise'])
        truth = inputs.data.batch([index], 'selection_dev', truth=True).future.to(device)
        stat_b = cpu_statistics(hard_metric_statistics(path_b, truth, batch.valid))
        stat_c = cpu_statistics(hard_metric_statistics(path_c, truth, batch.valid))
        endpoint_b = endpoint_minfde(goals_b['corrected'][0], truth[0, :, -1])
        stat_d_repetitions = []
        max_window_drift = 0.
        for repetition in range(5):
            shuffled_goals, shuffled_noise, pi = shuffled_correspondence(goals_b['corrected'], goals_b['noise'],
                batch.scene_ids[0], batch.agent_ids[0], repetition)
            # Inverse reconstruction proves exact preservation of whole endpoint
            # vectors and matching whole trajectory-noise vectors, including duplicates.
            assert torch.equal(gather_candidates(shuffled_goals, pi.argsort(1)), goals_b['corrected'])
            assert torch.equal(gather_candidates(shuffled_noise, pi.argsort(1)), goals_b['noise'])
            assert torch.equal(endpoint_minfde(shuffled_goals[0], truth[0, :, -1]), endpoint_b)
            endpoint_sets_checks += 1; endpoint_minfde_checks += 1
            if n == 1:
                statistic = stat_b
            else:
                path_d = decode(model, details['bridge_context'], shuffled_goals, shuffled_noise)
                statistic = cpu_statistics(hard_metric_statistics(path_d, truth, batch.valid))
                shuffled_ddim_calls += 1
            drift = float((statistic.agent_minfde-stat_b.agent_minfde).abs().max())
            max_window_drift = max(max_window_drift, drift)
            path_minfde_max_drift = max(path_minfde_max_drift, drift)
            stat_d_repetitions.append(statistic)
            shuffle_rows.append(dict(window=f'dev_{position+1:04d}', agents=n, repetition=repetition,
                **metric_row(statistic), endpoint_sets_exact=True, endpoint_minFDE_exact=True,
                path_agent_minFDE_max_abs_drift_m=drift, singleton_reused_B=n == 1))
        stat_d = average_statistics(stat_d_repetitions)
        stats = dict(A=archived[position], B=stat_b, C=stat_c, D=stat_d)
        for branch, statistic in stats.items():
            update_accumulators(accumulators[branch], statistic)
        likelihood = float(delta_nll(goals_b['R'], cache[index]['latent_truth'].to(device)))
        assert math.isfinite(likelihood)
        assert torch.equal(cache[index]['inputs']['observed'], batch.observed[0].cpu())
        row = dict(window=f'dev_{position+1:04d}', agents=n)
        for branch, statistic in stats.items():
            row.update(metric_row(statistic, branch+'_'))
        row.update(B_delta_nll_nat_per_person=likelihood, C_delta_nll_nat_per_person=0.)
        row.update(coverage_row(goals_b, truth[0, :, -1], 'B_'))
        row.update(coverage_row(goals_c, truth[0, :, -1], 'C_'))
        row.update(covariance_diagnostics(goals_b['R'], 'B_'))
        for branch, goals, statistic in (('B', goals_b, stat_b), ('C', goals_c, stat_c)):
            endpoint_error = endpoint_minfde(goals['corrected'][0], truth[0, :, -1]).cpu()
            row[branch+'_endpoint_vs_path_agent_minFDE_max_abs_drift_m'] = float((
                endpoint_error-statistic.agent_minfde[0]).abs().max())
        for other in ('A', 'C', 'D'):
            for metric in ('JADE', 'JFDE'):
                row[f'B_minus_{other}_{metric}'] = row[f'B_{metric}']-row[f'{other}_{metric}']
        row.update(D_endpoint_sets_exact=True, D_endpoint_minFDE_exact=True,
                   D_path_agent_minFDE_max_abs_drift_m=max_window_drift)
        rows.append(row)
        _progress('FINAL_ABCD', position+1, start, rank1_epoch=rank1_epoch)
    metrics = {branch: summaries(groups) for branch, groups in accumulators.items()}
    values = [metrics['B']['all']['JADE'], metrics['B']['all']['JFDE'], metrics['B']['N_ge_9']['JFDE']]
    passes = [values[0] <= LIMITS[0], values[1] < LIMITS[1], values[2] < LIMITS[2]]
    decision = dict(values=values, thresholds=LIMITS, differences=[x-y for x, y in zip(values, LIMITS)],
                    passes=passes, all_pass=all(passes), rank1_epoch=rank1_epoch,
                    decision='ZARA2 single-fold pass only' if all(passes) else 'retain H0; prediction gates not all passed',
                    additional_seeds=0, additional_folds=0, threshold_search=False, test_access=False)
    diagnostics = dict(delta_nll_nat_per_person=dict(B=nll_summary(rows, 'B_delta_nll_nat_per_person'),
                                                    C=nll_summary(rows, 'C_delta_nll_nat_per_person')),
        coverage=dict(B=coverage_summary(rows, 'B_'), C=coverage_summary(rows, 'C_')),
        numerical_diagnostics=dict(
            B_R_diagonal_block_max_abs_error=max(row['B_R_diagonal_block_max_abs_error'] for row in rows),
            B_R_cholesky_diagonal_min=min(row['B_R_cholesky_diagonal_min'] for row in rows),
            endpoint_vs_path_agent_minFDE_max_abs_drift_m={branch: max(
                row[branch+'_endpoint_vs_path_agent_minFDE_max_abs_drift_m'] for row in rows) for branch in ('B', 'C')}),
        marginal_preservation=dict(endpoint_sets_exact_checks=endpoint_sets_checks,
            endpoint_minFDE_exact_checks=endpoint_minfde_checks,
            path_agent_minFDE_max_abs_drift_m=path_minfde_max_drift,
            explanation='D preserves corrected endpoint sets and endpoint minFDE exactly; any native decoded-path minFDE drift is recorded separately.'))
    paired = [_paired_summary(rows, group) for group in GROUPS]
    replay = dict(windows=590, dense_windows=127, rank1_epoch=rank1_epoch, B_ddim_calls=590, C_ddim_calls=590,
        D_ddim_calls=shuffled_ddim_calls, D_repetitions=5, D_window_average_before_accumulation=True,
        B_correction_calls=590, C_correction_calls=590, D_correction_calls=0, ddim_steps=20,
        A_reused_archived_statistics=True, future_read_after_B_C_predictions=True,
        C_reuses_B_personal_and_edge_noise=True, D_endpoint_sets_exact_checks=endpoint_sets_checks,
        D_endpoint_minFDE_exact_checks=endpoint_minfde_checks,
        D_path_agent_minFDE_max_abs_drift_m=path_minfde_max_drift, seconds=time.monotonic()-start)
    if rank1_epoch is not None:
        cached = read(RUN/'dev'/f'epoch_{rank1_epoch:03d}_groups.json')
        for group in GROUPS:
            assert metrics['B'][group] == cached[group], 'final rank1 B must exactly reproduce its actual epoch validation'
        replay['rank1_epoch_prediction_metrics_exact_replay'] = True
    csvwrite(RUN/'FINAL_WINDOWS.csv', rows)
    csvwrite(RUN/'D_SHUFFLE_WINDOWS.csv', shuffle_rows)
    csvwrite(RUN/'PAIRED_SUMMARY.csv', paired)
    write(RUN/'FINAL_METRICS.json', metrics)
    write(RUN/'FINAL_NLL_AND_COVERAGE.json', diagnostics)
    write(RUN/'DECISION.json', decision)
    write(RUN/'FINAL_REPLAY.json', replay)
    return decision

"""Seal fixed 19+1 trajectories and score native paired K20 and diagnostic K21."""

from __future__ import annotations

import hashlib
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from bcep.contracts import require
from bcep_formal.data import FormalData
from ebjd.metrics import hard_metric_statistics, per_world_errors
from joint_flow_fourfold.report import csvwrite, public_metadata, read_csv
from joint_flow_frozen_test.run import forbid_heldout_truth

from .common import DOC as BASE_DOC, METRICS, SEEDS, canonical_hash, digest, immutable, read, window_key, write
from .fixed_k20_center import constant_velocity
from .interaction_diagnosis.assets import ARCHIVE, read_arrays, save_npz
from .interaction_eval import _row_metrics


FOLDS = ('hotel', 'univ')
DEV = 'selection_dev'
TEST = 'heldout_confirmation'
WINDOWS = {DEV: {'hotel': 688, 'univ': 622}, TEST: {'hotel': 445, 'univ': 947}}
GROUPS = ('all', 'N1', 'N2_8', 'N9plus')
LABEL = dict(zip(GROUPS, ('全体', 'N=1', '2–8 人', 'N≥9')))
DENSE_GROUPS = ('N9_14', 'N15plus')
LABEL.update(zip(DENSE_GROUPS, ('9–14 人', 'N≥15')))
EXPLORATORY_SCOPE = 'exploratory_univ_cv_after_development_fail'
MODELS = ('base', 'union', 'final')
PHASES = ('cv_development', 'center_development', 'test')
K = 20
TOL = 1e-6


def _phase(ref, identity):
    require(ref.purpose in (DEV, TEST), 'source may only refer to development or an admitted test fold')
    if ref.purpose == TEST:
        return 'test'
    return f"{identity['method']}_development"


def _dev_gate(fold, groups):
    checks = []

    def add(kind, group, metric, upper):
        row = groups[group]
        require(row['windows'] > 0, f'gate population missing: {fold}/{group}')
        delta = row[f'final_{metric}_m'] - row[f'base_{metric}_m']
        checks.append(dict(name=kind, group=group, metric=metric, windows=row['windows'],
                           delta_m=delta, max_delta_m=upper, passed=delta <= upper))

    require(groups['all']['windows'] == WINDOWS[DEV][fold], 'changed development population')
    if fold == 'hotel':
        require(groups['N2_8']['windows'] == 440, 'changed HOTEL 2–8-person population')
        add('target', 'all', 'JFDE', -.01)
        add('joint', 'all', 'JADE', TOL)
        for metric in ('JADE', 'JFDE'):
            add('joint', 'N2_8', metric, TOL)
        marginal = ('all',)
    else:
        require(groups['N9plus']['windows'] == 61 and groups['all']['max_agents'] == 13,
                'changed UNIV dense development population')
        add('target', 'N9plus', 'JFDE', -.01)
        for metric in ('JADE', 'JFDE'):
            add('joint', 'all', metric, TOL)
        add('joint', 'N9plus', 'JADE', TOL)
        marginal = ('all', 'N9plus')
    for group in marginal:
        for metric in ('minADE', 'minFDE'):
            add('coverage', group, metric, TOL)
    return dict(passed=all(check['passed'] for check in checks), checks=checks)


def _locked_dev_receipt(run, phase):
    path = run / phase / 'SCORE_RECEIPT.json'
    result = read(path)
    expected_folds = FOLDS if phase == 'cv_development' else ('univ',)
    require(result['status'] == 'COMPLETE' and result['purpose'] == DEV
            and result['phase'] == phase and set(result['groups']) == set(expected_folds)
            and result['decision'] == {fold: _dev_gate(fold, each) for fold, each in result['groups'].items()},
            'the prior development gate differs from its sealed scores')
    require(all(digest(path.parent / name) == sha for name, sha in result['output_sha256'].items())
            and digest(path.parent / 'ALL_PREDICTIONS_SEALED.json') == result['prediction_seal_sha256']
            and digest(run / 'RUN_CONFIG.json') == result['config_sha256'],
            'the prior development receipt or configuration changed')
    return path, result


def selected_development(run):
    """Return each fold's sole final development variant and its gate provenance."""
    run = Path(run)
    cv_path, cv = _locked_dev_receipt(run, 'cv_development')
    center = run / 'center_development' / 'SCORE_RECEIPT.json'
    chosen = {fold: (cv_path, cv) for fold in FOLDS}
    if cv['decision']['univ']['passed']:
        require(not center.exists(), 'UNIV passed the CV gate before center training')
    else:
        center_path, center_result = _locked_dev_receipt(run, 'center_development')
        chosen['univ'] = center_path, center_result
    return {
        fold: dict(dev_score_phase=result['phase'], dev_score_sha256=digest(path),
                   passed=result['decision'][fold]['passed'], identity=result['identities'][fold])
        for fold, (path, result) in chosen.items()
    }


def _test_freeze(run, fold, identity):
    path = run / 'FROZEN_TEST.json'
    barrier = read(path)
    if barrier.get('test_scope') == EXPLORATORY_SCOPE:
        from .fixed_k20_inputs import DOC as DEV_ARCHIVE, RUN as DEV_RUN

        require(fold == 'univ' and barrier['folds'] == {fold: identity}
                and barrier['seeds'] == list(SEEDS)
                and barrier['config_sha256'] == digest(run / 'RUN_CONFIG.json'),
                'exploratory authorization is limited to the fixed UNIV CV')
        choice = selected_development(DEV_RUN)
        cv_path, cv = _locked_dev_receipt(DEV_RUN, 'cv_development')
        archived = read(DEV_ARCHIVE / 'cv_development' / 'SCORE_RECEIPT.json')
        require(not any(row['passed'] for row in choice.values())
                and cv['decision'] == archived['decision'] and not cv['decision'][fold]['passed']
                and identity['dev_score_phase'] == 'cv_development'
                and identity['dev_score_sha256'] == digest(cv_path),
                'exploratory test must preserve the archived development failures')
        original = cv['identities'][fold]
        require(identity['method'] == 'cv' and identity['j_drop'] == 3 and identity['S0_epoch'] == 16
                and all(identity.get(key) == original.get(key) for key in
                        ('method', 'j_drop', 'S0_epoch', 'S0_sha256', 'checkpoint_sha256',
                         'train_deletion_sha256', 'frozen_deleted_slots_sha256')),
                'UNIV exploratory identity differs from the original CV')
        return digest(path)
    selected = {key: row for key, row in selected_development(run).items() if row['passed']}
    require(barrier['config_sha256'] == digest(run / 'RUN_CONFIG.json')
            and barrier['seeds'] == list(SEEDS) and set(barrier['folds']) == set(selected),
            'test freeze does not contain exactly the eligible folds')
    require(fold in selected and canonical_hash(barrier['folds'][fold]) == canonical_hash(identity),
            'fold identity outside the prediction freeze')
    choice = selected[fold]
    require(all(identity[key] == choice[key] for key in ('dev_score_phase', 'dev_score_sha256'))
            and all(identity.get(key) == choice['identity'].get(key) for key in
                    ('method', 'j_drop', 'S0_sha256', 'checkpoint_sha256')),
            'test identity differs from the sole gated development model')
    return digest(path)


def _protocol(ref, run, identity):
    method = identity['method']
    require(ref.fold in FOLDS and method in ('cv', 'center')
            and (method == 'cv' or ref.fold == 'univ'), 'model outside locked design')
    phase = _phase(ref, identity)
    seeds = list(SEEDS) if phase == 'test' else [SEEDS[0]]
    members = [ref.split['members'][index] for index in ref.indices]
    require(identity['phase'] == phase and ref.split['fold'] == ref.fold
            and list(ref.indices) == list(ref.split[ref.purpose + '_indices'])
            and list(ref.seeds) == seeds and len(members) == len(set(ref.indices)) == WINDOWS[ref.purpose][ref.fold]
            and ref.split['dt'] == .4 and ref.split['pred_length'] == 12
            and ref.identity['S0_sha256'] == identity['S0_sha256']
            and type(identity['j_drop']) is int and 0 <= identity['j_drop'] < K,
            'changed prediction population, baseline or slot')
    if phase == 'center_development':
        _, cv = _locked_dev_receipt(run, 'cv_development')
        require(not cv['decision']['univ']['passed']
                and identity['j_drop'] == cv['identities']['univ']['j_drop'],
                'UNIV center requires failed CV and the same frozen deleted slot')
    if method == 'center':
        require(len(identity['checkpoint_sha256']) == 64, 'missing learned center weight identity')
    test_freeze = _test_freeze(run, ref.fold, identity) if phase == 'test' else None
    return dict(phase=phase, fold=ref.fold, purpose=ref.purpose, seeds=seeds,
                K=K, frames=12, dt=.4, identity=identity,
                ordered_members_sha256=canonical_hash(members), source_hashes=ref.source_hashes,
                config_sha256=digest(run / 'RUN_CONFIG.json'), test_freeze_sha256=test_freeze)


def _array_sha(paths):
    return hashlib.sha256(paths.tobytes(order='C')).hexdigest()


def _source(ref, read_base, index, ordinal, seed, export_sha):
    member = ref.split['members'][index]
    n = len(member['agent_ids'])
    key = window_key(ref.split, index)
    prior = ref.rows[index, seed]
    require(prior['n'] == n and prior['member_sha256'] == key, 'S0 agent order changed')
    baseline = read_base(index, seed)
    require(baseline.dtype == np.float32 and baseline.shape == (K, n, 12, 2)
            and np.isfinite(baseline).all(), 'invalid frozen S0 trajectories')
    return baseline, dict(seed=seed, window=f'window_{ordinal:04d}', n=n, member_sha256=key,
                          source_sha256=_array_sha(baseline), export_sha256=export_sha,
                          file=f'seed_{seed}/{ordinal:04d}.npz')


def _check_retained(base, final, j_drop):
    # Preserve the bits of all original trajectories at their original slot indices.
    for section in (slice(j_drop), slice(j_drop+1, K)):
        require(_array_sha(base[section]) == _array_sha(final[section]), 'retained S0 slot changed')


def _read_shard(directory, expected, base, j_drop):
    path = directory / expected['file']
    row = read(path.with_suffix('.json'))
    require(all(row[key] == value for key, value in expected.items()) and row['sha256'] == digest(path)
            and math.isfinite(row['candidate_seconds']) and row['candidate_seconds'] >= 0,
            'frozen prediction shard or source changed')
    final = read_arrays(path, {'paths': (K, row['n'], 12, 2)})['paths']
    _check_retained(base, final, j_drop)
    return row, final


@torch.no_grad()
def export_fold(ref, run, identity, read_base, *, predict_new=None):
    """Write observation-only K20 predictions and seal every seed/window pair."""
    run = Path(run)
    protocol = _protocol(ref, run, identity)
    use_center = identity['method'] == 'center'
    require(use_center == (predict_new is not None), 'wrong candidate predictor for chosen method')
    j_drop = identity['j_drop']
    directory = run / protocol['phase'] / ref.fold
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / 'EXPORT_CONFIG.json'
    immutable(config_path, protocol)
    export_sha = digest(config_path)
    reader = FormalData(ref.split, phase='export', verify=False)
    rows, started = [], time.monotonic()
    with forbid_heldout_truth():
        for ordinal, index in enumerate(ref.indices, 1):
            record = ref.split['records'][index]
            require(digest(record['path']) == record['sha256'], 'immutable observation record changed')
            candidate = None
            for seed in protocol['seeds']:
                base, expected = _source(ref, read_base, index, ordinal, seed, export_sha)
                path = directory / expected['file']
                side = path.with_suffix('.json')
                if side.exists():
                    row, final = _read_shard(directory, expected, base, j_drop)
                else:
                    seconds = 0.
                    if candidate is None:
                        tick = time.monotonic()
                        batch = reader.batch([index], ref.purpose, truth=False)
                        require(batch.valid.shape == (1, expected['n']) and bool(batch.valid.all())
                                and not bool(batch.future.count_nonzero()), 'invalid observed-only window')
                        output = (predict_new(batch) if use_center else
                                  constant_velocity(batch.observed[0], protocol['dt']))
                        candidate = output.detach().cpu().numpy() if isinstance(output, torch.Tensor) else output
                        require(candidate.shape == (expected['n'], 12, 2) and candidate.dtype == np.float32
                                and np.isfinite(candidate).all(), 'new candidate must be finite FP32 N×12×2')
                        seconds = time.monotonic() - tick
                    final = base.copy()
                    final[j_drop] = candidate
                    if path.exists():
                        saved = read_arrays(path, {'paths': final.shape})['paths']
                        require(_array_sha(final) == _array_sha(saved), 'orphaned prediction differs on resume')
                    else:
                        save_npz(path, paths=final)
                    row = expected | dict(sha256=digest(path), candidate_seconds=seconds)
                    immutable(side, row)
                if candidate is None:
                    candidate = final[j_drop].copy()
                require(_array_sha(candidate) == _array_sha(final[j_drop]), 'candidate depends on inference seed')
                rows.append(row)
            if ordinal % 100 == 0 or ordinal == len(ref.indices):
                progress = dict(status='EXPORTING', fold=ref.fold, phase=protocol['phase'], windows=ordinal,
                                total=len(ref.indices), seconds=time.monotonic()-started, unix=time.time())
                write(directory / 'PROGRESS.json', progress)
                print(progress, flush=True)
    core = dict(protocol=protocol, export_sha256=export_sha, rows=rows, windows=len(ref.indices),
                prediction_pairs=len(rows), truth_reads=0, new_candidate_ddim_calls=0,
                candidate_seconds=math.fsum(row['candidate_seconds'] for row in rows))
    path = directory / 'PREDICTIONS_FROZEN.json'
    if path.exists():
        seal = read(path)
        require(all(seal[key] == value for key, value in core.items()), 'prediction manifest changed')
    else:
        seal = core | dict(seal_unix=time.time(), export_wall_seconds=time.monotonic()-started)
        immutable(path, seal)
    return {name: seal[name] for name in ('windows', 'prediction_pairs', 'truth_reads',
                                           'new_candidate_ddim_calls', 'candidate_seconds',
                                           'export_wall_seconds')} | dict(seal_sha256=digest(path))


def _verify_fold(run, ref, identity, read_base):
    protocol = _protocol(ref, run, identity)
    directory = run / protocol['phase'] / ref.fold
    config = directory / 'EXPORT_CONFIG.json'
    export_sha = digest(config)
    path = directory / 'PREDICTIONS_FROZEN.json'
    seal = read(path)
    count = len(ref.indices)*len(ref.seeds)
    require(seal['protocol'] == protocol == read(config) and seal['export_sha256'] == export_sha
            and seal['windows'] == len(ref.indices) and seal['prediction_pairs'] == len(seal['rows']) == count
            and seal['truth_reads'] == seal['new_candidate_ddim_calls'] == 0, 'incomplete observation-only seal')
    rows = iter(seal['rows'])
    exploratory = identity.get('test_scope') == EXPLORATORY_SCOPE
    endpoint_checks = {
        model: dict(bitwise_equal=True, max_abs_difference_m=0.) for model in ('S0', 'new_K20')
    } if exploratory else {}
    for ordinal, index in enumerate(ref.indices, 1):
        record = ref.split['records'][index]
        require(digest(record['path']) == record['sha256'], 'input changed after prediction seal')
        candidate_sha = None
        first_endpoints = None
        for seed in ref.seeds:
            base, expected = _source(ref, read_base, index, ordinal, seed, export_sha)
            row, final = _read_shard(directory, expected, base, identity['j_drop'])
            require(row == next(rows), 'per-shard receipt differs from manifest')
            sha = _array_sha(final[identity['j_drop']])
            require(candidate_sha is None or sha == candidate_sha, 'frozen candidate changed across seeds')
            candidate_sha = sha
            if exploratory:
                current = dict(S0=base[:, :, -1], new_K20=final[:, :, -1])
                if first_endpoints is None:
                    first_endpoints = current
                else:
                    for model, endpoint in current.items():
                        reference = first_endpoints[model]
                        stats = endpoint_checks[model]
                        stats['bitwise_equal'] &= _array_sha(endpoint) == _array_sha(reference)
                        stats['max_abs_difference_m'] = max(stats['max_abs_difference_m'],
                            float(np.max(np.abs(endpoint.astype(np.float64) - reference))))
    return {name: seal[name] for name in ('windows', 'prediction_pairs', 'truth_reads',
                                           'new_candidate_ddim_calls', 'candidate_seconds',
                                           'export_wall_seconds')} | dict(seal_sha256=digest(path)) | (
        dict(endpoint_cross_seed_checks=endpoint_checks) if exploratory else {})


def _group(n):
    return 'N1' if n == 1 else 'N2_8' if n <= 8 else 'N9plus'


def _decompose(row, metric):
    base, union, final = (row[f'{model}_{metric}_m'] for model in MODELS)
    gain, deletion, delta = base-union, final-union, final-base
    require(gain >= -2e-12 and deletion >= -2e-12 and abs(delta-(deletion-gain)) <= 2e-12,
            'K21 nesting or paired decomposition violated')
    row.update({f'I_{metric}_m': gain, f'D_{metric}_m': deletion, f'delta_{metric}_m': delta})


def _score_window(base, final, j_drop, truth, valid):
    n = int(valid.sum())
    require(valid.shape == (1, n) and bool(valid.all()), 'scoring expected packed valid agents')
    union = np.concatenate([base, final[j_drop:j_drop+1]], axis=0)
    final_tensor = torch.from_numpy(final)[None]
    row = {}
    for model, paths in zip(MODELS, (torch.from_numpy(base)[None], torch.from_numpy(union)[None], final_tensor)):
        metrics = _row_metrics(hard_metric_statistics(paths, truth, valid))
        row.update({f'{model}_{key}_m': metrics[key] for key in METRICS})
        row.update({f'{model}_{key}_sum_m': metrics[f'agent_{key}_sum'] for key in ('minADE', 'minFDE')})
    errors = per_world_errors(final_tensor, truth, valid)
    for metric, error in zip(('JADE', 'JFDE'), errors):
        world = (error*valid[..., None]).sum(1)/n
        require(float(world.min()) == row[f'final_{metric}_m'], 'winner computation differs from native scores')
        row[f'new_wins_{metric}'] = int(int(world.argmin()) == j_drop)
    for metric in METRICS:
        _decompose(row, metric)
    return row


def _summarize(fold, seed, group, rows):
    selected = [row for row in rows if group == 'all' or row['group'] == group or row.get('dense_group') == group]
    windows, agents = len(selected), sum(row['n'] for row in selected)
    result = dict(fold=fold, seed=seed, group=group, windows=windows, agents=agents,
                  max_agents=max((row['n'] for row in selected), default=0))
    for metric in METRICS:
        for model in MODELS:
            key = f'{model}_{metric}_m'
            if metric in ('minADE', 'minFDE'):
                result[key] = math.fsum(row[f'{model}_{metric}_sum_m'] for row in selected)/agents if agents else None
            else:
                result[key] = math.fsum(row[key] for row in selected)/windows if windows else None
        if windows:
            _decompose(result, metric)
        else:
            result.update({f'{key}_{metric}_m': None for key in ('I', 'D', 'delta')})
        deltas = [row[f'delta_{metric}_m'] for row in selected]
        for name, count in (('improved', sum(value < 0 for value in deltas)),
                            ('worsened', sum(value > 0 for value in deltas)),
                            ('unchanged', sum(value == 0 for value in deltas))):
            result[f'{name}_{metric}_windows'] = count
    for metric in ('JADE', 'JFDE'):
        count = sum(row[f'new_wins_{metric}'] for row in selected)
        result[f'new_wins_{metric}_windows'] = count
        result[f'new_wins_{metric}_fraction'] = count/windows if windows else None
    return result


def _mean_seeds(rows):
    first = rows[0]
    fixed = ('fold', 'group', 'windows', 'agents', 'max_agents')
    require(all(tuple(row[key] for key in fixed) == tuple(first[key] for key in fixed) for row in rows),
            'agent or window population changed across seeds')
    result = {key: first[key] for key in fixed}
    result['seeds'] = len(rows)
    for key in first:
        if key in (*fixed, 'seed'):
            continue
        values = [row[key] for row in rows]
        result[key] = statistics.fmean(values) if values[0] is not None else None
        if key.startswith('delta_') and key.endswith('_m'):
            result[f'{key}_sample_sd'] = (statistics.stdev(values)
                                         if values[0] is not None and len(rows) > 1 else None)
    return result


def _score_fold(run, phase, ref, identity, read_base, archived):
    directory = run / phase / ref.fold
    export_sha = digest(directory / 'EXPORT_CONFIG.json')
    reader = FormalData(ref.split, phase='scoring', verify=False)
    rows = {seed: [] for seed in ref.seeds}
    exploratory = identity.get('test_scope') == EXPLORATORY_SCOPE
    groups_to_score = GROUPS + DENSE_GROUPS if exploratory else GROUPS
    replay = 0.
    for ordinal, index in enumerate(ref.indices, 1):
        batch = reader.batch([index], ref.purpose, truth=True)
        require(bool(batch.valid.all()) and bool(torch.isfinite(batch.future).all()), 'invalid paired target')
        for seed in ref.seeds:
            base, expected = _source(ref, read_base, index, ordinal, seed, export_sha)
            _, final = _read_shard(directory, expected, base, identity['j_drop'])
            score = _score_window(base, final, identity['j_drop'], batch.future, batch.valid)
            n = expected['n']
            if ref.purpose == DEV:
                prior = archived[ref.fold, expected['window']]
                require(int(prior['n']) == n, 'archived S0 window population differs')
                replay = max(replay, *(abs(score[f'base_{metric}_m']-float(prior[f'S0_{metric}']))
                                       for metric in METRICS))
                require(replay <= TOL, 'development S0 trajectory differs from archived metrics')
            row = dict(fold=ref.fold, split=ref.purpose, method=identity['method'],
                       seed=seed, window=expected['window'], n=n, j_drop=identity['j_drop'],
                       group=_group(n), **score)
            if exploratory:
                row['dense_group'] = None if n < 9 else 'N9_14' if n <= 14 else 'N15plus'
            rows[seed].append(row)
    by_seed = []
    for seed, windows in rows.items():
        for group in groups_to_score:
            result = _summarize(ref.fold, seed, group, windows)
            by_seed.append(result)
            if ref.purpose == TEST and group in ('all', 'N1', 'N9plus') and result['windows']:
                old_group = {'all': 'all', 'N1': 'single', 'N9plus': 'N_ge_9'}[group]
                prior = archived[ref.fold, seed, old_group]
                require((int(prior['scene_count']), int(prior['agent_count'])) == (result['windows'], result['agents']),
                        'archived test population differs')
                replay = max(replay, *(abs(result[f'base_{metric}_m']-float(prior[f'new_{metric}']))
                                       for metric in METRICS))
                require(replay <= TOL, 'test S0 baseline failed its archived replay')
        if exploratory:
            counts = {row['group']: (row['windows'], row['agents']) for row in by_seed if row['seed'] == seed}
            require(counts['all'] == (947, 24334) and counts['N9plus'] == (907, 24099)
                    and tuple(map(sum, zip(counts['N9_14'], counts['N15plus']))) == counts['N9plus'],
                    'exploratory test population differs from the fixed protocol')
    summaries = {group: _mean_seeds([row for row in by_seed if row['group'] == group]) for group in groups_to_score}
    return [row for part in rows.values() for row in part], by_seed, summaries, replay


@torch.no_grad()
def score_phase(run, entries, *, phase, doc=None, development_folds=None):
    """Validate every admitted prediction and seal the whole phase before loading future targets."""
    run = Path(run)
    require(phase in PHASES and entries, 'invalid evaluation phase')
    if phase == 'test':
        require(development_folds is None, 'development subset only applies to development scoring')
        require(set(entries) == set(read(run / 'FROZEN_TEST.json')['folds']), 'missing or ineligible test fold')
    else:
        expected = FOLDS if phase == 'cv_development' else ('univ',)
        subset = expected if development_folds is None else tuple(development_folds)
        require(set(entries) == set(subset) and set(subset) <= set(expected),
                'incorrect development folds')
    output = run / phase
    started = time.monotonic()
    seals = {}
    for fold, (ref, identity, read_base) in entries.items():
        require(fold == ref.fold and _phase(ref, identity) == phase, 'mismatched evaluation entry')
        seals[fold] = _verify_fold(run, ref, identity, read_base)
    global_path = output / 'ALL_PREDICTIONS_SEALED.json'
    core = dict(phase=phase, purpose=TEST if phase == 'test' else DEV, K=K, diagnostic_K=K+1,
                truth_reads=0, folds=seals, config_sha256=digest(run / 'RUN_CONFIG.json'))
    if global_path.exists():
        seal = read(global_path)
        require(all(seal[key] == value for key, value in core.items()), 'global seal changed')
    else:
        seal = core | dict(seal_unix=time.time())
        immutable(global_path, seal)
    target = output / 'SCORE_RECEIPT.json'
    if target.exists():
        result = read(target)
        require(result['status'] == 'COMPLETE' and result['prediction_seal_sha256'] == digest(global_path)
                and all(digest(output / name) == sha for name, sha in result['output_sha256'].items())
                and result['first_truth_read_unix'] >= seal['seal_unix'], 'scored artifacts changed')
        return result
    if phase == 'test':
        archived = {(row['fold'], int(row['seed']), row['group']): row
                    for row in read_csv(BASE_DOC / 'PER_SEED.csv') if row['fold'] in entries}
    else:
        archived = {(row['fold'], row['window']): row
                    for row in read_csv(ARCHIVE / 'dev' / 'PER_WINDOW.csv') if row['fold'] in entries}
    first_truth = time.time()
    require(first_truth >= seal['seal_unix'], 'future read precedes global prediction seal')
    all_rows, seed_rows, groups, replay = [], [], {}, {}
    for fold, (ref, identity, read_base) in entries.items():
        per_window, per_seed, groups[fold], replay[fold] = _score_fold(
            run, phase, ref, identity, read_base, archived)
        all_rows.extend(per_window)
        seed_rows.extend(per_seed)
    if phase == 'cv_development' and 'hotel' in groups:
        expected = groups['hotel']['all']['I_JFDE_m']
        require(abs(expected - .013861) <= 1e-6, 'HOTEL K21 CV potential differs from the reference')
    decision = ({fold: _dev_gate(fold, summary) for fold, summary in groups.items()}
                if phase != 'test' else {})
    tables = {'GROUPS.csv': [row for summary in groups.values() for row in summary.values()],
              'PER_SEED.csv': seed_rows, 'PER_WINDOW.csv': all_rows}
    for name, rows in tables.items():
        csvwrite(output / name, rows)
    receipt = dict(status='COMPLETE', phase=phase, purpose=core['purpose'], K=K, diagnostic_K=K+1,
                   config_sha256=core['config_sha256'], decision=decision, groups=groups,
                   identities={fold: identity for fold, (_, identity, _) in entries.items()},
                   prediction_seal_sha256=digest(global_path), prediction_seal_unix=seal['seal_unix'],
                   first_truth_read_unix=first_truth, prediction_pairs=len(all_rows), native_replay_max_abs_m=replay,
                   scoring_seconds=time.monotonic()-started,
                   output_sha256={name: digest(output / name) for name in tables})
    immutable(target, receipt)
    if doc is not None:
        build_report(run, doc)
    return receipt


def _fmt(value, precision=4, *, sign=False):
    if value is None:
        return '—'
    return format(value, f"{'+' if sign else ''}.{precision}f")


def _tables(groups, test):
    main = [
        '| 折 | 分组（窗／人次） | 指标 | S0 K20（m） | 新 K20（m） | 配对 Δ（cm） | 改善／受损窗数 | 新场景胜率 |',
        '|---|---|---|---:|---:|---:|---:|---:|',
    ]
    diagnostic = [
        '| 折 | 分组 | 指标 | K21（m） | 潜在收益 I（cm） | 删槽代价 D（cm） | 正式 Δ（cm） |',
        '|---|---|---|---:|---:|---:|---:|',
    ]
    for fold, per_group in groups.items():
        for group, row in per_group.items():
            label = f"{LABEL[group]}（{row['windows']}／{row['agents']}）"
            for metric in METRICS:
                delta, sd = row[f'delta_{metric}_m'], row[f'delta_{metric}_m_sample_sd']
                delta_cm = _fmt(delta*100 if delta is not None else None, sign=True)
                if test and sd is not None:
                    delta_cm += f" ± {sd*100:.4f}"
                wins = row.get(f'new_wins_{metric}_fraction')
                counts = '／'.join(f"{row[f'{name}_{metric}_windows']:g}" for name in ('improved', 'worsened'))
                main.append(
                    f"| {fold.upper()} | {label} | {metric} | {_fmt(row[f'base_{metric}_m'], 9)} | "
                    f"{_fmt(row[f'final_{metric}_m'], 9)} | {delta_cm} | {counts} | "
                    f"{f'{wins:.2%}' if wins is not None else '—'} |")
                i, d = row[f'I_{metric}_m'], row[f'D_{metric}_m']
                diagnostic.append(
                    f"| {fold.upper()} | {LABEL[group]} | {metric} | {_fmt(row[f'union_{metric}_m'], 9)} | "
                    f"{_fmt(i*100 if i is not None else None)} | {_fmt(d*100 if d is not None else None)} | "
                    f"{_fmt(delta*100 if delta is not None else None, sign=True)} |")
    return '\n'.join(main), '\n'.join(diagnostic)


def _archive_phase(run, doc, phase, receipt):
    source, dest = run / phase, doc / phase
    dest.mkdir(parents=True, exist_ok=True)
    for name, sha in receipt['output_sha256'].items():
        require(digest(source / name) == sha, 'scored table changed before archive')
        (dest / name).write_bytes((source / name).read_bytes())
    for name in ('SCORE_RECEIPT.json', 'ALL_PREDICTIONS_SEALED.json'):
        write(dest / name, public_metadata(read(source / name)))


def build_report(run, doc):
    """Publish anonymous fold tables with the fixed independent test eligibility decision."""
    run, doc = Path(run), Path(doc)
    chosen = selected_development(run)
    receipts = {phase: read(run / phase / 'SCORE_RECEIPT.json') for phase in PHASES
                if (run / phase / 'SCORE_RECEIPT.json').exists()}
    eligible = {fold for fold, row in chosen.items() if row['passed']}
    require(set(receipts.get('test', {}).get('groups', {})) == eligible, 'test archive has wrong admitted folds')
    for phase, receipt in receipts.items():
        _archive_phase(run, doc, phase, receipt)

    def lead(fold):
        decision = chosen[fold]
        receipt = receipts[decision['dev_score_phase']]
        group = 'all' if fold == 'hotel' else 'N9plus'
        values = receipt['groups'][fold][group]
        check = receipt['decision'][fold]
        covers = all(row['passed'] for row in check['checks'] if row['name'] == 'coverage')
        i, d = values['I_JFDE_m']*100, values['D_JFDE_m']*100
        source = '新增候选的收益较大' if i > d else '删槽代价较大' if d > i else '两项相抵'
        method = '匀速' if decision['identity']['method'] == 'cv' else '中心轨迹'
        return (f"{fold.upper()} {LABEL[group]}采用{method}的 JFDE 变化为 {values['delta_JFDE_m']*100:+.4f} cm"
                f"、个人覆盖{'达标' if covers else '未达标'}，潜在收益 {i:.4f} cm、删槽代价 {d:.4f} cm，{source}")

    lines = ['# HOTEL 与 UNIV 固定 K20 的 19+1 实验',
             lead('hotel') + '，' + lead('univ') + '。',
             '首页配对差使用原开发集、seed 2035 与完整预测窗。负差表示误差下降。'
             '两折独立验收，通过的折进入五种子探索性冻结复核。']
    if eligible:
        tested = receipts['test']['groups']
        text = []
        for fold in FOLDS:
            if fold in eligible:
                group = 'all' if fold == 'hotel' else 'N9plus'
                row = tested[fold][group]
                text.append(f"{fold.upper()} {LABEL[group]} JFDE Δ "
                            f"{row['delta_JFDE_m']*100:+.4f} ± {row['delta_JFDE_m_sample_sd']*100:.4f} cm")
        lines.append('冻结测试结果为 ' + '，'.join(text) + '。')
    for fold, choice in chosen.items():
        ident = choice['identity']
        action = '完成冻结测试' if choice['passed'] else '保留 S0，止于开发集'
        lines.append(f"{fold.upper()} 固定删槽采用零基原槽序号 {ident['j_drop']}，{action}。")

    lines += [
        '正式指标使用原生 K20。原生 S0 先计算全部 20 条联合场景，再在原槽号替换一条。'
        '本次优先复用该完整缓存，19 条保留轨迹和原噪声键保持逐槽对应。'
        '匀速与中心轨迹直接生成第 20 个场景，各自新增 DDIM 次数均为 0。',
        '联合指标按窗等权，边缘指标按有效人次加权。新场景胜率按原生最小值位置计算，'
        '同分时采用零基槽号最小者。改善和受损窗数按未舍入逐窗差值统计。'
        '测试表中窗数和人次为单个种子的分母，计数列为五种子平均，Δ 后的 ± 为配对样本标准差。',
        'K21 诊断对完整 S0 追加同一新场景。潜在收益 I＝S0 K20−K21，删槽代价 D＝正式 K20−K21，'
        '配对净变化 Δ＝D−I。K21 仅用于诊断。开发验收分母使用全部适用窗。',
    ]
    labels = {'cv_development': '匀速开发集', 'center_development': 'UNIV 中心轨迹第 12 轮开发集',
              'test': '五种子探索性冻结测试'}
    for phase, receipt in receipts.items():
        main, diagnostic = _tables(receipt['groups'], phase == 'test')
        lines.extend([f"## {labels[phase]}正式 K20", main])
        if phase != 'test':
            gate = ['| 折 | 分组（窗） | 指标 | Δ（m） | 允许上限（m） | 通过 |',
                    '|---|---|---|---:|---:|---|']
            for fold, decision in receipt['decision'].items():
                for row in decision['checks']:
                    gate.append(f"| {fold.upper()} | {LABEL[row['group']]}（{row['windows']}） | {row['metric']} | "
                                f"{row['delta_m']:+.10g} | {row['max_delta_m']:+.10g} | "
                                f"{'是' if row['passed'] else '否'} |")
            lines += ['### 固定门槛', '\n'.join(gate)]
        lines += ['### K21 与删槽分解', diagnostic]
        seal = read(run / phase / 'ALL_PREDICTIONS_SEALED.json')
        for fold, info in seal['folds'].items():
            lines.append(f"{fold.upper()} 封存 {info['prediction_pairs']} 个窗种子对，"
                         f"新增候选生成 {info['candidate_seconds']:.2f} 秒，"
                         f"导出墙钟 {info['export_wall_seconds']:.2f} 秒。")
        lines.append(f"先封存全阶段预测再评分，评分及核验耗时 {receipt['scoring_seconds']:.2f} 秒。"
                     f"S0 指标归档回放最大绝对差为 {max(receipt['native_replay_max_abs_m'].values()):.3g} m。")
        lines += [f"[分组指标]({phase}/GROUPS.csv) · [逐种子]({phase}/PER_SEED.csv) · "
                  f"[匿名逐窗]({phase}/PER_WINDOW.csv) · [评分收据]({phase}/SCORE_RECEIPT.json) · "
                  f"[完整阶段封存]({phase}/ALL_PREDICTIONS_SEALED.json)"]
    lines += ['## 复核范围',
              '历史测试观察已启发本方案，因此通过后的五种子结果属于探索性冻结复核。'
              'UNIV 原开发集单窗最多 13 人，更大窗口的指标以实际冻结测试的结果为准。']
    if not eligible:
        lines.append('两折均止于开发集，评分范围涵盖 688 个 HOTEL 和 622 个 UNIV 原开发窗。')
    wall = run / 'WALL_TIME.json'
    if wall.is_file() and 'seconds' in read(wall):
        lines.append(f"实验总墙钟耗时 {read(wall)['seconds']:.2f} 秒。")
    lines.append('[固定配置](RUN_CONFIG.json)')
    (doc / 'README.md').write_text('\n\n'.join(lines) + '\n', encoding='utf-8')

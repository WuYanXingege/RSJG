"""Paired endpoint-interaction evaluation with a heldout gate and prediction seals."""

import math
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from bcep.contracts import require
from bcep_formal.data import FormalData
from ebjd.metrics import MetricAccumulator, hard_metric_statistics
from joint_flow_fourfold.report import csvwrite, public_metadata, read_csv
from joint_flow_frozen_test.run import forbid_heldout_truth

from .common import (
    ATOL, DOC as BASE_DOC, METRICS, RTOL, RUN as BASE_RUN, SEEDS, SOURCE,
    canonical_hash, digest, immutable, read, window_key, write,
)
from .evaluate import _read_prediction
from .model import decode_student, encode_observed, student_endpoints, student_noise


FOLDS = ('hotel', 'univ')
MODELS = ('S0', 'R12')
DEV = 'selection_dev'
TEST = 'heldout_confirmation'
TARGET_GROUP = {'hotel': 'N2_8', 'univ': 'N9plus'}
GROUPS = ('all', 'N1', 'N2_8', 'N9plus')
GROUP_LABELS = {'all': '全体', 'N1': 'N=1', 'N2_8': '2≤N≤8', 'N9plus': 'N≥9'}
TERMS = METRICS + ('L', 'G')
EPSILON_M = 1e-6
PROTOCOL = dict(K=20, ddim_steps=20, eta=0., prediction_frames=12)


def _output(root, purpose):
    require(purpose in (DEV, TEST), 'invalid paired split')
    return Path(root) / ('dev_evaluation' if purpose == DEV else 'test_evaluation')


def _gate(root, fold, identity, purpose):
    if purpose == DEV:
        return None
    freeze_path = Path(root) / 'FROZEN_R12_TEST.json'
    dev_path = _output(root, DEV) / 'SCORE_RECEIPT.json'
    require(freeze_path.is_file() and dev_path.is_file(), 'test awaits both development gates')
    freeze, dev = read(freeze_path), read(dev_path)
    require(dev['status'] == 'COMPLETE' and dev['purpose'] == DEV
            and dev['decision'] == _decision(dev['groups']) and dev['decision']['test_authorized']
            and freeze['status'] == 'BOTH_DEV_GATES_PASSED'
            and freeze['dev_score_sha256'] == digest(dev_path)
            and freeze['config_sha256'] == digest(Path(root) / 'RUN_CONFIG.json')
            and freeze['seeds'] == list(SEEDS) and set(freeze['folds']) == set(FOLDS),
            'R12 test gate or fixed protocol did not match')
    for name in ('S0_sha256', 'R12_sha256'):
        require(freeze['folds'][fold][name] == identity[name], 'frozen R12 or S0 hash differs')
    require(freeze['source_sha256'] and all(
        digest(SOURCE / filename) == sha for filename, sha in freeze['source_sha256'].items()),
        'frozen test sources changed')
    return digest(freeze_path)


def _fold_protocol(split, identity, purpose, barrier):
    fold = split['fold']
    indices = split[purpose + '_indices']
    require(fold in FOLDS and bool(indices) and len(indices) == len(set(indices)),
            'invalid paired split membership')
    members = [split['members'][index] for index in indices]
    member_hash = canonical_hash(members)
    if purpose == DEV:
        require(identity['dev_manifest_sha256'] == member_hash
                and identity['dev_windows'] == len(indices)
                and identity['dev_agents'] == sum(len(member['agent_ids']) for member in members),
                'selection_dev population changed')
    return indices, dict(fold=fold, purpose=purpose, identity_sha256=canonical_hash(identity),
        ordered_members_sha256=member_hash, seeds=[SEEDS[0]] if purpose == DEV else list(SEEDS),
        freeze_barrier_sha256=barrier, **PROTOCOL)


def _original_s0(split, indices, identity, protocol):
    """Resolve a heldout S0 cache only after the two-fold gate has passed."""
    fold = split['fold']
    for name in ('RUN_CONFIG.json', 'ALL_PREDICTIONS_SEALED.json', 'SCORE_RECEIPT.json', 'PER_SEED.csv'):
        require(digest(BASE_RUN / name) == digest(BASE_DOC / name), f'archived S0 {name} changed')
    original_config = read(BASE_DOC / 'RUN_CONFIG.json')
    original_seal = read(BASE_DOC / 'ALL_PREDICTIONS_SEALED.json')
    original_score = read(BASE_DOC / 'SCORE_RECEIPT.json')
    original_freeze_path = BASE_RUN / 'FROZEN_STUDENTS.json'
    original_freeze = read(original_freeze_path)
    saved_fold = original_freeze['folds'][fold]
    require(original_seal['freeze_barrier_sha256'] == digest(original_freeze_path)
            and original_freeze['config_sha256'] == digest(BASE_RUN / 'RUN_CONFIG.json')
            and original_score['global_prediction_seal_sha256'] == digest(BASE_RUN / 'ALL_PREDICTIONS_SEALED.json')
            and original_score['output_sha256']['PER_SEED.csv'] == digest(BASE_RUN / 'PER_SEED.csv')
            and saved_fold['selected_sha256'] == identity['S0_sha256']
            and saved_fold['selected_epoch'] == identity['S0_epoch']
            and original_freeze['seeds'] == protocol['seeds'],
            'S0 test reference failed identity checks')
    require(all(digest(SOURCE / filename) == expected
                for filename, expected in original_config['source_sha256'].items()),
            'original S0 inference sources changed')
    folder = BASE_RUN / fold / 'test'
    path = folder / 'PREDICTIONS_FROZEN.json'
    if not path.exists():
        return folder, {}, None
    sha = digest(path)
    require(sha == original_seal['folds'][fold]['sha256'], 'cached S0 prediction manifest changed')
    seal = read(path)
    expected = dict(fold=fold, selected_sha256=identity['S0_sha256'], selected_epoch=identity['S0_epoch'],
        ordered_members_sha256=protocol['ordered_members_sha256'],
        freeze_barrier_sha256=digest(original_freeze_path), seeds=protocol['seeds'], **PROTOCOL)
    require(seal['identity'] == expected and seal['truth_reads'] == 0
            and seal['ddim_calls'] == len(indices) * len(SEEDS),
            'cached S0 export protocol differs')
    cached = {(row['index'], row['seed']): row for row in seal['rows']}
    require(len(cached) == len(seal['rows']) == len(indices) * len(SEEDS),
            'duplicate or missing S0 cache row')
    for ordinal, index in enumerate(indices, 1):
        member = split['members'][index]
        for seed in SEEDS:
            row = cached[index, seed]
            require(row['window'] == f'test_{ordinal:04d}'
                    and row['file'] == f'seed_{seed}/{ordinal:04d}.npz'
                    and row['n'] == len(member['agent_ids'])
                    and row['member_sha256'] == window_key(split, index)
                    and row['export_identity_sha256'] == canonical_hash(expected),
                    'cached S0 seed/window/agent order mismatch')
    return folder, cached, sha


def _row(split, protocol, ordinal, index, seed):
    n = len(split['members'][index]['agent_ids'])
    return dict(seed=seed, window=f'window_{ordinal:04d}', file=f'seed_{seed}/{ordinal:04d}.npz',
                n=n, member_sha256=window_key(split, index), protocol_sha256=canonical_hash(protocol))


def _read_pair(directory, row, expected, cache_path, cache_row):
    require(all(row[key] == value for key, value in expected.items()), 'paired prediction identity changed')
    require(all(math.isfinite(row['inference_seconds'][model]) and row['inference_seconds'][model] >= 0
                for model in MODELS), 'invalid paired inference times')
    file = directory / row['file']
    require(digest(file) == row['sha256'], 'sealed R12/S0 prediction changed')
    with np.load(file, allow_pickle=False) as archive:
        fields = {'R12'} if row['S0_reused'] else set(MODELS)
        require(set(archive.files) == fields, 'invalid prediction fields')
        arrays = {model: archive[model] for model in fields}
    if row['S0_reused']:
        prior_path = cache_path / cache_row['file']
        require(cache_row['sha256'] == row['S0_sha256'] == digest(prior_path), 'historical S0 shard changed')
        arrays['S0'] = _read_prediction(prior_path, row['n'])
    else:
        require(row['S0_sha256'] is None, 'unexpected historical S0 identity')
    require(all(array.shape == (20, row['n'], 12, 2) and array.dtype == np.float32
                and np.isfinite(array).all() for array in arrays.values()), 'invalid paired K20 trajectory')
    return arrays


def _synchronize(tensor):
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def _prepare_window(assets, student, interaction, batch):
    observed, maps, valid = batch.observed[0], batch.semantic_maps[0], batch.valid[0]
    _synchronize(observed)
    tick = time.perf_counter()
    context, features = encode_observed(assets.provider, batch)
    g0 = student_endpoints(student, assets.provider, observed, maps, valid, context, features)
    bridge_context = assets.model.encode_context(batch.observed, batch.semantic_maps, batch.valid)
    _synchronize(observed)
    common_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    g1 = interaction(g0, observed, context, valid, dt=float(assets.split['dt']))
    _synchronize(observed)
    return bridge_context, {'S0': g0, 'R12': g1}, common_seconds, time.perf_counter() - tick


def _save(file, arrays):
    temporary = file.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(file)


def _progress(directory, phase, fold, done, total, seconds):
    row = dict(status=phase, fold=fold, windows=done, total=total, seconds=seconds, unix=time.time())
    write(directory / 'PROGRESS.json', row)
    print(row, flush=True)


@torch.no_grad()
def export_fold(assets, student, interaction, root, identity, purpose=DEV, should_yield=None):
    """Use observations only and seal actual K20 trajectories before scoring."""
    fold = assets.fold
    barrier = _gate(root, fold, identity, purpose)
    directory = _output(root, purpose) / fold
    directory.mkdir(parents=True, exist_ok=True)
    split = assets.split
    indices, protocol = _fold_protocol(split, identity, purpose, barrier)
    cache_path, cache, cache_sha = (
        _original_s0(split, indices, identity, protocol) if purpose == TEST else (None, {}, None))
    protocol['historical_S0_seal_sha256'] = cache_sha
    immutable(directory / 'EXPORT_CONFIG.json', protocol)
    for model in (student, interaction):
        model.eval().requires_grad_(False)
    reader = FormalData(split, phase='export', verify=False)
    rows = []
    started = time.monotonic()
    with forbid_heldout_truth():
        for ordinal, index in enumerate(indices, 1):
            if should_yield is not None and should_yield():
                return None
            record = split['records'][index]
            require(digest(record['path']) == record['sha256'], 'frozen observation record changed')
            prepared = None
            for seed in protocol['seeds']:
                expected = _row(split, protocol, ordinal, index, seed)
                file = directory / expected['file']
                file.parent.mkdir(exist_ok=True)
                side = file.with_suffix('.json')
                cache_row = cache.get((index, seed))
                if side.exists():
                    row = read(side)
                    _read_pair(directory, row, expected, cache_path, cache_row)
                else:
                    if prepared is None:
                        batch = reader.batch([index], purpose, truth=False).to(assets.device)
                        require(batch.valid.shape == (1, expected['n']) and bool(batch.valid.all())
                                and not batch.future.count_nonzero(), 'paired export read invalid observations')
                        prepared = _prepare_window(assets, student, interaction, batch)
                    bridge_context, goals, common_seconds, new_seconds = prepared
                    noise = student_noise(expected['member_sha256'], batch.agent_ids[0], seed, goals['S0'].device)
                    use_cache = False
                    if cache_row is not None:
                        prior_path = cache_path / cache_row['file']
                        use_cache = prior_path.is_file() and digest(prior_path) == cache_row['sha256']
                        if use_cache:
                            old = _read_prediction(prior_path, expected['n'])
                            np.testing.assert_allclose(old[..., -1, :], goals['S0'].cpu().numpy(),
                                                       atol=ATOL, rtol=RTOL)
                    predictions, timing = {}, {}
                    for model in MODELS:
                        if model == 'S0' and use_cache:
                            timing[model] = cache_row['inference_seconds']
                            continue
                        _synchronize(goals[model])
                        tick = time.perf_counter()
                        pred = decode_student(assets.model, bridge_context, goals[model], noise)
                        torch.testing.assert_close(pred[..., -1, :], goals[model][None], atol=ATOL, rtol=RTOL)
                        _synchronize(pred)
                        timing[model] = time.perf_counter() - tick + common_seconds
                        if model == 'R12':
                            timing[model] += new_seconds
                        predictions[model] = pred[0].cpu().numpy()
                    if file.exists():
                        with np.load(file, allow_pickle=False) as orphan:
                            require(set(orphan.files) == set(predictions) and all(
                                np.array_equal(orphan[model], values) for model, values in predictions.items()),
                                'orphan paired shard differs on deterministic replay')
                    else:
                        _save(file, predictions)
                    row = dict(expected, sha256=digest(file), S0_reused=use_cache,
                               S0_sha256=cache_row['sha256'] if use_cache else None,
                               inference_seconds=timing)
                    _read_pair(directory, row, expected, cache_path, cache_row)
                    immutable(side, row)
                rows.append(row)
            if ordinal % 25 == 0 or ordinal == len(indices):
                _progress(directory, 'EXPORTING', fold, ordinal, len(indices), time.monotonic()-started)
    seal_path = directory / 'PREDICTIONS_FROZEN.json'
    reused = sum(row['S0_reused'] for row in rows)
    core = dict(protocol=protocol, rows=rows, truth_reads=0, prediction_pairs=len(rows), windows=len(indices),
                ddim_calls=dict(S0=len(rows)-reused, R12=len(rows)), S0_reused=reused,
                inference_seconds={model: math.fsum(row['inference_seconds'][model] for row in rows)
                                   for model in MODELS})
    if seal_path.exists():
        prior = read(seal_path)
        require(all(prior[key] == value for key, value in core.items()), 'paired seal changed on resume')
        seal = prior
    else:
        seal = dict(core, seal_unix=time.time(), export_worker_seconds=time.monotonic()-started)
        immutable(seal_path, seal)
    return {key: seal[key] for key in ('windows', 'prediction_pairs', 'ddim_calls', 'S0_reused',
                                       'inference_seconds', 'export_worker_seconds')} | dict(
        seal_sha256=digest(seal_path))


def _verify_fold(split, root, identity, purpose):
    fold = split['fold']
    barrier = _gate(root, fold, identity, purpose)
    directory = _output(root, purpose) / fold
    indices, protocol = _fold_protocol(split, identity, purpose, barrier)
    cache_path, cache, cache_sha = (
        _original_s0(split, indices, identity, protocol) if purpose == TEST else (None, {}, None))
    protocol['historical_S0_seal_sha256'] = cache_sha
    seal_path = directory / 'PREDICTIONS_FROZEN.json'
    seal = read(seal_path)
    pairs = len(indices) * len(protocol['seeds'])
    require(seal['protocol'] == read(directory / 'EXPORT_CONFIG.json') == protocol
            and seal['truth_reads'] == 0 and seal['prediction_pairs'] == len(seal['rows']) == pairs
            and seal['ddim_calls'] == dict(S0=pairs-seal['S0_reused'], R12=pairs)
            and seal['S0_reused'] == sum(row['S0_reused'] for row in seal['rows'])
            and seal['windows'] == len(indices), 'incomplete paired seal')
    rows = {}
    values = iter(seal['rows'])
    for ordinal, index in enumerate(indices, 1):
        record = split['records'][index]
        require(digest(record['path']) == record['sha256'], 'scoring record changed')
        endpoints = None
        for seed in protocol['seeds']:
            row = next(values)
            arrays = _read_pair(directory, row, _row(split, protocol, ordinal, index, seed),
                                cache_path, cache.get((index, seed)))
            current = arrays['R12'][..., -1, :]
            if endpoints is None:
                endpoints = current.copy()
            require(np.allclose(current, endpoints, atol=ATOL, rtol=RTOL), 'R12 endpoints vary across seeds')
            rows[index, seed] = row
    return indices, protocol['seeds'], rows, cache_path, cache, dict(
        seal_sha256=digest(seal_path), windows=len(indices), prediction_pairs=pairs,
        ddim_calls=seal['ddim_calls'], S0_reused=seal['S0_reused'],
        inference_seconds=seal['inference_seconds'], export_worker_seconds=seal['export_worker_seconds'])


def _group(n):
    if n == 1:
        return 'N1'
    return 'N2_8' if n <= 8 else 'N9plus'


def _row_metrics(stats):
    n = int(stats.agent_valid.sum())
    ade = float(stats.agent_minade[stats.agent_valid].double().sum())
    fde = float(stats.agent_minfde[stats.agent_valid].double().sum())
    L, JFDE = float(stats.scene_marginal_fde.item()), float(stats.scene_jfde.item())
    return dict(JADE=float(stats.scene_jade.item()), JFDE=JFDE, minADE=ade/n, minFDE=fde/n,
                L=L, G=JFDE-L, agent_minADE_sum=ade, agent_minFDE_sum=fde)


def _group_row(fold, seed, group, accumulators):
    base, current = (accumulators[model] for model in MODELS)
    windows, agents = base.scene_count, base.agent_count
    require((windows, agents) == (current.scene_count, current.agent_count), 'paired group mismatch')
    row = dict(fold=fold, seed=seed, group=group, windows=windows, agents=agents)
    for model, acc in accumulators.items():
        metrics = dict.fromkeys(TERMS)
        if windows:
            native = acc.as_dict()
            metrics = {key: native[key] for key in METRICS}
            metrics['L'] = native['scene_weighted_minFDE']
            metrics['G'] = metrics['JFDE'] - metrics['L']
        row.update({f'{model}_{metric}_m': metrics[metric] for metric in TERMS})
    for metric in TERMS:
        row[f'delta_{metric}_cm'] = (
            100*(row[f'R12_{metric}_m'] - row[f'S0_{metric}_m']) if windows else None)
    if windows:
        require(abs(row['delta_JFDE_cm']-row['delta_L_cm']-row['delta_G_cm']) <= 1e-11,
                'window-weighted L + G decomposition changed')
    return row


def _decision(groups):
    checks = {}
    for fold in FOLDS:
        full, target = groups[fold]['all'], groups[fold][TARGET_GROUP[fold]]
        tests = {}
        for metric in METRICS:
            delta = full[f'R12_{metric}_m'] - full[f'S0_{metric}_m']
            passes = delta < -EPSILON_M if metric in ('JADE', 'JFDE') else delta <= EPSILON_M
            tests[f'all_{metric}'] = dict(delta_m=delta, passed=passes)
        delta = (target['R12_JFDE_m'] - target['S0_JFDE_m']) if target['windows'] else None
        tests[TARGET_GROUP[fold]+'_JFDE'] = dict(delta_m=delta, passed=delta is not None and delta < -EPSILON_M)
        checks[fold] = tests
    return dict(tolerance_m=EPSILON_M, folds=checks,
                test_authorized=all(test['passed'] for per_fold in checks.values() for test in per_fold.values()))


def _seed_summary(rows):
    out = []
    for fold in FOLDS:
        for group in GROUPS:
            pairs = [row for row in rows if row['fold'] == fold and row['group'] == group]
            windows, agents = pairs[0]['windows'], pairs[0]['agents']
            require(all((row['windows'], row['agents']) == (windows, agents) for row in pairs),
                    'group population varies across seeds')
            if len(pairs) == 1:
                out.append({key: value for key, value in pairs[0].items() if key != 'seed'})
                continue
            row = dict(fold=fold, group=group, windows=windows, agents=agents, seeds=len(pairs))
            for metric in TERMS:
                for model in MODELS:
                    key = f'{model}_{metric}_m'
                    row[key] = statistics.fmean(pair[key] for pair in pairs) if windows else None
                key = f'delta_{metric}_cm'
                deltas = [pair[key] for pair in pairs]
                row[key] = statistics.fmean(deltas) if windows else None
                row[f'{key}_sample_sd'] = statistics.stdev(deltas) if windows else None
            out.append(row)
    return out


@torch.no_grad()
def score_all(splits, root, identities, purpose=DEV):
    """Verify all seed/window pairs in both folds before opening any label."""
    started = time.monotonic()
    root = Path(root)
    output = _output(root, purpose)
    require(set(splits) == set(identities) == set(FOLDS), 'both folds must be supplied')
    verified = {fold: _verify_fold(splits[fold], root, identities[fold], purpose) for fold in FOLDS}
    global_path = output / 'ALL_PREDICTIONS_SEALED.json'
    core = dict(purpose=purpose, folds={fold: verified[fold][-1] for fold in FOLDS}, truth_reads=0)
    if global_path.exists():
        whole = read(global_path)
        require(all(whole[key] == value for key, value in core.items()), 'global paired seal changed')
    else:
        whole = dict(core, seal_unix=time.time())
        immutable(global_path, whole)
    target = output / 'SCORE_RECEIPT.json'
    if target.exists():
        result = read(target)
        require(result['prediction_seal_sha256'] == digest(global_path)
                and all(digest(output / name) == sha for name, sha in result['output_sha256'].items()),
                'scored paired artifacts changed')
        return result
    historical = {(row['fold'], int(row['seed']), row['group']): row
                  for row in read_csv(BASE_DOC / 'PER_SEED.csv')} if purpose == TEST else {}
    first_truth = time.time()
    require(first_truth >= whole['seal_unix'], 'truth read before both-fold prediction seal')
    rows, seed_rows, replay = [], [], {}
    for fold in FOLDS:
        split = splits[fold]
        indices, seeds, mapping, cache_path, cache, _ = verified[fold]
        reader = FormalData(split, phase='scoring', verify=False)
        directory = output / fold
        accumulators = {seed: {group: {model: MetricAccumulator() for model in MODELS}
                              for group in GROUPS} for seed in seeds}
        for ordinal, index in enumerate(indices, 1):
            batch = reader.batch([index], purpose, truth=True)
            n = int(batch.valid.sum())
            require(bool(batch.valid.all()) and torch.isfinite(batch.future).all(),
                    'invalid packed paired scoring batch')
            group = _group(n)
            for seed in seeds:
                row = mapping[index, seed]
                require(row['n'] == n, 'sealed agent count differs from truth batch')
                pair = _read_pair(directory, row, row, cache_path, cache.get((index, seed)))
                values = dict(fold=fold, seed=seed, window=row['window'], n=n, group=group)
                for model in MODELS:
                    stats = hard_metric_statistics(torch.from_numpy(pair[model])[None], batch.future, batch.valid)
                    for scope in ('all', group):
                        accumulators[seed][scope][model].update_statistics(stats)
                    values.update({f'{model}_{key}': value for key, value in _row_metrics(stats).items()})
                for term in TERMS:
                    values[f'delta_{term}_cm'] = 100 * (values[f'R12_{term}'] - values[f'S0_{term}'])
                rows.append(values)
            if ordinal % 100 == 0 or ordinal == len(indices):
                _progress(directory, 'SCORING', fold, ordinal, len(indices), time.monotonic()-started)
        replay[fold] = 0.
        for seed, population in accumulators.items():
            for group, group_acc in population.items():
                result = _group_row(fold, seed, group, group_acc)
                seed_rows.append(result)
                if purpose == DEV and group == 'all':
                    reference = identities[fold]['S0_dev_reference']
                elif purpose == TEST and result['windows'] and group in ('all', 'N1', 'N9plus'):
                    old_group = {'all': 'all', 'N1': 'single', 'N9plus': 'N_ge_9'}[group]
                    old = historical[fold, seed, old_group]
                    require((int(old['scene_count']), int(old['agent_count'])) ==
                            (result['windows'], result['agents']), 'archived S0 group counts changed')
                    reference = {metric: float(old[f'new_{metric}']) for metric in METRICS}
                else:
                    continue
                difference = max(abs(result[f'S0_{metric}_m'] - reference[metric]) for metric in METRICS)
                require(difference <= EPSILON_M, f'{fold} S0 score does not replay historical reference')
                replay[fold] = max(replay[fold], difference)
    summary = _seed_summary(seed_rows)
    groups = {fold: {row['group']: row for row in summary if row['fold'] == fold} for fold in FOLDS}
    names = ('SUMMARY.csv', 'PER_SEED.csv', 'PER_WINDOW.csv')
    for filename, records in zip(names, (summary, seed_rows, rows)):
        csvwrite(output / filename, records)
    receipt = dict(status='COMPLETE', purpose=purpose, groups=groups, seeds=list(SEEDS) if purpose == TEST else [SEEDS[0]],
        decision=_decision(groups) if purpose == DEV else None, reference_S0_max_abs_diff_m=replay,
        prediction_seal_sha256=digest(global_path), prediction_seal_unix=whole['seal_unix'],
        first_truth_read_unix=first_truth, scoring_seconds=time.monotonic()-started,
        output_sha256={name: digest(output / name) for name in names})
    immutable(target, receipt)
    return receipt


def _format(value, precision=4, sign=False):
    return 'N/A' if value is None else format(value, f'{"+" if sign else ""}.{precision}f')


def _target_line(groups, fold):
    group = TARGET_GROUP[fold]
    row = groups[fold][group]
    if not row['windows']:
        return f'{fold.upper()} {GROUP_LABELS[group]} 为 N/A'
    joint, marginal, coordination = (row[f'delta_{metric}_cm'] for metric in ('JFDE', 'L', 'G'))
    if joint < 0:
        cause = '，下降主要来自逐人终点覆盖' if marginal < coordination else '，下降主要来自同槽配合'
    elif joint > 0:
        cause = '，上升主要来自逐人终点误差' if marginal > coordination else '，上升主要来自同槽配合差距'
    else:
        cause = ''
    return (f'{fold.upper()} {GROUP_LABELS[group]} 的 JFDE 变化为 {joint:+.4f} cm，'
            f'ΔL {marginal:+.4f} cm，ΔG {coordination:+.4f} cm{cause}')


def _report_table(receipt):
    columns = ['| 折 | 分组（窗／人次） | 指标 | S0（m） | R12（m） | R12−S0（cm） |',
               '|---|---:|---|---:|---:|---:|']
    for fold in FOLDS:
        for group in GROUPS:
            row = receipt['groups'][fold][group]
            for term in TERMS:
                delta = _format(row[f'delta_{term}_cm'], sign=True)
                sd = row.get(f'delta_{term}_cm_sample_sd')
                if sd is not None:
                    delta += f' ± {sd:.4f}'
                columns.append(
                    f'| {fold.upper()} | {GROUP_LABELS[group]}（{row["windows"]}／{row["agents"]}） | '
                    f'{term} | {_format(row[f"S0_{term}_m"], 9)} | '
                    f'{_format(row[f"R12_{term}_m"], 9)} | {delta} |')
    return '\n'.join(columns)


def _archive_score(output, doc):
    receipt = read(output / 'SCORE_RECEIPT.json')
    seal_file = output / 'ALL_PREDICTIONS_SEALED.json'
    require(receipt['status'] == 'COMPLETE' and receipt['prediction_seal_sha256'] == digest(seal_file)
            and receipt['first_truth_read_unix'] >= receipt['prediction_seal_unix'], 'paired score receipt mismatch')
    doc.mkdir(parents=True, exist_ok=True)
    for name, sha in receipt['output_sha256'].items():
        require(sha == digest(output / name), 'paired CSV changed before archive')
        (doc / name).write_bytes((output / name).read_bytes())
    for name in ('SCORE_RECEIPT.json', 'ALL_PREDICTIONS_SEALED.json'):
        write(doc / name, public_metadata(read(output / name)))
    return receipt


def build_report(root, doc):
    """Archive anonymized scores, retention indexes, and reproducibility receipts."""
    root, doc = Path(root), Path(doc)
    config = read(root / 'RUN_CONFIG.json')
    dev = _archive_score(_output(root, DEV), doc / 'dev')
    decision = _decision(dev['groups'])
    require(dev['decision'] == decision, 'development gate changed before archive')
    if decision['test_authorized']:
        test = _archive_score(_output(root, TEST), doc / 'test')
        write(doc / 'FROZEN_R12_TEST.json', public_metadata(read(root / 'FROZEN_R12_TEST.json')))
    else:
        require(not _output(root, TEST).exists() and not (root / 'FROZEN_R12_TEST.json').exists(),
                'heldout artifacts exist despite failing the two-fold gate')
        test = None
    overall = test or dev
    scope = '五种子测试探索性复核' if test else '开发集单种子验收'
    lines = ['# HOTEL 与 UNIV 槽内终点交互',
             f'**结论来自{scope}。** ' + '。'.join(
                 _target_line(overall['groups'], fold) for fold in FOLDS) + '。']
    coverage = []
    for fold in FOLDS:
        row = overall['groups'][fold]['all']
        deltas = tuple(row[f'delta_{metric}_cm'] for metric in ('minADE', 'minFDE'))
        conclusion = '存在逐人覆盖退化' if any(value > 100*EPSILON_M for value in deltas) else '逐人覆盖守住容差'
        coverage.append(f'{fold.upper()} {conclusion}，minADE {deltas[0]:+.4f} cm，minFDE {deltas[1]:+.4f} cm')
    lines.append('。'.join(coverage) + '。')
    hotel_singles = overall['groups']['hotel']['N1']
    if hotel_singles['windows']:
        changes = '，'.join(
            f'{metric} {hotel_singles[f"delta_{metric}_cm"]:+.4f} cm' for metric in METRICS)
        lines.append(f'HOTEL 单人组有 {hotel_singles["windows"]} 窗。{changes}。')
    lines += [
        '联合 JADE、JFDE、L 和 G 按窗等权，边缘 minADE、minFDE 按有效行人人次加权。'
        'L 是窗内逐人的最小终点误差均值，G 为 JFDE 减去 L。配对差统一为 R12−S0，负值表示误差降低。',
        '两折均使用原生 20 步 DDIM 和每个种子独立的 K20，配对预测共享窗口、行人顺序及逐槽噪声。',
        '## 开发集配对结果', _report_table(dev),
    ]
    failed = [(fold, name) for fold, tests in decision['folds'].items()
              for name, detail in tests.items() if not detail['passed']]
    if failed:
        lines.append('两折联合开测门槛未通过，按预定规则在开发集验收后结束。未达条件为 ' +
                     '、'.join(f'{fold.upper()} {condition}' for fold, condition in failed) + '。')
    else:
        lines.append('两折开发集全部门槛通过。')
    if test:
        lines += ['## 测试集探索性复核',
                  '推理种子为 2035–2039。表中差值为五个同种子配对差的均值 ± 样本标准差，'
                  '衡量固定权重的推理波动。', _report_table(test)]
    lines.append('## 训练与推理记录')
    for fold in FOLDS:
        origin = 24 if fold == 'hotel' else 16
        training_root = root / fold
        training = read(training_root / 'TRAINING_COMPLETE.json')
        fold_doc = doc / fold
        fold_doc.mkdir(exist_ok=True)
        write(fold_doc / 'TRAINING_COMPLETE.json', public_metadata(training))
        for filename in ('TOP5.json', 'EPOCH_METRICS.json', 'PREFLIGHT.json', 'TRAIN_IDENTITY.json'):
            available = next((base / filename for base in (
                training_root, training_root / 'training', training_root / 'training/checkpoints',
                training_root / 'checkpoints') if (base / filename).is_file()), None)
            if available:
                write(fold_doc / filename, public_metadata(read(available)))
        lines.append(f'{fold.upper()} 从冻结 S{origin} 开始，只训练新增模块 12 轮。'
                     f'R12 权重 SHA256 为 `{training["R12_sha256"]}`。')
        timing_fields = ('train_seconds','training_seconds','wall_seconds','peak_gpu_allocated_mib',
                         'peak_gpu_reserved_mib','updates','train_ddim_calls')
        available = ['`'+key+'` '+str(training[key]) for key in timing_fields if key in training]
        if available:
            lines.append('，'.join(available)+'。')
    for purpose in (DEV, TEST) if test else (DEV,):
        output = _output(root, purpose)
        seal = read(output / 'ALL_PREDICTIONS_SEALED.json')
        label = '开发集' if purpose == DEV else '测试集'
        for fold in FOLDS:
            data = seal['folds'][fold]
            times = data['inference_seconds']
            lines.append(
                f'{label} {fold.upper()} 真实推理，S0 {times["S0"]:.1f} 秒，R12 {times["R12"]:.1f} 秒。'
                f'S0 复用 {data["S0_reused"]} 对，新增 DDIM 调用 S0 {data["ddim_calls"]["S0"]}、'
                f'R12 {data["ddim_calls"]["R12"]}。')
    for name in ('RUN_CONFIG.json', 'WALL_TIME.json', 'START.json', 'RESOURCE_SUMMARY.json'):
        if (root / name).exists():
            write(doc / name, public_metadata(read(root / name)))
    if (root / 'WALL_TIME.json').exists():
        wall = read(root / 'WALL_TIME.json')
        if 'seconds' in wall:
            lines.append(f'实验墙钟耗时 {wall["seconds"]/60:.1f} 分钟。')
    lines += [
        '[开发分组](dev/SUMMARY.csv) · [开发匿名逐窗](dev/PER_WINDOW.csv) · '
        '[固定配置](RUN_CONFIG.json) · [开发评分收据](dev/SCORE_RECEIPT.json)',
    ]
    if test:
        lines.append('[测试分组](test/SUMMARY.csv) · [测试逐种子](test/PER_SEED.csv) · '
                     '[测试匿名逐窗](test/PER_WINDOW.csv) · [测试冻结](FROZEN_R12_TEST.json)')
    (doc / 'README.md').write_text('\n\n'.join(lines)+'\n', encoding='utf-8')
    write(doc / 'ARTIFACT_SHA256.json', {
        str(path.relative_to(doc)): digest(path) for path in sorted(doc.rglob('*'))
        if path.is_file() and path.name != 'ARTIFACT_SHA256.json'})
    return dict(decision=decision, reported_phase=overall['purpose'])

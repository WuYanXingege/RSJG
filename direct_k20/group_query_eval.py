"""Observed-only paired development evaluation for the fixed group-query epoch."""

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
from ebjd.metrics import hard_metric_statistics
from joint_flow_fourfold.report import csvwrite, public_metadata, read_csv

from .common import ATOL, RTOL, canonical_hash, digest, immutable, read, window_key, write
from .group_query import prepare_groups, summarize_query_offsets
from .interaction_diagnosis.assets import ARCHIVE, read_arrays, save_npz
from .interaction_eval import _row_metrics
from .model import K, decode_student, encode_observed, student_endpoints, student_noise


FOLDS = ('hotel', 'univ')
DEV = 'selection_dev'
SEED = 2035
WINDOWS = {'hotel': 688, 'univ': 622}
TARGET_COUNTS = {('hotel', 'N2_8'): 440, ('hotel', 'N9plus'): 181, ('univ', 'N9plus'): 61}
GROUPS = ('all', 'N1', 'N2_8', 'N9plus')
LABEL = dict(zip(GROUPS, ('全体', '单人', '2–8 人', 'N≥9')))
MODELS = ('S0', 'R12')
METRICS = ('JADE', 'JFDE', 'minADE', 'minFDE')
TERMS = METRICS + ('L', 'G')
TOL = 1e-6
DIAGNOSTICS = ('dq_norm_mean', 'dq_group_pair_mean', 'dq_group_pair_max',
               'endpoint_displacement_mean_m', 'endpoint_displacement_max_m', 'endpoint_anchor_max_abs_m')


def _protocol(ref, identity, cache):
    require(ref.fold in FOLDS and ref.purpose == DEV and ref.seeds == [SEED]
            and len(ref.indices) == WINDOWS[ref.fold], 'invalid original development reference')
    members = [ref.split['members'][index] for index in ref.indices]
    require(len(set(ref.indices)) == len(members) == identity['dev_windows']
            and sum(len(member['agent_ids']) for member in members) == identity['dev_agents']
            and canonical_hash(members) == identity['dev_manifest_sha256']
            and ref.identity['S0_sha256'] == identity['S0_sha256'],
            'development membership or S0 checkpoint changed')
    require(identity['final_epoch'] == 12 and digest(identity['R12_checkpoint']) == identity['R12_sha256'],
            'evaluation requires the fixed completed twelfth epoch')
    require(cache.ref is ref or cache.ref.source_hashes == ref.source_hashes,
            'S0 trajectory cache has a different seal')
    return dict(fold=ref.fold, purpose=DEV, seed=SEED, K=K, frames=12, DDIM_steps=20, eta=0.,
                S0_source=ref.source_hashes, S0_cache=cache.identity, identity=identity)


def _array_sha(paths):
    return hashlib.sha256(memoryview(np.ascontiguousarray(paths)).cast('B')).hexdigest()


def _reference(ref, dev_cache, index, ordinal):
    prior = ref.rows[index, SEED]
    key = window_key(ref.split, index)
    require(key == prior['member_sha256'], 'frozen agent/slot ordering changed')
    n = prior['n']
    source = dev_cache.read_path(index)
    require(source.dtype == np.float32 and source.shape == (K, n, 12, 2)
            and np.isfinite(source).all(), 'invalid S0 trajectory')
    return source, dict(window=f'window_{ordinal:04d}', n=n, member_sha256=key,
                        source_sha256=_array_sha(source), file=f'{ordinal:04d}.npz')


def _read_shard(directory, expected):
    path = directory / expected['file']
    row = read(path.with_suffix('.json'))
    require(all(row[key] == val for key, val in expected.items())
            and row['sha256'] == digest(path), 'sealed prediction identity or hash changed')
    require((row['n']+1)//2 <= row['groups'] <= row['n']
            and all(math.isfinite(row[key]) and row[key] >= 0
                    for key in (*DIAGNOSTICS, 'inference_seconds')), 'invalid diagnostic metadata')
    fields = read_arrays(path, {'paths': (K, row['n'], 12, 2), 'goals': (K, row['n'], 2)})
    require(np.allclose(fields['paths'][..., -1, :], fields['goals'], atol=ATOL, rtol=RTOL),
            'sealed trajectory no longer ends on its generated goal')
    return row, fields['paths']


@torch.no_grad()
def export_fold(assets, student, root, identity, dev_cache, *, should_yield=None):
    """Generate one native K20 trajectory set per locked window before opening any target."""
    ref = dev_cache.ref
    require(assets.fold == ref.fold and assets.split == ref.split, 'frozen geometry or split changed')
    root = Path(root) / 'development' / ref.fold
    root.mkdir(parents=True, exist_ok=True)
    protocol = _protocol(ref, identity, dev_cache)
    protocol_path = root / 'EXPORT_CONFIG.json'
    immutable(protocol_path, protocol)
    protocol_sha = digest(protocol_path)

    student.eval().requires_grad_(False)
    reader = FormalData(ref.split, phase='export', verify=False)
    started = time.monotonic()
    rows = []
    for ordinal, index in enumerate(ref.indices, 1):
        if should_yield is not None and should_yield():
            return None
        source, expected = _reference(ref, dev_cache, index, ordinal)
        expected['export_sha256'] = protocol_sha
        path = root / expected['file']
        side = path.with_suffix('.json')
        if side.exists():
            row, _ = _read_shard(root, expected)
        else:
            record = ref.split['records'][index]
            require(digest(record['path']) == record['sha256'], 'sealed development input changed')
            batch = reader.batch([index], DEV, truth=False).to(assets.device)
            n = expected['n']
            require(batch.valid.shape == (1, n) and bool(batch.valid.all())
                    and not bool(batch.future.count_nonzero()), 'expected observed-only packed data')
            observed = batch.observed[0]
            base_goals = dev_cache.read_endpoints(index)['g0']
            require(np.allclose(base_goals, source[..., -1, :], atol=ATOL, rtol=RTOL),
                    'S0 endpoints lack the original slot correspondence')
            tick = time.monotonic()
            context, map_features = encode_observed(assets.provider, batch)
            groups = prepare_groups(observed)
            goals = student_endpoints(student, assets.provider, observed, batch.semantic_maps[0], batch.valid[0],
                                      context, map_features, groups=groups)
            bridge_context = assets.model.encode_context(batch.observed, batch.semantic_maps, batch.valid)
            noise = student_noise(expected['member_sha256'], batch.agent_ids[0], SEED, assets.device)
            trajectories = decode_student(assets.model, bridge_context, goals, noise)[0].cpu().numpy()
            elapsed = time.monotonic() - tick
            goals_np = goals.cpu().numpy()
            require(np.allclose(trajectories[..., -1, :], goals_np, atol=ATOL, rtol=RTOL),
                    'native DDIM changed its generated goal')

            diagnostics = summarize_query_offsets(student.query_offsets(observed, context, groups))
            movement = np.linalg.norm(goals_np.astype(np.float64) - base_goals, axis=-1)
            save_npz(path, paths=trajectories, goals=goals_np)
            row = expected | diagnostics | dict(
                sha256=digest(path), inference_seconds=elapsed,
                endpoint_displacement_mean_m=float(movement.mean()),
                endpoint_displacement_max_m=float(movement.max()),
                endpoint_anchor_max_abs_m=float(np.abs(trajectories[..., -1, :] - goals_np).max()),
            )
            require(all(math.isfinite(row[key]) and row[key] >= 0 for key in DIAGNOSTICS),
                    'invalid group query or endpoint diagnostics')
            immutable(side, row)
        rows.append(row)
        if ordinal % 25 == 0 or ordinal == len(ref.indices):
            progress = dict(status='EXPORTING', fold=ref.fold, windows=ordinal, total=len(ref.indices),
                            seconds=time.monotonic()-started, unix=time.time())
            write(root / 'PROGRESS.json', progress)
            print(progress, flush=True)
    payload = dict(fold=ref.fold, purpose=DEV, seed=SEED, windows=len(rows), rows=rows,
                   export_sha256=protocol_sha, truth_reads=0, new_ddim_calls=len(rows))
    path = root / 'PREDICTIONS_FROZEN.json'
    if path.exists():
        old = read(path)
        require(all(old[key] == val for key, val in payload.items()), 'completed development seal differs')
    else:
        immutable(path, payload | dict(seal_unix=time.time()))
    return dict(fold=ref.fold, windows=len(rows), S0_reused=len(rows), new_ddim_calls=len(rows),
                seal_sha256=digest(path), inference_seconds=math.fsum(row['inference_seconds'] for row in rows),
                final_export_worker_seconds=time.monotonic()-started)


def _verify_seals(refs, root, identities, dev_caches):
    require(set(refs) == set(identities) == set(dev_caches) == set(FOLDS), 'both locked dev folds required')
    proofs = {}
    for fold in FOLDS:
        ref = refs[fold]
        require(ref.fold == fold, 'crossed development fold')
        directory = root / fold
        protocol_path = directory / 'EXPORT_CONFIG.json'
        protocol_sha = digest(protocol_path)
        require(read(protocol_path) == _protocol(ref, identities[fold], dev_caches[fold]),
                'sealed inference protocol differs')
        path = directory / 'PREDICTIONS_FROZEN.json'
        seal = read(path)
        expected = dict(fold=fold, purpose=DEV, seed=SEED, windows=WINDOWS[fold], export_sha256=protocol_sha,
                        truth_reads=0, new_ddim_calls=WINDOWS[fold])
        require(all(seal[key] == val for key, val in expected.items())
                and len(seal['rows']) == WINDOWS[fold], 'development fold is incomplete')
        for ordinal, (index, row) in enumerate(zip(ref.indices, seal['rows']), 1):
            _, identity = _reference(ref, dev_caches[fold], index, ordinal)
            saved, _ = _read_shard(directory, identity | dict(export_sha256=protocol_sha))
            require(saved == row, 'sealed manifest and window record differ')
        proofs[fold] = dict(sha256=digest(path), windows=seal['windows'], seal_unix=seal['seal_unix'],
                            S0_reused=seal['windows'], new_ddim_calls=seal['new_ddim_calls'],
                            inference_seconds=math.fsum(row['inference_seconds'] for row in seal['rows']))
    global_path = root / 'ALL_PREDICTIONS_SEALED.json'
    core = dict(purpose=DEV, seed=SEED, K=K, folds=proofs, truth_reads=0)
    if global_path.exists():
        seal = read(global_path)
        require(all(seal[key] == val for key, val in core.items()), 'global seal changed')
    else:
        seal = core | dict(seal_unix=time.time())
        immutable(global_path, seal)
    return seal


def _group(n):
    return 'N1' if n == 1 else 'N2_8' if n <= 8 else 'N9plus'


def _summary(fold, group, rows):
    """Average native joint and marginal primitives at their declared populations."""
    selected = [row for row in rows if group == 'all' or row['group'] == group]
    n = len(selected)
    people = sum(row['n'] for row in selected)
    out = dict(fold=fold, group=group, windows=n, agents=people,
               max_agents=max((row['n'] for row in selected), default=0))
    for metric in TERMS:
        for model in MODELS:
            key, denominator = ((f'{model}_agent_{metric}_sum', people)
                                if metric in ('minADE', 'minFDE') else (f'{model}_{metric}_m', n))
            out[f'{model}_{metric}_m'] = math.fsum(row[key] for row in selected) / denominator if n else None
        window_deltas = [row[f'delta_{metric}_cm'] for row in selected]
        key = f'delta_{metric}_cm'
        worst = max(selected, key=lambda row: row[key], default=None)
        out.update({key: 100*(out[f'R12_{metric}_m']-out[f'S0_{metric}_m']) if n else None,
                    f'{metric}_improved_windows': sum(value < 0 for value in window_deltas),
                    f'{metric}_worsened_windows': sum(value > 0 for value in window_deltas),
                    f'{metric}_unchanged_windows': sum(value == 0 for value in window_deltas),
                    f'{metric}_median_delta_cm': statistics.median(window_deltas) if n else None,
                    f'{metric}_worst_delta_cm': worst[key] if worst else None,
                    f'{metric}_worst_window': worst['window'] if worst else None})
    if n:
        require(abs(out['delta_JFDE_cm'] - out['delta_L_cm'] - out['delta_G_cm']) <= 1e-11,
                'window-weighted L/G decomposition differs')
    multi = [row for row in selected if row['groups'] > 1]
    out['multi_group_windows'] = len(multi)
    for key in DIAGNOSTICS:
        use = multi if key.startswith('dq_group_pair') else selected
        out[key] = ((max if key.endswith(('_max', '_max_m', '_max_abs_m')) else statistics.fmean)(
            row[key] for row in use) if use else None)
    return out


def _decision(groups):
    for fold in FOLDS:
        require(groups[fold]['all']['windows'] == WINDOWS[fold], 'decision population changed')
    for (fold, group), size in TARGET_COUNTS.items():
        require(groups[fold][group]['windows'] == size, 'decision group population changed')
    require(groups['univ']['all']['max_agents'] == 13, 'UNIV development maximum population changed')

    checks = []

    def add(name, fold, group, metric, max_delta_m):
        row = groups[fold][group]
        delta = row[f'R12_{metric}_m'] - row[f'S0_{metric}_m']
        checks.append(dict(name=name, fold=fold, group=group, metric=metric, windows=row['windows'],
                           delta_m=delta, max_delta_m=max_delta_m, passed=delta <= max_delta_m))

    add('target', 'univ', 'N9plus', 'JFDE', -.01)
    for fold, group in (('univ', 'all'), ('hotel', 'all'), ('hotel', 'N2_8')):
        for metric in ('JADE', 'JFDE'):
            add('joint', fold, group, metric, TOL)
    for fold in FOLDS:
        for group in ('all', 'N9plus'):
            for metric in ('minADE', 'minFDE'):
                add('coverage', fold, group, metric, TOL)
    return dict(checks=checks, candidate=all(check['passed'] for check in checks), test_reads=0)


def _score_fold(ref, directory, dev_cache, archived):
    reader = FormalData(ref.split, phase='scoring', verify=False)
    export_sha = digest(directory / 'EXPORT_CONFIG.json')
    results = []
    for ordinal, index in enumerate(ref.indices, 1):
        source, expected = _reference(ref, dev_cache, index, ordinal)
        sealed, paths = _read_shard(directory, expected | dict(export_sha256=export_sha))
        record = ref.split['records'][index]
        require(digest(record['path']) == record['sha256'], 'sealed development target changed')
        batch = reader.batch([index], DEV, truth=True)
        n = sealed['n']
        require(batch.valid.shape == (1, n) and bool(batch.valid.all())
                and bool(torch.isfinite(batch.future).all()), 'invalid development target population')
        name = expected['window']
        prior = archived[ref.fold, name]
        row = dict(fold=ref.fold, split=DEV, seed=SEED, window=name, n=n, group=_group(n))
        for model, array in zip(MODELS, (source, paths)):
            scores = _row_metrics(hard_metric_statistics(torch.from_numpy(array)[None], batch.future, batch.valid))
            row.update({f'{model}_{key}_m' if key in TERMS else f'{model}_{key}': value
                        for key, value in scores.items()})
        deviation = max(abs(row[f'S0_{term}_m']-float(prior[f'S0_{term}'])) for term in TERMS)
        require(int(prior['n']) == n and deviation <= TOL, 'original dev paired trajectory did not replay')
        row.update({f'delta_{term}_cm': 100*(row[f'R12_{term}_m']-row[f'S0_{term}_m']) for term in TERMS})
        row.update({key: sealed[key] for key in ('groups', *DIAGNOSTICS)})
        row['S0_replay_max_abs_m'] = deviation
        results.append(row)
    return results


@torch.no_grad()
def score_all(refs, root, identities, dev_caches, *, doc=None):
    """Verify both prediction seals, then score exactly the original development windows."""
    started = time.monotonic()
    directory = Path(root) / 'development'
    seal = _verify_seals(refs, directory, identities, dev_caches)
    seal_sha = digest(directory / 'ALL_PREDICTIONS_SEALED.json')
    target = directory / 'SCORE_RECEIPT.json'
    if target.exists():
        receipt = read(target)
        require(receipt['prediction_seal_sha256'] == seal_sha and receipt['status'] == 'COMPLETE'
                and receipt['first_truth_read_unix'] >= seal['seal_unix']
                and all(digest(directory/name) == sha for name, sha in receipt['output_sha256'].items()),
                'original scoring receipt differs')
    else:
        prior = {(row['fold'], row['window']): row for row in read_csv(ARCHIVE / 'dev/PER_WINDOW.csv')
                 if row['fold'] in FOLDS and int(row['seed']) == SEED}
        require(len(prior) == sum(WINDOWS.values()), 'archived S0 rows incomplete')
        first_truth = time.time()
        require(first_truth >= seal['seal_unix'], 'truth was requested before the global seal')
        windows, groups = [], {}
        for fold in FOLDS:
            rows = _score_fold(refs[fold], directory/fold, dev_caches[fold], prior)
            windows.extend(rows)
            groups[fold] = {group: _summary(fold, group, rows) for group in GROUPS}
            for term in METRICS:
                old = groups[fold]['all'][f'S0_{term}_m']
                require(abs(old-identities[fold]['S0_dev_reference'][term]) <= TOL,
                        'packed original development score differs from selected S0 receipt')

        decision = _decision(groups)
        names = ('GROUPS.csv', 'PER_WINDOW.csv')
        csvwrite(directory/names[0], [row for per_fold in groups.values() for row in per_fold.values()])
        csvwrite(directory/names[1], windows)
        receipt = dict(status='COMPLETE', purpose=DEV, seed=SEED, K=K, groups=groups, decision=decision,
                       windows=len(windows), prediction_seal_unix=seal['seal_unix'],
                       first_truth_read_unix=first_truth, prediction_seal_sha256=seal_sha,
                       S0_replay_max_abs_m=max(row['S0_replay_max_abs_m'] for row in windows),
                       scoring_seconds=time.monotonic()-started,
                       output_sha256={name: digest(directory/name) for name in names})
        immutable(target, receipt)
    if doc is not None:
        build_report(root, doc)
    return receipt


def _table(groups):
    rows = ['| 折 | 分组（窗／人次） | 指标 | S0（m） | 第 12 轮（m） | Δ（cm） | 改善／恶化／持平窗 | 中位差（cm） | 最差单窗 Δ（cm） |',
            '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for fold, summaries in groups.items():
        for group, row in summaries.items():
            size = f"{row['windows']}／{row['agents']}"
            for metric in TERMS:
                if row['windows']:
                    counts = '／'.join(str(row[f'{metric}_{kind}_windows'])
                                      for kind in ('improved', 'worsened', 'unchanged'))
                    cells = [f"{row[f'S0_{metric}_m']:.9f}", f"{row[f'R12_{metric}_m']:.9f}",
                             f"{row[f'delta_{metric}_cm']:+.4f}", counts,
                             f"{row[f'{metric}_median_delta_cm']:+.4f}",
                             f"{row[f'{metric}_worst_delta_cm']:+.4f}（{row[f'{metric}_worst_window']}）"]
                else:
                    cells = ['—']*6
                rows.append('| ' + ' | '.join([fold.upper(), f'{LABEL[group]}（{size}）', metric, *cells]) + ' |')
    return '\n'.join(rows)


def _copy_json(src, dst):
    value = read(src)
    public = public_metadata(value)
    if value == public:
        dst.write_bytes(src.read_bytes())
    else:
        write(dst, public)


def _archive_training(root, doc, lines):
    for fold in FOLDS:
        folder, out = root/fold, doc/fold
        out.mkdir(exist_ok=True)
        training = read(folder/'TRAINING_COMPLETE.json')
        _copy_json(folder/'TRAINING_COMPLETE.json', out/'TRAINING_COMPLETE.json')
        for filename in ('TOP5.json', 'EPOCH_METRICS.json', 'PREFLIGHT.json', 'TRAIN_IDENTITY.json',
                         'DEV_EXPORT_COMPLETE.json', 'R12_IDENTITY.json'):
            available = next((base/filename for base in (folder, folder/'training', folder/'training/checkpoints',
                                                          folder/'checkpoints') if (base/filename).is_file()), None)
            if available is not None:
                _copy_json(available, out/filename)
        sha = training.get('final_sha256', training.get('R12_sha256'))
        checkpoint = training.get('final_checkpoint', training.get('R12_checkpoint'))
        require(checkpoint and sha and digest(checkpoint) == sha and training['completed_epochs'] == 12,
                'training archive differs from the evaluated checkpoint')
        start = 24 if fold == 'hotel' else 16
        lines.append(f'{fold.upper()} 从 S{start} 热启动，训练满 12 轮。最终权重 SHA256 为 `{sha}`。')
        units = {'train_seconds': '训练累计秒数', 'wall_seconds': '训练墙钟秒数',
                 'peak_gpu_allocated_mib': '峰值已分配显存 MiB', 'peak_gpu_reserved_mib': '峰值预留显存 MiB',
                 'parameter_updates': '实际优化步数', 'train_ddim_calls': '训练 DDIM 次数'}
        available = [f'{label} {training[key]}' for key, label in units.items() if key in training]
        if available:
            lines.append('，'.join(available) + '。')


def build_report(root, doc):
    """Archive anonymized development receipts and the fixed practical decision."""
    root, doc = Path(root), Path(doc)
    directory = root/'development'
    seal_file = directory/'ALL_PREDICTIONS_SEALED.json'
    seal = read(seal_file)
    score = read(directory/'SCORE_RECEIPT.json')
    require(score['purpose'] == DEV and score['status'] == 'COMPLETE'
            and score['prediction_seal_sha256'] == digest(seal_file)
            and score['first_truth_read_unix'] >= seal['seal_unix']
            and score['decision'] == _decision(score['groups']), 'score seal or decision changed')
    doc.mkdir(parents=True, exist_ok=True)
    for name, sha in score['output_sha256'].items():
        require(digest(directory/name) == sha, 'scoring table changed')
        (doc/name).write_bytes((directory/name).read_bytes())
    for name in ('SCORE_RECEIPT.json', 'ALL_PREDICTIONS_SEALED.json'):
        _copy_json(directory/name, doc/name)

    groups = score['groups']
    decision = score['decision']
    checks = decision['checks']
    first = checks[0]['delta_m'] * 100
    hotel = all(row['passed'] for row in checks if row['name'] == 'joint' and row['fold'] == 'hotel')
    coverage = {fold: all(row['passed'] for row in checks if row['name'] == 'coverage'
                          and row['fold'] == fold) for fold in FOLDS}
    coverage_words = '、'.join(f'{fold.upper()}{"达标" if passed else "未达标"}'
                              for fold, passed in coverage.items())
    verdict = '列为后续验证候选' if decision['candidate'] else '保留 S0，结束本次方案'
    lines = ['# HOTEL 与 UNIV 生成前组级查询开发集验收',
             f'UNIV 全部 61 个密集窗的 JFDE 变化为 {first:+.4f} cm，'
             f'HOTEL 联合指标{"守住" if hotel else "未守住"}，个人覆盖 {coverage_words}，本版{verdict}。',
             '本次比较组级查询和原学生联合微调后的整套方案与各折原生 S0。'
             '两折均采用原开发集、种子 2035 和每窗原生 K20 的完整 12 帧轨迹。',
             '联合指标和 L、G 按窗等权，边缘指标按有效人次加权。'
             'Δ 为第 12 轮减 S0，负值表示误差下降。窗数按未舍入的逐窗差值计数。',
             '## 配对结果', _table(groups),
             '## 固定门槛',
             '\n'.join(['| 检查 | 折 | 分组（窗） | 指标 | 实测 Δ（m） | 允许上限（m） | 通过 |',
                        '|---|---|---:|---|---:|---:|---|'] + [
                 f"| {dict(target='密集组研发尺度', joint='联合指标', coverage='个人覆盖')[row['name']]} | "
                 f"{row['fold'].upper()} | {LABEL[row['group']]}（{row['windows']}） | {row['metric']} | "
                 f"{row['delta_m']:+.9g} | {row['max_delta_m']:+.9g} | {'是' if row['passed'] else '否'} |"
                 for row in checks]),
             '## 组意图及端点',
             'dq 差异是同一个场景槽内不同组偏移之间的 L2 距离，只在至少有两个组的窗口统计。'
             '表中平均量按窗等权，端点位移以同人同槽的 S0 端点为参照。',
             '\n'.join(['| 折 | 分组 | 多组窗 | dq 模长均值 | 组间 dq 距离均值／最大 | 端点位移均值／最大（m） |',
                        '|---|---|---:|---:|---:|---:|'] + [
                 '| ' + ' | '.join([fold.upper(), LABEL[group], str(row['multi_group_windows']),
                     *['／'.join(f"{row[key]:.6f}" if row[key] is not None else '—' for key in keys)
                       for keys in (('dq_norm_mean',), ('dq_group_pair_mean','dq_group_pair_max'),
                                    ('endpoint_displacement_mean_m','endpoint_displacement_max_m'))]]) + ' |'
                 for fold, per_fold in groups.items() for group, row in per_fold.items()]),
             '## 训练、封存和评分记录']
    _archive_training(root, doc, lines)
    for fold in FOLDS:
        detail = seal['folds'][fold]
        lines.append(f"{fold.upper()} 开发推理 {detail['inference_seconds']:.1f} 秒，"
                     f"复用 S0 {detail['S0_reused']} 窗，第 12 轮执行 {detail['new_ddim_calls']} 次完整 DDIM。")
    lines.append(f"两折预测封存后统一评分，评分耗时 {score['scoring_seconds']:.1f} 秒，"
                 f"S0 与旧逐窗分数的最大差值为 {score['S0_replay_max_abs_m']:.9g} m。"
                 'UNIV 开发窗最多 13 人。验收范围为本次两折原开发集。')
    for name in ('RUN_CONFIG.json', 'WALL_TIME.json', 'START.json', 'DEV_PATH_REPLAY.json'):
        if (root/name).is_file():
            _copy_json(root/name, doc/name)
    if (root/'WALL_TIME.json').is_file():
        wall = read(root/'WALL_TIME.json')
        if 'seconds' in wall:
            lines.append(f"实验墙钟耗时 {wall['seconds']/60:.1f} 分钟。")
    lines.append('[分组指标](GROUPS.csv) · [匿名逐窗](PER_WINDOW.csv) · [固定配置](RUN_CONFIG.json) · '
                 '[评分收据](SCORE_RECEIPT.json) · [两折封存](ALL_PREDICTIONS_SEALED.json)')
    (doc/'README.md').write_text('\n\n'.join(lines)+'\n', encoding='utf-8')
    write(doc/'ARTIFACT_SHA256.json', {str(path.relative_to(doc)): digest(path)
          for path in sorted(doc.rglob('*')) if path.is_file() and path.name != 'ARTIFACT_SHA256.json'})
    return decision

"""Resume observed-only HOTEL development predictions, then score S24 and S28."""

import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from bcep.contracts import canonical_hash, digest, require
from ebjd.metrics import MetricAccumulator, hard_metric_statistics
from joint_flow_fourfold.report import public_metadata
from joint_flow_fourfold.scoring import csvwrite
from .common import ATOL, METRICS, RTOL, immutable, read, window_key, write
from .model import decode_student, encode_observed, student_endpoints, student_noise


PURPOSE = 'selection_dev'
SEED = 2035
EPSILON_M = 1e-6
STUDENTS = ('S24', 'S28')
GROUPS = ('all', 'N1', 'N2_8', 'N9plus')
GROUP_LABELS = {'all': '全体', 'N1': 'N=1', 'N2_8': '2≤N≤8', 'N9plus': 'N≥9'}
OUTPUTS = ('SUMMARY.csv', 'PER_WINDOW.csv')


def _indices(assets, identity):
    split = assets.split
    indices = split[PURPOSE + '_indices']
    require(assets.fold == split['fold'] == 'hotel' and identity['seed'] == SEED,
            'HOTEL frozen development fold or seed differs')
    require(bool(indices) and len(indices) == len(set(indices)), 'invalid development members')
    require(canonical_hash([split['members'][i] for i in indices]) == identity['dev_manifest_sha256'],
            'ordered development members differ from locked evaluation identity')
    return indices


def _group(n):
    if n == 1:
        return 'N1'
    return 'N2_8' if n <= 8 else 'N9plus'


def _row_identity(assets, position, index, identity_sha):
    window = f'dev_{position:04d}'
    return dict(window=window, n=len(assets.split['members'][index]['agent_ids']),
                member_sha256=window_key(assets.split, index), file=window + '.npz',
                identity_sha256=identity_sha)


def _read_pair(path, n):
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == set(STUDENTS), 'unexpected trajectory shard fields')
        pair = {name: archive[name] for name in STUDENTS}
    for array in pair.values():
        require(array.shape == (20, n, 12, 2) and array.dtype == np.float32
                and np.isfinite(array).all(), 'invalid K20 trajectory shard')
    return pair


def _validate_row(folder, row, expected):
    require(all(row.get(key) == value for key, value in expected.items()),
            'trajectory shard has the wrong development window or model identity')
    require(all(math.isfinite(row['inference_seconds'][name])
                and row['inference_seconds'][name] >= 0 for name in STUDENTS),
            'invalid inference timing')
    path = folder / expected['file']
    require(digest(path) == row['sha256'], 'trajectory shard hash changed')
    return _read_pair(path, expected['n'])


@torch.no_grad()
def _predict_pair(assets, s24, s28, index, key, n):
    batch = assets.batch(index, PURPOSE, truth=False)
    require(batch.valid.shape == (1, n) and bool(batch.valid.all())
            and not batch.future.count_nonzero(), 'invalid observed-only development window')
    context, features = encode_observed(assets.provider, batch)
    bridge_context = assets.model.encode_context(batch.observed, batch.semantic_maps, batch.valid)
    noise = student_noise(key, batch.agent_ids[0], SEED, device=batch.observed.device)
    predictions, timing = {}, {}
    for name, student in zip(STUDENTS, (s24, s28)):
        if batch.observed.is_cuda:
            torch.cuda.synchronize(batch.observed.device)
        tick = time.perf_counter()
        goals = student_endpoints(student, assets.provider, batch.observed[0],
            batch.semantic_maps[0], batch.valid[0], context, features)
        path = decode_student(assets.model, bridge_context, goals, noise)
        torch.testing.assert_close(path[..., -1, :], goals[None], atol=ATOL, rtol=RTOL)
        predictions[name] = path[0].cpu().numpy()
        timing[name] = time.perf_counter() - tick
    return predictions, timing


def _progress(run, status, **details):
    row = dict(status=status, unix=time.time(), **details)
    write(run / 'EVAL_PROGRESS.json', row)
    print(row, flush=True)


def _export_receipt(path, seal):
    return dict(prediction_seal_sha256=digest(path), windows=seal['windows'],
                ddim_calls=seal['ddim_calls'], inference_seconds=seal['inference_seconds'])


def export(assets, s24, s28, run: Path, identity: dict, should_yield=None):
    """Persist one prediction pair per window. Return None after a safe resource yield."""
    indices = _indices(assets, identity)
    folder = run / 'dev_predictions'
    folder.mkdir(parents=True, exist_ok=True)
    immutable(run / 'EVALUATION_IDENTITY.json', identity)
    identity_sha = digest(run / 'EVALUATION_IDENTITY.json')
    seal_path = run / 'PREDICTIONS_FROZEN.json'
    if seal_path.exists():
        return _export_receipt(seal_path, _verified_seal(assets, run, identity))
    for student in (s24, s28):
        student.eval().requires_grad_(False)
    rows = []
    started = time.monotonic()
    for position, index in enumerate(indices, 1):
        if should_yield is not None and should_yield():
            _progress(run, 'YIELDING_DEVELOPMENT_EXPORT', windows=len(rows), total=len(indices))
            return None
        expected = _row_identity(assets, position, index, identity_sha)
        path = folder / expected['file']
        side = path.with_suffix('.json')
        if side.exists():
            row = read(side)
            _validate_row(folder, row, expected)
        else:
            record = assets.split['records'][index]
            require(digest(record['path']) == record['sha256'], 'development record changed')
            pair, timing = _predict_pair(assets, s24, s28, index, expected['member_sha256'], expected['n'])
            if path.exists():
                prior = _read_pair(path, expected['n'])
                require(all(np.array_equal(prior[name], pair[name]) for name in STUDENTS),
                        'incomplete prediction shard changed on recovery')
            else:
                temporary = path.with_suffix('.tmp.npz')
                np.savez_compressed(temporary, **pair)
                temporary.replace(path)
            row = dict(expected, sha256=digest(path), inference_seconds=timing)
            _validate_row(folder, row, expected)
            immutable(side, row)
        rows.append(row)
        if position % 25 == 0 or position == len(indices):
            _progress(run, 'EXPORTING_DEVELOPMENT', windows=position, total=len(indices))
    seal = dict(status='SEALED', purpose=PURPOSE, seed=SEED, K=20, DDIM_steps=20, eta=0.,
                identity_sha256=identity_sha, rows=rows, windows=len(rows),
                prediction_pairs=len(rows), ddim_calls={name: len(rows) for name in STUDENTS},
                seal_unix=time.time(), truth_reads=0,
                inference_seconds={name: math.fsum(row['inference_seconds'][name] for row in rows)
                                   for name in STUDENTS},
                final_export_worker_seconds=time.monotonic() - started)
    immutable(seal_path, seal)
    return _export_receipt(seal_path, seal)


def _verified_seal(assets, run, identity):
    indices = _indices(assets, identity)
    require(read(run / 'EVALUATION_IDENTITY.json') == identity, 'evaluation identity changed')
    identity_sha = digest(run / 'EVALUATION_IDENTITY.json')
    seal = read(run / 'PREDICTIONS_FROZEN.json')
    require(seal['status'] == 'SEALED' and seal['purpose'] == PURPOSE and seal['seed'] == SEED
            and seal['K'] == 20 and seal['DDIM_steps'] == 20 and seal['eta'] == 0.
            and seal['identity_sha256'] == identity_sha and seal['truth_reads'] == 0
            and seal['windows'] == len(indices) == len(seal['rows'])
            and seal['ddim_calls'] == {name: len(indices) for name in STUDENTS},
            'development predictions were not fully sealed for this evaluation')
    for position, (index, row) in enumerate(zip(indices, seal['rows']), 1):
        expected = _row_identity(assets, position, index, identity_sha)
        _validate_row(run / 'dev_predictions', row, expected)
        record = assets.split['records'][index]
        require(digest(record['path']) == record['sha256'], 'scoring input changed after prediction')
    return seal


def _window_statistics(stats):
    valid = stats.agent_valid
    L = float(stats.scene_marginal_fde.item())
    JFDE = float(stats.scene_jfde.item())
    ac = int(valid.sum())
    ade_sum = float(stats.agent_minade[valid].double().sum())
    fde_sum = float(stats.agent_minfde[valid].double().sum())
    return dict(JADE=float(stats.scene_jade.item()), JFDE=JFDE, minADE=ade_sum / ac,
                minFDE=fde_sum / ac, L=L, G=JFDE - L,
                agent_minADE_sum=ade_sum, agent_minFDE_sum=fde_sum)


def _group_results(accumulators):
    result = {}
    for group, models in accumulators.items():
        n = models['S24'].scene_count
        agents = models['S24'].agent_count
        require(n == models['S28'].scene_count and agents == models['S28'].agent_count,
                'paired score populations differ')
        metrics = METRICS + ('L', 'G')
        row = dict(group=group, windows=n, agents=agents)
        for name, acc in models.items():
            if n:
                native = acc.as_dict()
                values = {metric: native[metric] for metric in METRICS}
                values['L'] = native['scene_weighted_minFDE']
                values['G'] = values['JFDE'] - values['L']
            else:
                values = dict.fromkeys(metrics)
            row.update({f'{name}_{metric}_m': values[metric] for metric in metrics})
        for metric in metrics:
            row[f'delta_{metric}_cm'] = (100. * (row[f'S28_{metric}_m'] - row[f'S24_{metric}_m'])
                                         if n else None)
        row['delta_JFDE_minus_delta_L_plus_G_cm'] = (
            row['delta_JFDE_cm'] - (row['delta_L_cm'] + row['delta_G_cm']) if n else None)
        require(not n or abs(row['delta_JFDE_minus_delta_L_plus_G_cm']) < 1e-12,
                'JFDE decomposition does not reconcile')
        result[group] = row
    return result


def _decision(groups):
    full, multi = groups['all'], groups['N2_8']
    increase = {metric: full[f'S28_{metric}_m'] - full[f'S24_{metric}_m'] for metric in METRICS}
    failures = [metric for metric in METRICS if increase[metric] > EPSILON_M]
    multi_drop = (multi['S24_JFDE_m'] - multi['S28_JFDE_m']) if multi['windows'] else None
    passed_multi = multi_drop is not None and multi_drop > EPSILON_M
    return dict(retained='S28' if passed_multi and not failures else 'S24',
                N2_8_JFDE_drop_m=multi_drop, N2_8_JFDE_gate=passed_multi,
                all_four_gate=not failures, all_metric_failures=failures,
                full_precision_tolerance_m=EPSILON_M)


@torch.no_grad()
def score(assets, run: Path, identity: dict):
    """Verify both sealed predictions before calling the development truth reader."""
    started = time.monotonic()
    seal = _verified_seal(assets, run, identity)
    seal_sha = digest(run / 'PREDICTIONS_FROZEN.json')
    receipt_path = run / 'SCORE_RECEIPT.json'
    if receipt_path.exists():
        receipt = read(receipt_path)
        require(receipt['prediction_seal_sha256'] == seal_sha
                and all(receipt['output_sha256'][name] == digest(run / name) for name in OUTPUTS),
                'development score changed after scoring')
        return receipt
    accumulators = {group: {name: MetricAccumulator() for name in STUDENTS} for group in GROUPS}
    singleton = dict(windows=0, mismatched_windows=0, out_of_tolerance_windows=0,
                     max_abs_coordinate_m=0., max_abs_metric_m=dict.fromkeys(METRICS, 0.))
    records = []
    first_truth = time.time()
    require(seal['seal_unix'] <= first_truth, 'development truth preceded prediction seal')
    for index, row in zip(_indices(assets, identity), seal['rows']):
        path = run / 'dev_predictions' / row['file']
        pair = _read_pair(path, row['n'])
        batch = assets.batch(index, PURPOSE, truth=True, device='cpu')
        n = row['n']
        require(batch.valid.shape == (1, n) and bool(batch.valid.all()) and
                batch.future.shape == (1, n, 12, 2) and bool(torch.isfinite(batch.future).all()),
                'invalid development truth population')
        group = _group(n)
        record = dict(window=row['window'], n=n, group=group)
        for name, array in pair.items():
            statistics = hard_metric_statistics(torch.from_numpy(array)[None], batch.future, batch.valid)
            for accum_group in ('all', group):
                accumulators[accum_group][name].update_statistics(statistics)
            record.update({f'{name}_{metric}': value for metric, value in
                           _window_statistics(statistics).items()})
        for metric in (*METRICS, 'L', 'G'):
            record[f'delta_{metric}_cm'] = 100. * (record[f'S28_{metric}'] - record[f'S24_{metric}'])
        if group == 'N1':
            diff = float(np.max(np.abs(pair['S28'].astype(np.float64) - pair['S24'])))
            singleton['windows'] += 1
            singleton['mismatched_windows'] += int(diff != 0.)
            singleton['out_of_tolerance_windows'] += int(
                not np.allclose(pair['S28'], pair['S24'], rtol=RTOL, atol=ATOL))
            singleton['max_abs_coordinate_m'] = max(singleton['max_abs_coordinate_m'], diff)
            for metric in METRICS:
                diff = abs(record[f'S28_{metric}'] - record[f'S24_{metric}'])
                singleton['max_abs_metric_m'][metric] = max(
                    singleton['max_abs_metric_m'][metric], diff)
        records.append(record)
        if len(records) % 50 == 0 or len(records) == len(seal['rows']):
            _progress(run, 'SCORING_DEVELOPMENT', windows=len(records), total=len(seal['rows']))
    require(singleton['out_of_tolerance_windows'] == 0, 'edge changed a singleton prediction')
    groups = _group_results(accumulators)
    replay = {metric: (groups['all'][f'S24_{metric}_m'] -
                       float(identity['S24_dev_reference'][metric])) for metric in METRICS}
    require(all(abs(value) <= EPSILON_M for value in replay.values()),
            'S24 replay differs from its recorded development score')
    csvwrite(run / 'SUMMARY.csv', list(groups.values()))
    csvwrite(run / 'PER_WINDOW.csv', records)
    receipt = dict(status='COMPLETE', fold='hotel', purpose=PURPOSE, seed=SEED,
                   K=20, prediction_seal_sha256=seal_sha, prediction_seal_unix=seal['seal_unix'],
                   first_truth_read_unix=first_truth, test_access=False, groups=groups,
                   decision=_decision(groups), singleton=singleton, S24_archive_replay_diff_m=replay,
                   replay_tolerance_m=EPSILON_M, scoring_seconds=time.monotonic()-started,
                   output_sha256={name: digest(run / name) for name in OUTPUTS})
    immutable(receipt_path, receipt)
    return receipt


def _explain_decomposition(row):
    if not row['windows']:
        return 'N/A'
    changes = f'ΔL {row["delta_L_cm"]:+.4f} cm，ΔG {row["delta_G_cm"]:+.4f} cm'
    lower = [metric for metric in ('L', 'G') if row[f'delta_{metric}_cm'] < 0.]
    if row['delta_JFDE_cm'] < 0. and lower:
        dominant = min(lower, key=lambda metric: row[f'delta_{metric}_cm'])
        return f'{changes}，JFDE 下降主要来自 {dominant} 下降'
    return changes


def build_report(run: Path, doc: Path):
    """Archive only aggregate receipts and anonymous score CSVs."""
    score_receipt = read(run / 'SCORE_RECEIPT.json')
    identity = read(run / 'EVALUATION_IDENTITY.json')
    seal = read(run / 'PREDICTIONS_FROZEN.json')
    config = read(run / 'RUN_CONFIG.json')
    train = read(run / 'TRAINING_COMPLETE.json')
    wall = read(run / 'WALL_TIME.json')
    optional = {name: read(run / name) for name in (
        'RESOURCE_SUMMARY.json', 'MAIN_PROGRESS_AT_COMPLETION.json'
    ) if (run / name).exists()}
    require(score_receipt['status'] == 'COMPLETE'
            and score_receipt['prediction_seal_sha256'] == digest(run / 'PREDICTIONS_FROZEN.json')
            and seal['identity_sha256'] == digest(run / 'EVALUATION_IDENTITY.json')
            and score_receipt['prediction_seal_unix'] <= score_receipt['first_truth_read_unix']
            and train['S28_sha256'] == identity['S28_sha256'], 'incomplete or changed result')
    require(all(digest(run / name) == score_receipt['output_sha256'][name] for name in OUTPUTS),
            'score CSV changed')
    decision = score_receipt['decision']
    groups = score_receipt['groups']
    require(decision == _decision(groups) and train['status'] == 'COMPLETE'
            and train['completed_epoch'] == 28
            and train['frozen_before_sha256'] == train['frozen_after_sha256'],
            'training freeze or development decision does not reconcile')
    multi, dense = groups['N2_8'], groups['N9plus']
    retained = decision['retained']
    multi_delta = f'{multi["delta_JFDE_cm"]:+.4f} cm' if multi['windows'] else 'N/A'
    failures = decision['all_metric_failures']
    full_verdict = '四项均守住预定容差' if not failures else '超出容差的指标为 ' + '、'.join(failures)
    singleton = score_receipt['singleton']
    events_path = run / 'RESOURCE_EVENTS.jsonl'
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()] \
        if events_path.exists() else []
    first = train['first_ddim_backward_update']
    resource = optional.get('RESOURCE_SUMMARY.json', {})
    main = optional.get('MAIN_PROGRESS_AT_COMPLETION.json', {})
    lines = [
        '# HOTEL Direct K20 交互层微调',
        f'**最终保留 {retained}。** 2–8 人窗的 JFDE 配对差为 {multi_delta}。'
        f'{_explain_decomposition(multi)}。全体{full_verdict}。',
        '本实验由历史测试观察启发，只在 HOTEL 原 selection_dev 验收。'
        '从经哈希核对的 S24 出发，只对 edge 微调四轮。第 28 轮是唯一新增验收权重。',
        f'S24 SHA256 为 `{identity["S24_sha256"]}`。S28 SHA256 为 `{identity["S28_sha256"]}`。',
        '评分使用 seed 2035 和每窗 K=20，两个权重使用同窗、同行人顺序和同逐槽桥噪声。'
        'JADE、JFDE、L、G 按窗等权，minADE 和 minFDE 按有效行人加权。'
        'L 是逐窗的个人最小 FDE 均值。G = JFDE − L。',
        '| 分组 | 窗数／人次 | 指标 | S24（m） | S28（m） | S28−S24（cm） |',
        '|---|---:|---|---:|---:|---:|',
    ]
    for group in GROUPS:
        row = groups[group]
        metrics = METRICS + ('L', 'G') if group in ('N2_8', 'N9plus') else METRICS
        for metric in metrics:
            before, after, delta = (f'{row[f"S24_{metric}_m"]:.9f}',
                                    f'{row[f"S28_{metric}_m"]:.9f}',
                                    f'{row[f"delta_{metric}_cm"]:+.6f}') if row['windows'] else ('N/A',)*3
            lines.append(f'| {GROUP_LABELS[group]} | {row["windows"]}／{row["agents"]} | {metric} | '
                         f'{before} | {after} | {delta} |')
    lines.extend([
        f'N≥9 窗的分解为 {_explain_decomposition(dense)}。保留判据使用未舍入数据，'
        '要求 2–8 人组的 JFDE 下降超过 1e−6 m，且全体四项的增幅均不超过 1e−6 m。',
        f'单人 {singleton["windows"]} 窗，'
        f'坐标最大绝对差 {singleton["max_abs_coordinate_m"]:.9g} m，'
        f'超回放容差 {singleton["out_of_tolerance_windows"]} 窗。'
        '四项最大绝对差依次为 ' +
        '、'.join(f'{singleton["max_abs_metric_m"][metric]:.9g} m' for metric in METRICS) + '。',
        f'实际 edge 更新 {train["updates"]} 次。训练合格窗累计 {train["eligible_windows"]} 窗，'
        f'额外损失触发 {train["active_windows"]} 窗，比例 {train["extra_trigger_fraction"]:.2%}。'
        f'训练 DDIM 调用 {train["train_ddim_calls"]} 次。',
        f'训练 CUDA 峰值已分配 {train["peak_gpu_allocated_mib"]:.1f} MiB、'
        f'峰值保留 {train["peak_gpu_reserved_mib"]:.1f} MiB。'
        f'首个完成真实 DDIM 反传的更新为第 {first["epoch"]} 轮第 {first["update"]} 次更新，'
        f'该批峰值已分配 {first["peak_gpu_allocated_mib"]:.1f} MiB、'
        f'峰值保留 {first["peak_gpu_reserved_mib"]:.1f} MiB。',
        f'训练 worker 累计 {train["train_seconds"]/60:.1f} 分钟。'
        f'两组各 {seal["windows"]} 次开发集 DDIM 后封存，随后统一评分。'
        f'完成轨迹用时分别为 S24 {seal["inference_seconds"]["S24"]/60:.1f} 分钟、'
        f'S28 {seal["inference_seconds"]["S28"]/60:.1f} 分钟。'
        f'封存核验和评分耗时 {score_receipt["scoring_seconds"]:.1f} 秒。',
    ])
    if resource.get('gpu_uuid'):
        lines.append(f'共享 GPU UUID 为 `{resource["gpu_uuid"]}`。')
    if resource:
        lines.append(f'与主线同卡启动 {resource["shared_launches"]} 次，'
                     f'请求让出 {resource["yield_requests"]} 次，'
                     f'OOM 回滚 {resource["oom_rollbacks"]} 次，'
                     f'等待资源 {resource["resource_wait_periods"]} 段。'
                     f'nvidia-smi 采样的辅助进程显存峰值为 {resource["peak_observed_process_mib"]} MiB。')
    if main.get('status'):
        main_epoch = (main.get('progress') or {}).get('epoch')
        epoch = f'第 {main_epoch} 轮，' if main_epoch is not None else ''
        lines.append(f'归档时五折主线在 {str(main.get("fold","")).upper()} {epoch}'
                     f'状态为 `{main["status"]}`。')
    if 'seconds' in wall:
        lines.append(f'墙钟耗时 {wall["seconds"]/60:.1f} 分钟。')
    lines.extend([
        '[指标汇总](SUMMARY.csv) · [匿名逐窗](PER_WINDOW.csv) · [固定配置](RUN_CONFIG.json) · '
        '[训练收据](TRAINING_COMPLETE.json) · [评分收据](SCORE_RECEIPT.json) · '
        '[资源记录](RESOURCE_EVENTS.json) · [墙钟时间](WALL_TIME.json)',
    ])
    doc.mkdir(parents=True, exist_ok=True)
    (doc / 'README.md').write_text('\n\n'.join(lines).replace('|\n\n|', '|\n|') + '\n',
                                 encoding='utf-8')
    for name in OUTPUTS:
        (doc / name).write_bytes((run / name).read_bytes())
    public_seal = {key: value for key, value in seal.items() if key != 'rows'}
    public_seal['host_manifest_sha256'] = score_receipt['prediction_seal_sha256']
    for name, meta in (
        ('RUN_CONFIG.json', config), ('EVALUATION_IDENTITY.json', identity),
        ('TRAINING_COMPLETE.json', train), ('SCORE_RECEIPT.json', score_receipt),
        ('PREDICTIONS_FROZEN.json', public_seal), ('RESOURCE_EVENTS.json', events),
        ('WALL_TIME.json', wall), *optional.items(),
    ):
        write(doc / name, public_metadata(meta))
    write(doc / 'ARTIFACT_SHA256.json', {
        path.name: digest(path) for path in sorted(doc.iterdir())
        if path.is_file() and path.name != 'ARTIFACT_SHA256.json'
    })
    return decision

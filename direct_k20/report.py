"""Chinese report and seed-paired fold-equal statistics."""
import math
from pathlib import Path
from shutil import copyfile
import statistics

from bcep.contracts import require
from joint_flow_fourfold.report import csvwrite, public_metadata, read_csv
from .common import DOC, FOLDS, GROUPS, METRICS, RUN, SEEDS, digest, read, write

BRANCHES = ('A', 'B', 'new')
GROUP_LABELS = {'all': '全体', 'multi': '多人', 'single': '单人', 'N_ge_9': 'N≥9'}
FIVE = 'five_fold'


def _mean(values):
    return math.fsum(values) / len(values)


def _read_seed_scores(path):
    rows = read_csv(path)
    lookup = {(row['fold'], int(row['seed']), row['group']): row for row in rows}
    required_keys = {(fold, seed, group) for fold in FOLDS for seed in SEEDS for group in GROUPS}
    require(len(lookup) == len(rows) and set(lookup) == required_keys,
            'incomplete or repeated fold/seed/group score')
    for fold in FOLDS:
        for group in GROUPS:
            subset = [lookup[fold, seed, group] for seed in SEEDS]
            for row in subset:
                for key in ('scene_count', 'agent_count'):
                    row[key] = int(row[key])
                n, agents = row['scene_count'], row['agent_count']
                require(n >= 0 and agents >= n and bool(agents) == bool(n),
                        'invalid group population')
                for branch in BRANCHES:
                    for metric in METRICS:
                        key = f'{branch}_{metric}'
                        if n:
                            row[key] = float(row[key])
                            require(math.isfinite(row[key]) and row[key] >= 0, 'invalid error')
                        else:
                            require(row[key] in ('', None), 'empty group must be N/A')
                            row[key] = None
            require(all(row['scene_count'] == subset[0]['scene_count'] and
                        row['agent_count'] == subset[0]['agent_count'] for row in subset),
                    'group population differs by seed')
    for fold in FOLDS:
        for seed in SEEDS:
            all_row = lookup[fold, seed, 'all']
            for key in ('scene_count', 'agent_count'):
                require(all_row[key] == lookup[fold, seed, 'single'][key] +
                        lookup[fold, seed, 'multi'][key], 'single/multi partition mismatch')
    return lookup


def _summary(scope, group, participants, seed_rows):
    first = seed_rows[0]
    available = bool(participants)
    for metric in METRICS:
        row = dict(fold=scope, group=group, metric=metric, scene_count=first['scene_count'],
                   agent_count=first['agent_count'], fold_count=len(participants),
                   participating_folds=';'.join(participants), seed_count=len(SEEDS), ddof=1)
        for branch in BRANCHES:
            values = [x[f'{branch}_{metric}'] for x in seed_rows] if available else []
            row[f'{branch}_mean_m'] = _mean(values) if values else None
            row[f'{branch}_std_m'] = statistics.stdev(values) if values else None
        paired_nb = [x[f'new_minus_B_{metric}'] if scope == FIVE else
                     x[f'new_{metric}'] - x[f'B_{metric}'] for x in seed_rows] if available else []
        paired_na = [x[f'new_minus_A_{metric}'] if scope == FIVE else
                     x[f'new_{metric}'] - x[f'A_{metric}'] for x in seed_rows] if available else []
        row['new_minus_B_mean_cm'] = 100 * _mean(paired_nb) if available else None
        row['new_minus_B_std_cm'] = 100 * statistics.stdev(paired_nb) if available else None
        row['new_minus_A_mean_cm'] = 100 * _mean(paired_na) if available else None
        row['new_minus_A_std_cm'] = 100 * statistics.stdev(paired_na) if available else None
        margin = row['A_mean_m'] - row['B_mean_m'] if available else 0.
        row['B_joint_advantage_over_A_cm'] = 100 * margin if metric in METRICS[:2] and available else None
        row['new_joint_advantage_over_A_cm'] = -row['new_minus_A_mean_cm'] if metric in METRICS[:2] and available else None
        row['joint_advantage_retained_pct'] = (-row['new_minus_A_mean_cm'] / margin
                                                if metric in METRICS[:2] and margin > 0. else None)
        yield row


def aggregate(path):
    """Compute fold-equal means inside each seed before five-seed statistics."""
    lookup = _read_seed_scores(path)
    summary, five_per_seed = [], []
    for group in GROUPS:
        for fold in FOLDS:
            rows = [lookup[fold, seed, group] for seed in SEEDS]
            summary.extend(_summary(fold, group, [fold] if rows[0]['scene_count'] else [], rows))
        participants = [fold for fold in FOLDS if lookup[fold, SEEDS[0], group]['scene_count']]
        if group == 'all':
            require(participants == list(FOLDS), 'five-fold mean needs all five folds')
        rows = []
        for seed in SEEDS:
            originals = [lookup[fold, seed, group] for fold in participants]
            pooled = dict(fold=FIVE, group=group, seed=seed,
                          scene_count=sum(row['scene_count'] for row in originals),
                          agent_count=sum(row['agent_count'] for row in originals),
                          fold_count=len(participants), participating_folds=';'.join(participants))
            for metric in METRICS:
                for branch in BRANCHES:
                    key = f'{branch}_{metric}'
                    pooled[key] = _mean([row[key] for row in originals]) if originals else None
                for reference in ('A', 'B'):
                    differences = [row[f'new_{metric}'] - row[f'{reference}_{metric}']
                                   for row in originals]
                    pooled[f'new_minus_{reference}_{metric}'] = _mean(differences) if differences else None
            rows.append(pooled)
        five_per_seed.extend(rows)
        summary.extend(_summary(FIVE, group, participants, rows))
    return lookup, summary, five_per_seed


def _scope_label(fold, group):
    label = '五折等权' if fold == FIVE else fold.upper()
    return f'{label} {GROUP_LABELS[group]}'


def _format(value, decimals=6, signed=False):
    if value is None:
        return 'N/A'
    return f'{value:+.{decimals}f}' if signed else f'{value:.{decimals}f}'


def _table(lines, rows):
    lines.extend(['| 范围 | 窗数／人次 | 指标 | A（m） | B（m） | 学生（m） | 学生−B（cm，均值 ± 样本SD） |',
                  '|---|---:|---|---:|---:|---:|---:|'])
    for row in rows:
        mean, sd = row['new_minus_B_mean_cm'], row['new_minus_B_std_cm']
        delta = 'N/A' if mean is None else f'{mean:+.3f} ± {sd:.3f}'
        lines.append(f"| {_scope_label(row['fold'],row['group'])} "
                     f"| {row['scene_count']}／{row['agent_count']} | {row['metric']} "
                     f"| {_format(row['A_mean_m'])} | {_format(row['B_mean_m'])} "
                     f"| {_format(row['new_mean_m'])} | {delta} |")


def _describe(values):
    return '、'.join(f'{name} {value:+.3f} cm' for name, value in values.items() if value is not None)


def _timing_row(fold, result):
    cache_directory = RUN / fold / 'cache/teacher_cache'
    cache = read(cache_directory / 'MANIFEST.json')
    scale_receipt = read(cache_directory / 'TRAIN_B_SCALES.json')
    seal = read(RUN / fold / 'test/PREDICTIONS_FROZEN.json')
    sample = read(RUN / fold / 'test/TIMING_SAMPLE.json')
    return dict(fold=fold, teacher_cache_seconds=cache['cache_wall_seconds'],
        normalization_scoring_seconds=scale_receipt['score_seconds'],
        training_seconds=result['train_seconds'],
        development_seconds=result['development_seconds'],
        student_test_export_seconds=seal['export_wall_seconds'],
        normalization_ddim_calls=cache['scale_B_ddim_calls'],
        preflight_ddim_calls=read(RUN/fold/'training/PREFLIGHT.json').get('ddim_calls',2),
        training_ddim_calls=result['train_ddim_calls'],
        development_ddim_calls=result['development_ddim_calls'],
        student_test_ddim_calls=seal['ddim_calls'],
        A_B_timing_ddim_calls=seal['A_B_timing_ddim_calls'],
        latency_windows=len(sample['sample']),
        **{f'{model}_median_seconds': statistics.median(x[f'{model}_seconds'] for x in sample['sample'])
           for model in ('A', 'B', 'student')})


def build_report():
    DOC.mkdir(parents=True, exist_ok=True)
    config = read(RUN / 'RUN_CONFIG.json')
    barrier = read(RUN / 'FROZEN_STUDENTS.json')
    receipt = read(RUN / 'SCORE_RECEIPT.json')
    assert all(digest(RUN / filename) == sha for filename, sha in receipt['output_sha256'].items())
    assert receipt['global_prediction_seal_sha256'] == digest(RUN / 'ALL_PREDICTIONS_SEALED.json')
    assert receipt['selection_seal_unix'] <= receipt['first_truth_read_unix']
    lookup, summary, means_by_seed = aggregate(RUN / 'PER_SEED.csv')
    index = {(row['fold'],row['group'],row['metric']): row for row in summary}
    overall = {metric:index[FIVE,'all',metric] for metric in METRICS}
    all_better = all(row['new_minus_B_mean_cm'] < 0 for row in overall.values())
    fold_degradation = {fold: {metric: index[fold,'all',metric]['new_minus_B_mean_cm']
        for metric in METRICS if index[fold,'all',metric]['new_minus_B_mean_cm'] > 0} for fold in FOLDS}
    decision = dict(five_fold_all_four_lower_than_B=all_better,
        five_fold_changes_cm={m: dict(student_minus_B=r['new_minus_B_mean_cm'],
                                      student_minus_A=r['new_minus_A_mean_cm']) for m,r in overall.items()},
        regressions_cm=fold_degradation)
    lines = [
        '# 直接生成 K20 的五折轨迹预测',
        '本实验由历史测试结果启发，属于五折五种子探索性对照。五个种子衡量固定权重的推理波动。',
        '**五折等权四项都低于 B。** 可作为独立数据验证的替代候选。' if all_better
        else '**五折等权未满足四项同时降低的目标，保留原 B。**',
        f'正式评分覆盖 {receipt["windows"]} 窗、{receipt["window_seed_pairs"]} 个窗种子配对。'
        '五折权重先冻结，全部预测封存后统一读真值。每个种子单独使用 K20，'
        '联合指标按窗等权，边缘指标按有效行人人次加权。先对同种子各折等权，再求五种子均值与样本标准差。',
        '## 全体主表',
        '绝对值单位米，配对差单位厘米，样本标准差由五个配对种子直接计算。2 cm 仅用于解释变化幅度。',
    ]
    _table(lines, [index[fold,'all',metric] for fold in FOLDS + (FIVE,) for metric in METRICS])
    lines.extend(['## 与 A 的边缘差距',
        '| 范围 | minADE 学生−A（cm） | minFDE 学生−A（cm） |',
        '|---|---:|---:|'])
    for fold in FOLDS + (FIVE,):
        a = index[fold,'all','minADE']
        f = index[fold,'all','minFDE']
        lines.append(f"| {_scope_label(fold,'all')} | {a['new_minus_A_mean_cm']:+.3f} ± {a['new_minus_A_std_cm']:.3f} | "
                     f"{f['new_minus_A_mean_cm']:+.3f} ± {f['new_minus_A_std_cm']:.3f} |")
    lines += ['出现的全体退化为 ' + ('、'.join(f'{f.upper()}（{_describe(values)}）'
        for f,values in fold_degradation.items() if values) or '无') + '。', '## 分组结果']
    _table(lines, [index[fold,group,metric] for group in ('single','multi','N_ge_9')
                   for fold in FOLDS + (FIVE,) for metric in METRICS])
    for fold, group in (('eth','multi'),('univ','N_ge_9')):
        lines.append(f"{_scope_label(fold,group)}的学生−B 联合差值为 "
            f"JADE {index[fold,group,'JADE']['new_minus_B_mean_cm']:+.3f} cm，"
            f"JFDE {index[fold,group,'JFDE']['new_minus_B_mean_cm']:+.3f} cm。")
    lines.extend(['## 训练、选模与推理',
        '学生使用 20 个查询直接生成 20 个整场场景。个人与整场教师匹配只参与训练。'
        'Flow、修正头、编码器和桥均冻结。第 13–24 轮每个 16 窗逻辑批抽两窗运行可导的完整 DDIM，'
        '用入选概率修正 ADE 项。验证只发生在第 12、16、20、24 轮，按最大相对 B 比值、平均比值、较早轮次依次选模。',
        '| 折 | 选中轮次 | SHA256 |',
        '|---|---:|---|'])
    selectors, timing = [], []
    for fold in FOLDS:
        folder = RUN / fold
        result = read(folder / 'TRAINING_COMPLETE.json')
        selected = barrier['folds'][fold]
        assert digest(selected['selected_checkpoint']) == selected['selected_sha256']
        assert digest(folder/'TRAINING_COMPLETE.json') == selected['training_receipt_sha256']
        lines.append(f"| {fold.upper()} | {selected['selected_epoch']} | `{selected['selected_sha256']}` |")
        selection_file = folder / 'training/DEV_SELECTION.csv'
        assert digest(selection_file) == result['dev_selection_sha256']
        for row in read_csv(selection_file):
            selectors.append(dict(fold=fold, selected=int(row['epoch'])==selected['selected_epoch'], **row))
        subdir = DOC / fold
        subdir.mkdir(exist_ok=True)
        write(subdir/'TRAINING_RESULT.json', public_metadata(result))
        write(subdir/'PREFLIGHT.json', public_metadata(read(folder/'training/PREFLIGHT.json')))
        write(subdir/'EPOCH_METRICS.json', public_metadata(read(folder/'training/EPOCH_METRICS.json')))
        write(subdir/'TOP5.json', public_metadata(read(folder/'training/checkpoints/TOP5.json')))
        write(subdir/'TRAIN_B_SCALES.json', public_metadata(read(folder/'cache/teacher_cache/TRAIN_B_SCALES.json')))
        write(subdir/'TIMING_SAMPLE.json', public_metadata(read(folder/'test/TIMING_SAMPLE.json')))
        write(subdir/'EXPORT_CONFIG.json', public_metadata(read(folder/'test/EXPORT_CONFIG.json')))
        timing.append(_timing_row(fold, result))
    lines.append('同卡速度比较使用每折 8 个稳定哈希选择的测试观测窗和 seed 2035。'
                 '计时从已在设备上的观测到完整轨迹，包含编码、采样和解码，不含读取或写盘。')
    lines.extend(['| 折 | A 中位秒/窗 | B 中位秒/窗 | 学生中位秒/窗 |',
                  '|---|---:|---:|---:|'])
    for row in timing:
        lines.append(f"| {row['fold'].upper()} | {row['A_median_seconds']:.4f} | "
                     f"{row['B_median_seconds']:.4f} | {row['student_median_seconds']:.4f} |")
    lines.extend([
        f"原生 A/B 回放最大分数差 {receipt['original_AB_max_abs_replay_m']:.3g} m。"
        '逐窗跨种子的学生端点分数差见评分收据。教师缓存、训练、验证、测试和速度对照的 DDIM 次数及耗时均见运行表。',
        '复现命令 `CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 '
        'OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python -m direct_k20.run`。',
        '[五折等权](FIVE_FOLD_EQUAL.csv) · [逐折逐种子](PER_SEED.csv) · [匿名逐窗](PER_WINDOW.csv) · '
        '[选模](SELECTION.csv) · [运行与 DDIM](TIMING.csv) · [评分收据](SCORE_RECEIPT.json)',
    ])
    (DOC/'README.md').write_text('\n\n'.join(lines).replace('|\n\n|','|\n|')+'\n', encoding='utf-8')
    csvwrite(DOC/'SUMMARY.csv', summary)
    csvwrite(DOC/'FIVE_FOLD_EQUAL.csv', [row for row in summary if row['fold'] == FIVE])
    csvwrite(DOC/'FIVE_FOLD_PER_SEED.csv', means_by_seed)
    csvwrite(DOC/'SELECTION.csv', selectors)
    csvwrite(DOC/'TIMING.csv', timing)
    for name in ('PER_SEED.csv','PER_WINDOW.csv'):
        copyfile(RUN/name,DOC/name)
    for name in ('RUN_CONFIG.json','START.json','WALL_TIME.json','FROZEN_STUDENTS.json',
                 'SCORE_RECEIPT.json','ALL_PREDICTIONS_SEALED.json'):
        write(DOC/name, public_metadata(read(RUN/name)))
    write(DOC/'DECISION.json',decision)
    for path in DOC.rglob('*'):
        if path.is_file():
            assert b'/media/ee615' not in path.read_bytes(),path
    write(DOC/'ARTIFACT_SHA256.json',{
        str(path.relative_to(DOC)):digest(path) for path in sorted(DOC.rglob('*'))
        if path.is_file() and path.name != 'ARTIFACT_SHA256.json'
    })

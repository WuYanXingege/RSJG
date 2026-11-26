"""Aggregate sealed K20 replay scores and archive the anonymous results."""
import math
from pathlib import Path
from shutil import copyfile
import statistics

from bcep.contracts import digest, require
from bcep_joint.common import read, write
from joint_flow_fourfold.report import csvwrite, public_metadata, read_csv


FOLDS = ('zara1', 'zara2', 'hotel', 'univ', 'eth')
SEEDS = (2035, 2036, 2037, 2038, 2039)
GROUPS = ('all', 'multi', 'single', 'N_ge_9')
METRICS = ('JADE', 'JFDE', 'minADE', 'minFDE')
BRANCHES = ('A', 'B', 'new')
GROUP_LABELS = {'all': '全体', 'multi': '多人', 'single': '单人', 'N_ge_9': 'N≥9'}
FIVE = 'five_fold'
SOURCE_FILES = ('PER_SEED.csv', 'PER_WINDOW.csv', 'SELECTED_SLOTS.csv')


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
                if group == 'single' and n:
                    require(all(row[f'new_{m}'] == row[f'A_{m}'] for m in METRICS),
                            'singleton scores did not exactly replay A')
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
    lines.extend(['| 范围 | 窗数／人次 | 指标 | A（m） | B（m） | 新方案（m） | 新−B（cm，均值 ± 样本SD） |',
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


def _decision(lookup, summary):
    by_key = {(row['fold'], row['group'], row['metric']): row for row in summary}
    changes = {fold: {group: {metric: by_key[fold,group,metric]['new_minus_B_mean_cm']
                              for metric in METRICS}
                      for group in GROUPS}
               for fold in FOLDS + (FIVE,)}
    overall = changes[FIVE]['all']
    all_better = all(value < 0 for value in overall.values())
    folds_worse = {fold: [metric for metric, value in changes[fold]['all'].items() if value > 0]
                   for fold in FOLDS}
    eth_contribution = {}
    for metric in METRICS:
        count = 'scene_count' if metric in METRICS[:2] else 'agent_count'
        values = []
        for seed in SEEDS:
            single = lookup['eth', seed, 'single']
            all_eth = lookup['eth', seed, 'all']
            fraction = single[count] / all_eth[count]
            values.append(fraction * (single[f'new_{metric}'] - single[f'B_{metric}']))
        contribution = 100 * _mean(values) / len(FOLDS)
        eth_contribution[metric] = dict(
            five_fold_delta_contribution_cm=contribution,
            share_of_five_fold_reduction_pct=100 * contribution / overall[metric] if overall[metric] < 0 else None,
        )
    improved = [metric for metric in METRICS if overall[metric] < 0]
    eth_single_dominates = bool(improved) and all(
        eth_contribution[metric]['share_of_five_fold_reduction_pct'] > 50 for metric in improved)
    multi_no_gain = all(row is None or row >= 0 for row in changes[FIVE]['multi'].values())
    univ_no_gain = all(row is None or row >= 0 for row in changes['univ']['all'].values())
    only_single_supported = eth_single_dominates and multi_no_gain and univ_no_gain
    if only_single_supported:
        recommendation = '单人分支有效，多人组合未得到支持。结束本轮方案。'
    elif all_better:
        recommendation = '保留为独立数据验证的候选，暂不采用为默认方案。'
    else:
        recommendation = '结束本版固定组合方案，保留原有模型。'
    return dict(five_fold_all_metrics_improved=all_better, five_fold_new_minus_B_cm=overall,
                per_fold_degraded_metrics=folds_worse, new_minus_B_cm_by_group=changes,
                ETH_single_contribution=eth_contribution, ETH_single_dominates=eth_single_dominates,
                multi_no_metric_improves=multi_no_gain, UNIV_no_metric_improves=univ_no_gain,
                only_single_supported=only_single_supported, recommendation=recommendation)


def build_report(run: Path, destination: Path):
    """Build the final Chinese report from completed anonymous score files."""
    run, destination = Path(run), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    config = read(run / 'RUN_CONFIG.json')
    receipt = read(run / 'SCORE_RECEIPT.json')
    wall = read(run / 'WALL_TIME.json')
    require(receipt['training_updates'] == 0, 'offline run changed weights')
    require(receipt['window_seed_pairs'] == 5 * receipt['windows'], 'incomplete five-seed score')
    for name, sha in receipt.get('output_sha256', {}).items():
        require(Path(name).name == name and digest(run / name) == sha, 'scoring output changed')
    lookup, summaries, five_per_seed = aggregate(run / 'PER_SEED.csv')
    by_key = {(r['fold'], r['group'], r['metric']): r for r in summaries}
    require(sum(lookup[f, SEEDS[0], 'all']['scene_count'] for f in FOLDS) == receipt['windows'],
            'report and scorer window count mismatch')
    require(5 * sum(lookup[f, SEEDS[0], 'single']['scene_count'] for f in FOLDS) ==
            receipt['singleton_exact_pairs'], 'singleton exact-match count mismatch')
    decision = _decision(lookup, summaries)
    overall = decision['five_fold_new_minus_B_cm']
    conclusion = ('**五折平均四项同时改善。**' if decision['five_fold_all_metrics_improved']
                  else '**五折平均未同时改善四项。**')
    lines = [
        '# 五折 A/B 整场轨迹组合探索性回放',
        f"{conclusion} 新方案相对 B 的全体五折等权差值为 {_describe(overall)}。"
        f"**{decision['recommendation']}**",
        '方案在查看历史测试结果后确定，属于探索性回放。五个固定推理种子描述同一权重下的推理波动。'
        '每个种子独立保留 20 个整场预测，五折使用同一个规则。',
        f"已评分 {receipt['windows']} 窗、{receipt['window_seed_pairs']} 个窗种子配对。"
        f"{receipt['singleton_exact_pairs']} 个单人窗种子配对的四项指标逐项精确等于 A。",
        '## 五折全体主表',
        '各方法绝对值为五种子均值。差值先在同一折、同一种子以新方案减 B 计算，'
        '同一种子内各折等权，再计算五种子的均值和样本标准差。'
        '正差表示误差变大，2 cm 仅作幅度参照。窗数和人次按种子列一次。',
    ]
    _table(lines, [by_key[fold,'all',metric] for fold in FOLDS + (FIVE,) for metric in METRICS])
    worse = [f"{fold.upper()}（{'、'.join(names)}）" for fold, names in
             decision['per_fold_degraded_metrics'].items() if names]
    lines.append('全体出现退化的折和指标为 ' + ('、'.join(worse) if worse else '无') + '。')
    lines.extend([
        '## 多人组合及重点子组',
        '每折多人窗单独统计。五折多人等权只取各折的多人窗，'
        '窗等权统计 JADE、JFDE，有效人次加权统计 minADE、minFDE。',
    ])
    _table(lines, [by_key[fold,'multi',metric] for fold in FOLDS + (FIVE,) for metric in METRICS])
    lines.append('ETH 单人组和 UNIV N≥9 组如下。UNIV 全体与 ETH 多人见前两表。')
    _table(lines, [by_key[fold,group,metric] for fold, group in (('eth','single'), ('univ','N_ge_9'))
                   for metric in METRICS])
    lines.extend([
        '## 联合预测优势与单人分支贡献',
        '联合优势以比 A 低多少厘米表示。当 B 已优于 A，保留比例按 '
        '（A−新方案）/（A−B）计算，允许负数和超过 100%。'
        'B 相对 A 无正优势时，比例记 N/A。',
        '| 范围 | 指标 | B 对 A 的联合优势（cm） | 新方案对 A 的联合优势（cm） | 保留比例（%） |',
        '|---|---|---:|---:|---:|',
    ])
    for fold in FOLDS + (FIVE,):
        for metric in METRICS[:2]:
            row = by_key[fold,'all',metric]
            lines.append(f"| {_scope_label(fold,'all')} | {metric} "
                         f"| {_format(row['B_joint_advantage_over_A_cm'],3,True)} "
                         f"| {_format(row['new_joint_advantage_over_A_cm'],3,True)} "
                         f"| {_format(row['joint_advantage_retained_pct'],2)} |")
    lines.append('表中单人贡献以 ETH 单人窗替换带来的度量变化乘以其在 ETH 的对应权重，'
                 '再除以五折数。贡献可被其他窗口的反向变化抵消。')
    lines.extend(['| 指标 | ETH 单人对五折平均差的贡献（cm） | 占五折净降幅（%） |',
                  '|---|---:|---:|'])
    for metric, contribution in decision['ETH_single_contribution'].items():
        lines.append(f"| {metric} | {_format(contribution['five_fold_delta_contribution_cm'],3,True)} "
                     f"| {_format(contribution['share_of_five_fold_reduction_pct'],2)} |")
    lines.append('五折多人差值为 ' + _describe(decision['new_minus_B_cm_by_group'][FIVE]['multi']) +
                 '。UNIV 全体差值为 ' + _describe(decision['new_minus_B_cm_by_group']['univ']['all']) + '。')
    lines.extend([
        '## 固定组合与运行成本',
        '单人窗直接输出 A。多人窗根据预测轨迹之间的距离，从 B 的 20 个整场中贪心删 4 个，'
        '再从 A 贪心补入 4 个整场。距离为 12 帧平均欧氏距离与末帧欧氏距离各占一半。'
        '两个阶段均以原始槽位解决同值。整个选择仅使用预测、有效掩码和成员顺序，'
        '封存全体槽位及源预测哈希后才统一评分。',
        '输入使用五折正式冻结 A/B 原始轨迹。ZARA2 的历史 A 基线身份沿用其原归档。'
        '同窗种子的原 A/B 指标使用原生评分器复核，训练更新为 0。',
        f"离线组合用时 {wall['combination_wall_seconds']:.3f} 秒（含源轨迹读取及哈希核验），"
        f"其中选择器 CPU 累计 {wall['selector_cpu_seconds']:.3f} 秒。"
        f"评分用时 {wall['scoring_wall_seconds']:.3f} 秒，总用时 {wall['wall_seconds']:.3f} 秒。",
        f"实际补推 {receipt['supplementary_prediction_pairs']} 个窗种子配对，"
        f"新增 {receipt['supplementary_model_forwards']} 路模型前向。"
        '在线多人窗仍需先运行 A 和 B 的两路整场轨迹预测，再选择用于输出的 20 个整场。'
        '输出预算为 K=20，推理开销需计入两路采样和解码。单人窗只需 A。',
        '冻结资产就位后可在训练主机上用以下命令复现。',
        '`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m ab_whole_scene_mix.run`',
        '[分折分种子 CSV](PER_SEED.csv) · [五折等权 CSV](FIVE_FOLD_EQUAL.csv) · '
        '[五折等权逐种子 CSV](FIVE_FOLD_PER_SEED.csv) · [匿名逐窗 CSV](PER_WINDOW.csv) · '
        '[来源槽位与源预测哈希](SELECTED_SLOTS.csv) · [评分收据](SCORE_RECEIPT.json)',
    ])
    (destination / 'README.md').write_text('\n\n'.join(lines).replace('|\n\n|', '|\n|') + '\n',
                                          encoding='utf-8')
    for name in SOURCE_FILES:
        copyfile(run / name, destination / name)
    csvwrite(destination / 'SUMMARY.csv', summaries)
    csvwrite(destination / 'FIVE_FOLD_EQUAL.csv', [row for row in summaries if row['fold'] == FIVE])
    csvwrite(destination / 'FIVE_FOLD_PER_SEED.csv', five_per_seed)
    for name, obj in (('RUN_CONFIG.json', config), ('SCORE_RECEIPT.json', receipt),
                      ('WALL_TIME.json', wall), ('DECISION.json', decision)):
        write(destination / name, public_metadata(obj))
    for path in destination.rglob('*'):
        if path.is_file():
            require(path.suffix in ('.md', '.json', '.csv') and b'/media/ee615' not in path.read_bytes(),
                    'nonpublic archive file')
    write(destination / 'ARTIFACT_SHA256.json', {
        str(path.relative_to(destination)): digest(path)
        for path in sorted(destination.rglob('*')) if path.is_file() and path.name != 'ARTIFACT_SHA256.json'
    })
    return decision

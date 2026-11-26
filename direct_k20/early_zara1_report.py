"""Publish the independent early ZARA1 evaluation using sealed native scores."""

import math
from pathlib import Path
from shutil import copyfile

from bcep.contracts import require
from joint_flow_fourfold.report import csvwrite, public_metadata, read_csv

from .common import GROUPS, METRICS, SEEDS, digest, groups_for, read, write
from .report import BRANCHES, GROUP_LABELS, _summary

FOLD = 'zara1'
EPOCH = 24
CHECKPOINT_SHA256 = 'f642535b7ca5bc8928bc53c62ab4c85569b545e4f5dab05937c0a5c83aa75e09'
WINDOWS = 705
AGENTS = 2356
COHORTS = ('all', 'single', 'multi', 'N_ge_9')
OUTPUTS = ('PER_SEED.csv', 'PER_WINDOW.csv')


def _read_seed_scores(run):
    """Check independently that anonymous window scores give the same totals."""
    lookup = {}
    for row in read_csv(run / 'PER_SEED.csv'):
        seed, group = int(row['seed']), row['group']
        require(row['fold'] == FOLD and (seed, group) not in lookup, 'invalid or duplicate fold seed score')
        for key in ('scene_count', 'agent_count'):
            row[key] = int(row[key])
        for name in BRANCHES:
            for metric in METRICS:
                key = f'{name}_{metric}'
                row[key] = float(row[key]) if row[key] else None
        lookup[seed, group] = row
    require(set(lookup) == {(seed, group) for seed in SEEDS for group in GROUPS},
            'missing seed or population in early ZARA1 scores')

    partitions = {(seed, group): [] for seed, group in lookup}
    counts = {}
    seen = set()
    for row in read_csv(run / 'PER_WINDOW.csv'):
        seed, ordinal, n = int(row['seed']), row['window'], int(row['n'])
        require(row['fold'] == FOLD and seed in SEEDS and n > 0 and (seed, ordinal) not in seen,
                'invalid or duplicate window score')
        require(counts.setdefault(ordinal, n) == n, 'window population changed across seeds')
        seen.add((seed, ordinal))
        for group in groups_for(n):
            partitions[seed, group].append((n, row))
    expected = {f'test_{index:04d}' for index in range(1, WINDOWS + 1)}
    require(set(counts) == expected and sum(counts.values()) == AGENTS and
            len(seen) == len(SEEDS) * WINDOWS, 'wrong early-test window population')

    for key, subset in partitions.items():
        score = lookup[key]
        windows = len(subset)
        agents = sum(n for n, _ in subset)
        require((windows, agents) == (score['scene_count'], score['agent_count']),
                'group counts do not agree with the anonymous records')
        for name in BRANCHES:
            for metric in METRICS:
                value = score[f'{name}_{metric}']
                if not windows:
                    require(value is None, 'empty group score must be N/A')
                    continue
                if metric in ('JADE', 'JFDE'):
                    total = math.fsum(float(row[f'{name}_{metric}']) for _, row in subset)
                    average = total / windows
                else:
                    total = math.fsum(float(row[f'{name}_agent_{metric}_sum']) for _, row in subset)
                    average = total / agents
                require(value is not None and math.isclose(value, average, rel_tol=1e-12, abs_tol=1e-12),
                        'native score does not reconcile with window/agent weighting')
    return lookup


def _paired_text(row, baseline):
    mean = row[f'new_minus_{baseline}_mean_cm']
    sd = row[f'new_minus_{baseline}_std_cm']
    return 'N/A' if mean is None else f'{mean:+.3f} ± {sd:.3f}'


def _absolute_text(value):
    return 'N/A' if value is None else f'{value:.6f}'


def build_report(run: Path, doc: Path):
    """Check score provenance, then write the Chinese report and anonymous CSVs."""
    config = read(run / 'RUN_CONFIG.json')
    receipt = read(run / 'SCORE_RECEIPT.json')
    wall = read(run / 'WALL_TIME.json')
    barrier = read(run / 'FROZEN_STUDENTS.json')
    selected = barrier['folds'][FOLD]
    require(set(barrier['folds']) == {FOLD} and selected['selected_epoch'] == EPOCH and
            selected['selected_sha256'] == CHECKPOINT_SHA256, 'unexpected early-test checkpoint')
    require(receipt['status'] == 'COMPLETE' and receipt['windows'] == WINDOWS and
            receipt['window_seed_pairs'] == len(SEEDS) * WINDOWS and receipt['K_per_seed'] == 20 and
            set(receipt['split_counts']) == {FOLD}, 'incomplete or wrong-scope score receipt')
    require(receipt['selection_seal_unix'] <= receipt['first_truth_read_unix'], 'truth read before sealing')
    require(receipt['global_prediction_seal_sha256'] == digest(run / 'ALL_PREDICTIONS_SEALED.json'),
            'global prediction seal changed')
    for name in OUTPUTS:
        require(receipt['output_sha256'][name] == digest(run / name), 'scored CSV changed')
    require(math.isfinite(wall['seconds']) and wall['seconds'] >= 0, 'invalid elapsed time')
    lookup = _read_seed_scores(run)
    summary = []
    for group in COHORTS:
        rows = [lookup[seed, group] for seed in SEEDS]
        summary.extend(_summary(FOLD, group, [FOLD] if rows[0]['scene_count'] else [], rows))

    all_rows = {row['metric']: row for row in summary if row['group'] == 'all'}
    diverged = [metric for metric in METRICS if all_rows[metric]['new_minus_B_mean_cm'] >= 0]
    decision = ('ZARA1 全体四项均低于 B。' if not diverged else
                f'ZARA1 全体有 {"、".join(diverged)} 未低于 B。')
    drift = receipt['max_seed_endpoint_metric_drift_m'][FOLD]
    lines = [
        '# ZARA1 直接 K20 学生的提前冻结测试',
        f'**{decision}** 这是历史测试启发的单折五种子探索性对照。五折主线继续按原配置训练与选模。',
        f'使用按开发集规则选定的第 {EPOCH} 轮，权重 SHA256 为 `{CHECKPOINT_SHA256}`。'
        '本次提前评估应用户要求执行，更新次数为 0。',
        f'正式测试覆盖 {WINDOWS} 窗、{AGENTS} 人次，种子为 2035–2039。'
        f'每个种子单独使用 K20，共运行 {receipt["window_seed_pairs"]:,} 次完整 20 步 DDIM。'
        '全部学生轨迹通过封存核验后读取未来并评分。',
        'JADE 和 JFDE 按窗等权汇总，minADE 和 minFDE 按有效行人人次加权。'
        '绝对值为五种子均值，单位米。配对差为同种子学生减基线，单位厘米。'
        '样本标准差由五个配对差直接计算，负值表示学生误差较低。',
        '| 分组 | 窗数／人次 | 指标 | A（m） | B（m） | 学生（m） | 学生−B（cm，均值 ± 样本 SD） | 学生−A（cm，均值 ± 样本 SD） |',
        '|---|---:|---|---:|---:|---:|---:|---:|',
    ]
    for row in summary:
        lines.append(
            f'| {GROUP_LABELS[row["group"]]} | {row["scene_count"]}／{row["agent_count"]} | '
            f'{row["metric"]} | {_absolute_text(row["A_mean_m"])} | {_absolute_text(row["B_mean_m"])} | '
            f'{_absolute_text(row["new_mean_m"])} | {_paired_text(row, "B")} | {_paired_text(row, "A")} |'
        )
    lines.extend([
        f'原生 A/B 回放的最大分数差为 {receipt["original_AB_max_abs_replay_m"]:.3g} m。'
        f'学生逐窗跨种子的 JFDE 和 minFDE 最大差分别为 {drift["JFDE"]:.3g} m、'
        f'{drift["minFDE"]:.3g} m。运行墙钟耗时 {wall["seconds"] / 60:.1f} 分钟。',
        '[指标汇总](SUMMARY.csv) · [五种子明细](PER_SEED.csv) · [匿名逐窗明细](PER_WINDOW.csv) · '
        '[评分收据](SCORE_RECEIPT.json) · [冻结权重](FROZEN_STUDENTS.json) · [运行配置](RUN_CONFIG.json)',
    ])

    doc.mkdir(parents=True, exist_ok=True)
    (doc / 'README.md').write_text('\n\n'.join(lines).replace('|\n\n|', '|\n|') + '\n', encoding='utf-8')
    csvwrite(doc / 'SUMMARY.csv', summary)
    for name in OUTPUTS:
        copyfile(run / name, doc / name)
    for name, value in (
        ('RUN_CONFIG.json', config), ('SCORE_RECEIPT.json', receipt), ('FROZEN_STUDENTS.json', barrier),
        ('WALL_TIME.json', wall), ('ALL_PREDICTIONS_SEALED.json', read(run / 'ALL_PREDICTIONS_SEALED.json')),
    ):
        write(doc / name, public_metadata(value))
    artifacts = {str(path.relative_to(doc)): digest(path) for path in sorted(doc.rglob('*'))
                 if path.is_file() and path.name != 'ARTIFACT_SHA256.json'}
    write(doc / 'ARTIFACT_SHA256.json', artifacts)
    return summary

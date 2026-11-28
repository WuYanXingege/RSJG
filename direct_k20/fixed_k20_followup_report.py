"""Archive the independent HOTEL train-only reselection and UNIV frozen replay."""

from hashlib import sha256
from pathlib import Path
import subprocess
import time

from bcep.contracts import require
from joint_flow_fourfold.report import csvwrite, public_metadata, read_csv

from .common import ASSETS, DOC as ORIGINAL_DOC, METRICS, SEEDS, digest, read, write


BASE = 'fea283129b483d552ca6c7b977413bc92b848876'
RUN_NAME = 'fixed_k20_cv_followup_20261128'
RUN = ASSETS / RUN_NAME
DOC = ORIGINAL_DOC.parent / RUN_NAME
PRIOR_DOC = ORIGINAL_DOC.parent / 'fixed_k20_cv_20261128'
LABEL = dict(all='全体', N1='单人', N2_8='2–8 人', N9plus='N≥9', N9_14='9–14 人', N15plus='N≥15')


def copy_receipt(src, dst, provenance):
    """Publish only reports or CSVs, redacting paths and private members."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix == '.csv':
        dst.write_bytes(src.read_bytes())
    else:
        require(src.suffix == '.json', 'only public receipts and tables can be archived')
        write(dst, public_metadata(read(src)))
    provenance[str(dst.relative_to(DOC))] = {
        'source_sha256': digest(src), 'public_sha256': digest(dst),
    }


def archive_scores(folder, target, fold, provenance):
    result = read(folder / 'SCORE_RECEIPT.json')
    barrier = folder / 'ALL_PREDICTIONS_SEALED.json'
    require(result['status'] == 'COMPLETE'
            and result['prediction_seal_sha256'] == digest(barrier)
            and result['first_truth_read_unix'] >= result['prediction_seal_unix'],
            'score receipt must be bound to an earlier prediction seal')
    seal = read(barrier)['folds'][fold]
    manifest_path = folder / fold / 'PREDICTIONS_FROZEN.json'
    require(digest(manifest_path) == seal['seal_sha256']
            and seal['truth_reads'] == 0 and seal['new_candidate_ddim_calls'] == 0,
            'invalid observation-only prediction manifest')
    manifest = read(manifest_path)
    require(len(manifest['rows']) == result['prediction_pairs'] == seal['prediction_pairs'],
            'prediction and score populations differ')
    pairs = {(row['seed'], row['window']) for row in manifest['rows']}
    require(len(pairs) == result['prediction_pairs'], 'repeated anonymous prediction pair')
    fields = ('seed', 'window', 'n', 'source_sha256', 'sha256')
    csvwrite(target / 'PREDICTION_HASHES.csv',
             [{field: row[field] for field in fields} for row in manifest['rows']])
    for filename, sha in result['output_sha256'].items():
        require(digest(folder / filename) == sha, f'changed score table {filename}')
        copy_receipt(folder / filename, target / filename, provenance)
    window_rows = read_csv(folder / 'PER_WINDOW.csv')
    require(len(window_rows) == len(pairs)
            and {(int(row['seed']), row['window']) for row in window_rows} == pairs,
            'prediction hash list and anonymized scores differ')
    for src in (barrier, folder / 'SCORE_RECEIPT.json', folder / fold / 'EXPORT_CONFIG.json'):
        copy_receipt(src, target / src.name, provenance)
    provenance[str((target / 'PREDICTION_HASHES.csv').relative_to(DOC))] = {
        'source_sha256': digest(manifest_path),
        'public_sha256': digest(target / 'PREDICTION_HASHES.csv'),
    }
    return result, seal


def _cm(value, *, signed=True):
    return 'N/A' if value is None else f"{100 * float(value):{'+.4f' if signed else '.4f'}}"


def _m(value):
    return 'N/A' if value is None else f'{float(value):.9f}'


def _count(value):
    return f'{value:g}'


def _official_table(groups, five_seeds=False):
    lines = [
        '| 分组（窗／人次） | 指标 | S0（m） | 新 K20（m） | Δ（cm） | 改善／持平／恶化窗数 |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for group, summary in groups.items():
        label = f"{LABEL[group]}（{summary['windows']}／{summary['agents']}）"
        for metric in METRICS:
            key = f'delta_{metric}_m'
            delta = _cm(summary[key])
            if five_seeds and summary[f'{key}_sample_sd'] is not None:
                delta += ' ± ' + _cm(summary[f'{key}_sample_sd'], signed=False)
            counts = '／'.join(_count(summary[f'{direction}_{metric}_windows'])
                              for direction in ('improved', 'unchanged', 'worsened'))
            lines.append(f"| {label} | {metric} | {_m(summary[f'base_{metric}_m'])} | "
                         f"{_m(summary[f'final_{metric}_m'])} | {delta} | {counts} |")
    return '\n'.join(lines)


def _diagnostic(groups):
    lines = [
        '| 分组 | 指标 | K21（m） | I（cm） | D（cm） | D−I（cm） |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for group, row in groups.items():
        for metric in METRICS:
            i, d, delta = [row[f'{key}_{metric}_m'] for key in ('I', 'D', 'delta')]
            require(delta is None if i is None else abs((d - i) - delta) <= 1e-12,
                    'K21 decomposition differs from the paired difference')
            lines.append(f"| {LABEL[group]} | {metric} | {_m(row[f'union_{metric}_m'])} "
                         f"| {_cm(i, signed=False)} | {_cm(d, signed=False)} | {_cm(delta)} |")
    return '\n'.join(lines)


def _test_seed_table(rows):
    lines = [
        '| 种子 | 分组（窗／人次） | ΔJADE（cm） | ΔJFDE（cm） | ΔminADE（cm） | ΔminFDE（cm） |',
        '|---:|---|---:|---:|---:|---:|',
    ]
    for row in rows:
        group = f"{LABEL[row['group']]}（{row['windows']}／{row['agents']}）"
        numbers = [_cm(float(row[f'delta_{metric}_m'])) if row[f'delta_{metric}_m'] else 'N/A'
                   for metric in METRICS]
        lines.append('| ' + ' | '.join([row['seed'], group, *numbers]) + ' |')
    return '\n'.join(lines)


def _gates(decision):
    lines = [
        '| 分组（窗） | 指标 | Δ（m，原精度） | 上限（m） | 判定 |',
        '|---|---|---:|---:|---|',
    ]
    for row in decision['checks']:
        outcome = '通过' if row['passed'] else '未通过'
        lines.append(f"| {LABEL[row['group']]}（{row['windows']}） | {row['metric']} | {row['delta_m']!r} "
                     f"| {row['max_delta_m']:.6f} | {outcome} |")
    return '\n'.join(lines)


def _hotel_slot7_comparison(dev, previous):
    lines = [
        '| 分组 | 指标 | 槽 7 Δ（cm） | 本次 Δ（cm） | 少损失 D₇−D（cm） |',
        '|---|---|---:|---:|---:|',
    ]
    for group, row in dev.items():
        old = previous[group]
        for metric in ('JFDE', 'minFDE'):
            require(abs(old[f'I_{metric}_m'] - row[f'I_{metric}_m']) <= 1e-12,
                    'the K21 reference differs across HOTEL slots')
            saved = old[f'D_{metric}_m'] - row[f'D_{metric}_m']
            lines.append(f"| {LABEL[group]} | {metric} | {_cm(old[f'delta_{metric}_m'])} "
                         f"| {_cm(row[f'delta_{metric}_m'])} | {_cm(saved)} |")
    return '\n'.join(lines)


def _prior_results():
    cv = read(PRIOR_DOC / 'cv_development' / 'SCORE_RECEIPT.json')
    center = read(PRIOR_DOC / 'center_development' / 'SCORE_RECEIPT.json')
    require(all(not cv['decision'][fold]['passed'] for fold in ('hotel', 'univ'))
            and not center['decision']['univ']['passed'],
            'follow-up depends on the archived development failures')
    return cv


def _source_versions():
    sources = {}
    for fold, folder, entry in (
        ('hotel', 'hotel_reselection', 'direct_k20/fixed_k20_hotel_reselection.py'),
        ('univ', 'univ_exploratory', 'direct_k20/fixed_k20_univ_exploratory.py'),
    ):
        commit = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', entry], text=True).strip()
        config = RUN / folder / 'RUN_CONFIG.json'
        hashes = read(config)['source_sha256']
        for path, expected in hashes.items():
            blob = subprocess.check_output(['git', 'show', f'{commit}:{path}'])
            require(sha256(blob).hexdigest() == expected, f'{fold} executed source hash differs from Git {path}')
        sources[fold] = dict(commit=commit, entry=entry, source_sha256=hashes, config_sha256=digest(config))
    return sources


def _univ_archive(provenance, previous):
    run, dest = RUN / 'univ_exploratory', DOC / 'univ_test'
    score, seal = archive_scores(run / 'test', dest, 'univ', provenance)
    for name in ('RUN_CONFIG.json', 'FROZEN_TEST.json', 'COMPLETE.json'):
        copy_receipt(run / name, dest / name, provenance)
    replay = run / 'test' / 'univ' / 'S0_REPLAY.json'
    copy_receipt(replay, dest / replay.name, provenance)
    completed = read(run / 'COMPLETE.json')
    require(completed['status'] == 'COMPLETE' and completed['development_gate'] == 'FAIL'
            and completed['config_sha256'] == digest(run / 'RUN_CONFIG.json')
            and completed['test_freeze_sha256'] == digest(run / 'FROZEN_TEST.json')
            and completed['score_receipt_sha256'] == digest(run / 'test' / 'SCORE_RECEIPT.json')
            and completed['training_updates'] == 0,
            'UNIV completion must be bound to the original exploratory freeze and actual score')
    row_by_group = score['groups']['univ']
    dense = row_by_group['N9plus']
    require(score['prediction_pairs'] == 4735
            and score['identities']['univ']['j_drop'] == 3
            and (row_by_group['all']['windows'], row_by_group['all']['agents']) == (947, 24334)
            and (dense['windows'], dense['agents']) == (907, 24099),
            'incorrect UNIV test scope')
    for field in ('windows', 'agents'):
        require(row_by_group['N9_14'][field] + row_by_group['N15plus'][field] == dense[field],
                'the UNIV density split does not cover the whole dense population')
    dev_delta = previous['groups']['univ']['N9plus']['delta_JFDE_m']
    retention = dense['delta_JFDE_m'] / dev_delta * 100
    seed_rows = read_csv(dest / 'PER_SEED.csv')
    require({int(row['seed']) for row in seed_rows} == set(SEEDS)
            and len(seed_rows) == len(SEEDS) * len(row_by_group), 'incomplete UNIV per-seed report')
    retained = (f"降幅为开发组 {_cm(-dev_delta, signed=False)} cm 的 **{retention/100:.2f} 倍**"
                if retention >= 0 else
                f"开发组 {_cm(-dev_delta, signed=False)} cm 的降幅在测试组反转"
                f"（有符号保留率 {retention:.2f}%）")
    header = (f"UNIV 密集测试组（907 窗）的 JFDE 变化为 **{_cm(dense['delta_JFDE_m'])} cm**，"
              f"{retained}。密集组 minFDE 变化为 **{_cm(dense['delta_minFDE_m'])} cm**。")
    endpoint_checks = seal['endpoint_cross_seed_checks']
    if all(stats['bitwise_equal'] for stats in endpoint_checks.values()):
        endpoints = '同窗 S0 和新 K20 的逐槽末帧跨五种子逐位一致，最大坐标差均为 0 m。'
    else:
        endpoints = '跨种子末帧坐标的最大绝对差分别为 ' + '、'.join(
            f"{model} {stats['max_abs_difference_m']:.9g} m" for model, stats in endpoint_checks.items()) + '。'
    smaller, larger = row_by_group['N9_14'], row_by_group['N15plus']
    lines = [
        '## UNIV 五种子探索性冻结复核',
        '固定沿用 S16、零基槽 3、1 个匀速整场候选。该匀速方案的原开发验收保留 **FAIL**。'
        '本次测试在看到开发结果后追加，完成评分即结束该折。',
        _official_table(row_by_group, five_seeds=True),
        f"9–14 人组 JFDE 变化 {_cm(smaller['delta_JFDE_m'])} cm，"
        f"N≥15 组变化 {_cm(larger['delta_JFDE_m'])} cm。"
        f"原开发集最多 13 人，本次测试最多 {row_by_group['all']['max_agents']} 人。",
        '窗数与人次为每种子的固定分母。主表的改善、持平、恶化数为每种子计数的均值，'
        '配对差的标准差为五个独立 K20 配对差的样本标准差。负值表示改善。',
        '### 逐种子配对差',
        _test_seed_table(seed_rows),
        '逐种子三向窗数详见逐种子 CSV。' + endpoints,
        '### K21 诊断',
        _diagnostic(row_by_group),
        f"共封存 4,735 个预测对，位于统一读取真值之前。导出墙钟 {seal['export_wall_seconds']:.2f} 秒，"
        f"评分及核验 {score['scoring_seconds']:.2f} 秒，UNIV 总墙钟 {completed['wall_seconds']:.2f} 秒。"
        f"S0 对原生历史评分的最大绝对差为 {score['native_replay_max_abs_m']['univ']:.3g} m。",
        '[分组 CSV](univ_test/GROUPS.csv) · [逐种子 CSV](univ_test/PER_SEED.csv) · '
        '[匿名逐窗 CSV](univ_test/PER_WINDOW.csv) · [预测哈希](univ_test/PREDICTION_HASHES.csv) · '
        '[评分收据](univ_test/SCORE_RECEIPT.json) · [完成收据](univ_test/COMPLETE.json)',
    ]
    return header, lines, read(replay)['new_DDIM_calls']


def _hotel_archive(provenance, previous):
    run, dest = RUN / 'hotel_reselection', DOC / 'hotel'
    for name in ('RUN_CONFIG.json', 'REPLACEMENT_RANKING.json', 'REPLACEMENT_RANKING.csv',
                 'SELECTION_FROZEN.json', 'RESULT.json'):
        copy_receipt(run / name, dest / name, provenance)
    selection = read(run / 'SELECTION_FROZEN.json')
    result = read(run / 'RESULT.json')
    rows = read_csv(run / 'REPLACEMENT_RANKING.csv')
    require(len(rows) == 20 and {int(row['slot']) for row in rows} == set(range(20)),
            'HOTEL ranking must cover all original slots')
    ranked = read(run / 'REPLACEMENT_RANKING.json')
    selected = selection['selected_slot']
    require(result['status'] == 'COMPLETE' and selected == ranked['selected_slot'] == result['selected_slot']
            and result['selection_sha256'] == digest(run / 'SELECTION_FROZEN.json')
            and selection['train_ranking_sha256'] == digest(run / 'REPLACEMENT_RANKING.json')
            and selection['config_sha256'] == digest(run / 'RUN_CONFIG.json')
            and ranked['table_sha256'] == digest(run / 'REPLACEMENT_RANKING.csv')
            and ranked['development_truth_reads'] == ranked['test_truth_reads'] == result['test_truth_reads'] == 0
            and ranked['training_updates'] == result['training_updates'] == 0,
            'HOTEL completed result must match the train-only frozen selection')
    previous_groups = previous['groups']['hotel']
    lines = [
        '## HOTEL 训练集重选固定槽',
        '在原 train 同时评估 20 个零基替换槽。可行条件为整套 K20 的训练 minFDE ≤ S24。'
        '可行槽按训练 JFDE、原槽号升序，在评分开发集前固定选中项。',
        f"训练覆盖 {ranked['windows']:,} 窗、{ranked['agents']:,} 人次。"
        f"S24 的训练 JFDE 为 {ranked['J_base_m']:.9f} m，minFDE 为 {ranked['F_base_m']:.9f} m。"
        f"排名耗时 {ranked['seconds']:.2f} 秒，HOTEL 总墙钟 {result['wall_seconds']:.2f} 秒。",
        '[20 槽训练表](hotel/REPLACEMENT_RANKING.csv) · [训练排名收据](hotel/REPLACEMENT_RANKING.json) · '
        '[冻结选择](hotel/SELECTION_FROZEN.json) · [完成收据](hotel/RESULT.json)',
        '### 原槽 7 的归档对照',
        _official_table(previous_groups),
        '[原开发报告](../fixed_k20_cv_20261128/README.md) · '
        '[原开发评分收据](../fixed_k20_cv_20261128/cv_development/SCORE_RECEIPT.json)',
    ]
    if selected is None:
        best_f = min(ranked['rows'], key=lambda row: (row['minFDE_m'], row['slot']))
        best_j = min(ranked['rows'], key=lambda row: (row['JFDE_m'], row['slot']))
        lines.extend([
            '20 个槽都未满足训练个人覆盖可行条件。本次 HOTEL 停在训练集排名，保留 S24。',
            f"训练 minFDE 最小的是槽 {best_f['slot']}，仍增加 {_cm(best_f['delta_minFDE_m'])} cm。"
            f"训练 JFDE 最小的是槽 {best_j['slot']}，JFDE 变化 {_cm(best_j['delta_JFDE_m'])} cm，"
            f"minFDE 变化 {_cm(best_j['delta_minFDE_m'])} cm。",
            '### 原槽 7 的六项开发验收',
            _gates(previous['decision']['hotel']),
        ])
        return 'HOTEL 没有满足训练 minFDE 约束的槽，在训练排名后结束，保留 S24。', lines, 0
    if selected == 7:
        lines.extend(['重选结果仍是槽 7，直接引用归档结果。相对槽 7 的 JFDE 和 minFDE 少损失均为 0 cm。',
                      _gates(previous['decision']['hotel'])])
        return 'HOTEL 重选仍为槽 7，沿用原开发 FAIL，保留 S24。', lines, 0

    dev, seal = archive_scores(run / 'cv_development', dest / 'development', 'hotel', provenance)
    groups = dev['groups']['hotel']
    decision = dev['decision']['hotel']
    require((groups['all']['windows'], groups['all']['agents']) == (688, 5203)
            and dev['identities']['hotel']['j_drop'] == selected
            and len(decision['checks']) == 6, 'incorrect HOTEL development scope')
    action = 'HOTEL 候选具备五种子测试资格，本次止于开发集。' if decision['passed'] else '开发验收未通过，保留 S24。'
    lines.extend([
        f'### 本次槽 {selected} 开发集正式 K20',
        _official_table(groups),
        '### 六项验收',
        _gates(decision),
        action,
        '### 与槽 7 比较及 K21 诊断',
        _hotel_slot7_comparison(groups, previous_groups),
        _diagnostic(groups),
        f"688 窗均在读取真值前封存。导出墙钟 {seal['export_wall_seconds']:.2f} 秒，"
        f"评分及核验 {dev['scoring_seconds']:.2f} 秒。",
        '[分组 CSV](hotel/development/GROUPS.csv) · [逐种子 CSV](hotel/development/PER_SEED.csv) · '
        '[匿名逐窗 CSV](hotel/development/PER_WINDOW.csv) · [评分收据](hotel/development/SCORE_RECEIPT.json)',
    ])
    return f'HOTEL 选中槽 {selected}，{action}', lines, result['new_DDIM_calls']


def main():
    DOC.mkdir(parents=True, exist_ok=True)
    prior = _prior_results()
    provenance = {}
    univ_heading, univ, univ_ddim = _univ_archive(provenance, prior)
    hotel_heading, hotel, hotel_ddim = _hotel_archive(provenance, prior)
    elapsed = time.time() - float((RUN / 'START_UNIX').read_text())
    sources = _source_versions()
    write(DOC / 'SOURCE_VERSIONS.json', dict(base_commit=BASE, executions=sources))
    start = [
        '# UNIV 匀速冻结复核与 HOTEL 训练集重选槽',
        univ_heading + ' ' + hotel_heading,
        '两折训练更新均为 0。新方案固定输出 K20。JADE、JFDE 分别择取整场共享槽并按窗等权，'
        'minADE、minFDE 分别逐人择槽并按有效人次加权。Δ 为新方案减 S0，负值表示改善。',
    ]
    end = [
        '## 计算成本与复现',
        'K21 只用于诊断。I 为 S0 减 K21，D 为新 K20 减 K21，逐窗与汇总均核验 Δ＝D−I。'
        '线上仍需先生成 S0 的全部 20 个场景，再替换一槽。新匀速场景无需调用桥。',
        f'两项任务到归档生成的总墙钟为 {elapsed:.2f} 秒。'
        f'已实际新增 DDIM 调用 UNIV {univ_ddim} 次、HOTEL {hotel_ddim} 次。',
        'UNIV 首次启动时修正了历史封存清单读取。实际运行使用逐窗逐种子记录校验人数和哈希。'
        '修复前的尝试在生成新预测、读取测试真值之前退出。',
        '复现入口如下。源码版本表按每折实际配置核对到独立提交。',
        '```bash\npython -m direct_k20.fixed_k20_hotel_reselection\n'
        'python -m direct_k20.fixed_k20_univ_exploratory\n'
        'python -m direct_k20.fixed_k20_followup_report\n```',
        '[UNIV 固定配置](univ_test/RUN_CONFIG.json) · [HOTEL 固定配置](hotel/RUN_CONFIG.json) · '
        '[实际源码版本](SOURCE_VERSIONS.json) · [公开归档哈希](ARTIFACT_SHA256.json)',
    ]
    (DOC / 'README.md').write_text('\n\n'.join(start + univ + hotel + end) + '\n', encoding='utf-8')
    write(DOC / 'ARTIFACT_SHA256.json', provenance | {
        'README.md': {'public_sha256': digest(DOC / 'README.md')},
        'SOURCE_VERSIONS.json': {'public_sha256': digest(DOC / 'SOURCE_VERSIONS.json')},
    })
    print(DOC)


if __name__ == '__main__':
    main()

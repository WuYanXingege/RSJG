"""Publish only a completed, prediction-first frozen ZARA2 test evaluation."""
import csv
import json
import math
from pathlib import Path
import shutil
import statistics
import sys

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from bcep.contracts import digest
from bcep_joint.common import read, write
from joint_flow_frozen_test.inputs import RUN, DOC, BASE_COMMIT, B_SHA

SEEDS = list(range(2035, 2040))
GROUPS = ('all', 'N_ge_9')
METRICS = ('JADE', 'JFDE', 'minADE', 'minFDE')
LABEL = dict(all='全体', N_ge_9='N≥9')


def rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def near(actual, expected, atol=2e-14):
    if not math.isfinite(float(actual)) or not math.isclose(float(actual), float(expected), rel_tol=0., abs_tol=atol):
        raise ValueError(f'report does not reproduce native score: {actual} != {expected}')


def change(value, unit='m'):
    return f"{'下降' if value < 0 else '上升' if value > 0 else '持平'} {abs(value):.12g} {unit}"


def verify(summary, seed_rows, window_rows, native, receipt, seal):
    if summary['seeds'] != SEEDS or summary['ddof'] != 1 or summary['K_per_seed'] != 20:
        raise ValueError('five independent K20 seeds and sample standard deviation are required')
    if len(seed_rows) != 10 or len(window_rows) != 4990:
        raise ValueError('wrong frozen evaluation table size')
    if receipt['status'] != 'COMPLETE' or receipt['windows'] != 998 or receipt['future_reads_this_complete_pass'] != 998:
        raise ValueError('scoring did not complete its fixed population')
    if not receipt['all_predictions_verified_before_any_truth'] or not receipt['future_read_once_per_window_for_all_five_seeds']:
        raise ValueError('prediction-first / one-future-read contract failed')
    if not (receipt['seal_unix'] < receipt['verification_complete_unix'] < receipt['first_truth_read_unix']):
        raise ValueError('truth read did not follow the complete prediction seal verification')
    if receipt['prediction_files_verified'] != 4990 or receipt['seeds'] != SEEDS or receipt['seed_pooling']:
        raise ValueError('wrong prediction population or seeds were pooled')
    if receipt['parameter_updates'] != 0 or receipt['selection_on_test'] or receipt['development_thresholds_applied']:
        raise ValueError('frozen test protocol changed')
    if digest(RUN/'PREDICTIONS_FROZEN.json') != receipt['prediction_manifest_sha256']:
        raise ValueError('sealed manifest changed after scoring')
    if digest(RUN/'RUN_CONFIG.json') != seal['source_config_sha256'] or seal['windows'] != 998 or seal['seeds'] != SEEDS:
        raise ValueError('prediction configuration identity changed')
    if digest(RUN/'PRE_SCORE_SEAL_VERIFICATION.json') != receipt['pre_score_seal_verification_sha256']:
        raise ValueError('pre-score verification receipt changed')
    for name, sha in receipt['output_sha256'].items():
        if digest(RUN/name) != sha:
            raise ValueError('scored output changed: '+name)
    unique = {(int(r['seed']), r['window']) for r in window_rows}
    expected = {(seed, f'test_{i:04d}') for seed in SEEDS for i in range(1, 999)}
    if unique != expected or len(unique) != len(window_rows):
        raise ValueError('missing, repeated, or nonanonymous test window')
    by_seed_group = {(int(r['seed']), r['group']): r for r in seed_rows}
    if set(by_seed_group) != {(seed, group) for seed in SEEDS for group in GROUPS}:
        raise ValueError('wrong seed/group rows')
    for seed in SEEDS:
        for group in GROUPS:
            rr = [r for r in window_rows if int(r['seed']) == seed and (group == 'all' or int(r['n']) >= 9)]
            counts = dict(scene_count=len(rr), agent_count=sum(int(r['n']) for r in rr))
            if counts != receipt['groups'][group]:
                raise ValueError('scored group population changed across seeds')
            seed_row = by_seed_group[seed, group]
            for key, value in counts.items():
                if int(seed_row[key]) != value or summary['groups'][group][key] != value:
                    raise ValueError('summary population mismatch')
            for branch in ('A', 'B'):
                result = native[str(seed)][group][branch]
                for metric in METRICS:
                    aggregate = (sum(float(r[branch+'_'+metric]) for r in rr)/len(rr) if metric in ('JADE', 'JFDE')
                                 else sum(float(r[branch+'_agent_'+metric+'_sum']) for r in rr)/counts['agent_count'])
                    near(aggregate, result[metric])
                    near(result[metric], seed_row[branch+'_'+metric], atol=0.)
            for metric in METRICS:
                near(seed_row['B_minus_A_'+metric], float(seed_row['B_'+metric])-float(seed_row['A_'+metric]), atol=0.)
                for r in rr:
                    near(r['B_minus_A_'+metric], float(r['B_'+metric])-float(r['A_'+metric]), atol=0.)
    for group in GROUPS:
        for branch in ('A', 'B', 'B_minus_A'):
            for metric in METRICS:
                values = [float(by_seed_group[seed, group][branch+'_'+metric]) for seed in SEEDS]
                result = summary['groups'][group][branch][metric]
                near(result['mean'], math.fsum(values)/5, atol=0.)
                near(result['std'], statistics.stdev(values), atol=0.)
    interpretation = summary['interpretation']
    for key, group, metric in (('all_JADE_lower', 'all', 'JADE'), ('all_JFDE_lower', 'all', 'JFDE'),
                               ('N_ge_9_JFDE_lower', 'N_ge_9', 'JFDE')):
        if interpretation[key] != (summary['groups'][group]['B_minus_A'][metric]['mean'] < 0):
            raise ValueError('test conclusion direction mismatch')
    for group in GROUPS:
        near(interpretation['marginal_minFDE_change_cm'][group],
             100*summary['groups'][group]['B_minus_A']['minFDE']['mean'], atol=0.)


def main():
    if read(RUN/'STATUS.json')['status'] != 'COMPLETE':
        raise RuntimeError('final test report requires a completed real run')
    config = read(RUN/'RUN_CONFIG.json')
    seal = read(RUN/'PREDICTIONS_FROZEN.json')
    receipt = read(RUN/'SCORE_RECEIPT.json')
    summary = read(RUN/'SUMMARY.json')
    native = read(RUN/'NATIVE_METRICS.json')
    window_rows = rows(RUN/'PER_WINDOW.csv')
    seed_rows = rows(RUN/'PER_SEED.csv')
    replay = read(RUN/'DEV_REPLAY.json')
    frozen = read(RUN/'FROZEN_MODELS.json')
    verify(summary, seed_rows, window_rows, native, receipt, seal)
    if not (replay['passed'] and replay['seed'] == 2035 and replay['windows'] == 590 and replay['dense_windows'] == 127
            and replay['exact_metrics'] and replay['metrics'] == replay['reference']):
        raise ValueError('seed2035 development replay did not exactly reproduce the frozen reference')
    if replay['reference_commit'] != BASE_COMMIT:
        raise ValueError('wrong frozen development reference')
    if not (frozen['exact_unchanged'] and frozen['before'] == frozen['after'] and frozen['requires_grad_false']
            and frozen['optimizer_updates'] == 0):
        raise ValueError('frozen model verification failed')
    if seal['identity']['B_checkpoint_sha256'] != B_SHA or seal['identity']['B_completed_epoch'] != 35:
        raise ValueError('test used a different dependency checkpoint')
    DOC.mkdir(parents=True, exist_ok=True)
    names = ('RUN_CONFIG.json', 'DEV_REPLAY.json', 'DEV_REPLAY.csv', 'FROZEN_MODELS.json',
             'PRE_SCORE_SEAL_VERIFICATION.json', 'SCORE_RECEIPT.json', 'NATIVE_METRICS.json',
             'PER_WINDOW.csv', 'PER_SEED.csv', 'SUMMARY.json', 'SUMMARY.csv')
    for name in names:
        shutil.copyfile(RUN/name, DOC/name)
    for name in ('PROTECTED_RUNS.json', 'A_BANKS.json', 'GPU_RELEASE.json'):
        if (RUN/name).exists():
            shutil.copyfile(RUN/name, DOC/name)
    # Raw member indices and index-bearing host filenames remain on the host.
    public_seal = dict(schema='anonymous-frozen-test-predictions-v1',
        original_manifest_sha256=digest(RUN/'PREDICTIONS_FROZEN.json'), identity=seal['identity'],
        source_config_sha256=seal['source_config_sha256'], seal_unix=seal['seal_unix'],
        windows=seal['windows'], seeds=seal['seeds'], prediction_files=len(seal['rows']),
        K_per_seed=20, prediction_frames=12,
        rows=[dict(seed=r['seed'], window=r['window'], n=r['n'], prediction_sha256=r['sha256']) for r in seal['rows']])
    write(DOC/'PREDICTIONS_FROZEN_PUBLIC.json', public_seal)
    main_rows = []
    seed_table = []
    for group in GROUPS:
        g = summary['groups'][group]
        for branch, label in (('A', 'A：旧 H0 coordinate epoch5'), ('B', 'B：固定依赖模型 epoch35'), ('B_minus_A', 'B−A 配对差')):
            values = ' | '.join(f"{g[branch][m]['mean']:+.9f} ± {g[branch][m]['std']:.9f}" if branch == 'B_minus_A'
                                else f"{g[branch][m]['mean']:.9f} ± {g[branch][m]['std']:.9f}" for m in METRICS)
            main_rows.append(f"| {LABEL[group]} | {g['scene_count']} | {g['agent_count']} | {label} | {values} |")
        for seed in SEEDS:
            row = next(r for r in seed_rows if r['group'] == group and int(r['seed']) == seed)
            values = ' | '.join(f"{float(row['A_'+m]):.9f} / {float(row['B_'+m]):.9f} / {float(row['B_minus_A_'+m]):+.9f}" for m in METRICS)
            seed_table.append(f'| {LABEL[group]} | {seed} | {values} |')
    conclusions = []
    for label, group, metric in (('全体 JADE', 'all', 'JADE'), ('全体 JFDE', 'all', 'JFDE'), ('N≥9 JFDE', 'N_ge_9', 'JFDE')):
        value = summary['groups'][group]['B_minus_A'][metric]['mean']
        conclusions.append(f'{label} {change(value)}')
    all_lower = all(summary['interpretation'][key] for key in ('all_JADE_lower', 'all_JFDE_lower', 'N_ge_9_JFDE_lower'))
    verdict = ('固定 B 在本次测试的五 seed 均值上仍降低全体 JADE、全体 JFDE 和 N≥9 JFDE。' if all_lower else
               '固定 B 未在本次测试的五 seed 均值上同时降低全体 JADE、全体 JFDE 和 N≥9 JFDE；各项实际方向如下。')
    marginal = '；'.join(f"{LABEL[group]}边缘 minFDE {change(summary['interpretation']['marginal_minFDE_change_cm'][group], 'cm')}" for group in GROUPS)
    dense = summary['groups']['N_ge_9']
    source_commit = config.get('runtime_source_commit', config.get('source_commit'))
    if not source_commit:
        raise ValueError('fixed runtime source commit absent from run configuration')
    code_base = f'https://github.com/WuYanXingege/RSJG/blob/{source_commit}/'
    document = f'''# ZARA2 固定模型五 seed 测试结果

{verdict}

比较对象均为同一组 **998 个测试窗口**。其中 N≥9 为 **{dense['scene_count']} 窗、{dense['agent_count']} 人次**；全体为 **{summary['groups']['all']['agent_count']} 人次**。B 固定使用开发集选出的 epoch35，测试过程中没有训练、模型选择、阈值选择或额外 seed。下述差值一律为同一 seed 下 B−A，负值表示 B 较低。

{'；'.join(conclusions)}。{marginal}。这些结论只比较同测试集、同 seed 的 A/B，**不套用开发集三项门槛**，也不作为其他折的结果。

## 五 seed 主结果

每个 seed 独立产生 K20、12 帧预测，使用原生 `hard_metric_statistics` 和 `MetricAccumulator` 在 CPU 上评分。JADE/JFDE 先对同一联合场景内的有效行人平均、再对 20 个联合场景取最小值，最后按窗口等权；边缘 minADE/minFDE 按有效行人加权。五个 seed 为 2035、2036、2037、2038、2039；没有合并成 K100。

下表单位为米，格式为五 seed **均值 ± 样本标准差（ddof=1）**。B−A 行的标准差直接由五个配对 seed 差计算，不是 A/B 标准差的独立合并，也不是置信区间。

| 分组 | 窗数 | 人次 | 输出 | JADE | JFDE | minADE | minFDE |
|---|---:|---:|---|---:|---:|---:|---:|
{chr(10).join(main_rows)}

## 每个 seed 的配对结果

每格按 **A / B / B−A** 顺序列出，单位为米。原生完整累加值和人数见 [逐 seed CSV](PER_SEED.csv) 与 [原生累加器记录](NATIVE_METRICS.json)。

| 分组 | seed | JADE：A / B / 差 | JFDE：A / B / 差 | minADE：A / B / 差 | minFDE：A / B / 差 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(seed_table)}

## 冻结身份与先预测后评分

A 为旧 v2 coordinate epoch5，checkpoint SHA256 `{seal['identity']['checkpoint_sha256']}`。B 固定为 epoch35，checkpoint SHA256 `{B_SHA}`，开发集归档提交为 `{BASE_COMMIT}`。Provider density19、修正头 epoch9、原编码器、原配对头、六层桥与依赖网络均冻结为 eval；本次没有优化步骤，测试前后模型状态哈希完全相同。B 采用已冻结的连续联合采样、256→20 整组压缩、一次有界修正及一次 20 步 DDIM；A 使用旧 H0 原生入口。SSEC 保持原监督器记录的暂停状态，既有训练队列及原实验任务状态均未改变。

正式测试之前，seed2035 在全部 590 个开发窗口（其中 127 窗 N≥9）复现 A/B 的归档原生指标，结果完全一致；见 [开发集复现](DEV_REPLAY.json)。该复现只检查冻结入口，没有重新选模。

全部 **4,990 个预测文件**先封存为 float32 的 A/B 两套 `[20,N,12,2]` 数组，文件中不含真值。评分入口先核对完整清单、固定配置身份、每个文件的 SHA256、形状、类型与有限性；完成全部核对后，才构造 `FormalData(..., 'scoring')` 并读取测试未来。每个窗口只读取一次未来，随后独立评分五个 seed。封存时间 `{receipt['seal_unix']}`、全部核验完成时间 `{receipt['verification_complete_unix']}`、首次未来读取时间 `{receipt['first_truth_read_unix']}`，均为 Unix 秒，顺序已核验。

原始封存清单 SHA256 为 `{receipt['prediction_manifest_sha256']}`。公开 [匿名预测封存清单](PREDICTIONS_FROZEN_PUBLIC.json) 保留每个匿名窗口/seed 的预测文件哈希；原始成员索引、预测文件、权重、候选库和轨迹保留在训练主机。

## 固定代码与交付

运行源码提交：`{source_commit}`。[评分及封存检查]({code_base}joint_flow_frozen_test/scoring.py) · [报告生成器]({code_base}scripts/report_joint_flow_frozen_test.py)。

- [运行配置](RUN_CONFIG.json) · [测试汇总 JSON](SUMMARY.json) · [测试汇总 CSV](SUMMARY.csv)
- [匿名逐窗配对差](PER_WINDOW.csv) · [逐 seed 指标](PER_SEED.csv) · [原生指标](NATIVE_METRICS.json)
- [开发集 seed2035 复现](DEV_REPLAY.json) · [开发集匿名逐窗记录](DEV_REPLAY.csv) · [冻结模型核验](FROZEN_MODELS.json)
- [预测封存清单](PREDICTIONS_FROZEN_PUBLIC.json) · [评分前完整预测核验](PRE_SCORE_SEAL_VERIFICATION.json) · [评分收据](SCORE_RECEIPT.json) · [报告核验](REPORT_VERIFICATION.json)

公开逐窗 CSV 仅含匿名窗口编号、seed、人数、指标和配对差，不含场景原名、行人 ID 或原始坐标。测试集没有参与拟合、早停、超参数选择或模型选择。
'''
    (DOC/'README.md').write_text(document)
    write(DOC/'REPORT_VERIFICATION.json', dict(status='COMPLETE', source_run_complete=True,
        windows=998, groups=receipt['groups'], seeds=SEEDS, K_per_seed=20, ddof=1,
        native_metrics_reconstructed_from_anonymous_windows=True,
        paired_seed_standard_deviation_recomputed=True, prediction_seal_precedes_truth_verified=True,
        dev_seed2035_exact_replay=True, frozen_weights_unchanged=True,
        prediction_manifest_sha256=receipt['prediction_manifest_sha256'],
        B_checkpoint_sha256=B_SHA, optimizer_updates=0, reporting_inference_calls=0))
    for path in DOC.rglob('*.csv'):
        path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
    for path in DOC.rglob('*'):
        if path.is_file():
            if path.suffix not in ('.json', '.csv', '.md', '.txt'):
                raise ValueError('binary artifact in public report')
            if any(token in path.read_bytes() for token in (b'/media/ee615', b'/home/ee615')):
                raise ValueError('host absolute path in public artifact')
    write(DOC/'ARTIFACT_SHA256.json', {str(path.relative_to(DOC)): digest(path) for path in sorted(DOC.rglob('*'))
        if path.is_file() and path.name != 'ARTIFACT_SHA256.json'})
    print(json.dumps(dict(report=str(DOC.relative_to(SOURCE)/'README.md'),
                         interpretation=summary['interpretation']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

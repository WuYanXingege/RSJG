"""Read-only score reconciliation and fold-equal five-seed reporting.

No prediction, future trajectory, checkpoint tensor, model, or optimizer is
loaded here. A missing fold is explicitly partial, never replaced by zero.
"""
import csv
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess

from bcep.contracts import canonical_hash, digest, require
from bcep_joint.common import read, write
from .common import SOURCE, BASE_COMMIT, FOLDS, SEEDS

GROUPS = ('all', 'N_ge_9')
METRICS = ('JADE', 'JFDE', 'minADE', 'minFDE')
BRANCHES = ('A', 'B', 'B_minus_A')
LEGACY = 'docs/research/continuous_joint_flow/zara2_frozen_test_20261122'
LABEL = {'all': '全体', 'N_ge_9': 'N≥9', 'new_four': '新四折', 'all_five': '含 ZARA2 的五折'}


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def csvwrite(path, records):
    require(bool(records), 'refuse empty report CSV')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(records)
    temp.replace(path)


def near(actual, expected, atol=2e-14):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected)) and
            math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=0.),
            f'report reconciliation mismatch: {actual} != {expected}')


def mean_std(values):
    require(len(values) == 5 and all(math.isfinite(v) for v in values), 'expected five finite seed values')
    return dict(mean=math.fsum(values)/5, std=statistics.stdev(values))


def summarize_fold(seed_rows):
    """Pure function; preserves within-fold native weighting already applied."""
    lookup = {(int(row['seed']), row['group']): row for row in seed_rows}
    require(len(lookup) == len(seed_rows) == 10 and set(lookup) == {(s, g) for s in SEEDS for g in GROUPS},
            'missing or repeated fold seed/group')
    groups = {}
    for group in GROUPS:
        rows = [lookup[seed, group] for seed in SEEDS]
        counts = {name: int(rows[0][name]) for name in ('scene_count', 'agent_count')}
        require(all(all(int(row[k]) == v for k, v in counts.items()) for row in rows), 'seed populations differ')
        available = counts['scene_count'] > 0
        require((counts['agent_count'] > 0) == available, 'inconsistent empty group')
        result = dict(**counts, available=available)
        for branch in BRANCHES:
            result[branch] = ({metric: mean_std([float(row[branch+'_'+metric]) for row in rows]) for metric in METRICS}
                              if available else None)
        for row in rows:
            for metric in METRICS:
                if available:
                    near(float(row['B_minus_A_'+metric]), float(row['B_'+metric])-float(row['A_'+metric]), atol=0.)
                else:
                    require(all(row[branch+'_'+metric] in ('', None) for branch in BRANCHES), 'empty dense group is not N/A')
        groups[group] = result
    return groups


def aggregate_fold_means(per_fold_seed_rows, scopes=None):
    """First average folds equally inside each seed; then summarize five seeds.

    Agent/window counts are descriptive only and never used as fold weights.
    Paired-difference SD is calculated from five fold-averaged paired seed
    differences, not by combining A/B SD or per-fold SD.
    """
    scopes = scopes or {'new_four': list(FOLDS), 'all_five': list(FOLDS)+['zara2']}
    fold_groups = {fold: summarize_fold(rows) for fold, rows in per_fold_seed_rows.items()}
    lookups = {fold: {(int(r['seed']), r['group']): r for r in rows} for fold, rows in per_fold_seed_rows.items()}
    result, per_seed, summary_rows = {}, [], []
    for scope, requested in scopes.items():
        included = [fold for fold in requested if fold in lookups]
        missing = [fold for fold in requested if fold not in lookups]
        item = dict(status='COMPLETE' if not missing else 'PARTIAL', requested_folds=requested,
                    included_folds=included, missing_folds=missing, groups={})
        for group in GROUPS:
            participants = [fold for fold in included if fold_groups[fold][group]['available']]
            empty = [fold for fold in included if fold not in participants]
            counts = {name: sum(fold_groups[fold][group][name] for fold in participants)
                      for name in ('scene_count', 'agent_count')}
            group_seed_rows = []
            for seed in SEEDS:
                row = dict(scope=scope, group=group, seed=seed, participating_fold_count=len(participants),
                           participating_folds=';'.join(participants), **counts)
                for branch in BRANCHES:
                    for metric in METRICS:
                        row[branch+'_'+metric] = (math.fsum(float(lookups[fold][seed, group][branch+'_'+metric])
                                                             for fold in participants)/len(participants)
                                                        if participants else None)
                group_seed_rows.append(row); per_seed.append(row)
            group_result = dict(available=bool(participants), participating_fold_count=len(participants),
                participating_folds=participants, completed_but_empty_folds=empty, **counts)
            for branch in BRANCHES:
                group_result[branch] = ({metric: mean_std([row[branch+'_'+metric] for row in group_seed_rows])
                                         for metric in METRICS} if participants else None)
                row = dict(scope=scope, scope_status=item['status'], group=group, branch=branch,
                           participating_fold_count=len(participants), participating_folds=';'.join(participants),
                           **counts, seed_count=5, ddof=1)
                for metric in METRICS:
                    row[metric+'_mean'] = group_result[branch][metric]['mean'] if participants else None
                    row[metric+'_std'] = group_result[branch][metric]['std'] if participants else None
                summary_rows.append(row)
            item['groups'][group] = group_result
        result[scope] = item
    return dict(schema='fold-equal-paired-five-seed-summary-v1', seeds=list(SEEDS), ddof=1, K_per_seed=20,
        aggregation='equal fold mean within each seed, then mean and sample SD across five paired seeds',
        fold_weights='equal; neither windows nor agents are used as fold weights',
        within_fold_joint_weighting='equal windows', within_fold_marginal_weighting='valid agents',
        difference_standard_deviation='five paired seed differences after equal-fold averaging',
        seed_variation='inference randomness at fixed weights; not independent training repetitions', scopes=result), per_seed, summary_rows


def validate_completed_fold(folder, fold):
    """Check saved anonymous statistics only; never open predictions or truth."""
    folder = Path(folder)
    receipt = read(folder/'SCORE_RECEIPT.json')
    require(receipt['status'] == 'COMPLETE', 'incomplete score receipt')
    require(receipt.get('fold', fold) == fold, 'score receipt belongs to another fold')
    require(receipt['seeds'] == list(SEEDS) and receipt['K_per_seed'] == 20 and not receipt['seed_pooling'], 'wrong seed/K protocol')
    require(receipt['parameter_updates'] == 0 and not receipt['selection_on_test'], 'test changed frozen weights/selection')
    require(receipt['all_predictions_verified_before_any_truth'] and
            receipt['future_read_once_per_window_for_all_five_seeds'], 'prediction/truth barrier missing')
    require(receipt['seal_unix'] < receipt['verification_complete_unix'] < receipt['first_truth_read_unix'],
            'truth access preceded prediction sealing')
    require(receipt['future_reads_this_complete_pass'] == receipt['windows'], 'incorrect future read population')
    for name, sha in receipt['output_sha256'].items():
        require(Path(name).name == name and digest(folder/name) == sha, 'score output hash changed')
    seed_rows, windows = read_csv(folder/'PER_SEED.csv'), read_csv(folder/'PER_WINDOW.csv')
    require(all(row.get('fold', fold) == fold for row in seed_rows+windows), 'CSV belongs to another fold')
    groups = summarize_fold(seed_rows)
    native = read(folder/'NATIVE_METRICS.json')
    saved = read(folder/'SUMMARY.json')
    expected_ids = {(seed, f'test_{i:04d}') for seed in SEEDS for i in range(1, receipt['windows']+1)}
    require(len(windows) == len(expected_ids) and {(int(r['seed']), r['window']) for r in windows} == expected_ids,
            'missing/duplicate/nonanonymous window')
    require(not any(key.lower() in ('index', 'scene_id', 'agent_ids', 'agent_id', 'future', 'observed', 'x', 'y')
                    for key in windows[0]), 'raw identifier/coordinate field in public window CSV')
    for group in GROUPS:
        for name in ('scene_count', 'agent_count'):
            require(groups[group][name] == receipt['groups'][group][name] == saved['groups'][group][name], 'group count mismatch')
        for branch in BRANCHES:
            if groups[group]['available']:
                for metric in METRICS:
                    for key in ('mean', 'std'):
                        near(groups[group][branch][metric][key], saved['groups'][group][branch][metric][key], atol=0.)
            else:
                require(saved['groups'][group][branch] is None, 'empty group summary must be N/A')
        for seed in SEEDS:
            subset = [r for r in windows if int(r['seed']) == seed and (group == 'all' or int(r['n']) >= 9)]
            require(len(subset) == groups[group]['scene_count'] and sum(int(r['n']) for r in subset) == groups[group]['agent_count'],
                    'window population differs from native statistics')
            if not subset:
                require(native[str(seed)][group] is None, 'native empty group must be null')
                continue
            seed_row = next(r for r in seed_rows if int(r['seed']) == seed and r['group'] == group)
            for branch in ('A', 'B'):
                for metric in METRICS:
                    value = (sum(float(r[branch+'_'+metric]) for r in subset)/len(subset) if metric in ('JADE', 'JFDE') else
                             sum(float(r[branch+'_agent_'+metric+'_sum']) for r in subset)/groups[group]['agent_count'])
                    near(value, native[str(seed)][group][branch][metric])
                    near(value, seed_row[branch+'_'+metric])
            for r in subset:
                for metric in METRICS:
                    near(r['B_minus_A_'+metric], float(r['B_'+metric])-float(r['A_'+metric]), atol=0.)
    return seed_rows, [dict(r, fold=fold) for r in windows], groups, receipt


PRIVATE_FIELDS = {'members', 'records', 'train_indices', 'selection_dev_indices', 'heldout_confirmation_indices',
                  'agent_ids', 'agent_id', 'scene_id', 'scene_ids', 'frame_ids', 'official_frame_ids',
                  'source_sequence', 'sequence', 'index', 'indices'}


def public_metadata(value):
    """Keep fixed hashes/counters, anonymize raw membership and host paths."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if isinstance(key, str) and key.startswith('/'):
                key = 'host_path_key_sha256_'+canonical_hash(key)
            if key in PRIVATE_FIELDS:
                result[key+'_redacted_sha256'] = canonical_hash(item)
                if isinstance(item, (list, dict)):
                    result[key+'_count'] = len(item)
            else:
                result[key] = public_metadata(item)
        return result
    if isinstance(value, list):
        return [public_metadata(item) for item in value]
    if isinstance(value, str) and value.startswith('/'):
        return '<host-path-sha256:'+canonical_hash(value)+'>'
    return value


def _copy_metadata(source, target, provenance):
    source, target = Path(source), Path(target)
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    changed = False
    if source.suffix == '.json':
        original = read(source); public = public_metadata(original)
        changed = original != public
        if changed:
            write(target, public)
        else:
            shutil.copyfile(source, target)
    elif source.suffix == '.csv':
        target.write_bytes(source.read_bytes().replace(b'\r\n', b'\n'))
    else:
        raise ValueError('only whitelisted JSON/CSV metadata may be published')
    provenance.append(dict(artifact=str(target), original_sha256=digest(source), public_sha256=digest(target),
                           identifiers_or_host_paths_redacted=changed))


def _format(value):
    return 'N/A' if value is None else f"{value['mean']:.9f} ± {value['std']:.9f}"


def _direction(value, unit='cm'):
    if value is None:
        return 'N/A（该折没有 N≥9 测试窗）'
    return f"{'下降' if value < 0 else '上升' if value > 0 else '持平'} {abs(value):.9g} {unit}"


def build_report(run_root, destination):
    """Archive completed folds and report explicit partial/full fold coverage."""
    run_root, destination = Path(run_root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    per_fold, fold_windows, fold_groups, availability = {}, {}, {}, {}
    provenance = []
    test_names = ('RUN_CONFIG.json', 'PER_WINDOW.csv', 'PER_SEED.csv', 'SUMMARY.json', 'SUMMARY.csv',
                  'NATIVE_METRICS.json', 'SCORE_RECEIPT.json', 'PRE_SCORE_SEAL_VERIFICATION.json',
                  'FROZEN_MODELS.json', 'A_BANKS.json')
    for fold in FOLDS:
        folder = run_root/fold/'test'
        if not (folder/'SCORE_RECEIPT.json').exists() or read(folder/'SCORE_RECEIPT.json').get('status') != 'COMPLETE':
            availability[fold] = dict(status='NOT_COMPLETED', included=False,
                current_status=read(run_root/fold/'STATUS.json') if (run_root/fold/'STATUS.json').exists() else None)
            continue
        per_fold[fold], fold_windows[fold], fold_groups[fold], receipt = validate_completed_fold(folder, fold)
        availability[fold] = dict(status='COMPLETE', included=True, groups=receipt['groups'])
        for name in test_names:
            _copy_metadata(folder/name, destination/fold/'test'/name, provenance)
        seal_path = folder/'PREDICTIONS_FROZEN.json'
        seal = read(seal_path)
        require(digest(seal_path) == receipt['prediction_manifest_sha256'], 'completed prediction seal changed')
        public_seal = dict(schema='anonymous-fold-predictions-v1', fold=fold, windows=seal['windows'], seeds=seal['seeds'],
            identity=public_metadata(seal['identity']), source_config_sha256=seal['source_config_sha256'],
            seal_unix=seal['seal_unix'], original_manifest_sha256=digest(seal_path),
            rows=[dict(seed=r['seed'], window=r['window'], n=r['n'], prediction_sha256=r['sha256']) for r in seal['rows']])
        write(destination/fold/'test/PREDICTIONS_FROZEN_PUBLIC.json', public_seal)
    # Existing ZARA2 is consumed as immutable archived scores, with no inference.
    legacy_folder = SOURCE/LEGACY
    legacy_names = ('PER_SEED.csv', 'PER_WINDOW.csv', 'SUMMARY.json', 'NATIVE_METRICS.json', 'SCORE_RECEIPT.json')
    for name in legacy_names:
        original = subprocess.check_output(['git', 'show', BASE_COMMIT+':'+LEGACY+'/'+name], cwd=SOURCE)
        require((legacy_folder/name).read_bytes() == original, 'existing ZARA2 score changed from fixed commit')
    per_fold['zara2'], fold_windows['zara2'], fold_groups['zara2'], legacy_receipt = validate_completed_fold(legacy_folder, 'zara2')
    availability['zara2'] = dict(status='REUSED_COMPLETE_ARCHIVE', included=True, reference_commit=BASE_COMMIT,
                               groups=legacy_receipt['groups'], new_inference_calls=0)
    for name in test_names:
        _copy_metadata(legacy_folder/name, destination/'zara2/test'/name, provenance)
    write(destination/'ZARA2_REUSE.json', dict(reference_commit=BASE_COMMIT, source_directory=LEGACY,
        hashes={name: digest(legacy_folder/name) for name in legacy_names}, new_predictions=0,
        A_epoch=5, A_original_dev_rank=2, B_epoch=35, B_original_dev_rank=1))
    summary, aggregate_seeds, aggregate_rows = aggregate_fold_means(per_fold)
    summary['availability'] = public_metadata(availability)
    write(destination/'SUMMARY.json', summary)
    csvwrite(destination/'PER_SEED.csv', aggregate_seeds)
    csvwrite(destination/'SUMMARY.csv', aggregate_rows)
    combined = [r for fold in list(FOLDS)+['zara2'] if fold in fold_windows for r in fold_windows[fold]]
    # Normalize column order between the old ZARA2 CSV and the new fold CSVs.
    columns = ['fold']+[k for k in combined[0] if k != 'fold']
    csvwrite(destination/'PER_WINDOW.csv', [{key: row[key] for key in columns} for row in combined])
    for name in ('RUN_CONFIG.json', 'ASSET_INVENTORY.json', 'STATUS.json', 'FROZEN_MODELS.json',
                 'PROTECTED_RUNS.json', 'GPU_RELEASE.json'):
        _copy_metadata(run_root/name, destination/name, provenance)
    training_table = []
    training = {}
    for fold in FOLDS:
        for relative in ('UPSTREAM_FROZEN.json', 'B_FROZEN.json', 'dependency/RUN_CONFIG.json',
                         'dependency/TRAINING_RESULT.json', 'dependency/FROZEN_MODELS.json', 'dependency/PREFLIGHT.json',
                         'dependency/dependence/EPOCH_METRICS.json', 'dependency/dependence/TOP5.json',
                         'dependency/dependence/STATUS.json'):
            _copy_metadata(run_root/fold/relative, destination/fold/relative, provenance)
        upstream_root = run_root/fold/'upstream'
        upstream_manifest = run_root/fold/'UPSTREAM_FROZEN.json'
        if not upstream_manifest.exists():
            upstream_manifest = upstream_root/'UPSTREAM_FROZEN.json'
        upstream = read(upstream_manifest) if upstream_manifest.exists() else None
        for name in ('RUN_CONFIG.json', 'ADAPTER_CONFIG.json', 'UPSTREAM_FROZEN.json', 'UPSTREAM_STATUS.json'):
            _copy_metadata(upstream_root/name, destination/fold/'upstream'/name, provenance)
        for stage in ('density', 'head', 'joint'):
            stage_folder = Path(upstream[stage]['stage_directory']) if upstream else upstream_root/stage
            for name in ('RUN_CONFIG.json', 'ADAPTER_CONFIG.json', 'EPOCH_METRICS.json', 'TOP5.json', 'FROZEN.json'):
                _copy_metadata(stage_folder/name, destination/fold/'upstream'/stage/name, provenance)
            _copy_metadata(upstream_root/'receipts'/f'{stage}.json',
                           destination/fold/'upstream/receipts'/f'{stage}.json', provenance)
        selected = run_root/fold/'dependency/TRAINING_RESULT.json'
        if selected.exists():
            value = read(selected); training[fold] = value
            require(upstream is not None, 'completed dependency lacks upstream frozen identity')
            a_epoch = upstream['joint']['frozen']['epoch']
            a_sha = upstream['joint']['sha256']
            training_table.append(f"| {fold} | {a_epoch} | `{a_sha}` | {value['completed_epochs']} | {value['rank1_epoch']} | `{value['rank1_sha256']}` |")
        else:
            training_table.append(f'| {fold} | N/A | N/A | 未完成 | N/A | N/A |')
    main_table, effect_table, effect_text = [], [], []
    for fold in list(FOLDS)+['zara2']:
        if fold not in fold_groups:
            effect_table.append(f'| {fold} | 尚无真实测试结果 | N/A | N/A | N/A | N/A | N/A |')
            continue
        groups = fold_groups[fold]
        for group in GROUPS:
            value = groups[group]
            metrics = ' | '.join((_format(value['A'][m])+' / '+_format(value['B'][m])) if value['available'] else 'N/A' for m in METRICS)
            main_table.append(f"| {fold} | {LABEL[group]} | {value['scene_count']} | {value['agent_count']} | {metrics} |")
        all_effect = [100*groups['all']['B_minus_A'][m]['mean'] for m in METRICS]
        dense = groups['N_ge_9']
        dense_values = [100*dense['B_minus_A'][m]['mean'] if dense['available'] else None for m in ('JFDE', 'minFDE')]
        values = ' | '.join('N/A' if x is None else f'{x:+.9f}' for x in all_effect+dense_values)
        effect_table.append(f'| {fold} | {values} |')
        effect_text.append(f"{fold}：全体 JADE {_direction(all_effect[0])}，JFDE {_direction(all_effect[1])}，边缘 minFDE {_direction(all_effect[3])}；N≥9 有 {dense['scene_count']} 窗，JFDE {_direction(dense_values[0])}，边缘 minFDE {_direction(dense_values[1])}。")
    aggregate_table = []
    participants_text = []
    for scope, item in summary['scopes'].items():
        for group in GROUPS:
            g = item['groups'][group]
            participants_text.append(f"{LABEL[scope]} / {LABEL[group]}：参与 {g['participating_fold_count']} 折（{', '.join(g['participating_folds']) or '无'}），已完成但无该组窗口的折为 {', '.join(g['completed_but_empty_folds']) or '无'}。")
            for branch in BRANCHES:
                values = ' | '.join(_format(g[branch][m]) if g['available'] else 'N/A' for m in METRICS)
                aggregate_table.append(f"| {LABEL[scope]}（{item['status']}） | {LABEL[group]} | {g['participating_fold_count']} | {branch} | {values} |")
    completed = sum(fold in per_fold for fold in FOLDS)
    status = 'COMPLETE' if completed == len(FOLDS) else 'PARTIAL'
    document = f'''# 连续联合终点模型：剩余四折与五折汇总

本归档状态为 **{status}**：新四折实际完成测试 **{completed}/4**；既有 ZARA2 从固定提交 `{BASE_COMMIT}` 直接复用逐 seed 和逐窗记录，没有重新推理。缺失折不会记作零，也不会冒充完整四折或五折结果。

每折 A/B 在同一测试集、同一 seed 下配对比较，差值为 B−A，负值代表 B 较低。新四折的 A 均为各自上游开发集选出的 coordinate **dev rank1**；ZARA2 延用此前固定的 A=epoch5（旧 dev rank2）、B=epoch35（依赖模型 dev rank1），这一基线身份差异保持原样并单列说明。

## 逐折原生测试指标

单位为米，每格为 **A 均值±样本标准差 / B 均值±样本标准差**。JADE/JFDE 在每折内按窗口等权，边缘 minADE/minFDE 在每折内按有效行人人次加权。每个 seed 独立 K20，不合并成 K100。

| 折 | 分组 | 窗数 | 人次 | JADE：A / B | JFDE：A / B | minADE：A / B | minFDE：A / B |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(main_table)}

## 逐折配对变化（厘米）

以下为五 seed 配对差的均值，单位 cm；负值为改善，正值为恶化。完整配对差标准差保留在各折 SUMMARY.json/CSV 及逐 seed CSV 中。

| 折 | 全体 ΔJADE | 全体 ΔJFDE | 全体 ΔminADE | 全体 ΔminFDE | N≥9 ΔJFDE | N≥9 ΔminFDE |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(effect_table)}

{chr(10).join(effect_text)}

## 先折等权、再汇总五个 seed

对每个 seed，先在参与折之间**等权平均**各折已计算的原生指标，然后对五个 seed 求均值和样本标准差（ddof=1）。折间不按窗口数、行人数或训练数据量加权。B−A 的标准差来自五个折均值配对差，不合并 A/B 标准差，不平均各折的标准差。无 N≥9 窗口的折从 dense 汇总中剔除，参与折数如下。

{chr(10).join(participants_text)}

单位为米，格式为均值±样本标准差。

| 范围与完成状态 | 分组 | 参与折数 | 输出 | JADE | JFDE | minADE | minFDE |
|---|---|---:|---|---:|---:|---:|---:|
{chr(10).join(aggregate_table)}

五个 seed 为 2035–2039，描述**固定权重的推理随机性**，不代表五次独立训练重复，也不提供训练 seed 间不确定性。本次不把开发集门槛应用于测试成绩，不依据测试集再训练、选模、改阈值或筛选 seed。

## 训练和冻结身份

新依赖网络逐折仅按开发集原生 JADE 选择完整 epoch，同分取早轮。上游资产与各折训练配置、完整逐轮指标、Top5、冻结收据按白名单归档；checkpoint、候选、预测与原始轨迹留在训练主机。

| 新折 | A dev rank1 epoch | A checkpoint SHA256 | B 完成训练轮数 | B 选中 epoch | B checkpoint SHA256 |
|---|---:|---|---:|---:|---|
{chr(10).join(training_table)}

每折测试前，全部五 seed 的 A/B `[20,N,12,2]` FP32 预测已封存；评分入口完整核验文件哈希和形状后才读取 future，每窗一次读取后独立评分五个 seed。各折 SCORE_RECEIPT.json 记录封存、全量核验和首次真值读取的严格先后关系。测试阶段不更新参数。

## 归档入口

- [折等权汇总](SUMMARY.json) · [汇总表](SUMMARY.csv) · [折均值逐 seed 指标](PER_SEED.csv)
- [所有已完成折匿名逐窗记录](PER_WINDOW.csv) · [ZARA2 固定归档复用](ZARA2_REUSE.json)
- [发布核验与完成范围](REPORT_VERIFICATION.json) · [源文件与公开文件哈希](SOURCE_ARTIFACTS.json)

每个折子目录的 test/ 保留原始逐折统计；dependency/ 保留依赖网络训练配置、收据及逐轮指标；upstream/ 保留上游适配配置及 density/head/joint 各阶段的逐轮指标、Top5 和冻结身份；UPSTREAM_FROZEN.json 与 B_FROZEN.json 保留固定模型身份。公开成员名单以哈希和数量代替，主机绝对路径已匿名化；匿名逐窗 CSV 不含场景原名、行人 ID 或原始坐标。
'''
    (destination/'README.md').write_text(document)
    for entry in provenance:
        entry['artifact'] = str(Path(entry['artifact']).relative_to(destination))
    write(destination/'SOURCE_ARTIFACTS.json', provenance)
    verification = dict(status=status, completed_new_folds=completed, requested_new_folds=list(FOLDS),
        included_new_folds=[fold for fold in FOLDS if fold in per_fold],
        missing_new_folds=[fold for fold in FOLDS if fold not in per_fold],
        archived_zara2_reused=True, zara2_reference_commit=BASE_COMMIT,
        native_aggregation_reconstructed_from_anonymous_windows=True,
        equal_fold_then_five_seed_aggregation=True, paired_seed_difference_std=True, ddof=1,
        dense_empty_folds_excluded=True, new_prediction_calls=0, truth_reads=0, optimizer_updates=0)
    write(destination/'REPORT_VERIFICATION.json', verification)
    for path in destination.rglob('*'):
        if path.is_file():
            require(path.suffix in ('.json', '.csv', '.md', '.txt'), 'nontext artifact in report')
            require(not any(token in path.read_bytes() for token in (b'/media/ee615', b'/home/ee615')), 'host path leaked into report')
    write(destination/'ARTIFACT_SHA256.json', {str(path.relative_to(destination)): digest(path)
          for path in sorted(destination.rglob('*')) if path.is_file() and path.name != 'ARTIFACT_SHA256.json'})
    return verification

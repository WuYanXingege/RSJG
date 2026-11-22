"""Archive a completed continuous joint-Flow trial from its native artifacts only.

This reporter never constructs a model or optimizer, runs inference, changes a
checkpoint, or revises the fixed selector/thresholds. It validates native outputs
and complete-state retention, then writes public text-only, anonymous artifacts.
"""
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import torch

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from bcep.contracts import digest
from bcep_joint.common import read, write
from joint_flow_dependency.inputs import RUN, DOC, ARCHIVE, ARCHIVE_DIR, LIMITS

GROUPS = ('all', 'N_ge_9')
GROUP_LABEL = {'all': '全体', 'N_ge_9': 'N≥9'}
BRANCH_LABEL = {'A': 'A：旧 H0', 'B': 'B：学习依赖', 'C': 'C：独立采样', 'D': 'D：打乱对应关系（5 次窗内平均）'}


def rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def csvwrite(path, records):
    if not records:
        raise ValueError('refuse empty public CSV: '+str(path))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def git_blob(commit, name):
    return subprocess.check_output(['git', 'show', commit+':'+ARCHIVE_DIR+name], cwd=SOURCE)


def truth(value):
    return str(value).lower() in ('1', 'true')


def near(actual, expected, *, atol=2e-14):
    if not math.isfinite(float(actual)) or not math.isfinite(float(expected)):
        raise ValueError('nonfinite reported value')
    if not math.isclose(float(actual), float(expected), rel_tol=0., abs_tol=atol):
        raise ValueError(f'native/CSV mismatch: {actual} != {expected}')


def direction(value, unit='m'):
    if value < 0:
        return f'下降 {abs(value):.12g} {unit}'
    if value > 0:
        return f'上升 {value:.12g} {unit}'
    return f'持平（0 {unit}）'


def verify_checkpoint_retention(history, top, result):
    if len(history) < 30:
        raise ValueError('minimum 30 complete epochs was not met')
    if [h['epoch'] for h in history] != list(range(1, len(history)+1)):
        raise ValueError('complete epoch history has gaps or duplicate epochs')
    ranking = sorted(history, key=lambda h: (h['validation'], h['epoch']))
    if [r['epoch'] for r in top] != [r['epoch'] for r in ranking[:5]]:
        raise ValueError('Top5 does not use only all-dev JADE with earlier exact ties')
    if len(top) != 5 or len({r['epoch'] for r in top}) != 5:
        raise ValueError('five distinct complete epoch snapshots are required')
    rank = ranking[0]
    if result['rank1_epoch'] != rank['epoch']:
        raise ValueError('final result and official rank1 disagree')
    if result['rank1_metrics'] != rank['metrics']:
        raise ValueError('rank1 training receipt does not match its complete epoch')
    snapshots = []
    for h in history:
        path = RUN/'dependence'/h['file']
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        required = {'model', 'optimizer', 'scheduler', 'rng', 'completed_epoch', 'update',
                    'epoch', 'order', 'cursor', 'history', 'stale', 'recipe', 'identity'}
        if not required <= checkpoint.keys():
            raise ValueError('incomplete resumable checkpoint: '+h['file'])
        if checkpoint['completed_epoch'] != h['epoch'] or checkpoint['update'] != h['update']:
            raise ValueError('checkpoint epoch/update mismatch')
        if checkpoint['cursor'] != 0 or checkpoint['order'] is not None:
            raise ValueError('a partial epoch was represented as complete')
        if checkpoint['epoch'] != h['epoch']+1 or checkpoint['history'] != history[:h['epoch']]:
            raise ValueError('checkpoint recovery selector/history does not match its epoch')
        if not {'cpu', 'cuda', 'python', 'numpy', 'order', 'view'} <= checkpoint['rng'].keys():
            raise ValueError('missing recovery RNG/data iteration state')
        near(h['validation'], h['metrics']['all']['JADE'], atol=0.)
        snapshots.append(dict(epoch=h['epoch'], update=h['update'], all_dev_JADE=h['validation'],
                              file='dependence/'+h['file'], sha256=digest(path),
                              learning_rate=checkpoint['optimizer']['param_groups'][0]['lr']))
    if digest(RUN/'dependence/last.pt') != snapshots[-1]['sha256']:
        raise ValueError('last does not retain the final complete recovery state')
    rank_sha = snapshots[rank['epoch']-1]['sha256']
    if digest(RUN/'dependence/best.pt') != rank_sha or result['rank1_sha256'] != rank_sha:
        raise ValueError('best/rank1 checkpoint identity mismatch')
    for row in top:
        if digest(RUN/'dependence'/row['file']) != row['sha256']:
            raise ValueError('Top5 checkpoint hash mismatch')
    if result['completed_epochs'] != len(history) or result['updates'] != history[-1]['update']:
        raise ValueError('training receipt disagrees with completed history')
    return snapshots, rank


def verify_evaluation(metrics, nll, decision, replay, windows, shuffled):
    if len(windows) != 590 or len({r['window'] for r in windows}) != 590:
        raise ValueError('expected 590 distinct anonymous dev windows')
    if sum(int(r['agents']) >= 9 for r in windows) != 127:
        raise ValueError('wrong dense population')
    if len(shuffled) != 5*590:
        raise ValueError('five predeclared D repetitions per window are required')
    by_window = {}
    for r in shuffled:
        by_window.setdefault(r['window'], []).append(r)
        if not truth(r['endpoint_sets_exact']) or not truth(r['endpoint_minFDE_exact']):
            raise ValueError('D did not preserve an agent endpoint set/minFDE exactly')
    for r in windows:
        repetitions = by_window[r['window']]
        if sorted(int(x['repetition']) for x in repetitions) != list(range(5)):
            raise ValueError('missing/duplicate D repetition')
        for metric in ('JADE', 'JFDE', 'minFDE'):
            near(sum(float(x[metric]) for x in repetitions)/5, float(r['D_'+metric]))
        if not truth(r['D_endpoint_sets_exact']) or not truth(r['D_endpoint_minFDE_exact']):
            raise ValueError('D marginal preservation receipt failed')
        if float(r['C_delta_nll_nat_per_person']) != 0.:
            raise ValueError('independent continuous-model delta NLL must be zero')
        for comparator in ('C', 'D'):
            for metric in ('JADE', 'JFDE'):
                near(float(r[f'B_minus_{comparator}_{metric}']), float(r['B_'+metric])-float(r[comparator+'_'+metric]))
    for group in GROUPS:
        rr = windows if group == 'all' else [r for r in windows if int(r['agents']) >= 9]
        agents = sum(int(r['agents']) for r in rr)
        for branch in BRANCH_LABEL:
            native = metrics[branch][group]
            if native['scene_count'] != len(rr) or native['agent_count'] != agents:
                raise ValueError('native accumulator population mismatch')
            for metric in ('JADE', 'JFDE'):
                near(sum(float(r[branch+'_'+metric]) for r in rr)/len(rr), native[metric])
            near(sum(float(r[branch+'_agent_minFDE_sum']) for r in rr)/agents, native['minFDE'])
        for branch in ('B', 'C'):
            near(math.fsum(float(r[branch+'_delta_nll_nat_per_person']) for r in rr)/len(rr),
                 nll['delta_nll_nat_per_person'][branch][group])
            for stage in ('raw256', 'selected20', 'corrected20'):
                coverage = nll['coverage'][branch][group][stage]
                near(math.fsum(float(r[f'{branch}_{stage}_minFDE_agent_sum']) for r in rr)/agents,
                     coverage['agent_micro_minFDE'])
                near(math.fsum(float(r[f'{branch}_{stage}_minFDE_scene_mean']) for r in rr)/len(rr),
                     coverage['scene_macro_minFDE'])
                if coverage['agent_count'] != agents or coverage['scene_count'] != len(rr):
                    raise ValueError('endpoint coverage population mismatch')
    actual = [metrics['B']['all']['JADE'], metrics['B']['all']['JFDE'], metrics['B']['N_ge_9']['JFDE']]
    passes = [actual[0] <= LIMITS[0], actual[1] < LIMITS[1], actual[2] < LIMITS[2]]
    if decision['thresholds'] != LIMITS or decision['values'] != actual or decision['passes'] != passes:
        raise ValueError('fixed gate or final B metric disagreement')
    if decision['differences'] != [a-b for a, b in zip(actual, LIMITS)] or decision['all_pass'] != all(passes):
        raise ValueError('fixed gate result mismatch')
    if replay['D_repetitions'] != 5 or not replay['D_window_average_before_accumulation']:
        raise ValueError('D was not averaged inside each window first')
    if (replay['B_ddim_calls'] != 590 or replay['C_ddim_calls'] != 590 or replay['D_correction_calls'] != 0
            or replay['B_correction_calls'] != 590 or replay['C_correction_calls'] != 590
            or replay['D_ddim_calls'] != 5*sum(int(r['agents']) > 1 for r in windows)
            or replay['ddim_steps'] != 20):
        raise ValueError('wrong final inference/correction protocol')
    if replay['rank1_epoch_prediction_metrics_exact_replay'] is not True:
        raise ValueError('final B did not exactly replay the selected epoch validation')
    preservation = nll['marginal_preservation']
    if preservation['endpoint_sets_exact_checks'] != 2950 or preservation['endpoint_minFDE_exact_checks'] != 2950:
        raise ValueError('wrong count of endpoint preservation checks')
    maximum_drift = max(float(r['D_path_agent_minFDE_max_abs_drift_m']) for r in windows)
    near(maximum_drift, preservation['path_agent_minFDE_max_abs_drift_m'], atol=0.)
    near(maximum_drift, replay['D_path_agent_minFDE_max_abs_drift_m'], atol=0.)


def main():
    if read(RUN/'STATUS.json')['status'] != 'COMPLETE':
        raise RuntimeError('only a completed real run may be published as final')
    config = read(RUN/'RUN_CONFIG.json')
    result = read(RUN/'TRAINING_RESULT.json')
    history = read(RUN/'dependence/EPOCH_METRICS.json')
    top = read(RUN/'dependence/TOP5.json')
    metrics = read(RUN/'FINAL_METRICS.json')
    nll = read(RUN/'FINAL_NLL_AND_COVERAGE.json')
    decision = read(RUN/'DECISION.json')
    replay = read(RUN/'FINAL_REPLAY.json')
    windows = rows(RUN/'FINAL_WINDOWS.csv')
    shuffled = rows(RUN/'D_SHUFFLE_WINDOWS.csv')
    paired = rows(RUN/'PAIRED_SUMMARY.csv')
    frozen = read(RUN/'FROZEN_MODELS.json')
    if frozen.get('exact_unchanged') is not True or frozen['before'] != frozen['after']:
        raise ValueError('frozen model states changed')
    snapshots, rank = verify_checkpoint_retention(history, top, result)
    verify_evaluation(metrics, nll, decision, replay, windows, shuffled)
    for group in GROUPS:
        if metrics['B'][group] != result['rank1_metrics'][group]:
            raise ValueError('final B and selected epoch native statistics disagree')
    if decision['rank1_epoch'] != rank['epoch'] or replay['rank1_epoch'] != rank['epoch']:
        raise ValueError('final decision/replay used another checkpoint')
    preflight = read(RUN/'PREFLIGHT.json')
    if preflight['passed'] is not True or preflight['optimizer_updates'] != 0:
        raise ValueError('interface check failed or updated the official initialization')
    reference = json.loads(git_blob(ARCHIVE, 'REFERENCE_IDENTITY.json'))
    prior_config = json.loads(git_blob(ARCHIVE, 'RUN_CONFIG.json'))
    prior_metrics = json.loads(git_blob(ARCHIVE, 'FINAL_METRICS.json'))
    prior_decision = json.loads(git_blob(ARCHIVE, 'DECISION.json'))
    if prior_decision['thresholds'] != LIMITS:
        raise ValueError('reference gate changed')
    if config['identity']['checkpoint_sha256'] != reference['identity']['checkpoint_sha256']:
        raise ValueError('wrong frozen coordinate baseline')
    if config['identity']['split_sha256'] != prior_config['identity']['split_sha256']:
        raise ValueError('wrong split identity')
    for group in GROUPS:
        for key in ('JADE', 'JFDE', 'minFDE'):
            near(metrics['A'][group][key], prior_metrics['H0'][group][key], atol=0.)

    DOC.mkdir(parents=True, exist_ok=True)
    names = ['RUN_CONFIG.json', 'TRAINING_RESULT.json', 'PREFLIGHT.json', 'FROZEN_MODELS.json',
             'FINAL_METRICS.json', 'FINAL_NLL_AND_COVERAGE.json', 'FINAL_REPLAY.json', 'DECISION.json',
             'FINAL_WINDOWS.csv', 'D_SHUFFLE_WINDOWS.csv', 'PAIRED_SUMMARY.csv', 'A_REFERENCE.json']
    for name in names:
        shutil.copyfile(RUN/name, DOC/name)
    for name in ('GPU_RELEASE.json', 'CPU_TESTS.txt', 'RECOVERY_EVENTS.json'):
        if (RUN/name).exists():shutil.copyfile(RUN/name, DOC/name)
    for name in ('EPOCH_METRICS.json', 'TOP5.json'):
        shutil.copyfile(RUN/'dependence'/name, DOC/name)
    write(DOC/'REFERENCE_IDENTITY.json', reference)
    csvwrite(DOC/'ALL_CHECKPOINTS.csv', snapshots)
    epochs = []
    for h in history:
        row = dict(epoch=h['epoch'], update=h['update'], all_dev_JADE=h['validation'],
                   learning_rate=snapshots[h['epoch']-1]['learning_rate'],
                   dev_delta_nll_nat_per_person=h['metrics']['dev_delta_nll_nat_per_person'],
                   train_seconds=h.get('train_seconds'), validation_seconds=h.get('validation_seconds'))
        for key, value in h.get('train_mean_per_update', {}).items():
            row['train_update_mean_'+key] = value
        for group in GROUPS:
            for key in ('JADE', 'JFDE', 'minFDE'):
                row[group+'_'+key] = h['metrics'][group][key]
            row[group+'_delta_nll_nat_per_person'] = h['metrics']['delta_nll_nat_per_person'][group]
            for stage in ('raw256', 'selected20', 'corrected20'):
                row[group+'_'+stage+'_agent_micro_minFDE'] = h['metrics']['coverage'][group][stage]['agent_micro_minFDE']
        epochs.append(row)
    csvwrite(DOC/'EPOCHS.csv', epochs)

    table = []
    for group in GROUPS:
        for branch, label in BRANCH_LABEL.items():
            m = metrics[branch][group]
            table.append(f"| {GROUP_LABEL[group]} | {m['scene_count']} | {label} | {m['JADE']:.12f} | {m['JFDE']:.12f} | {m['minFDE']:.12f} |")
    likelihood_table = []
    coverage_table = []
    mechanism = []
    paired_table = []
    paired_by_group = {r['group']: r for r in paired}
    for group in GROUPS:
        if group not in paired_by_group:raise ValueError('missing paired summary group')
        p = paired_by_group[group]
        bnll = nll['delta_nll_nat_per_person']['B'][group]
        cnll = nll['delta_nll_nat_per_person']['C'][group]
        likelihood_table.append(f"| {GROUP_LABEL[group]} | {bnll:.12g} | {cnll:.12g} | {bnll-cnll:+.12g} |")
        bc = float(p['B_minus_C_JFDE']); bd = float(p['B_minus_D_JFDE'])
        near(float(p['B_minus_C_delta_nll_nat_per_person']), bnll-cnll)
        near(bc, metrics['B'][group]['JFDE']-metrics['C'][group]['JFDE'])
        near(bd, metrics['B'][group]['JFDE']-metrics['D'][group]['JFDE'])
        mechanism.append(f"**{GROUP_LABEL[group]}：**B 相对 C 的连续 ΔNLL {direction(bnll-cnll, 'nat/人')}，JFDE {direction(bc)}；在每人终点集合相同的 B/D 对照中，保留 B 的跨人对应关系使 JFDE 相对 D {direction(bd)}。")
        paired_table.append(f"| {GROUP_LABEL[group]} | {float(p['B_minus_A_JADE']):+.12g} | {float(p['B_minus_A_JFDE']):+.12g} | {float(p['B_minus_C_JADE']):+.12g} | {bc:+.12g} | {float(p['B_minus_D_JADE']):+.12g} | {bd:+.12g} |")
        for branch in ('B', 'C'):
            c = nll['coverage'][branch][group]
            coverage_table.append(f"| {GROUP_LABEL[group]} | {branch} | {c['raw256']['agent_micro_minFDE']:.12f} | {c['selected20']['agent_micro_minFDE']:.12f} | {c['corrected20']['agent_micro_minFDE']:.12f} |")
    gate_table = []
    for label, sign, value, limit, difference, passed in zip(
        ('全体 JADE', '全体 JFDE', 'N≥9 JFDE'), ('≤', '<', '<'),
        decision['values'], decision['thresholds'], decision['differences'], decision['passes']):
        gate_table.append(f"| {label} | {value:.17g} | {sign} {limit:.17g} | {difference:+.17g} | {'通过' if passed else '未通过'} |")
    verdict = ('同一 rank1 的 B 同时满足三项固定门槛，ZARA2 单折通过；不据此扩展到其他折。'
               if decision['all_pass'] else '同一 rank1 的 B 未同时满足三项固定门槛；保留旧 H0。')
    maximum_drift = max(float(r['D_path_agent_minFDE_max_abs_drift_m']) for r in windows)
    endpoint_checks = sum(truth(r['endpoint_sets_exact']) for r in shuffled)
    minfde_checks = sum(truth(r['endpoint_minFDE_exact']) for r in shuffled)
    baseline_drift = [metrics['A']['all']['JADE']-LIMITS[0], metrics['A']['all']['JFDE']-LIMITS[1],
                      metrics['A']['N_ge_9']['JFDE']-LIMITS[2]]
    runtime_commit = config['runtime_source_commit']
    code_base = f'https://github.com/WuYanXingege/RSJG/blob/{runtime_commit}/'
    document = f'''# ZARA2 连续联合终点生成实验结果

{verdict}

本轮只训练一个跨人依赖网络，seed 固定为 3201。实际完成 **{result['completed_epochs']}** 轮、**{result['updates']}** 次更新，停止原因为 `{result['stop_reason']}`。每轮均将新模型的真实 K20 预测经过一次 20 步 DDIM，在全部 590 个 dev 窗口上计算原生指标；仅按全体 dev JADE 最低选出的 rank1 为第 **{result['rank1_epoch']}** 轮，严格同分取较早完整轮次。最终 B/C/D 均使用该 rank1；没有第二个模型、第二个训练 seed、测试集访问或扩折。

## 四项输出的主结果

单位均为米。JADE/JFDE 先在同一联合场景内平均有效行人，再对 20 个联合场景取最小值，最后对窗口等权平均。边缘 minFDE 使用原生累加器按有效行人加权；并非各窗 minFDE 的简单平均。D 的五次置乱先在每个窗口内平均原生统计量，再进入全体/N≥9 累加器。

| 分组 | 窗数 | 输出 | JADE | JFDE | 边缘 minFDE |
|---|---:|---|---:|---:|---:|
{chr(10).join(table)}

A 复用固定归档 `{ARCHIVE}` 的“始终 H0”及同一身份的逐窗原生统计，已做少量原生回放核对；没有使用旧 I/H0 选择头的成绩。B 是实际学习依赖入口。C 只旁路依赖网络，复用 B 的基础随机数、256→20 压缩、修正头和桥。D 在 B 已修正的 20 组终点上逐人独立置乱，终点与该人的桥噪声同步移动，随后重新解码；单人窗保留 B。

## B 的固定预测门槛

下表及最终判定均使用未舍入原生累加值，差值为 B 减门槛。同一 rank1 必须同时满足三项。

| 指标 | B 实际值 (m) | 固定门槛 (m) | 差值 (m) | 判定 |
|---|---:|---:|---:|---|
{chr(10).join(gate_table)}

A 的归档原生回放与原始门槛之间的三项差为 `{baseline_drift}` m。该回放差没有用于移动门槛，也没有算作新模型的收益。

## 连续似然与依赖机制

连续模型的 ΔNLL 为相对独立标准高斯的负对数似然差，单位 nat/人，各窗口等权。其计算对象是**压缩前的连续终点分布**；冻结 Flow 的 Jacobian 在差值中抵消。C 的理论 ΔNLL 恒为 0，不能为 A 或 D 虚构同一种连续模型似然。

| 分组 | B：dev ΔNLL (nat/人) | C：dev ΔNLL | B−C |
|---|---:|---:|---:|
{chr(10).join(likelihood_table)}

{chr(10).join(mechanism)}

| 分组 | B−A JADE | B−A JFDE | B−C JADE | B−C JFDE | B−D JADE | B−D JFDE |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(paired_table)}

差值为逐窗配对差的均值，负值表示 B 较低。B/C 回答学习跨人依赖后的实际变化，依赖可能进一步改变后续代表样本和修正结果。B/D 在每人终点集合固定时检查联合对应关系：{endpoint_checks} 个“窗口×置乱”检查的终点集合完全相同，{minfde_checks} 个检查的每人端点 minFDE 完全相同。桥解码后的每人 minFDE 最大绝对数值差为 **{maximum_drift:.17g} m**，单列于 [五次置乱逐窗记录](D_SHUFFLE_WINDOWS.csv)，没有混同于端点集合变化。

A/B 是完整方案差异，包含连续采样、压缩策略和后续路径的影响，不能单独归因于依赖网络。预测门槛判定与机制方向分别报告；单折结果不作为跨折结论。

## 有限样本覆盖

以下为端点集合自身的边缘 minFDE，按有效行人加权。完整文件同时保留逐窗等权版本。

| 分组 | 连续模型 | 原始 256 组 | 压缩后 20 组 | 修正后 20 组 |
|---|---|---:|---:|---:|
{chr(10).join(coverage_table)}

严格保持的是**压缩前每个人的连续 Flow 边缘分布**：共享边因子与独立个人噪声均为标准高斯，经各人的 Cholesky 白化后，联合高斯的每个对角块为 I₂。有限的 256 次抽样仍有经验覆盖波动；整组压缩和有界修正不承诺保留该连续边缘分布，其变化由上述实测量描述。

## 固定实现与训练

冻结基线为旧 v2 coordinate epoch5，原生提交 `{config['identity']['base']}`，checkpoint SHA256 `{config['identity']['checkpoint_sha256']}`。Provider density epoch19、修正头 epoch9、原编码器和六层桥全程 eval、无梯度，训练前后张量哈希完全一致。train 2681 窗、dev 590 窗，dev 中 127 窗 N≥9；未加载测试数据。SSEC 保持其原监督器记录的暂停状态，旧 I/H0 选择实验不重启。

依赖网络是共享的 260→64→4 两层 MLP，共 **16,964** 个可训练参数，SiLU 激活，从边两端各自的视角读取两个冻结 Flow 观测上下文、相对位置和相对速度，输出 2×2 实矩阵。图取旧观测近邻图的无向并集。末层权重以 std=0.001 初始化，偏置为 `[0.02,0,0,0.02]`，使两端载荷小幅非零。训练和采样共用同一因子 F 及 `R=F Fᵀ`；逐窗目标为 `[log det R + z*ᵀR⁻¹z* − z*ᵀz*]/(2N)`，真实同步终点经原 motion_frame 与冻结四层 Flow 的 `inverse=True` 得到 z*。未来只用于训练目标和离线计分，ID 只用于对齐与随机键。

每窗一次产生 256 个完整联合场景，以所有有效行人端点距离的均值定义场景距离，贪心选 20 个实际样本；首个最小化总距离，其后最大化总最近距离下降，同分取较小原抽样编号。端点不经过逐人 KMeans、Hungarian 或后置选择。20 组联合终点统一调用冻结修正头一次，再以 eta=0 进入一次 20 步 DDIM。个人噪声、无向边噪声、桥噪声与置乱使用独立随机流；桥噪声绑定 scene、agent、原始联合抽样编号和 seed2035，置乱时保持端点/噪声绑定。

模型、似然、Flow 和桥使用 FP32；离散代表选择的距离累加使用 CPU float64，沿用原有限支持选择器的数值方式。AdamW lr=1e-4、weight decay=1e-4、梯度裁剪 5、逻辑批 16 窗、5 轮 warmup、100 轮余弦调度、minimum30/patience15/max100。优化器只含依赖网络，窗口等权，不按行人数重新加权损失。

从独立工作区执行的正式命令如下；同一命令在可恢复中断后读取本次 last，不建立第二次训练。

```bash
env CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ../.conda/rsjg/bin/python -u -m joint_flow_dependency.run --wait-gpu
```

## 可恢复状态与交付

全部 **{len(snapshots)} 个完整 epoch** 的 model、optimizer、scheduler、完成轮次/更新数、RNG、早停与数据迭代状态已核验。保留全部 epoch、best、last 和按全体 dev JADE 排序的五个不同完整 epoch，严格同分取早轮；不删除任何旧研究资产。rank1 checkpoint SHA256 为 `{result['rank1_sha256']}`。

运行源码固定提交：`{runtime_commit}`。关键代码：[协方差与似然]({code_base}joint_flow_dependency/model.py) · [联合压缩与随机身份]({code_base}joint_flow_dependency/sampling.py) · [实际推理]({code_base}joint_flow_dependency/inference.py) · [训练]({code_base}joint_flow_dependency/train.py) · [四项对照]({code_base}joint_flow_dependency/evaluation.py)。

- [固定配置](RUN_CONFIG.json) · [逐轮指标 CSV](EPOCHS.csv) · [完整逐轮指标](EPOCH_METRICS.json)
- [匿名逐窗四项对照](FINAL_WINDOWS.csv) · [五次置乱逐窗记录](D_SHUFFLE_WINDOWS.csv) · [配对汇总](PAIRED_SUMMARY.csv)
- [连续似然与有限覆盖](FINAL_NLL_AND_COVERAGE.json) · [原生主指标](FINAL_METRICS.json) · [固定门槛判定](DECISION.json)
- [Top5](TOP5.json) · [全部 checkpoint 索引](ALL_CHECKPOINTS.csv) · [训练收据](TRAINING_RESULT.json)
- [接口检查](PREFLIGHT.json) · [冻结状态核验](FROZEN_MODELS.json) · [A 的归档身份](A_REFERENCE.json) · [最终入口核验](FINAL_REPLAY.json) · [发布文件核验](REPORT_VERIFICATION.json)

权重、候选终点、原始轨迹与冻结上下文/z* 缓存均保留在训练主机。公开 CSV 不含场景原名、行人 ID 或原始坐标。
'''
    (DOC/'README.md').write_text(document)
    write(DOC/'REPORT_VERIFICATION.json', dict(
        real_run_complete=True, dev_windows=590, dense_windows=127, D_repetitions_per_window=5,
        native_aggregation_reconstructed_from_anonymous_csv=True, D_window_average_verified=True,
        fixed_gate_recomputed_from_unrounded_native_B=True, frozen_states_unchanged=True,
        retained_complete_epochs=len(snapshots), distinct_top5_epochs=[r['epoch'] for r in top],
        rank1_epoch=result['rank1_epoch'], rank1_sha256=result['rank1_sha256'],
        checkpoint_recovery_fields_verified=True, A_matches_fixed_H0_archive=True,
        reporting_inference_calls=0, reporting_optimizer_updates=0))
    for path in DOC.rglob('*.csv'):
        path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
    for path in DOC.rglob('*'):
        if path.is_file():
            if path.suffix not in ('.json', '.csv', '.md', '.txt'):
                raise ValueError('non-text artifact in public report')
            content = path.read_bytes()
            if b'/media/ee615' in content or b'/home/ee615' in content:
                raise ValueError('host absolute path in public artifact: '+str(path))
    write(DOC/'ARTIFACT_SHA256.json', {str(p.relative_to(DOC)): digest(p) for p in sorted(DOC.rglob('*'))
                                      if p.is_file() and p.name != 'ARTIFACT_SHA256.json'})
    print(json.dumps(dict(decision=decision, report=str(DOC.relative_to(SOURCE)/'README.md'),
                         checkpoint_epochs=len(snapshots), rank1_epoch=result['rank1_epoch']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

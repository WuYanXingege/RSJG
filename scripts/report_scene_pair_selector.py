"""Read completed native artifacts, anonymize window IDs, and archive results."""
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import torch

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from bcep.contracts import digest
from bcep_joint.common import read, write
from scene_pair_selector.inputs import RUN, LIMITS
from scene_pair_selector.finalize import csvwrite

DOC = SOURCE/'docs/research/scene_pair_selector/zara2_20261122'
REFERENCE = 'f27fd97fc685cebcb4774c524067a18ad4c5da36'
REFERENCE_DIR = 'docs/research/bcep_scene_energy/zara2_trial_20261121/status_20261122_122319/'
OLD = '76d80b508ad79f52965c15109b58d2d835473fd7'
OLD_DIR = 'docs/research/bcep_v2/sharedworld_stage2_onerun_20261119/'


def blob(commit, path):
    return subprocess.check_output(['git', 'show', commit+':'+path], cwd=SOURCE)


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    assert read(RUN/'STATUS.json')['status'] == 'COMPLETE'
    config = read(RUN/'RUN_CONFIG.json')
    result = read(RUN/'TRAINING_RESULT.json')
    metrics = read(RUN/'FINAL_METRICS.json')
    decision = read(RUN/'DECISION.json')
    history = read(RUN/'selector/EPOCH_METRICS.json')
    top = read(RUN/'selector/TOP5.json')
    rank = min(history, key=lambda r: (r['validation'], r['epoch']))
    assert result['rank1_epoch'] == rank['epoch'] == top[0]['epoch']
    assert result['rank1_sha256'] == top[0]['sha256'] == digest(RUN/'selector/best.pt')
    assert len(history) >= 30 and len(top) == 5
    assert len({x['epoch'] for x in top}) == 5
    snapshots = []
    for h in history:
        path = RUN/'selector'/h['file']
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert state['completed_epoch'] == h['epoch'] and state['cursor'] == 0 and state['order'] is None
        assert {'model', 'optimizer', 'scheduler', 'rng', 'stale', 'history', 'update'} <= state.keys()
        snapshots.append(dict(epoch=h['epoch'], update=h['update'], selected_dev_JADE=h['validation'],
                              file='selector/'+h['file'], sha256=digest(path),
                              learning_rate=state['optimizer']['param_groups'][0]['lr']))
    assert digest(RUN/'selector/last.pt') == snapshots[-1]['sha256']
    for r in top:
        assert digest(RUN/'selector'/r['file']) == r['sha256']
    DOC.mkdir(parents=True, exist_ok=True)
    for name in ['RUN_CONFIG.json', 'TRAINING_RESULT.json', 'NORMALIZATION.json', 'PREFLIGHT.json',
                 'FROZEN_MODELS.json', 'CACHE_BASELINES.json', 'FINAL_REPLAY.json', 'FINAL_METRICS.json',
                 'DECISION.json', 'DEV_WINDOWS.csv', 'PAIRED_SUMMARY.csv', 'GPU_RELEASE.json', 'PRETRAIN_WAIT_TRANSITION.json']:
        shutil.copyfile(RUN/name, DOC/name)
    for name in ['TOP5.json', 'EPOCH_METRICS.json']:
        shutil.copyfile(RUN/'selector'/name, DOC/name)
    csvwrite(DOC/'ALL_CHECKPOINTS.csv', snapshots)
    write(DOC/'REFERENCE_IDENTITY.json', json.loads(blob(REFERENCE, REFERENCE_DIR+'REFERENCE_IDENTITY.json')))
    old_gate = json.loads(blob(REFERENCE, REFERENCE_DIR+'THRESHOLD_COMPARISON.json'))['old_epoch5']
    assert old_gate == LIMITS
    epoch_rows = []
    for h in history:
        row = dict(epoch=h['epoch'], update=h['update'], selected_dev_JADE=h['validation'],
                   learning_rate=snapshots[h['epoch']-1]['learning_rate'],
                   switches=h['metrics']['switches'], dense_switches=h['metrics']['dense_switches'],
                   train_mse_update_mean=h['train_mean_per_update']['two_output_mse'],
                   train_seconds=h['train_seconds'], validation_seconds=h['validation_seconds'])
        for group in ['all', 'N_ge_9']:
            for metric in ['JADE', 'JFDE']:
                row[group+'_'+metric] = h['metrics'][group][metric]
        epoch_rows.append(row)
    csvwrite(DOC/'EPOCHS.csv', epoch_rows)

    labels = []
    for purpose in ['train', 'selection_dev']:
        manifest = read(RUN/'cache'/f'{purpose}_MANIFEST.json')
        for pos, item in enumerate(manifest):
            path = RUN/'cache'/item['file']
            assert digest(path) == item['sha256']
            rec = torch.load(path, map_location='cpu', weights_only=False)
            labels.append(dict(split=purpose, window=f'{purpose}_{pos+1:04d}',
                agents=int(rec['features']['valid'].sum()), dA=float(rec['target'][0, 0]), dF=float(rec['target'][0, 1]),
                identical_permutations=int(torch.equal(rec['initial_pi'], rec['feedback_pi']))))
    csvwrite(DOC/'ANONYMOUS_LABELS.csv', labels)

    # Pair exact same-run path metrics; separately expose archival playback drift.
    current = rows(RUN/'DEV_WINDOWS.csv')
    old_bytes = blob(OLD, OLD_DIR+'OLD5_REFERENCE_WINDOWS.csv')
    ref = read(DOC/'REFERENCE_IDENTITY.json')
    assert hashlib.sha256(old_bytes).hexdigest() == ref['csv_sha256']
    archived = list(csv.DictReader(io.StringIO(old_bytes.decode())))
    assert len(current) == len(archived) == 590
    aligned = []
    for pos, (new, previous) in enumerate(zip(current, archived)):
        manifest = read(RUN/'cache/selection_dev_MANIFEST.json')[pos]
        rec = torch.load(RUN/'cache'/manifest['file'], map_location='cpu', weights_only=False)
        assert rec['index'] == int(previous['index']) and int(new['agents']) == int(previous['n_agents'])
        aligned.append(dict(window=new['window'], agents=new['agents'],
            I_JFDE=float(new['I_JFDE']), archived_I_JFDE=float(previous['I_JFDE']),
            replay_minus_archive_I_JFDE=float(new['I_JFDE'])-float(previous['I_JFDE']),
            H0_JADE=float(new['H0_JADE']), archived_H0_JADE=float(previous['H_JADE']),
            replay_minus_archive_JADE=float(new['H0_JADE'])-float(previous['H_JADE']),
            H0_JFDE=float(new['H0_JFDE']), archived_H0_JFDE=float(previous['H_JFDE']),
            replay_minus_archive_JFDE=float(new['H0_JFDE'])-float(previous['H_JFDE'])))
    csvwrite(DOC/'H0_ARCHIVE_REPLAY_DIFFERENCES.csv', aligned)
    def counts(items, prefix):
        d = [r[prefix+'H0_JFDE']-r[prefix+'I_JFDE'] for r in items]
        return dict(improved=sum(v < 0 for v in d), worsened=sum(v > 0 for v in d), tied=sum(v == 0 for v in d))
    def sign(value):
        return (value > 0)-(value < 0)
    sign_changes = [r for r in aligned if sign(r['H0_JFDE']-r['I_JFDE']) != sign(r['archived_H0_JFDE']-r['archived_I_JFDE'])]
    numeric = dict(archived_H0_vs_I_counts=counts(aligned, 'archived_'),
                   native_H0_vs_I_counts=counts(aligned, ''), changed_windows=sign_changes,
                   max_abs_I_JFDE_drift_at_changed_windows=max((abs(r['replay_minus_archive_I_JFDE']) for r in sign_changes), default=0.))
    write(DOC/'METRIC_NUMERICS.json', numeric)
    paired = rows(RUN/'PAIRED_SUMMARY.csv')
    table = []
    for group in ['all', 'N_ge_9']:
        for branch, label in [('I', '始终 I'), ('H0', '始终 H0'), ('actual', '实际选择头')]:
            m = metrics[branch][group]
            table.append(f"| {group} | {m['scene_count']} | {label} | {m['JADE']:.12f} | {m['JFDE']:.12f} |")
    ptable = []
    for p in paired:
        ptable.append(f"| {p['group']} | {p['switched']} | {p['JADE_improved']} / {p['JADE_worsened']} / {p['JADE_tied']} | {p['JFDE_improved']} / {p['JFDE_worsened']} / {p['JFDE_tied']} |")
    gate = []
    for name, op, value, threshold, diff, passed in zip(
        ['全体 JADE', '全体 JFDE', 'N≥9 JFDE'], ['≤', '<', '<'],
        decision['values'], decision['thresholds'], decision['differences'], decision['passes']):
        gate.append(f"| {name} | {value:.17g} | {op} {threshold:.17g} | {diff:+.17g} | {'通过' if passed else '未通过'} |")
    conclusion = '同一 rank1 同时满足三项固定门槛，ZARA2 单折通过。' if decision['all_pass'] else '同一 rank1 未同时满足三项固定门槛；保留旧 coordinate，结束该路线。'
    baseline_diff = [metrics['H0']['all']['JADE']-LIMITS[0], metrics['H0']['all']['JFDE']-LIMITS[1],
                     metrics['H0']['N_ge_9']['JFDE']-LIMITS[2]]
    replay_max = {k: max(abs(float(r[f'replay_residual_{k}'])) for r in current) for k in ['JADE', 'JFDE']}
    devlabels = [r for r in labels if r['split'] == 'selection_dev']
    oracle = sum(min(float(r['I_JFDE']), float(r['H0_JFDE'])) for r in current)/590
    dense = [r for r in current if int(r['agents']) >= 9]
    dense_oracle = sum(min(float(r['I_JFDE']), float(r['H0_JFDE'])) for r in dense)/127
    document = f'''# ZARA2 场景级配对选择实验结果

{conclusion}

仅执行 seed 3201 的一次选择头训练。共完成 {result['completed_epochs']} 轮、{result['updates']} 次更新，停止原因为 `{result['stop_reason']}`。仅按全体 dev 选择后 JADE 选出的 rank1 为第 **{result['rank1_epoch']}** 轮，严格同分取较早完整轮次；最终成绩均来自该 checkpoint 的“先选择、后一次 20 步 DDIM”真实入口。没有阈值搜索、第二次调参、测试集访问或扩折。

## 同一轮次的结果

各数值单位为米。全体 590 窗，N≥9 为 127 窗。各指标使用原生 `hard_metric_statistics` 和 `MetricAccumulator`：先在每个联合场景内平均有效行人，再在 20 个联合场景中取最小值，各窗口等权。

| 分组 | 窗数 | 方案 | JADE (m) | JFDE (m) |
|---|---:|---|---:|---:|
{chr(10).join(table)}

## 三项固定门槛

门槛判断使用未舍入原生累加值；表内差值为实际选择头减门槛。

| 指标 | 实际值 (m) | 固定门槛 (m) | 差值 (m) | 判定 |
|---|---:|---:|---:|---|
{chr(10).join(gate)}

## 真实切换与浮点回放

一个动作切换整个窗口、全部有效行人及 20 个联合场景。以下改善/恶化/持平只统计实际切到 I 的窗口，相对同次独立强制 H0；未切换的窗口不计入切换收益。

| 分组 | 切换窗数 | JADE 改善 / 恶化 / 持平 | JFDE 改善 / 恶化 / 持平 |
|---|---:|---|---|
{chr(10).join(ptable)}

逐窗分解为 `实际−同次H0 = 切换指示×(同次I−同次H0) + 回放残差`。最大绝对回放残差为 JADE {replay_max['JADE']:.17g} m、JFDE {replay_max['JFDE']:.17g} m。两组各自平均切换贡献、残差均保存在 [PAIRED_SUMMARY.csv](PAIRED_SUMMARY.csv)。相对固定历史门槛，同次 H0 的三项回放差为 {baseline_diff} m；没有将该差解释为模型收益，也没有修改门槛。另附 [H0 逐窗历史回放差](H0_ARCHIVE_REPLAY_DIFFERENCES.csv)。

## 标签、模型与边界

冻结旧 v2 coordinate epoch5；原生源码 `{config['identity']['base']}`，权重 SHA256 `{config['identity']['checkpoint_sha256']}`。Provider density19/head9、共享编码器、配对头和六层桥均 eval、无梯度，训练前后张量哈希完全一致。train 2681 窗和 dev 590 窗使用原 split、seed2035 refined bank 和按 scene/agent/原 bank 槽位生成的 keyed noise。

I 为同一次原配对前向第一次 Hungarian 的排列，H0 为 S0 加旧邻居反馈后的第二次排列；原 bank 顺序 G 未当作 I。离线分别解码两路，端点与原槽位噪声同步 gather。标签为 `dA=JADE(I)−JADE(H0)`、`dF=JFDE(I)−JFDE(H0)`，单位米。共享图打分器有 {config['selector_parameters']:,} 个参数，以历史、冻结 social_h 和同一联合场景内的配对终点关系为输入；窗口 ID、未来、误差标签均不进入打分器。归一化仅拟合 train。双输出 MSE 按窗口等权。

规则固定为 `dA_hat≤0 且 dF_hat<0` 时选 I，其余 H0；单人或排列完全相同直接 H0。对照事后选择的全体/N≥9 JFDE 为 {oracle:.12f}/{dense_oracle:.12f} m，仅用于描述选择空间，不作为正式成绩或调参依据。原归档 H0 相对 I 的改善/恶化/持平为 234/249/107；本次原生轨迹 metric 为 {numeric['native_H0_vs_I_counts']['improved']}/{numeric['native_H0_vs_I_counts']['worsened']}/{numeric['native_H0_vs_I_counts']['tied']}。其中 {len(sign_changes)} 个窗口的近零差变为持平，I 的 JFDE 最大回放差为 {numeric['max_abs_I_JFDE_drift_at_changed_windows']:.17g} m，详见 [数值差异记录](METRIC_NUMERICS.json)。未以容差改写标签或选择规则。

FP32、原 AdamW 参数分组，lr=1e-4、weight_decay=1e-4、5 轮 warmup、100 轮 cosine horizon，minimum30/patience15/max100。优化器只含选择头。三个真实窗口的强制 H0 与旧入口逐值一致；最终复核 590 窗，每窗先决策后只解码选中分支一次，另做两条离线强制对照。

## 复现与归档

运行代码提交 `{config['runtime_source_commit']}`。在原训练工作区按 [IMPLEMENTATION.md](IMPLEMENTATION.md) 的命令执行；同一次技术中断从 `selector/last.pt` 恢复。SSEC 由原监督器按暂停规则退出，状态为 PAUSED_NO_AUTOMATIC_RESUME；本实验未恢复或修改它。资源释放凭据见 [GPU_RELEASE.json](GPU_RELEASE.json)。等待入口在零标签、零训练更新时调整一次，见 [安全过渡记录](PRETRAIN_WAIT_TRANSITION.json)。

全部 {len(history)} 个完整轮次的完整可恢复 checkpoint、best、last 留在训练主机。前五名只按全体 dev 选择后 JADE、严格同分取早轮，包含五个不同完整 epoch；模型、优化器、调度、计数、RNG、早停和迭代状态已复核。不删除研究资产。rank1 SHA256 为 `{result['rank1_sha256']}`。

- [运行配置](RUN_CONFIG.json) · [逐轮 CSV](EPOCHS.csv) · [完整逐轮指标](EPOCH_METRICS.json)
- [匿名 dev 逐窗结果](DEV_WINDOWS.csv) · [匿名 train/dev 标签](ANONYMOUS_LABELS.csv)
- [Top5 及哈希](TOP5.json) · [全部 checkpoint 索引](ALL_CHECKPOINTS.csv)
- [最终判定](DECISION.json) · [最终单路复核](FINAL_REPLAY.json) · [冻结张量核验](FROZEN_MODELS.json)
- [原始参考身份](REFERENCE_IDENTITY.json) · [真实窗口接入检查](PREFLIGHT.json) · [CPU 检查](CPU_TESTS.txt)

权重、bank、冻结特征缓存与原始轨迹仅保留在训练主机。发布内容不包含场景原名、行人 ID、原始坐标或主机绝对路径。
'''
    (DOC/'README.md').write_text(document)
    # Normalize public CSV line endings only; retain host-native artifacts verbatim.
    for path in DOC.glob('*.csv'):
        path.write_bytes(path.read_bytes().replace(b'\r\n', b'\n'))
    write(DOC/'ARTIFACT_SHA256.json', {str(p.relative_to(DOC)): digest(p) for p in sorted(DOC.iterdir())
                                      if p.is_file() and p.name != 'ARTIFACT_SHA256.json'})
    for p in DOC.iterdir():
        if p.is_file():
            assert p.suffix in ['.json', '.csv', '.md', '.txt']
            assert b'/media/ee615' not in p.read_bytes() and b'/home/ee615' not in p.read_bytes()
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

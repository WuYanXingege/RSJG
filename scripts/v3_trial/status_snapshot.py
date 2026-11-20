"""Read-only interim analysis of the existing v3 run; never launches training/inference."""
import csv
import hashlib
import io
import json
import statistics as st
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report as final
from speed_snapshot import read, lines, write, csv_write, evidence

SOURCE, RUN = final.SOURCE, final.RUN
DOC = SOURCE / 'docs/research/bcep_v3/zara2_trial_20261120'


def main():
    torch.set_num_threads(1)
    started = datetime.now(ZoneInfo('Asia/Shanghai'))
    # Read a single complete prefix; a live log may grow after this snapshot.
    history = read(RUN/'joint/EPOCH_METRICS.json', 'epoch_metrics')
    top = read(RUN/'joint/TOP5.json', 'top5')
    status = read(RUN/'STATUS.json', 'native_status')
    supervisor = read(RUN/'SUPERVISOR_STATUS.json', 'supervisor_status')
    progress = read(RUN/'joint/PROGRESS.json', 'native_progress')
    launch = read(RUN/'LAUNCH.json', 'launch')
    frozen = read(RUN/'FROZEN_CONFIG.json', 'frozen_config')
    assert status['status'] == 'RUNNING' and supervisor['status'] == 'TRAINING', 'This script is for an active interim snapshot only.'
    telemetry = lines(RUN/'MODULE_UPDATES.jsonl', 'module_updates')
    samples = lines(RUN/'RESOURCE_SAMPLES.jsonl', 'resources')
    events = lines(RUN/'EPOCH_EVENTS.jsonl', 'epoch_events')
    captured = datetime.now(ZoneInfo('Asia/Shanghai'))
    assert launch['attempt'] == supervisor['attempt'] == 1, 'Use recovery-aware accounting after a restart.'
    assert [r['update'] for r in telemetry] == list(range(1, len(telemetry)+1))
    assert all(r['domain'] == 'real' for r in telemetry)
    assert [r['epoch'] for r in history] == list(range(1, len(history)+1))
    assert all(r['update'] == 168*r['epoch'] for r in history)
    ranked = sorted(history, key=lambda r: (r['validation'], r['epoch']))
    assert [r['epoch'] for r in top] == [r['epoch'] for r in ranked[:5]], 'Epoch rollover: rerun snapshot.'
    rank, latest = ranked[0], history[-1]
    for name, expected in frozen['implementation_files'].items():
        assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() == expected, name
    assert final.canonical_hash(frozen) == '294054c2cf7605caafe016507bb601e2d85518b70555369ca3690b74138e9e43'
    costs = final.resource_rows(history, events, samples, telemetry)
    epoch_rows = []
    group_sets = {}
    for h in history:
        groups = read(RUN/'dev'/f"epoch_{h['epoch']:03d}_groups.json", f"epoch_{h['epoch']}_groups")
        group_sets[h['epoch']] = {g['group']: g for g in groups}
        for g in groups:
            epoch_rows.append(dict(epoch=h['epoch'],update=h['update'],group=g['group'],windows=g['scene_count'],
                H_JADE=g['JADE'],G_JFDE=g['G_JFDE'],I_JFDE=g['I_JFDE'],H_JFDE=g['JFDE'],
                endpoint_aggregation_H_JFDE=g['H_JFDE'],feedback_gain=g['feedback_gain'],
                marginal_minFDE=g['minFDE'],window_oracle=g['scene_weighted_minFDE'],
                train_seconds=h['train_seconds'],validation_seconds=h['validation_seconds']))
    groups = group_sets[rank['epoch']]
    latest_groups = group_sets[latest['epoch']]
    old_blob, old_meta = final.reference_inputs(frozen)
    old = list(csv.DictReader(io.StringIO(old_blob.decode())))
    new = final.csv_rows(RUN/'dev'/f"epoch_{rank['epoch']:03d}_windows.csv")
    paired = final.paired_rows(new, old)
    paired_summary = final.summaries(paired)
    assert sum(r['n_agents'] >= 9 for r in paired) == 127
    # Check invariance in every completed epoch, not only the selected one.
    for h in history:
        final.paired_rows(final.csv_rows(RUN/'dev'/f"epoch_{h['epoch']:03d}_windows.csv"), old)
    decision_lock = read(DOC/'PREREGISTERED_DECISION.json', 'preregistered_decision')
    limits = list(decision_lock['thresholds_m'].values())
    assert list(decision_lock['thresholds_m']) == ['all_H_JADE_lte','all_H_JFDE_lt','N_ge_9_H_JFDE_lt']
    values = [groups['all']['JADE'],groups['all']['JFDE'],groups['N_ge_9']['JFDE']]
    passes = [values[0] <= limits[0], values[1] < limits[1], values[2] < limits[2]]
    checks = []
    for row in history:
        p = RUN/'joint'/row['file']
        assert p.is_file()
        checks.append(dict(epoch=row['epoch'],update=row['update'],file=row['file'],exists=True,bytes=p.stat().st_size))
    for row in top:
        assert final.digest(RUN/'joint'/row['file']) == row['sha256']
    saved = torch.load(RUN/'joint/last.pt', map_location='cpu', weights_only=False)
    selected = torch.load(RUN/'joint'/rank['file'], map_location='cpu', weights_only=False)
    for state, epoch in [(saved, latest['epoch']), (selected, rank['epoch'])]:
        assert state['completed_epoch'] == epoch and state['update'] == 168*epoch
        assert state['cursor'] == 0 and state['order'] is None
        assert state['identity'] == frozen['identity'] and state['recipe'] == frozen['recipe']
        assert {'model','optimizer','scheduler','rng','history','stale','trainable'} <= set(state)
        assert {'cpu','cuda','python','numpy','order','view'} <= set(state['rng'])
    assert final.digest(RUN/'joint/best.pt') == top[0]['sha256']
    assert final.digest(RUN/'joint/last.pt') == final.digest(RUN/'joint'/latest['file'])
    modules = final.module_summary(telemetry)
    heads = {'coupling.csee.':'coupling.csee.out.weight','coupling.hcra.':'coupling.hcra.delta.2.weight',
             'bridge.denoiser.grg_ctf.':'bridge.denoiser.grg_ctf.net.2.weight'}
    for prefix, key in heads.items():
        # The unchanged frozen constructors explicitly initialize these output weights to zero.
        modules[prefix]['rank1_zero_initialized_output_weight_l2'] = float(selected['model'][key].double().norm())
    old_states = {}
    for label, root in [('formal','bcep_ebjd_v2_integrated'),('candidate_cost','zara2_costproxy_stage2_20261119'),
                        ('sharedworld','zara2_sharedworld_stage2_onerun_20261119')]:
        raw = read(RUN.parent/root/'STATUS.json', 'protected_'+label+'_status')
        old_states[label] = raw.get('status')
    recent = history[-5:]
    mean_train = st.mean(h['train_seconds'] for h in recent)
    mean_dev = st.mean(h['validation_seconds'] for h in recent)
    elapsed_steps = len(telemetry) - latest['update']
    stop_if_no_best = max(frozen['recipe']['minimum'], rank['epoch']+frozen['recipe']['patience'])
    remaining_seconds = max(0, (stop_if_no_best-latest['epoch'])*(mean_train+mean_dev)-elapsed_steps*mean_train/168)
    result = dict(status='RUNNING_INTERIM_NOT_FINAL',captured_at=captured.isoformat(),snapshot_started=started.isoformat(),
        runtime_source_commit=launch['runtime_code_commit'],frozen_sha256=final.canonical_hash(frozen),
        native_status=status,supervisor=supervisor,latest_native_progress=progress,
        completed_epochs=len(history),recorded_real_optimizer_updates=len(telemetry),last_recorded_step_unix=telemetry[-1]['unix'],
        last_saved_update=saved['update'],last_saved_completed_epoch=saved['completed_epoch'],
        rank1_epoch=rank['epoch'],stale_completed_epochs=latest['epoch']-rank['epoch'],
        old_epoch5=limits,new_rank1=values,new_minus_old_m=[n-o for n,o in zip(values,limits)],passes=passes,all_pass=all(passes),
        selector=decision_lock['selector'],fixed_source_unchanged=True,old_run_statuses=old_states,
        report_additional_real_optimizer_updates=0,report_additional_inference=0,
        all_completed_epoch_files_present=True,top5_hashes_match=True,last_and_rank1_recovery_fields_verified=True,
        last_and_best_alias_hashes_match=True,minFDE_all_completed_epochs_equal_old5_within_m=1e-6,
        initial_model_sha256=frozen['initial_model_sha256'],bank_sha256=frozen['bank_sha256'],split_sha256=frozen['split_sha256'],
        recent5_mean_train_seconds=mean_train,recent5_mean_dev_seconds=mean_dev,
        conditional_stop_epoch_if_no_new_best=stop_if_no_best,conditional_remaining_hours=remaining_seconds/3600,
        performance_audit_patches_applied=False)
    out = DOC/started.strftime('status_%Y%m%d_%H%M%S')
    out.mkdir(exist_ok=False)
    write(out/'SNAPSHOT.json', result)
    write(out/'EVIDENCE.json', evidence)
    write(out/'MODULE_SUMMARY.json', modules)
    write(out/'TOP5.json', [dict(epoch=r['epoch'],update=r['update'],H_JADE=r['validation'],file=r['file'],sha256=r['sha256']) for r in top])
    write(out/'REFERENCE_IDENTITY.json',dict(commit=final.REFERENCE_COMMIT,csv_sha256=old_meta['csv_sha256'],
          checkpoint=old_meta['identity'],archive_differences_m=old_meta['archive_differences_m']))
    csv_write(out/'EPOCH_GROUPS.csv', epoch_rows)
    csv_write(out/'EPOCH_RESOURCES.csv', costs)
    csv_write(out/'CHECKPOINTS.csv', checks)
    csv_write(out/'CURRENT_RANK1_VS_OLD5_WINDOWS.csv', paired)
    csv_write(out/'PAIRED_SUMMARY.csv', paired_summary)
    epoch_modules = []
    for h in history:
        per = final.module_summary([r for r in telemetry if (r['update']-1)//168+1 == h['epoch']])
        epoch_modules.extend(dict(epoch=h['epoch'],module=k,**v) for k,v in per.items())
    csv_write(out/'EPOCH_MODULES.csv', epoch_modules)
    (out/'.gitattributes').write_text('*.csv -text whitespace=cr-at-eol\n')
    stat = {(r['group'],r['metric']):r for r in paired_summary}
    table = '\n'.join(f'| {label} | {o:.9f} | {n:.9f} | {(n-o)*100:+.6f} | {"通过" if ok else "未通过"} |'
        for label,o,n,ok in zip(['全体 H-JADE ≤','全体 H-JFDE <','N≥9 H-JFDE <'],limits,values,passes))
    comparisons = []
    for group in ['all','multi','N_ge_9']:
        a,b,c = [stat[group,k] for k in ['I_JFDE','feedback_gain','H_JFDE']]
        comparisons.append(f"- {group}：初配 I-JFDE 新−旧 **{a['mean_delta_m']*100:+.6f} cm**；反馈收益 I−H 新−旧 **{b['mean_delta_m']*100:+.6f} cm**；最终 H-JFDE 新−旧 **{c['mean_delta_m']*100:+.6f} cm**。当前反馈收益 {b['new_mean_m']*100:+.6f} cm，旧版 {b['old_mean_m']*100:+.6f} cm。")
    distribution = '\n'.join(f"| {r['group']} | {r['metric']} | {r['windows']} | {r['mean_delta_m']*100:+.6f} | {r['median_delta_m']*100:+.6f} | {r['improved_windows']} / {r['regressed_windows']} / {r['equal_windows']} |" for r in paired_summary)
    module_table = '\n'.join(f"| {k} | {v['nonzero_gradient_steps']} | {v['nonzero_parameter_update_steps']} | {v['gradient_l2_mean']:.7g} | {v['rank1_zero_initialized_output_weight_l2']:.7g} |" for k,v in modules.items())
    top_table = '\n'.join(f"| {i} | {r['epoch']} | {r['validation']:.9f} |" for i,r in enumerate(top,1))
    latest_values = [latest_groups['all']['JADE'], latest_groups['all']['JFDE'], latest_groups['N_ge_9']['JFDE']]
    report = f'''# BCEP–EBJD v3 ZARA2 正式训练：当前情况分析

快照时间：**{captured.strftime('%Y-%m-%d %H:%M:%S %Z')}**。训练仍在后台运行，已完成 **{len(history)} 轮**，当前进入 epoch **{latest['epoch']+1}**。已记录 **{len(telemetry)} 次真实 optimizer 更新**；最近完整可恢复检查点为 epoch {saved['completed_epoch']} / update {saved['update']}。本次报告额外真实更新和额外推理均为 **0**。

当前唯一选模结果为 **epoch {rank['epoch']}**，按全部 590 个开发窗的 H-JADE 排名。**当前三项门槛尚未同时通过；训练未结束，以下均为中期结果。** 最新完整轮次 epoch {latest['epoch']} 与当前最优权重分开报告。

| 指标（m） | 旧 coordinate epoch 5 | 当前 v3 rank1 | 新−旧（cm） | 当前判断 |
|---|---:|---:|---:|---|
{table}

门槛判断使用 [SNAPSHOT.json](SNAPSHOT.json) 中的未舍入值。旧正式 rank1 epoch 26 仅作背景：H-JADE **0.4084707879413993 m**、H-JFDE **0.8016974424154072 m**、N≥9 H-JFDE **1.3188941478729248 m**。没有从 Top5 改选 JFDE 最低权重。

最新 epoch {latest['epoch']} 的上述三项指标依次是 **{latest_values[0]:.9f} / {latest_values[1]:.9f} / {latest_values[2]:.9f} m**。已有 **{latest['epoch']-rank['epoch']} 个完整轮次**未刷新 H-JADE 最优值，说明当前开发集指标仍有波动；单凭这段曲线不能判定过拟合原因，也不能判定某个模块失效。

## 初配、反馈与完整轨迹

以下为当前 rank1 与旧 epoch 5 在相同开发窗的配对比较。误差差值为新减旧，负值代表改善；反馈收益为 I−H，正值代表最终配对优于初配。

{chr(10).join(comparisons)}

当前 rank1 全体 G/I/H-JFDE 为 **{groups['all']['G_JFDE']:.9f} / {groups['all']['I_JFDE']:.9f} / {groups['all']['JFDE']:.9f} m**；N≥9 为 **{groups['N_ge_9']['G_JFDE']:.9f} / {groups['N_ge_9']['I_JFDE']:.9f} / {groups['N_ge_9']['JFDE']:.9f} m**。初配后合成反馈目前使误差上升，全体约 **{-groups['all']['feedback_gain']*100:.3f} cm**、密集窗口约 **{-groups['N_ge_9']['feedback_gain']*100:.3f} cm**。这直接支持检查反馈环节，但不能把“旧反馈 + HCRA 增量”的合成结果单独归因于 HCRA。

完整路径 H-JADE 相比旧 epoch 5：全体变化 **{stat['all','H_JADE']['mean_delta_m']*100:+.6f} cm**，N≥9 变化 **{stat['N_ge_9','H_JADE']['mean_delta_m']*100:+.6f} cm**。这些是 CSEE、HCRA、GRG-CTF 联合启用后的组合效果；没有单模块消融，独立贡献仍无法证明。

| 分组 | 指标 | 窗数 | 均值差 cm | 中位数差 cm | 改善 / 退步 / 相等窗 |
|---|---|---:|---:|---:|---|
{distribution}

all 为 590 窗，multi 为 501 窗，N≥9 为 127 窗；使用窗口等权，差值严格为零才计相等，CSV 另列绝对差≤1e-6 m 的窗数。逐窗验证 ΔH=ΔI−Δ(I−H) 在 1e-6 m 内成立。逐窗 CSV 的 H-JFDE 与原生轨迹汇总 JFDE 存在微小浮点差；门槛采用原生 JFDE，分解采用同一逐窗 CSV，二者不混用。

## 候选、梯度与参数更新

已对所有 {len(history)} 个完整 epoch 的 590 窗 minFDE 与 oracle 核对旧 epoch 5，差值均在 1e-6 m 内。全体边缘 minFDE 为 **{groups['all']['minFDE']:.16f} m**；窗口等权 oracle 为 **{groups['all']['scene_weighted_minFDE']:.16f} m**。终点候选保持相同，当前变化不能归因于候选覆盖变化。

| 模块 | 非零梯度更新数 | 非零参数变化更新数 | 平均梯度 L2 | rank1 零初始化输出权重 L2 |
|---|---:|---:|---:|---:|
{module_table}

梯度与变化来自成功 AdamW step 后记录，统计截至本快照 {len(telemetry)} 次更新。三个输出头在冻结源码中明确零初始化，rank1 非零输出权重进一步支持真实学习变化。CSEE 小梯度不构成失败判据；第一步无梯度符合初始零出口的已验收行为。这里只证明模块参与更新，不证明其独立效果。

## 训练速度、停止条件与恢复资产

最近 5 个完整 epoch 平均训练 **{mean_train/60:.2f} 分钟**、验证 **{mean_dev/60:.2f} 分钟**，合计约 **{(mean_train+mean_dev)/60:.2f} 分钟/轮**；原生训练计时约 **{mean_train/168:.2f} 秒/更新**。已完成轮次合计训练 {sum(h['train_seconds'] for h in history)/3600:.3f} 小时、验证 {sum(h['validation_seconds'] for h in history)/3600:.3f} 小时。保存/启动等额外开销不包含在纯训练加验证估计中。

仍按 minimum {frozen['recipe']['minimum']}、patience {frozen['recipe']['patience']}、max {frozen['recipe']['max_epochs']} 原生规则运行。如果此后不刷新最优值，预计在 epoch **{stop_if_no_best}** 满足早停；按近期速度，距该条件约 **{remaining_seconds/3600:.1f} 小时**。这是条件估计；后续新最优值可能推迟停止。没有人为提前停止或另起训练。

显存来自已有每 5 秒的进程级采样，见 [EPOCH_RESOURCES.csv](EPOCH_RESOURCES.csv)；包含驱动/上下文/保留显存，可能漏掉瞬时峰值，不能当作 PyTorch allocated 峰值或瓶颈因果证据。

全部 {len(history)} 个完整 epoch 文件存在；Top5 为下表，五个权重 SHA 均核对通过，best 与 rank1、last 与最新完整 epoch 的哈希一致。实际加载核验了 rank1 和 last 的模型、优化器、调度器、RNG、epoch/update、数据游标和早停字段。本次未逐一反序列化其余所有 epoch，文件存在不冒充逐个恢复实测。

| 排名 | epoch | dev H-JADE（m） |
|---|---:|---:|
{top_table}

原五折状态 `{old_states['formal']}`，candidate_cost 状态 `{old_states['candidate_cost']}`，sharedworld 状态 `{old_states['sharedworld']}`。本次仅只读核对这些状态；没有修改或恢复旧实验。原生源码全部文件与冻结清单一致，性能审计分支的优化尚未应用于此运行。

## 数据身份与证据

正式源码提交 `{launch['runtime_code_commit']}`；冻结配置 SHA `{final.canonical_hash(frozen)}`；初态 SHA `{frozen['initial_model_sha256']}`。Provider density epoch19 / head epoch9、seed2035 bank、seed3201 初始化、FP32、AdamW、final_coordinate、6层桥、K20、20步 DDIM 和原选模规则均保持本次已验收配置。这里核对冻结清单和归档身份，不重新执行准备入口或构建训练模型。

旧 epoch5 逐窗参考取自固定归档 `{final.REFERENCE_COMMIT}`，检查 CSV、checkpoint、split、bank 哈希及 K20/DDIM20；没有再运行旧模型。所有开发指标来自现有原生验证输出，没有测试集评测。

`STATUS.json` 中运行时 `optimizer_updates=0` 是启动时静态字段，**不代表当前没有训练**。实际更新数以本次成功 step 的完整日志前缀为准；共 {len(telemetry)} 条、update 从 1 连续，无重复，控制器仍为首次 attempt。正在写入的尾行不计入快照，所以报告是截至读取时的已确认值。

证据：[快照](SNAPSHOT.json)、[逐轮分组指标](EPOCH_GROUPS.csv)、[匿名逐窗配对差值](CURRENT_RANK1_VS_OLD5_WINDOWS.csv)、[配对汇总](PAIRED_SUMMARY.csv)、[逐轮模块统计](EPOCH_MODULES.csv)、[Top5](TOP5.json)、[读取前缀哈希](EVIDENCE.json)。权重、候选 bank 和原始日志留在训练主机。

只读复现命令（从本实验 worktree 执行，会生成新的时间快照，不启动训练）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 ../.conda/rsjg/bin/python scripts/v3_trial/status_snapshot.py
```

当前判断：组合模型暂未达到旧 epoch5 的三项联合门槛，反馈后终点误差上升是已有数据直接支持的问题；参数得到更新与单模块带来收益是不同结论。继续由既有后台控制器按原协议完成本次唯一实验，再以最终 H-JADE rank1 作结论；不扩折、不换 seed、不调整损失、不启动消融或测试集。
'''
    (out/'REPORT_ZH.md').write_text(report)
    print(json.dumps(dict(output=str(out.relative_to(SOURCE)),completed_epochs=len(history),updates=len(telemetry),
        rank1_epoch=rank['epoch'],passes=passes,paired_summary=paired_summary,modules=modules),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()

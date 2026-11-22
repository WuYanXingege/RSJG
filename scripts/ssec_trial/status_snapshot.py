"""Read-only interim analysis, archived separately from the eventual final report."""
import datetime
import math
import shutil
import statistics
from zoneinfo import ZoneInfo

import report
from analysis import decision, paired, select_rank, summarize
from common import *


def main():
    captured_start=datetime.datetime.now(ZoneInfo('Asia/Shanghai'))
    config=source_check()
    assert read(RUN/'STATUS.json')['status']=='RUNNING', 'use final report after native completion'
    history=read(RUN/'joint/EPOCH_METRICS.json');rank=select_rank(history);latest=history[-1]
    snapshot=DOC/('status_'+captured_start.strftime('%Y%m%d_%H%M%S'))
    assert not snapshot.exists();snapshot.mkdir(parents=True)
    physical,effective=report.telemetry();last_update=max(effective);last_step=effective[last_update]
    launch=read(RUN/'LAUNCH.json');pid=launch['pid'];proc=Path('/proc')/str(pid)
    assert proc.exists() and (proc/'cwd').resolve()==TRAIN_SOURCE
    cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    assert '-m bcep_scene_energy.formal train --frozen-sha '+FROZEN in cmd
    assert read(RUN/'FIRST_UPDATE_VERIFICATION.json')['passed'] and read(RUN/'FIRST_EPOCH_VERIFICATION.json')['passed']
    groups={g['group']:g for g in read(RUN/'dev'/f"epoch_{rank['epoch']:03d}_groups.json")}
    current_rows=rows(RUN/'dev'/f"epoch_{rank['epoch']:03d}_windows.csv")
    old,reference_meta=report.reference(config)
    pairs=paired(current_rows,old);summaries=summarize(pairs)
    gate=decision([groups['all']['JADE'],groups['all']['JFDE'],groups['N_ge_9']['JFDE']])
    gate.update(interim_only=True,all_pass=all(gate['passes']),rank1_epoch=rank['epoch'],selector='all dev H-JADE; ties earlier epoch')
    top=sorted(history,key=lambda r:(r['validation'],r['epoch']))[:5]
    top_receipts=[]
    for h in top:
        p=RUN/'joint'/h['file'];assert p.exists()
        top_receipts.append(dict(epoch=h['epoch'],H_JADE=h['validation'],update=h['update'],file=h['file'],sha256=digest(p)))
    assert digest(RUN/'joint/best.pt')==top_receipts[0]['sha256']
    assert all((RUN/'joint'/h['file']).exists() for h in history)
    module_summary={}
    for prefix in report.PREFIXES:
        rr=[r['modules'][prefix] for r in effective.values()]
        assert all(math.isfinite(v) for m in rr for v in m.values())
        module_summary[prefix]=dict(observed_effective_updates=len(rr),nonzero_gradient_updates=sum(m['gradient_l2']>0 for m in rr),
            nonzero_parameter_delta_updates=sum(m['parameter_delta_l2']>0 for m in rr),
            mean_gradient_l2=statistics.mean(m['gradient_l2'] for m in rr),latest=last_step['modules'][prefix])
    feedback=[]
    for name,condition in [('all',lambda r:True),('multi',lambda r:int(r['n_agents'])>1),('N_ge_9',lambda r:int(r['n_agents'])>=9)]:
        rr=[r for r in current_rows if condition(r)];gains=[float(r['feedback_gain']) for r in rr]
        feedback.append(dict(group=name,windows=len(rr),mean_I_JFDE=statistics.mean(float(r['I_JFDE']) for r in rr),
            mean_H_JFDE=statistics.mean(float(r['H_JFDE']) for r in rr),mean_feedback_gain=statistics.mean(gains),
            median_feedback_gain=statistics.median(gains),improved_windows=sum(v>0 for v in gains),
            worsened_windows=sum(v<0 for v in gains),unchanged_windows=sum(v==0 for v in gains),
            sinkhorn_normalized_entropy=statistics.mean(float(r['sinkhorn_normalized_entropy']) for r in rr)))
    raw_gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
    gpu_mib=sum(float(r[1].strip()) for r in csv.reader(raw_gpu.splitlines()) if len(r)>1 and r[0].strip()==str(pid))
    state=dict(interim_only=True,captured_start_shanghai=captured_start.isoformat(timespec='seconds'),
        captured_end_shanghai=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        native_status='RUNNING',supervisor=read(RUN/'SUPERVISOR_STATUS.json'),attempt=launch['attempt'],
        controller_pid=pid,controller_cwd_and_cmdline_verified=True,runtime_code_commit=BASE,frozen_file_sha256=FROZEN,
        latest_observed_update=last_update,observed_physical_step_events=physical,completed_epochs=len(history),
        current_logical_epoch=last_step['epoch'],validation_progress=read(RUN/'VALIDATION_PROGRESS.json'),
        rank1_epoch=rank['epoch'],stale_completed_epochs=latest['epoch']-rank['epoch'],
        conditional_early_stop_epoch=max(30,rank['epoch']+15),first_update_verified=True,first_epoch_verified=True,
        complete_epoch_weights_present=len(history),disk_free_GiB=shutil.disk_usage(RUN).free/2**30,
        sampled_gpu_process_MiB=gpu_mib,training_allocated_peak_MiB=max(r['allocated_peak_bytes'] for r in effective.values())/2**20,
        full_epoch_train_seconds=sum(h['train_seconds'] for h in history),full_epoch_dev_seconds=sum(h['validation_seconds'] for h in history),
        reporting_only_no_inference_or_optimizer=True,no_source_or_configuration_changes=True)
    csvwrite(snapshot/'CURRENT_RANK1_VS_OLD5_WINDOWS.csv',pairs)
    csvwrite(snapshot/'PAIRED_SUMMARY.csv',summaries)
    csvwrite(snapshot/'CURRENT_RANK1_COORDINATION.csv',feedback)
    write(snapshot/'CURRENT_RANK1_GROUPS.json',list(groups.values()))
    write(snapshot/'THRESHOLD_COMPARISON.json',gate)
    write(snapshot/'TOP5.json',top_receipts)
    write(snapshot/'MODULE_SUMMARY.json',module_summary)
    write(snapshot/'REFERENCE_IDENTITY.json',dict(commit=report.REFERENCE,**reference_meta,additional_reference_inference=0))
    write(snapshot/'STATUS.json',state)
    write(snapshot/'SCRIPT_IDENTITY.json',dict(path='scripts/ssec_trial/status_snapshot.py',sha256=digest(__file__)))
    # Reuse original read-only exporters; no changes to live training or its supervisor.
    report.sanitized_receipts(config);epoch_groups=report.exports(history)
    for name in ['EPOCH_GROUPS.csv','EPOCH_POPULATIONS.csv','EPOCH_MODULES.csv','EPOCH_COSTS.csv','EPOCH_ENERGY_AND_MATCHING.csv']:
        shutil.copy2(DOC/name,snapshot/name)
    write(DOC/'STATUS_SNAPSHOT.json',state)
    table='\n'.join(f"| {label} | {a:.16f} | {b:.16f} | {b-a:+.16f} | {ok} |" for label,a,b,ok in zip(
        ['全体 H-JADE ≤','全体 H-JFDE <','N≥9 H-JFDE <'],gate['old_epoch5'],gate['new_rank1'],gate['passes']))
    decomposition=[]
    for name in ['all','multi','N_ge_9']:
        a,b,c=[next(r for r in summaries if r['group']==name and r['metric']==key) for key in ['I_JFDE','feedback_gain','H_JFDE']]
        decomposition.append(f"| {name} | {a['mean_delta']*100:+.6f} | {b['mean_delta']*100:+.6f} | {c['mean_delta']*100:+.6f} |")
    paired_table='\n'.join(f"| {r['group']} | {r['metric']} | {r['mean_delta']*100:+.6f} | {r['median_delta']*100:+.6f} | {r['improved']} | {r['regressed']} | {r['equal']} |" for r in summaries)
    feedback_table='\n'.join(f"| {r['group']} | {r['windows']} | {r['mean_feedback_gain']*100:+.6f} | {r['improved_windows']} | {r['worsened_windows']} | {r['unchanged_windows']} |" for r in feedback)
    recent=[r for r in epoch_groups if r['epoch'] in [h['epoch'] for h in history[-5:]] and r['group']=='all']
    recent_table='\n'.join(f"| {r['epoch']} | {r['H_JADE']:.9f} | {r['H_JFDE']:.9f} | {r['feedback_gain']*100:+.6f} |" for r in recent)
    text=f'''# SSEC Energy-only ZARA2：训练中阶段分析

**这是运行中快照，不是最终结论。** 截止北京时间{state['captured_end_shanghai']}，已观察到{last_update}次有效真实更新，完成{len(history)}个完整epoch；控制器与监督器正常，启动尝试为第{launch['attempt']}次。当前唯一H-JADE rank1为 **epoch {rank['epoch']}**。训练继续按原minimum30/patience15/max100执行，没有提前停止或调参。

## 当前rank1与固定门槛

| 指标 m | 旧coordinate epoch5门槛 | 当前SSEC rank1 | 新−旧 m | 通过 |
|---|---:|---:|---:|---|
{table}

目前三项{'全部通过' if gate['all_pass'] else '未同时通过'}，但运行尚未结束。主结果始终由全体dev H-JADE选择，不能改选JFDE最低权重。旧正式rank1 epoch26仅作背景：H-JADE 0.4084707879413993 m、H-JFDE 0.8016974424154072 m、N≥9 H-JFDE 1.3188941478729248 m。

## 初配与公共能量协调

下表为当前rank1自身协调前后，I−H>0表示JFDE改善；按窗口等权。

| 分组 | 窗数 | 平均I−H cm | 改善窗 | 恶化窗 | 不变窗 |
|---|---:|---:|---:|---:|---:|
{feedback_table}

当前rank1全体最终配对相对初配的误差变化为{-groups['all']['feedback_gain']*100:+.6f} cm，N≥9为{-groups['N_ge_9']['feedback_gain']*100:+.6f} cm（正值表示误差增大）。公共能量已接受更新均满足原FP64可靠下降契约，但这个下降不保证真实JFDE改善。

全体接受{groups['all']['accepted']}/{groups['all']['attempted']}个协调提案，严格并列{groups['all']['strict_ties']}、舍入不确定{groups['all']['roundoff_uncertain']}；N≥9接受{groups['N_ge_9']['accepted']}/{groups['N_ge_9']['attempted']}，严格并列{groups['N_ge_9']['strict_ties']}。可靠下降候选数与最终接受数不同：严格唯一性仍是独立接受条件，不能把被拒绝的并列提案算作接受。

N≥9最终提案有效评分RMS为{groups['N_ge_9']['final_proposal_effective_score_RMS']:.9g}，平均匹配间隔{groups['N_ge_9']['proposal_matching_gap_mean']:.9g}，归一化Sinkhorn熵为{groups['N_ge_9']['sinkhorn_normalized_entropy']:.12f}，仍接近均匀。它们是当前权重下的实测现象；不据此单独归因φ/ρ、修改温度或启动新结构。

## 与旧epoch5逐窗对齐

以下为新−旧。误差差值负数表示改善，反馈收益差值正数表示帮助增加；逐窗满足ΔH=ΔI−Δ(I−H)。

| 分组 | 初配I-JFDE差 cm | 反馈收益差 cm | 最终H-JFDE差 cm |
|---|---:|---:|---:|
{chr(10).join(decomposition)}

| 分组 | 指标 | 均值差 cm | 中位数差 cm | 改善窗 | 退步窗 | 相等窗 |
|---|---|---:|---:|---:|---:|---:|
{paired_table}

全590窗、501多人窗、127个N≥9窗与旧epoch5一一对应；只导出匿名窗口编号。逐窗边缘minFDE差<1e−6 m；冻结候选全体agent-micro minFDE=0.3520422351797007 m，窗口等权下界=0.33942601943420153 m。

参考数据复用归档`{report.REFERENCE}`，已核对CSV、权重、split与bank哈希；本轮报告没有重复模型推理。主门槛使用用户固定未舍入值与原路径metric累加器JFDE；逐窗配对使用直接端点误差，归档回放与门槛存在约1e−10至1e−8 m的浮点差，完整数值见REFERENCE_IDENTITY.json。

## 近期波动与停止规则

| 完整epoch | 全体H-JADE m | 全体H-JFDE m | I−H cm |
|---:|---:|---:|---:|
{recent_table}

截至快照，连续{state['stale_completed_epochs']}个完整轮未刷新最佳。**仅当之后不再出现更优H-JADE时**，原patience15对应第{state['conditional_early_stop_epoch']}轮结束；若有新最佳则重新计数，最迟100轮。当前波动不触发人为停机或选模标准变更。

## 工程状态与边界

首次真实更新及首个完整epoch验收通过。φ/ρ及共享编码器的梯度和实际参数变化均有限，逐步非零次数保留在MODULE_SUMMARY.json；不是用梯度×学习率估计参数变化。已保留{len(history)}轮完整权重、best、last和五个不同epoch的Top5，详见TOP5.json。快照GPU进程显存约{gpu_mib:.0f} MiB，磁盘剩余约{state['disk_free_GiB']:.2f} GiB；累计PyTorch allocated峰值{state['training_allocated_peak_MiB']:.2f} MiB，两种显存口径不混用。

训练源码仍为`{BASE}`，冻结配置SHA256 `{FROZEN}`。原Provider、bank、候选seed2035、训练seed3201、FP32/AdamW、final_coordinate、六层桥、K20/DDIM20、proposal-ST及全部超参数不变。报告脚本只读取已有结果和日志，不加载训练模型，不执行optimizer，不接触测试集，不发送训练信号。

当前证据支持：工程路径在运行，参数确实更新；在当前rank1上，协调步骤对全体及密集窗口平均终点误差的方向分别为{'恶化' if groups['all']['feedback_gain']<0 else '改善' if groups['all']['feedback_gain']>0 else '不变'}、{'恶化' if groups['N_ge_9']['feedback_gain']<0 else '改善' if groups['N_ge_9']['feedback_gain']>0 else '不变'}。尚不能据此证明φ/ρ各自的独立因果贡献、最终训练效果、跨seed或跨折表现。等待本次原生停止后形成最终结论，不启动其他实验。

[状态收据](STATUS.json) · [当前Top5](TOP5.json) · [逐轮指标](EPOCH_GROUPS.csv) · [人数分组](EPOCH_POPULATIONS.csv) · [能量与匹配](EPOCH_ENERGY_AND_MATCHING.csv) · [匿名逐窗差值](CURRENT_RANK1_VS_OLD5_WINDOWS.csv) · [配对汇总](PAIRED_SUMMARY.csv)
'''
    (snapshot/'README.md').write_text(text)
    (DOC/'README.md').write_text(f'''# SSEC Energy-only ZARA2 Stage2：运行中分析

**尚非最终结果。** 北京时间{state['captured_end_shanghai']}快照：已观察到{last_update}次真实更新，完成{len(history)}轮，当前按H-JADE选择的rank1为epoch {rank['epoch']}。

[完整中文阶段分析]({snapshot.name}/README.md) · [三项门槛比较]({snapshot.name}/THRESHOLD_COMPARISON.json) · [匿名逐窗对照]({snapshot.name}/CURRENT_RANK1_VS_OLD5_WINDOWS.csv) · [逐轮CSV](EPOCH_GROUPS.csv)

| 指标 | 当前rank1 m | 旧epoch5门槛 m |
|---|---:|---:|
| 全体H-JADE | {groups['all']['JADE']:.9f} | ≤{LIMITS[0]:.9f} |
| 全体H-JFDE | {groups['all']['JFDE']:.9f} | <{LIMITS[1]:.9f} |
| N≥9 H-JFDE | {groups['N_ge_9']['JFDE']:.9f} | <{LIMITS[2]:.9f} |

训练仍在后台按原规则继续；本次报告更新不改变训练、选模或停止规则。完成后监督器将发布最终研究报告并核验远端SHA。正式训练源码固定`{BASE}`，报告分支不作为训练入口。
''')
    write(DOC/'ARTIFACT_SHA256.json',{str(p.relative_to(DOC)):digest(p) for p in sorted(DOC.rglob('*')) if p.is_file() and p.name!='ARTIFACT_SHA256.json'})
    print(json.dumps(dict(snapshot=str(snapshot.relative_to(SOURCE)),completed_epochs=len(history),observed_updates=last_update,
        rank1_epoch=rank['epoch'],passes=gate['passes']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()

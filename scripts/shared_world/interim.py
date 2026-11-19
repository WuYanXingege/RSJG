"""Read-only publication snapshot; does not signal or alter the active run."""
import csv,json,time,hashlib,shutil,subprocess,sys
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import torch
import one_run as trial
from bcep_joint.common import read,write
from bcep.contracts import digest
from bcep_v2.formal_manifest import write_csv

w=trial.SOURCE.parent;up,cost,run=trial.paths(w);stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
out=trial.SOURCE/'docs/research/bcep_v2/sharedworld_stage2_onerun_20261119'/('interim_'+stamp);out.mkdir(parents=True)
status=read(run/'STATUS.json');progress=read(run/'joint/PROGRESS.json');history=read(run/'joint/EPOCH_METRICS.json');top=read(run/'joint/TOP5.json');frozen=read(run/'FROZEN_CONFIG.json')
assert status['status']=='RUNNING'
rank=min(history,key=lambda e:(e['validation'],e['epoch']))['epoch'];assert rank==top[0]['epoch']
state=torch.load(run/'joint/last.pt',map_location='cpu',weights_only=False)
assert state['identity']==frozen['identity'];saved=state['update'];completed=state['completed_epoch'];del state
rows=[]
for e in history:
 for g in read(run/'dev'/f"epoch_{e['epoch']:03d}_groups.json"):rows.append({k:v for k,v in g.items() if not isinstance(v,dict)})
groups={g['group']:g for g in rows if g['epoch']==rank};reference=frozen['decision']['epoch5_reference']
values=[groups['all']['JADE'],groups['all']['JFDE'],groups['N_ge_9']['JFDE']];checks=[values[0]<=reference[0],values[1]<reference[1],values[2]<reference[2]]
current={k:progress[k] for k in ['epoch','update','cursor','unix','grad_norm','metrics']}
receipt=dict(status='RUNNING_INTERIM_NOT_FINAL',snapshot_unix=time.time(),snapshot_timezone='Asia/Shanghai',progress=current,last_saved_update=saved,completed_epochs=completed,current_rank1_epoch=rank,rank1_values=values,old_epoch5=reference,passes_at_snapshot=checks,training_continues=True,stop_signal_sent=False,not_native_early_stop=True,real_update_cap=16800)
write(out/'SNAPSHOT.json',receipt)
for name in ['FROZEN_CONFIG.json','SMOKE.json','FIRST_UPDATE_VERIFICATION.json']:shutil.copy2(run/name,out/name)
write(out/'TOP5.json',[dict(epoch=r['epoch'],update=r['update'],dev_H_JADE=r['validation'],file='joint/'+r['file'],sha256=r['sha256']) for r in top])
for r in top:assert digest(run/'joint'/r['file'])==r['sha256']
write_csv(out/'EPOCH_GROUPS.csv',rows)
write_csv(out/'EPOCH_TRAIN.csv',[dict(epoch=e['epoch'],update=e['update'],dev_H_JADE=e['validation'],train_seconds=e['train_seconds'],validation_seconds=e['validation_seconds'],train_a_mean=e['train_pair_a_mean_scene_macro'],train_q_a=e['train_pair_q_a_scene_macro']) for e in history])
shutil.copy2(run/'dev'/f'epoch_{rank:03d}_windows.csv',out/'CURRENT_RANK1_WINDOWS.csv')
preserved={}
for k,h in read(run/'PROTECTED_BEFORE.json').items():
 label,rel=k.split('/',1);preserved[k]=digest((up if label=='formal' else cost)/rel)==h
assert all(preserved.values());write(out/'PRESERVATION.json',preserved)
with (trial.AUDIT/'GROUPS.csv').open() as f:old={r['group']:r for r in csv.DictReader(f) if r['epoch']=='5'}
old_I=[float(old['all']['I_JFDE']),float(old['N_ge_9']['I_JFDE'])]
parts=[]
for k,i,oldh in zip(['all','N_ge_9'],old_I,[reference[1],reference[2]]):
 new=groups[k];di=new['I_JFDE']-i;dg=new['feedback_gain']-(i-oldh);dh=new['JFDE']-oldh
 assert abs(dh-(di-dg))<1e-6
 parts.append(dict(group=k,old_I_JFDE=i,new_I_JFDE=new['I_JFDE'],delta_I_m=di,old_feedback_gain=i-oldh,new_feedback_gain=new['feedback_gain'],delta_feedback_gain_m=dg,delta_H_m=dh))
write_csv(out/'CURRENT_DECOMPOSITION.csv',parts)
lines=[f'''# ZARA2 shared_world_bounded 训练中期状态（{stamp}，Asia/Shanghai）

**训练仍在运行，本报告不是最终验收结论。** 已完整完成 {completed} 轮，当前进入第 {progress['epoch']} 轮。最新周期日志记录真实 update **{progress['update']}**（该时刻已执行，运行继续增长）；最后完整可恢复 checkpoint 为 update **{saved}**。本次快照不暂停、不重启、不更改早停或损失。

当前按 dev H-JADE 的 rank1 是 **epoch {rank}**。三项中有 {sum(checks)}/3 项达到旧 epoch 5 的预先门槛；{'当前三项均达线，但不提前停止。' if all(checks) else '尚未达到三项同时通过。'}最终必须训练到原生早停或 100 轮上限，再以最终 H-JADE rank1 判定，不能用中期某个 JFDE 较低的权重替换主结果。

| 开发集指标（m） | 旧 epoch 5 门槛 | 当前 rank1 | 新−旧（m） | 新−旧（cm） | 当前达线 |
|---|---:|---:|---:|---:|---|''']
for label,o,n,ok in zip(['全体 H-JADE ≤','全体 H-JFDE <','N≥9 H-JFDE <'],reference,values,checks):lines.append(f'| {label} | {o:.16f} | {n:.16f} | {n-o:+.9f} | {(n-o)*100:+.6f} | {ok} |')
lines.append('''
## 目前变化来自哪里

初配 I 与反馈后 H 使用同一批固定候选。反馈收益=I−H，正值表示反馈降低终点误差；新旧分解为 ΔH=ΔI−Δ反馈收益。

| 分组 | 初配误差新−旧 cm | 反馈收益新−旧 cm | 最终 H 误差新−旧 cm |
|---|---:|---:|---:|''')
for r in parts:lines.append(f"| {r['group']} | {100*r['delta_I_m']:+.5f} | {100*r['delta_feedback_gain_m']:+.5f} | {100*r['delta_H_m']:+.5f} |")
for r in parts:
 label='全体窗口' if r['group']=='all' else 'N≥9 窗口'
 initial='改善' if r['delta_I_m']<0 else '退步'
 feedback='增加' if r['delta_feedback_gain_m']>0 else '减少'
 final='降低' if r['delta_H_m']<0 else '升高'
 lines.append(f"\n{label}：初配误差{initial} {abs(r['delta_I_m'])*100:.5f} cm，反馈收益{feedback} {abs(r['delta_feedback_gain_m'])*100:.5f} cm，最终 H-JFDE {final} {abs(r['delta_H_m'])*100:.5f} cm。这是当前权重的观测分解，不是最终训练结论。")
lines.append(f'''
完整轨迹 H-JADE 新减旧为 {(values[0]-reference[0])*100:+.5f} cm。当前 {completed} 个完整轮次不足以替代原生早停后的最终判定。

## 已完成轮次

| epoch | 组别 | 窗数 | H-JADE | I-JFDE | H-JFDE | 反馈收益 |
|---:|---|---:|---:|---:|---:|---:|''')
for r in rows:lines.append(f"| {r['epoch']} | {r['group']} | {r['scene_count']} | {r['JADE']:.6f} | {r['I_JFDE']:.6f} | {r['JFDE']:.6f} | {r['feedback_gain']:+.6f} |")
lines.append(f'''
两轮全体逐人 minFDE 均为 0.3520422351797007 m，窗口等权 oracle 为 0.33942601943420153 m；逐窗候选与轨迹末帧的边缘误差在 1e-6 m 内一致。全体 590 窗，N≥9 组 127 窗。当前只完成 {completed} 轮，因此 Top5 清单含 {len(top)} 个不同完整轮次，全部保留；满五轮后保留至少五个最佳不同轮次，同时保留所有完整 epoch、best、last。

## 身份与运行核验

- 独立分支：`experiment/zara2-sharedworld-stage2-onerun-20261119`；报告基点 `7cea557556386d9fbc6e905e02cf3b24640a8d03`，已验收实现 `f17571c6836287408514736855e156fd5b825828`。
- 训练入口代码 `{frozen['code_commit']}`；配置 SHA256 `{frozen['config_sha256']}`。审计归档未改，本运行独立打开 RUN_ONE_ZARA2=true、allow_real_training=true，上限 16800 更新。
- 训练 seed 3201，候选/验证 seed 2035；train 2681、dev 590；Provider、修正头与 bank 均冻结。初始参数指纹和旧 coordinate 完全一致。
- 首个真实 step 已验证优化器计数、参数变化；共同损失与梯度范数和旧首步完全相等。反馈零初始化 I=H，故这是预期行为；后续日志的 q·a 已非零，实际 backward 使用 shared_world_bounded。
- 沿用原 AdamW、FP32、增广/数据顺序、物理分组、4 窗微批×4 累积、原尾批规则；5 轮 warmup、100 轮调度、minimum 30、patience 15。每完整轮 168 更新和一次原 20 步 DDIM 验证。
- 旧五折仍 PAUSED_RESUMABLE（ZARA1/head update 7097），candidate_cost 仍 SESSION_BUDGET_PAUSED（update 1903）；受保护状态、last、Top5 哈希未变。本次发布没有启动额外折、测试集或新 seed。

已完成轮次训练耗时合计 {sum(e['train_seconds'] for e in history)/60:.2f} 分钟、验证 {sum(e['validation_seconds'] for e in history)/60:.2f} 分钟，不含当前未完成轮次。当前报告只读收集已有结果，没有额外 DDIM 或优化器更新。

## 文件与后续

[周期快照](SNAPSHOT.json) · [逐轮分组 CSV](EPOCH_GROUPS.csv) · [当前 rank1 匿名逐窗 CSV](CURRENT_RANK1_WINDOWS.csv) · [反馈分解](CURRENT_DECOMPOSITION.csv) · [Top5 清单](TOP5.json) · [冻结配置](FROZEN_CONFIG.json) · [首步核验](FIRST_UPDATE_VERIFICATION.json) · [旧资产保全](PRESERVATION.json)。权重、bank 和内部绝对路径保留在主机。

训练继续至原生早停或轮数上限。届时才生成最终 rank1 对旧 epoch 5 的匿名逐窗均值差、中位差和改善/退步窗数，核验完整 H-JADE 与两项 H-JFDE；最终结果无论通过与否，都不会自动扩折。
''')
(out/'REPORT_ZH.md').write_text('\n'.join(lines))
write(out/'ARTIFACT_SHA256.json',{p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='ARTIFACT_SHA256.json'})
print(out.relative_to(trial.SOURCE))

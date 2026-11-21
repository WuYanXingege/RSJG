"""Read-only native-output report; finalization never loads a training model."""
import argparse, io, math, statistics, sys, time
from common import *
from analysis import select_rank, decision, paired, summarize

REFERENCE='76d80b508ad79f52965c15109b58d2d835473fd7'
REFERENCE_DIR='docs/research/bcep_v2/sharedworld_stage2_onerun_20261119/'
PREFIXES=['coupling.energy.phi.','coupling.energy.rho.','bridge.encoder.']


def blob(name):return subprocess.check_output(['git','show',REFERENCE+':'+REFERENCE_DIR+name],cwd=SOURCE)


def reference(config):
    raw=blob('OLD5_REFERENCE_WINDOWS.csv');meta=json.loads(blob('REFERENCE_REPLAY.json'));f=json.loads(blob('FROZEN_CONFIG.json'))
    assert hashlib.sha256(raw).hexdigest()==meta['csv_sha256']
    assert meta['identity']['split_sha256']==config['split_sha256']==f['identity']['split']
    assert config['bank_sha256']==dict(train=f['identity']['train_bank'],selection_dev=f['identity']['dev_bank'])
    assert meta['identity']['checkpoint_sha256']=='9281673fda5c439bdd463b0bfc9af1a65a3a0fea30941ad3c956cbb617943317'
    assert meta['identity']['candidate_seed']==2035 and meta['identity']['steps']==meta['identity']['K']==20
    return list(csv.DictReader(io.StringIO(raw.decode()))),meta


def telemetry():
    effective={};physical=0
    with (RUN/'MODULE_UPDATES.jsonl').open() as f:
        for line in f:
            try:r=json.loads(line)
            except json.JSONDecodeError:continue
            physical+=1
            effective[r['update']]=dict(update=r['update'],epoch=r['epoch'],modules=r['modules'],
                allocated_peak_bytes=r['allocated_peak_bytes'],seconds=r['logical_group_forward_backward_seconds'])
    return physical,effective


def exports(history):
    group_rows=[];pop_rows=[];module_rows=[];cost_rows=[];monitor_rows=[]
    for h in history:
        e=h['epoch'];rr=rows(RUN/'dev'/f'epoch_{e:03d}_windows.csv')
        assert len(rr)==590 and all(r['exact_endpoint']==r['exact_noise_identity']==r['certified_monotone']=='True' for r in rr)
        for g in read(RUN/'dev'/f'epoch_{e:03d}_groups.json'):
            group_rows.append(dict(epoch=e,update=h['update'],group=g['group'],windows=g['scene_count'],
                H_JADE=g['JADE'],G_JFDE=g['G_JFDE'],I_JFDE=g['I_JFDE'],H_JFDE=g['JFDE'],
                endpoint_H_JFDE=g['H_JFDE'],feedback_gain=g['feedback_gain'],minFDE_agent_micro=g['minFDE'],
                minFDE_scene=g['scene_weighted_minFDE'],train_seconds=h.get('train_seconds'),validation_seconds=h.get('validation_seconds')))
            monitor_rows.append(dict(epoch=e,group=g['group'],attempted=g['attempted'],accepted=g['accepted'],rejected=g['rejected'],
                strict_ties=g['strict_ties'],reliable_decreases=g['reliable_decreases'],roundoff_uncertain=g['roundoff_uncertain'],
                accepted_energy_delta_sum=g['accepted_energy_delta_sum'],effective_score_RMS=g['final_proposal_effective_score_RMS'],
                matching_gap_mean=g['proposal_matching_gap_mean'],sinkhorn_normalized_entropy=g['sinkhorn_normalized_entropy']))
        for label in ['1','2-8','9-19','20-41','42-57']:
            sub=[r for r in rr if r['population']==label]
            if sub:
                pop_rows.append(dict(epoch=e,population=label,windows=len(sub),**{key:statistics.mean(float(r[key]) for r in sub)
                    for key in ['H_JADE','G_JFDE','I_JFDE','H_JFDE','feedback_gain','minFDE_scene','final_proposal_effective_score_RMS','proposal_matching_gap_mean','sinkhorn_normalized_entropy']},
                    **{key:sum(int(r[key]) for r in sub) for key in ['attempted','accepted','rejected','strict_ties','roundoff_uncertain']}))
        training=read(RUN/'dev'/f'epoch_{e:03d}_training.json');assert training['optimizer_updates']==168
        for prefix,value in training['modules_mean_per_update'].items():module_rows.append(dict(epoch=e,module=prefix,**value))
        for r in rows(RUN/'dev'/f'epoch_{e:03d}_costs.csv'):cost_rows.append(r)
    for name,rr in [('EPOCH_GROUPS.csv',group_rows),('EPOCH_POPULATIONS.csv',pop_rows),('EPOCH_MODULES.csv',module_rows),('EPOCH_COSTS.csv',cost_rows),('EPOCH_ENERGY_AND_MATCHING.csv',monitor_rows)]:csvwrite(DOC/name,rr)
    return group_rows


def sanitized_receipts(config):
    DOC.mkdir(parents=True,exist_ok=True)
    write(DOC/'FROZEN_CONFIG.json',config)
    write(DOC/'SOURCE_IDENTITY.json',dict(runtime_code_commit=BASE,recovery_acceptance_commit=RECOVERY_BASE,
        frozen_file_sha256=FROZEN,implementation_hash=config['implementation_hash'],initial_model_sha256=config['initial_model_sha256'],
        provider=config['provider'],bank_sha256=config['bank_sha256'],split_sha256=config['split_sha256']))
    for name in ['FIRST_UPDATE_VERIFICATION.json','FIRST_EPOCH_VERIFICATION.json','PROCESS_ATTEMPTS.json']:
        if (RUN/name).exists():write(DOC/name,read(RUN/name))
    p=read(RUN/'PREFLIGHT.json');p.pop('exact_command',None);write(DOC/'PREFLIGHT.json',p)
    auth=read(RUN/'EXECUTION_AUTHORIZATION.json');auth.pop('exact_command',None);write(DOC/'EXECUTION_AUTHORIZATION.json',auth)
    (DOC/'REPRODUCE.md').write_text(f'''# 固定源码与唯一训练命令

正式训练代码固定为`{BASE}`，报告工作区从未作为训练入口。非零恢复验收固定为`{RECOVERY_BASE}`。

原正式源码中的唯一受控命令：
[CONTROLLED_COMMAND.txt](https://github.com/WuYanXingege/RSJG/blob/{BASE}/docs/research/bcep_scene_energy/stage2_ready_20261121/CONTROLLED_COMMAND.txt)

冻结配置文件SHA256：`{FROZEN}`。本次已启动，不能再次创建随机初态实验；技术恢复只能由同一命令读取本次`joint/last.pt`。

报告生成仅使用原生开发集CSV、完整checkpoint状态和日志，不调用optimizer、模型推理或测试集。训练结束后可用`python scripts/ssec_trial/report.py final`重建报告；运行中可用`python scripts/ssec_trial/report.py interim`读取快照。
''')


def interim(config):
    status=read(RUN/'STATUS.json');progress=read(RUN/'joint/PROGRESS.json') if (RUN/'joint/PROGRESS.json').exists() else {}
    history=read(RUN/'joint/EPOCH_METRICS.json') if (RUN/'joint/EPOCH_METRICS.json').exists() else []
    group_rows=exports(history) if history else []
    last=history[-1] if history else None
    physical,effective=telemetry() if (RUN/'MODULE_UPDATES.jsonl').exists() else (0,{})
    snapshot=dict(status='RUNNING_INTERIM_NOT_FINAL',native_status=status,reported_progress=progress,
        completed_epochs=len(history),latest_observed_update=max(effective,default=0),recorded_physical_steps=physical,
        first_update_passed=(RUN/'FIRST_UPDATE_VERIFICATION.json').exists(),first_epoch_passed=(RUN/'FIRST_EPOCH_VERIFICATION.json').exists(),
        thresholds=LIMITS,runtime_code_commit=BASE,unix=time.time())
    write(DOC/'STATUS_SNAPSHOT.json',snapshot)
    table='尚无完整epoch的开发集结果。'
    if last:
        gg=[g for g in group_rows if g['epoch']==last['epoch']]
        table='| 分组 | H-JADE m | I-JFDE m | H-JFDE m | I−H m |\n|---|---:|---:|---:|---:|\n'+'\n'.join(f"| {g['group']} | {g['H_JADE']:.9f} | {g['I_JFDE']:.9f} | {g['H_JFDE']:.9f} | {g['feedback_gain']:+.9f} |" for g in gg)
    (DOC/'README.md').write_text(f'''# SSEC Energy-only ZARA2 Stage2：运行中收据

**训练仍在运行，尚无最终结论。** 固定源码`{BASE}`，唯一正式运行，seed3201。快照记录有效逻辑更新{snapshot['latest_observed_update']}、已完成{len(history)}轮；首次真实更新验收{snapshot['first_update_passed']}，首次完整epoch验收{snapshot['first_epoch_passed']}。

最近完整epoch：{last['epoch'] if last else '暂无'}。以下为当时结果，不是最终rank1，不据此提前停止或调参。

{table}

原minimum30/patience15/max100、H-JADE唯一选模和全部冻结条件保持。后台监督器只执行唯一受控命令；技术故障仅从本次last恢复；结束后自动生成最终报告、匿名逐窗比较、全epoch/Top5哈希并推送本独立分支。未启动任何其他seed/折/测试集或旧队列。

[状态快照](STATUS_SNAPSHOT.json) · [源码/配置身份](SOURCE_IDENTITY.json) · [启动核验](PREFLIGHT.json) · [唯一命令与复现](REPRODUCE.md)
''')


def final(config):
    import torch
    torch.set_num_threads(1)
    status=read(RUN/'STATUS.json');assert status['status']=='COMPLETE' and status['stop_reason'] in ['early_stop','max_epochs']
    history=read(RUN/'joint/EPOCH_METRICS.json');top=read(RUN/'joint/TOP5.json');rank=select_rank(history)
    assert len(history)>=30 and len(history)<=100 and status['completed_epoch']==len(history)
    assert status['optimizer_updates']==168*len(history)
    assert [r['epoch'] for r in top]==[r['epoch'] for r in sorted(history,key=lambda r:(r['validation'],r['epoch']))[:5]]
    checkpoints=[]
    for h in history:
        p=RUN/'joint'/h['file'];s=torch.load(p,map_location='cpu',weights_only=False)
        assert s['identity']==config['identity'] and s['recipe']==config['recipe']
        assert s['completed_epoch']==h['epoch'] and s['update']==168*h['epoch'] and s['cursor']==0 and s['order'] is None
        assert {'model','optimizer','scheduler','rng','history','stale','trainable'}<=s.keys()
        assert {'cpu','cuda','python','numpy','order','view'}<=s['rng'].keys()
        checkpoints.append(dict(epoch=h['epoch'],update=s['update'],H_JADE=h['validation'],file=h['file'],sha256=digest(p)))
        del s
    rank_sha=next(r['sha256'] for r in checkpoints if r['epoch']==rank['epoch'])
    assert digest(RUN/'joint/best.pt')==rank_sha and digest(RUN/'joint/last.pt')==checkpoints[-1]['sha256']
    assert all(r['sha256']==next(s['sha256'] for s in checkpoints if s['epoch']==r['epoch']) for r in top)
    old,meta=reference(config);new=rows(RUN/'dev'/f"epoch_{rank['epoch']:03d}_windows.csv")
    pp=paired(new,old);ss=summarize(pp);gg={r['group']:r for r in read(RUN/'dev'/f"epoch_{rank['epoch']:03d}_groups.json")}
    values=[gg['all']['JADE'],gg['all']['JFDE'],gg['N_ge_9']['JFDE']]
    d=decision(values);d.update(all_pass=all(d['passes']),rank1_epoch=rank['epoch'],selector='all dev H-JADE; ties earlier epoch',
        completed_epochs=len(history),effective_real_optimizer_updates=status['optimizer_updates'],stop_reason=status['stop_reason'],
        no_test=True,no_followup=True)
    physical,effective=telemetry();assert set(effective)==set(range(1,status['optimizer_updates']+1))
    d['recorded_physical_step_events']=physical;d['replayed_step_events']=physical-len(effective)
    attempts=read(RUN/'PROCESS_ATTEMPTS.json');d['physical_step_total_exact']=len(attempts)==1 and physical==len(effective)
    module_summary={}
    for prefix in PREFIXES:
        mm=[r['modules'][prefix] for r in effective.values()]
        assert all(math.isfinite(v) for m in mm for v in m.values())
        module_summary[prefix]=dict(effective_steps=len(mm),nonzero_gradient_steps=sum(m['gradient_l2']>0 for m in mm),
            nonzero_parameter_delta_steps=sum(m['parameter_delta_l2']>0 for m in mm),
            mean_gradient_l2=statistics.mean(m['gradient_l2'] for m in mm),sum_parameter_delta_l2=sum(m['parameter_delta_l2'] for m in mm))
    protected=read(RUN/'PROTECTED_ASSETS_BEFORE.json');root=RUN.parent
    roots=dict(formal=root/'bcep_ebjd_v2_integrated',v3=root/'zara2_bcep_v3_stage2_20261120',candidate_cost=root/'zara2_costproxy_stage2_20261119',sharedworld=root/'zara2_sharedworld_stage2_onerun_20261119')
    for name,v in protected.items():
        label,rel=name.split('/',1);p=roots[label]/rel;assert p.stat().st_size==v['bytes'] and digest(p)==v['sha256'],name
    exports(history)
    csvwrite(DOC/'ALL_CHECKPOINTS.csv',checkpoints);csvwrite(DOC/'RANK1_VS_OLD5_WINDOWS.csv',pp);csvwrite(DOC/'PAIRED_SUMMARY.csv',ss)
    csvwrite(DOC/'RANK1_WINDOWS.csv',[dict(window=f'w{i:04d}',**{k:v for k,v in r.items() if k!='index'}) for i,r in enumerate(new,1)])
    write(DOC/'DECISION.json',d);write(DOC/'TOP5.json',top);write(DOC/'MODULE_SUMMARY.json',module_summary)
    write(DOC/'ASSET_PROTECTION.json',dict(unchanged=True,protected_files=len(protected),protected_bytes=sum(r['bytes'] for r in protected.values())))
    write(DOC/'REFERENCE_IDENTITY.json',dict(commit=REFERENCE,**meta,additional_reference_inference=0,gate_thresholds=LIMITS))
    write(DOC/'TRAINING_RECEIPT.json',dict(status=status,attempts=attempts,physical_step_events=physical,
        runtime_code_commit=BASE,frozen_file_sha256=FROZEN,rank1_sha256=rank_sha,
        last_sha256=checkpoints[-1]['sha256'],initial_model_sha256=config['initial_model_sha256']))
    write(DOC/'RESOURCE_SUMMARY.json',dict(train_seconds=sum(h['train_seconds'] for h in history),
        validation_seconds=sum(h['validation_seconds'] for h in history),
        measured_logical_step_forward_backward_seconds=sum(r['seconds'] for r in effective.values()),
        training_allocated_peak_bytes=max(r['allocated_peak_bytes'] for r in effective.values()),
        recovery_notes='logical steps deduplicated by latest native event; physical attempts separately retained'))
    header='本折三项全部通过；结束后等待独立验证决策。' if d['all_pass'] else '本折未同时通过三项门槛；保留旧coordinate基线，结束本轮。'
    table='\n'.join(f'| {k} | {o:.16f} | {n:.16f} | {n-o:+.16f} | {ok} |' for k,o,n,ok in zip(['全体 H-JADE ≤','全体 H-JFDE <','N≥9 H-JFDE <'],LIMITS,values,d['passes']))
    stats={(r['group'],r['metric']):r for r in ss};mechanism=[]
    for group in ['all','multi','N_ge_9']:
        a,b,c=[stats[group,k] for k in ['I_JFDE','feedback_gain','H_JFDE']]
        mechanism.append(f"- {group}：初配I-JFDE新−旧 {a['mean_delta']*100:+.6f} cm；反馈收益I−H新−旧 {b['mean_delta']*100:+.6f} cm；最终H-JFDE新−旧 {c['mean_delta']*100:+.6f} cm。SSEC自身平均反馈收益 {b['new_mean']*100:+.6f} cm。")
    distribution='\n'.join(f"| {r['group']} | {r['metric']} | {r['windows']} | {r['mean_delta']*100:+.6f} | {r['median_delta']*100:+.6f} | {r['improved']} | {r['regressed']} | {r['equal']} |" for r in ss)
    module_table='\n'.join(f"| {k} | {v['nonzero_gradient_steps']} | {v['nonzero_parameter_delta_steps']} | {v['mean_gradient_l2']:.9g} |" for k,v in module_summary.items())
    (DOC/'README.md').write_text(f'''# SSEC Energy-only：唯一ZARA2 Stage2正式训练最终报告

**{header}** 原生停止原因`{status['stop_reason']}`，完整{len(history)}轮，累计有效真实更新{status['optimizer_updates']}。记录到物理step事件{physical}，重放事件{physical-len(effective)}。唯一主结果为按全体dev H-JADE选出的 **epoch {rank['epoch']} / rank1**；没有另选JFDE最低权重。

| 固定指标 m | 旧coordinate epoch5 | SSEC rank1 | 新−旧 m | 通过 |
|---|---:|---:|---:|---|
{table}

三项按用户固定未舍入门槛同时判断。旧正式rank1 epoch26仅作背景：H-JADE 0.4084707879413993、H-JFDE 0.8016974424154072、N≥9 H-JFDE 1.3188941478729248 m。

## 初配、协调和完整轨迹

下列负的误差差值表示改善，正的反馈收益差值表示协调帮助增加；逐窗检查ΔH=ΔI−Δ(I−H)。

{chr(10).join(mechanism)}

完整路径全体H-JADE新−旧为{stats['all','H_JADE']['mean_delta']*100:+.6f} cm，N≥9为{stats['N_ge_9','H_JADE']['mean_delta']*100:+.6f} cm。这是Energy-only组合训练结果，不能把变化独立归因于φ、ρ、S0或条件桥。公共能量下降不等于真实JFDE改善。

| 分组 | 指标 | 窗数 | 均值差 cm | 中位数差 cm | 改善窗 | 退步窗 | 相等窗 |
|---|---|---:|---:|---:|---:|---:|---:|
{distribution}

按窗口等权，严格零记相等；CSV另列|Δ|≤1e−6数量。590窗、501多人窗、127个N≥9窗与旧epoch5逐窗对齐。逐人minFDE与逐窗边缘下界保持不变（逐窗差<1e−6 m），全体分别为0.3520422351797007 / 0.33942601943420153 m。固定终点使桥不能直接移动JFDE终点。

旧epoch5逐窗结果复用固定归档`{REFERENCE}`并核验CSV、权重、split与bank SHA，不重复生成参考推理。归档回放与历史门槛有约1e−10至1e−8 m的浮点差，见REFERENCE_IDENTITY；门槛仍严格使用用户原值。配对分解采用直接端点JFDE，主门槛采用原路径metric累加器的JFDE，两者都在CSV保留。

## SSEC更新与监测

| 模块 | 非零梯度step数 | 非零实际参数更新step数 | 平均梯度L2 |
|---|---:|---:|---:|
{module_table}

上述为原AdamW post hook实测，梯度为裁剪前值；不把梯度范数小当失败。逐轮SSEC接受/拒绝/严格并列/可靠下降/舍入不确定、FP64能量变化、有效评分RMS、matching gap和Sinkhorn熵见[EPOCH_ENERGY_AND_MATCHING.csv](EPOCH_ENERGY_AND_MATCHING.csv)。单人identity旁路零熵会拉低all平均，不能据此声称解决软硬梯度偏差。多人分组熵见人数分组CSV。

全体和N≥9逐轮指标：[EPOCH_GROUPS.csv](EPOCH_GROUPS.csv)；人数区间：[EPOCH_POPULATIONS.csv](EPOCH_POPULATIONS.csv)；模块：[EPOCH_MODULES.csv](EPOCH_MODULES.csv)；分组实测成本：[EPOCH_COSTS.csv](EPOCH_COSTS.csv)。公共能量使用神经能量单位，不能作为米单位误差收益。

## 固定协议与完整资产

正式源码固定`{BASE}`，非零恢复验收`{RECOVERY_BASE}`，冻结配置文件SHA256 `{FROZEN}`。实际只执行原CONTROLLED_COMMAND。SSEC Energy-only无CSEE，原Provider density19/head9及原seed2035 bank冻结，Stage2 seed3201，final_coordinate、六层桥、K20、DDIM20、FP32/AdamW、5轮warmup/100轮调度、168更新/轮、minimum30/patience15/max100均未变。

累计原生训练{sum(h['train_seconds'] for h in history)/3600:.3f}小时，开发验证{sum(h['validation_seconds'] for h in history)/3600:.3f}小时。逐步allocated峰值与分组成本、外部资源采样口径分开记录；不把nvidia-smi采样当作分配器精确峰值。

全部{len(history)}个完整epoch权重、best、last和Top5不同epoch都在训练主机保留。Top5只按H-JADE、严格同分取早轮。完整状态字段、RNG、计数、别名和SHA已复核：[ALL_CHECKPOINTS.csv](ALL_CHECKPOINTS.csv)、[TOP5.json](TOP5.json)、[TRAINING_RECEIPT.json](TRAINING_RECEIPT.json)。历史{len(protected)}个受保护文件哈希未变，旧队列未启动；权重与bank不上传。

代码、配置、数据/Provider/bank身份见[SOURCE_IDENTITY.json](SOURCE_IDENTITY.json)。[匿名逐窗配对差值](RANK1_VS_OLD5_WINDOWS.csv) · [配对汇总](PAIRED_SUMMARY.csv) · [首更新验收](FIRST_UPDATE_VERIFICATION.json) · [首完整轮验收](FIRST_EPOCH_VERIFICATION.json) · [复现](REPRODUCE.md)。

结论只限本次ZARA2单折、单seed组合效果。不声称独立模块因果贡献、跨seed稳健性或跨折泛化。本次训练与分析结束，不启动第二seed、其他折、消融、测试集、旧五折或新架构。
''')


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['interim','final']);a=p.parse_args()
    c=source_check();sanitized_receipts(c)
    (interim if a.mode=='interim' else final)(c)
    write(DOC/'ARTIFACT_SHA256.json',{str(p.relative_to(DOC)):digest(p) for p in sorted(DOC.rglob('*')) if p.is_file() and p.name!='ARTIFACT_SHA256.json'})
    print('SSEC report written:',a.mode,flush=True)


if __name__=='__main__':main()

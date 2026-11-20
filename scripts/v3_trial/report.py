"""Final read-only report for one completed frozen v3 trial, never model selection by JFDE."""
import csv,hashlib,io,json,math,statistics,subprocess,sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
SOURCE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(SOURCE))
import torch
from bcep.contracts import canonical_hash,digest
from bcep_joint.common import read,write,seed_all
from bcep_v3.formal_inputs import FormalInputs,paths
from bcep_v3.formal import freeze,model_hash
from bcep_v3.model import build,NEW_PREFIXES
from bcep_v2.formal_manifest import write_csv
RUN=paths()['run'];OUT=SOURCE/'docs/research/bcep_v3/zara2_trial_20261120'
REFERENCE_COMMIT='76d80b508ad79f52965c15109b58d2d835473fd7'
REFERENCE_DIR='docs/research/bcep_v2/sharedworld_stage2_onerun_20261119/'


def gitblob(commit,name):return subprocess.check_output(['git','show',commit+':'+name],cwd=SOURCE)
def csv_rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))
def jsonlines(path):
    if not path.exists():return []
    lines=path.read_text().splitlines();result=[]
    for i,line in enumerate(lines):
        try:result.append(json.loads(line))
        except json.JSONDecodeError:
            if i!=len(lines)-1:raise
    return result


def paired_rows(new,old):
    new={int(r['index']):r for r in new};old={int(r['index']):r for r in old}
    if set(new)!=set(old) or len(new)!=590:raise ValueError('paired dev populations differ')
    rows=[]
    for idx in sorted(new):
        a,b=new[idx],old[idx];n=int(a['n_agents']);assert n==int(b['n_agents'])
        row=dict(index=idx,n_agents=n)
        for metric in ['H_JADE','G_JFDE','I_JFDE','H_JFDE','feedback_gain','minFDE','oracle']:
            row['new_'+metric]=float(a[metric])
            if metric=='G_JFDE':continue # Old archived CSV did not contain G; same frozen bank guarantees its invariance.
            row['old_'+metric]=float(b[metric]);row['delta_'+metric]=float(a[metric])-float(b[metric])
        assert abs(row['delta_H_JFDE']-(row['delta_I_JFDE']-row['delta_feedback_gain']))<1e-6
        assert abs(row['delta_minFDE'])<1e-6 and abs(row['delta_oracle'])<1e-6
        rows.append(row)
    return rows


def summaries(rows):
    result=[]
    for group,predicate in [('all',lambda r:True),('multi',lambda r:r['n_agents']>1),('N_ge_9',lambda r:r['n_agents']>=9)]:
        rr=[r for r in rows if predicate(r)]
        for metric in ['H_JADE','I_JFDE','H_JFDE','feedback_gain']:
            delta=[r['delta_'+metric] for r in rr];benefit=1 if metric=='feedback_gain' else -1
            result.append(dict(group=group,metric=metric,windows=len(rr),old_mean_m=statistics.mean(r['old_'+metric] for r in rr),
                new_mean_m=statistics.mean(r['new_'+metric] for r in rr),mean_delta_m=statistics.mean(delta),median_delta_m=statistics.median(delta),
                improved_windows=sum(x*benefit>0 for x in delta),regressed_windows=sum(x*benefit<0 for x in delta),equal_windows=sum(x==0 for x in delta),
                abs_delta_le_1e_minus6_windows=sum(abs(x)<=1e-6 for x in delta)))
    return result


def module_summary(rows):
    out={}
    for prefix in NEW_PREFIXES:
        gs=[r['modules'][prefix]['gradient_l2'] for r in rows]
        ds=[r['modules'][prefix]['parameter_delta_l2'] for r in rows]
        assert all(math.isfinite(v) for v in gs+ds)
        out[prefix]=dict(steps=len(rows),gradient_l2_mean=statistics.mean(gs) if gs else 0.,
                         gradient_l2_max=max(gs,default=0.),nonzero_gradient_steps=sum(v>0 for v in gs),
                         parameter_delta_l2_sum=sum(ds),nonzero_parameter_update_steps=sum(v>0 for v in ds))
    return out


def resource_rows(history,events,samples,telemetry):
    by_event={int(e['epoch']):e for e in events};last_step={int(r['update']):r for r in telemetry}
    output=[]
    for h in history:
        epoch=h['epoch'];event=by_event.get(epoch);step=last_step.get(h['update'])
        if event is None or step is None:raise ValueError('missing epoch/resource boundary receipt')
        end=event['unix'];boundary=step['unix'];start=boundary-h['train_seconds']
        # Per-PID sample bins include driver/context/reserved memory. They are NOT allocator peaks.
        rr=[r for r in samples if r.get('pid')==event['pid'] and start<=r['unix']<=end and r.get('gpu_process_mib') is not None]
        train=[r for r in rr if r['unix']<=boundary];validation=[r for r in rr if r['unix']>boundary]
        output.append(dict(epoch=epoch,update=h['update'],train_seconds=h['train_seconds'],validation_seconds=h['validation_seconds'],
            train_gpu_process_sampled_peak_mib=max((r['gpu_process_mib'] for r in train),default=None),
            validation_gpu_process_sampled_peak_mib=max((r['gpu_process_mib'] for r in validation),default=None),
            process_rss_sampled_peak_kib=max((r.get('rss_kib',0) for r in rr),default=0),
            train_samples=len(train),validation_samples=len(validation),sampling_seconds=5,
            phase_boundary='last optimizer post-hook timestamp; epoch end from native stdout; train start estimated from native duration'))
    return output


def main():
    torch.set_num_threads(1)
    status=read(RUN/'STATUS.json')
    assert status['status']=='COMPLETE' and status['stop_reason'] in ['early_stop','max_epochs']
    inputs=FormalInputs();frozen,_=freeze(inputs);assert frozen==read(RUN/'FROZEN_CONFIG.json')
    hist=read(RUN/'joint/EPOCH_METRICS.json');top=read(RUN/'joint/TOP5.json');rank=min(hist,key=lambda r:(r['validation'],r['epoch']))
    assert [r['epoch'] for r in top]==[r['epoch'] for r in sorted(hist,key=lambda r:(r['validation'],r['epoch']))[:5]]
    assert len(hist)==status['completed_epoch'] and status['optimizer_updates']==168*len(hist)
    all_checkpoints=[];rank_state=None
    for row in hist:
        p=RUN/'joint'/row['file'];s=torch.load(p,map_location='cpu',weights_only=False)
        assert s['identity']==frozen['identity'] and s['recipe']==frozen['recipe']
        assert s['completed_epoch']==row['epoch'] and s['update']==168*row['epoch'] and s['cursor']==0 and s['order'] is None
        assert {'model','optimizer','scheduler','rng','history','stale','trainable'}<=set(s)
        assert {'cpu','cuda','python','numpy','order','view'}<=set(s['rng'])
        h=digest(p);all_checkpoints.append(dict(epoch=row['epoch'],update=s['update'],dev_H_JADE=row['validation'],file=row['file'],sha256=h))
        if row['epoch']==rank['epoch']:rank_state=s
        else:del s
    last=torch.load(RUN/'joint/last.pt',map_location='cpu',weights_only=False)
    assert last['status']=='COMPLETE' and last['update']==status['optimizer_updates'] and last['identity']==frozen['identity']
    assert digest(RUN/'joint/best.pt')==next(r['sha256'] for r in all_checkpoints if r['epoch']==rank['epoch'])
    assert digest(RUN/'joint/last.pt')==all_checkpoints[-1]['sha256']
    for r in top:assert r['sha256']==next(a['sha256'] for a in all_checkpoints if a['epoch']==r['epoch'])
    old_blob=gitblob(REFERENCE_COMMIT,REFERENCE_DIR+'OLD5_REFERENCE_WINDOWS.csv')
    old_meta=json.loads(gitblob(REFERENCE_COMMIT,REFERENCE_DIR+'REFERENCE_REPLAY.json'))
    old_frozen=json.loads(gitblob(REFERENCE_COMMIT,REFERENCE_DIR+'FROZEN_CONFIG.json'))
    assert hashlib.sha256(old_blob).hexdigest()==old_meta['csv_sha256']
    assert old_meta['identity']['split_sha256']==frozen['split_sha256']
    assert old_frozen['bank_sha256']==frozen['bank_sha256']
    assert old_meta['identity']['checkpoint_sha256']=='9281673fda5c439bdd463b0bfc9af1a65a3a0fea30941ad3c956cbb617943317'
    old=list(csv.DictReader(io.StringIO(old_blob.decode())))
    new=csv_rows(RUN/'dev'/f"epoch_{rank['epoch']:03d}_windows.csv");paired=paired_rows(new,old);paired_summary=summaries(paired)
    assert len([r for r in paired if r['n_agents']>=9])==127 and len([r for r in paired if r['n_agents']>1])==501
    groups={g['group']:g for g in read(RUN/'dev'/f"epoch_{rank['epoch']:03d}_groups.json")}
    decision_lock=read(OUT/'PREREGISTERED_DECISION.json');limits=[decision_lock['thresholds_m'][k] for k in ['all_H_JADE_lte','all_H_JFDE_lt','N_ge_9_H_JFDE_lt']]
    values=[groups['all']['JADE'],groups['all']['JFDE'],groups['N_ge_9']['JFDE']]
    passes=[values[0]<=limits[0],values[1]<limits[1],values[2]<limits[2]]
    # Telemetry is after successful physical optimizer steps; deduplicate effective updates after recovery.
    physical=jsonlines(RUN/'MODULE_UPDATES.jsonl');effective={int(r['update']):r for r in physical}
    assert set(effective)==set(range(1,status['optimizer_updates']+1))
    telemetry=[effective[i] for i in sorted(effective)]
    attempts=read(RUN/'PROCESS_ATTEMPTS.json');exact_physical=len(attempts)==1 and attempts[0]['exit_code']==0 and len(physical)==status['optimizer_updates']
    costs=resource_rows(hist,jsonlines(RUN/'EPOCH_EVENTS.jsonl'),jsonlines(RUN/'RESOURCE_SAMPLES.jsonl'),telemetry)
    epoch_rows=[];epoch_modules=[]
    for h in hist:
        e=h['epoch'];cost=next(r for r in costs if r['epoch']==e)
        for g in read(RUN/'dev'/f'epoch_{e:03d}_groups.json'):
            epoch_rows.append(dict(epoch=e,update=h['update'],group=g['group'],windows=g['scene_count'],H_JADE=g['JADE'],
                G_JFDE=g['G_JFDE'],I_JFDE=g['I_JFDE'],H_JFDE=g['JFDE'],feedback_gain=g['feedback_gain'],
                marginal_minFDE=g['minFDE'],window_oracle=g['scene_weighted_minFDE'],train_seconds=cost['train_seconds'],validation_seconds=cost['validation_seconds']))
        per=module_summary([r for r in telemetry if (r['update']-1)//168+1==e])
        for prefix,row in per.items():epoch_modules.append(dict(epoch=e,module=prefix,**row))
    seed_all(3201);initial,_=build(frozen['config']);assert model_hash(initial)==frozen['initial_model_sha256']
    init=initial.state_dict();summary=module_summary(telemetry)
    heads={'coupling.csee.':'coupling.csee.out.weight','coupling.hcra.':'coupling.hcra.delta.2.weight','bridge.denoiser.grg_ctf.':'bridge.denoiser.grg_ctf.net.2.weight'}
    for prefix in NEW_PREFIXES:
        difference=[(rank_state['model'][n]-t).double() for n,t in init.items() if n.startswith(prefix)]
        summary[prefix]['rank1_parameter_delta_l2_from_initial']=sum(float(x.square().sum()) for x in difference)**.5
        key=heads[prefix];summary[prefix]['rank1_zero_initialized_output_weight_delta_l2']=float((rank_state['model'][key]-init[key]).double().norm())
    protected=read(RUN/'PROTECTED_ASSETS_BEFORE.json');roots=inputs.roots;protection={}
    for name,info in protected.items():
        label,rel=name.split('/',1);key='sharedworld' if label=='sharedworld' else label
        p=roots[key]/rel;assert p.stat().st_size==info['size'] and digest(p)==info['sha256'],name
        protection[name]=True
    decision=dict(status='本折三项全部通过' if all(passes) else '本折未同时通过三项门槛，保留旧 coordinate 基线',rank1_epoch=rank['epoch'],
        selector='all dev H-JADE; exact ties earlier epoch',old_epoch5=limits,new_rank1=values,new_minus_old=[n-o for n,o in zip(values,limits)],
        passes=passes,all_pass=all(passes),effective_real_optimizer_updates=status['optimizer_updates'],recorded_physical_step_events=len(physical),
        physical_step_total_exact=exact_physical,completed_epochs=len(hist),stop_reason=status['stop_reason'],no_test=True,no_followup=True)
    write(OUT/'DECISION.json',decision);write(OUT/'FROZEN_CONFIG.json',frozen);write(OUT/'STATUS.json',status)
    write(OUT/'TOP5.json',[dict(epoch=r['epoch'],update=r['update'],dev_H_JADE=r['validation'],file=r['file'],sha256=r['sha256']) for r in top])
    write_csv(OUT/'ALL_CHECKPOINTS.csv',all_checkpoints);write_csv(OUT/'EPOCH_GROUPS.csv',epoch_rows);write_csv(OUT/'EPOCH_MODULES.csv',epoch_modules)
    write_csv(OUT/'EPOCH_RESOURCES.csv',costs);write_csv(OUT/'RANK1_VS_OLD5_WINDOWS.csv',paired);write_csv(OUT/'PAIRED_SUMMARY.csv',paired_summary)
    write_csv(OUT/'RANK1_WINDOWS.csv',new);(OUT/'OLD5_REFERENCE_WINDOWS.csv').write_bytes(old_blob)
    write(OUT/'MODULE_SUMMARY.json',summary);write(OUT/'PROTECTION.json',dict(all_unchanged=True,protected_files=len(protection),checks=protection))
    write(OUT/'REFERENCE_IDENTITY.json',dict(commit=REFERENCE_COMMIT,csv_sha256=old_meta['csv_sha256'],checkpoint=old_meta['identity'],
          replay_optimizer_updates=0,additional_old_inference_this_run=0,archive_differences_m=old_meta['archive_differences_m']))
    write(OUT/'RESOURCE_PROTOCOL.json',dict(source='external nvidia-smi per-process used_memory',sample_interval_seconds=5,
          allocator_peak_available=False,meaning='driver/context/reserved process GPU memory sampled every 5 seconds; peak can be missed',
          phase_boundary_approximate=True,training_and_validation_time_source='unmodified native run_stage timers',attempts=attempts))
    (OUT/'.gitattributes').write_text('*.csv -text whitespace=cr-at-eol\n')
    command=(SOURCE/'docs/research/bcep_v3/stage2_ready_20261120/CONTROLLED_COMMAND.txt').read_text().strip()
    (OUT/'CONTROLLED_COMMAND.txt').write_text(command+'\n')
    table='\n'.join(f'| {label} | {o:.16f} | {n:.16f} | {(n-o)*100:+.6f} | {ok} |' for label,o,n,ok in zip(['全体 H-JADE ≤','全体 H-JFDE <','N≥9 H-JFDE <'],limits,values,passes))
    stat={(r['group'],r['metric']):r for r in paired_summary}
    distribution='\n'.join(f"| {r['group']} | {r['metric']} | {r['windows']} | {r['mean_delta_m']*100:+.6f} | {r['median_delta_m']*100:+.6f} | {r['improved_windows']} | {r['regressed_windows']} | {r['equal_windows']} |" for r in paired_summary)
    attribution=[]
    for group in ['all','multi','N_ge_9']:
        a,b,c=[stat[group,k] for k in ['I_JFDE','feedback_gain','H_JFDE']]
        attribution.append(f"- {group}：初配I-JFDE新−旧 {a['mean_delta_m']*100:+.6f} cm；反馈收益I−H新−旧 {b['mean_delta_m']*100:+.6f} cm；最终H-JFDE新−旧 {c['mean_delta_m']*100:+.6f} cm。新模型平均反馈收益为 {b['new_mean_m']*100:+.6f} cm，旧模型为 {b['old_mean_m']*100:+.6f} cm。")
    module_table='\n'.join(f"| {k} | {v['nonzero_gradient_steps']} | {v['nonzero_parameter_update_steps']} | {v['rank1_parameter_delta_l2_from_initial']:.9g} | {v['rank1_zero_initialized_output_weight_delta_l2']:.9g} |" for k,v in summary.items())
    csee=summary['coupling.csee.'];csee_changed=csee['nonzero_gradient_steps']>0 and csee['rank1_zero_initialized_output_weight_delta_l2']>0
    report=f'''# BCEP–EBJD v3：唯一 ZARA2 Stage2 正式训练最终报告

**{decision['status']}。** 原生停止原因 `{status['stop_reason']}`；完整 **{len(hist)}轮**；有效真实optimizer更新 **{status['optimizer_updates']}**；记录到的物理step事件 **{len(physical)}**（物理总数精确可核对：{exact_physical}）。唯一主结果按全体dev H-JADE选为 **epoch {rank['epoch']} / rank1**，没有从Top5改挑JFDE最低权重。

| 固定验收指标 m | 旧coordinate epoch5 | 新v3 rank1 | 新−旧 cm | 通过 |
|---|---:|---:|---:|---|
{table}

三项按用户预先声明的归档未舍入值判断，必须同时通过。旧正式rank1 epoch26仅作背景：H-JADE 0.4084707879413993 m、H-JFDE 0.8016974424154072 m、N≥9 H-JFDE 1.3188941478729248 m。完整v3一次训练只能验证组合效果，不把任何变化直接归因于单个模块。

## 研究问题的明确回答

1. 是否同时通过全部门槛：**{all(passes)}**；每项差值见首页。
2. 初配和终配如何变化：下列均为同窗口新减旧，误差负值表示改善，反馈收益正值表示帮助增加。
{chr(10).join(attribution)}
3. HCRA反馈是否正向：只能直接判断“旧反馈+HCRA增量”的合成反馈。多人组I−H均值为 **{stat['multi','feedback_gain']['new_mean_m']*100:+.6f} cm**，正值说明最终配对平均优于初配，负值说明平均抵消。该数值不能隔离HCRA本身的因果贡献。
4. CSEE是否得到真实参数更新：**{csee_changed}**；实际step梯度/增量和rank1相对初态变化见下表，零初始化输出权重的非零变化排除了仅凭其他权重衰减推断学习。梯度小不作为失败条件。
5. GRG-CTF启用后的组合模型完整路径JADE是否改善：全体新−旧 **{stat['all','H_JADE']['mean_delta_m']*100:+.6f} cm**，N≥9 **{stat['N_ge_9','H_JADE']['mean_delta_m']*100:+.6f} cm**；负值为改善。这是完整v3的结果，不能单独归因于GRG-CTF。
6. 可直接证明的是组合模型指标、配对分解、三模块实际梯度/参数更新、候选边缘保持和工程恢复；CSEE、HCRA、GRG-CTF各自独立效果、协同贡献、跨seed稳健性与跨折泛化均未被本轮证明。没有追加消融或测试集评测。

## 同窗口配对分布

差值Δ=新−旧，I/H/JADE负值是改善，反馈收益正值是改善；按严格零统计相等，另在CSV列出|Δ|≤1e-6数量。all 590窗、multi 501窗、N≥9 127窗；没有使用行人数加权替代场景等权。

| 分组 | 指标 | 窗数 | 均值差 cm | 中位数差 cm | 改善窗 | 退步窗 | 相等窗 |
|---|---|---:|---:|---:|---:|---:|---:|
{distribution}

逐窗检查 ΔH=ΔI−Δ(I−H) 在1e-6 m内成立。与旧epoch5复用完全相同候选bank时，逐人minFDE和逐窗oracle差均小于1e-6 m；全体边缘minFDE仍为0.3520422351797007 m、逐窗等权oracle为0.33942601943420153 m。固定端点下桥不能直接改变JFDE。

旧epoch5匿名逐窗结果从固定归档 `{REFERENCE_COMMIT}` 读取，核验CSV/权重/split/bank身份，复用已归档20步开发集推理，本轮没有再次训练旧模型。旧参考聚合与门槛的浮点差记录在REFERENCE_IDENTITY.json；门槛判断仍使用用户归档值。[匿名逐窗差值](RANK1_VS_OLD5_WINDOWS.csv) · [配对汇总](PAIRED_SUMMARY.csv)

## 模块梯度与参数更新

| 模块 | 非零梯度step数 | 非零参数增量step数 | rank1相对初态参数L2 | 零初始化输出权重变化L2 |
|---|---:|---:|---:|---:|
{module_table}

来自已验收入口的实际AdamW step后hook；梯度是裁剪前范数。各epoch统计见[EPOCH_MODULES.csv](EPOCH_MODULES.csv)，不以某个范数小直接判模型失败。CSEE初始下游零出口造成第一步无梯度是已验收行为。若技术恢复重放更新，逐轮统计按最终有效update取最后事件，物理事件总数单列。

## 固定协议、耗时和保留

正式训练源码固定提交 `2e96b6f35554ceceb5a04fc08b6259dac75e486b`，配置规范化SHA `{canonical_hash(frozen)}`。实际启动只使用已验收CONTROLLED_COMMAND；准备态allow_real_training=false归档保持原样，本次用户授权独立记录于主机AUTHORIZATION/EXECUTION_AUTHORIZATION。

Provider density epoch19、head epoch9冻结；train 2681窗、dev590窗（N≥9 127），candidate seed2035、训练随机初态seed3201。CSEE+HCRA+GRG-CTF全部启用，final_coordinate旧有效损失、6层桥、K20、20步DDIM、FP32、AdamW、5轮warmup/100轮调度、168更新/epoch、minimum30/patience15/max100均未更改。没有重训Provider、改变终点、增加注意力层或调整权重。

原生计时合计：训练 **{sum(h['train_seconds'] for h in hist)/3600:.3f}小时**，逐轮验证 **{sum(h['validation_seconds'] for h in hist)/3600:.3f}小时**。计时含原控制器定义的范围，额外启动/保存开销不伪装成纯GPU时间。逐轮详情：[EPOCH_GROUPS.csv](EPOCH_GROUPS.csv)、[EPOCH_RESOURCES.csv](EPOCH_RESOURCES.csv)。

显存旁路每5秒读取本次PID的nvidia-smi进程used_memory，记录driver/context/reserved口径的采样峰值，同时记录RSS。**固定入口未采集PyTorch allocated峰值；旁路采样可能漏掉瞬时峰值，阶段边界为最后step与原生epoch日志近似划分。** 未修改冻结训练源码来插入计时或额外前向，不能把采样显存称为精确分配器峰值。资源口径和进程尝试记录见RESOURCE_PROTOCOL.json。

保留全部 **{len(hist)}个不同完整epoch**、best、可恢复last及按dev H-JADE排序的Top5 **{[r['epoch'] for r in top]}**；逐个核验完整模型/optimizer/scheduler/RNG/数据游标与选择状态、epoch更新计数和SHA。[ALL_CHECKPOINTS.csv](ALL_CHECKPOINTS.csv) · [TOP5.json](TOP5.json)。权重与候选库保留训练主机。旧v2/candidate_cost/sharedworld共{len(protection)}个保护文件大小与SHA未变，旧队列没有恢复。

## 复现与停止边界

```bash
{command}
```

本次已经COMPLETE，相同命令会返回，不另起训练。技术中断仅从同一运行last恢复，未落盘更新可能重放；若发生恢复，不能只根据checkpoint有效计数推断所有物理step，物理事件记录与精确性已在首页分列。

本轮结束后不启动第二seed、其他折、模块消融、测试集、旧五折、candidate_cost或sharedworld。三项全部通过也只支持本折后的独立审阅决定；否则保留旧coordinate基线。当前结论不推广为跨折性能保证。
'''
    (OUT/'REPORT_ZH.md').write_text(report)
    write(OUT/'ARTIFACT_SHA256.json',{p.name:digest(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='ARTIFACT_SHA256.json'})
    print('FINAL_REPORT_READY',decision,flush=True)
if __name__=='__main__':main()

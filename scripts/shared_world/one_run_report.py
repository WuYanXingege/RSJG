"""Finalize only a completed single-fold run; fill missing old-dev JADE once."""
import argparse,contextlib,csv,fcntl,hashlib,json,math,statistics,subprocess,sys,time,shutil
from pathlib import Path
import torch
import one_run as trial
from bcep_joint.common import read,write,seed_all
from bcep.contracts import digest,canonical_hash
from bcep_formal.data import FormalData
from bcep_v2.bank import Bank
from bcep_v2.joint import build_joint
from bcep_v2.formal_manifest import write_csv
from bcep_v2.runtime import configure_cuda
from ebjd.metrics import MetricAccumulator

def csvread(p):
    with p.open() as f:return list(csv.DictReader(f))

def old_reference(up,run,config,split):
    cp=up/'zara2/joint/epoch_005.pt';sha=digest(cp)
    assert sha=='9281673fda5c439bdd463b0bfc9af1a65a3a0fea30941ad3c956cbb617943317'
    dest=run/'reference';identity=dict(checkpoint_sha256=sha,split_sha256=canonical_hash(split),candidate_seed=2035,steps=20,K=20)
    if (dest/'COMPLETE.json').exists():
        done=read(dest/'COMPLETE.json');assert done['identity']==identity and digest(dest/'epoch_005_windows.csv')==done['csv_sha256'];return csvread(dest/'epoch_005_windows.csv'),done
    bankpath=up/'zara2/refined_banks/selection_dev_2035';bank=Bank(bankpath,read(bankpath/'IDENTITY.json'),split)
    assert bank.hash==read(run/'FROZEN_CONFIG.json')['identity']['dev_bank']
    seed_all(3201);model=build_joint(dict(config,pair_objective='final_coordinate')).cuda().eval().requires_grad_(False)
    ck=torch.load(cp,map_location='cpu',weights_only=False);model.load_state_dict(ck['model'],strict=True);del ck
    data=FormalData(split,'scoring',verify=False);rows=[];acc={k:MetricAccumulator() for k in ['all','N_ge_9']};start=time.monotonic()
    for pos,i in enumerate(split['selection_dev_indices']):
        b=data.batch([i],'selection_dev');r,s=trial.evaluate_window(model,b,bank.batch([i],b));rows.append(dict(epoch=5,index=i,**r));acc['all'].update_statistics(s)
        if r['n_agents']>=9:acc['N_ge_9'].update_statistics(s)
        if (pos+1)%100==0:print('old epoch5 missing per-window JADE',pos+1,'/590',flush=True)
    metrics={k:a.as_dict() for k,a in acc.items()};ref=read(run/'FROZEN_CONFIG.json')['decision']['epoch5_reference']
    difference=[metrics['all']['JADE']-ref[0],metrics['all']['JFDE']-ref[1],metrics['N_ge_9']['JFDE']-ref[2]]
    assert max(map(abs,difference))<1e-6
    write_csv(dest/'epoch_005_windows.csv',rows)
    done=dict(identity=identity,csv_sha256=digest(dest/'epoch_005_windows.csv'),groups=metrics,seconds=time.monotonic()-start,archive_differences_m=difference,real_optimizer_updates=0,only_missing_field='per-window old H-JADE; endpoint metrics collected from the same required forward')
    write(dest/'COMPLETE.json',done);return rows,done

def summarize(run,up,oldrows,reference,out):
    frozen=read(run/'FROZEN_CONFIG.json');status=read(run/'STATUS.json');history=read(run/'joint/EPOCH_METRICS.json');top=read(run/'joint/TOP5.json')
    assert status['status']=='COMPLETE' and status['stop_reason'] in ['early_stop','max_epochs']
    assert status['optimizer_updates']==168*status['completed_epoch'] and [e['epoch'] for e in history]==list(range(1,status['completed_epoch']+1))
    selected=min(history,key=lambda e:(e['validation'],e['epoch']));assert selected['epoch']==top[0]['epoch'] and selected['validation']==top[0]['validation']
    ck=torch.load(run/'joint/last.pt',map_location='cpu',weights_only=False)
    assert ck['status']=='COMPLETE' and ck['identity']==frozen['identity'] and ck['update']==status['optimizer_updates']
    for k in ['model','optimizer','scheduler','rng','order','cursor','stale','epoch_raw_sums','epoch_counts']:assert k in ck
    assert ck['cursor']==0;del ck
    complete=[]
    for e in history:
        path=run/'joint'/e['file'];assert path.exists()
        complete.append(dict(epoch=e['epoch'],update=e['update'],dev_H_JADE=e['validation'],file='joint/'+e['file'],sha256=digest(path)))
    top_rows=[next(e for e in complete if e['epoch']==t['epoch']) for t in top]
    assert len({e['epoch'] for e in top_rows})==min(5,len(history))
    for r,t in zip(top_rows,top):assert r['sha256']==t['sha256']
    assert digest(run/'joint/best.pt')==top_rows[0]['sha256']
    groups=[];epochs=[];costs=[]
    for e in history:
        groups.extend(read(run/'dev'/f"epoch_{e['epoch']:03d}_groups.json"));costs.append(read(run/'dev'/f"epoch_{e['epoch']:03d}_cost.json"))
        row=dict(epoch=e['epoch'],update=e['update'],dev_H_JADE=e['validation'],train_seconds=e['train_seconds'],validation_seconds=e['validation_seconds'])
        for k,v in e.items():
            if k.startswith('train_') and isinstance(v,(int,float)):row[k]=v
        epochs.append(row)
    flat=[]
    for g in groups:
        flat.append({k:v for k,v in g.items() if not isinstance(v,dict)})
    rank=selected['epoch'];rankgroups={g['group']:g for g in groups if g['epoch']==rank}
    newrows=csvread(run/'dev'/f'epoch_{rank:03d}_windows.csv');old={int(r['index']):r for r in oldrows};assert len(old)==len(newrows)==590
    paired=[]
    for nr in newrows:
        i=int(nr['index']);o=old[i];assert int(nr['n_agents'])==int(o['n_agents'])
        r=dict(index=i,n_agents=int(nr['n_agents']),new_epoch=rank,old_epoch=5)
        for k in ['I_JFDE','H_JFDE','H_JADE','feedback_gain','minFDE','oracle']:
            r['old_'+k]=float(o[k]);r['new_'+k]=float(nr[k]);r['delta_'+k]=float(nr[k])-float(o[k])
        assert abs(r['delta_H_JFDE']-(r['delta_I_JFDE']-r['delta_feedback_gain']))<1e-6
        assert abs(r['delta_minFDE'])<1e-6 and abs(r['delta_oracle'])<1e-6
        paired.append(r)
    summaries=[]
    for name in ['all','N_ge_9']:
        rr=[r for r in paired if name=='all' or r['n_agents']>=9];assert len(rr)==(590 if name=='all' else 127)
        for metric in ['I_JFDE','H_JFDE','H_JADE','feedback_gain']:
            values=[r['delta_'+metric] for r in rr]
            summaries.append(dict(group=name,metric=metric,windows=len(rr),old_mean_m=statistics.mean(r['old_'+metric] for r in rr),new_mean_m=statistics.mean(r['new_'+metric] for r in rr),delta_mean_m=statistics.mean(values),delta_mean_cm=100*statistics.mean(values),delta_median_m=statistics.median(values),delta_negative_windows=sum(x<0 for x in values),delta_positive_windows=sum(x>0 for x in values),delta_zero_windows=sum(x==0 for x in values),abs_delta_le_1e6_windows=sum(abs(x)<=1e-6 for x in values)))
    ref=frozen['decision']['epoch5_reference'];new=[rankgroups['all']['JADE'],rankgroups['all']['JFDE'],rankgroups['N_ge_9']['JFDE']]
    checks=[new[0]<=ref[0],new[1]<ref[1],new[2]<ref[2]]
    decision=dict(status='本折通过' if all(checks) else '本折未通过，保留旧 coordinate 基线',rank1_epoch=rank,old_epoch5=ref,new_rank1=new,new_minus_old=[n-o for n,o in zip(new,ref)],passes=checks,all_pass=all(checks),selector='dev H-JADE; exact ties earlier epoch',real_optimizer_updates=status['optimizer_updates'],completed_epochs=status['completed_epoch'],stop_reason=status['stop_reason'],no_test=True,no_followup_queue=True)
    preservation={}
    for k,h in read(run/'PROTECTED_BEFORE.json').items():
        label,relative=k.split('/',1);root=up if label=='formal' else up.parent/'zara2_costproxy_stage2_20261119';preservation[k]=digest(root/relative)==h
    assert all(preservation.values())
    out.mkdir(parents=True,exist_ok=True)
    for name in ['FROZEN_CONFIG.json','SMOKE.json','FIRST_UPDATE_VERIFICATION.json','PREPARED.json']:shutil.copy2(run/name,out/name)
    write(out/'STATUS.json',status);write(out/'DECISION.json',decision);write(out/'PRESERVATION.json',preservation);write(out/'TOP5.json',top_rows);write(out/'REFERENCE_REPLAY.json',reference)
    write_csv(out/'ALL_CHECKPOINTS.csv',complete);write_csv(out/'EPOCH_GROUPS.csv',flat);write_csv(out/'EPOCH_TRAIN.csv',epochs);write_csv(out/'RANK1_WINDOWS.csv',newrows);write_csv(out/'RANK1_VS_OLD5_WINDOWS.csv',paired);write_csv(out/'PAIRED_SUMMARY.csv',summaries)
    write_csv(out/'OLD5_REFERENCE_WINDOWS.csv',oldrows)
    total_train=sum(e['train_seconds'] for e in history);total_validation=sum(e['validation_seconds'] for e in history)
    resource=dict(train_seconds=total_train,validation_seconds=total_validation,old_reference_replay_seconds=reference['seconds'],train_peak_bytes=max(c['train_peak_bytes'] for c in costs),validation_peak_bytes=max(c['validation_peak_bytes'] for c in costs),complete_checkpoint_count=len(complete))
    write(out/'RESOURCES.json',resource)
    report=[f'''# ZARA2 shared_world_bounded：一次 Stage 2 受控训练

**{decision['status']}。** 停止原因 `{status['stop_reason']}`，完整 **{status['completed_epoch']} 轮**，有效真实更新 **{status['optimizer_updates']}**，唯一主结果为按 dev H-JADE 选出的 **epoch {rank} / rank1**。训练 {total_train/3600:.2f} 小时、逐轮验证 {total_validation/3600:.2f} 小时；原五折与 candidate_cost 保持暂停。没有测试集选模、第二个训练 seed、额外折或后续队列。

| 预先固定指标（m） | 旧 epoch 5 | 新 rank1 | 新−旧（m） | 新−旧（cm） | 通过 |
|---|---:|---:|---:|---:|---|''']
    for name,o,n,ok in zip(['全体 H-JADE ≤','全体 H-JFDE <','N≥9 H-JFDE <'],ref,new,checks):report.append(f'| {name} | {o:.16f} | {n:.16f} | {n-o:+.9f} | {(n-o)*100:+.6f} | {ok} |')
    report.append('''
三个门槛均来自旧 epoch 5 的归档未舍入数值；必须同时通过。微小差值只能作为本折观测，不能称稳定提升。Top5 只按 dev H-JADE 排序，没有改取 JFDE 最低权重。''')
    for name in ['all','N_ge_9']:
        ss={r['metric']:r for r in summaries if r['group']==name};di=ss['I_JFDE']['delta_mean_m'];dg=ss['feedback_gain']['delta_mean_m'];dh=ss['H_JFDE']['delta_mean_m'];da=ss['H_JADE']['delta_mean_m']
        report.append(f"\n{name}：初配 I-JFDE 新−旧 **{di*100:+.4f} cm**；邻居反馈收益变化 **{dg*100:+.4f} cm**（正为反馈更有帮助），最终 H-JFDE 新−旧 **{dh*100:+.4f} cm**。分解为 ΔH=ΔI−Δ反馈收益。完整轨迹 H-JADE 新−旧 **{da*100:+.4f} cm**（负为改善）。该分解描述已训练模型间的观测，不把损失标量系数当作参数梯度比例。")
    report.append('''
## 匿名逐窗配对分布

旧 epoch 5 缺少逐窗完整 H-JADE，故只读补算一次相同开发集的 20 步 DDIM，并从同一前向提取 I/H。没有补训旧模型；旧参考与归档的数值误差记录于 REFERENCE_REPLAY.json。Δ=新−旧；I/H/JADE 的负差为改善，反馈收益的正差才是改善。计数按严格零划分，另列 |Δ|≤1e-6 的数量作为浮点近零提示。

| 分组 | 指标 | 窗数 | 平均差 cm | 中位差 cm | Δ<0 | Δ>0 | Δ=0 | 近零 |
|---|---|---:|---:|---:|---:|---:|---:|---:|''')
    for r in summaries:report.append(f"| {r['group']} | {r['metric']} | {r['windows']} | {r['delta_mean_cm']:+.5f} | {r['delta_median_m']*100:+.5f} | {r['delta_negative_windows']} | {r['delta_positive_windows']} | {r['delta_zero_windows']} | {r['abs_delta_le_1e6_windows']} |")
    report.append(f'''
逐窗结果：[RANK1_VS_OLD5_WINDOWS.csv](RANK1_VS_OLD5_WINDOWS.csv)；逐轮分组：[EPOCH_GROUPS.csv](EPOCH_GROUPS.csv)；训练目标与系数日志：[EPOCH_TRAIN.csv](EPOCH_TRAIN.csv)。固定候选下逐窗 minFDE/窗口 oracle 新旧差均小于 1e-6 m，每轮全体 agent-micro minFDE 核对为 0.3520422351797007 m、窗口等权 oracle 为 0.33942601943420153 m（容差 1e-6）。

## 历史参照与选择边界

| 模型 | dev H-JADE | dev H-JFDE | dev N≥9 H-JFDE |
|---|---:|---:|---:|
| 旧 coordinate 正式 rank1 / epoch 26 | 0.4084707879413993 | 0.8016974424154072 | 1.3188941478729248 |
| 旧 coordinate epoch 5 | 0.4097670185111337 | 0.7841791655426308 | 1.296843430188697 |
| 旧 coordinate epoch 3，仅上下文 | 0.4218764735164784 | 0.7792569560204017 | 1.2382272847994107 |
| 新 shared_world_bounded / epoch {rank} | {new[0]:.16f} | {new[1]:.16f} | {new[2]:.16f} |

这些是开发集结果，未使用历史 test 数值作门槛，也没有本轮测试集评测。一次训练仅检验该目标相对旧基线的整体效果，不声称动态系数优于未实测的固定混合。

## 公平对照、身份与保留

报告基点 `7cea557556386d9fbc6e905e02cf3b24640a8d03`，已验收实现 `f17571c6836287408514736855e156fd5b825828`，运行代码 `{frozen['code_commit']}`。原实现文件逐项哈希匹配审计；只增加单折入口，模型/损失未再修改。配置 SHA256 `{frozen['config_sha256']}`。审计档案保持 RUN_ONE_ZARA2=false；本运行独立授权 true，上限 16800。

Provider/候选修正头沿用原 dev rank1，seed 2035 的 train/dev bank、成员与顺序哈希通过；仅 ZARA2 train 2681、dev 590（其中 N≥9 127）窗。训练 seed **3201**，候选/验证 seed **2035**。初态参数 hash `{frozen['initial_tensor_sha256']}` 与旧 coordinate 完全相同，无旧权重微调。首个真实 step 的损失与梯度范数和旧首步完全相等，优化器计数及参数变化见 [FIRST_UPDATE_VERIFICATION.json](FIRST_UPDATE_VERIFICATION.json)。

沿用原 joint_callbacks/run_stage：共享编码、配对头及六层桥；FP32、math SDP、原增广和数据顺序；微批 4、累积 4、尾批原除数，AdamW lr 1e-4、weight decay 1e-4、clip 5；5 轮 warmup、100 轮调度、minimum 30、patience 15、最大 100 轮。τ=0.10 m、a≤0.5、pair=1，其他损失原值。验证只跑一次原 20 步 DDIM，同时捕获初配端点，不额外跑一遍配对或轨迹。

完整保留 {len(complete)} 个不同完成轮次、best 与可恢复 last；[TOP5.json](TOP5.json) 按 dev H-JADE、同分早轮排序，文件与 SHA256 可核验。Top5 轮次为 {[x['epoch'] for x in top_rows]}。完整状态检查包括 optimizer、scheduler、RNG、数据游标、早停与统计分子分母。全部 checkpoint/候选库留在主机；公开包仅相对文件名、哈希和匿名统计。新入口冒烟 0 次真实/合成优化器更新；沿用上一轮 15 项验收。原五折 update 7097、candidate_cost update 1903 的 last、状态、Top5 哈希保持不变。[PRESERVATION.json](PRESERVATION.json)

训练峰值 allocated {resource['train_peak_bytes']/2**30:.3f} GiB，验证峰值 {resource['validation_peak_bytes']/2**30:.3f} GiB。旧参考补算 {reference['seconds']:.2f} s，不计作训练更新。原生结束后退出；通过则仅支持本折后的独立验证决策，未通过则保留旧 coordinate 基线，本程序不启动任何后续任务。

## 复现

在本次代码 checkout 与原主机环境中设置 WORKSPACE 为资产所在工作区。入口路径限定为本实验根目录；同一身份重复 train 调用只恢复本次 last.pt，已 COMPLETE 则直接返回，不会另起训练。先确认数据与 RUN_ONE_ZARA2=true 授权仍有效。

```bash
export OMP_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python scripts/shared_world/one_run.py prepare --workspace "$WORKSPACE"
python scripts/shared_world/one_run.py train --workspace "$WORKSPACE"
# 已启动/中断时仅执行上一条 train；不要对已启动运行再 prepare。
python scripts/shared_world/one_run_report.py --workspace "$WORKSPACE"
```

原五折和 candidate_cost 必须保持暂停。入口同时持有本运行锁与两条旧运行的控制锁，结束后释放；运行身份不符会拒绝恢复。训练源码与报告提交身份分开记录，最终发布以固定 GitHub SHA 为准。
''')
    (out/'REPORT_ZH.md').write_text('\n'.join(report))
    write(out/'ARTIFACT_SHA256.json',{p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='ARTIFACT_SHA256.json'})
    print(json.dumps(decision,ensure_ascii=False),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);a=p.parse_args();configure_cuda();torch.set_num_threads(1)
    up,cost,run=trial.paths(a.workspace)
    assert read(run/'STATUS.json')['status']=='COMPLETE','never finalize an unfinished trial'
    with contextlib.ExitStack() as stack:
        for path in [run/'run.lock',up/'controller.lock',cost/'trial.lock']:
            f=stack.enter_context(path.open('r'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        _,config,_,split,_,_=trial.inputs(up,cost)
        rows,reference=old_reference(up,run,config,split)
        out=trial.SOURCE/'docs/research/bcep_v2/sharedworld_stage2_onerun_20261119'
        summarize(run,up,rows,reference,out)
if __name__=='__main__':main()

#!/usr/bin/env python3
"""One bounded goal-objective comparison, with full epoch recovery and reports.

Control and candidate use identical architecture/state/RNG, three epochs each,
then five fixed inference seeds. No automatic long training or model selection.
"""
from __future__ import annotations
import argparse
import copy
import csv
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import signal
import shutil
import subprocess
import sys
import time

PAPER=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PAPER),str(PAPER/'scripts')]
import torch
from ebjd.config import load_config
from ebjd.goal_coverage import LEGACY, SEPARATED
from ebjd.proposals import file_sha256
from train_learned_goal_bridge import (build_scratch, optimizer_for, seed_all, restore_rng,
    capture_rng, perform_update, set_epoch_lr, evaluate, loader, atomic_json)
from check_spatial_goal_pair import digest_rng

INVOCATION_START=0.
PREVIOUS_COST=0.


def heartbeat(root):
    atomic_json(root/'GPU_COST.json',dict(status='RUNNING',gpu_seconds=PREVIOUS_COST+time.perf_counter()-INVOCATION_START,
        updated_unix=time.time()))


def arm_contract(base, arm):
    c=copy.deepcopy(base)
    c['goal_training_objective']=SEPARATED if arm=='candidate' else LEGACY
    if arm=='candidate':c.update(coverage_weight=1.,coverage_temperature=.1)
    return c


def make_model(payload, config, contract, device):
    seed_all(config['seed']);model=build_scratch(config,contract).to(device)
    model.load_state_dict(payload['model'],strict=True)
    opt=optimizer_for(model,config);opt.load_state_dict(copy.deepcopy(payload['optimizer']))
    gen=torch.Generator();restore_rng(payload['rng'],gen)
    return model,opt,gen


def fingerprint(batches, gen):
    h=hashlib.sha256()
    for b in batches:
        for t in [b.observed,b.future,b.valid]:h.update(t.numpy().tobytes())
    return dict(input_sha256=h.hexdigest(),scene_ids=[s for b in batches for s in b.scene_ids],rng_before=digest_rng(gen))


def compare_records(control,candidate):
    assert len(control)==len(candidate)
    for a,b in zip(control,candidate):
        for key in ['input_sha256','scene_ids','rng_before','epoch','update','lr']:
            if a[key]!=b[key]:raise RuntimeError(f'paired identity mismatch {key}, update={a["update"]}')


def finite_state(model,opt):
    for name,value in model.state_dict().items():
        if value.is_floating_point() and not torch.isfinite(value).all():raise FloatingPointError(name)
    for state in opt.state.values():
        for name,value in state.items():
            if isinstance(value,torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
                raise FloatingPointError('optimizer '+name)


def train_epoch(root,model,opt,gen,config,contract,arm,epoch,update,*,limit=None):
    device=torch.device('cuda');model.train();lr=set_epoch_lr(opt,epoch,config)
    records=[];pending=[];torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
    phase='gate' if limit else 'train'
    attempt=root/'logs'/f'{phase}_{arm}_epoch{epoch}_{time.time_ns()}.jsonl'
    data=loader(config,'train',gen)
    reference=None
    if arm=='candidate' and limit is None:
        reference=json.loads((root/'control'/f'epoch_{epoch:03d}.json').read_text())['records']
    def step(batches):
        nonlocal update
        ident=fingerprint(batches,gen)
        row=perform_update(model,opt,batches,config,contract,gen,device);update+=1
        row.update(**ident,epoch=epoch,update=update,lr=lr)
        if reference is not None:compare_records([reference[len(records)]],[row])
        records.append(row)
        with attempt.open('a') as f:f.write(json.dumps(row)+'\n')
        if len(records)%16==0:
            atomic_json(root/'STATUS.json',dict(status='RUNNING',phase=phase,arm=arm,epoch=epoch,
                update=update,completed_updates=len(records),loss=row['loss'],log_path=str(attempt),time_unix=time.time()))
            heartbeat(root)
    for cpu in data:
        pending.append(cpu)
        if len(pending)<config['training']['gradient_accumulation']:continue
        step(pending);pending=[]
        if limit and len(records)>=limit:break
    if pending and not limit:step(pending)
    del data,pending
    torch.cuda.synchronize();seconds=time.perf_counter()-started
    finite_state(model,opt)
    if not any(r['coupling_grad_norm']>0 for r in records):raise RuntimeError('coupling has no gradient across training window')
    if reference is not None:compare_records(reference,records)
    numeric=['loss','goal_nll','pair_loss','diffusion_loss','map_loss','geometry_loss',
        'goal_grad_norm','coupling_grad_norm','bridge_grad_norm','map_grad_norm']
    if arm=='candidate':numeric.append('goal_coverage_loss')
    return dict(arm=arm,epoch=epoch,update=update,updates=len(records),train_seconds=seconds,
        seconds_per_update=seconds/len(records),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
        training={k:statistics.fmean(r[k] for r in records) for k in numeric},records=records,rng_end=digest_rng(gen))


def write_checkpoint(path,model,opt,gen,config,identity,result):
    payload=dict(schema='ebjd-goal-coverage-pair-v1',model=model.state_dict(),optimizer=opt.state_dict(),
        epoch=result['epoch'],update_index=result['update'],rng=capture_rng(gen),run_identity=identity,
        selection_state={'rule':'fixed_parent_epoch_plus3','selected_epoch':47,'no_reselection':True},
        scheduler_state=dict(type='epoch_warmup_cosine',completed_epoch=result['epoch'],total_epochs=100,
            warmup_epochs=5,final_fraction=.1,group_lrs=[g['lr'] for g in opt.param_groups]),
        early_stop_state=dict(enabled=False,bounded_end_epoch=47),epoch_result=result)
    temporary=path.with_suffix('.tmp');torch.save(payload,temporary);os.replace(temporary,path)


def run_arm(root,parent,config,base,arm,identity):
    contract=arm_contract(base,arm);atomic_json(root/arm/'CONTRACT.json',contract)
    checkpoint=root/arm/'last.pt';payload=parent
    expected=dict(**identity,arm=arm,goal_training_objective=contract['goal_training_objective'])
    if checkpoint.exists():
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if payload['run_identity']!=expected:raise RuntimeError('resume identity mismatch')
        restored_result=copy.deepcopy(payload['epoch_result'])
        restored_result.update(checkpoint=str(checkpoint),checkpoint_sha256=file_sha256(checkpoint))
        result_path=root/arm/f'epoch_{payload["epoch"]:03d}.json'
        if not result_path.exists():
            restored_result['full_epoch_seconds']=None
            restored_result['recovered_from_checkpoint']=True
            atomic_json(result_path,restored_result)
    model,opt,gen=make_model(payload,config,contract,torch.device('cuda'))
    start_epoch=payload['epoch']+1;update=payload['update_index']
    if checkpoint.exists():
        logged_update=update
        for path in (root/'logs').glob(f'train_{arm}_epoch*.jsonl'):
            lines=path.read_text().splitlines()
            if lines:
                try:logged_update=max(logged_update,int(json.loads(lines[-1])['update']))
                except (ValueError,KeyError):pass
        with (root/'RESUME_HISTORY.jsonl').open('a') as f:
            f.write(json.dumps(dict(arm=arm,complete_epoch=start_epoch-1,update=update,
                unsaved_updates_to_replay=logged_update-update,time_unix=time.time()))+'\n')
    if payload is not parent:del payload
    for epoch in range(start_epoch,48):
        started=time.perf_counter()
        result=train_epoch(root,model,opt,gen,config,contract,arm,epoch,update);update=result['update']
        atomic_json(root/'STATUS.json',dict(status='RUNNING',phase='epoch_validation',arm=arm,epoch=epoch,
            update=update,time_unix=time.time()))
        rng=capture_rng(gen)
        try:validation=evaluate(model,config,torch.device('cuda'),2035,endpoint_diagnostics=True)
        finally:restore_rng(rng,gen)
        result['validation']=validation
        result['full_epoch_seconds_before_save']=time.perf_counter()-started
        write_checkpoint(checkpoint,model,opt,gen,config,expected,result)
        result.update(checkpoint=str(checkpoint),checkpoint_sha256=file_sha256(checkpoint),
            full_epoch_seconds=time.perf_counter()-started)
        atomic_json(root/arm/f'epoch_{epoch:03d}.json',result)
        heartbeat(root)
        metrics=validation['H']
        print(f'{arm} Epoch {epoch:03d}/047 | loss={result["training"]["loss"]:.6f} '
            f'| minADE={metrics["minADE"]:.6f} minFDE={metrics["minFDE"]:.6f} '
            f'JADE={metrics["JADE"]:.6f} JFDE={metrics["JFDE"]:.6f} '
            f'| full_epoch={result["full_epoch_seconds"]/60:.2f}min',flush=True)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer']);finite_state(model,opt)
    del model,opt,gen,saved;gc.collect();torch.cuda.empty_cache()


def final_evaluations(root,parent,config,base):
    all_results={}
    for arm in ['parent','control','candidate']:
        contract=arm_contract(base,arm)
        path=Path(json.loads((root/'identity/PARENT.json').read_text())['parent_checkpoint']) if arm=='parent' else root/arm/'last.pt'
        payload=parent if arm=='parent' else torch.load(path,map_location='cpu',weights_only=False)
        model,opt,gen=make_model(payload,config,contract,torch.device('cuda'))
        digest=file_sha256(path);results=[]
        for seed in range(2035,2040):
            destination=root/'evaluation'/f'{arm}_{seed}.json'
            if destination.exists():
                receipt=json.loads(destination.read_text())
                if receipt['checkpoint_sha256']!=digest:raise RuntimeError('evaluation checkpoint mismatch')
                result=receipt['result']
            elif arm!='parent' and seed==2035:
                receipt=json.loads((root/arm/'epoch_047.json').read_text())
                if receipt['checkpoint_sha256']!=digest:raise RuntimeError('fixed E47 identity mismatch')
                result=receipt['validation']
                atomic_json(destination,dict(checkpoint_sha256=digest,result=result,reused_epoch_validation=True))
            else:
                atomic_json(root/'STATUS.json',dict(status='RUNNING',phase='final_evaluation',arm=arm,seed=seed,time_unix=time.time()))
                result=evaluate(model,config,torch.device('cuda'),seed,endpoint_diagnostics=True)
                atomic_json(destination,dict(checkpoint_sha256=digest,result=result))
                heartbeat(root)
                print(f'evaluation {arm} seed={seed} H={result["H"]["minFDE"]:.6f}',flush=True)
            results.append(result)
        all_results[arm]=dict(checkpoint=str(path),checkpoint_sha256=digest,results=results)
        del model,opt,gen
        if arm!='parent':del payload
        gc.collect();torch.cuda.empty_cache()
    return all_results


def finish_report(root,runs):
    keys=['minADE','minFDE','JADE','JFDE'];stats={};rows=[]
    for arm,run in runs.items():
        stats[arm]={}
        for mode in ['G','H']:
            stats[arm][mode]={k:dict(mean=statistics.fmean(r[mode][k] for r in run['results']),
                sd=statistics.stdev(r[mode][k] for r in run['results'])) for k in keys}
        stats[arm]['coverage']={k:statistics.fmean(r['endpoint_diagnostics'][k] for r in run['results'])
            for k in ['recall_025','recall_05','recall_10','P50','P90','P95','goal_nll']}
        for r in run['results']:
            for mode in ['G','H']:rows.append(dict(arm=arm,seed=r['seed'],mode=mode,**{k:r[mode][k] for k in keys},
                **{k:v for k,v in r['endpoint_diagnostics'].items() if k not in ['unit','agent_best_errors']}))
    c=stats['candidate'];a=stats['control'];p=stats['parent']
    delta={ref:{k:c['H'][k]['mean']-stats[ref]['H'][k]['mean'] for k in keys} for ref in ['control','parent']}
    wins=sum(x['H']['minFDE']<y['H']['minFDE'] for x,y in zip(runs['candidate']['results'],runs['control']['results']))
    checks=dict(minFDE_gain=delta['control']['minFDE']<=-.01,
        other_metrics_control=all(delta['control'][k]<=0 for k in ['minADE','JADE','JFDE']),
        recall_nonworse=c['coverage']['recall_05']>=a['coverage']['recall_05'],
        seed_consistency=wins>=4,parent_minFDE=delta['parent']['minFDE']<0,
        other_metrics_parent=all(delta['parent'][k]<=0 for k in ['minADE','JADE','JFDE']))
    summary=dict(status='COMPLETE',decision='PROMISING_SHORT_SCREEN' if all(checks.values()) else 'DO_NOT_ADOPT',
        checks=checks,stats=stats,delta_H=delta,minFDE_improved_seeds=wins,
        identities={k:{n:v for n,v in r.items() if n!='results'} for k,r in runs.items()},
        scope='one training-seed paired continuation E44 to E47; five inference seeds, not independent trainings',
        old_long_training_restarted=False)
    atomic_json(root/'reports/SUMMARY.json',summary)
    with (root/'reports/PER_SEED_METRICS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    table='\n'.join('| '+arm+' | '+' | '.join(f'{stats[arm]["H"][k]["mean"]:.6f} ± {stats[arm]["H"][k]["sd"]:.6f}' for k in keys)+' |' for arm in runs)
    (root/'reports/REPORT_ZH.md').write_text(f'''# ETH逐人覆盖与联合配对分工：固定三轮对照

结论：**{summary['decision']}**。候选相对Control的minFDE改善seed数：{wins}/5。

同一epoch44完整状态，旧头与8层桥不变；两臂固定续训epoch45–47，原100轮LR时间轴。候选仅增加权重1/温度0.1的逐人soft-min候选覆盖，并阻断配对loss经候选坐标到终点头的直接梯度；共享编码器仍接收多个目标。保留NLL，K=20，20步DDIM，rollout/projection OFF。候选total loss多了一项，不能与Control的total loss直接比较大小。

指标为完整ETH validation（与test成员镜像），米，均值±样本SD（ddof=1）。这是一个训练seed下五个推理seed，不是五次独立训练；也不是新方案从零训练成绩。固定比较epoch47，不重新选模。

| H完整轨迹 | minADE | minFDE | JADE | JFDE |
|---|---:|---:|---:|---:|
{table}

逐项筛选：`{json.dumps(checks)}`。
候选−Control：`{json.dumps(delta['control'])}`；候选−Parent：`{json.dumps(delta['parent'])}`。负数为改善。
Recall、误差分位数、NLL及G/H逐seed数值见SUMMARY.json与PER_SEED_METRICS.csv。不能从单次短对照宣称统计显著或跨折泛化。

各轮耗时、峰值显存、配对RNG/输入指纹在control与candidate的epoch_045至047.json；完整模型、Adam、LR元数据、RNG在各自last.pt；父状态引用在identity/PARENT.json；实际GPU相关任务墙钟在GPU_COST.json。每轮更新257次，两臂均到epoch47/update12079，完整保存并strict恢复检查。发生异常则后续阶段不会执行。

本轮结束停止，不自动恢复旧100轮，不追加结构、权重或seed扫描。即使短筛通过，也仅表示值得进一步验证，不自动成为正式结果。
''')
    return summary


def main():
    global INVOCATION_START,PREVIOUS_COST
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--source-commit',required=True)
    args=parser.parse_args();root=Path(args.root)
    if (root/'reports/SUMMARY.json').exists():
        print('Already complete; no duplicate training.',flush=True);return
    head=subprocess.check_output(['git','-C',str(PAPER),'rev-parse','HEAD'],text=True).strip()
    if head!=args.source_commit:raise RuntimeError('frozen source HEAD mismatch')
    subprocess.run(['git','-C',str(PAPER),'diff','--exit-code','HEAD','--','ebjd','scripts','configs'],check=True)
    lock_path=root.parents[1]/'locks/ebjd_gpu0_training.lock';lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('GPU occupied')
        old_cost=json.loads((root/'GPU_COST.json').read_text()) if (root/'GPU_COST.json').exists() else {}
        previous=old_cost.get('gpu_seconds',0);started=time.perf_counter();failure=None
        INVOCATION_START=started;PREVIOUS_COST=previous
        def stop(signum,frame):raise KeyboardInterrupt(f'signal {signum}; resume from last complete epoch')
        signal.signal(signal.SIGTERM,stop)
        atomic_json(root/'GPU_COST.json',dict(status='RUNNING',gpu_seconds=previous,invocation_started_unix=time.time()))
        try:
            torch.set_num_threads(2)
            parent_meta=json.loads((root/'identity/PARENT.json').read_text())
            if file_sha256(parent_meta['parent_checkpoint'])!=parent_meta['sha256']:raise RuntimeError('parent hash mismatch')
            parent=torch.load(parent_meta['parent_checkpoint'],map_location='cpu',weights_only=False)
            assert parent['epoch']==44 and parent['update_index']==11308
            config=load_config(root/'identity/RESOLVED.json');base=json.loads((root/'identity/CONTRACT.json').read_text())
            protocol=json.loads((root/'identity/PROTOCOL.json').read_text())
            if protocol!=json.loads((PAPER/'configs/research/goal_coverage_pair_v1.json').read_text()):raise RuntimeError('protocol mismatch')
            identity=dict(source_commit=head,parent_sha256=parent_meta['sha256'],parent_epoch=44,
                config_sha256=file_sha256(root/'identity/RESOLVED.json'),base_contract_sha256=file_sha256(root/'identity/CONTRACT.json'),
                protocol_sha256=file_sha256(root/'identity/PROTOCOL.json'),end_epoch=47)
            path=root/'RUN_IDENTITY.json'
            if path.exists() and json.loads(path.read_text())!=identity:raise RuntimeError('run identity changed')
            atomic_json(path,identity)
            if not (root/'checks/REAL_GATE.json').exists():
                gates=[]
                for arm in ['control','candidate']:
                    contract=arm_contract(base,arm);model,opt,gen=make_model(parent,config,contract,torch.device('cuda'))
                    g=train_epoch(root,model,opt,gen,config,contract,arm,45,11308,limit=8)
                    plain=evaluate(model,config,torch.device('cuda'),2035,max_scenes=2)
                    detailed=evaluate(model,config,torch.device('cuda'),2035,max_scenes=2,endpoint_diagnostics=True)
                    for mode in ['G','H']:
                        for metric in ['minADE','minFDE','JADE','JFDE']:
                            if abs(plain[mode][metric]-detailed[mode][metric])>2e-5:
                                raise RuntimeError('endpoint diagnostics changed inference')
                    g['endpoint_diagnostic_inference_parity']=True
                    atomic_json(root/'checks'/f'gate_{arm}.json',g);gates.append(g)
                    del model,opt,gen;gc.collect();torch.cuda.empty_cache()
                compare_records(gates[0]['records'],gates[1]['records'])
                assert gates[0]['rng_end']==gates[1]['rng_end']
                atomic_json(root/'checks/REAL_GATE.json',dict(status='PASS',updates_per_arm=8,
                    same_inputs_rng=True,finite_gradients_and_parameters=True))
            for arm in ['control','candidate']:run_arm(root,parent,config,base,arm,identity)
            for epoch in range(45,48):
                a,b=[json.loads((root/arm/f'epoch_{epoch:03d}.json').read_text()) for arm in ['control','candidate']]
                compare_records(a['records'],b['records']);assert a['rng_end']==b['rng_end']
            summary=finish_report(root,final_evaluations(root,parent,config,base))
            atomic_json(root/'STATUS.json',dict(status='COMPLETE',decision=summary['decision'],time_unix=time.time()))
            print('COMPLETE '+summary['decision'],flush=True)
        except BaseException as exc:
            failure=repr(exc);atomic_json(root/'STATUS.json',dict(status='FAILED',reason=failure,time_unix=time.time()));raise
        finally:
            atomic_json(root/'GPU_COST.json',dict(status='FAILED' if failure else 'COMPLETE',
                gpu_seconds=previous+time.perf_counter()-started,failure=failure))
            if failure is None:
                docs=PAPER/'docs/research/20261114_goal_coverage_pair'
                docs.mkdir(parents=True,exist_ok=True)
                for name in ['REPORT_ZH.md','SUMMARY.json','PER_SEED_METRICS.csv']:
                    shutil.copy2(root/'reports'/name,docs/name)
                shutil.copy2(root/'GPU_COST.json',docs/'GPU_COST.json')


if __name__=='__main__':main()

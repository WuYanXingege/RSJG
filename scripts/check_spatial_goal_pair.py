#!/usr/bin/env python3
"""Bounded ETH endpoint-readout intervention: gate then one epoch per arm.

Never resumes the original 100-epoch run. The parent was trained from scratch;
this experiment is a controlled continuation, NOT a scratch-training result.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

PAPER=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PAPER))
import torch
from ebjd.config import load_config
from ebjd.proposals import file_sha256
from ebjd.spatial_goal import install_spatial_goal
from train_learned_goal_bridge import (build_scratch,optimizer_for,seed_all,restore_rng,
    capture_rng,perform_update,set_epoch_lr,evaluate,save_checkpoint,loader,atomic_json,
    bound_reference_metrics)


def digest_rng(generator):
    states=capture_rng(generator); h=hashlib.sha256()
    for value in [states['torch_cpu'],states['loader'],*states['torch_cuda']]:h.update(value.cpu().numpy().tobytes())
    h.update(repr(states['python']).encode());h.update(repr(states['numpy']).encode())
    return h.hexdigest()


def build_arm(parent,config,contract,arm,device):
    seed_all(config['seed'])
    model=build_scratch(config,contract).to(device)
    model.load_state_dict(parent['model'],strict=True)
    opt=optimizer_for(model,config);opt.load_state_dict(copy.deepcopy(parent['optimizer']))
    if arm=='candidate':
        # Identical new-module initialization each gate/full run. Training RNG
        # is restored below; architecture construction cannot shift its stream.
        seed_all(3101)
        opt=install_spatial_goal(model,opt,lambda m:optimizer_for(m,config))
    gen=torch.Generator();restore_rng(parent['rng'],gen)
    return model,opt,gen


def summarize(records):
    names=['loss','goal_nll','pair_loss','diffusion_loss','map_loss','geometry_loss',
           'goal_grad_norm','coupling_grad_norm','bridge_grad_norm','map_grad_norm']
    return {name:statistics.fmean(row[name] for row in records) for name in names}


def run_arm(root,parent,config,contract,arm,phase,identity):
    device=torch.device('cuda');model,opt,gen=build_arm(parent,config,contract,arm,device)
    model.train();epoch=parent['epoch']+1;lr=set_epoch_lr(opt,epoch,config)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
    records=[];pending=[];data=loader(config,'train',gen)
    original={n:p.detach().cpu().clone() for n,p in model.goal.named_parameters()}
    for cpu in data:
        pending.append(cpu)
        if len(pending)<config['training']['gradient_accumulation']:continue
        rng=digest_rng(gen)
        ids=[s for b in pending for s in b.scene_ids]
        batch_hash=hashlib.sha256()
        for b in pending:
            for t in [b.observed,b.future,b.valid]:batch_hash.update(t.numpy().tobytes())
        row=perform_update(model,opt,pending,config,contract,gen,device);pending=[]
        row.update(rng_before=rng,scene_ids=ids,input_sha256=batch_hash.hexdigest(),
                   update=parent['update_index']+len(records)+1,epoch=epoch,lr=lr)
        records.append(row)
        if len(records)==2:
            spatial=[p.grad for n,p in model.goal.named_parameters() if n.startswith('spatial_')]
            if arm=='candidate' and not all(g is not None and torch.isfinite(g).all() and g.abs().sum()>0 for g in spatial):
                raise RuntimeError('spatial query gradients missing after readout initialization')
        if len(records)%16==0:
            atomic_json(root/'STATUS.json',dict(status='RUNNING',phase=phase,arm=arm,epoch=epoch,
                completed_updates=len(records),last_loss=row['loss'],time_unix=time.time()))
            print(f'{phase} {arm} | epoch={epoch} updates={len(records)} loss={row["loss"]:.6f}',flush=True)
        if phase=='gate' and len(records)>=8:break
    if pending and phase!='gate':
        ids=[s for b in pending for s in b.scene_ids];rng=digest_rng(gen);h=hashlib.sha256()
        for b in pending:
            for t in [b.observed,b.future,b.valid]:h.update(t.numpy().tobytes())
        row=perform_update(model,opt,pending,config,contract,gen,device)
        row.update(rng_before=rng,scene_ids=ids,input_sha256=h.hexdigest(),update=parent['update_index']+len(records)+1,epoch=epoch,lr=lr)
        records.append(row)
    del data,pending
    torch.cuda.synchronize();train_seconds=time.perf_counter()-started
    changed=sum((p.detach().cpu()-original[n]).square().sum().item() for n,p in model.goal.named_parameters())**.5
    if not changed>0:raise RuntimeError('goal parameters did not update')
    rng_end=digest_rng(gen)
    result=dict(phase=phase,arm=arm,epoch=epoch,updates=len(records),train_seconds=train_seconds,
        seconds_per_update=train_seconds/len(records),training=summarize(records),
        rng_end=rng_end,goal_parameter_update_l2=changed,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,records=records)
    if phase=='epoch':
        rng=capture_rng(gen)
        try:val=evaluate(model,config,device,2035)
        finally:restore_rng(rng,gen)
        result['validation']=val
        selection=copy.deepcopy(parent['selection_state'])
        reference=bound_reference_metrics(config['paths']['matched_gdts_validation_result'],2035,
            file_sha256(config['paths']['validation_manifest']))
        m=val['H'];eligible=all(m[k]<=reference[k] for k in ['minADE','minFDE','JADE','JFDE'])
        if selection['best_jade'] is None or m['JADE']<selection['best_jade']['JADE']:
            selection['best_jade']=dict(epoch=epoch,**m)
        if eligible and (selection['best_eligible'] is None or m['JADE']<selection['best_eligible']['JADE']):
            selection['best_eligible']=dict(epoch=epoch,**m)
        checkpoint=root/arm/'last.pt'
        save_checkpoint(checkpoint,model,opt,epoch,parent['update_index']+len(records),gen,
            selection,dict(**identity,arm=arm,goal_head=('spatial_goal_queries_v1' if arm=='candidate' else 'legacy')),
            config)
        # Full state round-trip including all new state_dict keys and optimizer.
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer'])
        result.update(checkpoint=str(checkpoint),checkpoint_sha256=file_sha256(checkpoint),
            eligible=eligible,full_state_strict_restore=True,full_epoch_seconds=time.perf_counter()-started)
        del saved
    atomic_json(root/('checks' if phase=='gate' else arm)/f'{phase}_{arm}.json',result)
    del model,opt,gen,original;gc.collect();torch.cuda.empty_cache()
    return result


def assert_pair(control,candidate):
    if control['updates']!=candidate['updates']:raise RuntimeError('unequal update coverage')
    for a,b in zip(control['records'],candidate['records']):
        for key in ['scene_ids','input_sha256','rng_before','lr']:
            if a[key]!=b[key]:raise RuntimeError(f'paired training mismatch: {key}')
    if control['rng_end']!=candidate['rng_end']:raise RuntimeError('final training RNG mismatch')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source-commit',required=True)
    args=p.parse_args();root=Path(args.root)
    head=subprocess.check_output(['git','-C',str(PAPER),'rev-parse','HEAD'],text=True).strip()
    if head!=args.source_commit:raise RuntimeError('source binding mismatch')
    subprocess.run(['git','-C',str(PAPER),'diff','--exit-code','HEAD','--','ebjd','scripts','configs'],check=True)
    lock_path=root.parents[1]/'locks/ebjd_gpu0_training.lock';lock_path.parent.mkdir(exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('GPU occupied; no concurrent training allowed')
        started=time.perf_counter();failure=None
        atomic_json(root/'GPU_COST.json',dict(status='RUNNING',started_unix=time.time(),gpu_seconds=0))
        try:
            torch.set_num_threads(2)
            meta=json.loads((root/'identity/PARENT.json').read_text())
            if file_sha256(meta['parent_checkpoint'])!=meta['sha256']:raise RuntimeError('parent hash mismatch')
            parent=torch.load(meta['parent_checkpoint'],map_location='cpu',weights_only=False)
            config=load_config(root/'identity/RESOLVED.json');contract=json.loads((root/'identity/CONTRACT.json').read_text())
            identity=dict(source_commit=head,parent=meta,design='paired_one_epoch_spatial_readout',
                config_sha256=file_sha256(root/'identity/RESOLVED.json'),
                contract_sha256=file_sha256(root/'identity/CONTRACT.json'),
                comparison='fixed_epoch45_not_selected_best',new_initialization='zero residual readout',
                original_scratch_source='no_external_prediction_weights',long_training=False)
            atomic_json(root/'RUN_IDENTITY.json',identity)
            gate=[run_arm(root,parent,config,contract,a,'gate',identity) for a in ['control','candidate']]
            assert_pair(*gate);atomic_json(root/'checks/PAIR_GATE.json',dict(status='PASS',updates=8,
                same_input_order_rng=True,finite_gradients=True,
                note='candidate divergence is expected research intervention, not execution equivalence'))
            runs=[run_arm(root,parent,config,contract,a,'epoch',identity) for a in ['control','candidate']]
            assert_pair(*runs)
            a,b=[r['validation']['H'] for r in runs];keys=['minADE','minFDE','JADE','JFDE']
            delta={k:b[k]-a[k] for k in keys}
            passed=delta['minFDE']<=-.01 and all(delta[k]<=0 for k in ['minADE','JADE','JFDE'])
            summary=dict(status='COMPLETE',decision='PROMISING_SHORT_SCREEN' if passed else 'DO_NOT_ADOPT',
                control={k:v for k,v in runs[0].items() if k!='records'},
                candidate={k:v for k,v in runs[1].items() if k!='records'},delta_H=delta,
                paired_inputs_rng=True,training_speed_ratio=runs[0]['train_seconds']/runs[1]['train_seconds'],
                scope='one training continuation and one inference seed; not a from-scratch comparison',
                old_long_training_restarted=False)
            atomic_json(root/'reports/SUMMARY.json',summary)
            atomic_json(root/'STATUS.json',dict(status='COMPLETE',decision=summary['decision'],time_unix=time.time()))
        except BaseException as exc:
            failure=repr(exc);atomic_json(root/'STATUS.json',dict(status='FAILED',reason=failure,time_unix=time.time()))
            raise
        finally:
            atomic_json(root/'GPU_COST.json',dict(status='FAILED' if failure else 'COMPLETE',
                gpu_seconds=time.perf_counter()-started,failure=failure))


if __name__=='__main__':main()

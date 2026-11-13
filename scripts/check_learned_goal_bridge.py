#!/usr/bin/env python3
"""Bounded real-data startup check; no benchmark state enters formal training."""
import argparse
import copy
import json
from pathlib import Path
import time
import torch
from train_learned_goal_bridge import (
    build_scratch, load_config, seed_all, optimizer_for, set_epoch_lr, loader,
    perform_update, evaluate, save_checkpoint, restore_rng, atomic_json)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    root=args.root;config=load_config(root/'configs/RESOLVED.json');contract=json.loads((root/'configs/CONTRACT.json').read_text())
    report=root/'receipts/REAL_DATA_PREFLIGHT.json';start=time.perf_counter()
    previous=json.loads(report.read_text()) if report.exists() else None
    result=dict(status='RUNNING',formal_optimizer_updates=0,temporary_optimizer_updates=0)
    if previous:
        result['previous_attempts']=previous.get('previous_attempts',[])+[
            {k:previous.get(k) for k in ['status','error','gpu_development_wall_seconds','temporary_optimizer_updates']}]
    atomic_json(report,result)
    try:
        assert torch.cuda.is_available()
        device=torch.device('cuda');seed_all(117)
        model=build_scratch(config,contract).to(device).train();optimizer=optimizer_for(model,config)
        set_epoch_lr(optimizer,1,config);generator=torch.Generator().manual_seed(117)
        train_loader=loader(config,'train',generator);it=iter(train_loader)
        records=[];torch.cuda.reset_peak_memory_stats()
        fixed_batch=None;goals_before=None;fixed_noise=None
        for step in range(2):
            batches=[next(it) for _ in range(config['training']['gradient_accumulation'])]
            if step==0:
                fixed_batch=batches[0].compact_scenes(0,1).to(device)
                model.eval()
                with torch.no_grad():
                    ctx=model.encode_context(fixed_batch.observed,fixed_batch.semantic_maps,fixed_batch.valid)
                    fixed_noise=torch.randn(1,len(ctx.valid[0]),20,2,device=device)
                    goals_before=model.goal.sample(ctx,model.bridge.representation.goal_scale,noise=fixed_noise).clone()
                model.train()
            torch.cuda.synchronize();begin=time.perf_counter()
            record=perform_update(model,optimizer,batches,config,contract,generator,device)
            torch.cuda.synchronize();record['seconds']=time.perf_counter()-begin;records.append(record)
            result['temporary_optimizer_updates']+=1
        for parameter in model.parameters(): assert torch.isfinite(parameter).all()
        model.eval()
        with torch.no_grad():
            ctx=model.encode_context(fixed_batch.observed,fixed_batch.semantic_maps,fixed_batch.valid)
            goals_after=model.goal.sample(ctx,model.bridge.representation.goal_scale,noise=fixed_noise)
        displacement=float((goals_after-goals_before).norm(dim=-1).max())
        assert displacement>0
        # Actual 20 worlds / 20 DDIM; GT used solely by metric computation.
        indices=[]; populations=[]
        for index,batch in enumerate(loader(config,'validation',torch.Generator().manual_seed(99173))):
            people=int(batch.valid.sum())
            if not indices or (people>1 and len(indices)<3):
                indices.append(index);populations.append(people)
            if len(indices)==3:break
        assert any(n>1 for n in populations)
        evaluation=evaluate(model,config,device,2035,max_scenes=len(indices),scene_indices=set(indices))
        state=root/'preflight/full_state.pt'
        save_checkpoint(state,model,optimizer,0,2,generator,{}, {'temporary':True},config)
        payload=torch.load(state,map_location='cpu',weights_only=False)
        # Strict same-model reload avoids doubling GPU memory merely for testing.
        model.load_state_dict(payload['model'],strict=True);optimizer.load_state_dict(payload['optimizer'])
        restore_rng(payload['rng'],generator)
        assert payload['scheduler_state']['total_epochs']==100
        result.update(status='PASS',records=records,parameter_count=sum(p.numel() for p in model.parameters()),
            external_prediction_weights_loaded=[],goal_change_same_input_noise_max_m=displacement,
            evaluation_smoke=evaluation,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            evaluation_scene_indices=indices,evaluation_scene_populations=populations,
            strict_full_state_restore=True,formal_initialization='independent seed3101 fresh process',
            scope='2 complete accumulated real optimizer updates; 3 validation scenes, 20 worlds/20 steps; not a performance result')
    except BaseException as exc:
        result.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result['gpu_development_wall_seconds']=time.perf_counter()-start
        result['total_gpu_development_wall_seconds']=result['gpu_development_wall_seconds']+sum(
            p.get('gpu_development_wall_seconds',0) for p in result.get('previous_attempts',[]))
        atomic_json(report,result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Bounded real ETH update/inference preflight, never starts formal training.

Reuses only train-fitted scalar scales from an identity receipt, not any model
weights. A receipt and finally block account for failed as well as passed work.
"""
import argparse
import json
from pathlib import Path
import time

import torch
from train_learned_goal_bridge import (
    build_scratch, load_config, seed_all, optimizer_for, set_epoch_lr, loader,
    perform_update, evaluate, save_checkpoint, restore_rng, atomic_json, file_sha256)
from ebjd.spatial_density_goal import SPATIAL_SCHEMA


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--scale-identity',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=8)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args()
    if not 1<=args.updates<=12:raise ValueError('preflight supports 1–12 temporary updates only')
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('use an empty isolated preflight directory')
    args.output.mkdir(parents=True,exist_ok=True)
    report=args.output/'PREFLIGHT.json';started=time.perf_counter()
    result=dict(status='RUNNING',formal_optimizer_updates=0,temporary_optimizer_updates=0,
        scope='Implementation preflight, not performance or accuracy acceptance',device=args.device)
    atomic_json(report,result)
    try:
        torch.set_num_threads(2);seed_all(117)
        device=torch.device(args.device)
        if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        config=load_config(args.config);contract=json.loads(args.contract.read_text())
        if contract['schema']!=SPATIAL_SCHEMA:raise ValueError('requires spatial-density research contract')
        identity=json.loads(args.scale_identity.read_text())
        digest=file_sha256(config['paths']['train_manifest'])
        if identity['manifest_sha256']['train']!=digest:raise ValueError('scale source train manifest mismatch')
        scales=identity['representation_scales']
        if scales['source']!='train_manifest_only':raise ValueError('scales must be train-only')
        model=build_scratch(config,contract)
        model.bridge.representation.set_scales(scales['goal'],scales['bridge'])
        model.bridge.encoder.goal_scale.fill_(scales['goal'])
        model=model.to(device).train();optimizer=optimizer_for(model,config)
        set_epoch_lr(optimizer,1,config);generator=torch.Generator().manual_seed(117)
        train_loader=loader(config,'train',generator);iterator=iter(train_loader)
        records=[];input_receipts=[]
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        initial=model.goal.spatial_readout.weight.detach().clone()
        for _ in range(args.updates):
            batches=[next(iterator) for _ in range(config['training']['gradient_accumulation'])]
            input_receipts.append([dict(scene_ids=b.scene_ids,agents=b.valid.sum(1).tolist(),
                padded_slots=list(b.valid.shape),map_shape=list(b.semantic_maps.shape)) for b in batches])
            if device.type=='cuda':torch.cuda.synchronize()
            begin=time.perf_counter()
            record=perform_update(model,optimizer,batches,config,contract,generator,device)
            if device.type=='cuda':torch.cuda.synchronize()
            record['seconds']=time.perf_counter()-begin;records.append(record)
            result['temporary_optimizer_updates']+=1
            atomic_json(report,{**result,'records':records})
        del iterator,train_loader
        if any(not torch.isfinite(p).all() for p in model.parameters()):raise RuntimeError('nonfinite parameters')
        delta=float((model.goal.spatial_readout.weight-initial).detach().norm())
        if delta==0:raise RuntimeError('spatial density parameters did not update')
        if not any(r['coupling_grad_norm']>0 for r in records):raise RuntimeError('missing coupling gradient')
        # Real validation membership, one singleton and one multiagent window.
        indices=[];populations=[];seen=set()
        for index,b in enumerate(loader(config,'validation',torch.Generator().manual_seed(99173))):
            people=int(b.valid.sum());kind=people>1
            if kind not in seen:
                indices.append(index);populations.append(people);seen.add(kind)
            if len(seen)==2:break
        if not indices:raise RuntimeError('empty validation manifest')
        evaluation=evaluate(model,config,device,2035,max_scenes=len(indices),scene_indices=set(indices))
        checkpoint=args.output/'temporary_full_state.pt'
        save_checkpoint(checkpoint,model,optimizer,0,args.updates,generator,{},
            {'schema':SPATIAL_SCHEMA,'temporary':True},config)
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model.load_state_dict(payload['model'],strict=True);optimizer.load_state_dict(payload['optimizer'])
        restore_rng(payload['rng'],generator)
        import resource
        result.update(status='PASS',records=records,input_receipts=input_receipts,
            config_sha256=file_sha256(args.config),contract_sha256=file_sha256(args.contract),
            scale_source=str(args.scale_identity),scale_source_sha256=file_sha256(args.scale_identity),
            train_manifest_sha256=digest,scales=scales,checkpoint_sha256=file_sha256(checkpoint),
            external_prediction_weights_loaded=[],parameter_count=sum(p.numel() for p in model.parameters()),
            goal_parameter_count=sum(p.numel() for p in model.goal.parameters()),
            spatial_readout_update_l2=delta,strict_full_restore=True,
            validation_indices=indices,validation_populations=populations,evaluation_smoke=evaluation,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None,
            process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            no_formal_training_started=True)
    except BaseException as exc:
        result.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result['wall_seconds']=time.perf_counter()-started
        result['gpu_task_wall_seconds']=result['wall_seconds'] if args.device=='cuda' else 0
        atomic_json(report,result)
    print(json.dumps({k:result[k] for k in ('status','temporary_optimizer_updates','wall_seconds','peak_allocated_mib')},indent=2),flush=True)


if __name__=='__main__':main()

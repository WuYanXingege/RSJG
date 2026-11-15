#!/usr/bin/env python3
"""Real-data finite-K preflight and paired complete-update runtime benchmark.

All models are disposable. Formal training starts freshly and independently.
"""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import resource
import statistics
import time

import torch
from train_learned_goal_bridge import (
    build_scratch, load_config, seed_all, optimizer_for, set_epoch_lr, loader,
    perform_update, evaluate, save_checkpoint, capture_rng, restore_rng, atomic_json, file_sha256)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--scale-identity',type=Path,required=True)
    p.add_argument('--updates',type=int,default=32)
    p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--common-parent-gate',action='store_true',
        help='Separate local execution differences from discrete assignment trajectory divergence')
    p.add_argument('--candidate-denoiser-slots',type=int,choices=[32],
        help='Benchmark-only extension; map U-Net checkpoint remains enabled')
    args=p.parse_args()
    root=args.root;out=root/'preflight';out.mkdir(exist_ok=True)
    config=load_config(root/'configs/RESOLVED.json')
    contract=json.loads((root/'configs/CONTRACT.json').read_text())
    source=json.loads(args.scale_identity.read_text())
    candidate_mode='denoiser_graph32' if args.candidate_denoiser_slots else 'deferred_metrics_v1'
    assert source['manifest_sha256']['train']==file_sha256(config['paths']['train_manifest'])
    scales=source['representation_scales'];assert scales['source']=='train_manifest_only'
    started=time.perf_counter()
    result=dict(status='RUNNING',formal_optimizer_updates=0,temporary_optimizer_updates=0,
        limits=dict(loss_relative=0.001,gradient_norm_relative=0.01),runs=[],
        scales=scales,scope='implementation/runtime acceptance only; not predictive accuracy')
    def persist():
        result['wall_seconds']=time.perf_counter()-started
        result['gpu_task_wall_seconds']=result['wall_seconds']
        atomic_json(out/'RESULTS.json',result)
    persist()
    torch.set_num_threads(2)
    device=torch.device('cuda')
    try:
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        result.update(gpu=torch.cuda.get_device_name(),pytorch=torch.__version__,cuda=torch.version.cuda)
        seed_all(3101)
        initial=build_scratch(config,contract)
        initial.bridge.representation.set_scales(scales['goal'],scales['bridge'])
        initial.bridge.encoder.goal_scale.fill_(scales['goal'])
        initial_state=copy.deepcopy(initial.state_dict());del initial

        def arm(mode,label,steps,warmup=0,smoke=False):
            seed_all(3101)
            c=copy.deepcopy(config)
            c['training']['scratch_update_runtime']='scene_serial' if mode=='denoiser_graph32' else mode
            model=build_scratch(c,contract).to(device).train()
            if mode=='denoiser_graph32':model.bridge.denoiser.uncheckpointed_max_agent_slots=32
            model.load_state_dict(initial_state,strict=True)
            opt=optimizer_for(model,c);set_epoch_lr(opt,1,c)
            seed_all(7231)
            generator=torch.Generator().manual_seed(3101)
            dl=loader(c,'train',generator);it=iter(dl)
            records=[];inputs=[];waits=[];totals=[]
            torch.cuda.reset_peak_memory_stats()
            for i in range(warmup+steps):
                torch.cuda.synchronize();begin=time.perf_counter()
                batches=[next(it) for _ in range(c['training']['gradient_accumulation'])]
                waited=time.perf_counter()-begin
                inputs.append([dict(scenes=b.scene_ids,agents=b.valid.sum(1).tolist(),map_shape=list(b.semantic_maps.shape)) for b in batches])
                records.append(perform_update(model,opt,batches,c,contract,generator,device))
                torch.cuda.synchronize();elapsed=time.perf_counter()-begin
                result['temporary_optimizer_updates']+=1
                if i>=warmup:waits.append(waited);totals.append(elapsed)
                if i%8==0:persist()
            del it,dl,batches
            stats=dict(label=label,mode=mode,updates=steps,warmup=warmup,
                seconds_per_update=statistics.mean(totals),median_seconds=statistics.median(totals),
                data_wait_seconds=statistics.mean(waits),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                inputs=inputs,records=records,seconds_each=totals)
            final={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if smoke:
                before=final
                stats['real_validation_smoke']=evaluate(model,c,device,2035,max_scenes=2)
                ck=out/'temporary_complete_state.pt'
                save_checkpoint(ck,model,opt,0,steps,generator,{},dict(temporary=True),c)
                saved=torch.load(ck,map_location='cpu',weights_only=False)
                model.load_state_dict(saved['model'],strict=True);opt.load_state_dict(saved['optimizer'])
                restore_rng(saved['rng'],generator)
                stats['strict_full_restore']=True
                stats['model_parameters_changed']=any(not torch.equal(before[k],initial_state[k]) for k in before)
                del saved
            del model,opt;gc.collect();torch.cuda.empty_cache()
            result['runs'].append(stats);persist()
            return stats,final

        def compare(a,b,sa,sb):
            keys=['loss','goal_nll','finite_k_risk','pair_loss','diffusion_loss','goal_grad_norm','bridge_grad_norm','map_grad_norm','coupling_grad_norm']
            diff={k:max(abs(x[k]-y[k])/max(abs(x[k]),abs(y[k]),1e-6) for x,y in zip(a['records'],b['records'])) for k in keys}
            params={}
            for prefix in ['goal.','coupling.','bridge.']:
                num=sum((sa[k].double()-sb[k].double()).square().sum() for k in sa if k.startswith(prefix) and sa[k].is_floating_point())
                den=sum(sa[k].double().square().sum() for k in sa if k.startswith(prefix) and sa[k].is_floating_point())
                params[prefix]=float((num/den.clamp_min(1e-20)).sqrt())
            return dict(max_relative=diff,parameter_l2_relative=params,input_identities_equal=a['inputs']==b['inputs'])
        if args.common_parent_gate:
            # Each of eight *successive reference states* is branched afresh.
            # Every arm executes an actual AdamW step; no formal state is used.
            seed_all(3101);model=build_scratch(config,contract).to(device).train()
            model.load_state_dict(initial_state,strict=True)
            opt=optimizer_for(model,config);set_epoch_lr(opt,1,config)
            seed_all(7231);generator=torch.Generator().manual_seed(3101)
            dl=loader(config,'train',generator);it=iter(dl)
            all_records=[[],[],[]];deltas=[]
            for step in range(8):
                batches=[next(it) for _ in range(config['training']['gradient_accumulation'])]
                parent=copy.deepcopy(model.state_dict());parent_opt=copy.deepcopy(opt.state_dict())
                rng=capture_rng(generator);ends=[]
                for j,mode in enumerate(('scene_serial','scene_serial',candidate_mode)):
                    model.load_state_dict(parent,strict=True);opt.load_state_dict(copy.deepcopy(parent_opt))
                    restore_rng(rng,generator)
                    cc=copy.deepcopy(config)
                    cc['training']['scratch_update_runtime']='scene_serial' if mode=='denoiser_graph32' else mode
                    model.bridge.denoiser.uncheckpointed_max_agent_slots=(32 if mode=='denoiser_graph32' else config['training']['denoiser_uncheckpointed_max_agent_slots'])
                    record=perform_update(model,opt,batches,cc,contract,generator,device)
                    all_records[j].append(record);result['temporary_optimizer_updates']+=1
                    ends.append({n:p.detach().cpu().clone() for n,p in model.named_parameters()})
                    if j==0:
                        next_state=copy.deepcopy(model.state_dict());next_opt=copy.deepcopy(opt.state_dict())
                        next_rng=capture_rng(generator)
                dummy=[dict(records=[all_records[j][-1]],inputs=['same_cpu_batches']) for j in range(3)]
                deltas.append(dict(reference_repeat=compare(dummy[0],dummy[1],ends[0],ends[1]),
                    candidate=compare(dummy[0],dummy[2],ends[0],ends[2])))
                model.load_state_dict(next_state);opt.load_state_dict(next_opt);restore_rng(next_rng,generator)
                result['common_parent_steps']=deltas;persist()
            def maximum(kind):
                return dict(max_relative={k:max(x[kind]['max_relative'][k] for x in deltas) for k in deltas[0][kind]['max_relative']},
                    parameter_l2_relative={k:max(x[kind]['parameter_l2_relative'][k] for x in deltas) for k in deltas[0][kind]['parameter_l2_relative']},
                    input_identities_equal=True)
            result['baseline_repeat']=maximum('reference_repeat');result['runtime_gate']=maximum('candidate')
            cand=dict(model_parameters_changed=True)
            del it,dl,model,opt,parent,parent_opt,next_state,next_opt,batches,ends
            gc.collect();torch.cuda.empty_cache()
        else:
            base,bs=arm('scene_serial','gate_baseline',8)
            repeat,rs=arm('scene_serial','gate_baseline_repeat',8)
            cand,cs=arm(candidate_mode,'gate_candidate',8,smoke=True)
            result['baseline_repeat']=compare(base,repeat,bs,rs)
            result['runtime_gate']=compare(base,cand,bs,cs)
            del bs,rs,cs
        gate=result['runtime_gate'];v=gate['max_relative']
        correct=(gate['input_identities_equal'] and v['loss']<=.001 and
                 all(v[k]<=.01 for k in v if k.endswith('grad_norm')) and cand['model_parameters_changed'])
        result['runtime_correctness_pass']=correct;persist()
        if correct:
            speed=[]
            for repetition in range(2):
                # Reverse arm order in repetition two to reduce warm-cache bias.
                modes=['scene_serial',candidate_mode]
                if repetition:modes.reverse()
                pair={}
                for mode in modes:
                    record,_=arm(mode,f'performance_{repetition+1}_{mode}',args.updates,args.warmup)
                    pair[mode]=record['seconds_per_update']
                speed.append(pair['scene_serial']/pair[candidate_mode])
            result['paired_speedups']=speed
            result['selected_runtime']=candidate_mode if min(speed)>=1.03 else 'scene_serial'
        else:
            result['selected_runtime']='scene_serial'
        result['status']='PASS'
    except BaseException as exc:
        result.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:persist()
    print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2),flush=True)


if __name__=='__main__':main()

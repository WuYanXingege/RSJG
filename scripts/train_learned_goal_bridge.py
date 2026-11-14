#!/usr/bin/env python3
"""Scratch endpoint-density + learnable coupling + gated conditional bridge.

Dedicated research entry, never resumes frozen-provider checkpoints. The goal
head uses observations/maps only; bridge training is GT-condition likelihood,
not an incorrect fixed-predicted-endpoint reconstruction of a different GT.
"""
from __future__ import annotations
import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import time

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER))
import numpy as np
import torch
from ebjd.config import build_model, load_config
from ebjd.data import augment_batch_with_points
from ebjd.learned_goal import LearnedGoalCouplingBridge
from ebjd.goal_coverage import LEGACY, SEPARATED, marginal_goal_coverage
from ebjd.metrics import MetricAccumulator
from ebjd.objectives import diffusion_loss, gated_geometry_mean, map_loss, relative_motion_loss
from ebjd.proposals import file_sha256
from ebjd.representation import cosine_vp
from ebjd.sampling import differentiable_sample
from ebjd.train import fit_training_scales
from ebjd.unified import gather_candidates, joint_pair_loss
from run_zara1_native_prior_pilot import loader
from run_gated_release_background import atomic_json
from run_zara1_unified_coupling_bridge import bound_reference_metrics


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def check_contract(config, contract):
    t = config['training']
    assert config['paths']['initialization_role'] == 'all_random_scratch'
    assert not config['paths']['accepted_fold_matched_gdts_unet']
    assert not config['paths']['accepted_fold_matched_gdts_unet_sha256']
    assert t['epochs'] == 100 and t['early_stopping']['enabled'] is False
    assert t['rollout_training_enabled'] is False
    assert t['paired_marginal_actual_step_projection_enabled'] is False
    assert t['ebjd_input_pipeline'] == 'legacy_cpu_rigid'
    assert contract['schema'] == 'ebjd-learned-goal-scratch-v1'
    assert contract['goal'] == dict(components=20, min_std=0.05, max_std=4.0,
        mixture_weights='uniform', candidate_sampling='one_reparameterized_draw_per_component')
    assert contract['loss_weights'] == dict(goal_nll=1.0, joint_pair=1.0, coarse=0.25, geometry=0.05, map=0.10)
    assert config['denoiser']['block_type'] == 'endpoint_gated_dit_v1'
    assert config['denoiser']['coarse_blocks'] == config['denoiser']['refinement_blocks'] == 4
    assert config['denoiser']['hidden_dim'] == 192 and config['denoiser']['ff_dim'] == 768
    assert t['coupling_singleton_backward'] == 'identity'
    assert config['evaluation']['worlds'] == config['evaluation']['steps'] == 20
    assert contract['external_prediction_weights_loaded'] is False
    mode = contract.get('goal_training_objective', LEGACY)
    assert mode in (LEGACY, SEPARATED)
    if mode == SEPARATED:
        assert contract['coverage_weight'] == 1.0 and contract['coverage_temperature'] == 0.1


def build_scratch(config, contract):
    check_contract(config, contract)
    bridge = build_model(config, proposal_conditioned=True)
    g = contract['goal']
    return LearnedGoalCouplingBridge(bridge, components=g['components'], min_std=g['min_std'], max_std=g['max_std'])


def optimizer_for(model, config):
    t = config['training']; groups = []
    for is_map in (False, True):
        for decay in (True, False):
            parameters = [p for name, p in model.named_parameters() if p.requires_grad
                and ('bridge.encoder.map_unet' in name) == is_map
                and (p.ndim > 1 and not name.endswith('bias') and 'norm' not in name.lower()) == decay]
            groups.append(dict(params=parameters, lr=float(t['unet_lr'] if is_map else t['new_module_lr']),
                base_lr=float(t['unet_lr'] if is_map else t['new_module_lr']),
                weight_decay=float(t['weight_decay']) if decay else 0.0))
    return torch.optim.AdamW(groups, betas=tuple(float(v) for v in t['optimizer_betas']), eps=float(t['optimizer_eps']))


def set_epoch_lr(optimizer, epoch, config):
    t = config['training']; warmup = t['lr_warmup_epochs']; epochs = t['epochs']
    if epoch <= warmup: factor = epoch / warmup
    else:
        progress = min(1., (epoch - warmup) / (epochs - warmup))
        factor = t['lr_final_fraction'] + (1-t['lr_final_fraction']) * (1+math.cos(math.pi*progress))/2
    for group in optimizer.param_groups: group['lr'] = group['base_lr'] * factor
    return optimizer.param_groups[0]['lr']


def objective(model, batch, contract, *, time_draw=None, noise=None, goal_noise=None):
    context = model.encode_context(batch.observed, batch.semantic_maps, batch.valid)
    distribution = model.goal(context)
    proposals = model.goal.sample(context, model.bridge.representation.goal_scale,
        noise=goal_noise, distribution=distribution)
    nll = model.goal.nll(context, batch.future[..., -1, :], model.bridge.representation.goal_scale,
        distribution=distribution)
    mode = contract.get('goal_training_objective', LEGACY)
    if mode not in (LEGACY, SEPARATED):
        raise ValueError(f'unknown goal objective: {mode}')
    separated = mode == SEPARATED
    coverage = (marginal_goal_coverage(proposals, batch.future[..., -1, :], batch.valid,
        model.bridge.representation.goal_scale, contract['coverage_temperature']) if separated else None)
    pair_proposals = proposals.detach() if separated else proposals
    coupled, _, _ = model.coupling.assign(context, pair_proposals,
        model.bridge.representation.goal_scale, straight_through=True)
    pair = joint_pair_loss(coupled, batch.future[..., -1, :], batch.valid,
        model.bridge.representation.goal_scale, temperature=0.1)
    endpoint = batch.future[..., -1, :].detach()
    clean = model.bridge.representation.encode_target_conditioned(
        batch.future.detach(), context.baseline, endpoint, eta=0.)[:, None]
    if time_draw is None: time_draw = torch.randint(1, 101, (len(batch.observed),), device=batch.observed.device).float()/100
    if noise is None: noise = torch.randn_like(clean, dtype=torch.float32)
    alpha, sigma = cosine_vp(time_draw)
    alpha, sigma = alpha.to(clean).view(-1,1,1,1,1), sigma.to(clean).view(-1,1,1,1,1)
    noisy = alpha*clean.float() + sigma*noise
    target = alpha*noise - sigma*clean.float()
    final, coarse, coarse_path, _ = model.bridge.denoiser(noisy, time_draw, context,
        proposal_goals=endpoint[:, None], proposal_eta=0.)
    final_path, _ = model.bridge.representation.decode_conditioned(
        alpha*noisy-sigma*final.float(), context.baseline, endpoint[:, None], eta=0.)
    w = contract['loss_weights']
    diff = diffusion_loss(final, coarse, target, batch.valid, w['coarse'])
    geometry = gated_geometry_mean(0.5*(
        relative_motion_loss(coarse_path, batch.future, context, 6.,2.,4.8) +
        relative_motion_loss(final_path, batch.future, context, 6.,2.,4.8)), alpha)
    maps = map_loss(batch.future, context, 0.5)
    agent = diff + w['map']*maps + w['goal_nll']*nll
    if separated:
        agent = agent + contract['coverage_weight'] * coverage
    scene = w['geometry']*geometry + w['joint_pair']*pair
    terms = dict(goal_nll=nll, pair_loss=pair, diffusion_loss=diff, map_loss=maps, geometry_loss=geometry)
    if separated:
        terms['goal_coverage_loss'] = coverage
    return agent, scene, terms


def gradient_norm(module):
    grads = [p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(grads).sum().sqrt()) if grads else 0.


def perform_update(model, optimizer, batches, config, contract, generator, device):
    """One logical optimizer update; original scene/agent weighting preserved."""
    optimizer.zero_grad(set_to_none=True)
    stats = dict(loss=0., goal_nll=0., pair_loss=0., diffusion_loss=0., map_loss=0., geometry_loss=0.)
    if contract.get('goal_training_objective', LEGACY) == SEPARATED:
        stats['goal_coverage_loss'] = 0.
    scenes = agents = max_agents = 0
    for cpu in batches:
        # Geometry-only dummy points are discarded, never used as proposals.
        dummy = cpu.observed[:, None, :, -1].clone()
        prepared, _ = augment_batch_with_points(cpu, dummy, generator)
        b, n = prepared.observed.shape[:2]
        total_agents = int(prepared.valid.sum()); scenes += b; agents += total_agents
        times = torch.randint(1,101,(b,), device=device).float()/100
        noise = torch.randn(b,1,n,12,2,device=device)
        goal_noise = torch.randn(b,n,20,2,device=device)
        for index in range(b):
            positions = prepared.valid[index].nonzero(as_tuple=False).flatten().to(device)
            batch = prepared.compact_scenes(index,index+1).to(device)
            count = len(positions); max_agents = max(max_agents,count)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                    enabled=device.type=='cuda' and config['training']['mixed_precision'].startswith('bf16')):
                agent, scene, terms = objective(model, batch, contract,
                    time_draw=times[index:index+1], noise=noise[index:index+1,:,positions],
                    goal_noise=goal_noise[index:index+1,positions])
                loss = (count/total_agents*agent + scene/b)/len(batches)
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite training loss')
            loss.backward()
            stats['loss'] += float(loss.detach())
            for name, value in terms.items():
                weight = 1/b if name in ('pair_loss','geometry_loss') else count/total_agents
                stats[name] += float(value.detach())*weight/len(batches)
    stats.update(goal_grad_norm=gradient_norm(model.goal), coupling_grad_norm=gradient_norm(model.coupling),
        bridge_grad_norm=gradient_norm(model.bridge), map_grad_norm=gradient_norm(model.bridge.encoder.map_unet))
    if not all(math.isfinite(x) for x in stats.values()): raise FloatingPointError('nonfinite gradients/objective')
    if stats['goal_grad_norm'] == 0 or stats['bridge_grad_norm'] == 0 or stats['map_grad_norm'] == 0:
        raise RuntimeError('missing required learning gradients')
    torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['total_gradient_clip_norm'], error_if_nonfinite=True)
    optimizer.step()
    stats.update(scenes=scenes, agents=agents, max_scene_agents=max_agents)
    return stats


@torch.no_grad()
def evaluate(model, config, device, seed, *, max_scenes=None, scene_indices=None, endpoint_diagnostics=False):
    model.eval(); g_acc, h_acc = MetricAccumulator(), MetricAccumulator()
    generator = torch.Generator(device=device).manual_seed(seed)
    started=time.perf_counter(); count=0; agent_count=0
    identities=[]; endpoint_errors=[]; nll_sum=0.
    for scene_index,cpu in enumerate(loader(config,'validation',torch.Generator().manual_seed(99173))):
        if scene_indices is not None and scene_index not in scene_indices: continue
        batch=cpu.to(device)
        context=model.encode_context(batch.observed,batch.semantic_maps,batch.valid)
        proposals=model.goal.sample(context,model.bridge.representation.goal_scale,generator=generator)
        h_goals, permutation, _=model.coupling.assign(context,proposals,model.bridge.representation.goal_scale,straight_through=False)
        noise=torch.randn(len(batch.observed),20,batch.observed.shape[1],12,2,device=device,generator=generator)
        g_path,g_goal,_=differentiable_sample(model.bridge,context,noise,20,proposal_goals=proposals,proposal_eta=0.)
        h_path,h_goal,_=differentiable_sample(model.bridge,context,gather_candidates(noise,permutation),20,proposal_goals=h_goals,proposal_eta=0.)
        if not torch.equal(g_goal.sort(1).values,h_goal.sort(1).values): raise RuntimeError('endpoint set changed')
        if not torch.isfinite(g_path).all() or not torch.isfinite(h_path).all(): raise FloatingPointError('nonfinite inference')
        g_path,target=batch.paper_coordinates(g_path,batch.future)
        h_path,_=batch.paper_coordinates(h_path,batch.future)
        g_acc.update(g_path,target,batch.valid); h_acc.update(h_path,target,batch.valid)
        if endpoint_diagnostics:
            errors=torch.linalg.vector_norm(g_path[..., -1, :]-target[:,None,...,-1,:],dim=-1).amin(1)
            endpoint_errors.extend(errors[batch.valid].cpu().tolist())
            nll_sum+=float(model.goal.nll(context,batch.future[...,-1,:],
                model.bridge.representation.goal_scale))*int(batch.valid.sum())
        count+=len(batch.observed); agent_count+=int(batch.valid.sum())
        identities.extend(batch.scene_ids)
        if max_scenes is not None and count>=max_scenes: break
    result = dict(seed=seed,G=g_acc.as_dict(),H=h_acc.as_dict(),scenes=count,agent_occurrences=agent_count,
        seconds=time.perf_counter()-started,proposal_source='online_learned_uniform_mixture_stratified20',
        scene_ids=identities)
    if endpoint_diagnostics:
        errors=torch.tensor(endpoint_errors,dtype=torch.float64)
        result['endpoint_diagnostics']=dict(goal_nll=nll_sum/max(agent_count,1),
            recall_025=float((errors<=.25).double().mean()),recall_05=float((errors<=.5).double().mean()),
            recall_10=float((errors<=1.).double().mean()),
            P50=float(errors.quantile(.5)),P90=float(errors.quantile(.9)),P95=float(errors.quantile(.95)),
            agent_best_errors=endpoint_errors,unit='paper_metric_coordinates')
    return result


def capture_rng(generator):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],loader=generator.get_state())


def restore_rng(rng,generator):
    random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch_cpu'].cpu())
    if rng['torch_cuda']: torch.cuda.set_rng_state_all([s.cpu() for s in rng['torch_cuda']])
    generator.set_state(rng['loader'].cpu())


def save_checkpoint(path,model,optimizer,epoch,update,generator,selection,identity,config):
    payload=dict(schema='ebjd-learned-goal-scratch-v1',model=model.state_dict(),optimizer=optimizer.state_dict(),
        epoch=epoch,update_index=update,selection_state=selection,run_identity=identity,rng=capture_rng(generator),
        scheduler_state=dict(type='epoch_warmup_cosine',completed_epoch=epoch,total_epochs=config['training']['epochs'],
            warmup_epochs=config['training']['lr_warmup_epochs'],final_fraction=config['training']['lr_final_fraction'],
            group_lrs=[g['lr'] for g in optimizer.param_groups]),early_stop_state=dict(enabled=False))
    temporary=path.with_suffix('.tmp');torch.save(payload,temporary);os.replace(temporary,path)


def copy_atomic(source,target):
    temporary=target.with_suffix('.tmp');shutil.copy2(source,temporary);os.replace(temporary,target)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--contract',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--source-commit',required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(); config=load_config(args.config);contract=json.loads(Path(args.contract).read_text())
    check_contract(config,contract)
    head=subprocess.check_output(['git','-C',str(PAPER),'rev-parse','HEAD'],text=True).strip()
    if head!=args.source_commit: raise RuntimeError('source HEAD does not match launch contract')
    subprocess.run(['git','-C',str(PAPER),'diff','--exit-code','HEAD','--','ebjd','scripts','configs'],check=True)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True);device=torch.device(args.device)
    if args.resume and not (output/'last.pt').exists(): raise RuntimeError('resume requires full last.pt')
    if not args.resume and any(output.iterdir()): raise RuntimeError('fresh run requires empty output')
    seed_all(config['seed']);model=build_scratch(config,contract)
    scale_loader=loader(config,'train',torch.Generator().manual_seed(0))
    scales=fit_training_scales(model.bridge,scale_loader);del scale_loader
    model=model.to(device);optimizer=optimizer_for(model,config)
    validation_sha=file_sha256(config['paths']['validation_manifest'])
    ref_path=config['paths']['matched_gdts_validation_result']
    if file_sha256(ref_path)!=config['paths']['matched_gdts_validation_result_sha256']:
        raise RuntimeError('baseline result identity mismatch')
    gdts=bound_reference_metrics(ref_path,2035,validation_sha)
    identity=dict(schema=contract['schema'],source_commit=head,config_sha256=file_sha256(args.config),
        contract_sha256=file_sha256(args.contract),fold=config['experiment']['fold'],training_seed=config['seed'],
        initialization='all_prediction_parameters_random',external_weights_loaded=[],proposal_banks=[],
        endpoint_distribution=contract['goal'],denoiser=model.bridge.denoiser.architecture_signature(),
        representation_scales=dict(goal=scales[0],bridge=scales[1],source='train_manifest_only'),
        manifest_sha256={s:file_sha256(config['paths'][s+'_manifest']) for s in ['train','validation','test']},
        baseline_role='comparison_and_selector_only_no_weight_load',baseline_result_sha256=file_sha256(ref_path),
        worlds=20,ddim_steps=20,rollout=False,projection=False,epochs=100,early_stopping=False,
        denoiser_uncheckpointed_max_agent_slots=config['training']['denoiser_uncheckpointed_max_agent_slots'])
    generator=torch.Generator().manual_seed(config['seed']);update=0;start_epoch=1
    selection=dict(best_jade=None,best_eligible=None)
    if args.resume:
        payload=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
        if payload['schema']!=contract['schema'] or payload['run_identity']!=identity: raise RuntimeError('checkpoint identity mismatch')
        model.load_state_dict(payload['model'],strict=True);optimizer.load_state_dict(payload['optimizer'])
        update=payload['update_index'];start_epoch=payload['epoch']+1;selection=payload['selection_state']
        restore_rng(payload['rng'],generator)
        previous_logged_update=update
        if (output/'train.jsonl').exists():
            for line in (output/'train.jsonl').read_text().splitlines():
                try: previous_logged_update=max(previous_logged_update,int(json.loads(line)['update']))
                except (ValueError,KeyError): pass
        with (output/'resume_history.jsonl').open('a') as h:
            h.write(json.dumps(dict(resumed_complete_epoch=payload['epoch'],resumed_update=update,
                prior_log_max_update=previous_logged_update,
                unsaved_updates_to_replay=previous_logged_update-update,
                checkpoint_sha256=file_sha256(output/'last.pt'),time_unix=time.time()))+'\n')
        del payload
    atomic_json(output/'RUN_IDENTITY.json',identity)
    if not args.resume:
        save_checkpoint(output/'last.pt',model,optimizer,0,0,generator,selection,identity,config)
    print(f"ETH scratch learned-goal gated8 | seed={config['seed']} | epochs=100 | output={output}",flush=True)
    started=time.perf_counter();fields=None
    for epoch in range(start_epoch,101):
        epoch_start=time.perf_counter();lr=set_epoch_lr(optimizer,epoch,config);model.train()
        records=[];pending=[]
        train_loader=loader(config,'train',generator)
        for cpu in train_loader:
            pending.append(cpu)
            if len(pending)<config['training']['gradient_accumulation']: continue
            record=perform_update(model,optimizer,pending,config,contract,generator,device);pending=[];update+=1
            record.update(epoch=epoch,update=update,lr=lr,rollout_update=0.,projection_active=0.)
            records.append(record)
            with (output/'train.jsonl').open('a') as h:h.write(json.dumps(record)+'\n')
            if update==1:atomic_json(output/'FIRST_UPDATE.json',record)
        if pending:
            record=perform_update(model,optimizer,pending,config,contract,generator,device);update+=1
            record.update(epoch=epoch,update=update,lr=lr,rollout_update=0.,projection_active=0.)
            records.append(record)
            with (output/'train.jsonl').open('a') as h:h.write(json.dumps(record)+'\n')
        del train_loader
        train_seconds=time.perf_counter()-epoch_start
        rng=capture_rng(generator)
        try:validation=evaluate(model,config,device,2035)
        finally:restore_rng(rng,generator)
        metrics=validation['H'];eligible=all(metrics[k]<=gdts[k] for k in ['minADE','minFDE','JADE','JFDE'])
        better_jade=selection['best_jade'] is None or metrics['JADE']<selection['best_jade']['JADE']
        better_eligible=eligible and (selection['best_eligible'] is None or metrics['JADE']<selection['best_eligible']['JADE'])
        if better_jade:selection['best_jade']=dict(epoch=epoch,**metrics)
        if better_eligible:selection['best_eligible']=dict(epoch=epoch,**metrics)
        atomic_json(output/f'validation/epoch_{epoch:03d}.json',validation)
        save_checkpoint(output/'last.pt',model,optimizer,epoch,update,generator,selection,identity,config)
        if better_jade:copy_atomic(output/'last.pt',output/'best_jade.pt')
        if better_eligible:copy_atomic(output/'last.pt',output/'best_eligible.pt')
        if epoch==100:copy_atomic(output/'last.pt',output/'final.pt')
        checkpoint_sha=file_sha256(output/'last.pt')
        with (output/'checkpoint_index.jsonl').open('a') as h:
            h.write(json.dumps(dict(epoch=epoch,update=update,sha256=checkpoint_sha,best_jade=better_jade,
                best_eligible=better_eligible,note='last alias overwritten next epoch; only selected/final aliases retained'))+'\n')
        seconds=time.perf_counter()-epoch_start
        row=dict(epoch=epoch,update=update,train_loss=statistics.fmean(x['loss'] for x in records),
            goal_nll=statistics.fmean(x['goal_nll'] for x in records),pair_loss=statistics.fmean(x['pair_loss'] for x in records),
            **{mode+'_'+k:validation[mode][k] for mode in ['G','H'] for k in ['minADE','minFDE','JADE','JFDE']},eligible=eligible,
            train_seconds=train_seconds,validation_seconds=validation['seconds'],epoch_minutes=seconds/60,
            lr=lr,eta_hours=(time.perf_counter()-started)/(epoch-start_epoch+1)*(100-epoch)/3600)
        with (output/'epoch_metrics.csv').open('a',newline='') as h:
            writer=csv.DictWriter(h,fieldnames=row.keys())
            if h.tell()==0:writer.writeheader()
            writer.writerow(row)
        atomic_json(output/'STATUS.json',dict(status='TRAINING',**row,selection=selection))
        print(f"Epoch {epoch:03d}/100 | loss={row['train_loss']:.6f} goalNLL={row['goal_nll']:.4f} | minADE={metrics['minADE']:.6f} | minFDE={metrics['minFDE']:.6f} | JADE={metrics['JADE']:.6f} | JFDE={metrics['JFDE']:.6f} | eligible={eligible} | time={seconds/60:.2f}min | ETA={row['eta_hours']:.2f}h",flush=True)
    selected=output/('best_eligible.pt' if selection['best_eligible'] else 'best_jade.pt')
    payload=torch.load(selected,map_location='cpu',weights_only=False);model.load_state_dict(payload['model'],strict=True)
    selected_epoch=payload['epoch'];del payload
    digest=file_sha256(selected);runs=[]
    for seed in range(2035,2040):
        path=output/f'evaluations/seed_{seed}.json'
        existing=json.loads(path.read_text()) if path.exists() else None
        if existing and existing.get('checkpoint_sha256')==digest and existing.get('run_identity')==identity:
            result=existing['result']
        else:
            result=evaluate(model,config,device,seed)
            atomic_json(path,dict(checkpoint_sha256=digest,run_identity=identity,result=result))
        runs.append(result)
    summary={mode:{k:dict(mean=statistics.fmean(r[mode][k] for r in runs),
        sample_standard_deviation=statistics.stdev(r[mode][k] for r in runs))
        for k in ['minADE','minFDE','JADE','JFDE']} for mode in ['G','H']}
    final=dict(status='COMPLETE',selected_checkpoint=str(selected),selected_checkpoint_sha256=digest,
        selected_epoch=selected_epoch,selection_identity=selected.name,runs=runs,summary=summary,
        epochs=100,optimizer_updates=update,elapsed_this_invocation_seconds=time.perf_counter()-started,
        evaluation_identity='validation/test mirrored; one training seed; five inference seeds')
    atomic_json(output/'FINAL_EVALUATION.json',final);atomic_json(output/'STATUS.json',final)


if __name__=='__main__':main()

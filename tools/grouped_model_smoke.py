#!/usr/bin/env python3
"""Real classes/real train bytes, isolated SMOKE_ONLY, never production gate."""
import argparse,copy,gc,json,os,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.p2_grouped_training import *
from src.p2_grouped_artifacts import make_grant
from src.parser import get_parser
from src.models.goal_pretrain import Goal_Pretrain
from src.models.model import GDTS

def append(p,v):
    with p.open('a') as f:f.write(json.dumps(v,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())

def arguments(mode,device,out):
    a=get_parser().parse_args([])
    for k,v in dict(dataset='eth5',test_set='hotel',goal_model_type='independent',training_stage='baseline',
        p2_mode=mode,device=device,use_cuda=device.startswith('cuda'),model_dir=str(out),seq_length=20,
        obs_length=8,pred_length=12,down_factor=8,num_workers=0,batch_size=64,data_augmentation=False,
        seed=3101,validation_seed=3101,jdv2_active=False,amp_enabled=False,amp_dtype='fp32',learning_rate=.001 if mode=='goal' else .0001,
        use_scene_latent=False,use_dynamic_relation=False,use_joint_energy=False,use_dependency_corrector=False,
        branch_stage_step=18,num_epochs=150 if mode=='goal' else 250).items():setattr(a,k,v)
    a.branch_stage_step=int((a.ddpm_step-a.trunk_stage_step)//(a.ddpm_step/a.ddim_step))
    return a

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['plan','cpu','cuda','reload','fresh'])
    ap.add_argument('--archive',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();D=a.archive.resolve();O=a.output.resolve();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    m=json.loads((D/'QUALIFIED_INPUT_REVIEW_MANIFEST.json').read_text());reg=Registry(m,m['manifest_hash'])
    pm=json.loads((D/'PREPARATION_MANIFEST.json').read_text());preg=Registry(pm,pm['manifest_hash'])
    grant=PreparationGrant(preg,make_grant(preg));reader=PackReader(reg,'train',preparation=grant)
    plan=dict(seed=3101,train_packs=[0,1],train_members=[reader.packs[i] for i in [0,1]],
        inner_pack=0,inner_execution='BLOCKED_UNTIL_PROVENANCE_QUALIFIED',smoke_grant=grant.value['manifest_hash'],
        matrix={'cpu':{'goal_fp32':1,'joint_fp32':1},'cuda':{'goal_fp32':2,'joint_fp32':2,'goal_bf16':1,'joint_bf16':1}},
        total_planned_updates=8,hard_cumulative_attempt_cap=24,fp32_replay_atol=1e-6,bf16_replay_atol=2e-3,replay_rtol=1e-6,
        cross_precision_comparison='NOT_CLAIMED; support/finite update only',gpu_seconds=1800,
        gpu_reserved_limit=10*1024**3,namespace=str(O/'PREFLIGHT_SMOKE'),full_inner='only if qualification+time permit')
    if a.mode=='plan':atomic_json(plan,D/'SMOKE_PLAN.json');return
    if json.loads((D/'SMOKE_PLAN.json').read_text())!=json.loads(json.dumps(plan)):raise RuntimeError('smoke plan mismatch')
    if a.mode in ('cpu','cuda','reload'):
        # Dedicated smoke purpose, checked for the full preselected set before
        # any provider/model access. Preparation authorization alone is not it.
        for i in plan['train_packs']:
            for wid,j in reader.packs[i]:grant.authorize(reg.by_id[wid],'PREFLIGHT_SMOKE')
    ledger=D/'OPTIMIZER_LEDGER.jsonl';results=[];start=time.monotonic();device='cuda:0' if a.mode in ('cuda','reload') else 'cpu'
    attempt_id=str(time.time_ns());attempt_dir=D/'model_smoke_attempts';attempt_dir.mkdir(exist_ok=True)
    def report(value):
        value.update(smoke_purpose_authorized_before_provider=True,source=source_fingerprint(),
            data_binding=data_binding(reg),smoke_grant=grant.value['manifest_hash'])
        atomic_json(value,attempt_dir/(a.mode+'_'+attempt_id+'.json'))
        atomic_json(value,D/('MODEL_SMOKE_'+a.mode+'.json'))
    if a.mode=='fresh':
        args=arguments('goal','cpu',O/'FORMAL_FRESH_INITIAL');seed_all(3101,'cpu')
        model=Goal_Pretrain(args,'cpu',dataset=geometry_dataset(reg)).cpu();state=state_hash(model.state_dict())
        dest=O/'FORMAL_FRESH_INITIAL'/('goal_seed3101_unupdated_'+data_binding(reg)[:12]+'.pt')
        atomic_tensor(dict(classification='FORMAL_INITIAL_UNUPDATED',seed=3101,optimizer_updates=0,
            model_state_dict=model.state_dict(),state_sha256=state,config=vars(args),rng=rng_snapshot('cpu')),dest)
        atomic_json(dict(path=str(dest),sha256=file_hash(dest),state_sha256=state,updates=0,seed=3101,
            smoke_state_inherited=False,data_binding=data_binding(reg),source=source_fingerprint(),
            initialization_device='cpu; formal constructor also initializes parameters on CPU before device transfer',
            formal_config_path=str(D/'FRESH_GOAL_BLOCKED.yaml'),formal_config_sha256=file_hash(D/'FRESH_GOAL_BLOCKED.yaml'),
            scope='fresh unupdated initialization audit, not a selected parent or resume checkpoint',
            status='UNUPDATED_INITIAL_NOT_QUALIFIED_PARENT'),D/'FRESH_INITIAL.json');return
    if device.startswith('cuda'):
        query=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.free','--format=csv,noheader,nounits'],text=True,capture_output=True)
        compute=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True,capture_output=True)
        atomic_json(dict(gpus=query.stdout,compute=compute.stdout,returncodes=[query.returncode,compute.returncode]),D/('GPU_QUERY_'+a.mode+'.json'))
        if query.returncode or compute.returncode or compute.stdout.strip():
            report(dict(status='RESOURCE_BLOCKED',reason='DEVICE_BUSY_OR_QUERY_FAILED',results=[],gpu_held_seconds=0))
            raise RuntimeError('RESOURCE_BLOCKED_DEVICE_BUSY_OR_QUERY_FAILED')
        vals=query.stdout.splitlines()[0].split(',');free=int(vals[3])*1024**2
        budget=min(10*1024**3,free-1024**3)
        if budget<2*1024**3:raise RuntimeError('RESOURCE_BLOCKED_MEMORY')
        gpu_ledger=D/'GPU_BUDGET_LEDGER.jsonl'
        prior=sum(json.loads(x)['charged_seconds'] for x in gpu_ledger.read_text().splitlines()) if gpu_ledger.exists() else 0.
        if prior>=1800:raise RuntimeError('GPU_CUMULATIVE_BUDGET')
        torch.cuda.set_device(0);actual_free,total=torch.cuda.mem_get_info()
        budget=min(budget,actual_free-1024**3)
        if budget<2*1024**3:raise RuntimeError('RESOURCE_BLOCKED_ACTUAL_MEMORY')
        torch.cuda.set_per_process_memory_fraction(budget/torch.cuda.get_device_properties(0).total_memory,0)
        torch.cuda.reset_peak_memory_stats();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
    else:prior=0;budget=0
    cases=[('goal',False,1),('joint',False,1)] if a.mode=='cpu' else [('goal',False,2),('joint',False,2),('goal',True,1),('joint',True,1)]
    if a.mode=='reload':cases=[('goal',False,0),('joint',False,0),('goal',True,0),('joint',True,0)]
    try:
        for kind,bf16,steps in cases:
            label=kind+('_bf16' if bf16 else '_fp32');out=O/'PREFLIGHT_SMOKE'/a.mode/attempt_id/label
            if a.mode=='reload':
                cuda_result=json.loads((D/'MODEL_SMOKE_cuda.json').read_text())
                if cuda_result['status']!='PASS':raise RuntimeError('reload requires completed CUDA matrix')
                match=next(x for x in cuda_result['results'] if x['case']==label)
                if match['status']=='NOT_SUPPORTED':results.append(match);continue
                out=Path(match['checkpoint']['path']).parent
            if bf16 and not torch.cuda.is_bf16_supported():results.append(dict(case=label,status='NOT_SUPPORTED'));continue
            args=arguments(kind,device,out);args.amp_enabled=bf16;args.amp_dtype='bf16' if bf16 else 'fp32'
            seed_all(3101,device);model=(Goal_Pretrain if kind=='goal' else GDTS)(args,torch.device(device),dataset=geometry_dataset(reg,('train',))).to(device)
            initial=state_hash(model.state_dict());arc=architecture(model)
            opt=torch.optim.Adam(model.parameters(),lr=args.learning_rate);sched=torch.optim.lr_scheduler.ExponentialLR(opt,gamma=.99)
            rows=[];case_start=time.monotonic();last=None
            if a.mode=='reload':
                saved=torch.load(out/'state.pt',map_location='cpu',weights_only=True)
                if saved['classification']!='SMOKE_ONLY' or state_hash(saved['model_state_dict'])!=saved['state_sha256']:raise RuntimeError('reload state')
                model.load_state_dict(saved['model_state_dict'],strict=True);opt.load_state_dict(saved['optimizer']);sched.load_state_dict(saved['scheduler']);restore_rng(saved['rng'])
                if state_hash(model.state_dict())!=saved['state_sha256']:raise RuntimeError('reload model bytes')
                if tree_hash(opt.state_dict())!=saved['optimizer_hash'] or tree_hash(sched.state_dict())!=saved['scheduler_hash'] or tree_hash(rng_snapshot(device))!=saved['rng_hash']:raise RuntimeError('independent optimizer/scheduler/RNG readback')
            for i in range(steps):
                attempts=sum(json.loads(x)['event']=='ATTEMPT' for x in ledger.read_text().splitlines()) if ledger.exists() else 0
                if attempts>=24 or (device.startswith('cuda') and time.monotonic()-start+prior>=1800):raise RuntimeError('CUMULATIVE_SMOKE_CAP_BEFORE_FETCH')
                batch,ids,obs,target=reader.training(plan['train_packs'][i]);model.train();prepared=model.prepare_inputs(batch,ids);x,seq=prepared[0],prepared[-1]
                attempt=attempts+1;append(ledger,dict(event='ATTEMPT',number=attempt,case=a.mode+'/'+label,pack=plan['train_packs'][i],role='train',members=reader.packs[plan['train_packs'][i]]))
                opt.zero_grad(set_to_none=True);before=state_hash(model.state_dict());t=time.monotonic()
                family_before={prefix:state_hash({k:v for k,v in model.state_dict().items() if k.startswith(prefix)}) for prefix in (('goal_module.',) if kind=='goal' else ('goal_module.','registrar.','diffnet.'))}
                with torch.autocast(device_type=torch.device(device).type,dtype=torch.bfloat16,enabled=bf16):
                    losses=model.get_loss(x,seq);coef=model.set_losses_coeffs();loss=sum(v*coef[k] for k,v in losses.items())
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite smoke loss')
                loss.backward();grads=[p.grad for p in model.parameters() if p.grad is not None]
                if not grads or not all(torch.isfinite(g).all() for g in grads):raise FloatingPointError('smoke gradients')
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
                if device.startswith('cuda'):torch.cuda.synchronize()
                after=state_hash(model.state_dict())
                if before==after:raise RuntimeError('no parameter update')
                family_changed={prefix:family_before[prefix]!=state_hash({k:v for k,v in model.state_dict().items() if k.startswith(prefix)}) for prefix in family_before}
                if not all(family_changed.values()):raise RuntimeError('expected trainable family unchanged')
                row=dict(event='SUCCESS',number=attempt,case=a.mode+'/'+label,loss=float(loss.detach()),
                    components={k:float(v.detach()) for k,v in losses.items()},grad_norm=float(norm),seconds=time.monotonic()-t,
                    parameters_changed=True,family_changed=family_changed,parameter_dtype=str(next(model.parameters()).dtype),loss_dtype=str(loss.dtype),agents=len(reader.packs[i]))
                append(ledger,row);rows.append(row);last=(batch,ids)
                if device.startswith('cuda') and torch.cuda.max_memory_reserved()>budget:raise RuntimeError('GPU_MEMORY_LIMIT')
                del loss,losses,x,seq,prepared,batch,obs,target
            inference=None
            if a.mode=='cuda' and not bf16:
                model.eval();obs=reader.observation(0)
                scene=model.dataset.scenes[reg.by_id[reader.packs[0][0][0]]['source']['scene_family']]
                inputs=observation_inputs(obs,scene,device)
                assert inputs['input_traj_maps'].shape[1]==8 and inputs['x_augmented'].shape[0]==8
                with torch.no_grad():pred=model(inputs) if kind=='goal' else model(inputs,if_test=True)[0]
                if not torch.isfinite(pred).all():raise FloatingPointError('observation-only inference')
                inference=dict(shape=list(pred.shape),dtype=str(pred.dtype),finite=True,source_role='train',inner_access=False)
                del inputs,pred,obs
            # Save at explicit optimizer boundary, no scheduler/epoch advance for partial smoke.
            if a.mode!='reload':
                progress=dict(epoch_completed=0,cursor=steps,attempts=steps,successful=steps,skipped=0,failed=0,agent_exposures=sum(len(reader.packs[i]) for i in range(steps)))
                saved_ref=save_checkpoint(out/'state.pt',model,opt,sched,reg,args,progress,initial,'SMOKE_ONLY')
            else:saved_ref=dict(path=str(out/'state.pt'),sha256=file_hash(out/'state.pt'))
            # Fixed-input/noise replay after save: snapshot RNG before identical real loss.
            batch,ids,obs,target=reader.training(0);prepared=model.prepare_inputs(batch,ids);model.eval();snap=rng_snapshot(device)
            with torch.no_grad(),torch.autocast(device_type=torch.device(device).type,dtype=torch.bfloat16,enabled=bf16):
                losses=model.get_loss(prepared[0],prepared[-1]);behavior=torch.stack([v.detach().float() for k,v in sorted(losses.items())]).cpu()
            replay_max_abs=None;replay_within_tolerance=None
            if a.mode=='reload':
                expected=torch.load(out/'replay.pt',weights_only=True)
                equal=torch.equal(behavior,expected['behavior'])
                replay_max_abs=float((behavior-expected['behavior']).abs().max().item())
                replay_within_tolerance=torch.allclose(behavior,expected['behavior'],atol=plan['bf16_replay_atol' if bf16 else 'fp32_replay_atol'],rtol=plan['replay_rtol'])
                if not replay_within_tolerance:raise RuntimeError('independent-process replay outside preregistered tolerance: '+str(replay_max_abs))
            else:
                atomic_tensor(dict(behavior=behavior,rng=snap,scope='real model eval loss fixed input/noise; not CUDA long-training resume'),out/'replay.pt');equal=None
            results.append(dict(case=label,status='PASS',updates=steps,rows=rows,architecture=arc,checkpoint=saved_ref,
                seconds=time.monotonic()-case_start,replay_equal=equal,replay_max_abs=replay_max_abs,replay_within_tolerance=replay_within_tolerance,inference=inference,inner_selection='EXTERNAL_EVIDENCE_BLOCKED',
                peak_reserved=torch.cuda.max_memory_reserved() if device.startswith('cuda') else 0))
            del model,opt,sched,prepared,losses,behavior,batch,obs,target;gc.collect()
            if device.startswith('cuda'):torch.cuda.empty_cache()
    except Exception as e:
        append(ledger,dict(event='FAILURE',stage=a.mode,error=type(e).__name__+': '+str(e)))
        report(dict(status='FAILED',error=traceback.format_exc(),results=results,gpu_held_seconds=time.monotonic()-start if device.startswith('cuda') else 0));raise
    report(dict(status='PASS',results=results,seconds=time.monotonic()-start,
        gpu_held_seconds=time.monotonic()-start if device.startswith('cuda') else 0,
        inner_selection='BLOCKED_NOT_RUN_ON_UNQUALIFIED_INNER',classification='SMOKE_ONLY'))
if __name__=='__main__':main()

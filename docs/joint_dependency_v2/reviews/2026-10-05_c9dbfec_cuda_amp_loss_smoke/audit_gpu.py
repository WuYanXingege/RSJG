"""Bounded, one-shot audit. --plan is CPU-only; --run refuses an existing run.
Run from repository root using the pinned environment, never a training launcher.
"""
import argparse
from collections import Counter
import contextlib
import hashlib
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
import torch
from src.joint_goal_loss import expected_conditional_composite as ce
from src.jdv2_objective_state import ObjectiveRNG, CUDA_BACKEND, MC
from src.trainer import trainer
from mc_wiring_fixture import args as toy_args
from test_jdv2_expected_conditional import explicit_conditional, tiny
from prepare_real import settings, describe, sha256, CONFIG, CONFIG_SHA, CHECKPOINT, CHECKPOINT_SHA, CACHE, CACHE_SHA


def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def make_args(amp=False, cuda=False):
    a = toy_args('/tmp/rsjg_cuda_audit_unused')
    if cuda:
        a.device = 'cuda:0'; a.jdv2_mc_backend = CUDA_BACKEND
        a.amp_enabled = amp; a.amp_dtype = 'bf16' if amp else 'fp32'
    return a


def fixture(name):
    u,q,s,e,c,_ = tiny(torch.float32)
    c = c.detach() * .65
    mask = torch.ones_like(q, dtype=torch.bool)
    if name.startswith('singleton'):
        u,q,s = u[-1:],q[-1:],s[-1:]
        e,c,mask = e[:,:0],c[:0],mask[-1:]
    elif name.startswith('empty_unequal'):
        e,c = e[:,:0],c[:0]
    elif name.startswith('masked'):
        # Explicit agent/candidate permutations, re-canonicalizing edge direction.
        p=torch.tensor([2,0,3,1,4]); inv=torch.argsort(p); k=torch.tensor([2,0,1])
        u,q,s=u[p][:,k],q[p][:,k],s[p]
        e=inv[e]; c=c[:,k][:,:,k]
        flip=e[0]>e[1]; c[flip]=c[flip].transpose(1,2)
        e=e.sort(dim=0).values
        mask[1,1]=False; q=q*mask; q=q/q.sum(-1,keepdim=True)
        u=u.T.contiguous().T; c=c.transpose(1,2).contiguous().transpose(1,2)
    raw=dict(u=u.detach(),c=c,q=q,scenes=s,edges=e,mask=mask)
    if name.startswith('mixture'):
        E,K=c.shape[:2]
        def wave(shape,phase):
            return .3*torch.sin(torch.arange(torch.tensor(shape).prod().item()).reshape(shape)*.37+phase)
        raw.update(deploy=wave((E,K,K,4),.3),teacher=wave((E,4),.7),
                   left=wave((E,4,K,2),.9),right=wave((E,4,K,2),1.1))
    return raw


def graph(raw, device, dtype):
    keys = ['u','deploy','teacher','left','right'] if 'deploy' in raw else ['u','c']
    r={k:v.to(device=device,dtype=dtype if v.is_floating_point() else v.dtype).detach().clone()
       for k,v in raw.items()}
    for k in keys: r[k].requires_grad_(True)
    if 'deploy' in r:
        # Exactly the declared effective FP32 algebra, no BF16 upstream substitution.
        with torch.autocast(device_type=device.type,enabled=False):
            en=-torch.einsum('emkr,emlr->eklm',r['left'],r['right'])/(2**.5)
            prior=-torch.logsumexp(torch.log_softmax(r['deploy'],-1)-en,-1)
            post=-torch.logsumexp(torch.log_softmax(r['teacher'],-1)[:,None,None,:]-en,-1)
        costs=[post,prior]
    else: costs=[r['c']]
    for c in costs:
        if c.requires_grad: c.retain_grad()
    return r,keys,costs


def scalar(r,costs,z,reference=False,backend='cpu_v1'):
    vals=[]
    for c in costs:
        if reference:
            val=torch.stack([explicit_conditional(r['u'],r['q'],r['scenes'],r['edges'],c,row,r['mask']) for row in z]).mean()
        else:
            chunks=[(i,min(i+2,len(c)),c[i:i+2]) for i in range(0,len(c),2)]
            val=ce(r['u'],r['q'],r['scenes'],r['edges'],chunks,z,r['mask'],compute_backend=backend)
        vals.append(val)
    return sum(vals)/len(vals)


def compare(a,b):
    a,b=a.detach().cpu().double(),b.detach().cpu().double()
    delta=(a-b).abs()
    row=dict(max_abs=float(delta.max()) if delta.numel() else 0.,
             max_relative=float((delta/b.abs().clamp_min(1e-12)).max()) if delta.numel() else 0.,
             reference_scale=float(b.abs().max()) if b.numel() else 0.,
             passed=bool(torch.allclose(a,b,atol=1e-5,rtol=1e-4)))
    assert row['passed'], row
    return row


def plan():
    assert not torch.cuda.is_initialized()
    names=json.loads((HERE/'AUDIT_PLAN.json').read_text())['gpu_valid_cases']
    payload={}; manifest={}
    for name in names:
        raw=fixture(name); owner=ObjectiveRNG(make_args())
        start=owner.generator.get_state().clone()
        z=owner.draw(raw['u'],raw['q'],raw['scenes'],raw['edges'],raw['mask'])
        assert raw['u'].abs().max()<=2 and (not raw['c'].numel() or raw['c'].abs().max()<=2)
        refs={}
        # FP32 implementation plus double raw algebra; exact effective-input
        # double reference is separately recorded below (important for mixtures).
        for label,dtype,independent in [('cpu_fp32',torch.float32,False),('cpu_fp64_loop',torch.float64,True)]:
            r,keys,costs=graph(raw,torch.device('cpu'),torch.float32)
            if independent:
                # For raw-chain comparison, use independent double algebra too;
                # the FP32-vs-double upstream discrepancy is bounded here.
                r,keys,costs=graph(raw,torch.device('cpu'),dtype)
            out=scalar(r,costs,z,reference=independent)
            out.backward()
            refs[label]=dict(value=out.detach(),grads={k: None if r[k].grad is None else r[k].grad.detach() for k in keys},
                cost_grads=[None if c.grad is None else c.grad.detach() for c in costs])
        r0,_,cost0=graph(raw,torch.device('cpu'),torch.float32)
        r64={k:v.detach().double() if v.is_floating_point() else v for k,v in r0.items()}
        r64['u'].requires_grad_(True)
        cost64=[c.detach().double().requires_grad_(True) for c in cost0]
        same=scalar(r64,cost64,z,reference=True);same.backward()
        refs['cpu_fp64_identical_effective_inputs']=dict(value=same.detach(),grads={'u':r64['u'].grad.detach()},
            cost_grads=[None if c.grad is None else c.grad.detach() for c in cost64])
        compare(refs['cpu_fp32']['value'],refs['cpu_fp64_loop']['value'])
        payload[name]=dict(raw=raw,z=z,start=start,end=owner.generator.get_state().clone(),refs=refs)
        manifest[name]=dict(inputs={k:describe(v) for k,v in raw.items()},z=z.tolist(),
            start_sha256=digest(start),end_sha256=digest(payload[name]['end']),
            cpu_fp32_value=float(refs['cpu_fp32']['value']),cpu_fp64_value=float(refs['cpu_fp64_loop']['value']),
            draw_calls=1,agent_draws=z.numel())
    torch.save(payload,HERE/'SYNTHETIC_INPUTS.pt')
    dump(HERE/'SYNTHETIC_PLAN.json',dict(cases=manifest,payload_sha256=sha256(HERE/'SYNTHETIC_INPUTS.pt'),cuda_initialized=False))
    print('CPU plan fixed: 16 cases; CUDA initialization 0',flush=True)


def resource():
    cmd=['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.free','--format=csv,noheader,nounits']
    g=subprocess.check_output(cmd,text=True).strip()
    owners=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip()
    return dict(gpu=g,compute_owners=owners)


def policy():
    return dict(threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
        float32_matmul_precision=torch.get_float32_matmul_precision())


def run():
    output=HERE/'gpu_run'
    output.mkdir(exist_ok=False)  # Persistent attempt marker: no automatic rerun.
    ledger=Counter(); result=dict(status='INCOMPLETE',ledger=ledger,synthetic=[],illegal=[],real=[])
    started=None; current=None; owner=None
    def save(): dump(output/'RESULTS.json',result)
    def budget():
        torch.cuda.synchronize()
        result['gpu_seconds']=time.monotonic()-started
        result['peak_allocated']=torch.cuda.max_memory_allocated()
        result['peak_reserved']=torch.cuda.max_memory_reserved()
        assert result['gpu_seconds']<=900 and result['peak_reserved']<=4*1024**3
        save()
    def timeout(*_): raise TimeoutError('900-second CUDA budget exhausted')
    try:
        result['resources_before']=resource()
        assert not result['resources_before']['compute_owners'], 'BLOCKED_RESOURCE_UNAVAILABLE'
        assert int(result['resources_before']['gpu'].split(',')[-1])>=6144, 'BLOCKED_RESOURCE_UNAVAILABLE'
        result['policy_before']=policy()
        cpuplan=json.loads((HERE/'SYNTHETIC_PLAN.json').read_text())
        assert sha256(HERE/'SYNTHETIC_INPUTS.pt')==cpuplan['payload_sha256']
        inputs=torch.load(HERE/'SYNTHETIC_INPUTS.pt',map_location='cpu',weights_only=True)
        # Legitimate train input preparation before CUDA clock, no model creation.
        from prepare_real import dataset_set_name, default_collate
        pre=json.loads((HERE/'PREFLIGHT.json').read_text())
        a=settings(); ds=dataset_set_name(a,'train')
        for v in pre['selection'].values():
            cp=ROOT/CACHE/'train'/f"{v['index']:06d}.pt"
            assert sha256(cp)==v['cache_sha256']
            assert sha256(cp.with_name(f"{v['index']:06d}.teacher.pt"))==v['teacher_sha256']
            assert sha256(Path(ds.path_to_folder)/v['filename'])==v['source_batch_sha256']
        batches={k:default_collate([ds[v['index']]]) for k,v in pre['selection'].items()}
        ledger.update(source_batch_reads=2,candidate_cache_reads=2,teacher_cache_reads=2)
        for path,expected in [(CONFIG,CONFIG_SHA),(CHECKPOINT,CHECKPOINT_SHA),(CACHE+'/manifest.json',CACHE_SHA)]:
            assert sha256(ROOT/path)==expected
        started=time.monotonic(); signal.signal(signal.SIGALRM,timeout); signal.alarm(900)
        ledger['cuda_initialization_attempted']+=1
        torch.cuda.init(); ledger['cuda_initialization_completed']+=1
        result['environment']=dict(torch=str(torch.__version__),cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),device=torch.cuda.get_device_name(),
            capability=torch.cuda.get_device_capability(),bf16_native=torch.cuda.is_bf16_supported(including_emulation=False))
        assert result['environment']['bf16_native'], 'BF16_UNSUPPORTED'
        dev=torch.device('cuda:0')
        evalgen=torch.Generator(device=dev).manual_seed(628371)
        for name,p in inputs.items():
            current=name; owner=None; row=dict(name=name,status='ATTEMPTED');result['synthetic'].append(row)
            ledger['synthetic_valid_attempted']+=1; save()
            amp=name.endswith('bf16'); owner=ObjectiveRNG(make_args(amp,True))
            owner.generator.set_state(p['start'])
            raw={k:v.to(dev) for k,v in p['raw'].items()}
            before=(torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone(),evalgen.get_state().clone())
            z=owner.draw(raw['u'],raw['q'],raw['scenes'],raw['edges'],raw['mask'])
            assert torch.equal(z.cpu(),p['z']) and torch.equal(owner.generator.get_state(),p['end'])
            assert all(torch.equal(x,y) for x,y in zip(before,(torch.get_rng_state(),torch.cuda.get_rng_state(),evalgen.get_state())))
            ledger['synthetic_draws']+=1;ledger['synthetic_agent_draws']+=z.numel()
            ledger['synthetic_forward_attempted']+=1
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
                r,keys,costs=graph(p['raw'],dev,torch.float32)
                out=scalar(r,costs,z,backend=CUDA_BACKEND)
            ledger['synthetic_forward_completed']+=1
            assert out.dtype==torch.float32 and out.device==dev and torch.isfinite(out)
            ledger['synthetic_backward_attempted']+=1
            out.backward();ledger['synthetic_backward_completed']+=1
            row['comparisons']={}
            for label,ref in p['refs'].items():
                checks={'value':compare(out,ref['value'])}
                for k in keys:
                    if k not in ref['grads']:continue
                    if ref['grads'][k] is None:
                        assert r[k].grad is None
                    else:
                        assert r[k].grad.device==dev and torch.isfinite(r[k].grad).all()
                        checks[k]=compare(r[k].grad,ref['grads'][k])
                for i,(c,g) in enumerate(zip(costs,ref['cost_grads'])):
                    if g is not None:checks['C'+str(i)]=compare(c.grad,g)
                row['comparisons'][label]=checks
            if name.startswith('toy_'):
                params=torch.nn.ParameterList([torch.nn.Parameter(r[k].detach().clone()) for k in keys])
                for param,k in zip(params,keys):param.grad=r[k].grad.clone() if r[k].grad is not None else None
                opt=torch.optim.Adam(params,lr=.003)
                shim=SimpleNamespace(args=make_args(amp,True),device=dev,net=params,optimizer=opt,
                    scaler=SimpleNamespace(is_enabled=lambda:False),jdv2_objective_rng=owner,
                    _mc_total_steps=lambda:10)
                old=[p.detach().clone() for p in params]; skip=name.startswith('toy_skip')
                trainer._mc_optimizer_step(shim,skip=skip)
                changed=any(not torch.equal(x,y) for x,y in zip(old,params))
                assert changed!=skip and owner.successful_optimizer_updates==int(not skip)
                ledger['toy_optimizer_updates']+=int(not skip)
                row['toy']=dict(skipped=skip,changed=changed,progress=shim.args.jdv2_stage_progress)
            else: owner.finish_update(False)
            assert not owner.pending_backward
            row.update(status='PASSED',value=float(out),rng_ids_exact=True,rng_state_exact=True,
                global_cpu_cuda_eval_rng_unchanged=True,draws=owner.draw_calls,agent_draws=owner.sampled_agent_draws,
                successful_updates=owner.successful_optimizer_updates,pending=False)
            ledger['synthetic_valid_completed']+=1
            torch.save(dict(value=out.detach().cpu(),grads={k:None if r[k].grad is None else r[k].grad.cpu() for k in keys},z=z.cpu()),output/(name+'.pt'))
            budget()
        for name in json.loads((HERE/'AUDIT_PLAN.json').read_text())['illegal_cases']:
            current=name;owner=ObjectiveRNG(make_args(False,True))
            row=dict(name=name,status='ATTEMPTED');result['illegal'].append(row)
            ledger['illegal_attempted']+=1;save()
            r={k:v.to(dev).clone() for k,v in fixture('asymmetric').items()}
            state=owner.generator.get_state().clone()
            if name=='cuda_float64': r['u']=r['u'].double();r['q']=r['q'].double()
            if name=='cuda_float16': r['u']=r['u'].half();r['q']=r['q'].half()
            if name=='nonfinite_unary':r['u'][0,0]=float('nan')
            if name=='trainable_q':r['q'].requires_grad_(True)
            if name=='cross_scene':r['scenes'][0]=991
            if name=='empty_mask':r['mask'][0]=False
            try:
                if name in ('bf16_cost','cpu_ids'):
                    z=inputs['asymmetric_fp32']['z'].to(dev if name!='cpu_ids' else 'cpu')
                    c=r['c'].bfloat16() if name=='bf16_cost' else r['c']
                    ce(r['u'],r['q'],r['scenes'],r['edges'],[(0,len(c),c)],z,r['mask'],compute_backend=CUDA_BACKEND)
                else: owner.draw(r['u'],r['q'],r['scenes'],r['edges'],r['mask'])
            except (ValueError,TypeError) as exc:row.update(status='REJECTED_EXPECTED',error=str(exc))
            else:raise AssertionError('illegal input accepted: '+name)
            assert owner.draw_calls==0 and not owner.pending_backward and torch.equal(state,owner.generator.get_state())
            ledger['illegal_rejected']+=1;budget()
        result['operator_gate']='PASSED'
        real_smoke(result,ledger,batches,a,output,budget)
        result['status']='CUDA_FP32_BF16_LOSS_SMOKE_CERTIFIED_NO_TRAINING'
    except BaseException as exc:
        result.update(status='INCOMPLETE',failed_at=current,error=repr(exc),traceback=traceback.format_exc())
        if owner is not None:
            result['last_synthetic_owner']=dict(draws=owner.draw_calls,agent_draws=owner.sampled_agent_draws,
                successful_updates=owner.successful_optimizer_updates,pending=owner.pending_backward,
                state_sha256=digest(owner.generator.get_state()))
    finally:
        if started is not None:
            signal.alarm(0)
            result['gpu_seconds']=time.monotonic()-started
            if torch.cuda.is_initialized():
                result['peak_allocated']=torch.cuda.max_memory_allocated();result['peak_reserved']=torch.cuda.max_memory_reserved()
        result['policy_after']=policy()
        result['policies_unchanged']=result.get('policy_before')==result['policy_after']
        for category in ('synthetic','illegal','real'):
            for row in result[category]:
                if row['status']=='ATTEMPTED':row['status']='FAILED'
        save(); print(json.dumps({k:result.get(k) for k in ('status','ledger','gpu_seconds','peak_allocated','peak_reserved','error','traceback')},indent=2,allow_nan=False),flush=True)


def real_smoke(result,ledger,batches,a,output,budget):
    from src.models.model import GDTS
    import src.models.model as model_module
    dev=torch.device('cuda:0');a.device=str(dev);a.use_cuda=True
    a.jdv2_goal_objective='mean_energy';a.jdv2_mc_backend=CUDA_BACKEND
    ledger['real_model_construction_attempted']+=1
    net=GDTS(a,dev).to(dev);ledger['real_model_construction_completed']+=1
    ledger['real_checkpoint_read_attempted']+=1
    ckpt=torch.load(ROOT/CHECKPOINT,map_location='cpu',weights_only=False)
    ledger['real_checkpoint_read_completed']+=1
    state=ckpt['model_state_dict']
    assert ckpt.get('training_stage')=='joint_goal'
    assert ckpt.get('latent_objective')==a.jdv2_latent_objective
    shim=SimpleNamespace(args=a,net=net)
    assert ckpt.get('architecture_config')==trainer._jdv2_architecture_config(shim)
    assert ckpt.get('ablation_config')==trainer._jdv2_ablation_config(shim)
    assert ckpt.get('epoch')==13
    loaded=net.load_state_dict(state,strict=True)
    result['checkpoint']=dict(path=CHECKPOINT,sha256=CHECKPOINT_SHA,epoch=ckpt.get('epoch'),
        training_stage=ckpt.get('training_stage'),strict_missing=loaded.missing_keys,strict_unexpected=loaded.unexpected_keys,
        state_keys_shapes={k:list(v.shape) for k,v in state.items()})
    del ckpt,state
    net.configure_training_epoch(1);net.train(True);a.jdv2_stage_progress=.25
    calls=Counter(); result['native_module_calls']=calls
    handles=[]
    def count(name):
        def hook(*_):calls[name]+=1
        return hook
    for name,module in net.named_modules():handles.append(module.register_forward_pre_hook(count(name or '<root>')))
    def forbidden(*_,**__):
        ledger['forbidden_forward_attempted']+=1
        raise RuntimeError('forbidden sampler/diffusion/corrector forward')
    net.jdv2_sampler.forward=forbidden;net.diffnet.forward=forbidden;net.jdv2_corrector.forward=forbidden
    for name in ('sample','ts_sample'):
        if hasattr(net,name):setattr(net,name,forbidden)
    original_encode=net.encode
    def encode(*args,**kwargs):
        ledger['real_encode_attempted']+=1
        answer=original_encode(*args,**kwargs)
        ledger['real_encode_completed']+=1
        return answer
    net.encode=encode
    original_ce=model_module.expected_conditional_composite
    captured=[]
    def capture(*args,**kwargs):
        out=original_ce(*args,**kwargs)
        captured.append((args,out)) # only references; CPU copy after backward
        return out
    model_module.expected_conditional_composite=capture
    def state_hash():
        h=hashlib.sha256()
        for k,v in net.state_dict().items():h.update(k.encode());h.update(bytes.fromhex(digest(v)))
        return h.hexdigest()
    try:
        for idx,(selection,mode,amp) in enumerate([('edge','mean_energy',False),('edge',MC,False),('edge',MC,True),('empty',MC,True)],1):
            row=dict(name='R'+str(idx),selection=selection,objective=mode,amp=amp,status='ATTEMPTED')
            result['real'].append(row);budget()
            a.jdv2_goal_objective=mode;a.amp_enabled=amp;a.amp_dtype='bf16' if amp else 'fp32'
            owner=ObjectiveRNG(a) if mode==MC else None
            if owner:net.jdv2_objective_rng=owner
            inputs,seq=net.prepare_inputs(*batches[selection]);net.zero_grad(set_to_none=True);captured.clear()
            before=state_hash();row['before_state_sha256']=before
            ledger['real_get_loss_attempted']+=1
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
                losses=net.get_loss(inputs,seq)
                coeff=net.set_losses_coeffs();total=sum(coeff[k]*v for k,v in losses.items())
            ledger['real_get_loss_completed']+=1
            ledger['real_backward_attempted']+=1;total.backward();ledger['real_backward_completed']+=1
            row.update(losses={k:float(v.detach()) for k,v in losses.items()},coefficients=coeff,total=float(total.detach()),
                loss_dtypes={k:str(v.dtype) for k,v in losses.items()})
            assert all(v.dtype==torch.float32 and v.device==dev and torch.isfinite(v) for v in losses.values())
            gradients={};families=Counter()
            for k,p in net.named_parameters():
                if p.grad is not None:
                    assert p.requires_grad and torch.isfinite(p.grad).all(),k
                    nonzero=bool(torch.any(p.grad!=0));families[k.split('.')[0]]+=int(nonzero)
                    gradients[k]=dict(connected=True,nonzero=nonzero,max_abs=float(p.grad.abs().max()),sha256=digest(p.grad))
                else:gradients[k]=dict(connected=False,trainable=p.requires_grad)
            required=['social_encoder','jdv2_unary'] if selection=='empty' else list(net._jdv2_goal_module_names())
            for family in required:assert families[family]>0,(family,'no nonzero gradient')
            row['gradient_families']=dict(families);row['parameter_gradients']=gradients
            row['after_state_sha256']=state_hash();assert before==row['after_state_sha256']
            evidence=[];checks=[]
            for args,out in captured:
                u,q,s,e,chunks,z,mask=args
                costs=torch.cat([v.detach().cpu() for _,_,v in chunks]) if chunks else torch.empty((0,u.shape[1],u.shape[1]))
                record=dict(u=u.detach().cpu(),q=q.detach().cpu(),scenes=s.cpu(),edges=e.cpu(),C=costs,z=z.cpu(),mask=mask.cpu(),value=out.detach().cpu())
                ref=torch.stack([explicit_conditional(record['u'].double(),record['q'].double(),record['scenes'],record['edges'],costs.double(),zr,record['mask']) for zr in record['z']]).mean()
                checks.append(compare(record['value'],ref));evidence.append(record)
            if owner:
                assert len(evidence)==2 and torch.equal(evidence[0]['z'],evidence[1]['z'])
                clone=ObjectiveRNG(a);zref=torch.multinomial(evidence[0]['q'],4,replacement=True,generator=clone.generator).T.contiguous()
                assert torch.equal(zref,evidence[0]['z']) and torch.equal(clone.generator.get_state(),owner.generator.get_state())
                owner.finish_update(False)
                row['rng']=dict(draws=owner.draw_calls,agent_draws=owner.sampled_agent_draws,success=owner.successful_optimizer_updates,
                    skip=1,pending=owner.pending_backward,state_sha256=digest(owner.generator.get_state()),same_post_prior_ids=True,same_q_cpu_bridge_exact=True)
                ledger['real_objective_draws']+=owner.draw_calls;ledger['real_agent_draws']+=owner.sampled_agent_draws
                if selection=='empty':assert row['losses']['jdv2_relation_kl']==0 and row['losses']['jdv2_pl_post']==row['losses']['jdv2_pl_prior']
            torch.save(dict(effective_inputs=evidence,losses=row['losses'],gradients=gradients),output/(row['name']+'.pt'))
            row.update(status='PASSED',same_effective_input_references=checks,weights_buffers_unchanged=True)
            net.zero_grad(set_to_none=True);budget()
    except BaseException:
        if 'row' in locals():
            row['status']='FAILED'
            if owner:row['rng_at_failure']=dict(draws=owner.draw_calls,agent_draws=owner.sampled_agent_draws,pending=owner.pending_backward,success=owner.successful_optimizer_updates)
        raise
    finally:
        model_module.expected_conditional_composite=original_ce
        for h in handles:h.remove()
        for key in ('real_optimizer_updates','data_training','data_evaluation','sampler_forward','diffusion_forward','corrector_forward','full_trajectory'):
            ledger.setdefault(key,0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',action='store_true');p.add_argument('--run',action='store_true');opt=p.parse_args()
    assert opt.plan != opt.run
    if opt.plan:plan()
    else:run()

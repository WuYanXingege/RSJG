#!/usr/bin/env python3
"""Bounded fresh-joint initialization, update, reload and full-inner checks."""
import argparse,gc,json,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.p2_grouped_training import *
from src.p2_observation import atomic_json
from tools.grouped_model_smoke import arguments

def load(D):
    m=json.loads((D/'QUALIFIED_INPUT_REVIEW_MANIFEST.json').read_text());reg=Registry(m,m['manifest_hash'])
    args=arguments('joint','cpu',D);args.goal_pretrain_checkpoint=reg.manifest['parents']['goal']['path']
    return m,reg,args

def construct(args,reg,device):
    args.device=str(device);args.use_cuda=str(device).startswith('cuda')
    model,initial,non_goal,parent=initialized_model(args,reg,torch.device(device))
    parent_goal={k[len('goal_module.'):]:v for k,v in parent['model_state_dict'].items() if k.startswith('goal_module.')}
    checks=dict(parent_selected_epoch=parent['progress']['epoch_completed'],parent_sha256=reg.manifest['parents']['goal']['sha256'],
        goal_loaded_exact=state_hash(model.goal_module.state_dict())==state_hash(parent_goal),fresh_non_goal_sha256=non_goal,
        combined_initial_state_sha256=initial,coefficients=model.set_losses_coeffs())
    if checks['parent_selected_epoch']!=96 or not checks['goal_loaded_exact'] or checks['coefficients']!={'diffusion_loss':1,'goal_BCE_loss':20}:
        raise RuntimeError('joint initialization/objective contract')
    return model,checks

def gpu(D):
    q=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,capture_output=True)
    if q.returncode or q.stdout.strip():raise RuntimeError('RESOURCE_BLOCKED_DEVICE_BUSY_OR_QUERY_FAILED')
    torch.cuda.set_device(0);free,total=torch.cuda.mem_get_info();budget=min(10*1024**3,free-1024**3)
    if budget<2*1024**3:raise RuntimeError('RESOURCE_BLOCKED_MEMORY')
    torch.cuda.set_per_process_memory_fraction(budget/total,0);torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    return budget

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['fresh','cuda','reload','inner','ready'])
    ap.add_argument('--archive',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seconds',type=float,default=1200)
    a=ap.parse_args();D=a.archive.resolve();O=a.output.resolve();O.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.set_num_interop_threads(1);m,reg,args=load(D);binding=data_binding(reg)
    cfg=D/'FRESH_JOINT_CONFIG.yaml';source=source_fingerprint();parent=reg.manifest['parents']['goal']
    source_tag=source['files']['src/p2_grouped_training.py'][:12]
    if a.mode=='fresh':
        model,checks=construct(args,reg,'cpu');dest=O/('FRESH_JOINT_INITIAL_'+source_tag+'.pt')
        atomic_tensor(dict(classification='FORMAL_INITIAL_UNUPDATED',seed=3101,optimizer_updates=0,
            model_state_dict=model.state_dict(),state_sha256=checks['combined_initial_state_sha256'],
            fresh_non_goal_sha256=checks['fresh_non_goal_sha256'],goal_parent=parent),dest)
        atomic_json(dict(status='PASS',path=str(dest),sha256=file_hash(dest),updates=0,checks=checks,
            data_binding=binding,manifest_hash=m['manifest_hash'],config_sha256=file_hash(cfg),source=source),D/'JOINT_FRESH_INITIAL.json');return
    if a.mode=='ready':
        names=['JOINT_FRESH_INITIAL.json','JOINT_SMOKE_CUDA.json','JOINT_SMOKE_RELOAD.json','JOINT_INNER_SELECTION.json']
        receipts={n:json.loads((D/n).read_text()) for n in names};current=source_fingerprint()['files']
        checks={n:(v.get('status')=='PASS' and v.get('data_binding')==binding and v.get('manifest_hash')==m['manifest_hash']
            and v.get('source',{}).get('files')==current) for n,v in receipts.items()}
        fresh=receipts[names[0]];inner=receipts[names[3]];config=json.loads(cfg.read_text())
        checks.update(parent_qualified=qualify_parent(parent,reg,'goal')['progress']['epoch_completed']==96,
            initial_file=file_hash(path(fresh['path']))==fresh['sha256'],inner_complete=inner.get('packs')==391 and inner.get('denominator')==24955,
            frozen_config=config.get('p2_mode')=='joint' and config.get('phase')=='train' and config.get('num_epochs')==250
                and config.get('learning_rate')==.0001 and config.get('goal_pretrain_checkpoint')==parent['path'])
        run=subprocess.run([sys.executable,'-B','main.py','--p2_config',str(cfg),'--p2_launch_check'],cwd=ROOT,text=True,capture_output=True)
        checks['production_entry']=run.returncode==0
        result=dict(status='JOINT_TRAINING_READY' if all(checks.values()) else 'FAILED_INCOMPLETE_ACCEPTANCE',checks=checks,
            protocol_id=m['protocol_id'],data_binding=binding,manifest_hash=m['manifest_hash'],config_path=str(cfg),config_sha256=file_hash(cfg),
            goal_parent=parent,formal_training_started=False,cache='WAITING_FOR_QUALIFIED_BASE',A0='WAITING_FOR_CACHE',A1='WAITING_FOR_CACHE')
        atomic_json(result,D/'JOINT_READINESS.json');print(json.dumps(result));raise SystemExit(0 if result['status']=='JOINT_TRAINING_READY' else 2)
    budget=gpu(D);args.device='cuda:0';args.use_cuda=True
    model,checks=construct(args,reg,'cuda:0');fresh=json.loads((D/'JOINT_FRESH_INITIAL.json').read_text())
    if checks['combined_initial_state_sha256']!=fresh['checks']['combined_initial_state_sha256'] or checks['fresh_non_goal_sha256']!=fresh['checks']['fresh_non_goal_sha256']:
        raise RuntimeError('CPU/CUDA fresh initialization identity')
    opt=torch.optim.Adam(model.parameters(),lr=.0001);sched=torch.optim.lr_scheduler.ExponentialLR(opt,gamma=.99)
    smoke=O/'JOINT_SMOKE'/source_tag;smoke.mkdir(parents=True,exist_ok=True);state_path=smoke/'state.pt';replay_path=smoke/'replay.pt'
    if a.mode=='cuda':
        reader=PackReader(reg,'train');rows=[];start=time.monotonic()
        for i in range(2):
            batch,ids,_,_=reader.training(i);prepared=model.prepare_inputs(batch,ids);x,seq=prepared[0],prepared[-1]
            before={p:state_hash({k:v for k,v in model.state_dict().items() if k.startswith(p)}) for p in ('goal_module.','registrar.','diffnet.')}
            opt.zero_grad(set_to_none=True);losses=model.get_loss(x,seq);coef=model.set_losses_coeffs()
            loss=sum(losses[k]*coef[k] for k in losses);loss.backward()
            grads={p:sum(float(v.grad.detach().abs().sum()) for n,v in model.named_parameters() if n.startswith(p) and v.grad is not None) for p in before}
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize()
            changed={p:before[p]!=state_hash({k:v for k,v in model.state_dict().items() if k.startswith(p)}) for p in before}
            if not torch.isfinite(loss) or not all(x>0 for x in grads.values()) or not all(changed.values()):raise RuntimeError('joint trainable-family gradient/update')
            rows.append(dict(pack=i,loss=float(loss.detach()),components={k:float(v.detach()) for k,v in losses.items()},coefficients=coef,
                grad_l1=grads,grad_norm=float(norm),changed=changed,agents=len(reader.packs[i])))
        snap=rng_snapshot('cuda:0');payload=dict(classification='SMOKE_ONLY',model_state_dict=model.state_dict(),state_sha256=state_hash(model.state_dict()),
            optimizer=opt.state_dict(),optimizer_hash=tree_hash(opt.state_dict()),scheduler=sched.state_dict(),scheduler_hash=tree_hash(sched.state_dict()),
            rng=snap,rng_hash=tree_hash(snap),initial_state_sha256=checks['combined_initial_state_sha256'],fresh_non_goal_sha256=checks['fresh_non_goal_sha256'],goal_parent=parent)
        atomic_tensor(payload,state_path)
        batch,ids,_,_=reader.training(0);prepared=model.prepare_inputs(batch,ids);model.eval()
        with torch.no_grad():behavior=torch.stack([v.detach().float() for k,v in sorted(model.get_loss(prepared[0],prepared[-1]).items())]).cpu()
        atomic_tensor(dict(behavior=behavior),replay_path)
        atomic_json(dict(status='PASS',updates=2,rows=rows,checks=checks,checkpoint=dict(path=str(state_path),sha256=file_hash(state_path)),
            replay=dict(path=str(replay_path),sha256=file_hash(replay_path)),seconds=time.monotonic()-start,peak_reserved=torch.cuda.max_memory_reserved(),
            gpu_memory_budget=budget,data_binding=binding,manifest_hash=m['manifest_hash'],source=source),D/'JOINT_SMOKE_CUDA.json');return
    saved=torch.load(state_path,map_location='cpu',weights_only=True);model.load_state_dict(saved['model_state_dict'],strict=True);opt.load_state_dict(saved['optimizer']);sched.load_state_dict(saved['scheduler']);restore_rng(saved['rng'])
    if state_hash(model.state_dict())!=saved['state_sha256'] or tree_hash(opt.state_dict())!=saved['optimizer_hash'] or tree_hash(sched.state_dict())!=saved['scheduler_hash'] or tree_hash(rng_snapshot('cuda:0'))!=saved['rng_hash']:
        raise RuntimeError('joint reload state')
    if a.mode=='reload':
        reader=PackReader(reg,'train');batch,ids,_,_=reader.training(0);prepared=model.prepare_inputs(batch,ids);model.eval()
        with torch.no_grad():behavior=torch.stack([v.detach().float() for k,v in sorted(model.get_loss(prepared[0],prepared[-1]).items())]).cpu()
        expected=torch.load(replay_path,weights_only=True)['behavior'];diff=float((behavior-expected).abs().max())
        if not torch.equal(behavior,expected):raise RuntimeError('joint reload replay')
        atomic_json(dict(status='PASS',state_exact=True,optimizer_exact=True,scheduler_exact=True,rng_exact=True,replay_max_abs=diff,
            data_binding=binding,manifest_hash=m['manifest_hash'],source=source),D/'JOINT_SMOKE_RELOAD.json');return
    if not 0<a.seconds<=1800:raise RuntimeError('inner wall bound')
    seed_all(3101,'cuda:0');reader=PackReader(reg,'inner_valid');start=time.monotonic();metric=evaluate(model,reader,'cuda:0',False,start+a.seconds)
    if metric.denominator!=24955:raise RuntimeError('incomplete joint inner')
    atomic_json(dict(status='PASS',metric='ADE_native_pixels_P20',value=metric.value(),packs=len(reader),denominator=metric.denominator,
        optimizer_updates=0,target_read_after_prediction=True,outer_access=False,checkpoint_sha256=file_hash(state_path),
        seconds=time.monotonic()-start,peak_reserved=torch.cuda.max_memory_reserved(),data_binding=binding,manifest_hash=m['manifest_hash'],source=source),D/'JOINT_INNER_SELECTION.json')
if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();raise

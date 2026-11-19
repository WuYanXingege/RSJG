"""Frozen ZARA2 shared-world signal audit; deliberately has no training entry point.
Run from checkout: python scripts/shared_world/audit.py --workspace "$WORKSPACE" --out "$OUT"
"""
import argparse,csv,hashlib,io,json,math,os,subprocess,sys,time,fcntl
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from bcep.contracts import digest,canonical_hash
from bcep_joint.common import read,write,seed_all
from bcep_formal.data import FormalData
from bcep_v2.bank import Bank
from bcep_v2.joint import build_joint,endpoint_world_error,shared_world_terms,pair_loss
from bcep_v2.runtime import configure_cuda
from bcep_v2.workflow import recipe
from ebjd.unified import gather_candidates
BASE='468c5eddde48bdeec47c4bb3c39418352be16bb9'
ARCHIVE='4c9eede551bebdd7634ab81b7b4bd790165c6bb0'
PREFIX='docs/research/bcep_v2/formal_fivefold_top5_20261118/20261119_zara2_joint_endpoint_dev/'

def csvout(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def gitfile(name):return subprocess.check_output(['git','show',ARCHIVE+':'+PREFIX+name],cwd=ROOT)
def inventory(root):return {str(p.relative_to(root)):[p.stat().st_size,p.stat().st_mtime_ns] for p in root.rglob('*') if p.is_file()}
def tensor_hash(model):
    h=hashlib.sha256()
    for n,t in model.state_dict().items():h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def lock_free(p):
    with p.open('r') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
        fcntl.flock(f,fcntl.LOCK_UN);return True

def main(a):
    t0=time.monotonic();configure_cuda();torch.set_num_threads(1)
    up=a.workspace/'RSJG_EBJD_assets/experiments/bcep_ebjd_v2_integrated'
    trial=a.workspace/'RSJG_EBJD_assets/experiments/zara2_costproxy_stage2_20261119'
    out=a.out;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'QA.json').exists(),'completed audit must not be overwritten'
    before={k:inventory(r) for k,r in [('formal',up),('candidate_cost',trial)]}
    write(out/'LOCAL_INVENTORY_BEFORE.json',before)
    assert read(up/'STATUS.json')['status']=='PAUSED_RESUMABLE'
    assert read(trial/'STATUS.json')['status']=='SESSION_BUDGET_PAUSED'
    assert all(lock_free(p) for p in [up/'controller.lock',trial/'trial.lock'])
    protected={};receipts=[]
    for label,r,rel in [('formal',up,'zara1/head/last.pt'),('candidate_cost',trial,'joint/last.pt')]:
        ck=torch.load(r/rel,map_location='cpu',weights_only=False)
        receipts.append(dict(run=label,saved_update=ck['update'],completed_epoch=ck['completed_epoch'],epoch=ck['epoch'],cursor=ck['cursor'],status=read(r/'STATUS.json')['status'],last_sha256=digest(r/rel)))
        assert ck['update']==(7097 if label=='formal' else 1903)
        del ck
        for p in [r/rel,r/'STATUS.json',*r.rglob('TOP5.json')]:protected[label+'/'+str(p.relative_to(r))]=digest(p)
    write(out/'PAUSE_RECEIPT.json',dict(runs=receipts,locks_free=True,signals_sent=0,real_optimizer_updates=0,auto_resume=False))
    run=read(up/'FORMAL_RUN.json');assert run['commit']==BASE
    plan=read(run['input_plan']);desc=plan['folds']['zara2']['split'];assert digest(desc['path'])==desc['sha256'];split=read(desc['path'])
    cfg=read(up/'zara2/CONFIG.json');config=dict(cfg['joint'],scales=split['bridge_scales'],pair_objective='shared_world_bounded');rec=recipe(cfg,'joint')
    assert split['unit']=='m' and len(split['selection_dev_indices'])==590 and len(split['train_indices'])==2681
    banks={}
    for purpose in ['train','selection_dev']:
        path=up/'zara2/refined_banks'/f'{purpose}_2035';ident=read(path/'IDENTITY.json');banks[purpose]=Bank(path,ident,split)
        assert ident['K']==20 and ident['seed']==2035 and ident['unit']=='m'
        assert ident['base']==read(up/'zara2/density/FROZEN.json') and ident['head']==read(up/'zara2/head/FROZEN.json')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    implementation={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'bcep_v2').glob('*.py'))}
    frozen=dict(experiment='zara2_shared_world_audit_20261119',RUN_ONE_ZARA2=False,real_update_cap=0,
        base_source=BASE,code_commit=source,diagnostic_reference='7ea3a2d1a5a5d654d9e23ba4ed38f94610fc6736',
        config=config,config_sha256=canonical_hash(config),implementation_sha256=canonical_hash(implementation),implementation_files=implementation,
        original_recipe=rec,bank_sha256={k:b.hash for k,b in banks.items()},split_sha256=desc['sha256'],
        tau_m=.1,a_cap=.5,pair_weight=1.,epochs=[3,5],train_gradient_sample_rule='first train index with 2<=N<=4 and first with N>=9; both epochs; no augmentation or step',
        retention='all complete epochs plus dev H-JADE best five distinct epochs, tie earlier; best/last; full resumable state',
        future_only_decision=dict(selector='dev_H_JADE',rank1_reference=[.4084707879413993,.8016974424154072,1.3188941478729248],epoch5_reference=[.4097670185111337,.7841791655426308,1.296843430188697],pass_rule='new rank1 H-JADE <= old epoch5 and both H-JFDE(all),H-JFDE(N>=9) < old epoch5; no test; no automatic other folds'))
    write(out/'FROZEN_CONFIG.json',frozen)
    manifest=json.loads(gitfile('ARTIFACT_SHA256.json'));raw=gitfile('CHECKPOINTS.csv')
    assert hashlib.sha256(raw).hexdigest()==manifest['CHECKPOINTS.csv']
    checks={int(r['epoch']):r for r in csv.DictReader(io.StringIO(raw.decode()))}
    archived=a.workspace/'RSJG_EBJD_assets/analyses/zara2_v2_joint_endpoint_dev_20261119'
    data=FormalData(split,'scoring',verify=False);worlds=[];windows=[];gradrows=[];cps=[]
    seed_all(3201);model=build_joint(config).cuda().eval();init_hash=tensor_hash(model)
    torch.cuda.reset_peak_memory_stats();inference_seconds=0.;gradient_seconds=0.;archive_mismatches=0
    sample=[next(i for i in split['train_indices'] if 2<=len(split['members'][i]['agent_ids'])<=4),next(i for i in split['train_indices'] if len(split['members'][i]['agent_ids'])>=9)]
    for epoch in [3,5]:
        cp=up/'zara2/joint'/f'epoch_{epoch:03d}.pt';assert digest(cp)==checks[epoch]['sha256']
        ck=torch.load(cp,map_location='cpu',weights_only=False);model.load_state_dict(ck['model'],strict=True)
        assert ck['completed_epoch']==epoch and ck['identity']['dev_bank']==banks['selection_dev'].hash
        cps.append(dict(epoch=epoch,update=ck['update'],sha256=digest(cp)));del ck
        model.eval().requires_grad_(False);start=time.monotonic()
        for pos,idx in enumerate(split['selection_dev_indices']):
            assert digest(split['records'][idx]['path'])==split['records'][idx]['sha256']
            b=data.batch([idx],'selection_dev');x=b.to('cuda');g=banks['selection_dev'].batch([idx],b).cuda();n=int(x.valid.sum())
            with torch.inference_mode():
                # Archive identity binds hard outputs to the actual old coordinate checkpoints.
                p=archived/'assignments'/str(epoch)/f'{idx}.npz';assert digest(p)==manifest[f'assignments/{epoch}/{idx}.npz']
                z=np.load(p)
                if a.reuse_verified_epoch3 and epoch==3:
                    pi=torch.from_numpy(z['pi_H']).to('cuda');pi0=torch.from_numpy(z['pi_initial']).to('cuda')
                    h=gather_candidates(g,pi);initial=gather_candidates(g,pi0)
                else:
                    ctx=model.encode_context(x.observed,x.semantic_maps,x.valid)
                    h,pi,_,initial,pi0=model.coupling.assign(ctx,g,model.bridge.representation.goal_scale,straight_through=False,return_initial=True)
                    assert np.array_equal(pi.cpu().numpy(),z['pi_H']) and np.array_equal(pi0.cpu().numpy(),z['pi_initial'])
                I=endpoint_world_error(initial,x.future[:,:,-1],x.valid)[0];H=endpoint_world_error(h,x.future[:,:,-1],x.valid)[0]
                C,aa=shared_world_terms(I,H);q=(-C/.1).softmax(0);qo=(-H/.1).softmax(0)
                loss,_=pair_loss(h,x.future[:,:,-1],x.valid,initial=initial)
                oracle=float((g-x.future[:,None,:,-1]).norm(dim=-1).min(1).values.mean())
                assert torch.equal((g-x.future[:,None,:,-1]).norm(dim=-1).min(1).values,(h-x.future[:,None,:,-1]).norm(dim=-1).min(1).values)
                if n==1:assert aa.count_nonzero()==0 and torch.equal(I,H)
                eo=float(-(qo*qo.clamp_min(1e-30).log()).sum());en=float(-(q*q.clamp_min(1e-30).log()).sum())
                row=dict(epoch=epoch,index=idx,n_agents=n,I_JFDE=float(I.min()),H_JFDE=float(H.min()),feedback_gain=float(I.min()-H.min()),oracle=oracle,a_positive_fraction=float((aa>0).float().mean()),a_mean=float(aa.mean()),q_a=float((q*aa).sum()),old_q_entropy=eo,new_q_entropy=en,old_effective_worlds=math.exp(eo),new_effective_worlds=math.exp(en),initial_winner=int(I.argmin()),final_winner=int(H.argmin()),winner_switch=int(I.argmin()!=H.argmin()),objective_winner=int(C.argmin()),objective_switch_from_H=int(C.argmin()!=H.argmin()),new_pair_loss=float(loss),old_pair_loss=float(-.1*(torch.logsumexp(-H/.1,0)-math.log(20))))
                windows.append(row)
                for k in range(20):worlds.append(dict(epoch=epoch,index=idx,n_agents=n,world=k,I_m=float(I[k]),H_m=float(H[k]),a=float(aa[k]),C_m=float(C[k]),q_new=float(q[k]),q_old=float(qo[k])))
            if (pos+1)%100==0:print('epoch',epoch,'dev',pos+1,'/590',flush=True)
        torch.cuda.synchronize();inference_seconds+=time.monotonic()-start
        # Fixed real training scenes: gradient observability only. No optimizer is instantiated.
        csvout(out/'WORLD_WEIGHTS.csv',worlds);csvout(out/'WINDOWS.csv',windows)
        model.train().requires_grad_(True);start=time.monotonic()
        for idx in sample:
            assert digest(split['records'][idx]['path'])==split['records'][idx]['sha256']
            b=data.batch([idx],'train');x=b.to('cuda');g=banks['train'].batch([idx],b).cuda();g.requires_grad_(True)
            ctx=model.encode_context(x.observed,x.semantic_maps,x.valid)
            h,pi,_,initial,pi0=model.coupling.assign(ctx,g,model.bridge.representation.goal_scale,straight_through=True,return_initial=True)
            I=endpoint_world_error(initial,x.future[:,:,-1],x.valid);H=endpoint_world_error(h,x.future[:,:,-1],x.valid)
            C,aa=shared_world_terms(I,H);q=(-C.detach()/.1).softmax(1)
            old,_=pair_loss(h,x.future[:,:,-1],x.valid);new,_=pair_loss(h,x.future[:,:,-1],x.valid,initial=initial)
            pars=list(model.named_parameters());params=[p for n,p in pars]
            grads={}
            for name,value in [('old',old),('new',new),('initial_component',(q*aa*I).sum()),('final_component',(q*(1-aa)*H).sum())]:
                v=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True)
                grads[name]=torch.cat([(z if z is not None else torch.zeros_like(p)).detach().flatten() for z,p in zip(v,params)])
            assert torch.autograd.grad(new,g,allow_unused=True)[0] is None
            assert torch.allclose(grads['new'],grads['initial_component']+grads['final_component'],atol=1e-6,rtol=1e-5)
            gradrows.append(dict(epoch=epoch,index=idx,n_agents=int(x.valid.sum()),q_a=float((q*aa).sum()),old_grad_norm=float(grads['old'].norm()),new_grad_norm=float(grads['new'].norm()),initial_component_norm=float(grads['initial_component'].norm()),final_component_norm=float(grads['final_component'].norm()),cosine_old_new=float(torch.nn.functional.cosine_similarity(grads['old'],grads['new'],dim=0)),decomposition_max_abs=float((grads['new']-grads['initial_component']-grads['final_component']).abs().max())))
            del ctx,h,initial,grads,params,pars,g,new,old,C,I,H
        torch.cuda.synchronize();gradient_seconds+=time.monotonic()-start
    csvout(out/'WORLD_WEIGHTS.csv',worlds);csvout(out/'WINDOWS.csv',windows);csvout(out/'GRADIENT_OBSERVABILITY.csv',gradrows)
    summary=[];dist=[]
    for epoch in [3,5]:
        for label,pred in [('all',lambda n:True),('single',lambda n:n==1),('multi',lambda n:n>1),('N_ge_9',lambda n:n>=9)]:
            rr=[r for r in windows if r['epoch']==epoch and pred(r['n_agents'])];ww=[r for r in worlds if r['epoch']==epoch and pred(r['n_agents'])]
            if label in ['all','N_ge_9']:assert len(rr)==(590 if label=='all' else 127)
            s=dict(epoch=epoch,group=label,windows=len(rr),worlds=len(ww))
            for k in ['I_JFDE','H_JFDE','feedback_gain','oracle','a_positive_fraction','a_mean','q_a','old_q_entropy','new_q_entropy','old_effective_worlds','new_effective_worlds','winner_switch','objective_switch_from_H','old_pair_loss','new_pair_loss']:s[k]=sum(r[k] for r in rr)/len(rr)
            summary.append(s);av=np.array([r['a'] for r in ww]);qa=np.array([r['q_a'] for r in rr])
            d=dict(epoch=epoch,group=label,a_zero_count=int((av==0).sum()),a_cap_count=int((av==.5).sum()),world_count=len(av),window_count=len(qa))
            for name,v in [('a',av),('q_a',qa)]:
                for pct in [0,25,50,75,90,95,100]:d[f'{name}_p{pct}']=float(np.percentile(v,pct))
            dist.append(d)
    csvout(out/'GROUPS.csv',summary);csvout(out/'DISTRIBUTIONS.csv',dist);write(out/'CHECKPOINTS.json',cps)
    for k,r in [('formal',up),('candidate_cost',trial)]:assert inventory(r)==before[k]
    for k,h in protected.items():
        label,rel=k.split('/',1);assert digest((up if label=='formal' else trial)/rel)==h
    assert all(lock_free(p) for p in [up/'controller.lock',trial/'trial.lock'])
    write(out/'PRESERVATION.json',dict(status='PASS',protected_sha256=protected,whole_run_inventory_unchanged=True,locks_free=True))
    write(out/'QA.json',dict(status='PASS',real_optimizer_updates=0,real_dev_windows_per_epoch=590,real_dev_forward_calls=590 if a.reuse_verified_epoch3 else 1180,reused_archive_windows=590 if a.reuse_verified_epoch3 else 0,real_train_gradient_windows_per_epoch=2,real_train_gradient_forward_calls=4,epochs=[3,5],heldout_read=False,augmentation=False,real_DDIM_calls=0,archive_hard_assignments_exact=True,per_agent_minFDE_exact=True,single_a_exact_zero=True,initial_tensor_sha256=init_hash,dev_forward_seconds=inference_seconds,gradient_seconds=gradient_seconds,total_seconds=time.monotonic()-t0,gpu=torch.cuda.get_device_name(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),torch=torch.__version__,code_commit=source))
    write(out/'STATUS.json',dict(status='COMPLETE_NO_TRAIN',real_optimizer_updates=0,auto_resume=False))
    print('AUDIT PASS',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--reuse-verified-epoch3',action='store_true',help='Reuse SHA-verified old pi after an already completed epoch-3 frozen pass');main(p.parse_args())

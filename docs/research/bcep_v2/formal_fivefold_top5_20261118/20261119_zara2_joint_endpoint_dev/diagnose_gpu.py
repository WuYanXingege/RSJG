"""Read-only dev endpoint/assignment mechanism audit; no training entrypoints."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']='1'
import sys,json,csv,math,itertools,subprocess,fcntl
from pathlib import Path
import numpy as np
import torch
BASE=Path(__file__).resolve().parents[3]
SRC=BASE/'RSJG_BCEP_integrated_v2_20261118'
sys.path.insert(0,str(SRC))
from bcep.contracts import digest,canonical_hash
from bcep_joint.common import read,write,seed_all
from bcep_formal.data import FormalData
from bcep_v2.formal_manifest import contract,stage_identity,write_csv
from bcep_v2.bank import Bank,bank_identity
from bcep_v2.joint import build_joint,pair_loss
from ebjd.unified import LearnableJointCouplingHead,hard_assignment_indices,gather_candidates,permutation_matrix
OUT=Path(__file__).resolve().parent
ROOT=BASE/'RSJG_EBJD_assets/experiments/bcep_ebjd_v2_integrated'
P='selection_dev'
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)

def inventory():
    return {str(p.relative_to(ROOT)):[p.stat().st_size,p.stat().st_mtime_ns] for p in ROOT.rglob('*') if p.is_file()}

def softcost(h,y):
    e=(h-y[None]).norm(dim=-1).mean(-1)
    return -.1*(torch.logsumexp(-e/.1,0)-math.log(len(h)))

def swaps(epoch,idx,g,h,pi,y,agent_ids):
    # All values frozen; autograd only on local endpoint coordinates, never parameters.
    z=h.double().detach().requires_grad_(True); yy=y.double()
    loss=softcost(z,yy); grad,=torch.autograd.grad(loss,z)
    fp=h[None].detach().requires_grad_(True)
    lf,_=pair_loss(fp,y[None],torch.ones(1,len(y),dtype=torch.bool))
    gf,=torch.autograd.grad(lf,fp)
    assert torch.allclose(gf[0].double(),grad,atol=2e-6,rtol=2e-4)
    rng=np.random.default_rng(1192035+idx)
    agents=sorted(rng.choice(len(y),min(2,len(y)),replace=False).tolist())
    pairs=list(itertools.combinations(range(20),2)); chosen=rng.choice(len(pairs),20,replace=False)
    rows=[]
    # Equivalent dL/dP[n,k,j]=<dL/dX[k,n],G[j,n]>, contracted with P_swap-P.
    grad_p=torch.einsum('knd,jnd->nkj',grad,g.double())
    for a in agents:
        for pair_idx in chosen:
            u,v=pairs[int(pair_idx)]; j,l=int(pi[u,a]),int(pi[v,a])
            zz=z.detach().clone();zz[[u,v],a]=zz[[v,u],a]
            d=zz-z.detach(); actual=float(softcost(zz,yy)-loss.detach()); proxy=float((grad*d).sum())
            pproxy=float(grad_p[a,u,l]-grad_p[a,u,j]+grad_p[a,v,j]-grad_p[a,v,l])
            assert abs(proxy-pproxy)<1e-10
            eps=1e-5
            fd=float((softcost(z.detach()+eps*d,yy)-softcost(z.detach()-eps*d,yy))/(2*eps))
            assert abs(fd-proxy)<1e-6
            rows.append(dict(epoch=epoch,index=idx,agent_position=a,agent_id=agent_ids[a],world_u=u,world_v=v,candidate_u=j,candidate_v=l,true_delta_m=actual,coordinate_proxy_direction_m=proxy,local_central_difference=fd,near_zero_true=abs(actual)<=1e-8,near_zero_proxy=abs(proxy)<=1e-8,informative=abs(actual)>1e-8 and abs(proxy)>1e-8,sign_disagreement=actual*proxy<0))
    return rows

from bcep_v2.runtime import configure_cuda
configure_cuda()
before=inventory();write(OUT/'INPUT_INVENTORY.json',before)
run=read(ROOT/'FORMAL_RUN.json');assert read(ROOT/'STATUS.json')['status']=='PAUSED_RESUMABLE'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=SRC,text=True).strip()==run['commit']
ctx=contract(ROOT,'zara2');cfg,split,folder,geo,ident=ctx
stages={s:read(folder/'receipts'/f'{s}.json') for s in ['density','base-cache','head','banks','joint']}
identity=stage_identity(ctx,'joint',stages)
bid=bank_identity(split,P,2035,geo,ident['sources'],stages['density']['selected'],stages['head']['selected'])
bank=Bank(folder/'refined_banks/selection_dev_2035',bid,split)
ids=split[P+'_indices'];assert len(ids)==590
for i in ids:assert digest(split['records'][i]['path'])==split['records'][i]['sha256']
data=FormalData(split,'scoring',verify=False,cache_bytes=8*1024*1024)
multis=[i for i in ids if len(split['members'][i]['agent_ids'])>1]
sampled=set(multis[j] for j in np.linspace(0,len(multis)-1,32,dtype=int))
write(OUT/'IDENTITY.json',dict(run=run['experiment'],source_commit=run['commit'],source=ident,bank_hash=bank.hash,bank_identity=bid,windows=len(ids),valid_agent_occurrences=sum(len(split['members'][i]['agent_ids']) for i in ids),membership_hash=canonical_hash([split['members'][i] for i in ids]),optimizer_updates=0,device='cuda_fp32_forward_cpu_gradient_audit',implementation_sha256=digest(Path(__file__)),gradient_sampling=dict(windows=sorted(sampled),rule='32 evenly spaced positions in official multi-agent dev window order; at most 2 uniform agents and 20 uniform unordered world pairs without replacement, RNG 1192035+window_index, same swaps across epochs',temperature_m=.1,near_zero_threshold_m=1e-8,coordinate_proxy='gradient wrt final assignment matrix P before Sinkhorn Jacobian; finite swap vs local directional derivative; other people fixed')))
receipt=read(OUT/'STOP_RECEIPT.json');p=ROOT/'zara1/head/last.pt';s=torch.load(p,map_location='cpu',weights_only=False)
receipt['recovery_snapshot']=dict(path=str(p),sha256=digest(p),**{k:s[k] for k in ['schema','status','epoch','completed_epoch','update','cursor','stale','stop_reason']},keys=list(s),rng_keys=list(s['rng']),order_length=len(s['order']),optimizer_state_entries=len(s['optimizer']['state']),scheduler_present=s['scheduler'] is not None)
receipt['diagnostic_optimizer_updates']=0;receipt['progress_log_is_periodic_not_exact_at_signal']=True
receipt['progress_to_saved_update_delta']=s['update']-json.loads(receipt['status_cli_before'])['progress']['update']
receipt['queue_resume_invoked']=False
write(OUT/'STOP_RECEIPT.json',receipt);del s
history=read(folder/'joint/EPOCH_METRICS.json');history={r['epoch']:r for r in history}
checks=[];window_rows=[];agent_rows=[];swap_rows=[];invariants=[]
for epoch in [26,3,5]:
    p=folder/'joint'/f'epoch_{epoch:03d}.pt'
    c=dict(epoch=epoch,path=str(p),exists=p.exists());checks.append(c)
    if not p.exists():continue
    s=torch.load(p,map_location='cpu',weights_only=False)
    assert s['identity']==identity and s['completed_epoch']==epoch and s['cursor']==0 and s['order'] is None
    c.update(sha256=digest(p),update=s['update'],completed_epoch=s['completed_epoch'],archive_JFDE=history[epoch]['metrics']['JFDE'])
    joint_cfg=dict(cfg['joint'],scales=split['bridge_scales']);seed_all(joint_cfg['downstream_seed'])
    model=build_joint(joint_cfg);model.load_state_dict(s['model'],strict=True);model.eval().requires_grad_(False).to("cuda");del s
    (OUT/'assignments'/str(epoch)).mkdir(parents=True,exist_ok=True)
    for pos,i in enumerate(ids):
        x=data.batch([i],P);n=int(x.valid.sum());assert n==x.valid.shape[1]
        g=bank.batch([i],x).to("cuda");x=x.to("cuda");y=x.future[0,:,-1]
        with torch.no_grad():
            context=model.encode_context(x.observed,x.semantic_maps,x.valid)
            raw=LearnableJointCouplingHead.forward(model.coupling,context,g,model.bridge.representation.goal_scale)
            pi0=hard_assignment_indices(raw,x.valid);initial=gather_candidates(g,pi0)
            h,pi,scores=model.coupling.assign(context,g,model.bridge.representation.goal_scale,straight_through=False)
            assert torch.equal(h,gather_candidates(g,pi))
        g=g.cpu();h=h.cpu();pi=pi.cpu();pi0=pi0.cpu();initial=initial.cpu();y=y.cpu()
        errors=(g[0]-y[None]).norm(dim=-1); nearest=errors.sort(dim=0).values
        oracle=float(nearest[0].mean())
        rng=np.random.default_rng(2035+i)
        shuffled=[]
        for repeat in range(20):
            pir=torch.tensor(np.stack([rng.permutation(20) for _ in range(n)],axis=1))[None]
            gr=gather_candidates(g,pir)[0]
            shuffled.append(float((gr-y[None]).norm(dim=-1).mean(-1).min()))
        # Random common world relabeling cannot change matching or JFDE.
        common=torch.tensor(rng.permutation(20));uniform=h[:,common]
        values={'G':float(errors.mean(-1).min()),'initial_Hungarian':float((initial[0]-y[None]).norm(dim=-1).mean(-1).min()),'H':float((h[0]-y[None]).norm(dim=-1).mean(-1).min()),'independent_shuffle':float(np.mean(shuffled)),'oracle':oracle,'H_global_relabel':float((uniform[0]-y[None]).norm(dim=-1).mean(-1).min())}
        assert abs(values['H']-values['H_global_relabel'])<1e-7
        if n==1:assert max(values.values())-min(values.values())<1e-7
        assert min(values.values())>=oracle-1e-6
        # Relative-to-agent-0 normalization removes global world labels.
        rel0=pi0[0][torch.argsort(pi0[0,:,0])];relh=pi[0][torch.argsort(pi[0,:,0])]
        invariants.append(dict(epoch=epoch,index=i,n_agents=n,initial_vs_H_raw_entries=int((pi0!=pi).sum()),initial_vs_H_relative_entries=int((rel0!=relh).sum()),global_relabel_abs_delta=abs(values['H']-values['H_global_relabel'])))
        for method,value in values.items():window_rows.append(dict(epoch=epoch,index=i,n_agents=n,subset='single' if n==1 else 'multi',method=method,JFDE_m=value))
        for repeat,value in enumerate(shuffled):window_rows.append(dict(epoch=epoch,index=i,n_agents=n,subset='single' if n==1 else 'multi',method=f'shuffle_repeat_{repeat:02d}',JFDE_m=value))
        if epoch==26:
            for a in range(n):agent_rows.append(dict(index=i,agent_position=a,agent_id=x.agent_ids[0][a],n_agents=n,first_m=float(nearest[0,a]),second_m=float(nearest[1,a]),fifth_m=float(nearest[4,a])))
        np.savez_compressed(OUT/'assignments'/str(epoch)/f'{i}.npz',index=i,pi_initial=pi0.numpy(),pi_H=pi.numpy())
        if i in sampled:swap_rows.extend(swaps(epoch,i,g[0],h[0],pi[0],y,x.agent_ids[0]))
        if (pos+1)%100==0:print(f'epoch {epoch}: {pos+1}/{len(ids)}',flush=True)
    mean=np.mean([r['JFDE_m'] for r in window_rows if r['epoch']==epoch and r['method']=='H'])
    c['reproduced_H_JFDE']=float(mean);c['archive_abs_delta']=abs(mean-c['archive_JFDE']);assert c['archive_abs_delta']<1e-6
    if epoch==26:assert abs(np.mean([r['JFDE_m'] for r in window_rows if r['epoch']==26 and r['method']=='oracle'])-.33942601943420153)<1e-7
    print('verified',c,flush=True)
    write_csv(OUT/'CHECKPOINTS.csv',checks)
    write_csv(OUT/'WINDOW_METRICS.csv',window_rows);write_csv(OUT/'AGENT_NEAREST.csv',agent_rows)
    write_csv(OUT/'SWAPS.csv',swap_rows);write_csv(OUT/'PERMUTATION_CHECKS.csv',invariants)
summary=[]
for epoch in [26,3,5]:
 for subset in ['all','single','multi']:
  for method in ['G','initial_Hungarian','H','independent_shuffle','oracle','H_global_relabel']:
   rows=[r for r in window_rows if r['epoch']==epoch and r['method']==method and (subset=='all' or r['subset']==subset)]
   if rows:summary.append(dict(epoch=epoch,subset=subset,method=method,windows=len(rows),agents=sum(r['n_agents'] for r in rows),JFDE_m=float(np.mean([r['JFDE_m'] for r in rows]))))
write_csv(OUT/'SUMMARY.csv',summary)
near=[]
for subset in ['all','single','multi']:
 a=[r for r in agent_rows if subset=='all' or (r['n_agents']==1)==(subset=='single')]
 for rank,key in [(1,'first_m'),(2,'second_m'),(5,'fifth_m')]:
  w={}
  for r in a:w.setdefault(r['index'],[]).append(r[key])
  near.append(dict(subset=subset,rank=rank,windows=len(w),agents=len(a),agent_micro_m=float(np.mean([r[key] for r in a])),scene_macro_m=float(np.mean([np.mean(v) for v in w.values()]))))
write_csv(OUT/'NEAREST_SUMMARY.csv',near)
sums=[]
for epoch in [26,3,5]:
 rows=[r for r in swap_rows if r['epoch']==epoch]
 for tol in [1e-8,1e-6]:
  informative=[r for r in rows if abs(r['true_delta_m'])>tol and abs(r['coordinate_proxy_direction_m'])>tol]
  disagree=[r for r in informative if r['sign_disagreement']]
  sums.append(dict(epoch=epoch,threshold_m=tol,sampled_windows=len({r['index'] for r in rows}),valid_swaps=len(rows),near_zero_true=sum(abs(r['true_delta_m'])<=tol for r in rows),near_zero_proxy=sum(abs(r['coordinate_proxy_direction_m'])<=tol for r in rows),informative=len(informative),disagreements=len(disagree),disagreement_rate=len(disagree)/len(informative)))
write_csv(OUT/'GRADIENT_SUMMARY.csv',sums)
after=inventory();assert before==after,'upstream assets changed'
write(OUT/'RESULT.json',dict(status='COMPLETE',optimizer_updates=0,auto_resume=False,input_inventory_unchanged=True,protected_file_count=len(before),checkpoints=checks,summary=summary,nearest=near,gradient=sums))
print('COMPLETE',flush=True)

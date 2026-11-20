"""Bounded synthetic GPU acceptance; never imports a real dataset or starts a queue."""
import argparse,copy,gc,hashlib,json,os,subprocess,sys,time,types,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from bcep_v3.model import build,tensor_sha,NEW_PREFIXES
from bcep_v3.fixtures import joint_config,SyntheticData,SyntheticBank
from bcep_v3.stage2 import callbacks
from bcep_joint.common import seed_all
from bcep_joint.checks import equal
from bcep_joint.inference import generate,keyed_noise
from ebjd.unified import gather_candidates
from bcep_v2.provider import Provider
from bcep_v2.state import run_stage,optimizer
from bcep_v2.runtime import configure_cuda
from bcep_v2.joint import assignment
BASE='468c5eddde48bdeec47c4bb3c39418352be16bb9'
OUT=ROOT.parent/'RSJG_EBJD_assets/experiments/bcep_v3_architecture_audit_20261120'
PUBLIC=ROOT/'docs/research/bcep_v3/architecture_20261120'


def write(path,obj):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(1048576),b''):h.update(x)
    return h.hexdigest()


def pristine_coupling():
    source=subprocess.check_output(['git','show',BASE+':bcep_v2/joint.py'],cwd=ROOT,text=True)
    m=types.ModuleType('bcep_v2._pristine_joint');m.__package__='bcep_v2';exec(compile(source,'anchor/bcep_v2/joint.py','exec'),m.__dict__)
    return m.FeedbackCoupling()


class Budget:
    def __init__(self):
        self.path=OUT/'GPU_UPDATE_LEDGER.json'
        self.rows=json.loads(self.path.read_text()) if self.path.exists() else []
        self.label='unset'
    def install(self):
        self.original=torch.optim.AdamW.step
        def step(opt,*args,**kwargs):
            if len(self.rows)>=32:raise RuntimeError('synthetic GPU update cap exhausted')
            # Record BEFORE step: a crash can only overcount, never hide an update.
            self.rows.append(dict(ordinal=len(self.rows)+1,section=self.label,time_unix=time.time(),synthetic_only=True))
            write(self.path,self.rows)
            return self.original(opt,*args,**kwargs)
        torch.optim.AdamW.step=step
    def close(self):torch.optim.AdamW.step=self.original


def audit_migration(config):
    anchor=OUT.parent/'bcep_ebjd_v2_integrated/zara2'
    top=json.loads((anchor/'joint/TOP5.json').read_text())[0];path=anchor/'joint'/top['file']
    assert top['epoch']==26 and sha(path)==top['sha256']
    state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['identity']['fold']=='zara2' and state['identity']['stage']=='joint'
    old,_=build(config,variant='v2_compatible',v2_state=state['model'])
    legacy=pristine_coupling();legacy.load_state_dict(old.coupling.state_dict(),strict=True);old.coupling=legacy
    new,origin=build(config,v2_state=state['model'])
    origin.update(checkpoint_file='zara2/joint/'+top['file'],checkpoint_sha256=top['sha256'],base_source=BASE)
    origin['parameters']={v:sum(p.numel() for p in m.parameters()) for v,m in [('v2',old),('v3',new)]}
    origin['new_parameter_groups']={prefix:sum(p.numel() for n,p in new.named_parameters() if n.startswith(prefix)) for prefix in NEW_PREFIXES}
    write(PUBLIC/'INITIALIZATION_ORIGIN.json',origin)
    provider=Provider()
    for stage,module in [('density',provider.base),('head',provider.head)]:
        row=json.loads((anchor/stage/'TOP5.json').read_text())[0];p=anchor/stage/row['file'];assert sha(p)==row['sha256']
        ck=torch.load(p,map_location='cpu',weights_only=False);module.load_state_dict(ck['model'],strict=True)
    provider.eval().requires_grad_(False)
    receipt={k:tensor_sha(v) for k,v in provider.state_dict().items()}
    write(OUT/'PROVIDER_TENSORS_BEFORE.json',receipt)
    return old,new,provider


def equivalence(old,new):
    old.cuda().eval();new.cuda().eval();rows=[]
    for n in [1,2,9,57]:
        d=SyntheticData([n],256);batch=d.batch([0],'selection_dev');g=SyntheticBank().batch([0],batch).cuda();x=batch.to('cuda')
        with torch.no_grad():
            co=old.encode_context(x.observed,x.semantic_maps,x.valid)
            cn=new.encode_context(x.observed,x.semantic_maps,x.valid)
            s,_=old.coupling.forward_features(co,g,old.bridge.representation.goal_scale)
            pi0,_,_=assignment(s,x.valid,False)
            a=old.coupling.assign(co,g,old.bridge.representation.goal_scale,straight_through=False)
            b=new.coupling.assign(cn,g,new.bridge.representation.goal_scale,straight_through=False,return_details=True)
            assert torch.equal(pi0,b[3]['initial_pi']) and torch.equal(a[1],b[1])
            assert torch.equal(a[0],b[0]);torch.testing.assert_close(a[2],b[2],atol=0,rtol=0)
            z=keyed_noise(x.scene_ids,x.agent_ids,x.valid,2035).cuda()
            assert torch.equal(gather_candidates(z,a[1]),gather_candidates(z,b[1]))
            before=0
            def count(*args):
                nonlocal before;before+=1
            handle=new.coupling.register_forward_hook(count)
            # generate calls assign directly; wrap it to count the single pairing.
            calls=[];original=new.coupling.assign
            def counted(*a,**kw):calls.append(1);return original(*a,**kw)
            new.coupling.assign=counted
            try:
                ga,ha,_,ra=generate(old,x.observed,x.semantic_maps,x.valid,x.scene_ids,x.agent_ids,g,2035)
                gb,hb,_,rb=generate(new,x.observed,x.semantic_maps,x.valid,x.scene_ids,x.agent_ids,g,2035)
            finally:new.coupling.assign=original;handle.remove()
            assert len(calls)==1
            torch.testing.assert_close(ga,gb,atol=2e-6,rtol=2e-6);torch.testing.assert_close(ha,hb,atol=2e-6,rtol=2e-6)
            marg=(g-x.future[:,None,:,-1]).norm(dim=-1).min(1).values
            torch.testing.assert_close(marg,(b[0]-x.future[:,None,:,-1]).norm(dim=-1).min(1).values,atol=0,rtol=0)
            rows.append(dict(agents=n,K=20,DDIM_steps=20,map_pixels=256,old_feedback_nonzero=True,
                score_max_abs=float((a[2]-b[2]).abs().max()),G_max_abs=float((ga-gb).abs().max()),H_max_abs=float((ha-hb).abs().max()),
                initial_final_pi_exact=True,keyed_noise_exact=True,candidate_marginal_exact=True,pairing_calls=1))
        print('equivalence',n,rows[-1],flush=True)
    # The inherited decoder-energy option still works, but frozen experiment uses legacy.
    with torch.no_grad():
        d=SyntheticData([2],32);x=d.batch([0],'selection_dev').to('cuda');g=SyntheticBank().batch([0],d.batch([0],'selection_dev')).cuda()
        c=old.encode_context(x.observed,x.semantic_maps,x.valid);z=torch.randn(1,2,2,12,2,device='cuda')
        for m in (old,new):m.bridge.denoiser.ctf_feedback_basis='decoder_energy';m.bridge.denoiser.set_ctf_feedback_eta(1.)
        a=old.bridge.denoiser(z,.5,c,proposal_goals=g[:,:2],proposal_eta=0.)
        b=new.bridge.denoiser(z,.5,c,proposal_goals=g[:,:2],proposal_eta=0.)
        for aa,bb in zip(a,b):torch.testing.assert_close(aa,bb,atol=2e-6,rtol=2e-6)
        for m in (old,new):m.bridge.denoiser.ctf_feedback_basis='legacy';m.bridge.denoiser.set_ctf_feedback_eta(0.)
    return dict(status='PASS',rows=rows,decoder_energy_equivalence=True)


def restart(config,budget):
    data=SyntheticData([2,3],32);bank=SyntheticBank();grad_rows=[]
    c=dict(config,batch_scenes=2,gradient_accumulation=1,group_max_slots=4)
    recipe=dict(seed=3201,lr=1e-4,weight_decay=1e-4,clip=5.,logical_scenes=2,max_epochs=6,minimum=6,patience=15,scheduler_horizon=100)
    identity=dict(stage='joint',fold='TEST_ONLY',variant='v3',real_updates=0)
    for arm,limit,resume in [('continuous',4,False),('resumed',2,False),('resumed',4,True)]:
        budget.label='restart_'+arm+('_resume' if resume else '')
        seed_all(3201);m,_=build(c);m.cuda();bw,val=callbacks(m,data,'cuda',c,bank,bank)
        def backward(*args):
            result=bw(*args)
            row={}
            tensors={n:float(p.grad.abs().sum()) if p.grad is not None else None for n,p in m.named_parameters() if n.startswith(NEW_PREFIXES)}
            for prefix in NEW_PREFIXES:
                named=[(n,p) for n,p in m.named_parameters() if n.startswith(prefix)]
                assert all(p.grad is None or torch.isfinite(p.grad).all() for n,p in named)
                row[prefix]=sum(float(p.grad.abs().sum()) for n,p in named if p.grad is not None)
            grad_rows.append(dict(arm=arm,local_step=len(grad_rows)+1,norm_l1=row,tensor_gradient_l1=tensors))
            return result
        folder=OUT/('final_'+arm)
        run_stage(folder,identity,recipe,m,[0,1],backward,val,limit,resume=resume)
        if arm=='continuous':
            for prefix in NEW_PREFIXES:
                assert any(row['norm_l1'][prefix]>0 for row in grad_rows),prefix
            write(OUT/'POST_UPDATE_NEW_TENSORS.json',{n:tensor_sha(t) for n,t in m.state_dict().items() if n.startswith(NEW_PREFIXES)})
        del m;gc.collect();torch.cuda.empty_cache()
    a=torch.load(OUT/'final_continuous/last.pt',map_location='cpu',weights_only=False)
    b=torch.load(OUT/'final_resumed/last.pt',map_location='cpu',weights_only=False)
    for key in ['model','optimizer','scheduler','rng','update','cursor','order','completed_epoch','stale','history']:
        equal(a[key],b[key])
    for arm in ['continuous','resumed']:
        top=json.loads((OUT/('final_'+arm)/'TOP5.json').read_text());assert len(top)==4 and len({x['epoch'] for x in top})==4
        for row in top:assert sha(OUT/('final_'+arm)/row['file'])==row['sha256']
    return dict(status='PASS',continuous_updates=4,split_updates=[2,2],all_state_exact=True,grouped_backward_real_implementation=True,gradients=grad_rows)


def benchmark(config,budget,reuse_v2=None,repeats=2):
    rows=[] if reuse_v2 is None else [r for r in json.loads(Path(reuse_v2).read_text())['rows'] if r['variant']=='v2_compatible']
    for counts in [[1],[8,8,8,8],[57]]:
        for variant in (['v2_compatible','v3'] if reuse_v2 is None else ['v3']):
            seed_all(3210);c=dict(config,batch_scenes=4,gradient_accumulation=1);m,_=build(c,variant=variant);m.cuda().train()
            data=SyntheticData(counts,256);bank=SyntheticBank();bw,_=callbacks(m,data,'cuda',c,bank,bank)
            opt=optimizer(m,1e-4);order=torch.Generator().manual_seed(1);views=torch.Generator().manual_seed(2)
            times=[];peaks=[];budget.label='benchmark_'+variant+'_'+'x'.join(map(str,counts))
            for _ in range(repeats):
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                opt.zero_grad(set_to_none=True);bw(list(range(len(counts))),order,views)
                torch.nn.utils.clip_grad_norm_(m.parameters(),5.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize()
                times.append(time.perf_counter()-start);peaks.append(torch.cuda.max_memory_allocated())
            row=dict(variant=variant,counts=counts,map_pixels=256,full_step_seconds=times,peak_allocated_bytes=peaks,
                     timed_updates=len(times),includes_cpu_augmentation=True,group_max_slots=32,gradient_accumulation=1)
            rows.append(row);write(PUBLIC/'BENCHMARK.json',dict(status='IN_PROGRESS',rows=rows));print('benchmark',row,flush=True)
            del m,opt;gc.collect();torch.cuda.empty_cache()
    return dict(status='PASS',device=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
                precision='FP32',deterministic=True,SDPA='math',rows=rows,
                baseline_reused=reuse_v2 is not None,limitations='one or two full synthetic updates per shape (see timed_updates); noisy estimates; unequal repeat counts when baseline reused; no real-data throughput guarantee; N57 exceeds physical group threshold as original single-scene rule')


def main():
    p=argparse.ArgumentParser();p.add_argument('--section',choices=['migration','restart','benchmark'],required=True);p.add_argument('--reuse-v2-benchmark',type=Path);p.add_argument('--benchmark-repeats',type=int,choices=[1,2],default=2);a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)
    fixed=json.loads((ROOT/'configs/bcep_ebjd_v3/RESOLVED_CONFIG.json').read_text());assert fixed['allow_real_training'] is False
    configure_cuda();torch.set_num_threads(1);seed_all(3201);config=joint_config();budget=Budget()
    with (OUT/'audit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);budget.install()
        try:
            if a.section=='migration':
                old,new,provider=audit_migration(config);r=equivalence(old,new)
                assert all(tensor_sha(v)==json.loads((OUT/'PROVIDER_TENSORS_BEFORE.json').read_text())[k] for k,v in provider.state_dict().items())
                assert all(p.grad is None and not p.requires_grad for p in provider.parameters())
                r['provider_frozen_unchanged']=True;write(PUBLIC/'MIGRATION_EQUIVALENCE.json',r)
            elif a.section=='restart':write(PUBLIC/'RESTART_GRADIENT.json',restart(config,budget))
            else:write(PUBLIC/'BENCHMARK.json',benchmark(config,budget,a.reuse_v2_benchmark,a.benchmark_repeats))
        finally:budget.close()
    print('DONE',a.section,'physical_synthetic_updates',len(budget.rows),'real_updates',0,flush=True)
if __name__=='__main__':main()

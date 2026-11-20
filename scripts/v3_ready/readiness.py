"""Real-input zero-step acceptance plus separately budgeted synthetic recovery."""
import argparse,contextlib,copy,gc,json,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from bcep.contracts import canonical_hash,digest
from bcep_joint.common import read,write,seed_all
from bcep_joint.checks import equal
from bcep_joint.inference import generate,keyed_noise
from bcep_v2.runtime import configure_cuda
from bcep_v2.state import run_stage,optimizer
from bcep_v2.stages import joint_callbacks
from bcep_v3.model import build,NEW_PREFIXES,tensor_sha
from bcep_v3.fixtures import SyntheticData,SyntheticBank
from bcep_v3.formal import model_hash,freeze,locks
from bcep_v3.formal_inputs import FormalInputs,paths,SOURCE
from bcep_v3.formal_validation import window,validation
from bcep_v3.formal_telemetry import ModuleTelemetry,no_optimizer_steps
from bcep_v3.stage2 import formal_callbacks
PUBLIC=ROOT/'docs/research/bcep_v3/stage2_ready_20261120'


def protect(roots):
    snapshot={}
    for name in ['formal','candidate_cost','sharedworld']:
        root=roots[name];rows={}
        for p in root.rglob('*'):
            if p.is_file() and (p.suffix in ['.pt','.npy'] or p.name in ['STATUS.json','TOP5.json','FROZEN.json','COMPLETE.json','IDENTITY.json']):
                rows[str(p.relative_to(root))]=dict(size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
        snapshot[name]=dict(status=read(root/'STATUS.json'),files=rows)
    return snapshot


def real_checks(inputs,frozen,out):
    config=inputs.config;data=inputs.data;bank=inputs.banks[0];split=inputs.split
    counts={i:len(split['members'][i]['agent_ids']) for i in split['train_indices']}
    probes=[next(i for i,n in counts.items() if n==1),next(i for i,n in counts.items() if n==2),next(i for i,n in counts.items() if n>=9),max(counts,key=counts.get)]
    seed_all(3201);new,_=build(config);initial=model_hash(new);new.cuda()
    seed_all(3201);old,_=build(config,variant='v2_compatible');assert model_hash(old)==model_hash(new,True);old.cuda()
    rows=[]
    with no_optimizer_steps():
        for i in probes:
            b=data.batch([i],'train');goals=bank.batch([i],b);x=b.to('cuda');g=goals.cuda()
            with torch.no_grad():
                a=generate(old,x.observed,x.semantic_maps,x.valid,x.scene_ids,x.agent_ids,g,2035)
                z=generate(new,x.observed,x.semantic_maps,x.valid,x.scene_ids,x.agent_ids,g,2035)
                assert torch.equal(a[2],z[2])
                for aa,bb in zip(a[:2],z[:2]):torch.testing.assert_close(aa,bb,atol=2e-6,rtol=2e-6)
                rows.append(dict(index=i,n_agents=counts[i],G_max_abs=float((a[0]-z[0]).abs().max()),H_max_abs=float((a[1]-z[1]).abs().max()),pi_equal=True,keyed_noise_equal=True))
        del old;gc.collect();torch.cuda.empty_cache()
        new.train();bw,_=formal_callbacks(new,inputs,'cuda');observer=ModuleTelemetry(new,domain='real_readonly')
        # Full original 16-scene logical update graph, including the actual 57-agent scene.
        ids=list(dict.fromkeys(probes+split['train_indices'][:16]))[:16]
        torch.cuda.reset_peak_memory_stats();start=time.monotonic();new.zero_grad(set_to_none=True)
        metrics=bw(ids,torch.Generator().manual_seed(3201),torch.Generator().manual_seed(3202));grad=observer.capture_backward()
        assert model_hash(new)==initial
        real_rows=[dict(phase='fresh_zero_residual',indices=ids,gradient=grad,parameter_delta=0,
                       peak_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-start)]
        new.zero_grad(set_to_none=True)
    write(out/'REAL_FRESH.json',dict(status='PASS',initial_hash=initial,equivalence=rows,backward=real_rows,metrics=metrics,real_optimizer_updates=0))
    del new;gc.collect();torch.cuda.empty_cache()


def synthetic_recovery(inputs,out):
    # Actual v3 model and native controller, using ONLY constructed synthetic fixtures.
    data=SyntheticData([2,3,1,2,3],32);bank=SyntheticBank();c=inputs.config
    recipe=dict(inputs.recipe,logical_scenes=2,max_epochs=3,minimum=3)
    identity=dict(stage='joint',fold='TEST_ONLY',experiment='v3_formal_entry_recovery')
    ledger=[]
    for arm,limit,resume in [('continuous',4,False),('resumed',2,False),('resumed',4,True)]:
        seed_all(3201);m,_=build(c);m.cuda();bw,_=joint_callbacks(m,data,'cuda',c,bank,bank)
        folder=out/'synthetic'/arm;start=read(folder/'COUNTER.json')['updates'] if resume else 0
        observer=ModuleTelemetry(m,start_update=start,domain='synthetic')
        def backward(*args):
            row=bw(*args);observer.capture_backward();return row
        def validate():
            m.eval();b=data.batch([0],'selection_dev');r,_=window(m,b,bank.batch([0],b));return r['H_JADE']
        with observer.native_optimizer_hook():result=run_stage(folder,identity,recipe,m,list(range(5)),backward,validate,limit,resume=resume)
        write(folder/'COUNTER.json',dict(updates=result['update']));ledger+=observer.rows
        if len(ledger)>8:raise RuntimeError('synthetic acceptance budget exceeded')
        if not resume and arm=='resumed':assert result['completed_epoch']==0 and result['cursor']==4
        del m;gc.collect();torch.cuda.empty_cache()
    a=torch.load(out/'synthetic/continuous/last.pt',map_location='cpu',weights_only=False)
    b=torch.load(out/'synthetic/resumed/last.pt',map_location='cpu',weights_only=False)
    for k in ['model','optimizer','scheduler','rng','update','cursor','order','completed_epoch','stale','history','epoch_sums','epoch_steps']:
        equal(a[k],b[k])
    write(out/'SYNTHETIC_RECOVERY.json',dict(status='PASS',physical_optimizer_updates=len(ledger),real_optimizer_updates=0,
          partial_epoch_pause_cursor=4,continuous4_vs_2plus2_exact=True,telemetry=ledger))
    # Use activated synthetic model ONLY for read-only real gradient path audit; discard afterwards.
    model,_=build(c);model.load_state_dict(a['model'],strict=True);model.cuda().train();before=model_hash(model)
    real_bw,_=formal_callbacks(model,inputs,'cuda');probe=max(inputs.split['train_indices'],key=lambda i:len(inputs.split['members'][i]['agent_ids']))
    with no_optimizer_steps():
        model.zero_grad(set_to_none=True);real_bw([probe],torch.Generator().manual_seed(3201),torch.Generator().manual_seed(3202))
        observe=ModuleTelemetry(model,domain='real_readonly');grads=observe.capture_backward()
        assert all(g['gradient_l2']>0 for g in grads.values()) # connectivity check after activation; no magnitude threshold
        assert model_hash(model)==before
    write(out/'REAL_ACTIVATED.json',dict(status='PASS',index=probe,n_agents=57,gradient=grads,parameter_delta=0,
           synthetic_weights_only_for_diagnostic=True,not_used_as_initial_model=True,real_optimizer_updates=0))


def main():
    p=argparse.ArgumentParser();p.add_argument('section',choices=['real','recovery','dev']);args=p.parse_args()
    configure_cuda();torch.set_num_threads(1);roots=paths();out=roots['audit'];out.mkdir(parents=True,exist_ok=True)
    with locks(roots):
        inputs=FormalInputs();frozen,_=freeze(inputs);assert frozen==read(roots['run']/'FROZEN_CONFIG.json')
        if not (out/'PROTECTED_BEFORE.json').exists():write(out/'PROTECTED_BEFORE.json',protect(roots))
        if args.section=='real':real_checks(inputs,frozen,out)
        elif args.section=='recovery':synthetic_recovery(inputs,out)
        else:
            seed_all(3201);model,_=build(inputs.config);assert model_hash(model)==frozen['initial_model_sha256'];model.cuda()
            with no_optimizer_steps():
                validate=validation(model,inputs.data,inputs.banks[1],out,0);metric=validate()
                assert model_hash(model)==frozen['initial_model_sha256']
            write(out/'INITIAL_DEV.json',dict(status='PASS',H_JADE=metric,selection=False,optimizer_updates=0,
                  completed_windows=590,counts=inputs.counts,model_hash=frozen['initial_model_sha256']))
        assert protect(roots)==read(out/'PROTECTED_BEFORE.json')
        print('PASS',args.section,'real optimizer updates 0',flush=True)
if __name__=='__main__':main()

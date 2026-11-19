"""One source-bound ZARA2 Stage-2 run. No Stage-1, test, or next-fold entry."""
import argparse,contextlib,fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import torch
SOURCE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SOURCE))
from bcep.contracts import canonical_hash,digest
from bcep_joint.common import seed_all,read,write
from bcep_joint.inference import generate
from bcep_formal.data import FormalData
from bcep_formal.control import install
from bcep_v2.bank import Bank
from bcep_v2.joint import build_joint
from bcep_v2.stages import joint_callbacks
from bcep_v2.state import run_stage
from bcep_v2.runtime import configure_cuda
from bcep_v2.formal_manifest import write_csv
from ebjd.metrics import MetricAccumulator,hard_metric_statistics
from ebjd.unified import gather_candidates

AUDIT=SOURCE/'docs/research/bcep_v2/shared_world_objective_20261119'
EXPERIMENT='zara2_sharedworld_stage2_onerun_20261119'
BASE='468c5eddde48bdeec47c4bb3c39418352be16bb9'
REPORT_BASE='7cea557556386d9fbc6e905e02cf3b24640a8d03'
INITIAL='fcd1967d91712784ee963948b76fa3ba05c812224c81eeb0023db0751dc062c7'

def tensor_hash(model):
    h=hashlib.sha256()
    for n,t in model.state_dict().items():h.update(n.encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def paths(workspace):
    w=workspace.resolve()
    if w!=SOURCE.parent:raise ValueError('this entry is bound to its own worktree workspace')
    assets=w/'RSJG_EBJD_assets/experiments'
    return assets/'bcep_ebjd_v2_integrated',assets/'zara2_costproxy_stage2_20261119',assets/EXPERIMENT

def inputs(up,cost):
    frozen=read(AUDIT/'FROZEN_CONFIG.json')
    for name,h in frozen['implementation_files'].items():assert digest(SOURCE/name)==h,'audited implementation changed'
    assert read(AUDIT/'ACCEPTANCE.json')['status']=='PASS'
    assert read(up/'STATUS.json')['status']=='PAUSED_RESUMABLE'
    assert read(cost/'STATUS.json')['status']=='SESSION_BUDGET_PAUSED'
    run=read(up/'FORMAL_RUN.json');assert run['commit']==BASE
    plan=read(run['input_plan']);desc=plan['folds']['zara2']['split']
    assert digest(desc['path'])==desc['sha256']==frozen['split_sha256']
    split=read(desc['path']);assert split['unit']=='m'
    assert len(split['train_indices'])==2681 and len(split['selection_dev_indices'])==590
    assert sum(len(split['members'][i]['agent_ids'])>=9 for i in split['selection_dev_indices'])==127
    assert sum(len(split['members'][i]['agent_ids']) for i in split['selection_dev_indices'])==4262
    for i in split['train_indices']+split['selection_dev_indices']:assert digest(split['records'][i]['path'])==split['records'][i]['sha256']
    banks=[]
    for purpose in ['train','selection_dev']:
        p=up/'zara2/refined_banks'/f'{purpose}_2035';bid=read(p/'IDENTITY.json')
        assert (bid['seed'],bid['K'],bid['unit'],bid['purpose'])==(2035,20,'m',purpose)
        assert bid['base']==read(up/'zara2/density/FROZEN.json') and bid['head']==read(up/'zara2/head/FROZEN.json')
        b=Bank(p,bid,split);assert b.hash==frozen['bank_sha256'][purpose];banks.append(b)
    provider=[]
    for stage,epoch in [('density',19),('head',9)]:
        best=read(up/f'zara2/{stage}/TOP5.json')[0];assert best['epoch']==epoch
        assert digest(up/f'zara2/{stage}'/best['file'])==best['sha256'];provider.append(dict(stage=stage,epoch=epoch,sha256=best['sha256']))
    config=dict(frozen['config'],allow_real_training=True)
    assert {k for k in config if config[k]!=frozen['config'][k]}=={'allow_real_training'}
    original=read(up/'zara2/CONFIG.json');original_joint=dict(original['joint'],scales=split['bridge_scales'],allow_real_training=True,pair_objective='shared_world_bounded')
    assert config==original_joint
    recipe=frozen['original_recipe'];assert recipe==dict(seed=3201,lr=.0001,weight_decay=.0001,clip=5.,logical_scenes=16,max_epochs=100,minimum=30,patience=15,scheduler_horizon=100)
    return frozen,config,recipe,split,banks,provider

def resolve(frozen,config,recipe,split,banks,provider):
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
    tracked=subprocess.check_output(['git','ls-files','*.py'],cwd=SOURCE,text=True).splitlines()
    impl={p:digest(SOURCE/p) for p in tracked if not p.startswith(('tests/','docs/'))}
    identity=dict(stage='joint',fold='zara2',experiment=EXPERIMENT,report_base=REPORT_BASE,base_source=BASE,pair_objective='shared_world_bounded',implementation=canonical_hash(impl),config=canonical_hash(config),split=canonical_hash(split),train_bank=banks[0].hash,dev_bank=banks[1].hash)
    return dict(identity=identity,code_commit=source_commit,config=config,config_sha256=canonical_hash(config),recipe=recipe,RUN_ONE_ZARA2=True,real_update_cap=16800,archive_config_sha256=frozen['config_sha256'],config_diff={'allow_real_training':{'old':False,'new':True}},authorization_diff={'RUN_ONE_ZARA2':[False,True],'real_update_cap':[0,16800]},implementation_files=impl,provider=provider,initial_tensor_sha256=INITIAL,retention=frozen['retention'],decision=frozen['future_only_decision'],runtime=dict(precision='FP32',deterministic=True,TF32=False,SDP='math',candidate_seed=2035,training_seed=3201),window_counts=dict(train=2681,selection_dev=590,dev_N_ge_9=127))

@torch.no_grad()
def evaluate_window(model,batch,goals,device='cuda'):
    x=batch.to(device);g=goals.to(device);initial=[];original=model.coupling.assign
    def capture(*args,**kwargs):
        assert kwargs['straight_through'] is False
        r=original(*args,**kwargs,return_initial=True);initial.append(r[3]);return r[:3]
    model.coupling.assign=capture
    try:_,h,pi,_=generate(model,x.observed,x.semantic_maps,x.valid,x.scene_ids,x.agent_ids,g,2035,mode='H')
    finally:model.coupling.assign=original
    assert len(initial)==1
    stats=hard_metric_statistics(h,x.future,x.valid)
    raw=(g-x.future[:,None,:,-1]).norm(dim=-1).min(1).values
    torch.testing.assert_close(raw[x.valid],stats.agent_minfde[x.valid],atol=1e-6,rtol=1e-6)
    err=(initial[0]-x.future[:,None,:,-1]).norm(dim=-1)
    initial_jfde=((err*x.valid[:,None]).sum(-1)/x.valid.sum(-1)[:,None]).min(1).values
    one=MetricAccumulator();one.update_statistics(stats);m=one.as_dict()
    return dict(n_agents=int(x.valid.sum()),I_JFDE=float(initial_jfde.mean()),H_JFDE=m['JFDE'],H_JADE=m['JADE'],feedback_gain=float(initial_jfde.mean())-m['JFDE'],minFDE=m['minFDE'],oracle=m['scene_weighted_minFDE']),stats

def validation(model,data,bank,out,start_epoch):
    epoch=start_epoch
    def validate():
        nonlocal epoch
        start=time.monotonic();train_peak=torch.cuda.max_memory_allocated();torch.cuda.reset_peak_memory_stats()
        rows=[];acc={k:MetricAccumulator() for k in ['all','N_ge_9']}
        for pos,i in enumerate(data.split['selection_dev_indices']):
            b=data.batch([i],'selection_dev');r,s=evaluate_window(model,b,bank.batch([i],b));r=dict(epoch=epoch,index=i,**r);rows.append(r)
            acc['all'].update_statistics(s)
            if r['n_agents']>=9:acc['N_ge_9'].update_statistics(s)
            if (pos+1)%100==0:write(out/'VALIDATION_PROGRESS.json',dict(epoch=epoch,completed_windows=pos+1,total_windows=590,pid=os.getpid(),unix=time.time()))
        groups=[]
        for name,a in acc.items():
            rr=[r for r in rows if name=='all' or r['n_agents']>=9];assert len(rr)==(590 if name=='all' else 127)
            groups.append(dict(epoch=epoch,group=name,**a.as_dict(),I_JFDE=sum(r['I_JFDE'] for r in rr)/len(rr),feedback_gain=sum(r['feedback_gain'] for r in rr)/len(rr)))
        validate.metrics=acc['all'].as_dict()
        assert abs(validate.metrics['minFDE']-.3520422351797007)<1e-6
        assert abs(validate.metrics['scene_weighted_minFDE']-.33942601943420153)<1e-6
        write_csv(out/'dev'/f'epoch_{epoch:03d}_windows.csv',rows);write(out/'dev'/f'epoch_{epoch:03d}_groups.json',groups)
        write(out/'dev'/f'epoch_{epoch:03d}_cost.json',dict(train_peak_bytes=train_peak,validation_peak_bytes=torch.cuda.max_memory_allocated(),validation_seconds=time.monotonic()-start))
        write(out/'VALIDATION_PROGRESS.json',dict(epoch=epoch,completed_windows=590,total_windows=590,pid=os.getpid(),unix=time.time()))
        epoch+=1;torch.cuda.reset_peak_memory_stats();return validate.metrics['JADE']
    return validate

def smoke(config,out,frozen):
    from bcep_joint.checks import synthetic_batch
    from bcep_v2.objective import bridge_and_pair_loss
    seed_all(3201);m=build_joint(config).cuda();assert tensor_hash(m)==INITIAL and m.pair_objective=='shared_world_bounded'
    x=synthetic_batch(2,pixels=32).to('cuda');g=torch.randn(1,20,2,2,device='cuda')
    m.train();value=bridge_and_pair_loss(m,x,g,time_draw=torch.tensor([.4],device='cuda'))
    value[0].backward();assert m.last_pair_metrics['pair_a_mean']==0 and 'pair_q_a' in m.last_pair_metrics
    assert all(p.grad is None or p.grad.isfinite().all() for p in m.parameters())
    # Same observed-only 20-step inference and scalar aggregation as the formal validator.
    row,stats=evaluate_window(m,x,g)
    _,h,_,_=generate(m,x.observed,x.semantic_maps,x.valid,x.scene_ids,x.agent_ids,g,2035,mode='H')
    reference=MetricAccumulator();reference.update(h,x.future,x.valid)
    assert row['H_JADE']==reference.as_dict()['JADE'] and row['H_JFDE']==reference.as_dict()['JFDE']
    assert tensor_hash(m)==INITIAL
    write(out/'SMOKE.json',dict(status='PASS',identity=frozen['identity'],initial_tensor_sha256=INITIAL,real_optimizer_updates=0,synthetic_optimizer_updates=0,synthetic_backward_calls=1,inference_capture_equal=True,reused_audit_tests=15))
    del m,x,g;torch.cuda.empty_cache()

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train']);p.add_argument('--workspace',type=Path,required=True);args=p.parse_args()
    configure_cuda();torch.set_num_threads(1);up,cost,out=paths(args.workspace);out.mkdir(exist_ok=True,parents=True)
    with contextlib.ExitStack() as stack:
        for path,mode in [(out/'run.lock','a'),(up/'controller.lock','r'),(cost/'trial.lock','r')]:
            lock=stack.enter_context(path.open(mode));fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        audit,config,rec,split,banks,provider=inputs(up,cost);frozen=resolve(audit,config,rec,split,banks,provider)
        if (out/'FROZEN_CONFIG.json').exists():
            prior=read(out/'FROZEN_CONFIG.json')
            assert prior['identity']==frozen['identity'] and prior['config']==config and prior['recipe']==rec
            frozen=prior
        else:
            assert args.command=='prepare','prepare and smoke before training'
            write(out/'FROZEN_CONFIG.json',frozen);write(out/'STATUS.json',dict(status='PREPARED',optimizer_updates=0))
        if args.command=='prepare':
            assert not (out/'joint/last.pt').exists(),'cannot smoke a started run'
            smoke(config,out,frozen)
            protected={}
            for label,root,rel in [('formal',up,'zara1/head/last.pt'),('candidate_cost',cost,'joint/last.pt')]:
                for path in [root/rel,root/'STATUS.json',*root.rglob('TOP5.json')]:protected[label+'/'+str(path.relative_to(root))]=digest(path)
            write(out/'PROTECTED_BEFORE.json',protected)
            write(out/'PREPARED.json',dict(unix=time.time(),free_disk_bytes=shutil.disk_usage(out).free,expected_max_checkpoint_bytes=102*(up/'zara2/joint/epoch_001.pt').stat().st_size))
            print('PREPARE AND SMOKE PASS',flush=True);return
        assert read(out/'SMOKE.json')['identity']==frozen['identity'] and read(out/'SMOKE.json')['status']=='PASS'
        install();seed_all(3201);model=build_joint(config).cuda();assert tensor_hash(model)==INITIAL
        assert model.pair_objective=='shared_world_bounded' and model.training_options=={'coarse':.25,'geometry':.05,'map':.1}
        data=FormalData(split,'training',verify=False);native_backward,_=joint_callbacks(model,data,torch.device('cuda'),config,*banks)
        def backward(ids,order,views):
            assert model.pair_objective=='shared_world_bounded'
            row=native_backward(ids,order,views);assert 'pair_q_a_scene_sum' in row and row['pair_weight']==1.
            return row
        resume=(out/'joint/last.pt').exists();epoch=1
        if resume:
            state=torch.load(out/'joint/last.pt',map_location='cpu',weights_only=False)
            assert state['identity']==frozen['identity'] and state['recipe']==rec
            if state['status']=='COMPLETE':
                write(out/'STATUS.json',dict(status='COMPLETE',optimizer_updates=state['update'],completed_epoch=state['completed_epoch'],stop_reason=state['stop_reason']));return
            epoch=state['epoch'];del state
        validate=validation(model,data,banks[1],out,epoch)
        write(out/'STATUS.json',dict(status='RUNNING',pid=os.getpid(),started_unix=time.time(),resume=resume,identity=frozen['identity']))
        try:
            state=run_stage(out/'joint',frozen['identity'],rec,model,split['train_indices'],backward,validate,16800,resume=resume,record_progress=True)
            write(out/'STATUS.json',dict(status=state['status'],optimizer_updates=state['update'],completed_epoch=state['completed_epoch'],stop_reason=state['stop_reason'],finished_unix=time.time()))
        except BaseException as e:
            write(out/'STATUS.json',dict(status='TECHNICAL_INTERRUPTION',exception_type=type(e).__name__,reason=str(e),unix=time.time()));raise
if __name__=='__main__':main()

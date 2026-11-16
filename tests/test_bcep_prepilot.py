"""Only synthetic optimizer updates. Export tests call the actual CLI entry."""
import copy,json,random
from pathlib import Path
import numpy as np
import pytest
import torch
from bcep.provider import BCEP
from bcep.contracts import canonical_hash,digest,checkpoint_identity
from bcep.supervision import LikelihoodAccumulator
from bcep.pilot_runner import run_pilot,atomic_save
import importlib.util

def script_main(name):
    path=Path(__file__).resolve().parents[1]/'scripts'/name
    spec=importlib.util.spec_from_file_location(name.removesuffix('.py'),path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.main

export=script_main('evaluate_bcep_bank.py')


torch.set_num_threads(2)


def test_geometry_message_invariances_and_conditioned_flow():
    torch.manual_seed(3101);m=BCEP().double().eval()
    obs=torch.zeros(3,8,2,dtype=torch.float64);obs[:, :,0]=torch.arange(8,dtype=torch.float64)*.1
    obs[1,:,0]+=.5;obs[2]=float('nan');valid=torch.tensor([True,True,False])
    far=obs.clone();far[1,:,0]+=8
    with torch.no_grad():
        c=m.encoder(obs,None,valid,.4);cf=m.encoder(far,None,valid,.4)
        assert (c[0]-cf[0]).abs().max()>1e-3
        torch.testing.assert_close(m.encoder(obs+torch.tensor([7.,-3.]),None,valid,.4),c,atol=1e-10,rtol=1e-10)
        perm=[1,0,2];torch.testing.assert_close(m.encoder(obs[perm],None,valid[perm],.4),c[perm],atol=1e-10,rtol=1e-10)
        assert torch.isfinite(m.encoder(obs[:1],None,valid[:1],.4)).all()
        assert torch.equal(c[2],torch.zeros_like(c[2]))
        # Nonzero conditional fixture, not identity-initialized flow, no training.
        for net in m.flow.nets:net[-1].weight.normal_(0,.02)
        near=m.sample_support(obs,None,valid,'s',['a','b','pad'],2035)[0]
        away=m.sample_support(far,None,valid,'s',['a','b','pad'],2035)[0]
        assert (near[:,0]-away[:,0]).abs().max()>1e-5
        m.sample_support(obs[:1],None,valid[:1],'different',['a'],2035)
        assert torch.equal(near,m.sample_support(obs,None,valid,'s',['a','b','pad'],2035)[0])
        permuted=m.sample_support(obs[perm],None,valid[perm],'s',['b','a','pad'],2035)[0]
        torch.testing.assert_close(permuted,near[:,perm],atol=1e-10,rtol=1e-10)
    future=obs[:,-1:, :].expand(-1,12,-1).clone()+.2
    m.loss(obs,None,valid,future).backward()
    for module in [m.encoder.edge_bias,m.encoder.edge_value]:
        assert torch.isfinite(module.weight.grad).all() and module.weight.grad.abs().sum()>0


def components(n,final,total):
    return dict(loss_total=torch.tensor(float(total)),nll_final_sum=torch.tensor(float(final)),valid_agent_count=n,
                auxiliary={4:dict(nll_sum=torch.tensor(2.*n),valid_agent_count=n)},world_jacobian_included=True)


def test_likelihood_micro_macro_and_unchanged_objective():
    acc=LikelihoodAccumulator();acc.update(components(1,1,1.5));acc.update(components(3,9,3.5));metrics=acc.metrics()
    assert metrics['dev_nll_final_agent_micro']==2.5
    assert metrics['dev_nll_final_scene_macro']==2.
    assert metrics['dev_loss_total_scene_macro']==2.5
    assert metrics['dev_auxiliary']['4']['valid_agent_count']==4
    torch.manual_seed(3);m=BCEP();o=torch.randn(3,8,2);f=torch.randn(3,12,2);v=torch.ones(3,dtype=torch.bool)
    c=m.loss_components(o,None,v,f)
    ctx=m.encoder(o,None,v,.4)
    old=[-m.log_prob_world(f[:,h-1],o,ctx,h).mean() for h in [4,8,12]]
    torch.testing.assert_close(c['loss_total'],old[-1]+.25*torch.stack(old[:-1]).mean(),rtol=0,atol=0)
    torch.testing.assert_close(c['loss_total'],c['nll_final_sum']/3+.25*torch.stack([x['nll_sum']/3 for x in c['auxiliary'].values()]).mean())


class SyntheticDataset:
    def __init__(self):
        g=torch.Generator().manual_seed(44);self.items=[]
        for i in range(6):
            n=1+i%3;o=torch.randn(n,8,2,generator=g)*.1
            self.items.append(dict(observed=o,future=torch.randn(n,12,2,generator=g)*.1,
                semantic_maps=None,source_sequence='train' if i<4 else 'dev',scene_id=f's{i}',agent_ids=[str(j) for j in range(n)],frame_ids=list(range(20))))
    def __getitem__(self,i):
        item=dict(self.items[i])
        # Exercise all process RNGs along the actual runner's sample-read path.
        noise=(torch.rand(())+np.random.random()+random.random())*.001
        item['observed']=item['observed']+noise
        return item


def runner_fixture():
    ds=SyntheticDataset();members=[dict(index=i,sequence=x['source_sequence'],scene_id=x['scene_id'],agent_ids=x['agent_ids'],frame_ids=x['frame_ids']) for i,x in enumerate(ds.items)]
    split=dict(train_indices=list(range(4)),dev_indices=[4,5],members=members,scales=[[1.,1.]]*12,pred_length=12,dt=.4,unit='m')
    cfg=dict(device='cpu',seed=3101,max_updates=12,max_epochs=3,budget_seconds=3600,lr=1e-4,lr_gamma=.99,lambda_aux=.25,patience=3,allow_real_training=False)
    return ds,split,cfg,dict(split='SYNTHETIC_ONLY',weights_origin='random_initialization')


def assert_tree_equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_tree_equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_tree_equal(x,y)
    else:assert a==b


def test_real_runner_continuous_vs_interrupted_resume(tmp_path):
    ds,split,cfg,identity=runner_fixture();continuous=[];interrupted=[]
    run_pilot(ds,split,cfg,identity,tmp_path/'continuous',stop_after_updates=8,on_update=lambda i,u:continuous.append((i,u)))
    run_pilot(ds,split,cfg,identity,tmp_path/'resumed',stop_after_updates=3,on_update=lambda i,u:interrupted.append((i,u)))
    partial=torch.load(tmp_path/'resumed/last.pt',weights_only=False)
    assert partial['completed_epoch']==0 and partial['train_cursor']==3 and partial['scheduler']['last_epoch']==0
    run_pilot(ds,split,cfg,identity,tmp_path/'resumed',resume=True,stop_after_updates=5,on_update=lambda i,u:interrupted.append((i,u)))
    a=torch.load(tmp_path/'continuous/last.pt',weights_only=False);b=torch.load(tmp_path/'resumed/last.pt',weights_only=False)
    assert continuous==interrupted and len(continuous)==8
    for k in ['model','optimizer','scheduler','rng','order','epoch','completed_epoch','update','train_cursor','dev_cursor','best','stale','selected_epoch','train_accumulator','dev_accumulator']:
        assert_tree_equal(a[k],b[k])
    assert a['completed_epoch']==2 and a['scheduler']['last_epoch']==2
    assert torch.isfinite(torch.cat([v.flatten() for v in a['model'].values() if v.is_floating_point()])).all()
    selected=torch.load(tmp_path/'resumed/best.pt',weights_only=False)
    assert selected['selection_value']==selected['best'] and selected['split_hash']==identity['split']


def test_runner_exhausted_budget_and_update_cap_do_not_restart(tmp_path):
    ds,split,cfg,identity=runner_fixture();cfg['max_updates']=2
    run_pilot(ds,split,cfg,identity,tmp_path/'cap')
    path=tmp_path/'cap/last.pt';sha=digest(path)
    result=run_pilot(ds,split,cfg,identity,tmp_path/'cap',resume=True)
    assert result['update']==2 and result['completed_epoch']==0 and digest(path)==sha
    cfg['max_updates']=12
    run_pilot(ds,split,cfg,identity,tmp_path/'budget',stop_after_updates=1)
    path=tmp_path/'budget/last.pt';state=torch.load(path,weights_only=False);state['elapsed_wall_seconds']=cfg['budget_seconds'];atomic_save(state,path)
    sha=digest(path);result=run_pilot(ds,split,cfg,identity,tmp_path/'budget',resume=True)
    assert result['update']==1 and digest(path)==sha


def test_runner_equal_selector_keeps_earliest(tmp_path,monkeypatch):
    ds,split,cfg,identity=runner_fixture();cfg.update(max_epochs=2,max_updates=8)
    def constant(self,observed,semantic_maps,valid,future,aux_weight):
        n=len(observed);zero=next(self.parameters()).sum()*0
        return dict(loss_total=zero+2,nll_final_sum=zero+2*n,valid_agent_count=n,auxiliary={},world_jacobian_included=True)
    monkeypatch.setattr(BCEP,'loss_components',constant)
    state=run_pilot(ds,split,cfg,identity,tmp_path/'ties')
    best=torch.load(tmp_path/'ties/best.pt',weights_only=False)
    assert state['completed_epoch']==2 and state['selected_epoch']==1 and state['stale']==1
    assert best['completed_epoch']==1 and best['selection_metric_name']=='dev_nll_final_agent_micro'


def synthetic_export_fixture(root):
    root.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(1);fields={k:[] for k in ['observed','future','semantic_maps','agent_ids','frame_ids','timestamps','source_sequences','scene_ids']}
    members=[]
    for i,n in enumerate([1,2,1,2]):
        seq='train_seq' if i<2 else 'dev_seq';frames=list(range(i*20,(i+1)*20));ids=[str(x) for x in range(n)]
        maps=np.zeros((n,6,256,256),np.uint8);maps[:,1]=1
        row=dict(observed=rng.normal(size=(n,8,2)).astype('float32')*.1,future=rng.normal(size=(n,12,2)).astype('float32')*.1,semantic_maps=maps,agent_ids=ids,frame_ids=frames,timestamps=np.arange(20)*.4,source_sequences=seq,scene_ids=f's{i}')
        for k,v in row.items():fields[k].append(v)
        members.append(dict(index=i,sequence=seq,scene_id=f's{i}',frame_ids=frames,agent_ids=ids))
    arrays={}
    for k,v in fields.items():
        a=np.empty(len(v),object)
        for i,item in enumerate(v):a[i]=item
        arrays[k]=a
    shard=root/'scenes.npz';np.savez_compressed(shard,**arrays)
    manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(schema='ebjd-scene-manifest-v1',metadata=dict(unit='m',dt_seconds=.4,obs_length=8,pred_length=12,crop_width_m=32.,crop_pixels=256,semantic_channels=6,align_corners=True,out_of_bounds_fill=0),shards=[dict(path=shard.name,sha256=digest(shard),scenes=4)],statistics=dict(scenes=4),window_keys=[f's{i}' for i in range(4)])))
    split=root/'split.json';split.write_text(json.dumps(dict(source_manifest=str(manifest),source_sha256=digest(manifest),members=members,train_indices=[0,1],dev_indices=[2,3],dev_sequences=['dev_seq'],scales=[[1.,1.]]*12,scale_source='internal_train_only',unit='m',dt=.4,pred_length=12)))
    contract=root/'contract.json';contract.write_text(json.dumps(dict(split_hash=digest(split),source_manifest_hash=digest(manifest),unit='m',dt=.4,pred_length=12,train_windows=2,dev_windows=2,train_member_hash=canonical_hash(members[:2]),dev_member_hash=canonical_hash(members[2:]),historical_manifest_hashes=[],data_kind='synthetic')))
    checkpoint=root/'checkpoint.pt';torch.manual_seed(8)
    state=dict(schema='bcep-pilot-v2',identity=checkpoint_identity(contract,split),update=17,completed_epoch=2,selection_metric_name='dev_nll_final_agent_micro',protocol=dict(dt=.4,pred_length=12,unit='m'),model=BCEP().state_dict())
    torch.save(state,checkpoint)
    args=['--manifest',str(manifest),'--split-lock',str(split),'--contract',str(contract),'--checkpoint',str(checkpoint),'--subset','internal_dev','--purpose','internal_development','--output',str(root/'export')]
    return args,manifest,split,contract,checkpoint


def replace_arg(args,key,value):
    args=list(args);args[args.index(key)+1]=str(value);return args


def test_actual_export_five_seeds_subset_counts_shared_support(tmp_path,monkeypatch):
    args,*_=synthetic_export_fixture(tmp_path/'valid');calls=[]
    original=BCEP.select_support
    def inspect(self,atoms,valid,ids,k):
        out,meta=original(self,atoms,valid,ids,k);calls.append((atoms.detach().clone(),out.detach().clone(),meta));return out,meta
    monkeypatch.setattr(BCEP,'select_support',inspect)
    summary=export(args);folder=Path(args[args.index('--output')+1])
    assert len(calls)==10 and summary['seeds']==list(range(2035,2040))
    assert summary['pilot_screening']['status']=='NOT_EVALUATED_NONFORMAL_OR_INCOMPLETE'
    assert summary['inference_seed_statistics']['iid20']['all']['minFDE']['sample_sd']>=0
    for j,seed in enumerate(summary['seeds']):
        a=np.load(folder/f'iid20_seed{seed}.npz');b=np.load(folder/f'coverage20_seed{seed}.npz')
        metadata=json.loads(str(a['metadata_json']))
        assert list(a['scene_ids'])==['s2','s3'] and a['valid'].sum()==3
        assert metadata['checkpoint_training_updates']==17 and metadata['export_optimizer_updates']==0
        assert metadata['nonformal'] and not metadata['independent_final_test']
        for i,n in enumerate([1,2]):
            atoms,out,meta=calls[2*j+i]
            np.testing.assert_array_equal(a['endpoints'][i,:,:n],atoms[:20].numpy())
            np.testing.assert_array_equal(b['endpoints'][i,:,:n],out.numpy())
            for agent in range(n):np.testing.assert_array_equal(out[:,agent].numpy(),atoms[meta['agents'][agent]['indices'],agent].numpy())
        result=json.loads((folder/f'result_seed{seed}.json').read_text())
        assert result['likelihood']['eval_valid_agent_count']==3
        assert result['metrics']['iid20']['all']['windows']==2


@pytest.mark.parametrize('fault',['manifest','split','train_leak','checkpoint','random_as_dev','smoke_as_complete','limit_as_complete','missing_member','protocol','purpose'])
def test_actual_export_rejects_bad_identity(tmp_path,fault):
    args,manifest,split,contract,checkpoint=synthetic_export_fixture(tmp_path/fault)
    if fault=='manifest':
        wrong=tmp_path/'wrong.json';wrong.write_text('{}');args=replace_arg(args,'--manifest',wrong)
    elif fault=='split':split.write_text(split.read_text()+' ')
    elif fault in ['train_leak','missing_member']:
        s=json.loads(split.read_text());s['dev_indices']=[0,3] if fault=='train_leak' else [2];split.write_text(json.dumps(s))
        c=json.loads(contract.read_text());c['split_hash']=digest(split);contract.write_text(json.dumps(c))
    elif fault in ['checkpoint','protocol']:
        c=torch.load(checkpoint,weights_only=False)
        if fault=='checkpoint':c['identity']['weights_origin']='matched_gdts'
        else:c['protocol']['unit']='pixel'
        torch.save(c,checkpoint)
    elif fault in ['random_as_dev','smoke_as_complete']:
        at=args.index('--checkpoint');del args[at:at+2];args+=['--random-init-smoke']
        if fault=='smoke_as_complete':
            args=replace_arg(replace_arg(args,'--subset','interface_smoke'),'--purpose','interface_check');args+=['--require-complete']
    elif fault=='limit_as_complete':args+=['--limit','1','--require-complete']
    else:args=replace_arg(args,'--purpose','training_diagnostics')
    expected='limit_requested' if fault=='limit_as_complete' else 'random_initialization' if fault=='smoke_as_complete' else None
    with pytest.raises(ValueError,match=expected):export(args)
    assert not Path(args[args.index('--output')+1]).exists()


def test_real_training_cli_stays_disabled():
    main=script_main('train_bcep.py')
    config=Path(__file__).resolve().parents[1]/'configs/research/bcep_prepilot_v2.yaml'
    with pytest.raises(SystemExit,match='REAL_TRAINING_DISABLED'):main(['--config',str(config)])

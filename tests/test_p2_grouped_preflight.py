"""Metadata and synthetic permission tests; never fake-qualify real data."""
import copy,json,ast,subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from src.p2_protocol import ROOT,Registry,sealed,digest,FileProviders,combine_train
from src.p2_grouped import COUNTS,role_for,order,qualification_issues
from src.p2_grouped_artifacts import PreparationGrant,make_grant,scan_targets,build_target,validate_target,save_payload
from src.p2_grouped_training import WeightedMean,Selector,L2Selector,goal_bce_parts,PackReader,geometry_dataset,qualify_parent
from src.p2_data import base_packing_plan,observation_inputs
from src.losses import Goal_BCE_loss
from src.p2_bounded import BoundedController
D=ROOT/'docs/joint_dependency_v2/reviews/2026-10-05_7a0f83b_grouped_univ_complete_preflight'

@pytest.fixture(scope='module')
def reg():
    m=json.loads((D/'PREPARATION_MANIFEST.json').read_text());return Registry(m,m['manifest_hash'])

def inline(reg):
    m=copy.deepcopy(reg.manifest);m.pop('record_files',None);m['records']=copy.deepcopy(reg.records);return m

def test_exact_migration_and_packing(reg):
    assert reg.counts==COUNTS
    for role,n,e in [('train',175,11122),('inner_valid',391,24955)]:
        packs=base_packing_plan(reg,role);assert len(packs)==n and sum(map(len,packs))==e
        assert len([v for p in packs for v in p])==len(set(v for p in packs for v in p))
        assert all(len({reg.by_id[w]['source']['source_sha256'] for w,j in p})==1 for p in packs)
    moved=[r for r in reg.records if r['role']=='inner_valid' and Path(r['source']['original_source_path']).name!='uni_examples.txt']
    assert len(moved)==947
    for r in moved:
        for p in ('gradient','teacher_build_train','jdv2_gradient'):
            with pytest.raises(PermissionError):reg.authorize(r['window_id'],p)

@pytest.mark.parametrize('field',['role','logical_role','frame','agent','path','order','group'])
def test_new_registry_tamper(reg,field):
    m=inline(reg);r=m['records'][0]
    if field=='role':r['role']='train'
    elif field=='logical_role':r['source']['logical_role']='train'
    elif field=='frame':r['source']['frame_ids'][0]+=1
    elif field=='agent':r['source']['agent_ids'][0]+=1
    elif field=='path':r['source']['original_source_path']+='alias'
    elif field=='order':m['train_order'].reverse();m['train_order_hash']=digest(m['train_order'])
    else:r['isolation_group']='WRONG'
    m=sealed(m)
    with pytest.raises(RuntimeError):Registry(m,m['manifest_hash'])

def test_unknown_production_before_provider(reg):
    with pytest.raises(RuntimeError,match='EXTERNAL_EVIDENCE_BLOCKED'):PackReader(reg,'train')
    with pytest.raises(PermissionError):PackReader(reg,'outer')

@pytest.mark.parametrize('role,purpose',[('outer','metric_target_preparation'),('outer','gradient_target_preparation'),('inner_valid','gradient_target_preparation'),('inner_valid','PREFLIGHT_SMOKE'),('train','teacher')])
def test_purpose_denied_before_source(reg,role,purpose):
    g=PreparationGrant(reg,make_grant(reg));r=reg.refs_unordered(role)[0]
    with pytest.raises(PermissionError):g.authorize(r,purpose)

def test_wrong_grant_parent(reg):
    g=make_grant(reg);g['parent_manifest_hash']='wrong'
    with pytest.raises(RuntimeError):PreparationGrant(reg,g)

def synthetic(tmp_path,role='train',bad=None):
    p=tmp_path/'raw.txt';rows=[f'{t} 1 {t+20} 10\n' for t in range(20)]
    if bad=='missing':rows.pop()
    if bad=='duplicate':rows.append(rows[-1])
    if bad=='nonfinite':rows[-1]='19 1 nan 10\n'
    p.write_text(''.join(rows))
    from src.p2_protocol import file_hash
    r=dict(window_id='synthetic',role=role,source=dict(frame_ids=list(range(20)),agent_ids=[1],source_sha256=file_hash(p),original_source_path=str(p),scene_family='zara1',source_batch_path=str(tmp_path/'full.pkl')),artifacts={})
    reg=SimpleNamespace(records=[r],by_id={'synthetic':r},manifest={'manifest_hash':'synthetic','registered_source_rows_hash':digest(r['source'])})
    return r,PreparationGrant(reg,make_grant(reg))

@pytest.mark.parametrize('bad',['missing','duplicate','nonfinite'])
def test_raw_future_fail_closed(tmp_path,bad):
    r,g=synthetic(tmp_path,bad=bad)
    with pytest.raises(RuntimeError):scan_targets([r],g,Counter())

def test_target_math_slots_and_mask(tmp_path):
    import numpy as np
    assets=lambda:dict(H_inv=np.eye(3),tensor_image=torch.ones(6,8,8))
    from src.models.model_utils.cnn_big_images_utils import create_CNN_inputs_loop
    r,g=synthetic(tmp_path);ledger=Counter();xy=scan_targets([r],g,ledger)['synthetic'];v=build_target(xy,r,assets())
    assert ledger['window_local_target_slots']==12 and v['abs_pixel_coord'][0,0,0]==28
    assert torch.equal(v['input_traj_maps'],create_CNN_inputs_loop(v['abs_pixel_coord'].float()/8,assets()['tensor_image']))
    assert v['seq_list'].eq(1).all()

def test_artifact_halfwrite_and_tamper(tmp_path):
    import numpy as np
    assets=lambda:dict(H_inv=np.eye(3),tensor_image=torch.ones(6,8,8))
    r,g=synthetic(tmp_path);xy=scan_targets([r],g,Counter())['synthetic'];v=build_target(xy,r,assets())
    item=save_payload(v,r,'target',tmp_path,g);p=Path(item['artifact']['path']);p.write_bytes(b'broken')
    with pytest.raises(RuntimeError):save_payload(v,r,'target',tmp_path,g)

def test_weighted_goal_partition_mask_and_native():
    torch.manual_seed(8);logits=torch.randn(1,7,12,3,4);y=torch.rand(7,12,3,4)
    obs={'seq_list':torch.ones(8,7)};target={'seq_list':torch.ones(12,7),'input_traj_maps':y};target['seq_list'][3,2]=0
    n,d=goal_bce_parts(logits,obs,target)
    native=Goal_BCE_loss(logits,y,torch.cat([obs['seq_list'],target['seq_list']]).cumprod(0))
    assert abs(n/d-native.item())<1e-7 and d==6
    acc=WeightedMean()
    for sl in (slice(0,2),slice(2,6),slice(6,7)):
        acc.add(*goal_bce_parts(logits[:,sl],{'seq_list':obs['seq_list'][:,sl]},dict(seq_list=target['seq_list'][:,sl],input_traj_maps=y[sl])))
    assert abs(acc.value()-n/d)<1e-7
    with pytest.raises(ValueError):WeightedMean().value()

def test_selection_ties_and_l2_permissions():
    s=Selector(patience=2);assert s.update(1.,1)==(True,False);assert s.update(1.,2)==(False,False);assert s.update(1.1,3)==(False,True);assert s.epoch==1
    l=L2Selector(.01,2);assert l.update(1.,2.,1)[0];assert not l.update(.995,1.,2)[0];assert l.update(1.,1.9,3)[0]
    assert not l.update(1.,1.9,4)[0]
    with pytest.raises(PermissionError):l.update(0.,0.,5,role='outer')

def test_pre_fetch_cap_and_partial_no_epoch():
    visits=[]
    def gen():
        for i in range(9):visits.append(i);yield i
    c=BoundedController(2,175,2537)
    for b in c.batches(gen()):c.outcome(True)
    assert visits==[0,1] and not c.complete_epoch and c.ledger()['after_progress']==2/2537

def test_prediction_before_metric_target_spy():
    reader=object.__new__(PackReader);reader.role='inner_valid';reader.read_trace=[];calls=[]
    def read(i,kind):calls.append(kind);return {'only':kind}
    reader._read=read
    def predict(obs):
        assert calls==['observation'] and obs=={'only':'observation'};calls.append('predict');return 3
    def metric(pred,obs,target):assert pred==3 and calls==['observation','predict','target'];return 4
    assert reader.predict_then_metric(0,predict,metric)==4
    reader.role='outer'
    with pytest.raises(PermissionError):reader.predict_then_metric(0,predict,metric)
    with pytest.raises(PermissionError):reader.training(0)

def test_selection_deadline_before_observation_fetch(monkeypatch):
    import src.p2_grouped_training as gt
    monkeypatch.setattr(gt,'time',SimpleNamespace(monotonic=lambda:11.))
    model=SimpleNamespace(eval=lambda:None)
    # No registry/provider/dataset exists: touching any of them would fail.
    reader=SimpleNamespace(packs=[[('synthetic',0)]])
    with pytest.raises(RuntimeError,match='PARTIAL_SELECTION'):
        gt.evaluate(model,reader,'cpu',True,10.)

def test_geometry_constructor_no_visual_or_raw(reg,monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('legacy dataset constructed')
    import src.models.goal_pretrain as goal
    monkeypatch.setattr(goal,'create_dataset',forbidden)
    ds=geometry_dataset(reg,('train','inner_valid'))
    assert set(ds.scenes)=={'eth','zara1','zara2','univ'} and 'hotel' not in ds.scenes
    args=SimpleNamespace(obs_length=8,pred_length=12,dataset='eth5')
    model=goal.Goal_Pretrain(args,'cpu',dataset=ds)
    assert model.enc_chs==(14,32,32,64,64,64) and model.dec_chs==(64,64,64,32,32)

def test_smoke_parent_rejected_without_qualification(tmp_path,reg):
    from src.p2_protocol import file_hash
    p=tmp_path/'smoke.pt';torch.save(dict(schema='grouped-training-state-v1',classification='SMOKE_ONLY'),p)
    with pytest.raises(RuntimeError,match='classification'):qualify_parent(dict(path=str(p),sha256=file_hash(p)),reg,'goal')

def test_stage_scoped_qualification_does_not_require_outer_semantics(reg):
    issues=qualification_issues(reg,('train','inner_valid'))
    assert 'SEMANTIC_QUALIFICATION_hotel' not in issues and 'SEMANTIC_QUALIFICATION_univ' in issues

def test_evidence_updates_do_not_change_data_but_assets_do(reg):
    from src.p2_grouped import data_binding
    changed=copy.deepcopy(reg);before=data_binding(reg)
    changed.manifest['qualification']['scenes']['hotel']['review_note']='future evidence update only'
    assert data_binding(changed)==before
    assert 'SEMANTIC_QUALIFICATION_univ' in qualification_issues(changed,('train','inner_valid'))
    changed.manifest['assets']['univ']['pred_mask.png']['sha256']='0'*64
    assert data_binding(changed)!=before

def test_provider_filename_and_outer_rejected_before_bytes(reg):
    from src.p2_grouped_training import GroupedFileProvider
    provider=GroupedFileProvider(reg)
    outer=copy.deepcopy(reg.refs_unordered('outer')[0])
    with pytest.raises(PermissionError,match='before provider'):
        provider.read(outer,'target')
    train=copy.deepcopy(reg.refs_unordered('train')[0])
    train['artifacts']['target']={'path':'/does/not/exist/renamed.pt','sha256':'0'*64}
    with pytest.raises(RuntimeError,match='filename identity'):
        provider.read(train,'target')

def test_provider_rejects_stale_asset_binding_before_tensor(tmp_path,reg):
    from src.p2_grouped_training import GroupedFileProvider
    from src.p2_observation import atomic_json
    r=copy.deepcopy(reg.refs_unordered('train')[0]);p=tmp_path/(r['window_id']+'.pt')
    r['artifacts']['observation']={'path':str(p),'sha256':'0'*64}
    atomic_json(sealed({'asset_binding_hash':'old-input-contract'}),p.with_suffix('.json'))
    with pytest.raises(RuntimeError,match='observation assets differ'):
        GroupedFileProvider(reg).read(r,'observation')

def test_grant_runtime_refs_only_not_source_alias(reg):
    grant=PreparationGrant(reg,make_grant(reg))
    r=copy.deepcopy(reg.refs_unordered('train')[0])
    r['artifacts']['target']={'path':'runtime-filled.pt','sha256':'a'*64}
    grant.authorize(r,'gradient_target_preparation')
    r['source']['original_source_path']+='.alias'
    with pytest.raises((RuntimeError,PermissionError)):
        grant.authorize(r,'gradient_target_preparation')

@pytest.mark.parametrize('kind',['actual_hash','known_bad','incomplete','wrong_schema'])
def test_parent_negative_contracts(tmp_path,reg,monkeypatch,kind):
    import src.p2_grouped_training as gt
    from src.p2_protocol import file_hash
    dest=tmp_path/'parent.pt'
    payload={'schema':'grouped-training-state-v1','classification':'FORMAL_EPOCH',
        'artifact_role':'goal','family':gt.FAMILY,'data_binding':gt.data_binding(reg),
        'gradient_roles':['train'],'selection_role':'inner_valid','outer_access':False,
        'progress':{'epoch_completed':0,'cursor':1}}
    if kind=='wrong_schema':payload['schema']='legacy'
    torch.save(payload,dest);sha=file_hash(dest)
    if kind=='known_bad':monkeypatch.setattr(gt,'BAD_BASES',{sha})
    if kind=='actual_hash':sha='0'*64
    with pytest.raises(RuntimeError):
        gt.qualify_parent({'path':str(dest),'sha256':sha},reg,'goal')

def test_l2_actual_adapter_synthetic_joint_reference(monkeypatch):
    import src.p2_grouped_training as gt
    from src.models.model import GDTS
    calls=[]
    obs=dict(abs_pixel_coord=torch.zeros(8,2,2),seq_list=torch.ones(8,2),
        input_traj_maps=torch.ones(2,8,2,2),scene_index=torch.zeros(2,dtype=torch.long))
    target=dict(abs_pixel_coord=torch.zeros(12,2,2),seq_list=torch.ones(12,2),input_traj_maps=torch.ones(2,12,2,2))
    row={'role':'inner_valid','window_id':'synthetic','source':{'scene_family':'synthetic'}}
    reg=SimpleNamespace(refs_unordered=lambda role:[row],authorize=lambda *a:calls.append('authorize'))
    monkeypatch.setattr(gt,'require_provenance',lambda *a:None) # synthetic only, no real registry/files
    class Provider:
        def read(self,r,kind):
            if kind=='target':assert calls[-1]=='prediction'
            calls.append(kind);return obs if kind=='observation' else target
    class Model:
        args=SimpleNamespace(down_factor=8)
        dataset=SimpleNamespace(scenes={'synthetic':SimpleNamespace(make_world_coord_torch=lambda x:x)})
        compute_model_metrics=GDTS.compute_model_metrics
        def eval(self):return self
        def __call__(self,x,if_test):
            assert x['abs_pixel_coord'].shape[0]==8 and x['input_traj_maps'].shape[1]==8
            assert not torch.is_grad_enabled();calls.append('prediction')
            pred=torch.zeros(2,20,2,2)
            pred[0,8:,0,0]=2/8;pred[1,8:,1,0]=4/8
            return pred,{}
    result=gt.l2_inner_evaluate(Model(),reg,'cpu',Provider())
    assert calls==['authorize','observation','prediction','target']
    assert result['JADE']['value']==1 and result['JFDE']['value']==1
    assert result['JADE']['denominator']==1
    row['role']='outer';calls.clear()
    with pytest.raises(PermissionError):gt.l2_inner_evaluate(Model(),reg,'cpu',Provider())
    assert calls==[]

@pytest.mark.parametrize('partial',[False,True])
def test_actual_new_training_loop_epoch_boundaries_synthetic(tmp_path,monkeypatch,reg,partial):
    """Explicit synthetic model/readers, not a fake-qualified real-data run."""
    import src.p2_grouped_training as gt
    import src.models.goal_pretrain as gm
    class Tiny(torch.nn.Module):
        def __init__(self,*a,**k):super().__init__();self.w=torch.nn.Parameter(torch.ones(1))
        def prepare_inputs(self,b,ids):return b,None,None
        def get_loss(self,x,seq):return {'loss':self.w.square().sum()}
        def set_losses_coeffs(self):return {'loss':1.}
    events=[];sizes=[len(p) for p in base_packing_plan(reg,'train')]
    class FixtureReader:
        def __init__(self,r,role):self.packs=[list(range(n)) for n in sizes]
        def __len__(self):return 175
        def training(self,i):events.append(('fetch',i));return {},{},None,None
    monkeypatch.setattr(gm,'Goal_Pretrain',Tiny)
    monkeypatch.setattr(gt,'launch_guard',lambda *a:reg)
    monkeypatch.setattr(gt,'geometry_dataset',lambda *a:SimpleNamespace(scenes={}))
    monkeypatch.setattr(gt,'formal_output',lambda *a:tmp_path/'formal_synthetic')
    monkeypatch.setattr(gt,'PackReader',FixtureReader)
    def evaluate(*a):
        events.append(('selection',));v=WeightedMean();v.add(24955,24955);return v
    monkeypatch.setattr(gt,'evaluate',evaluate)
    def save(dest,model,opt,sched,reg,args,p,*a):
        events.append(('save',p['epoch_completed'],sched.last_epoch,p['successful'],p['agent_exposures']))
        torch.save({'synthetic_fixture':True},dest)
    monkeypatch.setattr(gt,'save_checkpoint',save)
    args=SimpleNamespace(p2_mode='goal',device='cpu',learning_rate=.001,early_stopping_patience=0,num_epochs=2,clip=1.,validation_seed=3101)
    if partial:
        clock=iter([0.,86401.]);monkeypatch.setattr(gt,'time',SimpleNamespace(monotonic=lambda:next(clock)))
        with pytest.raises(RuntimeError,match='PARTIAL_EPOCH'):gt.train(args,reg)
        assert events==[]
    else:
        result=gt.train(args,reg)
        assert result['best_epoch']==1
        assert [e for e in events if e[0]=='save']==[('save',1,1,175,11122),('save',2,2,350,22244)]
        assert len([e for e in events if e[0]=='fetch'])==350
        assert len([e for e in events if e[0]=='selection'])==2

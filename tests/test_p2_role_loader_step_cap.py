"""P2 synthetic contracts: no real neural constructor/forward or real data."""
import copy
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from src.p2_protocol import *
from src.p2_data import RoleDataset,base_packing_plan

from src.p2_cache import execute
from src.p2_bounded import BoundedController
from src.trainer import trainer,_requested_data_splits
from src.models.goal_pretrain import goal_pretrainer
from src.joint_dependency_v2_cache import JointDependencyV2CacheBuilder
from src.parser import get_parser,check_and_add_additional_args
import mc_wiring_fixture as mc

COUNTS=Counter()
TRACES=[]

def fixture_manifest():
    records=[]
    for i,(role,sha,physical) in enumerate([
        ('train','a'*64,'test'),('train','b'*64,'train'),
        ('inner_valid',INNER,'train'),('outer',HOTEL,'train')]):
        s=dict(source_sha256=sha,id_sha256=str(i)*64,content_sha256=str(i+1)*64,
            source_batch_path='synthetic/'+physical+'/'+str(i)+'.pkl',physical_split=physical,
            physical_index=0,cache_filename=str(i)+'.pkl',logical_role=role,
            frame_ids=list(range(20)),agent_ids=[1,2],window_steps=20,
            scene_family='scene'+str(i),source_id='sha256:'+sha)
        records.append(dict(source=s,window_id=identity(s),role=role,aliases=['alias'+str(i)],
            boundary=[8,12],recording=dict(status='VERIFIED',recording_id='rec'+str(i),
            id_namespace='local',evidence_sha256='e'*64),
            map_provenance=dict(status='PROVIDED_STATIC_ASSET',evidence_sha256='f'*64,
                                protocol_permitted=True),
            artifacts={k:None for k in ('observation','target','teacher','deployment')}))
    order=[r['window_id'] for r in records if r['role']=='train']
    return sealed(dict(schema=SCHEMA,family=FAMILY,source_commit='synthetic',records=records,
        train_order=order,train_order_hash=digest(order),down_factor=8,parents={}))
def registry(m=None):
    m=fixture_manifest() if m is None else m
    return Registry(m,m['manifest_hash'],registered_rows=digest([r['source'] for r in m['records']]))
def tensors(n=2):
    return dict(abs_pixel_coord=torch.arange(20*n*2).reshape(20,n,2).float(),
        seq_list=torch.ones(20,n),input_traj_maps=torch.zeros(n,20,2,2),
        frame_ids=torch.arange(20)[:,None].repeat(1,n),
        tensor_image=torch.zeros(6,2,2),tensor_map=torch.zeros(3,2,2),
        scene_index=torch.zeros(n,dtype=torch.long),scene_ptr=torch.tensor([0,n]),
        batch_format_version=torch.tensor(2))
class Providers:
    def __init__(self):self.calls=Counter()
    def read(self,r,kind):
        self.calls[(r['role'],kind)]+=1;COUNTS['provider_'+kind]+=1
        if r['role']=='outer' and kind!='observation':raise AssertionError('protected future provider')
        if r['role']=='inner_valid' and kind=='teacher':raise AssertionError('protected teacher provider')
        obs,target=split_window_payload(tensors(),r['source'])
        target['future_position_world']=target['abs_pixel_coord'].permute(1,0,2)
        return {'observation':obs,'target':target,'teacher':{'future_pair_descriptor':torch.zeros(1,6)},
            'deployment':{'edge_index':torch.tensor([[0],[1]])}}[kind]

@pytest.mark.parametrize('mutation',['hash','member','address','alias','recording','role','boundary','order'])
def test_registry_tamper(mutation):
    m=fixture_manifest();pinned=digest([r['source'] for r in m['records']])
    if mutation=='hash':m['manifest_hash']='x'
    elif mutation=='member':m['records'][0]['source']['id_sha256']='c'*64
    elif mutation=='address':m['records'][0]['source']['cache_filename']='wrong.pkl'
    elif mutation=='alias':m['records'][1]['aliases']=m['records'][0]['aliases']
    elif mutation=='recording':m['records'][2]['recording']['recording_id']='rec0'
    elif mutation=='role':m['records'][0]['role']='outer'
    elif mutation=='boundary':m['records'][0]['boundary']=[9,11]
    elif mutation=='order':m['train_order']=m['train_order'][:1];m['train_order_hash']=digest(m['train_order'])
    if mutation!='hash':m=sealed(m)
    with pytest.raises(RuntimeError):Registry(m,m['manifest_hash'],registered_rows=pinned)

def test_unknown_before_provider():
    m=fixture_manifest();m['records'][0]['recording']['status']='UNKNOWN';reg=registry(sealed(m));p=Providers()
    with pytest.raises(RuntimeError,match='RECORDING'):read_record(reg,m['train_order'][0],'gradient',p)
    assert not p.calls

@pytest.mark.parametrize('role,purpose,target,teacher',[
 ('train','gradient',1,0),('train','jdv2_gradient',1,1),('train','candidate_build',0,0),
 ('inner_valid','selection',1,0),('inner_valid','candidate_build',0,0),('outer','candidate_build',0,0)])
def test_role_permissions(role,purpose,target,teacher):
    reg=registry();p=Providers();r=reg.refs(role)[0]
    result=read_record(reg,r['window_id'],purpose,p)
    assert (result['target'] is not None)==bool(target)
    assert p.calls[(role,'teacher')]==teacher
    assert r['source']['physical_index']==0
    if role=='train':assert r['source']['physical_split']=='test'

@pytest.mark.parametrize('role,purpose',[('outer','selection'),('outer','gradient'),('outer','metric_evaluation'),
 ('inner_valid','jdv2_gradient'),('inner_valid','teacher_build_train')])
def test_denied_before_observation(role,purpose):
    reg=registry();p=Providers()
    with pytest.raises(PermissionError):read_record(reg,reg.refs(role)[0]['window_id'],purpose,p)
    assert not p.calls

def test_missing_observation_never_full_load(monkeypatch):
    reg=registry();p=FileProviders()
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('must not load full batch'))
    with pytest.raises(RuntimeError,match='MISSING_OBSERVATION'):read_record(reg,reg.refs('outer')[0]['window_id'],'candidate_build',p)
    assert not p.counts

def test_real_provider_identity_and_hash(tmp_path):
    reg=registry();r=reg.refs('train')[0];obs,_=split_window_payload(tensors(),r['source'])
    p=tmp_path/'obs.pt';torch.save(dict(kind='observation',window_id=r['window_id'],
        source_identity=digest(r['source']),payload=obs),p)
    r['artifacts']['observation']=dict(path=str(p),sha256=file_hash(p))
    provider=FileProviders()
    assert provider.read(r,'observation')['abs_pixel_coord'].shape==(8,2,2)
    r['artifacts']['observation']['sha256']='0'*64
    with pytest.raises(RuntimeError,match='hash'):provider.read(r,'observation')
    assert provider.counts['observation']==1

def test_split_is_window_bound_not_frame_union():
    reg=registry();r=reg.records[0];full=tensors()
    obs,target=split_window_payload(full,r['source'])
    assert obs['input_traj_maps'].shape[1]==8 and target['input_traj_maps'].shape[1]==12
    bad=copy.deepcopy(r['source']);bad['frame_ids'][0]=-1
    with pytest.raises(RuntimeError,match='frame'):split_window_payload(full,bad)

def test_role_dataset_and_base_packing():
    reg=registry();p=Providers()
    d=RoleDataset(reg,'train','gradient',p,base_pack=True)
    batches=[d[i] for i in range(len(d))]
    assert len(batches)==2 and batches[0][0]['abs_pixel_coord'].shape==(1,20,2,2)
    assert sum(len(x) for x in base_packing_plan(reg,'train'))==4
    inner=RoleDataset(reg,'inner_valid','selection',p)[0]
    assert inner['observation']['abs_pixel_coord'].shape[0]==8
    assert p.calls[('inner_valid','teacher')]==0

def test_cache_production_tasks_descriptor_serializer(tmp_path):
    reg=registry();p=Providers()
    def candidate(obs,ref):
        assert obs['abs_pixel_coord'].shape[0]==8
        COUNTS['fake_candidate_calls']+=1
        return dict(edge_index=torch.tensor([[0],[1]]),contains_future_supervision=False,
            goal_candidates_map=torch.zeros(2,21,2),goal_candidates_world=torch.zeros(2,21,2),
            candidate_log_prior=torch.zeros(2,21))
    producer=dict(classification='QUALIFIED',data_binding=data_binding(reg),sha256='d'*64)
    manifest=execute(reg,p,candidate,tmp_path/'new',producer)
    assert manifest['ledger']==dict(deployment_calls=4,target_reads=2,teacher_calls=2,teacher_writes=2,protected_teacher_calls=0)
    COUNTS.update({'descriptor_calls':2,'teacher_writes':2})
    assert not p.calls[('outer','target')] and not p.calls[('inner_valid','target')]
    for r in reg.records:
        assert ('teacher' in manifest['records'][r['window_id']])==(r['role']=='train')
    with pytest.raises(RuntimeError,match='new root'):execute(reg,p,candidate,tmp_path/'new',producer)

@pytest.mark.parametrize('entry',['parser','trainer','goal','builder','main'])
def test_blocked_guard_before_side_effect(entry,monkeypatch):
    a=get_parser().parse_args(['--p2_mode','goal','--phase','goal_pretrain'])
    import src.p2_protocol as pp
    def reject(*a,**k):raise RuntimeError('BLOCKED_RECORDING_IDENTITY_UNVERIFIED')
    monkeypatch.setattr(pp,'load_registry',reject)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('device before metadata'))
    funcs={'parser':check_and_add_additional_args,'trainer':trainer,'goal':goal_pretrainer,
        'builder':JointDependencyV2CacheBuilder,
        'main':lambda args:dispatch(args,lambda _:pytest.fail('net'),None,None)}
    with pytest.raises(RuntimeError,match='RECORDING'):funcs[entry](a)

def test_default_splits_and_disabled_cap():
    a=SimpleNamespace(phase='train',clean_split_protocol=False)
    assert _requested_data_splits(a)==['train','valid','test'] and not bounded(a)
    a.clean_split_protocol=True
    assert _requested_data_splits(a)==['train','valid']
    a.p2_mode='l1'
    assert _requested_data_splits(a)==['train']

def loop_fixture(tmp_path,mode,cap=3,native=5,fail=None):
    item=mc.harness(tmp_path,mode)
    a=item.args
    a.p2_mode='l1';a.p2_reference_total_steps=native;a.p2_max_update_attempts=cap
    a.p2_skip_validation=True;a.p2_no_automatic_resume=True
    a.load_checkpoint=None;a.num_epochs=1;a.use_trajectory_bank_cache=False
    a.coupling_grad_accum_steps=1;a.freeze_upstream_generator=True;a.joint_diagnostics=False
    a.use_wandb=False
    item.p2_exposure=[]
    item.p2_controller=BoundedController(cap,native,native,clock=lambda:0)
    class Loader:
        def __len__(self):return native
        def __iter__(self):
            for i in range(native):
                COUNTS['loop_fetch']+=1
                yield ({},{'p2_window_ids':['synthetic-'+str(i)]})
    item.data_loaders={'train':Loader()}
    item.net.prepare_inputs=lambda batch,ids:(item.inputs,None)
    item.net.configure_training_epoch=lambda _:None
    item.net.train=lambda *a,**k:None  # no real baseline modules in constructor-free fixture
    calls=[0]
    def loss(inputs,seq):
        calls[0]+=1;COUNTS['fake_loss_calls']+=1
        result=mc.losses(item)
        if fail=='forward':raise ValueError('synthetic forward failure')
        if fail=='nonfinite':return {k:v*float('nan') for k,v in result.items()}
        if fail=='skip':return {k:v.detach() for k,v in result.items()}
        return result
    item.net.get_loss=loss
    item._autocast_context=lambda:__import__('contextlib').nullcontext()
    original=item.optimizer.step
    def counted(*a,**k):
        result=original(*a,**k)
        COUNTS['actual_toy_optimizer_updates']+=1
        return result
    item.optimizer.step=counted
    item._evaluate_epoch=lambda *a,**k:pytest.fail('bounded validation')
    item._save_checkpoint=lambda *a,**k:pytest.fail('ordinary bounded checkpoint')
    return item

@pytest.mark.parametrize('mode',['mean_energy','expected_conditional_mc'])
@pytest.mark.parametrize('cap,native',[(1,5),(3,5),(3,3)])
def test_actual_train_loop_cap_and_snapshot(tmp_path,mode,cap,native):
    item=loop_fixture(tmp_path,mode,cap,native);before=COUNTS.copy()
    step=item.scheduler.last_epoch
    result=item._train_loop(1,1)
    assert COUNTS['loop_fetch']-before['loop_fetch']==cap
    assert COUNTS['actual_toy_optimizer_updates']-before['actual_toy_optimizer_updates']==cap
    assert result['counts']['successful_updates']==result['counts']['backward']==cap
    assert result['counts']['loss_samples']==cap and result['after_progress']==cap/native
    assert result['frozen_equal'] and len(result['update_norms'])==cap
    assert result['first_components'] and result['last_components']
    assert item.scheduler.last_epoch-step==int(cap==native)
    payload=torch.load(tmp_path/'bounded_stop_weights_only.pt',weights_only=False)
    assert payload['resumable'] is False and payload['artifact_role']=='bounded_stop_weights_only'
    with pytest.raises(RuntimeError,match='not a resume'):item._load_state_file(str(tmp_path/'bounded_stop_weights_only.pt'))
    if mode=='mean_energy':assert not hasattr(item,'jdv2_objective_rng')
    else:assert item.jdv2_objective_rng.draw_calls==cap and not item.jdv2_objective_rng.pending_backward
    TRACES.append(result)

@pytest.mark.parametrize('mode',['mean_energy','expected_conditional_mc'])
@pytest.mark.parametrize('failure',['skip','forward','nonfinite'])
def test_actual_failure_no_replacement(tmp_path,mode,failure):
    item=loop_fixture(tmp_path,mode,fail=failure);before=COUNTS.copy()
    if failure=='skip':
        result=item._train_loop(1,1);assert result['stop_reason']=='SKIPPED_UPDATE'
    else:
        with pytest.raises((ValueError,FloatingPointError)):item._train_loop(1,1)
    assert COUNTS['loop_fetch']-before['loop_fetch']==1
    assert COUNTS['actual_toy_optimizer_updates']==before['actual_toy_optimizer_updates']
    assert not (tmp_path/'bounded_stop_weights_only.pt').exists()
    if mode=='expected_conditional_mc' and failure!='skip':assert item.jdv2_objective_rng.pending_backward
    TRACES.append(item.p2_controller.ledger())

@pytest.mark.parametrize('reason',['clock','memory','empty','identity'])
def test_limit_empty_identity(tmp_path,reason):
    item=loop_fixture(tmp_path,'mean_energy')
    if reason=='clock':item.p2_controller.clock=lambda:901
    if reason=='memory':item.p2_controller.memory=lambda:11*1024**3
    if reason=='empty':item.data_loaders['train']=[]
    if reason=='identity':item.p2_controller.order=['wrong']*5
    before=COUNTS.copy()
    if reason=='identity':
        with pytest.raises(RuntimeError):item._train_loop(1,1)
    else:item._train_loop(1,1)
    assert COUNTS['actual_toy_optimizer_updates']==before['actual_toy_optimizer_updates']
    assert COUNTS['loop_fetch']-before['loop_fetch']==int(reason=='identity')

def test_bounds_parser_and_beta():
    a=get_parser().parse_args(['--p2_mode','l1','--num_workers','0','--p2_max_update_attempts','500',
        '--p2_reference_total_steps','3484','--p2_skip_validation','True','--p2_no_automatic_resume','True'])
    validate_bounds(a,3484)
    assert .1*min((499/3484)/.2,1)==pytest.approx(.07161308840413318)
    a.num_workers=1
    with pytest.raises(RuntimeError):validate_bounds(a,3484)

def test_old_parent_cannot_be_relabelled():
    reg=registry()
    for sha in BAD_BASES:
        with pytest.raises(RuntimeError,match='KNOWN_PROTOCOL'):
            validate_parent({},reg,'p2_goal',sha)

def populate_artifacts(reg,tmp_path):
    for r in reg.records:
        obs,target=split_window_payload(tensors(),r['source'])
        for kind,payload in [('observation',obs),('target',target)]:
            p=tmp_path/(r['window_id']+kind+'.pt')
            torch.save(dict(kind=kind,payload=payload,window_id=r['window_id'],source_identity=digest(r['source'])),p)
            r['artifacts'][kind]=dict(path=str(p),sha256=file_hash(p))

def goal_args():
    a=get_parser().parse_args(['--p2_mode','goal','--phase','goal_pretrain',
        '--goal_model_type','independent','--training_stage','baseline','--num_epochs','150',
        '--num_workers','0','--data_augmentation','False','--final_test_access','blocked'])
    a.reproducibility=False
    return a

def test_fresh_goal_no_base_dependency_and_train_only_dispatch(tmp_path,monkeypatch):
    reg=registry();populate_artifacts(reg,tmp_path)
    import src.p2_protocol as pp
    monkeypatch.setattr(pp,'load_registry',lambda args:reg)
    a=goal_args();assert launch_guard(a) is reg
    calls=[]
    processor=SimpleNamespace(train=lambda:calls.append('train'),
        train_test=lambda:pytest.fail('P2 goal wrapper'),test=lambda:pytest.fail('extra test'))
    dispatch(a,lambda _:processor,None,None)
    assert calls==['train'] and not reg.manifest['parents']

def test_joint_dependency_parent_before_device(tmp_path,monkeypatch):
    reg=registry();populate_artifacts(reg,tmp_path)
    import src.p2_protocol as pp
    monkeypatch.setattr(pp,'load_registry',lambda args:reg)
    a=goal_args();a.p2_mode='joint';a.phase='train';a.num_epochs=250
    with pytest.raises(RuntimeError,match='GOAL_PARENT'):launch_guard(a)
    p=tmp_path/'goal.pt';torch.save({'fake_state':torch.ones(1)},p)
    reg.manifest['parents']['goal']=dict(path=str(p),sha256=file_hash(p),artifact_role='p2_goal',
        data_binding=data_binding(reg),gradient_roles=['train'],selection_role='inner_valid',
        classification='QUALIFIED',initial_state_sha256='1'*64)
    a.goal_pretrain_checkpoint=str(p)
    assert launch_guard(a) is reg
    reg.manifest['parents']['goal']['gradient_roles']=['outer']
    with pytest.raises(RuntimeError,match='selection'):launch_guard(a)

def test_input_adapter_and_selection_never_feed_future():
    from src.p2_data import evaluate_inner
    reg=registry();p=Providers();sample=RoleDataset(reg,'inner_valid','selection',p)[0]
    scene=SimpleNamespace(make_world_coord_torch=lambda x:x)
    calls=[]
    def prediction(inputs):
        assert inputs['abs_pixel_coord'].shape[0]==8
        assert inputs['input_traj_maps'].shape[1]==8 and inputs['x_augmented'].shape[0]==8
        assert not torch.is_grad_enabled()
        calls.append('obs_predict')
        return torch.tensor(1.)
    def metric(*args):
        assert args[-1]['abs_pixel_coord'].shape[0]==20
        calls.append('metric_target');return [2.]
    net=SimpleNamespace(eval=lambda:None,init_test_metrics=lambda:{'goal_BCE':[]},
        dataset=SimpleNamespace(scenes={sample['identity']['scene_name'][0]:scene}),
        forward=prediction,compute_model_metrics=metric,
        prepare_inputs=lambda *a:pytest.fail('no full GT prepare_inputs'))
    owner=SimpleNamespace(net=net,args=SimpleNamespace(down_factor=8),device=torch.device('cpu'),
        data_loaders={'valid':[sample]})
    assert evaluate_inner(owner,goal=True)=={'valid_goal_BCE':2.}
    assert calls==['obs_predict','metric_target']
    COUNTS['fake_selection_forward']+=1

def test_new_base_atomic_metadata_not_auto_qualified(tmp_path):
    from src.p2_checkpoint import initialize_ledger,save_base
    reg=registry();a=SimpleNamespace(p2_mode='goal',seed=3101)
    net=torch.nn.Linear(1,1);net.best_valid_metric=lambda:'goal_BCE'
    owner=SimpleNamespace(args=a,net=net,p2_registry=reg)
    initialize_ledger(owner);owner.p2_exposure=['wid0','wid1']
    p=tmp_path/'goal.pt'
    save_base(owner,{'model_state_dict':net.state_dict(),'best_metrics':{'goal_BCE':1.}},p,1)
    saved=torch.load(p,weights_only=False)
    meta=json.loads(Path(str(p)+'.provenance.json').read_text())
    assert saved['p2_provenance']['gradient_roles']==['train']
    assert meta['classification']=='PROVENANCE_REVIEW_REQUIRED'
    assert meta['sha256']==file_hash(p) and meta['resumable'] is False
    assert not list(tmp_path.glob('*.tmp'))

def test_pair_failure_stops_other_arm_before_fetch(tmp_path):
    p=tmp_path/'pair_stop.json'
    left=BoundedController(1,2,2,clock=lambda:0,pair_failure_path=str(p))
    left.fail('bad')
    right=BoundedController(1,2,2,clock=lambda:0,pair_failure_path=str(p))
    assert list(right.batches([None,None]))==[]
    assert right.counts['fetch_attempts']==0 and right.reason=='PAIR_ALREADY_STOPPED'

def test_nonfinite_ledger_json_and_failure_count():
    c=BoundedController(1,2,2,clock=lambda:0);c.loss(float('nan'));c.fail('one');c.fail('two')
    json.dumps(c.ledger(),allow_nan=False)
    assert c.counts['failed']==1

def test_observation_alias_full_batch_hash_rejected(tmp_path):
    reg=registry();r=reg.records[0];r['source']['source_batch_sha256']='f'*64
    r['artifacts']['observation']={'path':str(tmp_path/'copy.pt'),'sha256':'f'*64}
    with pytest.raises(RuntimeError,match='relabelled'):FileProviders().read(r,'observation')

def test_p2_loader_dispatch_and_no_valid_for_l1(monkeypatch):
    import src.p2_data as data
    import src.data_loader as legacy
    reg=registry()
    monkeypatch.setattr(data,'launch_guard',lambda args:reg)
    monkeypatch.setattr(legacy,'dataset_set_name',lambda *a,**k:pytest.fail('physical dataset'))
    a=SimpleNamespace(p2_mode='l1')
    loader=legacy.get_dataloader(a,'train')
    assert loader.num_workers==0 and len(loader)==2
    with pytest.raises(PermissionError):legacy.get_dataloader(a,'valid')
    with pytest.raises(PermissionError):legacy.get_dataloader(a,'test')

def test_default_network_loss_definitions_unchanged():
    import ast,subprocess
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    paths=list((root/'src/models').rglob('*.py'))+[root/'src/joint_goal_loss.py',root/'src/jdv2_objective_state.py']
    for p in paths:
        relative=str(p.relative_to(root))
        old=subprocess.check_output(['git','show','28728bf:'+relative],cwd=root,text=True)
        before=ast.parse(old);after=ast.parse(p.read_text())
        # Grouped-only dataset injection is the sole permitted model edit.
        # Normalize its explicit optional argument and conditional assignment;
        # all layers, losses and every other AST node still match the baseline.
        if relative in {'src/models/model.py','src/models/goal_pretrain.py'}:
            cls=next(n for n in after.body if isinstance(n,ast.ClassDef) and n.name in {'GDTS','Goal_Pretrain'})
            init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
            assert [x.arg for x in init.args.kwonlyargs]==['dataset']
            assert ast.literal_eval(init.args.kw_defaults[0]) is None
            init.args.kwonlyargs=[];init.args.kw_defaults=[]
            assignment=next(n for n in init.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='dataset' for t in n.targets))
            assert ast.unparse(assignment.value.test)=='dataset is not None'
            assert ast.unparse(assignment.value.body)=='dataset'
            assignment.value=assignment.value.orelse
        if relative=='src/models/goal_pretrain.py':
            before=next(n for n in before.body if isinstance(n,ast.ClassDef) and n.name=='Goal_Pretrain')
            after=next(n for n in after.body if isinstance(n,ast.ClassDef) and n.name=='Goal_Pretrain')
        assert ast.dump(before,include_attributes=False)==ast.dump(after,include_attributes=False),relative

def test_p2_execution_protocol_counts_not_legacy(tmp_path):
    owner=SimpleNamespace(args=SimpleNamespace(p2_mode='l1',model_dir=str(tmp_path)),
        p2_registry=registry(),data_loaders={'train':[1,2]})
    trainer._write_evaluation_protocol(owner)
    report=json.loads((tmp_path/'p2_evaluation_protocol.json').read_text())
    assert report['logical_window_counts']=={'train':2,'inner_valid':1,'outer':1}
    assert report['loader_batches']=={'train':2} and not report['validation_constructed']
    assert report['outer_metric_access']=='LOCKED'

def test_p2_base_not_training_resume(tmp_path):
    owner=mc.harness(tmp_path)
    p=tmp_path/'p2.pt';torch.save({'p2_provenance':{'artifact_role':'p2_base'},'resumable':False},p)
    with pytest.raises(RuntimeError,match='resume scope'):owner._load_state_file(p)

def test_shared_initial_authentication(tmp_path):
    from src.p2_checkpoint import load_shared_initial,state_hash
    reg=registry();net=torch.nn.Linear(1,1)
    p=tmp_path/'initial.pt';reg.manifest['shared_initial_state']={'path':str(p),'base_sha256':'a'*64}
    payload=dict(artifact_role='p2_shared_initial',base_sha256='a'*64,data_binding=data_binding(reg),
        model_state_dict=net.state_dict(),state_sha256=state_hash(net.state_dict()),
        initialization_policy='fresh_heads_zero_corrector',optimizer_updates=0)
    torch.save(payload,p)
    owner=SimpleNamespace(p2_registry=reg,net=net)
    load_shared_initial(owner)
    payload['optimizer_updates']=1;torch.save(payload,p)
    with pytest.raises(RuntimeError,match='untrained'):load_shared_initial(owner)

def test_cache_graph_contract_before_parent_or_device(monkeypatch):
    import src.p2_protocol as pp
    monkeypatch.setattr(pp,'load_registry',lambda args:registry())
    a=goal_args();a.p2_mode='cache';a.phase='build-jdv2-cache';a.graph_radius=7.
    with pytest.raises(RuntimeError,match='candidate/graph'):launch_guard(a)

def test_frozen_change_ledger_counts_and_no_snapshot(tmp_path):
    from src.p2_bounded import finish_bounded
    from src.p2_checkpoint import state_hash
    net=torch.nn.Linear(1,1);net.requires_grad_(False)
    before=state_hash(net.state_dict())
    net.weight.add_(1.)
    control=BoundedController(1,2,2,clock=lambda:0)
    control.reason='ATTEMPT_CAP';control.counts['successful_updates']=1
    owner=SimpleNamespace(args=SimpleNamespace(model_dir=str(tmp_path)),net=net,
        p2_controller=control,p2_frozen_before=before)
    ledger=finish_bounded(owner)
    assert ledger['stop_reason']=='FROZEN_WEIGHT_CHANGED' and ledger['counts']['failed']==1
    assert not ledger['frozen_equal'] and not (tmp_path/'bounded_stop_weights_only.pt').exists()

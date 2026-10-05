"""Observation preparation contracts: only synthetic coordinates except CLI stages."""
import copy,json,math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from src.p2_protocol import Registry,sealed,digest,identity,file_hash,HOTEL,INNER,FileProviders,split_window_payload
from src.p2_observation import (PreparationGate,make_grant,integer,scan_observations,build_payload,
    validate_observation,content_hash,save_artifact,verify_artifact,Budget,load_assets,inspect_assets)
from src.data_src.data_utils import pixel2world_numpy
from src.models.model_utils.cnn_big_images_utils import make_gaussian_map_patches,create_tensor_image

PERMIT=lambda p,h:None


def fixture(tmp_path,n=2,role='train',future='BAD BAD',missing=False,duplicate=False):
    p=tmp_path/'source.txt'
    rows=[]
    for t in range(20):
        for a in range(n):
            xy=f'{8+t} {12+a}' if t<8 else future
            if not(missing and t==2 and a==0):rows.append(f'{t}.0 {a+1}.0 {xy}\n')
    if duplicate:rows.append(rows[0])
    p.write_text(''.join(rows))
    sha=file_hash(p)
    # Real Registry role policy pinned to source SHAs; synthetic train only for raw scan.
    s=dict(source_sha256=sha,id_sha256='b'*64,content_sha256='c'*64,
        source_batch_path=str(tmp_path/'full.pkl'),source_batch_sha256='d'*64,
        original_source_path=str(p),physical_split='train',physical_index=0,cache_filename='full.pkl',
        logical_role='train',frame_ids=list(range(20)),agent_ids=list(range(1,n+1)),
        window_steps=20,scene_family='zara1',source_id='sha256:'+sha)
    r=dict(source=s,window_id=identity(s),role='train',aliases=[str(p)],boundary=[8,12],
        recording=dict(status='UNKNOWN'),map_provenance=dict(status='UNKNOWN'),
        artifacts={k:None for k in ('observation','target','teacher','deployment')})
    order=[r['window_id']]
    m=sealed(dict(schema='rsjg-p2-role-v1',family='p2_hotel_uni_examples',source_commit='synthetic',
        records=[r],train_order=order,train_order_hash=digest(order),down_factor=8))
    reg=Registry(m,m['manifest_hash'],registered_rows=digest([s]))
    g=make_grant(reg);gate=PreparationGate(reg,g,g['manifest_hash'])
    return reg,gate,r,p


def assets():
    sem=torch.zeros(6,8,8);sem[0]=1
    return dict(H=np.eye(3),H_inv=np.eye(3),tensor_image=sem,tensor_map=torch.ones(3,8,8)/2)


@pytest.mark.parametrize('token',['1.5','NaN','Infinity','bad','9223372036854775808'])
def test_invalid_ids(token):
    with pytest.raises(RuntimeError):integer(token)

@pytest.mark.parametrize('token',['1','1.000','1e0','+1'])
def test_integer_formats(token):assert integer(token)==1


@pytest.mark.parametrize('change',['grant','address','purpose','source'])
def test_gate_before_decode(tmp_path,change):
    reg,gate,r,p=fixture(tmp_path)
    if change=='grant':
        g=copy.deepcopy(gate.grant);g['allow_training']=True
        with pytest.raises(RuntimeError):PreparationGate(reg,g,g['manifest_hash'])
    elif change=='address':
        changed=copy.deepcopy(r);changed['source']['physical_index']=9
        with pytest.raises(RuntimeError):scan_observations([changed],gate,Counter())
    elif change=='purpose':
        with pytest.raises(PermissionError):gate.authorize(r,'gradient')
    else:
        p.write_bytes(p.read_bytes()+b'20 1 INVALID INVALID\n')
        ledger=Counter()
        with pytest.raises(RuntimeError,match='hash'):scan_observations([r],gate,ledger)
        assert ledger['unique_raw_obs_xy_decodes']==0


@pytest.mark.parametrize('n',[1,3])
def test_observation_only_invalid_future_and_order(tmp_path,n):
    reg,gate,r,p=fixture(tmp_path,n=n)
    counts=Counter();xy=scan_observations([r],gate,counts)[r['window_id']]
    assert xy.shape==(8,n,2) and counts['window_local_observation_slots']==8*n
    assert counts['unique_raw_obs_xy_decodes']==8*n
    with pytest.raises(RuntimeError,match='RECORDING'):reg.authorize(r['window_id'],'gradient')
    payload,_=build_payload(xy,r,assets())
    assert payload['seq_list'].eq(1).all()
    assert payload['abs_pixel_coord'].dtype==torch.float64


@pytest.mark.parametrize('failure',['missing','duplicate','nonfinite','fractional_id'])
def test_bad_rows(tmp_path,failure):
    reg,gate,r,p=fixture(tmp_path,missing=failure=='missing',duplicate=failure=='duplicate')
    if failure in {'nonfinite','fractional_id'}:
        text=p.read_text().replace('0.0 1.0 8 12','0.0 1.0 nan 12' if failure=='nonfinite' else '0.5 1.0 8 12')
        p.write_text(text)
        # Rebuild a synthetic registry so the file hash gate passes; test row semantics.
        m=copy.deepcopy(reg.manifest);s=m['records'][0]['source'];s['source_sha256']=file_hash(p)
        m['records'][0]['window_id']=identity(s);m['train_order']=[identity(s)];m['train_order_hash']=digest(m['train_order'])
        m=sealed(m);reg=Registry(m,m['manifest_hash'],registered_rows=digest([s]))
        g=make_grant(reg);gate=PreparationGate(reg,g,g['manifest_hash']);r=reg.records[0]
    with pytest.raises((RuntimeError,ValueError)):scan_observations([r],gate,Counter())


def test_future_counterfactual_same_observation(tmp_path):
    first=tmp_path/'a';second=tmp_path/'b';first.mkdir();second.mkdir()
    reg,g,r,_=fixture(first,future='BAD BAD')
    _,g2,r2,_=fixture(second,future='999 -300')
    x=scan_observations([r],g,Counter())[r['window_id']]
    y=scan_observations([r2],g2,Counter())[r2['window_id']]
    assert np.array_equal(x,y)
    assert content_hash(build_payload(x,r,assets())[0])==content_hash(build_payload(y,r2,assets())[0])


def test_overlapping_routes_are_window_local(tmp_path):
    reg,gate,r,p=fixture(tmp_path,future='40 40')
    m=copy.deepcopy(reg.manifest);r2=copy.deepcopy(r);s=r2['source']
    s['frame_ids']=list(range(4,24));s['id_sha256']='e'*64;s['content_sha256']='f'*64
    s['physical_index']=1;s['source_batch_path']=str(tmp_path/'second.pkl');s['cache_filename']='second.pkl'
    r2['window_id']=identity(s);m['records'].append(r2);m['train_order'].append(r2['window_id']);m['train_order_hash']=digest(m['train_order'])
    m=sealed(m);reg=Registry(m,m['manifest_hash'],registered_rows=digest([x['source'] for x in m['records']]))
    g=make_grant(reg);gate=PreparationGate(reg,g,g['manifest_hash']);counts=Counter()
    values=scan_observations(reg.records,gate,counts)
    assert values[r['window_id']].shape==(8,2,2)
    assert values[r2['window_id']][4:].tolist()==[[[40.,40.],[40.,40.]]]*4
    assert counts['unique_raw_obs_xy_decodes']==24 and counts['window_local_observation_slots']==32


@pytest.mark.parametrize('scene',['eth','hotel','univ','zara1'])
def test_original_geometry_swap_and_units(tmp_path,scene):
    _,g,r,_=fixture(tmp_path,n=1);r=copy.deepcopy(r);r['source']['scene_family']=scene
    a=assets();a['H']=np.diag([2.,3.,1.]);a['H_inv']=np.linalg.inv(a['H'])
    xy=np.ones((8,1,2),dtype=np.float64)*24
    payload,_=build_payload(xy,r,a)
    pix=payload['abs_pixel_coord'].numpy()
    assert np.array_equal(pix[0,0],[8.,12.] if scene in ('eth','hotel') else [12.,8.])
    recovered=pixel2world_numpy(pix.reshape(-1,2)[:,[1,0]] if scene in ('eth','hotel') else pix.reshape(-1,2),a['H'])
    assert np.array_equal(recovered,xy.reshape(-1,2))


def test_gaussian_independent_math_and_image_scaling():
    centers=torch.tensor([[2.,3.],[1.5,2.5]])
    actual=make_gaussian_map_patches(centers,8,8,gaussian_std=1.)
    expected=torch.empty_like(actual)
    for t,(cx,cy) in enumerate(centers.tolist()):
        for y in range(8):
            for x in range(8):expected[0,t,y,x]=math.exp(-((x-cx)**2+(y-cy)**2)/2)
        expected[0,t]/=expected[0,t].max()
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
    default=make_gaussian_map_patches(centers[:1],8,8)
    direct=make_gaussian_map_patches(centers[:1],8,8,gaussian_std=8/64)
    assert torch.equal(default,direct)
    rgb=np.full((16,16,3),128,dtype=np.uint8)
    assert torch.equal(create_tensor_image(rgb,8),torch.full((3,2,2),128/255))
    semantic=np.zeros((16,16,6),dtype=np.float32);semantic[:,:,3]=1
    result=create_tensor_image(semantic,8)
    assert result.shape==(6,2,2) and result[3].eq(1).all()


def test_synthetic_full20_input_parity_no_model(tmp_path):
    from src.p2_data import observation_inputs,add_outer
    from src.models.model import GDTS
    reg,g,r,_=fixture(tmp_path)
    xy=scan_observations([r],g,Counter())[r['window_id']]
    obs,_=build_payload(xy,r,assets())
    full=dict(obs)
    for name,dim in [('abs_pixel_coord',0),('seq_list',0),('frame_ids',0),('input_traj_maps',1)]:
        shape=list(obs[name].shape);shape[dim]=12
        tail=torch.ones(shape,dtype=obs[name].dtype)
        full[name]=torch.cat([obs[name],tail],dim=dim)
    full['abs_pixel_coord']=full['abs_pixel_coord']/8
    scene=SimpleNamespace(make_world_coord_torch=lambda x:x)
    owner=SimpleNamespace(args=SimpleNamespace(seq_length=20,obs_length=8,down_factor=8),
        device=torch.device('cpu'),dataset=SimpleNamespace(scenes={'x':scene}),active_goal_model_type='independent')
    trained,_=GDTS.prepare_inputs(owner,add_outer(full),{'scene_name':['x']})
    selected=observation_inputs(obs,scene,torch.device('cpu'))
    assert selected['x_augmented'].shape==(8,2,8)
    assert torch.equal(trained['x_augmented'][:8],selected['x_augmented'])
    from src.models.common import derivative_of
    rel=obs['abs_pixel_coord'].float()/8-obs['abs_pixel_coord'][-1].float()/8
    assert torch.equal(selected['x_augmented'][...,2:4],derivative_of(rel,dt=1))
    synthetic20=copy.deepcopy(full);synthetic20['abs_pixel_coord'][8:]*=-99
    checked,_=GDTS.prepare_inputs(owner,add_outer(synthetic20),{'scene_name':['x']})
    assert torch.equal(checked['x_augmented'][:8],selected['x_augmented'])
    # Existing target separation API is synthetic-only here.
    raw=dict(full);raw['frame_ids']=torch.arange(20)[:,None].repeat(1,2)
    _,target=split_window_payload(raw,r['source'])
    assert target['abs_pixel_coord'].shape==(12,2,2)


@pytest.mark.parametrize('tamper',['wrapper','sidecar','half','shape','old_bytes'])
def test_artifact_tamper_and_atomic_resume(tmp_path,tamper):
    reg,g,r,p=fixture(tmp_path);out=tmp_path/'obs'
    xy=scan_observations([r],g,Counter())[r['window_id']];payload,_=build_payload(xy,r,assets())
    binding={'synthetic':True}
    row=save_artifact(payload,r,out,g,binding,PERMIT)
    assert verify_artifact(r,out,g,binding,PERMIT)==row
    assert save_artifact(payload,r,out,g,binding,PERMIT)==row
    target=Path(row['observation']['path']);side=target.with_suffix('.json')
    if tamper=='wrapper':target.write_bytes(b'bad bytes')
    elif tamper=='sidecar':
        m=json.loads(side.read_text());m['window_id']='wrong';side.write_text(json.dumps(m))
    elif tamper=='half':side.unlink()
    elif tamper=='shape':
        wrong=dict(payload);wrong['scene_ptr']=torch.tensor([0,9])
        with pytest.raises(RuntimeError):validate_observation(wrong,r)
        return
    else:
        forged=copy.deepcopy(r);forged['artifacts']['observation']=dict(path=str(p),sha256=r['source']['source_batch_sha256'])
        with pytest.raises(RuntimeError,match='relabelled'):FileProviders().read(forged,'observation')
        return
    with pytest.raises(RuntimeError):verify_artifact(r,out,g,binding,PERMIT)


def test_atomic_failure_no_certified_completion(tmp_path,monkeypatch):
    reg,g,r,p=fixture(tmp_path);out=tmp_path/'obs';xy=scan_observations([r],g,Counter())[r['window_id']]
    payload,_=build_payload(xy,r,assets())
    monkeypatch.setattr(torch,'save',lambda *a:(_ for _ in ()).throw(OSError('synthetic half write')))
    with pytest.raises(OSError):save_artifact(payload,r,out,g,{},PERMIT)
    assert not list(out.glob('*.pt')) and not list(out.glob('.observation-*'))


@pytest.mark.parametrize('limit',['wall','storage','rss'])
def test_budget_and_no_reset(tmp_path,monkeypatch,limit):
    now=[0.];b=Budget(tmp_path/'budget',seconds=10,max_bytes=100000,clock=lambda:now[0])
    if limit=='wall':now[0]=11
    elif limit=='storage':b.bytes=100001
    else:monkeypatch.setattr('src.p2_observation.resource.getrusage',lambda _:SimpleNamespace(ru_maxrss=9*1024**2))
    with pytest.raises(RuntimeError,match='BUDGET'):b.check()
    with pytest.raises(RuntimeError,match='reset'):Budget(tmp_path/'budget',seconds=20,clock=lambda:0)


def test_full_pickle_guard_before_deserialize(tmp_path):
    p=tmp_path/'full.pkl';p.write_bytes(b'not a pickle')
    with pytest.raises(RuntimeError,match='guard'):torch.load(p,map_location='cpu',weights_only=True)


def test_unknown_all_roles_preparation_but_production_denied():
    from src.p2_protocol import ROOT
    from src.p2_observation import registry_from_file
    parent=ROOT/'docs/joint_dependency_v2/reviews/2026-10-05_28728bf_p2_role_loader_step_cap_cpu/P2_RUNTIME_MANIFEST.json'
    m=json.loads(parent.read_text());reg=registry_from_file(parent,m['manifest_hash']);g=make_grant(reg)
    gate=PreparationGate(reg,g,g['manifest_hash'])
    for role in ['train','inner_valid','outer']:
        r=reg.refs(role)[0];gate.authorize(r,'artifact_preparation')
        with pytest.raises(RuntimeError,match='RECORDING'):reg.authorize(r['window_id'],'candidate_build')
        with pytest.raises(PermissionError):gate.authorize(r,'metric_evaluation')

def test_original_outside_raster_behavior_no_clamp(tmp_path):
    _,_,r,_=fixture(tmp_path,n=1)
    xy=np.full((8,1,2),16.,dtype=np.float64);xy[:,:,0]=-.8
    payload,count=build_payload(xy,r,assets())
    assert count==8 and payload['abs_pixel_coord'][0,0,0]==-.8
    assert torch.isfinite(payload['input_traj_maps']).all()
    xy[:,:,0]=-8000.
    with pytest.raises(RuntimeError,match='nonfinite'):build_payload(xy,r,assets())

@pytest.mark.parametrize('method',['is_available','device_count','init','synchronize'])
def test_guard_intercepts_deliberate_cuda_queries(method):
    with pytest.raises(RuntimeError,match='CPU preparation guard: cuda'):
        getattr(torch.cuda,method)()

def test_guard_intercepts_native_query():
    if hasattr(torch._C,'_cuda_getDeviceCount'):
        with pytest.raises(RuntimeError,match='cuda_native'):torch._C._cuda_getDeviceCount()


def test_real_outer_metric_rejected_before_provider():
    from src.p2_protocol import ROOT,read_record
    from src.p2_observation import registry_from_file
    parent=ROOT/'docs/joint_dependency_v2/reviews/2026-10-05_28728bf_p2_role_loader_step_cap_cpu/P2_RUNTIME_MANIFEST.json'
    m=json.loads(parent.read_text());reg=registry_from_file(parent,m['manifest_hash'])
    provider=FileProviders()
    with pytest.raises(RuntimeError,match='RECORDING'):
        read_record(reg,reg.refs('outer')[0]['window_id'],'selection',provider)
    assert not provider.counts
    # Original role policy is inspected separately; UNKNOWN wins first in real metadata.


def test_manifest_hash_and_duplicate_members_rejected(tmp_path):
    reg,g,r,p=fixture(tmp_path)
    with pytest.raises(RuntimeError):
        Registry(reg.manifest,'0'*64,registered_rows=digest([r['source']]))
    m=copy.deepcopy(reg.manifest);m['records'].append(copy.deepcopy(r));m=sealed(m)
    with pytest.raises(RuntimeError):
        Registry(m,m['manifest_hash'],registered_rows=digest([r['source'],r['source']]))


@pytest.mark.parametrize('field',['kind','window_id','source_identity'])
def test_rehashed_wrong_wrapper_rejected(tmp_path,field):
    reg,g,r,p=fixture(tmp_path);out=tmp_path/'obs'
    xy=scan_observations([r],g,Counter())[r['window_id']]
    payload,_=build_payload(xy,r,assets())
    row=save_artifact(payload,r,out,g,{},PERMIT)
    dest=Path(row['observation']['path']);side=dest.with_suffix('.json')
    blob=dict(window_id=r['window_id'],kind='observation',source_identity=digest(r['source']),payload=payload)
    blob[field]='wrong';torch.save(blob,dest)
    meta=json.loads(side.read_text());meta['artifact_sha256']=file_hash(dest)
    side.write_text(json.dumps(sealed(meta)))
    with pytest.raises(RuntimeError,match='identity'):verify_artifact(r,out,g,{},PERMIT)


def test_changed_asset_binding_before_load(tmp_path):
    reg,g,r,p=fixture(tmp_path);out=tmp_path/'obs'
    xy=scan_observations([r],g,Counter())[r['window_id']]
    payload,_=build_payload(xy,r,assets());save_artifact(payload,r,out,g,{'id':1},PERMIT)
    calls=[]
    with pytest.raises(RuntimeError,match='binding'):
        verify_artifact(r,out,g,{'id':2},lambda *a:calls.append(a))
    assert not calls

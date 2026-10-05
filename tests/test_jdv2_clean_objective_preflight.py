"""Pure metadata/CPU contract tests; no models, optimizers or serialized data."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import pytest
from tools import jdv2_clean_objective_preflight as p

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'docs/joint_dependency_v2/reviews/2026-10-05_4f11f44_clean_base_split_objective_preflight'

def read(name):return json.loads((ARCHIVE/name).read_text())

def test_import_has_no_torch_or_model_side_effect():
    code="import sys; from tools import jdv2_clean_objective_preflight; assert 'torch' not in sys.modules; assert 'src.models.model' not in sys.modules"
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)

@pytest.mark.parametrize('sha,names,expected',[(p.HOTEL,['arbitrary_alias.txt'],'outer'),(p.INNER,['another.txt'],'inner_valid'),('different',['biwi_hotel.txt'],'outer'),('different',['uni_examples.txt'],'inner_valid'),('different',['crowds_zara01.txt'],'train')])
def test_entire_source_family(sha,names,expected):assert p.role(sha,names)==expected

def test_alias_cannot_cross_roles():
    with pytest.raises(ValueError):p.assert_source_isolation([{'source_sha256':p.HOTEL,'logical_role':x} for x in ['outer','train']])

def test_agent_time_overlap_and_unknown():
    rows=[dict(source_sha256='s',logical_role=k,agent_ids=[2],frame_ids=[1,2]) for k in ['train','outer']]
    rows.append(dict(source_sha256='t',logical_role='train',agent_ids=None,frame_ids=None))
    out=p.overlap(rows)
    assert out['cross_role']['outer__train']==2 and out['unknown_windows']==1

@pytest.mark.parametrize('role,teacher,purpose', [('outer',True,'training'),('inner_valid',True,'training'),('train',True,'deployment'),('outer',False,'deployment')])
def test_permission_does_not_follow_physical_train(role,teacher,purpose):
    with pytest.raises(PermissionError):p.cache_permission({'logical_role':role,'physical_split':'train'},teacher,purpose)

def test_train_permission_keeps_original_index():
    r=dict(logical_role='train',physical_split='test',physical_index=83)
    p.cache_permission(r,True,'training');assert r['physical_index']==83

@pytest.mark.parametrize('grad,selection,complete,expected',[(['outer'],[],False,'KNOWN_PROTOCOL_VIOLATION'),(['inner_valid'],[],True,'KNOWN_PROTOCOL_VIOLATION'),([],['outer'],False,'KNOWN_PROTOCOL_VIOLATION'),(['train'],['inner_valid'],False,'PROVENANCE_UNVERIFIED'),(['train'],['inner_valid'],True,'QUALIFIED')])
def test_evidence_classes(grad,selection,complete,expected):assert p.classify(grad,selection,complete)==expected

def test_actual_blocked_plan_cli():
    r=subprocess.run([sys.executable,'-B',str(ROOT/'tools/jdv2_clean_objective_preflight.py'),'--check-plan',str(ARCHIVE/'PILOT_PLAN.json')],cwd=ROOT,text=True,capture_output=True)
    assert r.returncode==2 and 'TEMPLATE_BLOCKED' in r.stdout

@pytest.mark.parametrize('fault',['missing_base','producer','loader','step_cap','unqualified'])
def test_ready_label_cannot_override_missing_contract(fault):
    q=dict(status='READY_CLEAN_OBJECTIVE_PILOT_PLAN_NO_EXECUTION',base_path='base',base_sha256='a',cache_path='cache',cache_manifest_sha256='b',initial_state_path='init',initial_state_sha256='c',cache_producer_sha256='a',base_classification='QUALIFIED',loader_supported=True,step_cap_supported=True)
    if fault=='missing_base':q['base_path']=None
    if fault=='producer':q['cache_producer_sha256']='other'
    if fault=='loader':q['loader_supported']=False
    if fault=='step_cap':q['step_cap_supported']=False
    if fault=='unqualified':q['base_classification']='PROVENANCE_UNVERIFIED'
    with pytest.raises(RuntimeError):p.validate_plan(q)

def test_only_allowed_arm_differences():
    pair=read('RESOLVED_CONFIGS.json')['configs'];assert p.compare_configs(pair['A0'],pair['A1'])['passed']
    broken=dict(pair['A1'],learning_rate=.01)
    with pytest.raises(ValueError):p.compare_configs(pair['A0'],broken)

def test_actual_members_and_overlap():
    manifest=read('SPLIT_ROLE_MANIFEST.json')
    assert manifest['counts']==dict(train=3484,inner_valid=320,outer=445)
    assert all(v==0 for v in manifest['overlap']['cross_role'].values())
    assert manifest['overlap']['unknown_windows']==0
    assert len(manifest['duplicate_physical_windows'])==139
    rows=[json.loads(line) for part in manifest['record_files'] for line in (ARCHIVE/part['path']).read_text().splitlines()]
    p.assert_source_isolation(rows)
    assert all(r['logical_role']=='outer' for r in rows if r['source_sha256']==p.HOTEL)
    assert all(r['logical_role']=='inner_valid' for r in rows if r['source_sha256']==p.INNER)
    assert all(r['window_steps']==20 for r in rows)

def test_shared_order_budget_and_warmup():
    order=read('BATCH_ORDER_SEED3101.json');plan=read('PILOT_PLAN.json');progress=read('L1_PROGRESS_TABLE.json')['rows']
    assert len(order['records'])==3484 and order['H']==500 and plan['H']==500
    assert p.stable(order['records'])==order['order_sha256']
    assert len(progress)==500 and progress[0]['beta']==0
    assert progress[-1]['beta']==pytest.approx(.1*(499/3484)/.2)
    assert plan['total_steps']==3484 and plan['accumulation']==1

def test_actual_counts_and_base_component_chain():
    counts=read('EXECUTION.json')['counts'];base=read('BASE_PROVENANCE.json')
    assert counts['checkpoint_deserializations']==5
    assert counts['teacher_cache_reads']==counts['deployment_cache_reads']==1
    for k in ('protected_future_tensor_reads','source_batch_deserializations','outer_metric_computations','real_model_construction_attempts','real_model_call_attempts','cuda_forbidden_attempts','optimizer_updates','data_training','data_evaluation'):assert counts[k]==0
    assert base['counts']=={'KNOWN_PROTOCOL_VIOLATION':5}
    assert all(all(v.values()) for v in base['frozen_component_equality'].values())


def test_cross_recording_collisions_are_not_proven_identity():
    rows=[dict(source_sha256=s,original_source_path=s,logical_role=role,scene_family='univ',agent_ids=[1],frame_ids=[5]) for s,role in [('a','train'),('b','inner_valid')]]
    answer=p.recording_overlap(rows)
    assert answer[0]['shared_agent_frame_identifiers']==1
    assert answer[0]['identity_status'].startswith('UNKNOWN')


def test_cross_recording_scene_namespace_is_respected():
    rows=[dict(source_sha256=s,original_source_path=s,logical_role='train',scene_family=s,agent_ids=[1],frame_ids=[5]) for s in ['univ','hotel']]
    assert p.recording_overlap(rows)==[]

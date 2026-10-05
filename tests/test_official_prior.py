"""Independent input-policy identity, never relabel UNKNOWN as clean VERIFIED."""
import copy,json
from pathlib import Path
import pytest
from src.p2_protocol import ROOT,Registry,sealed,data_binding,file_hash
from src.p2_grouped import qualification_issues,OFFICIAL_PROTOCOL
from src.p2_grouped_training import PackReader,qualify_parent
D=ROOT/'docs/joint_dependency_v2/reviews/2026-10-05_2c25c1f_official_prior_fresh_goal'
OLD=ROOT/'docs/joint_dependency_v2/reviews/2026-10-05_7a0f83b_grouped_univ_complete_preflight'
def registry(d):
    m=json.loads((d/'QUALIFIED_INPUT_REVIEW_MANIFEST.json').read_text());return Registry(m,m['manifest_hash'])
def test_independent_identity_and_original_gate():
    old,new=registry(OLD),registry(D)
    assert qualification_issues(old) and not qualification_issues(new)
    assert data_binding(old)=='73206ec37a642b6702949c898661e2d45f23c27076a935d320e21ce6c88f26a7'
    assert data_binding(new)!=data_binding(old) and new.manifest['protocol_id']==OFFICIAL_PROTOCOL
    assert new.records==old.records and new.manifest['qualification']==old.manifest['qualification']
    assert all(x['semantic_status'].startswith('UNKNOWN') for x in new.manifest['qualification']['scenes'].values())

@pytest.mark.parametrize('field',['preprocessor_training_scope','strict_clean_claim','gradient_roles','selection_roles','outer_metrics','user_authorized','all_arms_same_frozen_inputs','semantic_asset_sha256','author_archive_sha256'])
def test_consent_tampering_not_a_generic_allow_unknown(tmp_path,field):
    new=registry(D);ref=new.manifest['input_policy_acceptance'];v=json.loads((ROOT/ref['path']).read_text())
    v[field]='forged';p=tmp_path/'acceptance.json';p.write_text(json.dumps(v))
    new.manifest['input_policy_acceptance']={'path':str(p),'sha256':file_hash(p)}
    assert any('OFFICIAL_PRIOR_ACCEPTANCE' in x for x in qualification_issues(new))

@pytest.mark.parametrize('role,purpose',[('inner_valid','gradient'),('inner_valid','teacher_build_train'),('outer','selection'),('outer','gradient'),('outer','teacher_build_train')])
def test_original_role_protection_retained(role,purpose):
    reg=registry(D)
    with pytest.raises(PermissionError):reg.authorize(reg.refs_unordered(role)[0]['window_id'],purpose)

def test_variant_typo_and_missing_evidence_rejected():
    reg=registry(D);m=copy.deepcopy(reg.manifest);m['protocol_variant']='allow_unknown';m=sealed(m)
    with pytest.raises(RuntimeError,match='variant'):Registry(m,m['manifest_hash'])
    reg.manifest['input_policy_acceptance']['sha256']='0'*64
    with pytest.raises(RuntimeError,match='OFFICIAL_PRIOR_ACCEPTANCE'):PackReader(reg,'train')

def test_parent_from_clean_binding_not_accepted(tmp_path):
    import torch
    old,new=registry(OLD),registry(D);p=tmp_path/'parent.pt'
    torch.save(dict(schema='grouped-training-state-v1',classification='FORMAL_EPOCH',artifact_role='goal',
        family=old.manifest['family'],data_binding=data_binding(old)),p)
    with pytest.raises(RuntimeError,match='protocol/assets/order'):
        qualify_parent({'path':str(p),'sha256':file_hash(p)},new,'goal')

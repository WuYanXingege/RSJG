#!/usr/bin/env python3
"""Register an explicitly authorized official-prior variant; never edit clean history."""
import argparse,copy,json,shutil,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.p2_protocol import ROOT,Registry,file_hash,data_binding,sealed
from src.p2_grouped import OFFICIAL_PRIOR,OFFICIAL_PROTOCOL,qualification_issues
from src.p2_observation import atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();D=a.archive.resolve();O=a.output.resolve();D.mkdir(parents=True,exist_ok=True);O.mkdir(parents=True,exist_ok=True)
    dest=D/'QUALIFIED_INPUT_REVIEW_MANIFEST.json'
    if dest.exists():raise RuntimeError('new archive required')
    old=ROOT/'docs/joint_dependency_v2/reviews/2026-10-05_7a0f83b_grouped_univ_complete_preflight'
    m=json.loads((old/dest.name).read_text());reg=Registry(m,m['manifest_hash']);old_binding=data_binding(reg)
    release=json.loads((old/'AUTHOR_RELEASE_BINDING.json').read_text())
    consent=dict(schema='rsjg-official-prior-acceptance-v1',protocol_id=OFFICIAL_PROTOCOL,
        user_authorized=True,user_request='现在建立独立协议，验收通过后后台启动 fresh goal',
        accepted_assumption='Official released semantic rasters are fixed environmental priors shared by every arm; their segmentation training scope is not verified.',
        preprocessor_training_scope='UNKNOWN',strict_clean_claim=False,all_arms_same_frozen_inputs=True,
        gradient_roles=['train'],selection_roles=['inner_valid'],outer_metrics=False,
        author_archive_sha256=release['archive_sha256'],
        semantic_asset_sha256={s:x['pred_mask.png']['sha256'] for s,x in m['assets'].items()},
        formal_authority=dict(stage='fresh_goal_only',seed=3101,precision='fp32',epochs_max=150,wall_seconds=86400,
            no_auto_joint=True,no_auto_A1=True),parent_review_commit='2c25c1f29151ec7727606833aa576a0edd0e51e7')
    atomic_json(consent,D/'INPUT_POLICY_ACCEPTANCE.json')
    m.update(protocol_variant=OFFICIAL_PRIOR,protocol_id=OFFICIAL_PROTOCOL,
        input_policy_acceptance=dict(path=str((D/'INPUT_POLICY_ACCEPTANCE.json').relative_to(ROOT)),sha256=file_hash(D/'INPUT_POLICY_ACCEPTANCE.json')))
    m=sealed(m);atomic_json(m,dest);new=Registry(m,m['manifest_hash'])
    assert not qualification_issues(new,('train','inner_valid'))
    assert qualification_issues(reg,('train','inner_valid')) and old_binding!=data_binding(new)
    inherited={}
    for name in ['PREPARATION_MANIFEST.json','ARTIFACT_EXPORT_RESULT.json','ARTIFACT_VERIFY_RESULT.json',
        'FULL_NATIVE_PACK_ARTIFACT_AUDIT.json','OPTIMIZER_LEDGER.jsonl','GPU_BUDGET_LEDGER.jsonl']:
        shutil.copyfile(old/name,D/name);inherited[name]=dict(path=str((old/name).relative_to(ROOT)),sha256=file_hash(old/name))
    cfg=json.loads((old/'FRESH_GOAL_BLOCKED.yaml').read_text())
    run='grouped_fresh_goal_official_prior_seed3101_'+time.strftime('%Y%m%d_%H%M%S',time.gmtime())
    cfg.update(p2_manifest=str(dest.relative_to(ROOT)),p2_manifest_hash=m['manifest_hash'],run_name=run)
    atomic_json(cfg,D/'FRESH_GOAL_CONFIG.yaml')
    atomic_json(dict(status='REGISTERED_NOT_READY',protocol_id=OFFICIAL_PROTOCOL,manifest_hash=m['manifest_hash'],
        data_binding=data_binding(new),strict_clean_data_binding=old_binding,counts=new.counts,run_name=run,
        frozen_assets=m['assets'],inherited_byte_identical_evidence=inherited,
        prior_preflight_attempts=16,remaining_preflight_attempts=8,preflight_total_cap=24,
        policy='new protocol identity within shared grouped role schema; strict clean default unchanged; no record/tensor regeneration',
        formal_training_started=False),D/'REGISTRATION.json')
    print(json.dumps(dict(status='REGISTERED_NOT_READY',data_binding=data_binding(new),run_name=run)))
if __name__=='__main__':main()

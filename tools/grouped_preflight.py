#!/usr/bin/env python3
"""Independent readiness conditions and conditional stage launcher (no auto-chain)."""
import argparse,json,sys,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from src.p2_protocol import ROOT,Registry,digest,file_hash,path
from src.p2_grouped import qualification_issues
from src.p2_grouped_training import qualify_parent
from src.p2_observation import atomic_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['configure','check','qualify-goal','qualify-base'])
    ap.add_argument('--archive',type=Path,required=True);ap.add_argument('--parent',type=Path);ap.add_argument('--sha256')
    a=ap.parse_args();D=a.archive.resolve();mp=D/'QUALIFIED_INPUT_REVIEW_MANIFEST.json';m=json.loads(mp.read_text());reg=Registry(m,m['manifest_hash'])
    if a.action.startswith('qualify-'):
        if not a.parent or not a.sha256:ap.error('actual parent path and SHA required')
        role=a.action.split('-')[1]
        v=qualify_parent(dict(path=str(a.parent),sha256=a.sha256),reg,role)
        print(json.dumps(dict(status='QUALIFIED_READ_ONLY',parent_reference=dict(path=str(a.parent.resolve()),sha256=a.sha256,
            artifact_role='p2_goal' if role=='goal' else 'p2_base',classification='QUALIFIED',data_binding=v['data_binding'],initial_state_sha256=v['initial_state_sha256'],state_sha256=v['state_sha256']))));return
    if a.action=='configure':
        common=dict(p2_manifest=str(mp.relative_to(ROOT)),p2_manifest_hash=m['manifest_hash'],dataset='eth5',test_set='hotel',
            goal_model_type='independent',training_stage='baseline',device='cuda:0',seed=3101,validation_seed=3101,
            batch_size=64,num_workers=0,coupling_grad_accum_steps=1,data_augmentation=False,down_factor=8,
            use_scene_latent=False,use_dynamic_relation=False,use_joint_energy=False,use_dependency_corrector=False,
            optimizer='Adam',scheduler='ExponentialLR',clip=1.,load_checkpoint=None,pretrain_path=None,
            amp_enabled=False,amp_dtype='fp32',final_test_access='blocked',validate_every=1,start_validation=1,
            early_stopping_patience=0,shuffle_train_batches=False,shuffle_test_batches=False,use_wandb=False,
            reproducibility=True,clean_split_protocol=False,num_samples=20,force_reprocess=False)
        configs={
            'FRESH_GOAL_BLOCKED.yaml':dict(common,p2_mode='goal',phase='goal_pretrain',num_epochs=150,learning_rate=.001,run_name='grouped_fresh_goal_seed3101_7a0f83b'),
            'FRESH_JOINT_CONDITIONAL.yaml':dict(common,p2_mode='joint',phase='train',num_epochs=250,learning_rate=.0001,run_name='grouped_fresh_joint_seed3101_7a0f83b',goal_pretrain_checkpoint=None)}
        old=ROOT/'docs/joint_dependency_v2/reviews/2026-10-05_28728bf_p2_role_loader_step_cap_cpu'
        for arm in ('A0','A1'):
            cfg=yaml.safe_load((old/(arm+'_BLOCKED.yaml')).read_text())
            cfg.update(p2_manifest=str(mp.relative_to(ROOT)),p2_manifest_hash=m['manifest_hash'],p2_reference_total_steps=2537,
                test_set='hotel',run_name='grouped_l1_'+arm.lower()+'_seed3101',jdv2_cache_schema='jdv2-p2-grouped-cache-v1')
            configs[arm+'_CONDITIONAL.yaml']=cfg
        for name,cfg in configs.items():
            dest=D/name
            # YAML is generated from an explicit configuration object, not shell text.
            atomic_json(cfg,dest) # JSON is valid YAML; stable byte representation.
        atomic_json(dict(goal_status='EXTERNAL_EVIDENCE_BLOCKED',joint_status='WAITING_FOR_QUALIFIED_GOAL',
            stage_a_status='WAITING_FOR_QUALIFIED_BASE_AND_CACHE',auto_chain=False,
            dependencies={'joint':['qualified goal actual file SHA','fresh history/diffusion'],
                'cache':['qualified complete base','all consumed observations including outer semantic qualification'],
                'stage_a':['same base SHA for both arms','grouped deployment4249 teacher2537','shared untrained initial','reference2537']},
            L1=dict(reference_total_steps=2537,H=500,progress_at_cap=500/2537,beta_at_cap=.1*min((500/2537)/.2,1),
                optimizer='Adam',learning_rate=.0001,arms_seconds=900,pair_seconds=2700,reserved_bytes=10*1024**3,
                A0='mean_energy',A1='expected_conditional_mc S4 independent RNG',new_protocol_not_old_L1=True),
            L2=dict(enabled=False,selector='src.p2_grouped_training.L2Selector',primary='JADE',tie_break='JFDE',min_delta=0.,patience=12,role='inner_valid'),
            formal_budget=dict(goal_epochs_max=150,goal_updates_max=26250,joint_epochs_max=250,joint_updates_max=43750,
                goal_wall_seconds=86400,joint_wall_seconds=172800,selection_each_complete_epoch=True,
                partial_epoch='no selection/scheduler/checkpoint; stop',resume='FRESH_START_ONLY',outer_metrics=False)),D/'PIPELINE_CONDITIONS.json')
        return
    conditions={}
    issues=qualification_issues(reg,('train','inner_valid'))
    conditions['protocol']=dict(status='PASS',family=m['family'],manifest_hash=m['manifest_hash'],counts=reg.counts,rows=m['registered_source_rows_hash'])
    conditions['source_and_maps']=dict(status='EXTERNAL_EVIDENCE_BLOCKED' if issues else 'PASS',issues=issues)
    for name,filename in [('artifact_export','ARTIFACT_EXPORT_RESULT.json'),('artifact_full_readback','ARTIFACT_VERIFY_RESULT.json'),
        ('full_native_pack_artifact','FULL_NATIVE_PACK_ARTIFACT_AUDIT.json'),
        ('CPU_tests','CPU_TEST_RESULT.json'),('CPU_model','MODEL_SMOKE_cpu.json'),('CUDA_model','MODEL_SMOKE_cuda.json'),
        ('independent_reload','MODEL_SMOKE_reload.json'),('fresh_initial','FRESH_INITIAL.json')]:
        p=D/filename
        if not p.exists():conditions[name]=dict(status='NOT_COMPLETED');continue
        v=json.loads(p.read_text());conditions[name]=dict(status=v.get('status','PASS'),evidence=str(p.relative_to(ROOT)),sha256=file_hash(p))
    ip=D/'INNER_SELECTION_RESULT.json'
    conditions['qualified_inner_selection']=json.loads(ip.read_text()) if ip.exists() else dict(status='EXTERNAL_EVIDENCE_BLOCKED' if issues else 'NOT_COMPLETED')
    cfg=D/'FRESH_GOAL_BLOCKED.yaml'
    command=[sys.executable,'-B','main.py','--p2_config',str(cfg),'--p2_launch_check']
    run=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
    conditions['production_entry']=dict(status='PASS' if run.returncode==0 else 'BLOCKED',argv=command,exit=run.returncode,stdout=run.stdout,stderr=run.stderr,config_sha256=file_hash(cfg))
    pp=D/'PRODUCTION_LOADER_RESULT.json'
    conditions['production_full_loader']=json.loads(pp.read_text()) if pp.exists() else dict(status='EXTERNAL_EVIDENCE_BLOCKED' if issues else 'NOT_COMPLETED',note='artifact/preparation traversal is not qualified production traversal')
    status='EXTERNAL_EVIDENCE_BLOCKED' if issues else 'FAILED_INCOMPLETE_ACCEPTANCE'
    if not issues and all(v['status'] in {'PASS','ARTIFACT_VALIDATION_ONLY_NOT_PRODUCTION','UNUPDATED_INITIAL_NOT_QUALIFIED_PARENT'} for v in conditions.values()):status='GOAL_TRAINING_READY'
    result=dict(status=status,conditions=conditions,joint='WAITING_FOR_QUALIFIED_GOAL',stage_a='WAITING_FOR_QUALIFIED_BASE_AND_CACHE',formal_training_started=False)
    atomic_json(result,D/'READINESS_MATRIX.json');print(json.dumps(result));raise SystemExit(0 if status=='GOAL_TRAINING_READY' else 2)
if __name__=='__main__':main()

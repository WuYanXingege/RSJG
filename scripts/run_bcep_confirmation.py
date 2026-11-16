"""Exactly four authorized bounded runs, then one CPU confirmation; no followup queue."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='2'
import argparse,fcntl,json,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bcep_pilot_runtime import configure,assert_gpu_idle,environment
import torch
from bcep.contracts import digest,canonical_hash
from bcep_confirmation.common import ROOT,read,write,source_identity
from bcep_confirmation.protocol import PurposeDataset,validate_split
from bcep_confirmation.runner import run_pilot
from bcep_confirmation.checks import gpu_checks
from bcep_confirmation.evaluation import evaluate

def git(*args,cwd=ROOT):return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()
def protected(plan,after=False):
    root=Path(plan['assets']);path=root/'identity/PROTECTED_BEFORE.json'
    if not after:
        previous=read(ROOT/'docs/research/bcep_v1/bcep_frozen_selector_control_20261116_215702/PROTECTION_BEFORE_AFTER.json')
        paths={r['path']:r['after_sha256'] for r in previous['files']}
        old=ROOT.parent/'RSJG_EBJD_BCEP'
        for p in (old/'docs/research/bcep_v1/bcep_frozen_selector_control_20261116_215702').iterdir():
            if p.is_file():paths[str(p)]=digest(p)
        outputs=read(old/'docs/research/bcep_v1/bcep_frozen_selector_control_20261116_215702/OUTPUTS_FROZEN.json')
        for r in outputs['banks']:paths[r['path']]=r['sha256']
        for folder in ('bcep','bcep_control','selector_control'):
            for p in (ROOT/folder).rglob('*.py'):paths[str(p)]=digest(p)
        for p,h in paths.items():assert digest(p)==h,'protected artifact differs: '+p
        worktrees={name:dict(head=git('rev-parse','HEAD',cwd=ROOT.parent/name),status=git('status','--porcelain',cwd=ROOT.parent/name)) for name in ('RSJG_EBJD_Paper','RSJG_EBJD_BCEP')}
        data=dict(files=[dict(path=p,sha256=h) for p,h in paths.items()],worktrees_before=worktrees);write(path,data)
    else:
        data=read(path)
        for r in data['files']:r['after_sha256']=digest(r['path']);r['unchanged']=r['after_sha256']==r['sha256']
        data['worktrees_after']={name:dict(head=git('rev-parse','HEAD',cwd=ROOT.parent/name),status=git('status','--porcelain',cwd=ROOT.parent/name)) for name in data['worktrees_before']}
        data['passed']=all(r['unchanged'] for r in data['files']) and data['worktrees_before']==data['worktrees_after']
        write(root/'reports/PROTECTION_RECEIPT.json',data);write(Path(plan['docs'])/'PROTECTION_RECEIPT.json',data);assert data['passed']
    return data

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--execute-authorized-four-runs',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if not args.execute_authorized_four_runs:raise SystemExit('REAL_TRAINING_DISABLED: explicit bounded execution switch required')
    plan=read(args.plan);root=Path(plan['assets']);docs=Path(plan['docs']);claim=root/'RUN_CLAIM.json'
    lock=(root/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    expected=[('bcep_flow',3102),('mdn8',3102),('mdn8',3103),('bcep_flow',3103)]
    assert [(r['model_kind'],r['seed']) for r in plan['authorized_run_matrix']]==expected
    assert digest(plan['split_lock'])==plan['split_sha256'] and digest(plan['contract'])==plan['contract_sha256']
    split=read(plan['split_lock']);validate_split(split);contract=read(plan['contract']);policy=configure()
    identity=dict(schema='bcep-confirmation-checkpoint-v1',split=canonical_hash(split),contract=digest(plan['contract']),source=source_identity(),source_commit=git('rev-parse','HEAD'),weights_origin='random_initialization',training_purpose='train',selection_purpose='selection_dev',heldout_purpose='heldout_confirmation')
    if claim.exists():
        if not args.resume:raise RuntimeError('run already claimed; no duplicate start')
        prior=read(claim);assert prior['identity']==identity and prior['plan_sha256']==digest(args.plan)
        if read(root/'PIPELINE_STATE.json')['status']=='COMPLETE':raise RuntimeError('completed task cannot resume')
    else:
        assert not args.resume
        with claim.open('x') as f:json.dump(dict(pid=os.getpid(),created_unix=time.time(),plan_sha256=digest(args.plan),identity=identity),f,indent=2)
        protected(plan)
    state=dict(status='RUNNING',phase='PREFLIGHT',pid=os.getpid(),formal_optimizer_updates=0,started_unix=time.time(),automatic_followup=False)
    write(root/'PIPELINE_STATE.json',state);all_runs=[]
    try:
        assert read(root/'reports/CPU_PREFLIGHT.json')['status']=='PASS'
        assert_gpu_idle();write(root/'identity/ENVIRONMENT.json',dict(**environment(),precision=policy))
        if not (root/'reports/GPU_PREFLIGHT.json').exists():gpu_checks(root)
        assert read(root/'reports/GPU_PREFLIGHT.json')['status']=='PASS'
        data=PurposeDataset(split,'training')
        for row in plan['authorized_run_matrix']:
            state.update(phase='TRAIN',current_run=row['name']);write(root/'PIPELINE_STATE.json',state)
            output=root/'training'/row['name'];cfg=dict(device='cuda',seed=row['seed'],model_kind=row['model_kind'],max_updates=6*len(split['train_indices']),max_epochs=6,budget_seconds=3600,lr=1e-4,lr_gamma=1.,lambda_aux=.25,patience=3,support=256,candidates=20,allow_real_training=False)
            assert cfg['max_updates']==row['max_updates'] and row['budget_seconds']==3600
            run_identity={**identity,'model_kind':row['model_kind'],'training_seed':row['seed']}
            write(root/'identity'/f'{row["name"]}_CONFIG.json',cfg)
            started=time.monotonic();before=None
            if (output/'last.pt').exists():
                assert args.resume;before=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
                if before['stop_reason']=='INTERRUPTED_OR_NUMERICAL_ERROR':raise RuntimeError('explicit failure diagnosis required before retrying this run')
            def heartbeat(index,update):
                if update%100==0:write(root/'TRAINING_PROGRESS.json',dict(run=row['name'],update=update,index=index,invocation_wall_seconds=time.monotonic()-started))
            result=run_pilot(data,split,cfg,run_identity,output,resume=before is not None,on_update=heartbeat,workflow_started=started)
            measured=read(output/'RUN_STATE.json')['elapsed_wall_seconds']
            entry=dict(**row,identity=run_identity,config_hash=canonical_hash(cfg),completed_epoch=result['completed_epoch'],last_epoch=result['epoch'],last_update=result['update'],phase=result['phase'],stop_reason=result['stop_reason'],
                cumulative_wall_seconds=measured,initialization=read(output/'INITIALIZATION.json'),last=dict(path=str(output/'last.pt'),sha256=digest(output/'last.pt')),best=None,resume_invocations=int(before is not None))
            if (output/'best.pt').exists():
                best=torch.load(output/'best.pt',map_location='cpu',weights_only=False)
                assert best['identity']==run_identity and best['completed_epoch']==best['selected_epoch']
                entry['best']=dict(path=str(output/'best.pt'),sha256=digest(output/'best.pt'),epoch=best['completed_epoch'],update=best['update'],selection_value=best['selection_value'])
            else:entry['evidence']='INSUFFICIENT_TRAINING_EVIDENCE'
            expected_init=read(root/'reports/CPU_PREFLIGHT.json')['initial_hashes'][str(row['seed'])][row['model_kind']]
            assert entry['initialization']['model_sha256']==expected_init['model'] and entry['initialization']['encoder_sha256']==expected_init['encoder']
            all_runs.append(entry);state['formal_optimizer_updates']=sum(r['last_update'] for r in all_runs)
            write(root/'reports/TRAINING_RECEIPTS.json',all_runs);write(docs/'TRAINING_RECEIPTS.json',all_runs)
            print('RUN_TERMINATED '+json.dumps({k:entry[k] for k in ['name','completed_epoch','last_update','stop_reason','cumulative_wall_seconds']}),flush=True)
            del result,before;torch.cuda.empty_cache()
        frozen=dict(status='ALL_RUNS_TERMINAL',created_unix=time.time(),split_hash=canonical_hash(split),runs=all_runs)
        write(root/'reports/CHECKPOINTS_FROZEN.json',frozen);write(docs/'CHECKPOINTS_FROZEN.json',frozen)
        state.update(phase='CPU_CONFIRMATION');write(root/'PIPELINE_STATE.json',state)
        evaluate(plan,split,frozen)
        state['status']='COMPLETE' if (root/'reports/AGGREGATES.json').exists() else 'EVALUATION_INCOMPLETE'
    except BaseException as exc:
        state.update(status='FAILED_OR_INCOMPLETE',error=repr(exc),traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
    finally:
        try:protected(plan,after=True)
        except BaseException as exc:state.update(status='FAILED_OR_INCOMPLETE',protection_error=repr(exc))
        state.update(finished_unix=time.time(),active=False,default_allow_real_training=False,automatic_followup=False)
        write(root/'PIPELINE_STATE.json',state);write(docs/'PIPELINE_RECEIPT.json',state)
    return 0 if state['status']=='COMPLETE' else 1
if __name__=='__main__':sys.exit(main())

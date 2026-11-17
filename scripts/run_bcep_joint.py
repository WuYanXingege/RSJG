"""Explicitly authorized, bounded two-provider/single-downstream-seed pilot."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,fcntl,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bcep_joint.common import read,write,digest,canonical_hash,ROOT,source_identity

def protection(plan):
    root=Path(plan['assets']);old=read(root/'identity/PROTECTED_BEFORE.json');files=[]
    for r in old['files']:
        actual=digest(r['path']);assert actual==r['sha256'],r['path'];files.append(r)
    for name,state in old['worktrees'].items():
        actual=dict(head=subprocess.check_output(['git','-C',str(ROOT.parent/name),'rev-parse','HEAD'],text=True).strip(),status=subprocess.check_output(['git','-C',str(ROOT.parent/name),'status','--porcelain'],text=True).strip())
        assert actual==state,(name,actual,state)
    result=dict(status='PASS',files=files,worktrees=old['worktrees'],checked_unix=time.time())
    write(root/'reports/PROTECTION_RECEIPT.json',result);write(Path(plan['docs'])/'PROTECTION_RECEIPT.json',result)

def main(args):
    plan=read(args.plan);root=Path(plan['assets']);docs=Path(plan['docs'])
    assert digest(plan['split_path'])==plan['split_file_sha256'] and digest(plan['config_path'])==plan['config_sha256']
    split=read(plan['split_path']);c=read(plan['config_path']);assert canonical_hash(split)==plan['split_hash']
    assert len(plan['runs'])==2 and [r['provider_seed'] for r in plan['runs']]==[3102,3103]
    if args.cpu_checks:
        from bcep_joint.checks import cpu_checks
        cpu_checks(plan);protection(plan);return
    if not args.execute_authorized_two_runs:raise SystemExit('Default training is OFF; explicit task authorization flag required')
    assert read(root/'reports/CPU_PRECHECK.json')['status']=='PASS'
    handle=(root/'receipts/PILOT.lock').open('w');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'identity/EXECUTION_IDENTITY.json').exists(),'single execution already claimed; inspect its state, do not duplicate'
    assert subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()==''
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    source=source_identity();identity=dict(execution_commit=commit,source_hash=canonical_hash(source),config_sha256=plan['config_sha256'],split_hash=plan['split_hash'],pid=os.getpid(),started_unix=time.time(),allow_real_training=True,authorization=plan['authorization'])
    write(root/'identity/EXECUTION_IDENTITY.json',identity);write(root/'identity/SOURCE_FILES.json',source);write(docs/'EXECUTION_IDENTITY.json',identity);write(docs/'SOURCE_FILES.json',source)
    status=dict(status='RUNNING',stage='GPU_PRECHECK',pid=os.getpid(),default_allow_real_training=False)
    write(root/'PIPELINE_STATE.json',status)
    training=[];failure=None
    try:
        from bcep_joint.runtime import configure,assert_gpu_idle,environment
        assert_gpu_idle();policy=configure();write(root/'identity/ENVIRONMENT.json',dict(precision=policy,environment=environment()))
        from bcep_joint.checks import gpu_checks
        checked=root/'reports/GPU_PRECHECK.json'
        if not checked.exists() or read(checked).get('source_hash')!=canonical_hash(source):gpu_checks(plan)
        else:assert read(checked)['status']=='PASS'
        from bcep_joint.data import JointData,CandidateBank
        from bcep_joint.runner import run
        import torch
        data=JointData(split)
        for row in plan['runs']:
            assert source_identity()==source
            status.update(stage='TRAINING',run=row['name']);write(root/'PIPELINE_STATE.json',status)
            output=root/'training'/row['name'];rid=dict(**identity,run=row['name'],provider_sha256=row['provider']['sha256'],bank_train_sha256=row['banks']['train_2035']['sha256'],bank_dev_sha256=row['banks']['selection_dev_2035']['sha256'])
            for bank in row['banks'].values():assert digest(bank['path'])==bank['sha256']
            train=CandidateBank(row['banks']['train_2035']['path'],split,'train',row['provider']['sha256'],2035)
            dev=CandidateBank(row['banks']['selection_dev_2035']['path'],split,'selection_dev',row['provider']['sha256'],2035)
            torch.cuda.reset_peak_memory_stats();t=time.monotonic()
            try:state=run(c,split,data,train,dev,rid,output)
            finally:
                elapsed=time.monotonic()-t
                receipt=dict(seconds=elapsed,limit_seconds=7200,finite_overrun_seconds=max(0.,elapsed-7200),includes='model initialization, training, dev validation, saves; stop at optimizer boundary',run=row['name'],peak_allocated_bytes=torch.cuda.max_memory_allocated())
                write(root/'receipts'/f"TRAINING_{row['name']}_BUDGET.json",receipt)
            s=read(output/'STATUS.json');entry=dict(name=row['name'],identity=rid,status=s,initialization=read(output/'INITIALIZATION_ORIGIN.json'),best=None,last=dict(path=str(output/'last.pt'),sha256=digest(output/'last.pt')),budget=receipt)
            if (output/'best.pt').exists():entry['best']=dict(path=str(output/'best.pt'),sha256=digest(output/'best.pt'),**s['selection']['best'])
            training.append(entry);write(root/'reports/TRAINING.json',training)
            assert state['update']<=c['max_updates'] and state['status'] in ('COMPLETE','BUDGET_EXHAUSTED')
            del state,train,dev;torch.cuda.empty_cache()
        assert training[0]['initialization']['model']==training[1]['initialization']['model']==read(root/'reports/CPU_PRECHECK.json')['downstream_initial_hash']
        assert training[0]['initialization']['optimizer_initial_sha256']==training[1]['initialization']['optimizer_initial_sha256']
        orders=[read(root/'training'/r['name']/'ORDER_RECEIPT.json') for r in plan['runs']]
        paired_epochs=min(map(len,orders));assert orders[0][:paired_epochs]==orders[1][:paired_epochs]
        write(root/'reports/TRAINING_ORDER_RECEIPT.json',dict(status='PASS',matched_epoch_starts=paired_epochs,orders=orders,initial_optimizer_equal=True))
        write(docs/'TRAINING_ORDER_RECEIPT.json',read(root/'reports/TRAINING_ORDER_RECEIPT.json'))
        frozen=dict(status='ALL_RUNS_TERMINAL',runs=training,default_allow_real_training=False,frozen_unix=time.time())
        write(root/'reports/CHECKPOINTS_FROZEN.json',frozen);write(docs/'CHECKPOINTS_FROZEN.json',frozen)
        status.update(stage='FROZEN_EVALUATION');write(root/'PIPELINE_STATE.json',status)
        from bcep_joint.evaluation import evaluate
        evaluate(plan,split,c,frozen)
        status.update(status=read(root/'reports/EVALUATION_STATUS.json')['status'],stage='FINISHED')
    except BaseException as e:
        failure=repr(e);status.update(status='FAILED_OR_INCOMPLETE',failure=failure,traceback=traceback.format_exc());traceback.print_exc()
    finally:
        try:protection(plan)
        except BaseException as e:status.update(status='PROTECTION_CHECK_FAILED',protection_failure=repr(e))
        status.update(ended_unix=time.time(),active=False,default_allow_real_training=False,no_autoqueue=True)
        write(root/'PIPELINE_STATE.json',status);write(docs/'PIPELINE_STATE.json',status)
        for name in ['CPU_PRECHECK.json','GPU_PRECHECK.json','TRAINING.json','EVALUATION_STATUS.json','BANK_ADAPTER_RECEIPT.json']:
            p=root/'reports'/name
            if p.exists():write(docs/name,read(p))
        write(docs/'BUDGET.json',{p.name:read(p) for p in (root/'receipts').glob('*BUDGET.json')})
    if status['status']!='COMPLETE':raise SystemExit(1)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--cpu-checks',action='store_true');parser.add_argument('--execute-authorized-two-runs',action='store_true');main(parser.parse_args())

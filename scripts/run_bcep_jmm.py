"""Finite five-fold supervisor. No polling/cron and no GPU contention."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,sys,json,time,subprocess,fcntl,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bcep_joint.common import read,write,canonical_hash,digest
from bcep_formal.control import install
from bcep_formal.banks import Paused

def sources():
 root=Path(__file__).resolve().parents[1];paths=[]
 for name in ['ebjd','bcep','bcep_joint','bcep_formal','selector_control']:paths+=sorted((root/name).rglob('*.py'))
 paths += [Path(__file__),root/'scripts/run_zara1_unified_coupling_bridge.py',root/'scripts/check_bcep_jmm.py',root/'scripts/finalize_bcep_jmm_runtime.py']
 return {str(p.relative_to(root)):digest(p) for p in paths}

def main(args):
 if not args.execute_authorized_fivefold:raise SystemExit('Explicit --execute-authorized-fivefold required')
 plan=read(args.plan);a=Path(plan['assets']);identity_files=sources();code=canonical_hash(identity_files)
 assert digest(plan['split_lock'])==plan['split_sha256'] and plan['fold_order']==['zara2','zara1','hotel','univ','eth']
 assert plan['authorization']['explicit_fivefold_authorization']
 assert read(a/'receipts/CPU_PREFLIGHT.json')['status']=='PASS'
 for f in plan['folds'].values():
  assert digest(f['split']['path'])==f['split']['sha256']
  for k in ['provider_config','joint_config']:assert digest(f[k])==f[k+'_sha256']
 # No automatic resource waiter: an occupied GPU is an explicit exit, as package requires.
 from bcep_joint.runtime import assert_gpu_idle,configure
 gate_path=a/'receipts/GPU_PREFLIGHT.json'
 if not gate_path.exists() or read(gate_path).get('status')!='PASS' or read(gate_path).get('source_hash')!=code:
  write(a/'PIPELINE_STATE.json',dict(status='GPU_GATE_REQUIRED',formal_optimizer_updates=0));raise SystemExit('Run scripts/check_bcep_jmm.py --gpu after current GPU task finishes')
 runtime=read(a/'receipts/RUNTIME_FROZEN.json');assert runtime['plan_sha256']==digest(args.plan) and runtime['source_hash']==code
 handle=(a/'PIPELINE.lock').open('w');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
 project_lock=(a.parents[1]/'bcep_jmm_formal_gpu.lock').open('w');fcntl.flock(project_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:assert_gpu_idle()
 except Exception as e:
  write(a/'PIPELINE_STATE.json',dict(status='RESOURCE_BUSY_NOT_QUEUED',reason=str(e)));raise
 configure();install();import torch
 from bcep_formal.data import FormalData
 from bcep_formal.provider_runner import run_provider
 from bcep_formal.joint_runner import run_joint
 from bcep_formal.banks import export_bank
 from bcep_formal.evaluate import evaluate
 from bcep_joint.data import CandidateBank
 from bcep_formal.summary import summarize
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=plan['worktree'],text=True).strip()
 execution=dict(commit=commit,source_hash=code,plan_sha256=digest(args.plan),seed_pair=[3102,3201])
 ex=a/'identity/EXECUTION_IDENTITY.json'
 if ex.exists():assert read(ex)==execution,'cannot silently migrate an existing execution'
 else:write(ex,execution)
 state=read(a/'PIPELINE_STATE.json');stage='START';fold=None;stage_fold=None;stage_started=time.monotonic();costs=[]
 costs_file=a/'receipts/COMPUTE_COSTS.json'
 if costs_file.exists():costs=read(costs_file)
 def finish_stage():
  if stage!='START':
   costs.append(dict(fold=stage_fold,stage=stage,occupied_wall_seconds=time.monotonic()-stage_started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),includes_cpu_io=True));write(costs_file,costs)
 try:
  for fold in plan['fold_order']:
   assert sources()==identity_files
   r=plan['folds'][fold];s=read(r['split']['path']);out=Path(r['output']);ps=read(r['provider_config']);js=read(r['joint_config']);ident=dict(execution=execution,fold=fold,split=canonical_hash(s),initialization='random_no_pretrained_weights')
   done=out/'FOLD_COMPLETE.json'
   if done.exists():assert read(done)['identity']==ident;continue
   def progress(name):
    nonlocal stage,stage_started,stage_fold
    finish_stage();stage_started=time.monotonic();torch.cuda.reset_peak_memory_stats();stage=name;stage_fold=fold;write(a/'PIPELINE_STATE.json',dict(status='RUNNING',stage=stage,fold=fold,pid=os.getpid()))
   progress('PROVIDER');data=FormalData(s);state=run_provider(data,s,ps,ident,out/'provider',resume=(out/'provider/last.pt').exists())
   if state['phase']!='complete':raise Paused('provider saved complete resumable state')
   provider=out/'provider/best.pt';assert provider.exists();p_sha=digest(provider);write(out/'PROVIDER_FROZEN.json',dict(path=str(provider),sha256=p_sha,selected_epoch=state['selected_epoch'],identity=ident));del state,data;torch.cuda.empty_cache()
   progress('BANKS_TRAIN_VAL');banks={}
   for purpose in ['train','selection_dev']:banks[purpose]=export_bank(s,provider,purpose,2035,out/'banks'/f'{purpose}_2035.npz')
   progress('JOINT');data=FormalData(s);tr=CandidateBank(banks['train']['path'],s,'train',p_sha,2035);dev=CandidateBank(banks['selection_dev']['path'],s,'selection_dev',p_sha,2035)
   ji=dict(**ident,provider_sha256=p_sha,bank_train_sha256=tr.sha,bank_dev_sha256=dev.sha)
   status=out/'joint/STATUS.json'
   if not status.exists() or read(status)['status']!='COMPLETE':
    state=run_joint(js,s,data,tr,dev,ji,out/'joint',resume=(out/'joint/last.pt').exists())
    if state['phase']!='complete':raise Paused('joint saved complete resumable state')
    del state
   best=out/'joint/best.pt';assert best.exists();write(out/'JOINT_FROZEN.json',dict(path=str(best),sha256=digest(best),selection=read(status)['selection']['best'],identity=ji));del data,tr,dev;torch.cuda.empty_cache()
   progress('TEST_BANKS');test={str(seed):export_bank(s,provider,'heldout_confirmation',seed,out/'banks'/f'test_{seed}.npz') for seed in range(2035,2040)}
   progress('FINAL_EVALUATION');result=evaluate(s,js,best,test,out/'evaluation');write(done,dict(identity=ident,result=result))
  summarize(plan)
  write(a/'PIPELINE_STATE.json',dict(status='COMPLETE',folds=plan['fold_order'],automatic_followup=False))
 except Paused as e:write(a/'PIPELINE_STATE.json',dict(status='PAUSED_RESUMABLE',stage=stage,fold=fold,reason=str(e)))
 except BaseException as e:
  write(a/'PIPELINE_STATE.json',dict(status='FAILED',stage=stage,fold=fold,reason=repr(e)));traceback.print_exc();raise
 finally:
  finish_stage()
  write(a/'receipts/LAST_INVOCATION.json',dict(ended_unix=time.time(),state=read(a/'PIPELINE_STATE.json'),execution=execution))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--execute-authorized-fivefold',action='store_true');main(p.parse_args())

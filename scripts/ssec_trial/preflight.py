"""Explicit real-training authorization preflight; no optimizer construction."""
import os, shutil, sys, time
from common import *
sys.path.insert(0,str(TRAIN_SOURCE))
from bcep_scene_energy.formal_contract import locks,paths
from bcep_v3.formal_inputs import FormalInputs


def main():
    assert not (RUN/'LAUNCH.json').exists() and not (RUN/'joint/last.pt').exists()
    config=source_check();cmd=command()
    ready_path=TRAIN_SOURCE/'docs/research/bcep_scene_energy/stage2_ready_20261121/READINESS.json'
    ready=read(ready_path);assert ready['status']=='READY_FOR_SEPARATE_TRAINING_AUTHORIZATION'
    assert ready['frozen_file_sha256']==FROZEN and all(ready['gates'].values())
    audit_source=SOURCE.parent/'RSJG_SSEC_synthetic_recovery_20261121'
    assert git('rev-parse','HEAD',cwd=audit_source)==RECOVERY_BASE and not git('status','--porcelain',cwd=audit_source)
    recovery_path=audit_source/'docs/research/bcep_scene_energy/synthetic_nonzero_recovery_20261121/READINESS.json'
    recovery=read(recovery_path);assert recovery['status']=='PASS_SYNTHETIC_NONZERO_RECOVERY'
    assert recovery['frozen_formal_config_sha256']==FROZEN and recovery['real_optimizer_updates']==0
    roots=paths()
    with locks(roots):
        inputs=FormalInputs()
        assert inputs.config==config['config'] and inputs.recipe==config['recipe']
        states={key:read(roots[key]/'STATUS.json')['status'] for key in ['formal','v3','candidate_cost','sharedworld','run']}
        assert states==dict(formal='PAUSED_RESUMABLE',v3='COMPLETE',candidate_cost='SESSION_BUDGET_PAUSED',sharedworld='COMPLETE',run='PREPARED_NO_TRAINING_AUTHORIZATION')
        protected={}
        for key in ['formal','v3','candidate_cost','sharedworld']:
            for p in sorted(roots[key].rglob('*')):
                if p.is_file():protected[key+'/'+str(p.relative_to(roots[key]))]=dict(bytes=p.stat().st_size,sha256=digest(p))
        write(RUN/'PROTECTED_ASSETS_BEFORE.json',protected)
        # Exclude own newly written execution receipts from historical protection.
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.free,memory.total','--format=csv,noheader'],text=True).strip()
        jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
        assert not jobs,'GPU already has a compute workload'
        free=shutil.disk_usage(RUN).free
        assert free>20*2**30
        receipt=dict(passed=True,runtime_code_commit=BASE,recovery_commit=RECOVERY_BASE,
            frozen_file_sha256=FROZEN,implementation_hash=config['implementation_hash'],
            ready_receipt_sha256=digest(ready_path),recovery_receipt_sha256=digest(recovery_path),
            states=states,locks_free=True,GPU=gpu,GPU_compute_jobs=jobs,disk_free_bytes=free,
            counts=inputs.counts,provider=inputs.provider,bank_sha256=config['bank_sha256'],split_sha256=config['split_sha256'],
            initial_model_sha256=config['initial_model_sha256'],protected_files=len(protected),
            authorization='user explicitly authorized unique SSEC Energy-only ZARA2 Stage2; no followup experiments',
            exact_command=cmd,real_updates_before_launch=0,unix=time.time())
        write(RUN/'PREFLIGHT.json',receipt)
        print('PREFLIGHT PASS',len(protected),'protected files; no optimizer constructed',flush=True)


if __name__=='__main__':main()

"""Single-use precision wrapper around the unchanged training CLI; no queue."""
import argparse,json,os,sys,time,traceback,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bcep_pilot_runtime import configure,environment,assert_gpu_idle,script_module,ROOT
import torch,yaml
from bcep.contracts import code_identity,digest
from bcep.pilot_runner import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    plan=json.loads(a.plan.read_text());root=Path(plan['asset_root']);gate=json.loads((root/'reports/GPU_GATE_REPORT.json').read_text())
    assert gate['status']=='PASS' and gate['real_optimizer_updates']==0
    assert gate['source_identity']==code_identity() and gate['config_sha256']==digest(a.config)
    assert_gpu_idle();policy=configure();assert policy==gate['precision']
    cfg=yaml.safe_load(a.config.read_text());claim=root/'identity/PILOT_CLAIM.json'
    if not a.resume:
        with claim.open('x') as f:json.dump(dict(pid=os.getpid(),config_hash=digest(a.config),run_id=plan['run_id'],real_run_number=1),f)
        assert not (root/'training/last.pt').exists()
    else:
        original=json.loads(claim.read_text());assert original['config_hash']==digest(a.config)
        assert (root/'training/last.pt').exists()
    identity=dict(run_id=plan['run_id'],actual_execution_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_identity=code_identity(),precision=policy,environment=environment(),config_sha256=digest(a.config),
        wrappers={name:digest(ROOT/'scripts'/name) for name in ['bcep_one_pilot.py','bcep_pilot_runtime.py','bcep_gpu_gate.py','bcep_pilot_readonly.py']},
        command=sys.argv,epoch_cap=10,update_cap=37900,workflow_seconds_cap=3600,gate_report_sha256=digest(root/'reports/GPU_GATE_REPORT.json'))
    identity_path=root/'identity/PILOT_EXECUTION_IDENTITY.json'
    if a.resume:
        prior=json.loads(identity_path.read_text());assert prior['source_identity']==identity['source_identity'] and prior['wrappers']==identity['wrappers'] and prior['precision']==policy
    else:atomic_json(identity,identity_path)
    started=time.monotonic();report=dict(status='RUNNING',resume=a.resume,precision=policy)
    atomic_json(report,root/'reports/TRAINING_RESULT.json')
    try:
        torch.cuda.reset_peak_memory_stats()
        argv=['--config',str(a.config)]+(['--resume'] if a.resume else [])
        state=script_module('scripts/train_bcep.py').main(argv)
        torch.cuda.synchronize()
        report.update(status='FINISHED',stop_reason=state.get('stop_reason'),completed_epoch=state['completed_epoch'],
            next_or_partial_epoch=state['epoch'],phase=state['phase'],train_cursor=state['train_cursor'],dev_cursor=state['dev_cursor'],
            optimizer_updates=state['update'],selected_epoch=state['selected_epoch'],selection_value=state['selection_value'],
            selection_metric_name=state['selection_metric_name'],stale=state['stale'],runner_elapsed_wall_seconds=state['elapsed_wall_seconds'])
        for name in ['last.pt','best.pt']:
            path=root/'training'/name
            if path.exists():report[name]=dict(path=str(path),sha256=digest(path))
        report['run_state']=json.loads((root/'training/RUN_STATE.json').read_text())
        report['has_selected_best']=(root/'training/best.pt').exists() and state['completed_epoch']>=1
    except Exception as e:
        report.update(status='NUMERICAL_OR_EXECUTION_FAILURE',failure=str(e),traceback=traceback.format_exc());raise
    finally:
        report['invocation_wall_seconds']=time.monotonic()-started
        if torch.cuda.is_initialized():report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        atomic_json(report,root/'reports/TRAINING_RESULT.json')
        print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()

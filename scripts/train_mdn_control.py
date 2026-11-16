"""One authorized MDN8 run, separate checkpoint schema and frozen identity."""
import argparse,json,os,sys,time,traceback,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bcep_pilot_runtime import configure,environment,assert_gpu_idle,ROOT
import torch,yaml
from bcep.contracts import load_locked_split,digest,require
from bcep.pilot_runner import atomic_json
from bcep_control.contracts import checkpoint_identity,control_code_identity
from bcep_control.pilot_runner import run_pilot
from ebjd.data import SceneManifestDataset


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    cfg=yaml.safe_load(a.config.read_text());require(cfg.get('allow_real_training') is True,'REAL_TRAINING_DISABLED')
    plan=json.loads((ROOT/cfg['plan']).read_text());root=Path(plan['assets'])
    gate=json.loads((root/'reports/MDN_TEST_REPORT.json').read_text());assert gate['status']=='PASS'
    assert gate['source_identity']==control_code_identity() and gate['config_sha256']==digest(a.config)
    assert json.loads((root/'reports/FROZEN_BCEP_REGRESSION.json').read_text())['status']=='PASS'
    assert_gpu_idle();policy=configure();assert policy==gate['precision']
    started=time.monotonic()
    claim=root/'identity/MDN_CLAIM.json'
    if not a.resume:
        with claim.open('x') as f:json.dump(dict(pid=os.getpid(),config_sha256=digest(a.config),real_run_number=1),f)
        assert not (root/'mdn_training/last.pt').exists()
    else:assert json.loads(claim.read_text())['config_sha256']==digest(a.config)
    identity_record=dict(execution_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                         source_identity=control_code_identity(),config_sha256=digest(a.config),command=sys.argv,
                         precision=policy,environment=environment(),initial_encoder_hash=gate['initial_encoder_hash'],
                         initial_mdn_hash=gate['initial_mdn_hash'])
    identity_path=root/'identity/MDN_EXECUTION_IDENTITY.json'
    if a.resume:assert json.loads(identity_path.read_text())['source_identity']==identity_record['source_identity']
    else:atomic_json(identity_record,identity_path)
    report=dict(status='RUNNING',model_kind='mdn8_full_covariance_v1',resume=a.resume)
    atomic_json(report,root/'reports/MDN_TRAINING_RESULT.json')
    try:
        split_path=ROOT/cfg['split_lock'];contract_path=ROOT/cfg['contract']
        split,contract,_=load_locked_split(split_path,contract_path)
        assert cfg['max_epochs']==6 and cfg['max_updates']==22740 and cfg['budget_seconds']==3600
        assert cfg['seed']==3101 and cfg['patience']==3 and cfg['lr']==1e-4 and cfg['lr_gamma']==1 and cfg['lambda_aux']==.25
        assert cfg['max_updates']==len(split['train_indices'])*cfg['max_epochs']
        torch.cuda.reset_peak_memory_stats()
        state=run_pilot(SceneManifestDataset(split['source_manifest']),split,cfg,checkpoint_identity(contract_path,split_path),
                        ROOT/cfg['output'],resume=a.resume,workflow_started=started)
        torch.cuda.synchronize()
        for key in ['stop_reason','completed_epoch','epoch','phase','train_cursor','dev_cursor','update','selected_epoch','selection_value','stale','elapsed_wall_seconds']:
            report[key]=state[key]
        report['status']='FINISHED';report['run_state']=json.loads((root/'mdn_training/RUN_STATE.json').read_text())
        for name in ['best.pt','last.pt']:
            path=root/'mdn_training'/name
            if path.exists():report[name]=dict(path=str(path),sha256=digest(path))
    except BaseException as error:
        report.update(status='FAILED',failure=repr(error),traceback=traceback.format_exc());raise
    finally:
        report['wall_seconds']=time.monotonic()-started
        report['peak_allocated_bytes']=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
        atomic_json(report,root/'reports/MDN_TRAINING_RESULT.json');print(json.dumps(report),flush=True)


if __name__=='__main__':main()

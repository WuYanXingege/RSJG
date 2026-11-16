"""One frozen CPU export plus bounded initial-versus-best NLL diagnostic."""
import argparse,json,sys,time,traceback,random,subprocess,signal
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bcep_pilot_runtime import configure,script_module,ROOT
import numpy as np
import torch,yaml
from bcep.contracts import load_locked_split,validate_checkpoint,digest,code_identity,check_item
from bcep.provider import BCEP
from bcep.supervision import LikelihoodAccumulator
from bcep.pilot_runner import atomic_json
from ebjd.data import SceneManifestDataset


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    plan=json.loads(a.plan.read_text());root=Path(plan['asset_root']);cfg=yaml.safe_load(a.config.read_text())
    report_path=root/'reports/READONLY_RESULT.json'
    if report_path.exists():raise RuntimeError('one readonly stage only; refusing to overwrite')
    started=time.monotonic();report=dict(status='RUNNING',optimizer_updates=0,device='cpu',budget_seconds=1800)
    atomic_json(report,report_path)
    def expired(signum,frame):raise TimeoutError('readonly workflow wall budget exhausted')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,1800)
    try:
        configure();split,contract,_=load_locked_split(ROOT/cfg['split_lock'],ROOT/cfg['contract'])
        best=root/'training/best.pt';state=torch.load(best,map_location='cpu',weights_only=False)
        validate_checkpoint(state,ROOT/cfg['contract'],ROOT/cfg['split_lock']);assert state['completed_epoch']>=1 and state['update']>=3790
        report['checkpoint']=dict(path=str(best),sha256=digest(best),epoch=state['selected_epoch'],training_updates=state['update'])
        argv=['--manifest',split['source_manifest'],'--split-lock',str(ROOT/cfg['split_lock']),
              '--contract',str(ROOT/cfg['contract']),'--checkpoint',str(best),'--subset','internal_dev','--purpose','internal_development',
              '--seeds','2035','2036','2037','2038','2039','--require-complete','--output',str(root/'evaluation/banks')]
        report['export_command']=[sys.executable,str(ROOT/'scripts/evaluate_bcep_bank.py'),*argv]
        t=time.monotonic();summary=script_module('scripts/evaluate_bcep_bank.py').main(argv)
        report['cpu_full_export_seconds']=time.monotonic()-t;report['summary']=summary
        atomic_json(report,report_path)
        # Exact constructor and seed sequence of the frozen runner, independent of
        # training objects. Diagnostic only: never presented as a trained checkpoint.
        random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
        initial=BCEP(split['pred_length'],split['dt']);initial.scales.copy_(torch.tensor(split['scales']));initial.scales_fitted.fill_(True);initial.eval()
        dataset=SceneManifestDataset(split['source_manifest']);acc=LikelihoodAccumulator();t=time.monotonic()
        with torch.no_grad():
            for i in split['dev_indices']:
                item=dataset[i];check_item(item,split['members'][i]);valid=torch.ones(len(item['observed']),dtype=torch.bool)
                acc.update(initial.loss_components(item['observed'],item['semantic_maps'],valid,item['future'],cfg['lambda_aux']))
        report['initial_density_diagnostic']=dict(reconstruction='same frozen CPU constructor, seed3101 and train-only scale buffers as run_pilot before model.to(cuda); no pretrained load',source_identity=code_identity(),metrics=acc.metrics('initial_dev'),wall_seconds=time.monotonic()-t,complete_windows=len(split['dev_indices']))
        # CPU total prediction latency includes encoder + M256 draw + CPU selection.
        selected=next(i for i in split['dev_indices'] if len(split['members'][i]['agent_ids'])>=2)
        item=dataset[selected];model=BCEP(split['pred_length'],split['dt']);model.load_state_dict(state['model'],strict=True);model.eval()
        valid=torch.ones(len(item['observed']),dtype=torch.bool);timings=[]
        with torch.no_grad():
            for seed in [2035,2036,2037]:
                t=time.monotonic();atoms,_=model.sample_support(item['observed'],item['semantic_maps'],valid,item['scene_id'],item['agent_ids'],seed,256)
                support_time=time.monotonic()-t;model.select_support(atoms,valid,item['agent_ids']);total=time.monotonic()-t
                timings.append(dict(seed=seed,support_generation_seconds=support_time,coverage_total_seconds=total))
        report['latency']=dict(device='cpu',scope='one locked multi-agent dev scene; not dataset throughput or equal-cost IID20 comparison',index=selected,agents=len(valid),timings=timings)
        report['status']='COMPLETE'
    except Exception as e:
        report.update(status='INCOMPLETE',failure=str(e),traceback=traceback.format_exc());raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        report['wall_seconds']=time.monotonic()-started;atomic_json(report,report_path)
        print(json.dumps({k:report[k] for k in ['status','wall_seconds','optimizer_updates']}),flush=True)

if __name__=='__main__':main()

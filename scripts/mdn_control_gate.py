"""Bounded synthetic GPU recovery and real no-grad interface gate; no real updates."""
import argparse,json,sys,time,signal,traceback
from pathlib import Path
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bcep_pilot_runtime import configure,environment,assert_gpu_idle,script_module,ROOT
import torch,yaml
from bcep.provider import BCEP
from bcep.contracts import load_locked_split,digest,check_item
from bcep.pilot_runner import atomic_json
from bcep_control.mdn import MDNProvider
from bcep_control.contracts import control_code_identity,state_digest
from bcep_control.pilot_runner import run_pilot
from ebjd.data import SceneManifestDataset


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    cfg=yaml.safe_load(a.config.read_text());plan=json.loads((ROOT/cfg['plan']).read_text());root=Path(plan['assets'])
    report_path=root/'reports/MDN_TEST_REPORT.json';assert not report_path.exists()
    tests=ET.parse(root/'gate/CPU_TESTS.xml').getroot().find('testsuite')
    assert int(tests.attrib['tests'])==35 and int(tests.attrib['errors'])==int(tests.attrib['failures'])==0
    cpu_seconds=float(tests.attrib['time']);started=time.monotonic()
    def timeout(signum,frame):raise TimeoutError('600s preflight budget exhausted')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,600-cpu_seconds)
    report=dict(status='RUNNING',real_optimizer_updates=0,cpu_tests=35,original_cpu_tests=25,cpu_seconds=cpu_seconds,config_sha256=digest(a.config))
    atomic_json(report,report_path)
    try:
        assert_gpu_idle();report['precision']=configure();report['environment']=environment();torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(3101);flow=BCEP()
        torch.manual_seed(3101);model=MDNProvider()
        report['initial_encoder_hash']=state_digest(model.encoder.state_dict());assert report['initial_encoder_hash']==state_digest(flow.encoder.state_dict())
        report['initial_mdn_hash']=state_digest(model.mdn.state_dict());report['encoder_initialization_exact']=True
        fixtures=script_module('tests/test_bcep_prepilot.py');ds,split,c,identity=fixtures.runner_fixture();c['device']='cuda:0'
        trace_a=[];trace_b=[]
        run_pilot(ds,split,c,identity,root/'gate/continuous',stop_after_updates=8,on_update=lambda i,u:trace_a.append((i,u)))
        run_pilot(ds,split,c,identity,root/'gate/resumed',stop_after_updates=3,on_update=lambda i,u:trace_b.append((i,u)))
        run_pilot(ds,split,c,identity,root/'gate/resumed',resume=True,stop_after_updates=5,on_update=lambda i,u:trace_b.append((i,u)))
        left=torch.load(root/'gate/continuous/last.pt',map_location='cpu',weights_only=False)
        right=torch.load(root/'gate/resumed/last.pt',map_location='cpu',weights_only=False)
        keys=['model','optimizer','scheduler','rng','order','epoch','completed_epoch','update','train_cursor','dev_cursor','best','stale','selected_epoch','train_accumulator','dev_accumulator']
        diff=script_module('scripts/bcep_gpu_gate.py').tree_difference({k:left[k] for k in keys},{k:right[k] for k in keys})
        assert diff['within_tolerance'] and trace_a==trace_b
        report['synthetic_resume']=diff;report['synthetic_optimizer_updates']=16;report['synthetic_sample_trace']=trace_a
        split,contract,_=load_locked_split(ROOT/cfg['split_lock'],ROOT/cfg['contract']);dataset=SceneManifestDataset(split['source_manifest'])
        model.scales.copy_(torch.tensor(split['scales']));model.scales_fitted.fill_(True);model.to('cuda').eval();rows=[]
        # Identity-only selections fixed before looking at targets.
        counts=[len(x['agent_ids']) for x in split['members']]
        choices=[next(i for i,n in enumerate(counts) if n==1),next(i for i,n in enumerate(counts) if n>=2),max(range(len(counts)),key=counts.__getitem__)]
        with torch.no_grad():
            for index in choices:
                item=dataset[index];check_item(item,split['members'][index]);n=len(item['observed']);v=torch.ones(n,dtype=torch.bool,device='cuda')
                o=item['observed'].cuda();maps=item['semantic_maps'].cuda();atoms,_=model.sample_support(o,maps,v,item['scene_id'],item['agent_ids'],2035)
                chosen,_=model.select_support(atoms,v,item['agent_ids']);loss=model.loss(o,maps,v,item['future'].cuda())
                assert torch.isfinite(atoms).all() and torch.isfinite(chosen).all() and torch.isfinite(loss)
                rows.append(dict(index=index,agents=n,map_shape=list(maps.shape),finite=True,real_optimizer_updates=0))
        report.update(status='PASS',real_interfaces=rows,source_identity=control_code_identity())
    except BaseException as error:
        report.update(status='BLOCKED',failure=repr(error),traceback=traceback.format_exc());raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        report['wall_seconds']=time.monotonic()-started;report['total_preflight_seconds']=report['wall_seconds']+cpu_seconds
        report['peak_allocated_bytes']=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0
        atomic_json(report,report_path);print(json.dumps(dict(status=report['status'],wall_seconds=report['wall_seconds'])),flush=True)


if __name__=='__main__':main()

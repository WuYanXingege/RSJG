#!/usr/bin/env python3
"""Explicitly authorized fresh-goal-only detached launch after acceptance."""
import argparse,json,os,runpy,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.p2_protocol import ROOT,Registry,file_hash,data_binding
from src.p2_grouped import OFFICIAL_PROTOCOL
from src.p2_grouped_training import goal_config_path,source_fingerprint
from src.p2_observation import atomic_json

def verify(D):
    ready=json.loads((D/'READINESS_MATRIX.json').read_text());cfg=goal_config_path(D)
    m=json.loads((D/'QUALIFIED_INPUT_REVIEW_MANIFEST.json').read_text());reg=Registry(m,m['manifest_hash'])
    if ready['status']!='GOAL_TRAINING_READY' or ready['protocol_id']!=OFFICIAL_PROTOCOL:raise RuntimeError('not official-prior ready')
    if ready['config_sha256']!=file_hash(cfg) or ready['manifest_hash']!=m['manifest_hash'] or ready['data_binding']!=data_binding(reg):raise RuntimeError('stale ready identity')
    config=json.loads(cfg.read_text())
    if config['p2_mode']!='goal' or config['phase']!='goal_pretrain' or config['num_epochs']!=150:raise RuntimeError('fresh goal only')
    if config['p2_manifest_hash']!=m['manifest_hash']:raise RuntimeError('config manifest')
    if (ROOT/'output/hotel/runs'/config['run_name']).exists():raise RuntimeError('fresh output already exists')
    for mode in ('cpu','cuda','reload'):
        receipt=json.loads((D/('MODEL_SMOKE_'+mode+'.json')).read_text())
        if receipt['data_binding']!=data_binding(reg) or receipt['source']['files']!=source_fingerprint()['files']:raise RuntimeError('stale smoke source')
    fresh=json.loads((D/'FRESH_INITIAL.json').read_text())
    if fresh['data_binding']!=data_binding(reg) or fresh['formal_config_sha256']!=file_hash(cfg) or file_hash(fresh['path'])!=fresh['sha256'] or fresh['updates']!=0:raise RuntimeError('fresh snapshot identity')
    return cfg,config,ready,fresh

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--worker',action='store_true');a=ap.parse_args();D=a.archive.resolve();O=a.output.resolve();O.mkdir(parents=True,exist_ok=True)
    os.chdir(ROOT);cfg,config,ready,fresh=verify(D)
    if not a.worker:
        if (O/'LAUNCH_REQUEST.json').exists():raise RuntimeError('refuse duplicate launch')
        subprocess.run([sys.executable,'-B','main.py','--p2_config',str(cfg),'--p2_launch_check'],cwd=ROOT,check=True)
        log=O/'fresh_goal.log'
        env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
        cmd=['setsid','nohup',sys.executable,'-u','-B',str(Path(__file__).resolve()),'--worker','--archive',str(D),'--output',str(O)]
        with log.open('x') as f:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,close_fds=True)
        atomic_json(dict(status='LAUNCH_REQUESTED_NOT_YET_TRAINING',pid=child.pid,argv=cmd,log=str(log),
            config_sha256=file_hash(cfg),readiness_sha256=file_hash(D/'READINESS_MATRIX.json'),source=source_fingerprint()),O/'LAUNCH_REQUEST.json')
        print(json.dumps(dict(pid=child.pid,log=str(log))));return
    status=dict(pid=os.getpid(),session_id=os.getsid(0),run_name=config['run_name'],source=source_fingerprint(),
        protocol_id=OFFICIAL_PROTOCOL,data_binding=ready['data_binding'],started_unix=time.time(),epochs_max=150,wall_seconds=86400,
        output_dir=str(ROOT/'output/hotel/runs'/config['run_name']),no_auto_joint=True,no_auto_A1=True)
    try:
        query=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,capture_output=True)
        if query.returncode or query.stdout.strip():raise RuntimeError('RESOURCE_BLOCKED_DEVICE_BUSY_OR_QUERY_FAILED')
        import torch
        torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.cuda.set_device(0)
        free,total=torch.cuda.mem_get_info();budget=min(10*1024**3,free-1024**3)
        if budget<2*1024**3:raise RuntimeError('RESOURCE_BLOCKED_MEMORY')
        torch.cuda.set_per_process_memory_fraction(budget/total,0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        os.environ['RSJG_EXPECTED_FRESH_STATE_SHA256']=fresh['state_sha256']
        atomic_json(dict(status,status='STARTING',gpu_memory_budget=budget),O/'LAUNCH_STATUS.json')
        sys.argv=['main.py','--p2_config',str(cfg)];runpy.run_path(str(ROOT/'main.py'),run_name='__main__')
        atomic_json(dict(status,status='COMPLETED',ended_unix=time.time()),O/'LAUNCH_STATUS.json')
    except BaseException:
        atomic_json(dict(status,status='FAILED',error=traceback.format_exc(),ended_unix=time.time()),O/'LAUNCH_STATUS.json');raise
if __name__=='__main__':main()

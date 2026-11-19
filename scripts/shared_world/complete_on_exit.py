"""Wait for this run's process-exit event, then finalize and publish once.
No periodic process polling; no other fold/seed. Explicitly paused runs stay paused.
"""
import argparse,fcntl,hashlib,json,os,select,subprocess,sys,time
from pathlib import Path
SOURCE=Path(__file__).resolve().parents[2]
EXPERIMENT='zara2_sharedworld_stage2_onerun_20261119'
BRANCH='experiment/zara2-sharedworld-stage2-onerun-20261119'
REMOTE='git@github.com:WuYanXingege/RSJG.git'

def read(p):return json.loads(p.read_text())
def write(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n');t.replace(p)
def git(*args):return subprocess.check_output(['git',*args],cwd=SOURCE,text=True).strip()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def wait_for_exit(pid,workspace):
    p=Path('/proc')/str(pid)
    if not p.exists():return
    tokens=(p/'cmdline').read_bytes().split(b'\0')
    if b'scripts/shared_world/one_run.py' not in tokens or b'train' not in tokens or str(workspace).encode() not in tokens:
        raise RuntimeError('PID no longer identifies the bound training command')
    if Path(os.readlink(p/'cwd')).resolve()!=SOURCE:raise RuntimeError('training cwd mismatch')
    fd=os.pidfd_open(pid)
    try:
        poll=select.poll();poll.register(fd,select.POLLIN);poll.poll()
    finally:os.close(fd)

def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);a=p.parse_args();w=a.workspace.resolve()
    if w!=SOURCE.parent:raise ValueError('workspace mismatch')
    run=w/'RSJG_EBJD_assets/experiments'/EXPERIMENT;statuspath=run/'COMPLETION_STATUS.json'
    with (run/'completion.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (run/'DELIVERY.json').exists():return
        frozen=read(run/'FROZEN_CONFIG.json');assert frozen['RUN_ONE_ZARA2'] is True
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        pid=read(run/'LAUNCH.json')['pid'];retries=[]
        try:
            while True:
                write(statuspath,dict(status='WAITING_FOR_EXIT_EVENT',worker_pid=os.getpid(),training_pid=pid,periodic_polling=False,restarts=retries,unix=time.time()))
                wait_for_exit(pid,w)
                current=read(run/'STATUS.json')
                if current['status']=='COMPLETE':break
                if current['status']=='SESSION_BUDGET_PAUSED':
                    write(statuspath,dict(status='PAUSED_NO_AUTOMATIC_RESUME',run_status=current,unix=time.time()));return
                # User authorized technical recovery of this identity only; bounded retries.
                if current['status'] not in ['RUNNING','TECHNICAL_INTERRUPTION'] or len(retries)>=2 or not (run/'joint/last.pt').exists():
                    raise RuntimeError('training incomplete; manual review required: '+current['status'])
                saved=read(run/'joint/PROGRESS.json')
                retries.append(dict(at_unix=time.time(),prior_status=current,latest_logged_update=saved.get('update'),note='recovery from last.pt; uncheckpointed physical steps may be replayed'))
                write(run/'TECHNICAL_RECOVERIES.json',retries)
                with (run/'LOCAL_TRAIN.log').open('ab',buffering=0) as log:
                    child=subprocess.Popen([sys.executable,'scripts/shared_world/one_run.py','train','--workspace',str(w)],cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                pid=child.pid
                write(run/'LAUNCH.json',dict(pid=pid,recovery=True,started_unix=time.time()))
            for rel,h in frozen['implementation_files'].items():assert sha(SOURCE/rel)==h,'runtime source changed'
            assert git('branch','--show-current')==BRANCH
            write(statuspath,dict(status='FINALIZING',unix=time.time(),restarts=retries))
            subprocess.run([sys.executable,'scripts/shared_world/one_run_report.py','--workspace',str(w)],cwd=SOURCE,env=env,check=True)
            out=SOURCE/'docs/research/bcep_v2/sharedworld_stage2_onerun_20261119'
            if retries:
                # Preserve the distinction between committed updates and replayed physical work.
                write(out/'TECHNICAL_RECOVERIES.json',retries)
                report=out/'REPORT_ZH.md';report.write_text(report.read_text()+'\n\n技术恢复记录见 TECHNICAL_RECOVERIES.json。报告更新数为本次最终状态中的有效更新；未落盘更新可能重放，物理 optimizer.step 总数不能仅凭周期日志精确重建。\n')
                write(out/'ARTIFACT_SHA256.json',{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='ARTIFACT_SHA256.json'})
            files=[p for p in out.iterdir() if p.is_file()]
            for p in files:
                assert p.suffix in ['.md','.csv','.json'] or p.name=='.gitattributes'
                t=p.read_text();assert '/media/ee615/' not in t and '/home/ee615/' not in t
            for name,h in read(out/'ARTIFACT_SHA256.json').items():assert sha(out/name)==h
            subprocess.run(['git','diff','--cached','--quiet'],cwd=SOURCE,check=True)
            subprocess.run(['git','add',*[str(p.relative_to(SOURCE)) for p in files]],cwd=SOURCE,check=True)
            subprocess.run(['git','diff','--cached','--check'],cwd=SOURCE,check=True)
            if subprocess.run(['git','diff','--cached','--quiet'],cwd=SOURCE).returncode:
                subprocess.run(['git','commit','-m','Publish completed ZARA2 shared-world Stage 2 controlled trial'],cwd=SOURCE,check=True)
            commit=git('rev-parse','HEAD')
            for attempt in range(3):
                try:
                    subprocess.run(['git','push',REMOTE,BRANCH],cwd=SOURCE,check=True)
                    remote=git('ls-remote',REMOTE,'refs/heads/'+BRANCH).split()[0];assert remote==commit
                    break
                except subprocess.CalledProcessError:
                    if attempt==2:raise
                    time.sleep(5*(attempt+1))
            base='https://github.com/WuYanXingege/RSJG/blob/'+commit+'/'+str(out.relative_to(SOURCE))+'/'
            write(run/'DELIVERY.json',dict(remote_sha=commit,report=base+'REPORT_ZH.md',csv=base+'EPOCH_GROUPS.csv',paired_csv=base+'RANK1_VS_OLD5_WINDOWS.csv',decision=read(out/'DECISION.json'),unix=time.time()))
            write(statuspath,dict(status='PUBLISHED',remote_sha=commit,unix=time.time()))
        except BaseException as e:
            write(statuspath,dict(status='NEEDS_REVIEW',exception_type=type(e).__name__,reason=str(e),unix=time.time()));raise
if __name__=='__main__':main()

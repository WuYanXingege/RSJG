"""One authorized command; event-driven completion, same-last technical recovery.

Only resource telemetry samples periodically. There is no new model entry,
training queue, hyperparameter override, or periodic foreground process monitor.
"""
import csv,fcntl,json,os,re,shlex,subprocess,sys,threading,time,traceback
from pathlib import Path
SOURCE=Path(__file__).resolve().parents[2]
RUN=SOURCE.parent/'RSJG_EBJD_assets/experiments/zara2_bcep_v3_stage2_20261120'
BRANCH='experiment/zara2-bcep-v3-stage2-20261120'
REMOTE='git@github.com:WuYanXingege/RSJG.git'
BASE='2e96b6f35554ceceb5a04fc08b6259dac75e486b'
FROZEN='294054c2cf7605caafe016507bb601e2d85518b70555369ca3690b74138e9e43'
DOC=SOURCE/'docs/research/bcep_v3/zara2_trial_20261120'


def read(p):return json.loads(p.read_text())
def write(p,value):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');t.replace(p)
def append(p,row):
    with p.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
def git(*args):return subprocess.check_output(['git',*args],cwd=SOURCE,text=True).strip()


def bound_command():
    command=(SOURCE/'docs/research/bcep_v3/stage2_ready_20261120/CONTROLLED_COMMAND.txt').read_text().strip()
    parts=shlex.split(command);env=dict(os.environ)
    expected=['RUN_ONE_ZARA2_V3=true','OMP_NUM_THREADS=1','CUBLAS_WORKSPACE_CONFIG=:4096:8']
    assert parts[:3]==expected
    argv=['../.conda/rsjg/bin/python','-m','bcep_v3.formal','train','--frozen-sha',FROZEN]
    assert parts[3:]==argv
    for token in expected:
        key,value=token.split('=',1);env[key]=value
    return command,argv,env


def resources(pid,stop):
    while not stop.is_set():
        row=dict(unix=time.time(),pid=pid,interval_seconds=5)
        try:
            result=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
            entries=list(csv.reader(result.splitlines()))
            values=[float(r[1].strip()) for r in entries if len(r)>1 and r[0].strip()==str(pid)]
            row['gpu_process_mib']=sum(values) if values else None
            proc=Path('/proc')/str(pid)/'status'
            if proc.exists():
                fields=dict(line.split(':',1) for line in proc.read_text().splitlines() if ':' in line)
                row['rss_kib']=int(fields.get('VmRSS','0 kB').split()[0])
            row['status']='OK'
        except Exception as e:row.update(status='UNAVAILABLE',reason=type(e).__name__+': '+str(e))
        append(RUN/'RESOURCE_SAMPLES.jsonl',row)
        stop.wait(5)


def publish():
    subprocess.run([sys.executable,'scripts/v3_trial/report.py'],cwd=SOURCE,check=True)
    assert git('branch','--show-current')==BRANCH
    assert git('diff','--cached','--name-only')==''
    files=[p for p in DOC.iterdir() if p.is_file()]
    for p in files:
        assert p.suffix in ['.md','.json','.csv','.txt'] or p.name=='.gitattributes'
        raw=p.read_bytes();assert b'/media/ee615/' not in raw and b'/home/ee615/' not in raw
    subprocess.run(['git','add',*[str(p.relative_to(SOURCE)) for p in files]],cwd=SOURCE,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=SOURCE,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=SOURCE).returncode:
        subprocess.run(['git','commit','-m','Publish completed single ZARA2 BCEP EBJD v3 controlled trial'],cwd=SOURCE,check=True)
    commit=git('rev-parse','HEAD')
    for attempt in range(3):
        try:
            subprocess.run(['git','push',REMOTE,BRANCH],cwd=SOURCE,check=True)
            remote=git('ls-remote',REMOTE,'refs/heads/'+BRANCH).split()[0];assert remote==commit
            break
        except subprocess.CalledProcessError:
            if attempt==2:raise
            time.sleep(5*(attempt+1))
    base='https://github.com/WuYanXingege/RSJG/blob/'+commit+'/'+str(DOC.relative_to(SOURCE))+'/'
    write(RUN/'DELIVERY.json',dict(remote_sha=commit,report=base+'REPORT_ZH.md',decision=read(DOC/'DECISION.json'),unix=time.time()))
    return commit


def main():
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (RUN/'DELIVERY.json').exists():return
        if (RUN/'LAUNCH.json').exists():raise RuntimeError('this supervisor must not be launched twice')
        assert git('branch','--show-current')==BRANCH
        assert read(RUN/'STATUS.json')['status']=='PREPARED_NO_TRAINING_AUTHORIZATION'
        assert not (RUN/'joint/last.pt').exists()
        command,argv,env=bound_command();attempts=[]
        write(RUN/'EXECUTION_AUTHORIZATION.json',dict(user_authorized=True,scope='one ZARA2 Stage2 only',runtime_code_commit=BASE,
              frozen_sha256=FROZEN,exact_command=command,max_updates=16800,other_seeds=False,other_folds=False,test=False,old_queues=False,unix=time.time()))
        try:
            while True:
                attempt=len(attempts)+1
                started=time.time()
                child=subprocess.Popen(argv,cwd=SOURCE,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
                write(RUN/'LAUNCH.json',dict(pid=child.pid,supervisor_pid=os.getpid(),attempt=attempt,started_unix=started,runtime_code_commit=BASE,command=command))
                write(RUN/'SUPERVISOR_STATUS.json',dict(status='TRAINING',pid=child.pid,supervisor_pid=os.getpid(),attempt=attempt,unix=time.time()))
                stop=threading.Event();sampler=threading.Thread(target=resources,args=(child.pid,stop),daemon=True);sampler.start()
                with (RUN/'LOCAL_TRAIN.log').open('a',buffering=1) as log:
                    for line in child.stdout:
                        log.write(line)
                        match=re.search(r'zara2 joint epoch=(\d+) update=(\d+)',line)
                        if match:
                            append(RUN/'EPOCH_EVENTS.jsonl',dict(epoch=int(match[1]),update=int(match[2]),unix=time.time(),pid=child.pid,attempt=attempt))
                code=child.wait();stop.set();sampler.join(timeout=15)
                state=read(RUN/'STATUS.json')
                attempts.append(dict(attempt=attempt,pid=child.pid,exit_code=code,started_unix=started,finished_unix=time.time(),status=state['status']))
                write(RUN/'PROCESS_ATTEMPTS.json',attempts)
                if state['status']=='COMPLETE' and code==0:break
                if state['status']=='PAUSED_RESUMABLE':
                    write(RUN/'SUPERVISOR_STATUS.json',dict(status='PAUSED_NO_AUTOMATIC_RESUME',reason='native pause is respected',unix=time.time()));return
                if state['status'] not in ['TECHNICAL_INTERRUPTION','RUNNING'] or not (RUN/'joint/last.pt').exists() or attempt>=3:
                    raise RuntimeError('technical interruption requires review; no fresh training or new queue allowed')
                write(RUN/'SUPERVISOR_STATUS.json',dict(status='RECOVER_SAME_LAST',attempt=attempt,unix=time.time()))
                time.sleep(10)
            write(RUN/'SUPERVISOR_STATUS.json',dict(status='FINALIZING',unix=time.time()))
            commit=publish()
            write(RUN/'SUPERVISOR_STATUS.json',dict(status='PUBLISHED',remote_sha=commit,unix=time.time()))
        except BaseException as e:
            write(RUN/'SUPERVISOR_STATUS.json',dict(status='NEEDS_REVIEW',exception_type=type(e).__name__,reason=str(e),unix=time.time()))
            traceback.print_exc();raise
if __name__=='__main__':main()

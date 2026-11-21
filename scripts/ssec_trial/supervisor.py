"""One exact authorized command, same-last recovery, finite final publication."""
import fcntl, os, re, subprocess, sys, threading, time, traceback
from common import *


def telemetry(pid, stop):
    while not stop.is_set():
        r=dict(unix=time.time(),pid=pid)
        try:
            raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
            found=[float(x[1]) for x in csv.reader(raw.splitlines()) if len(x)>1 and x[0].strip()==str(pid)]
            r['gpu_process_mib']=sum(found) if found else None
            r['disk_free_bytes']=__import__('shutil').disk_usage(RUN).free
            if (RUN/'joint/FIRST_UPDATE.json').exists() and not (RUN/'FIRST_UPDATE_VERIFICATION.json').exists():
                subprocess.run([sys.executable,str(SOURCE/'scripts/ssec_trial/verify.py'),'first-update'],cwd=SOURCE,check=True)
        except Exception as e:r['telemetry_error']=str(e)
        append(RUN/'RESOURCE_SAMPLES.jsonl',r)
        stop.wait(30)


def publish():
    subprocess.run([sys.executable,str(SOURCE/'scripts/ssec_trial/report.py'),'final'],cwd=SOURCE,check=True)
    assert git('branch','--show-current')==BRANCH
    assert not git('diff','--cached','--name-only')
    files=[p for p in DOC.rglob('*') if p.is_file()]
    for p in files:
        assert p.suffix in ['.md','.json','.csv','.txt']
        raw=p.read_bytes();assert b'/media/ee615/' not in raw and b'/home/ee615/' not in raw
    subprocess.run(['git','add',str(DOC.relative_to(SOURCE))],cwd=SOURCE,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=SOURCE,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=SOURCE).returncode:
        subprocess.run(['git','commit','-m','Publish completed single SSEC Energy-only ZARA2 trial'],cwd=SOURCE,check=True)
    sha=git('rev-parse','HEAD')
    for attempt in range(3):
        try:
            subprocess.run(['git','push','origin',BRANCH],cwd=SOURCE,check=True)
            remote=git('ls-remote','origin','refs/heads/'+BRANCH).split()[0];assert remote==sha
            break
        except subprocess.CalledProcessError:
            if attempt==2:raise
            time.sleep(5*(attempt+1))
    base='https://github.com/WuYanXingege/RSJG/blob/'+sha+'/'+PUBLIC_PATH+'/'
    write(RUN/'DELIVERY.json',dict(remote_sha=sha,remote_verified=True,report=base+'README.md',
          decision=read(DOC/'DECISION.json'),runtime_code_commit=BASE,unix=time.time()))
    return sha


def main():
    with (RUN/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (RUN/'LAUNCH.json').exists():raise RuntimeError('unique run already launched; never start another supervisor')
        source_check();cmd=command()
        assert read(RUN/'STATUS.json')['status']=='PREPARED_NO_TRAINING_AUTHORIZATION'
        assert not (RUN/'joint/last.pt').exists()
        assert read(RUN/'PREFLIGHT.json')['passed']
        attempts=[]
        write(RUN/'EXECUTION_AUTHORIZATION.json',dict(user_authorized=True,scope='one ZARA2 SSEC Energy-only Stage2',
            runtime_code_commit=BASE,recovery_receipt_commit=RECOVERY_BASE,frozen_file_sha256=FROZEN,
            exact_command=cmd,max_updates=16800,other_seeds=False,other_folds=False,test=False,old_queues=False,unix=time.time()))
        try:
            while True:
                source_check()
                attempt=len(attempts)+1;started=time.time()
                # Execute the exact command line read from CONTROLLED_COMMAND.
                child=subprocess.Popen(['/bin/bash','-c',cmd],cwd=TRAIN_SOURCE,stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
                write(RUN/'LAUNCH.json',dict(pid=child.pid,supervisor_pid=os.getpid(),attempt=attempt,started_unix=started,
                    runtime_code_commit=BASE,command=cmd))
                write(RUN/'SUPERVISOR_STATUS.json',dict(status='TRAINING',pid=child.pid,attempt=attempt,unix=time.time()))
                stop=threading.Event();thread=threading.Thread(target=telemetry,args=(child.pid,stop),daemon=True);thread.start()
                with (RUN/'LOCAL_TRAIN.log').open('a',buffering=1) as log:
                    for line in child.stdout:
                        log.write(line)
                        match=re.search(r'zara2 joint epoch=(\d+) update=(\d+)',line)
                        if match:
                            epoch=int(match[1]);append(RUN/'EPOCH_EVENTS.jsonl',dict(epoch=epoch,update=int(match[2]),unix=time.time(),pid=child.pid,attempt=attempt))
                            if epoch==1:
                                check=subprocess.run([sys.executable,str(SOURCE/'scripts/ssec_trial/verify.py'),'first-epoch'],cwd=SOURCE)
                                if check.returncode:write(RUN/'SUPERVISION_ALERT.json',dict(reason='first epoch verification needs review',unix=time.time()))
                code=child.wait();stop.set();thread.join(timeout=15)
                state=read(RUN/'STATUS.json')
                attempts.append(dict(attempt=attempt,pid=child.pid,exit_code=code,started_unix=started,finished_unix=time.time(),status=state['status']))
                write(RUN/'PROCESS_ATTEMPTS.json',attempts)
                if code==0 and state['status']=='COMPLETE':break
                if state['status']=='PAUSED_RESUMABLE':
                    write(RUN/'SUPERVISOR_STATUS.json',dict(status='PAUSED_NO_AUTOMATIC_RESUME',unix=time.time()));return
                if state['status'] not in ['TECHNICAL_INTERRUPTION','RUNNING'] or not (RUN/'joint/last.pt').exists() or attempt>=3:
                    raise RuntimeError('technical interruption requires review; no fresh random run allowed')
                write(RUN/'SUPERVISOR_STATUS.json',dict(status='RECOVER_SAME_LAST',attempt=attempt,unix=time.time()))
                time.sleep(10)
            write(RUN/'SUPERVISOR_STATUS.json',dict(status='FINALIZING',unix=time.time()))
            sha=publish();write(RUN/'SUPERVISOR_STATUS.json',dict(status='PUBLISHED',remote_sha=sha,unix=time.time()))
        except BaseException as e:
            write(RUN/'SUPERVISOR_STATUS.json',dict(status='NEEDS_REVIEW',exception_type=type(e).__name__,reason=str(e),unix=time.time()))
            traceback.print_exc();raise


if __name__=='__main__':main()

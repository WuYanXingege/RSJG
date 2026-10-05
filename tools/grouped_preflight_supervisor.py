#!/usr/bin/env python3
"""Append-only owned-process resource ledger; no Torch/CUDA imports."""
import argparse,datetime,hashlib,json,os,signal,subprocess,time,uuid
from pathlib import Path

def append(p,row):
    with p.open('a') as f:
        f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def proc(pid):
    try:
        text=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
        status={s.split(':')[0]:s.split(':',1)[1].strip() for s in Path('/proc',str(pid),'status').read_text().splitlines() if ':' in s}
        return dict(pid=pid,ppid=int(text[1]),start_ticks=int(text[19]),
            rss_bytes=int(text[21])*os.sysconf('SC_PAGE_SIZE'),
            hwm_bytes=int(status['VmHWM'].split()[0])*1024 if 'VmHWM' in status else 0,
            os_threads=int(status.get('Threads','0')),pid_namespace=os.readlink('/proc/'+str(pid)+'/ns/pid'))
    except (OSError,ValueError,KeyError):return None
def tree(pid):
    seen=set();pending=[pid]
    while pending:
        p=pending.pop()
        if p in seen:continue
        seen.add(p)
        try:pending.extend(map(int,Path('/proc',str(p),'task',str(p),'children').read_text().split()))
        except OSError:pass
    return [r for p in seen if (r:=proc(p)) is not None]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--stage',required=True)
    ap.add_argument('--threads',type=int,default=4);ap.add_argument('command',nargs=argparse.REMAINDER)
    ap.add_argument('--gpu-seconds',type=float,default=None,help='Owned child hard wall, charged conservatively to cumulative GPU budget')
    a=ap.parse_args();assert 1<=a.threads<=8
    cmd=a.command[1:] if a.command and a.command[0]=='--' else a.command
    if not cmd:ap.error('command required')
    a.archive.mkdir(parents=True,exist_ok=True);a.output.mkdir(parents=True,exist_ok=True)
    ledger=a.archive/'PROCESS_LEDGER.jsonl';job=uuid.uuid4().hex
    origin=datetime.datetime(2026,10,5,9,5,tzinfo=datetime.timezone.utc).timestamp()
    if time.time()-origin>=21600:raise RuntimeError('Cumulative task CPU/data wall exhausted')
    env=dict(os.environ)
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]=str(a.threads)
    logfile=a.output/(a.stage+'_'+job+'.log');start=time.time()
    gpu_ledger=a.archive/'GPU_BUDGET_LEDGER.jsonl'
    gpu_prior=sum(json.loads(x)['charged_seconds'] for x in gpu_ledger.read_text().splitlines()) if gpu_ledger.exists() else 0.
    if a.gpu_seconds is not None:
        if not 0<a.gpu_seconds<=1800 or gpu_prior>=1785:raise RuntimeError('cumulative GPU budget exhausted')
        a.gpu_seconds=min(a.gpu_seconds,1800-gpu_prior-15) # teardown/sampling safety margin
    append(ledger,dict(event='START',job=job,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        stage=a.stage,command=cmd,supervisor=proc(os.getpid()),threads_configured=a.threads,
        command_sha256=hashlib.sha256(json.dumps(cmd,separators=(',',':')).encode()).hexdigest(),
        entrypoint_sources={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in cmd if s.endswith('.py') and Path(s).is_file()},
        implementation_sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for pattern in
            ('src/p2*.py','src/models/model.py','src/models/goal_pretrain.py','src/trainer.py','src/losses.py','src/parser.py') for p in Path('.').glob(pattern)},
        stdout_path=str(logfile),sampling_seconds=1,scope='Owned supervisor and recursive descendants; sequential jobs, no unrelated/SDD inspection'))
    peak=0;high={};stop=None;iterations=0
    with logfile.open('w') as log:
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        while True:
            rows=tree(child.pid)+[proc(os.getpid())];rows=[r for r in rows if r]
            rss=sum(r['rss_bytes'] for r in rows);peak=max(peak,rss)
            for r in rows:high[str((r['pid_namespace'],r['pid'],r['start_ticks']))]=max(high.get(str((r['pid_namespace'],r['pid'],r['start_ticks'])),0),r['hwm_bytes'])
            append(ledger,dict(event='SAMPLE',job=job,time=time.time(),processes=rows,aggregate_rss=rss))
            code=child.poll()
            if code is not None:break
            if rss>12*1024**3:stop='AGGREGATE_RSS_LIMIT'
            if time.time()-origin>=21600:stop='TASK_WALL_LIMIT'
            if a.gpu_seconds is not None and time.time()-start>=a.gpu_seconds:stop='GPU_WALL_LIMIT'
            if iterations%60==0:
                size=sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())
                if size>64*1024**3:stop='NEW_STORAGE_LIMIT'
            if stop:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                code=child.returncode;break
            iterations+=1;time.sleep(1)
    result=dict(event='END',job=job,stage=a.stage,exit_code=code,stop=stop,
        seconds=time.time()-start,aggregate_sampled_peak_rss=peak,process_high_water=high,
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),log=str(logfile),
        coverage='1s samples may miss transient aggregate peaks; per-process high-water sampled while alive; import/native CUDA instrumentation separate')
    append(ledger,result);print(json.dumps(result),flush=True)
    if a.gpu_seconds is not None:append(gpu_ledger,dict(job=job,stage=a.stage,charged_seconds=result['seconds'],scope='conservative entire child wall including CPU setup; actual GPU-only time no greater'))
    return code if code else (2 if stop else 0)
if __name__=='__main__':raise SystemExit(main())

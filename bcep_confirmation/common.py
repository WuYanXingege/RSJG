import hashlib,json,os,signal,time
from pathlib import Path
from contextlib import contextmanager
from bcep.contracts import canonical_hash,digest

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n');t.replace(p)
def state_digest(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def source_identity():
    paths=[p for folder in ['bcep','bcep_control','selector_control','bcep_confirmation'] for p in sorted((ROOT/folder).rglob('*.py'))]+[ROOT/'scripts/run_bcep_confirmation.py',ROOT/'scripts/prepare_bcep_confirmation.py',ROOT/'scripts/bcep_pilot_runtime.py']
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}
@contextmanager
def budget(root,name,limit):
    path=Path(root)/'receipts'/f'{name}_BUDGET.json'
    ledger=read(path) if path.exists() else dict(limit_seconds=limit,seconds=0.,attempts=[])
    if ledger['attempts'] and ledger['attempts'][-1]['status']=='RUNNING':raise RuntimeError('unclean budget requires explicit reconciliation')
    remaining=limit-ledger['seconds']
    if remaining<=0:raise TimeoutError('cumulative budget exhausted')
    start=time.monotonic();row=dict(status='RUNNING',pid=os.getpid(),started_unix=time.time());ledger['attempts'].append(row);write(path,ledger)
    def expired(*_):raise TimeoutError(name+' budget exhausted')
    prior=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,remaining)
    try:yield;row['status']='COMPLETE'
    except BaseException as e:row.update(status='FAILED',error=repr(e));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,prior)
        row['seconds']=time.monotonic()-start;ledger['seconds']+=row['seconds'];write(path,ledger)

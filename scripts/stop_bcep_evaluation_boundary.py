"""External exact-process seed-boundary stop; never edits evaluated source/assets."""
import argparse,ctypes,hashlib,json,os,select,signal,struct,time,zipfile
from pathlib import Path

def write(path,value):
 t=path.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));os.replace(t,path)
def identity(pid):
 p=Path('/proc')/str(pid)
 return dict(pid=pid,cmdline=(p/'cmdline').read_bytes().decode().split('\0')[:-1],cwd=os.readlink(p/'cwd'),start_ticks=(p/'stat').read_text().rsplit(')',1)[1].split()[19])
def main(args):
 out=Path(args.receipts);old=Path(args.old_assets);process=identity(args.pid)
 assert process['cwd']==str(Path(args.worktree).resolve())
 assert 'scripts/run_bcep_joint.py' in process['cmdline'] and '--execute-authorized-two-runs' in process['cmdline']
 planarg=process['cmdline'][process['cmdline'].index('--plan')+1]
 assert (Path(process['cwd'])/planarg).resolve()==old/'RUN_PLAN.json'
 progress=json.loads((old/'EVALUATION_PROGRESS.json').read_text());target=old/'evaluation'/f"{progress['run']}_{progress['seed']}.npz"
 state=dict(status='WAITING_CURRENT_SEED_ONLY',process=process,initial_progress=progress,target=str(target),started_unix=time.time(),hot_source_changes=False,signals=[])
 write(out/'STOP_STATE.json',state)
 libc=ctypes.CDLL(None,use_errno=True);fd=libc.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC);assert fd>=0
 assert libc.inotify_add_watch(fd,os.fsencode(target.parent),0x80|0x8)>=0 # MOVED_TO, CLOSE_WRITE
 try:
  while not target.exists():
   try:assert identity(args.pid)==process,'PID identity changed'
   except FileNotFoundError:
    state.update(status='EXITED_BEFORE_TARGET_PERSISTED',ended_unix=time.time());write(out/'STOP_STATE.json',state);return
   if select.select([fd],[],[],1)[0]:os.read(fd,65536)
  assert identity(args.pid)==process
  os.kill(args.pid,signal.SIGSTOP);state['signals'].append('SIGSTOP after final seed filename exists')
  state.update(status='STOPPED_AT_PERSISTED_SEED',boundary_observed_unix=time.time());write(out/'STOP_STATE.json',state)
  # Seed files are published only by atomic rename after save completes.
  with zipfile.ZipFile(target) as z:assert z.testzip() is None
  state['completed_target']=dict(path=str(target),bytes=target.stat().st_size,sha256=hashlib.file_digest(target.open('rb'),'sha256').hexdigest()) if hasattr(hashlib,'file_digest') else dict(path=str(target),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest())
  assert identity(args.pid)==process
  os.kill(args.pid,signal.SIGTERM);state['signals'].append('SIGTERM exact verified coordinator')
  os.kill(args.pid,signal.SIGCONT);state['signals'].append('SIGCONT to deliver pending termination')
  for _ in range(100):
   try:
    stat=(Path('/proc')/str(args.pid)/'stat').read_text().rsplit(')',1)[1].split()[0]
    if stat=='Z':break
   except FileNotFoundError:break
   time.sleep(.1)
  else:raise RuntimeError('process still alive after SIGTERM; no escalation issued')
  state.update(status='TERMINATED_AFTER_COMPLETE_SEED',ended_unix=time.time(),remaining_seeds_not_requested=True)
  write(out/'STOP_STATE.json',state)
 finally:os.close(fd)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);p.add_argument('--old-assets',required=True);p.add_argument('--worktree',required=True);p.add_argument('--receipts',required=True);main(p.parse_args())

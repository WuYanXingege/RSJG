"""CPU-only immutable inventory after the exact old evaluation process exits."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));os.replace(t,p)
def main(args):
 old=Path(args.old_assets);out=Path(args.receipts);stop=json.loads((out/'STOP_STATE.json').read_text());assert stop['status']=='TERMINATED_AFTER_COMPLETE_SEED'
 proc=Path('/proc')/str(stop['process']['pid'])
 assert not proc.exists() or proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[0]=='Z'
 plan=json.loads((old/'RUN_PLAN.json').read_text());frozen=json.loads((old/'reports/CHECKPOINTS_FROZEN.json').read_text());files=set(p for p in old.rglob('*') if p.is_file())
 for r in plan['runs']:
  files.add(Path(r['provider']['path']))
  for b in r['banks'].values():files.add(Path(b['path']))
 for key in ['split_path','config_path']:files.add(Path(plan[key]))
 index={str(p):dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(files)}
 for b in json.loads((out/'TRAINING_ASSETS_BEFORE.json').read_text()):assert index[b['path']]['sha256']==b['sha256'],'training state was changed'
 results=[]
 for r in plan['runs']:
  f=next(x for x in frozen['runs'] if x['name']==r['name'])
  assert index[r['provider']['path']]['sha256']==r['provider']['sha256']
  for b in r['banks'].values():assert index[b['path']]['sha256']==b['sha256']
  for seed in range(2035,2040):
   p=old/'evaluation'/f"{r['name']}_{seed}.npz";row=dict(run=r['name'],seed=seed,status='NOT_PERSISTED_CANCELLED',complete_members=0)
   if p.exists():
    with np.load(p,allow_pickle=False) as z:
     meta=json.loads(str(z['metadata_json']));v=z['valid'];g=z['G'];h=z['H'];assert meta['seed']==seed and meta['run']==r['name'] and meta['checkpoint_sha256']==f['best']['sha256'];assert len(meta['members'])==len(v)==998;assert g.shape==h.shape and g.shape[1]==20 and g.shape[-2:]==(12,2);assert np.isfinite(g).all() and np.isfinite(h).all()
     counts=[len(m['agent_ids']) for m in meta['members']];assert np.array_equal(v.sum(1),counts)
     row.update(status='COMPLETE_PERSISTED_TRAJECTORIES',complete_members=len(v),agent_occurrences=int(v.sum()),K=20,T=12,unit=meta['unit'],checkpoint_sha256=meta['checkpoint_sha256'],**index[str(p)])
   results.append(row)
 write(out/'PRESERVED_ASSET_MANIFEST.json',dict(files=list(index.values()),training_state_unchanged=True,deleted_files=0))
 write(out/'EVALUATION_SCOPE_CHANGE.json',dict(status='PARTIAL_BY_USER_PRIORITY_CHANGE',reason='formal JMM fivefold task has priority',actual_exit=stop,results=results,complete_bank_count=sum(r['status']=='COMPLETE_PERSISTED_TRAJECTORIES' for r in results),cpu_scoring='DEFERRED to avoid competing with formal GPU acceptance CPU/IO',no_balanced_two_run_five_seed_claim=True,original_training_status_not_rewritten=True,unpersisted_in_memory_work_not_claimed=True,optimizer_updates_added=0,created_unix=time.time()))
 print(json.dumps(dict(preserved_files=len(files),completed_banks=sum(r['status']=='COMPLETE_PERSISTED_TRAJECTORIES' for r in results),status='PARTIAL_BY_USER_PRIORITY_CHANGE')))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--old-assets',required=True);p.add_argument('--receipts',required=True);main(p.parse_args())

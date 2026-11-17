"""Read official loader's actual membership method on immutable official rows."""
import sys,json,types,importlib.util,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from bcep_joint.common import read,write
from bcep.contracts import digest

def main(paths):
 a=Path(paths['assets']);repo=a/'sources/Joint_AgentFormer';lock=read(a/'JMM_SPLIT_LOCK.json');start=time.monotonic()
 # Only get_valid_id/PreData/FutureData are used. Map construction is intentionally
 # not called: current EBJD map/crop semantics stay unchanged.
 package=types.ModuleType('_official_membership');package.__path__=[str(repo/'data')];sys.modules[package.__name__]=package
 dummy=types.ModuleType('_official_membership.map');dummy.GeometricMap=object;sys.modules[dummy.__name__]=dummy
 spec=importlib.util.spec_from_file_location('_official_membership.preprocessor',repo/'data/preprocessor.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 checked=[]
 for name,row in lock['files'].items():
  table=np.loadtxt(row['path'],usecols=[0,1,13,15]);obj=object.__new__(module.preprocess);obj.gt=table;obj.past_frames=8;obj.future_frames=12;obj.min_past_frames=8;obj.min_future_frames=12;obj.frame_skip=1;obj.init_frame=int(table[:,0].min())
  byframe={int(f):table[table[:,0]==f] for f in np.unique(table[:,0])};expected=[]
  for f in range(obj.init_frame+7,int(table[:,0].max())-11):
   pre=[byframe.get(f-i,np.empty((0,4))) for i in range(8)];future=[byframe.get(f+i,np.empty((0,4))) for i in range(1,13)]
   ids=obj.get_valid_id(pre,future)
   if ids:expected.append((list(range(f-7,f+13)),[str(int(v)) for v in ids]))
  # Source files are identical across folds; compare one use and validate all hashes.
  actual=None
  for fold,fr in lock['folds'].items():
   s=read(fr['path']);rows=[m for m in s['members'] if m['official_partition']==name]
   if rows:
    got=[(m['official_frame_ids'],m['agent_ids']) for m in rows]
    if actual is None:actual=got
    assert got==actual
  assert expected==actual,(name,len(expected),len(actual or []))
  checked.append(dict(partition=name,windows=len(expected),member_order_exact=True,source_sha256=row['sha256']))
 write(a/'receipts/OFFICIAL_LOADER_PARITY.json',dict(status='PASS',partitions=checked,loader_sha256=digest(repo/'data/preprocessor.py'),cpu_seconds=time.monotonic()-start,contract='min_past=8,min_future=12,frame_skip=1; get_valid_id from pinned original source'))
 print('PASS',len(checked),'official partitions')
if __name__=='__main__':main(read(sys.argv[1]))

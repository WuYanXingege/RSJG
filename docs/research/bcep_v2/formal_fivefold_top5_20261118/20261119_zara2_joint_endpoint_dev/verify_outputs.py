import csv,json,hashlib
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent;B=O.parents[2];R=B/'RSJG_EBJD_assets/experiments/bcep_ebjd_v2_integrated'
read=lambda p:json.loads(p.read_text())
run=read(R/'FORMAL_RUN.json');plan=read(Path(run['input_plan']));split=read(Path(plan['folds']['zara2']['split']['path']))
with (O/'WINDOW_METRICS.csv').open() as f:wr=list(csv.DictReader(f))
lookup={(int(r['epoch']),int(r['index']),r['method']):float(r['JFDE_m']) for r in wr}
maxdiff=0.;tested=0
for e in [26,3,5]:
 for i in split['selection_dev_indices']:
  g=np.load(R/f'zara2/refined_banks/selection_dev_2035/{i}.npy').astype('float64')
  with np.load(split['records'][i]['path']) as z:y=z['future'][:,-1].astype('float64')
  err=np.linalg.norm(g-y[None],axis=-1)
  with np.load(O/f'assignments/{e}/{i}.npz') as z:
   for key,method in [('pi_initial','initial_Hungarian'),('pi_H','H')]:
    pi=z[key][0];assert np.array_equal(np.sort(pi,axis=0),np.broadcast_to(np.arange(20)[:,None],pi.shape))
    selected=np.take_along_axis(err,pi,axis=0);value=selected.mean(axis=1).min()
    maxdiff=max(maxdiff,abs(value-lookup[e,i,method]));tested+=1
  maxdiff=max(maxdiff,abs(err.mean(1).min()-lookup[e,i,'G']),abs(err.min(0).mean()-lookup[e,i,'oracle']))
assert maxdiff<1e-6
assert read(R/'STATUS.json')['status']=='PAUSED_RESUMABLE'
inv=read(O/'INPUT_INVENTORY.json');now={str(p.relative_to(R)):[p.stat().st_size,p.stat().st_mtime_ns] for p in R.rglob('*') if p.is_file()}
assert inv==now
out={'numpy_float64_independent_metric_check':True,'assignment_tables_tested':tested,'maximum_absolute_metric_difference_m':maxdiff,'run_status':'PAUSED_RESUMABLE','upstream_file_inventory_unchanged':True}
(O/'QA.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

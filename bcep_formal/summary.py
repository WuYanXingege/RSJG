"""CPU-only frozen summaries; inference seeds are not training repetitions."""
import csv,json
from pathlib import Path
import numpy as np
from bcep_joint.common import write

def summarize(plan):
 rows=[]
 for fold in plan['fold_order']:
  p=Path(plan['folds'][fold]['output'])/'evaluation/PER_SEED_METRICS.csv'
  with p.open() as f:rows.extend(csv.DictReader(f))
 out=Path(plan['assets'])/'reports';fields=['minADE','minFDE','JADE','JFDE'];summary=[]
 for fold in plan['fold_order']+['MACRO_FIVEFOLD']:
  for method in ['G','H']:
   ss={}
   for seed in range(2035,2040):
    r=[x for x in rows if x['group']=='all' and x['method']==method and int(x['inference_seed'])==seed and (fold=='MACRO_FIVEFOLD' or x['fold']==fold)]
    assert len(r)==(5 if fold=='MACRO_FIVEFOLD' else 1)
    ss[seed]={k:float(np.mean([float(x[k]) for x in r])) for k in fields}
   summary.append(dict(fold=fold,method=method,**{name:val for k in fields for name,val in [(k+'_mean',float(np.mean([r[k] for r in ss.values()]))),(k+'_sd',float(np.std([r[k] for r in ss.values()],ddof=1)))]}))
 for folder in [out,Path(plan['docs'])]:
  folder.mkdir(parents=True,exist_ok=True)
  for filename,data in [('PER_SEED_METRICS.csv',rows),('FIVEFOLD_METRICS.csv',summary)]:
   with (folder/filename).open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
  write(folder/'FINAL_SUMMARY.json',dict(status='COMPLETE',seed_pair=[3102,3201],inference_seeds=list(range(2035,2040)),summary=summary,units='m',aggregation='per-seed fivefold equal macro; SD across inference seeds, not training stability',automatic_followup=False))

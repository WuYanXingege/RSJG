"""Final test only after best freeze. Resumable no-future trajectory exports."""
from pathlib import Path
import json,csv,time
import numpy as np
import torch
from bcep.contracts import canonical_hash,digest
from bcep_joint.common import write,seed_all
from bcep_joint.model import build
from bcep_joint.data import CandidateBank
from bcep_joint.inference import generate,shuffle_paths,metric_invariants
from ebjd.metrics import MetricAccumulator,hard_metrics
from ebjd.unified import gather_candidates
from .data import FormalData
from .banks import Paused
from .control import pause_requested

def evaluate(split,c,checkpoint,banks,output,device='cuda'):
 output=Path(output);output.mkdir(exist_ok=True);checkpoint_sha=digest(checkpoint);identity=canonical_hash(split)
 if (output/'COMPLETE.json').exists():
  done=json.loads((output/'COMPLETE.json').read_text());assert done['checkpoint_sha256']==checkpoint_sha and done['split_hash']==identity;return done
 seed_all(3201);m=build(c).to(device);state=torch.load(checkpoint,map_location=device,weights_only=False);m.load_state_dict(state['model'],strict=True);m.eval()
 obs=FormalData(split,'export');indices=split['heldout_confirmation_indices'];members=[split['members'][i] for i in indices];frozen=[]
 for seed in range(2035,2040):
  receipt=banks[str(seed)];bank=CandidateBank(receipt['path'],split,'heldout_confirmation',receipt['provider_sha256'],seed);folder=output/f'seed{seed}';folder.mkdir(exist_ok=True)
  expected=dict(checkpoint_sha256=checkpoint_sha,split_hash=identity,bank_sha256=bank.sha,seed=seed,K=20,steps=20,unit='m',contains_truth=False)
  ip=folder/'IDENTITY.json'
  if ip.exists():assert json.loads(ip.read_text())==expected
  else:write(ip,expected)
  rows=[]
  for j,i in enumerate(indices):
   file=folder/f'{j:06d}.npz'
   if not file.exists():
    if pause_requested():raise Paused('final evaluation paused at window boundary')
    batch=obs.batch([i],'heldout_confirmation',False);goals=bank.batch(batch).to(device);b=batch.to(device)
    g,h,pi,r=generate(m,b.observed,b.semantic_maps,b.valid,b.scene_ids,b.agent_ids,goals,seed)
    tmp=file.with_suffix('.tmp.npz');np.savez_compressed(tmp,G=g.cpu().numpy(),H=h.cpu().numpy(),permutation=pi.cpu().numpy(),valid=b.valid.cpu().numpy());tmp.replace(file)
   rows.append(dict(path=str(file),sha256=digest(file),member=members[j]))
   if j%50==0:write(output/'PROGRESS.json',dict(phase='NO_FUTURE_EXPORT',seed=seed,window=j+1,total=len(indices)))
  frozen.append(dict(seed=seed,identity=expected,rows=rows))
 write(output/'OUTPUTS_FROZEN.json',dict(status='ALL_OUTPUTS_FROZEN',outputs=frozen))
 del m,state;torch.cuda.empty_cache() if device=='cuda' else None
 data=FormalData(split,'scoring');results=[];invariants=[]
 for out in frozen:
  seed=out['seed'];b=banks[str(seed)];bank=CandidateBank(b['path'],split,'heldout_confirmation',b['provider_sha256'],seed)
  acc={group:{mode:MetricAccumulator() for mode in ['G','H']+[f'H_shuffle_{j}' for j in range(5)]} for group in ['all','N1','Nge2']};max_delta=0.
  for j,i in enumerate(indices):
   x=data.item(i,'heldout_confirmation');truth=x['future'][None];row=out['rows'][j];assert digest(row['path'])==row['sha256']
   with np.load(row['path'],allow_pickle=False) as z:g=torch.from_numpy(z['G']);h=torch.from_numpy(z['H']);pi=torch.from_numpy(z['permutation']);v=torch.from_numpy(z['valid'])
   goals=bank.goals[j:j+1,:,:v.shape[1]];r=metric_invariants(g,h,goals,gather_candidates(goals,pi),truth,v);max_delta=max(max_delta,r['minFDE_G_H_abs']);modes=dict(G=g,H=h)
   for rep in range(5):
    sh=shuffle_paths(h,v,[x['scene_id']],seed,rep);hm=hard_metrics(h,truth,v);sm=hard_metrics(sh,truth,v)
    assert abs(hm.minADE-sm.minADE)<=1e-6 and abs(hm.minFDE-sm.minFDE)<=1e-6;modes[f'H_shuffle_{rep}']=sh
   for group in ['all','N1' if int(v.sum())==1 else 'Nge2']:
    for mode,path in modes.items():acc[group][mode].update(path,truth,v)
  for group,aa in acc.items():
   for mode,v in aa.items():
    if v.scene_count:results.append(dict(fold=split['fold'],inference_seed=seed,method=mode,group=group,**{k:x for k,x in v.as_dict().items() if k!='sums'}))
  invariants.append(dict(seed=seed,max_GH_minFDE_difference=max_delta,endpoint_and_gather=True,shuffle_marginals=True))
 with (output/'PER_SEED_METRICS.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(results[0]),lineterminator='\n');w.writeheader();w.writerows(results)
 done=dict(status='COMPLETE',checkpoint_sha256=checkpoint_sha,split_hash=identity,rows=len(results),invariants=invariants);write(output/'COMPLETE.json',done);return done

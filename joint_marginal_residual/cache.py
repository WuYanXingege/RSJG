"""Atomic observed-only teacher/features shards and separately stored labels."""
import time,os
from pathlib import Path
from contextlib import contextmanager
import torch
from bcep.contracts import canonical_hash,digest
from bcep_joint.common import read,write,save
from bcep_formal.data import FormalData
from ebjd.metrics import MetricAccumulator
from joint_flow_dependency.inference import generate
from joint_flow_fourfold.inputs import state_hash
from .assets import RUN,load_teacher
from .features import coverage_statistics,frozen_features,FEATURE_VERSION

SEEDS={'train':2036,'selection_dev':2035}

def immutable(path,value):
 if path.exists():assert read(path)==value,str(path)
 else:write(path,value)

@contextmanager
def observed_only():
 original=FormalData.item
 def item(self,index,purpose,truth=True):
  if truth or purpose not in SEEDS:raise ValueError('feature export reads only train/dev observations')
  return original(self,index,purpose,truth=False)
 FormalData.item=item
 try:yield
 finally:FormalData.item=original

@torch.no_grad()
def build(fold,device='cuda'):
 start=time.monotonic();start_unix=time.time();inputs,model,provider,dep,identity=load_teacher(fold,device)
 output=RUN/fold/'cache';output.mkdir(parents=True,exist_ok=True)
 immutable(output/'IDENTITY.json',identity)
 immutable(output/'CONFIG.json',dict(identity=identity,seeds=SEEDS,M=256,K=20,DDIM=20,eta=0,
  feature_version=FEATURE_VERSION,prediction_device=device,dtype='float32',truth_in_features=False,
  execution_policy=os.environ.get('RSJG_DISABLE_OPTIMIZATIONS',''),run_config_sha256=digest(RUN/'RUN_CONFIG.json')))
 timeline_path=output/'INVOCATIONS.json'
 timeline=read(timeline_path) if timeline_path.exists() else dict(first_start_unix=start_unix,invocations=[])
 invocation=dict(start_unix=start_unix,measured_active_seconds=0.,completed=False)
 timeline['invocations'].append(invocation)
 def heartbeat(completed=False):
  invocation.update(last_checkpoint_unix=time.time(),measured_active_seconds=time.monotonic()-start,completed=completed)
  write(timeline_path,timeline)
 heartbeat()
 if device=='cuda':torch.cuda.reset_peak_memory_stats()
 config_hash=digest(output/'CONFIG.json');manifests={};io_seconds=0.;decode_seconds=0.;feature_seconds=0.
 for purpose,seed in SEEDS.items():
  folder=output/purpose;folder.mkdir(exist_ok=True);rows=[]
  with observed_only():
   for pos,index in enumerate(inputs.split[purpose+'_indices']):
    path=folder/f'{pos:05d}.pt';receipt=path.with_suffix('.json')
    if receipt.exists():
     row=read(receipt);assert row['config_sha256']==config_hash and row['sha256']==digest(path)
    else:
     tick=time.monotonic();host=inputs.data.batch([index],purpose,truth=False);batch=host.to(device)
     if device=='cuda':torch.cuda.synchronize()
     io=time.monotonic()-tick;tick=time.monotonic()
     y,endpoints,detail=generate(model,provider,dep,batch,seed=seed)
     if device=='cuda':torch.cuda.synchronize()
     inference=time.monotonic()-tick;tick=time.monotonic()
     features=frozen_features(host.observed[0],detail['flow_context'].cpu(),y[0].cpu(),
      coverage_statistics(endpoints['raw'].cpu(),endpoints['raw_indices'].cpu(),host.valid[0]),host.valid[0])
     feature_time=time.monotonic()-tick
     item={k:v.detach().cpu() for k,v in features.items() if torch.is_tensor(v)}
     item.update(B=y[0].cpu(),observed=host.observed[0],raw_ids=endpoints['raw_indices'].cpu())
     assert set(item).isdisjoint({'truth','future','label'})
     tick=time.monotonic();save(item,path);shard_sha=digest(path);write_seconds=time.monotonic()-tick
     row=dict(window=f'{purpose}_{pos+1:05d}',n=int(host.valid.sum()),seed=seed,file=path.name,
      sha256=shard_sha,config_sha256=config_hash,raw_ids=item['raw_ids'].tolist(),
      inference_seconds=inference,feature_seconds=feature_time,io_seconds=io,shard_write_hash_seconds=write_seconds,ddim_calls=detail['ddim_calls'])
     write(receipt,row)
    rows.append(row)
    if (pos+1)%25==0 or pos+1==len(inputs.split[purpose+'_indices']):
     write(RUN/fold/'PROGRESS.json',dict(stage='CACHE_OBSERVATIONS_ONLY',purpose=purpose,windows=pos+1,total=len(inputs.split[purpose+'_indices']),unix=time.time()))
     heartbeat()
  immutable(folder/'COMPLETE.json',dict(config_sha256=config_hash,rows=rows))
  manifests[purpose]=rows
 # Every prediction/feature shard exists before labels are read by the scoring side.
 assert identity['states']=={k:state_hash(m) for k,m in dict(backbone=model,provider=provider,dependence=dep).items()}
 label_root=RUN/fold/'labels';label_root.mkdir(exist_ok=True);baseline={}
 for purpose in SEEDS:
  folder=label_root/purpose;folder.mkdir(exist_ok=True)
  acc={g:MetricAccumulator() for g in ('all','N_ge_9')}
  for pos,index in enumerate(inputs.split[purpose+'_indices']):
   p=folder/f'{pos:05d}.pt';side=p.with_suffix('.json');row=manifests[purpose][pos]
   if side.exists():assert read(side)['sha256']==digest(p) and read(side)['prediction_sha256']==row['sha256']
   else:
    truth=inputs.data.batch([index],purpose,truth=True).future[0]
    save(dict(truth=truth),p);write(side,dict(sha256=digest(p),prediction_sha256=row['sha256']))
   features=torch.load(output/purpose/row['file'],weights_only=False)
   truth=torch.load(p,weights_only=False)['truth'];valid=features['valid'][None]
   acc['all'].update(features['B'][None],truth[None],valid)
   if row['n']>=9:acc['N_ge_9'].update(features['B'][None],truth[None],valid)
  baseline[purpose]={g:a.as_dict() if a.scene_count else None for g,a in acc.items()}
  heartbeat()
 write(output/'BASELINE_METRICS.json',baseline)
 scales={q:max(baseline['train']['all'][q],1e-6) for q in ('minADE','minFDE','JADE','JFDE')}
 write(output/'TRAIN_SCALES.json',scales)
 rows=[r for values in manifests.values() for r in values]
 heartbeat(completed=True)
 write(output/'TIMING.json',dict(wall_seconds=sum(r['measured_active_seconds'] for r in timeline['invocations']),
  wall_seconds_definition='sum of observed active invocation durations; interrupted invocations stop at their last durable heartbeat',
  current_invocation_wall_seconds=invocation['measured_active_seconds'],
  first_start_to_completion_elapsed_seconds=time.time()-timeline['first_start_unix'],
  invocation_count=len(timeline['invocations']),interrupted_invocations=sum(not r['completed'] for r in timeline['invocations']),
  inference_seconds=sum(r['inference_seconds'] for r in rows),feature_seconds=sum(r['feature_seconds'] for r in rows),
  io_seconds=sum(r['io_seconds'] for r in rows),shard_write_hash_seconds=sum(r.get('shard_write_hash_seconds',0.) for r in rows),
  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
  bytes=sum(p.stat().st_size for p in output.rglob('*.pt')),
  actual_unique_teacher_decodes=len(rows),per_head_epoch_DDIM_calls=0,upstream_weights_unchanged=True,
  prior_prediction_cache_reuse=0,reason='canonical CUDA teacher; previous CPU dev caches are not treated as bit-identical CUDA predictions'))
 return baseline

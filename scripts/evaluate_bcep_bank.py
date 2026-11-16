"""Independent BCEP bank export. Untrained smoke output is explicitly nonformal."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ebjd.data import SceneManifestDataset
from bcep.provider import BCEP
from bcep.evaluation import endpoint_metrics

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--random-init-smoke',action='store_true');p.add_argument('--split-lock',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=2035);p.add_argument('--limit',type=int);a=p.parse_args()
 if bool(a.checkpoint)==a.random_init_smoke:raise ValueError('specify exactly one source')
 torch.set_num_threads(2);torch.manual_seed(3101);lock=json.loads(a.split_lock.read_text());ds=SceneManifestDataset(a.manifest)
 model=BCEP(ds.metadata['pred_length'],ds.metadata['dt_seconds']);model.scales.copy_(torch.tensor(lock['scales']));model.scales_fitted.fill_(True)
 if a.checkpoint:
  c=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
  assert c['identity']['weights_origin']=='random_initialization' and c['identity']['teacher_weights'] is False
  assert c['identity']['split']==hashlib.sha256(a.split_lock.read_bytes()).hexdigest()
  model.load_state_dict(c['model'],strict=True);checkpoint_sha=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
 else:checkpoint_sha='UNTRAINED_SMOKE_NOT_FORMAL'
 model.eval();rows=[]
 indices=range(min(len(ds),a.limit or len(ds)))
 if a.random_init_smoke:
  assert a.limit and a.limit<=4
  indices=[lock['train_indices'][0],lock['dev_indices'][0]][:a.limit]
 for index in indices:
  item=ds[index];obs=item['observed'];valid=torch.ones(len(obs),dtype=torch.bool)
  with torch.no_grad():
   before=torch.get_rng_state().clone();g,meta=model.predict(obs,item['semantic_maps'],valid,item['scene_id'],item['agent_ids'],seed=a.seed)
   assert torch.equal(before,torch.get_rng_state())
   again,_=model.predict(obs,item['semantic_maps'],valid,item['scene_id'],item['agent_ids'],seed=a.seed)
   assert torch.equal(g,again) and g.shape==(20,len(obs),2) and torch.isfinite(g).all()
   loss=model.loss(obs,item['semantic_maps'],valid,item['future']);assert torch.isfinite(loss)
  rows.append(dict(endpoints=g.numpy(),truth=item['future'][:,-1].numpy(),scene_id=item['scene_id'],agent_ids=item['agent_ids'],frame_ids=item['frame_ids'],n=len(obs),nll=float(loss),metadata=meta))
 maximum=max(r['n'] for r in rows);endpoints=np.zeros((len(rows),20,maximum,2),np.float32);truth=np.zeros((len(rows),maximum,2),np.float32);valid=np.zeros((len(rows),maximum),bool);ids=np.full((len(rows),maximum),'',dtype='<U100')
 for i,r in enumerate(rows):
  n=r['n'];endpoints[i,:,:n]=r['endpoints'];truth[i,:n]=r['truth'];valid[i,:n]=True;ids[i,:n]=r['agent_ids']
 metadata=dict(checkpoint_sha256=checkpoint_sha,manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),split='interface_smoke' if a.random_init_smoke else ds.metadata.get('split','unknown'),coordinate_unit='m',worlds=20,seed=a.seed,provider='bcep',support=256,real_optimizer_updates=0,formal=not a.random_init_smoke and a.limit is None,frame_ids=[r['frame_ids'] for r in rows])
 a.output.parent.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(a.output,endpoints=endpoints,truth=truth,valid=valid,scene_ids=np.array([r['scene_id'] for r in rows]),agent_ids=ids,metadata_json=json.dumps(metadata))
 receipt=dict(status='PASS',metadata=metadata,scenes=[{k:r[k] for k in ['scene_id','n','nll']} for r in rows],bank_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),interpretation='untrained interface check; NLL is not model quality evidence' if a.random_init_smoke else 'frozen BCEP output')
 receipt['endpoint_metrics']=endpoint_metrics(endpoints,truth,valid)
 a.output.with_suffix('.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
if __name__=='__main__':main()

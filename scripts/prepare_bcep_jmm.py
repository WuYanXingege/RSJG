"""Build all five official file-based splits; never infer train/val boundaries."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse, importlib.util, json, sys, tempfile, time, math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from bcep.contracts import digest, canonical_hash
from bcep_joint.common import write
from bcep.motion_frame import frame, to_local
from ebjd.export_ethucy import synchronized_windows,semantic_crops,SEQUENCE_TO_SCENE
from ebjd.representation import EndpointBridgeDecomposition,cv_baseline

ORDER=['zara2','zara1','hotel','univ','eth']

def load_module(path):
 s=importlib.util.spec_from_file_location('official_split',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def main(paths):
 a=Path(paths['assets']);src=a/'sources';repo=src/'Joint_AgentFormer';raw=repo/'datasets/eth_ucy';root=Path(paths['root']);docs=Path(paths['docs']);docs.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 official=load_module(src/'dataset_splits.py');af=load_module(repo/'data/ethucy_split.py')
 lock=dict(schema='bcep-jmm-fivefold-lock-v1',jmm_commit='894182b5ea7dac3c93bbe0bcc18c156c932398b7',agentformer_commit='f4c29aa587026a4bce20c631494bf27eac09c93f',sources=[dict(path=str(x),sha256=digest(x)) for x in [src/'dataset_splits.py',repo/'data/ethucy_split.py',repo/'data/dataloader.py',repo/'data/preprocessor.py']],coordinate_policy='official world-meter columns 13/15, no learned normalization; frame ids official*10 per official loader frame_scale; dt=.4',window_policy='existing synchronized_windows: all 20 frames present, complete agents, stride one, no cross-file windows',folds={},files={},maps={})
 maps={};unique={};recorddir=a/'data/records';recorddir.mkdir(exist_ok=True);convdir=a/'data/converted';convdir.mkdir(exist_ok=True)
 for fold in ORDER:
  groups=official.get_ethucy_split(fold);assert groups==af.get_ethucy_split(fold)
  members=[];records=[];indices={};occurrences={};sequence_files=[]
  for purpose,names in zip(['train','selection_dev','heldout_confirmation'],groups):
   indices[purpose]=[];occurrences[purpose]=set()
   for name in names:
    f=raw/fold/(name+'.txt');h=digest(f);rec=name.removesuffix('_train').removesuffix('_val');scene=SEQUENCE_TO_SCENE[rec]
    sequence_files.append(dict(purpose=purpose,recording=rec,path=str(f),sha256=h,official_partition=name))
    if name in lock['files']:assert lock['files'][name]['sha256']==h,'same official partition differs across folds'
    lock['files'][name]=dict(path=str(f),sha256=h)
    if name not in unique:
     arr=np.loadtxt(f,usecols=[0,1,13,15],dtype=np.float64);assert np.isfinite(arr).all()
     # Save the explicit source->evaluation frame correspondence, not guessed cuts.
     c=convdir/(name+'.txt');np.savetxt(c,arr,fmt='%.12g');wins=synchronized_windows(c)
     if scene not in maps:
      hm=root/'GDTS/data/eth5'/scene/'H.txt';mp=root/'GDTS/data/eth5'/scene/'pred_mask.png'
      maps[scene]=(np.loadtxt(hm),np.asarray(Image.open(mp),dtype=np.uint8));lock['maps'][scene]=dict(homography_path=str(hm),homography_sha256=digest(hm),semantic_path=str(mp),semantic_sha256=digest(mp))
     hp,labels=maps[scene];prepared=[]
     for j,w in enumerate(wins):
      frames=w['frame_ids'].tolist();ids=[str(x) for x in w['agent_ids']];coords=w['coordinates'];key=canonical_hash(dict(source=h,frames=frames,agents=ids,map=lock['maps'][scene]))
      target=recorddir/(key+'.npz')
      if not target.exists():
       crop=semantic_crops(coords[:,7],labels,hp,scene);codes=np.where(crop.sum(1)>0,crop.argmax(1)+1,0).astype(np.uint8)
       if j==0:assert np.array_equal(crop,(codes[:,None]==np.arange(1,7,dtype=np.uint8)[None,:,None,None]).astype(np.uint8))
       tmp=target.with_suffix('.tmp.npz');np.savez_compressed(tmp,observed=coords[:,:8],future=coords[:,8:],map_codes=codes);tmp.replace(target)
      prepared.append((dict(sequence=rec,scene_id=f'jmm:{rec}:{frames[0]}-{frames[-1]}',official_partition=name,official_frame_ids=frames,frame_ids=[v*10 for v in frames],agent_ids=ids,source_file_sha256=h),dict(path=str(target),sha256=digest(target))))
      if j%100==0:write(a/'PREPARATION_PROGRESS.json',dict(stage='OFFICIAL_RECORDS',partition=name,window=j+1,total=len(wins),fold=fold))
     unique[name]=prepared
     print('PREPARED',name,len(prepared),'windows',flush=True)
    for m,r in unique[name]:
     i=len(members);members.append(dict(m,index=i));records.append(r);indices[purpose].append(i)
     occurrences[purpose].update((rec,f,agent) for f in m['official_frame_ids'] for agent in m['agent_ids'])
  overlaps={f'{p}:{q}':len(occurrences[p]&occurrences[q]) for j,p in enumerate(occurrences) for q in list(occurrences)[j+1:]}
  assert not any(overlaps.values()),f'official cross-purpose occurrence overlap: {fold} {overlaps}'
  s=dict(schema='bcep-jmm-official-fold-v1',fold=fold,members=members,records=records,unit='m',dt=.4,pred_length=12,scale_source='train_only',source_files=sequence_files,official_overlap_counts=overlaps,**{p+'_indices':ix for p,ix in indices.items()})
  # Fit both normalization families from train only. No maps or val/test labels read.
  total=torch.zeros(12,2,dtype=torch.float64);n=0;gs=bs=0.;gn=bn=0;rep=EndpointBridgeDecomposition()
  for i in indices['train']:
   with np.load(records[i]['path'],allow_pickle=False) as z:obs=torch.from_numpy(z['observed'].copy()).double();fut=torch.from_numpy(z['future'].copy()).double()
   for horizon in range(1,13):
    b,r,scale=frame(obs,horizon,.4,torch.ones(2,dtype=torch.float64));total[horizon-1]+=to_local(fut[:,horizon-1],b,r,scale).square().sum(0)
   n+=len(obs);base,_=cv_baseline(obs[None]);latent=rep._encode_unscaled(fut[None],base)
   gs+=float(latent[...,11:,:].square().sum());gn+=latent[...,11:,:].numel();bs+=float(latent[...,:11,:].square().sum());bn+=latent[...,:11,:].numel()
  s['scales']=(total/n).sqrt().clamp_min(.1).tolist();s['bridge_scales']=dict(goal=max((gs/gn)**.5,.5),bridge=max((bs/bn)**.5,.1));s['scale_agent_count']=n
  s['statistics']={p:dict(windows=len(ix),agent_occurrences=sum(len(members[i]['agent_ids']) for i in ix),mean_agents=float(np.mean([len(members[i]['agent_ids']) for i in ix])),max_agents=max(len(members[i]['agent_ids']) for i in ix)) for p,ix in indices.items()}
  folder=a/'folds'/fold;folder.mkdir(exist_ok=True);sp=folder/'SPLIT_LOCK.json';write(sp,s)
  lock['folds'][fold]=dict(path=str(sp),sha256=digest(sp),canonical_sha256=canonical_hash(s),statistics=s['statistics'],official_overlap_counts=overlaps)
  write(a/'JMM_SPLIT_LOCK.json',lock)
 write(docs/'JMM_SPLIT_LOCK.json',lock);write(a/'PREPARATION_PROGRESS.json',dict(status='COMPLETE',unique_partitions=len(unique),unique_records=sum(len(x) for x in unique.values()),formal_optimizer_updates=0))
 print(json.dumps({k:v['statistics'] for k,v in lock['folds'].items()},indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--paths',required=True);main(json.loads(Path(p.parse_args().paths).read_text()))

"""Whole-sequence internal split; no overlap windows across train/dev."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bcep.motion_frame import frame,to_local

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 m=json.loads(a.manifest.read_text());meta=m['metadata'];assert meta['unit']=='m' and meta.get('split')=='train', 'only training fold can define internal split'
 rows=[];values=[]
 for shard in m['shards']:
  path=(a.manifest.parent/shard['path']).resolve()
  assert digest(path)==shard['sha256']
  with np.load(path,allow_pickle=True) as z:
   for seq,sid,frames,agents,obs,fut in zip(z['source_sequences'],z['scene_ids'],z['frame_ids'],z['agent_ids'],z['observed'],z['future']):
    rows.append({'index':len(rows),'sequence':str(seq),'scene_id':str(sid),'frame_ids':np.asarray(frames).tolist(),'agent_ids':[str(x) for x in agents]});values.append((np.asarray(obs),np.asarray(fut)))
 sequences=sorted(set(x['sequence'] for x in rows));assert len(sequences)>1
 dev_sequences=sequences[-max(1,len(sequences)//5):]
 train=[x['index'] for x in rows if x['sequence'] not in dev_sequences];dev=[x['index'] for x in rows if x['sequence'] in dev_sequences]
 assert {rows[i]['sequence'] for i in train}.isdisjoint({rows[i]['sequence'] for i in dev})
 total=torch.zeros(meta['pred_length'],2,dtype=torch.float64);count=0
 for i in train:
  obs,fut=values[i];obs=torch.tensor(obs,dtype=torch.float64);fut=torch.tensor(fut,dtype=torch.float64)
  for h in range(1,meta['pred_length']+1):
   b,r,s=frame(obs,h,meta['dt_seconds'],torch.ones(2,dtype=torch.float64));total[h-1]+=to_local(fut[:,h-1],b,r,s).square().sum(0)
  count+=len(obs)
 scales=(total/count).sqrt().clamp_min(.1)
 result={'source_manifest':str(a.manifest.resolve()),'source_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'unit':'m','dt':meta['dt_seconds'],'pred_length':meta['pred_length'],'train_indices':train,'dev_indices':dev,'members':rows,'dev_sequences':dev_sequences,'scales':scales.tolist(),'scale_source':'internal_train_only','matched_reference_dev_overlap':True,'fair_matched_dev_comparison':False,'semantic_provenance':m.get('map_sources',{}),'allow_real_training':False}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print({'train':len(train),'dev':len(dev),'dev_sequences':dev_sequences,'train_scale_agents':count})
if __name__=='__main__':main()

import json
from pathlib import Path
import numpy as np
import torch
from bcep.contracts import require,canonical_hash,digest,check_item
from bcep_confirmation.protocol import PurposeDataset
from ebjd.data import collate_scenes
from .common import read

class JointData:
    def __init__(self,split,phase='training',frozen=None):
        self.split=split;self.dataset=PurposeDataset(split,phase,frozen)
    def item(self,index,purpose,truth=True):
        d=self.dataset.get(index,purpose,truth)
        d['metadata']={'dt':.4,'unit':'m','obs_length':8,'pred_length':12}
        if not truth:d['future']=torch.zeros(len(d['observed']),12,2) # API carrier only, never real future
        return d
    def batch(self,indices,purpose,truth=True):return collate_scenes([self.item(i,purpose,truth) for i in indices])

class CandidateBank:
    def __init__(self,path,split,purpose,provider_sha,seed):
        self.path=str(path);self.sha=digest(path)
        with np.load(path,allow_pickle=False) as z:
            self.meta=json.loads(str(z['metadata_json']));self.goals=torch.from_numpy(z['candidates'].copy());self.valid=z['valid'].copy()
        m=self.meta;members=[split['members'][i] for i in split[purpose+'_indices']]
        require(m['schema']=='bcep-joint-candidates-v1','bank schema mismatch')
        require(m['purpose']==purpose and m['split_hash']==canonical_hash(split),'bank purpose/split mismatch')
        require(m['provider_sha256']==provider_sha and m['seed']==seed,'provider/seed mismatch')
        require(m['members']==members and m['member_hash']==canonical_hash(members),'bank members mismatch')
        require(m['unit']=='m' and m['K']==20 and m['M']==256 and self.goals.dtype==torch.float32,'bank coordinate/dtype protocol')
        require(self.goals.ndim==4 and self.goals.shape[0]==len(members) and self.goals.shape[1]==20 and self.goals.shape[-1]==2,'bank shape')
        require(self.valid.shape==self.goals.shape[::2][:2],'bank mask shape')
        self.rows={v['scene_id']:i for i,v in enumerate(members)}
        require(len(self.rows)==len(members) and bool(torch.isfinite(self.goals).all()),'duplicate/nonfinite bank')
        for i,m in enumerate(members):require(int(self.valid[i].sum())==len(m['agent_ids']) and self.valid[i,:len(m['agent_ids'])].all(),'invalid bank padding')
    def batch(self,batch):
        out=torch.zeros(len(batch.scene_ids),20,batch.valid.shape[1],2)
        for i,scene in enumerate(batch.scene_ids):
            require(scene in self.rows,'unknown bank scene');row=self.rows[scene];m=self.meta['members'][row]
            require(batch.agent_ids[i]==m['agent_ids'] and batch.frame_ids[i]==m['frame_ids'],'agent/frame identity differs')
            pos=batch.valid[i].nonzero().flatten();require(len(pos)==len(m['agent_ids']),'empty/masked bank member')
            out[i,:,pos]=self.goals[row,:,:len(pos)]
        return out

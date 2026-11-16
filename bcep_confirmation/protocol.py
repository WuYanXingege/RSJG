"""New four-purpose schema; no weakened historical split checks."""
import json
from pathlib import Path
import numpy as np
import torch
from bcep.contracts import canonical_hash, digest, check_item, require

PURPOSES=('train','selection_dev','heldout_confirmation','excluded')

def validate_split(s):
    require(s['schema']=='bcep-recording-confirmation-v1','schema mismatch')
    sets={p:set(s[p+'_indices']) for p in PURPOSES}
    require(set.union(*sets.values())==set(range(len(s['members']))),'universe coverage differs')
    for p in PURPOSES:
        require(len(sets[p])==len(s[p+'_indices']),'duplicate index')
        require(canonical_hash([s['members'][i] for i in s[p+'_indices']])==s['member_hashes'][p],'member hash differs')
        for q in PURPOSES:
            if p!=q:require(not sets[p]&sets[q],'purpose overlap')
    recordings={p:{s['recording_ids'][s['members'][i]['sequence']] for i in sets[p]} for p in PURPOSES[:3]}
    for p in recordings:
        for q in recordings:
            if p!=q:require(not recordings[p]&recordings[q],'recording alias leakage')
    # A physical recording/frame/agent occurrence cannot cross purposes.
    seen={}
    for p in PURPOSES[:3]:
        for i in s[p+'_indices']:
            m=s['members'][i];rec=s['recording_ids'][m['sequence']]
            for a in m['agent_ids']:
                for f in m['frame_ids']:
                    key=(rec,str(a),int(f));require(key not in seen or seen[key]==p,'cross-purpose temporal overlap');seen[key]=p
    require(s['scale_source']=='train_only','scale scope differs')
    require(s['unit']=='m' and s['dt']==.4 and s['pred_length']==12,'physical protocol differs')
    return sets

class PurposeDataset:
    """Field-selective NPZ reads. Confirmation future is inaccessible until scoring.

    Original SceneManifestDataset loads all NPZ fields even for observations;
    this adapter deliberately avoids materializing future in export/preparation.
    """
    def __init__(self,split,phase='training',frozen_outputs=None):
        self.split=split;self.sets=validate_split(split);self.phase=phase;self.access=[]
        self.manifest_path=Path(split['source_manifest']);self.manifest=json.loads(self.manifest_path.read_text())
        require(digest(self.manifest_path)==split['source_sha256'],'manifest changed')
        self.index=[]
        for descriptor in self.manifest['shards']:
            self.index.extend((descriptor,local) for local in range(descriptor['scenes']))
        require(len(self.index)==len(split['members']),'shard index count differs')
        if phase=='scoring':
            require(frozen_outputs is not None,'missing output freeze')
            f=json.loads(Path(frozen_outputs).read_text())
            require(f['status']=='ALL_OUTPUTS_FROZEN' and f['split_hash']==canonical_hash(split),'outputs not frozen for split')
    def get(self,index,purpose,include_future=True):
        require(purpose in self.sets and index in self.sets[purpose],'purpose/member mismatch')
        require(purpose!='excluded','excluded data forbidden')
        if self.phase=='training':require(purpose in ('train','selection_dev'),'confirmation forbidden during training/selection')
        elif self.phase=='scale':require(purpose=='train','normalization only reads train')
        elif self.phase=='export':require(purpose=='heldout_confirmation' and not include_future,'export must be observed-only')
        elif self.phase=='scoring':require(purpose=='heldout_confirmation','scoring scope mismatch')
        else:raise ValueError('unknown dataset phase')
        descriptor,local=self.index[index];path=self.manifest_path.parent/descriptor['path']
        self.access.append((index,purpose,include_future))
        with np.load(path,allow_pickle=True) as z:
            item=dict(observed=torch.as_tensor(np.asarray(z['observed'][local]),dtype=torch.float32),
                scene_id=str(z['scene_ids'][local]),agent_ids=[str(x) for x in z['agent_ids'][local]],
                frame_ids=[int(x) for x in z['frame_ids'][local]],source_sequence=str(z['source_sequences'][local]),
                timestamps=[float(x) for x in z['timestamps'][local]],purpose=purpose)
            if self.phase!='scale':item['semantic_maps']=torch.as_tensor(np.asarray(z['semantic_maps'][local]),dtype=torch.uint8)
            if include_future:item['future']=torch.as_tensor(np.asarray(z['future'][local]),dtype=torch.float32)
        check_item(item,self.split['members'][index]);n=len(item['agent_ids'])
        require(item['observed'].shape==(n,8,2) and bool(torch.isfinite(item['observed']).all()),'invalid observations')
        if include_future:require(item['future'].shape==(n,12,2) and bool(torch.isfinite(item['future']).all()),'invalid future')
        if 'semantic_maps' in item:
            maps=item['semantic_maps'];require(maps.shape==(n,6,256,256),'invalid maps')
            require(bool(((maps==0)|(maps==1)).all()) and bool((maps.sum(1)<=1).all()),'map values differ')
        require(np.allclose(np.diff(item['timestamps']),.4,atol=1e-7),'time alignment differs')
        return item

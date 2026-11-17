"""One-time integrity checks, compact categorical crops and bounded input cache."""
from collections import OrderedDict
from pathlib import Path
import json
import numpy as np
import torch
from bcep.contracts import canonical_hash, digest, require, check_item
from ebjd.data import collate_scenes

class FormalData:
    def __init__(self, split, phase='training', cache_bytes=256*1024*1024, verify=True):
        require(split['schema']=='bcep-jmm-official-fold-v1','wrong split schema')
        self.split=split; self.phase=phase; self.cache=OrderedDict(); self.cache_bytes=cache_bytes; self.used=0
        self.sets={p:set(split[p+'_indices']) for p in ('train','selection_dev','heldout_confirmation')}
        require(set.union(*self.sets.values())==set(range(len(split['members']))),'incomplete membership')
        for p,s in self.sets.items():
            require(len(s)==len(split[p+'_indices']),'duplicate member')
            require(all(not s&t for q,t in self.sets.items() if q!=p),'purpose overlap')
        if verify:
            for row in split['records']:
                require(digest(row['path'])==row['sha256'],'record changed')
        self.metadata={'dt':.4,'unit':'m','obs_length':8,'pred_length':12}
    def _raw(self,i):
        if i in self.cache:
            self.cache.move_to_end(i);return self.cache[i]
        with np.load(self.split['records'][i]['path'],allow_pickle=False) as z:
            # Keep future separate: observed-only exports do not materialize it.
            d={k:z[k].copy() for k in ('observed','map_codes')}
        size=sum(x.nbytes for x in d.values())
        while self.cache and self.used+size>self.cache_bytes:
            _,x=self.cache.popitem(last=False);self.used-=sum(v.nbytes for v in x.values())
        if size<=self.cache_bytes:self.cache[i]=d;self.used+=size
        return d
    def item(self,i,purpose,truth=True):
        require(purpose in self.sets and i in self.sets[purpose],'purpose mismatch')
        if self.phase in ('training','scale'):require(purpose!='heldout_confirmation','test inaccessible before freeze')
        if self.phase=='scale':require(purpose=='train','scale must use train')
        if self.phase=='export':require(not truth,'export must be observed-only')
        m=self.split['members'][i];d=self._raw(i);n=len(m['agent_ids'])
        # Cached arrays never returned as writable views. Labels 1..6 encode classes
        # 0..5; code 0 is outside map and decodes to six zeros exactly.
        codes=torch.from_numpy(d['map_codes']);maps=(codes[:,None]==torch.arange(1,7,dtype=torch.uint8)[None,:,None,None]).to(torch.uint8)
        item=dict(observed=torch.from_numpy(d['observed'].copy()),semantic_maps=maps,scene_id=m['scene_id'],agent_ids=list(m['agent_ids']),frame_ids=list(m['frame_ids']),source_sequence=m['sequence'],timestamps=[f*.4 for f in m['official_frame_ids']],metadata=self.metadata)
        if truth:
            with np.load(self.split['records'][i]['path'],allow_pickle=False) as z:item['future']=torch.from_numpy(z['future'].copy())
        else:item['future']=torch.zeros(n,12,2)
        check_item(item,m);return item
    def batch(self,indices,purpose,truth=True):return collate_scenes([self.item(i,purpose,truth) for i in indices])
    def __getitem__(self,i):
        p='train' if i in self.sets['train'] else 'selection_dev'
        return self.item(i,p)

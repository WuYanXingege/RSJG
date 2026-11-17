"""Frozen, resumable per-window candidate export. No test targets are read."""
from pathlib import Path
import json,time
import numpy as np
import torch
from bcep.provider import BCEP
from bcep.contracts import digest,canonical_hash
from bcep_joint.common import write
from selector_control.algorithms import kmeans_centers
from selector_control.core import local_seed
from .data import FormalData
from .control import pause_requested

class Paused(Exception):pass

def export_bank(split,checkpoint,purpose,seed,output,device='cuda'):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);sha=digest(checkpoint)
    expected=dict(split_hash=canonical_hash(split),provider_sha256=sha,purpose=purpose,seed=seed)
    if output.exists():
        with np.load(output,allow_pickle=False) as z:m=json.loads(str(z['metadata_json']))
        assert all(m[k]==v for k,v in expected.items())
        return dict(path=str(output),sha256=digest(output),**expected)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False);assert state['split_hash']==canonical_hash(split)
    model=BCEP();model.load_state_dict(state['model'],strict=True);model.to(device).eval();model.requires_grad_(False)
    data=FormalData(split,'export');ids=split[purpose+'_indices'];members=[split['members'][i] for i in ids];nmax=max(len(m['agent_ids']) for m in members)
    partial=output.with_suffix('.partial');partial.mkdir(exist_ok=True);identity=partial/'IDENTITY.json'
    if identity.exists():assert json.loads(identity.read_text())==expected
    else:write(identity,expected)
    for j,i in enumerate(ids):
        row=partial/f'{j:06d}.npy'
        if row.exists():continue
        if pause_requested():raise Paused('candidate export paused at window boundary')
        x=data.item(i,purpose,False);obs=x['observed'].to(device);maps=x['semantic_maps'].to(device);valid=torch.ones(len(obs),dtype=torch.bool,device=device)
        with torch.no_grad():atoms,_=model.sample_support(obs,maps,valid,x['scene_id'],x['agent_ids'],seed,256)
        atoms=atoms.cpu().numpy();goals=np.stack([kmeans_centers(atoms[:,ai],20,local_seed('bcep_flow',x['scene_id'],agent,seed),n_init=5,max_iter=100,tolerance_m=1e-6)['centers'].astype(np.float32) for ai,agent in enumerate(x['agent_ids'])],axis=1)
        tmp=row.with_suffix('.tmp.npy');np.save(tmp,goals);tmp.replace(row)
        if j%50==0:write(output.parent/'PROGRESS.json',dict(purpose=purpose,seed=seed,window=j+1,total=len(ids)))
    arr=np.zeros((len(ids),20,nmax,2),np.float32);mask=np.zeros((len(ids),nmax),bool)
    for j,m in enumerate(members):
        n=len(m['agent_ids']);v=np.load(partial/f'{j:06d}.npy');assert v.shape==(20,n,2) and np.isfinite(v).all();arr[j,:,:n]=v;mask[j,:n]=True
    meta=dict(schema='bcep-joint-candidates-v1',**expected,members=members,member_hash=canonical_hash(members),unit='m',K=20,M=256,provider_kind='bcep_flow',contains_truth=False,selector=dict(n_init=5,max_iter=100,tolerance_m=1e-6,ordering='original kmeans center order'),dtype='float32')
    tmp=output.with_suffix('.tmp.npz');np.savez_compressed(tmp,candidates=arr,valid=mask,metadata_json=json.dumps(meta));tmp.replace(output)
    receipt=dict(path=str(output),sha256=digest(output),**expected,windows=len(ids),agents=int(mask.sum()));write(output.with_suffix('.receipt.json'),receipt);return receipt

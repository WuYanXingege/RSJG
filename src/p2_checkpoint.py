"""P2-only checkpoint provenance; legacy layouts untouched."""
import hashlib
from pathlib import Path
import torch
from src.p2_protocol import base_metadata,digest,data_binding,path
from src.jdv2_objective_state import atomic_save
from src.joint_dependency_v2_cache import atomic_json_save,sha256_file

def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(str((tuple(v.shape),v.dtype)).encode())
        h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def initialize_ledger(owner):
    owner.p2_initial_hash=state_hash(owner.net.state_dict())
    owner.p2_exposure=[]
def record_exposure(owner,batch_id):
    owner.p2_exposure.extend(batch_id['p2_window_ids'])
def save_base(owner,payload,target,epoch):
    metadata=base_metadata(owner.args,owner.p2_registry,epoch,owner.p2_initial_hash,
        owner.p2_exposure,owner.net.best_valid_metric())
    payload=dict(payload,p2_provenance=metadata,resumable=False)
    atomic_save(payload,target)
    metadata=dict(metadata,path=str(target),sha256=sha256_file(target),
        classification='PROVENANCE_REVIEW_REQUIRED',best_metrics=payload.get('best_metrics'),
        optimizer_steps_scope='epoch payload; no real CUDA resume certification')
    atomic_json_save(metadata,str(target)+'.provenance.json')
def load_shared_initial(owner):
    ref=owner.p2_registry.manifest['shared_initial_state']
    payload=torch.load(path(ref['path']),map_location='cpu',weights_only=True)
    if payload.get('artifact_role')!='p2_shared_initial' or payload.get('base_sha256')!=ref['base_sha256'] or payload.get('data_binding')!=data_binding(owner.p2_registry):
        raise RuntimeError('P2 initial embedded identity')
    if payload.get('initialization_policy')!='fresh_heads_zero_corrector' or payload.get('optimizer_updates')!=0:
        raise RuntimeError('P2 shared state must be authenticated untrained initialization')
    owner.net.load_state_dict(payload['model_state_dict'],strict=True)
    # No optimizer/scheduler/RNG restored. New objective owner remains fresh.
    if state_hash(owner.net.state_dict())!=payload.get('state_sha256'):
        raise RuntimeError('P2 initial loaded state hash')

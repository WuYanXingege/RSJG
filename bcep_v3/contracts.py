"""Fail closed at the candidate/data boundary; never infer units from magnitude."""
import torch


def validate_batch(batch, goals):
    meta = batch.metadata
    if meta.get('unit') != 'm' or meta.get('dt_seconds') != 0.4:
        raise ValueError('explicit world-metre / dt_seconds=0.4 contract required')
    if meta.get('candidate_frame') != 'world_xy' or meta.get('frame_mapping') != 'observed8_future12':
        raise ValueError('candidate/frame mapping mismatch')
    b, n = batch.valid.shape
    if goals.shape != (b, 20, n, 2):
        raise ValueError('candidate layout mismatch')
    for ids, frames, ts, mask in zip(batch.agent_ids, batch.frame_ids, batch.timestamps, batch.valid):
        if len(ids) != int(mask.sum()) or len(set(ids)) != len(ids):
            raise ValueError('candidate/member identity mismatch')
        if len(frames) != 20 or len(set(frames)) != 20 or frames != sorted(frames):
            raise ValueError('frame mapping mismatch')
        if len(ts) != 20 or any(abs(ts[i+1]-ts[i]-0.4)>1e-6 for i in range(19)):
            raise ValueError('timestamp mapping mismatch')
    if not torch.isfinite(goals.transpose(1,2)[batch.valid]).all():
        raise ValueError('nonfinite valid candidate')

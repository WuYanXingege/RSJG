"""Synthetic-only data for the real grouped-backward callbacks."""
import torch
import yaml
from pathlib import Path
from ebjd.data import collate_scenes
from bcep_joint.checks import synthetic_batch
from .contracts import validate_batch


def joint_config():
    c = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/bcep_ebjd_v2/RESOLVED_CONFIG.yaml').read_text())
    return dict(c['joint'], scales=c['folds']['zara2']['bridge_scales'], pair_objective='final_coordinate')


class SyntheticData:
    TEST_ONLY = True
    def __init__(self, counts=(2, 3), pixels=32):
        self.items = []
        for i,n in enumerate(counts):
            x = synthetic_batch(n,pixels)
            self.items.append(dict(observed=x.observed[0]+i*0.05, future=x.future[0]+i*0.05,
                semantic_maps=x.semantic_maps[0], scene_id=f'SYNTHETIC_{i}', agent_ids=x.agent_ids[0],
                frame_ids=x.frame_ids[0], timestamps=x.timestamps[0], source_sequence='SYNTHETIC',
                metadata=dict(unit='m',dt_seconds=0.4,candidate_frame='world_xy',frame_mapping='observed8_future12')))
        self.split=dict(train_indices=list(range(len(counts))),selection_dev_indices=list(range(len(counts))),
                        members=[dict(agent_ids=x['agent_ids']) for x in self.items])
    def batch(self, ids, purpose):
        if purpose not in ('train','selection_dev'):raise ValueError('no test fixture')
        return collate_scenes([self.items[i] for i in ids])


class SyntheticBank:
    TEST_ONLY = True
    def batch(self, ids, batch):
        # Fixed random offsets, not targets. No Provider computation in Stage2.
        gen=torch.Generator().manual_seed(2035)
        offset=torch.randn(len(ids),20,batch.valid.shape[1],2,generator=gen)*1.4
        goals=batch.observed[:,None,:,-1]+offset
        validate_batch(batch,goals)
        return goals

"""Read exactly the original train/dev split and seed-2035 refined banks."""
from pathlib import Path
import subprocess
import torch
from bcep.contracts import canonical_hash, digest
from bcep_joint.common import read, seed_all
from bcep_formal.data import FormalData
from bcep_v2.bank import Bank
from bcep_v2.joint import build_joint
from bcep_v2.provider import Provider

SOURCE = Path(__file__).resolve().parents[1]
ROOT = SOURCE.parent
FORMAL = ROOT/'RSJG_EBJD_assets/experiments/bcep_ebjd_v2_integrated'
RUN = ROOT/'RSJG_EBJD_assets/experiments/zara2_scene_pair_selector_20261122'
BASE = '468c5eddde48bdeec47c4bb3c39418352be16bb9'
CHECKPOINT = FORMAL/'zara2/joint/epoch_005.pt'
CHECKPOINT_SHA = '9281673fda5c439bdd463b0bfc9af1a65a3a0fea30941ad3c956cbb617943317'
BANK_HASH = {'train': '2883ba7a847c6dcf34e08159529d1b10803e9ff5626cbffbd9c8bd66aa8d605a',
             'selection_dev': 'b6aee909ed8bcdac6bb08b93fba554303829f94707cf54e87eccd399a07c1560'}
LIMITS = [0.4097670185111337, 0.7841791655426308, 1.2968434301886971]
RECIPE = dict(seed=3201, lr=1e-4, weight_decay=1e-4, clip=5., logical_scenes=16,
              max_epochs=100, minimum=30, patience=15, scheduler_horizon=100)


class TrainDevData(FormalData):
    def item(self, i, purpose, truth=True):
        if purpose not in ('train', 'selection_dev'):
            raise ValueError('test inaccessible in this experiment')
        return super().item(i, purpose, truth)


class Inputs:
    def __init__(self):
        run = read(FORMAL/'FORMAL_RUN.json')
        assert run['commit'] == BASE
        plan = read(run['input_plan'])
        descriptor = plan['folds']['zara2']['split']
        assert digest(descriptor['path']) == descriptor['sha256'] == 'b31703d83fce7aadc735897efaa5c3970c04f477c70afb05879a05669657a816'
        self.split = read(descriptor['path'])
        assert canonical_hash(self.split) == '716e44710ff74faeffae1a6d7d814687443621e44623a0a9a1233cf8bf4f9bbb'
        assert (len(self.split['train_indices']), len(self.split['selection_dev_indices'])) == (2681, 590)
        assert sum(len(self.split['members'][i]['agent_ids']) >= 9 for i in self.split['selection_dev_indices']) == 127
        for i in self.split['train_indices']+self.split['selection_dev_indices']:
            row = self.split['records'][i]
            assert digest(row['path']) == row['sha256']
        self.data = TrainDevData(self.split, 'training', verify=False)
        self.banks = {}
        for purpose in BANK_HASH:
            folder = FORMAL/'zara2/refined_banks'/f'{purpose}_2035'
            identity = read(folder/'IDENTITY.json')
            assert (identity['seed'], identity['K'], identity['M'], identity['test_only']) == (2035, 20, 256, False)
            assert identity['base'] == read(FORMAL/'zara2/density/FROZEN.json')
            assert identity['head'] == read(FORMAL/'zara2/head/FROZEN.json')
            self.banks[purpose] = Bank(folder, identity, self.split)
            assert self.banks[purpose].hash == BANK_HASH[purpose]
        self.config = dict(read(FORMAL/'zara2/CONFIG.json')['joint'], scales=self.split['bridge_scales'])
        assert digest(CHECKPOINT) == CHECKPOINT_SHA
        self.identity = dict(base=BASE, checkpoint_sha256=CHECKPOINT_SHA,
                             split_sha256=canonical_hash(self.split), banks=BANK_HASH)

    def load_frozen(self, device):
        seed_all(3201)
        model = build_joint(self.config)
        state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
        assert state['completed_epoch'] == 5
        model.load_state_dict(state['model'], strict=True)
        model.eval().requires_grad_(False).to(device)
        assert len(model.bridge.denoiser.blocks) == 6
        # Provider is frozen too; its already-refined banks are used without regeneration.
        provider = Provider()
        self.provider_identity = []
        for stage, module, epoch in [('density', provider.base, 19), ('head', provider.head, 9)]:
            row = read(FORMAL/f'zara2/{stage}/TOP5.json')[0]
            path = FORMAL/f'zara2/{stage}'/row['file']
            assert row['epoch'] == epoch and digest(path) == row['sha256']
            saved = torch.load(path, map_location='cpu', weights_only=False)
            module.load_state_dict(saved['model'], strict=True)
            self.provider_identity.append(dict(stage=stage, epoch=epoch, sha256=row['sha256']))
        provider.eval().requires_grad_(False)
        assert not any(p.requires_grad for p in list(model.parameters())+list(provider.parameters()))
        assert not any(m.training for m in list(model.modules())+list(provider.modules()))
        return model, provider

    def batch(self, purpose, index, device, truth=True):
        batch = self.data.batch([index], purpose, truth=truth)
        goals = self.banks[purpose].batch([index], batch).to(device)
        return batch.to(device), goals


def source_hashes():
    paths = [p for folder in ('scene_pair_selector', 'bcep_v2', 'bcep_joint', 'bcep_formal', 'bcep', 'ebjd', 'selector_control', 'bcep_confirmation')
             for p in sorted((SOURCE/folder).rglob('*.py'))]
    return {str(p.relative_to(SOURCE)): digest(p) for p in paths}

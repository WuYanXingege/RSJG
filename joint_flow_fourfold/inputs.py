"""Read immutable fold assets; expose only train/development observations and labels."""
from copy import deepcopy
import hashlib
from pathlib import Path

import torch
import yaml

from bcep.contracts import canonical_hash, digest
from bcep_formal.data import FormalData
from bcep_joint.common import read
from bcep_v2.bank import Bank
from bcep_v2.joint import build_joint
from bcep_v2.provider import Provider

SOURCE = Path(__file__).resolve().parents[1]


def state_hash(module):
    return canonical_hash({name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                           for name, value in module.state_dict().items()})


class TrainDevData(FormalData):
    def item(self, index, purpose, truth=True):
        if purpose not in ('train', 'selection_dev'):
            raise ValueError('dependency fitting may only access train and selection_dev')
        return super().item(index, purpose, truth=truth)


class Inputs:
    """Manifest-driven CPU checks; load_frozen is the explicit device-load step."""
    def __init__(self, frozen_manifest, output, device):
        self.manifest = deepcopy(read(frozen_manifest) if isinstance(frozen_manifest, (str, Path)) else frozen_manifest)
        self.output = Path(output)
        self.device = torch.device(device)
        manifest = self.manifest
        self.fold = manifest['fold']
        paths = manifest.get('paths', {})
        self.split_path = Path(manifest.get('split_path') or paths['split'])
        config_path = manifest.get('config_path') or paths.get('config')
        if config_path is None:
            raise ValueError('frozen manifest must specify config_path')
        self.config_path = Path(config_path)
        self.split = read(self.split_path)
        assert self.split['fold'] == self.fold
        if manifest.get('split_sha256'):
            assert digest(self.split_path) == manifest['split_sha256']
        if manifest.get('config_sha256'):
            assert digest(self.config_path) == manifest['config_sha256']
        self.full_config = yaml.safe_load(self.config_path.read_text())
        self.config = dict(self.full_config['joint'], scales=self.split['bridge_scales'])
        self.geometry = deepcopy(manifest['geometry'])
        self.data = TrainDevData(self.split, phase='training', verify=False)
        self.counts = {purpose: len(self.split[purpose+'_indices']) for purpose in ('train', 'selection_dev')}
        assert all(self.counts.values())
        for purpose in ('train', 'selection_dev'):
            for index in self.split[purpose+'_indices']:
                record = self.split['records'][index]
                assert digest(record['path']) == record['sha256'], 'train/dev record changed'
        self.stages = {}
        self.provider_identity = []
        for stage in ('density', 'head', 'joint'):
            asset = deepcopy(manifest[stage])
            checkpoint = Path(asset.get('checkpoint') or paths[stage])
            assert digest(checkpoint) == asset['sha256'], stage+' checkpoint changed'
            frozen = asset['frozen']
            assert frozen['stage'] == stage and frozen['trained'] and frozen['frozen']
            assert frozen['weights_sha256'] == asset['sha256']
            asset['checkpoint'] = str(checkpoint)
            self.stages[stage] = asset
            if stage != 'joint':
                self.provider_identity.append(dict(stage=stage, epoch=frozen['epoch'], sha256=asset['sha256']))
        assert self.stages['head']['frozen']['base'] == self.stages['density']['frozen']
        self.banks = {}
        for purpose, alias in (('train', 'train_bank'), ('selection_dev', 'dev_bank')):
            asset = manifest['banks'][purpose]
            folder = Path(asset.get('path') or paths[alias])
            identity = asset['identity']
            assert identity['fold'] == self.fold and identity['purpose'] == purpose
            assert (identity['seed'], identity['M'], identity['K']) == (2035, 256, 20)
            assert identity['base'] == self.stages['density']['frozen']
            assert identity['head'] == self.stages['head']['frozen']
            assert identity['geometry'] == canonical_hash(self.geometry)
            bank = Bank(folder, identity, self.split)
            assert bank.hash == asset['sha256']
            self.banks[purpose] = bank
        self.identity = dict(fold=self.fold, frozen_manifest_sha256=canonical_hash(manifest),
            upstream_identity=deepcopy(manifest['identity']), split_sha256=canonical_hash(self.split),
            split_file_sha256=digest(self.split_path), config_file_sha256=digest(self.config_path),
            checkpoint_sha256=self.stages['joint']['sha256'], provider=self.provider_identity,
            banks={purpose: bank.hash for purpose, bank in self.banks.items()})

    def load_frozen(self, device=None):
        device = self.device if device is None else torch.device(device)
        model = build_joint(self.config)
        provider = Provider()
        modules = dict(density=provider.base, head=provider.head, joint=model)
        for stage, module in modules.items():
            asset = self.stages[stage]
            assert digest(asset['checkpoint']) == asset['sha256']
            saved = torch.load(asset['checkpoint'], map_location='cpu', weights_only=False)
            assert saved['completed_epoch'] == asset['frozen']['epoch']
            assert canonical_hash(saved['identity']) == asset['frozen']['identity']
            module.load_state_dict(saved['model'], strict=True)
        for module in (model, provider):
            module.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
            assert not any(part.training for part in module.modules())
            assert not any(parameter.requires_grad for parameter in module.parameters())
        assert len(model.bridge.denoiser.blocks) == 6
        return model, provider

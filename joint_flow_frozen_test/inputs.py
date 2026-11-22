"""Read-only adapters for one frozen ZARA2 A/B heldout-confirmation export.

Construction hashes records and checkpoints but never materializes heldout
future arrays. Missing formal banks are reported, never generated or repaired.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from bcep.contracts import canonical_hash, digest
from bcep_formal.data import FormalData
from bcep_joint.common import read
from bcep_v2.bank import Bank
from bcep_v2.provider import GEOMETRY
from joint_flow_dependency.model import EdgeDependence
from scene_pair_selector.inputs import (
    Inputs as BaseInputs, FORMAL, BASE, CHECKPOINT, CHECKPOINT_SHA,
)

SOURCE = Path(__file__).resolve().parents[1]
ROOT = SOURCE.parent
RUN = ROOT/'RSJG_EBJD_assets/experiments/zara2_joint_flow_frozen_test_20261122'
DOC = SOURCE/'docs/research/continuous_joint_flow/zara2_frozen_test_20261122'
BASE_COMMIT = 'd4fbe6d8e3b142645b9e801116aac308fcb19c9d'
SEEDS = tuple(range(2035, 2040))
PURPOSE = 'heldout_confirmation'
B_CHECKPOINT = ROOT/'RSJG_EBJD_assets/experiments/zara2_joint_flow_dependency_20261122/dependence/epoch_035.pt'
B_SHA = 'ea13cd7fd99a02c44c3c9af9a14a0cc9b5166b629e5c338694cca97f1c8c502a'
ARCHIVE_DIR = 'docs/research/continuous_joint_flow/zara2_20261122/'


def archive_json(name):
    value = subprocess.check_output(['git', 'show', BASE_COMMIT+':'+ARCHIVE_DIR+name], cwd=SOURCE)
    return json.loads(value)


class ObservedOnlyHeldoutData(FormalData):
    """The exporter cannot accidentally request truth or a different purpose."""
    def __init__(self, split):
        super().__init__(split, phase='export', verify=False)

    def item(self, index, purpose, truth=False):
        if purpose != PURPOSE or truth:
            raise ValueError('frozen test exporter accepts heldout observations only')
        return super().item(index, purpose, truth=False)

    def batch(self, indices, purpose, truth=False):
        return super().batch(indices, purpose, truth=truth)


class Inputs:
    def __init__(self):
        self._base = BaseInputs()
        self.split = self._base.split
        assert self.split['fold'] == 'zara2'
        assert len(self.split[PURPOSE+'_indices']) == 998
        assert len(self.split['selection_dev_indices']) == 590
        self.dev_data = self._base.data
        self.export_data = ObservedOnlyHeldoutData(self.split)
        self.dev_bank2035 = self._base.banks['selection_dev']
        self.config = self._base.config
        self.base_identity = deepcopy(self._base.identity)
        # Hashing compressed record bytes verifies integrity without loading
        # any future array; the only test reader exposed is phase='export'.
        for index in self.split[PURPOSE+'_indices']:
            row = self.split['records'][index]
            assert digest(row['path']) == row['sha256'], 'heldout record changed'
        assert digest(CHECKPOINT) == CHECKPOINT_SHA
        assert digest(B_CHECKPOINT) == B_SHA
        self.archive_config = archive_json('RUN_CONFIG.json')
        self.archive_training = archive_json('TRAINING_RESULT.json')
        self.archive_metrics = archive_json('FINAL_METRICS.json')
        self.archive_decision = archive_json('DECISION.json')
        frozen = archive_json('FROZEN_MODELS.json')
        assert self.archive_config['identity'] == self.base_identity
        assert self.archive_training['rank1_epoch'] == self.archive_decision['rank1_epoch'] == 35
        assert self.archive_training['rank1_sha256'] == B_SHA
        assert self.archive_training['recipe']['seed'] == 3201
        assert self.archive_decision['all_pass'] is True
        assert self.archive_decision['test_access'] is False
        for group in ('all', 'N_ge_9'):
            assert self.archive_training['rank1_metrics'][group] == self.archive_metrics['B'][group]
        assert frozen['exact_unchanged'] and frozen['before'] == frozen['after']
        self.provider_identity = deepcopy(frozen['provider'])
        assert [(r['stage'], r['epoch']) for r in self.provider_identity] == [('density', 19), ('head', 9)]
        self.provider_frozen = {}
        for expected in self.provider_identity:
            stage = expected['stage']
            frozen_stage = read(FORMAL/f'zara2/{stage}/FROZEN.json')
            rank = read(FORMAL/f'zara2/{stage}/TOP5.json')[0]
            assert frozen_stage['trained'] and frozen_stage['frozen']
            assert frozen_stage['epoch'] == rank['epoch'] == expected['epoch']
            assert frozen_stage['weights_sha256'] == rank['sha256'] == expected['sha256']
            assert digest(FORMAL/f'zara2/{stage}'/rank['file']) == expected['sha256']
            self.provider_frozen[stage] = frozen_stage
        assert self.provider_frozen['head']['base'] == self.provider_frozen['density']
        # Verify the archived definition; the frozen dependence itself must
        # remain exact. Inference/sampling may expose the authorized seed
        # argument, whose 2035 equivalence is checked by full dev replay.
        for relative in ('joint_flow_dependency/model.py', 'joint_flow_dependency/inference.py',
                         'joint_flow_dependency/sampling.py'):
            original = subprocess.check_output(['git', 'show', BASE_COMMIT+':'+relative], cwd=SOURCE)
            assert hashlib.sha256(original).hexdigest() == self.archive_config['source_sha256'][relative]
        assert digest(SOURCE/'joint_flow_dependency/model.py') == self.archive_config['source_sha256']['joint_flow_dependency/model.py']
        self.bank_template = read(self.dev_bank2035.folder/'IDENTITY.json')
        full_config = read(FORMAL/'zara2/CONFIG.json')
        self.geometry = dict(layout=GEOMETRY,
            split=full_config['folds']['zara2']['split_sha256'],
            crop_sources=full_config['geometry_source'],
            records=canonical_hash([row['sha256'] for row in self.split['records']]))
        assert canonical_hash(self.geometry) == self.bank_template['geometry']
        assert self.bank_template['base'] == self.provider_frozen['density']
        assert self.bank_template['head'] == self.provider_frozen['head']
        self.identity = dict(self.base_identity, frozen_dev_commit=BASE_COMMIT,
            B_checkpoint_sha256=B_SHA, B_completed_epoch=35, provider=self.provider_identity,
            purpose=PURPOSE, heldout_windows=998, seeds=list(SEEDS),
            heldout_member_hash=canonical_hash([self.split['members'][i] for i in self.split[PURPOSE+'_indices']]))
        self.bank_identities = {seed: self.expected_test_bank_identity(seed) for seed in SEEDS}
        self.test_banks = {}
        self.test_bank_reports = {}
        for seed in SEEDS:
            folder = FORMAL/'zara2/refined_banks'/f'{PURPOSE}_{seed}'
            if not (folder/'COMPLETE.json').exists():
                self.test_bank_reports[seed] = dict(seed=seed, path=str(folder),
                    status='missing' if not folder.exists() else 'incomplete',
                    complete=False, generated=False)
                continue
            self.attach_test_bank(seed, folder)
        self.missing_test_bank_seeds = tuple(seed for seed in SEEDS if seed not in self.test_banks)

    def expected_test_bank_identity(self, seed):
        """Original Provider bank contract, adapted only to heldout members/seed."""
        if seed not in SEEDS:
            raise ValueError('only predeclared seeds 2035 through 2039 are allowed')
        identity = deepcopy(self.bank_template)
        identity.update(purpose=PURPOSE, seed=int(seed), member_hash=self.identity['heldout_member_hash'])
        assert identity['fold'] == 'zara2' and identity['split'] == canonical_hash(self.split)
        assert (identity['M'], identity['K'], identity['dtype'], identity['unit']) == (256, 20, 'float32', 'm')
        assert identity['no_future_as_input'] and identity['trained'] and not identity['test_only']
        return identity

    def attach_test_bank(self, seed, folder):
        """Validate a complete existing/new-run bank; this method never writes it."""
        folder = Path(folder)
        identity = self.expected_test_bank_identity(seed)
        assert read(folder/'IDENTITY.json') == identity, 'test bank differs from frozen native contract'
        bank = Bank(folder, identity, self.split)
        assert len(bank.rows) == 998
        self.test_banks[seed] = bank
        self.test_bank_reports[seed] = dict(seed=seed, path=str(folder), status='verified_complete',
            complete=True, bank_sha256=bank.hash, identity_sha256=canonical_hash(identity),
            complete_manifest_sha256=digest(folder/'COMPLETE.json'), windows=len(bank.rows),
            row_identity_and_file_hashes_verified=True, future_loaded=False)
        self.missing_test_bank_seeds = tuple(s for s in SEEDS if s not in self.test_banks)
        return bank

    def load_frozen(self, device):
        """Strictly load A coordinate epoch5, frozen Provider, and B dev rank1."""
        assert digest(CHECKPOINT) == CHECKPOINT_SHA and digest(B_CHECKPOINT) == B_SHA
        model, provider = self._base.load_frozen(device)
        assert self._base.provider_identity == self.provider_identity
        state = torch.load(B_CHECKPOINT, map_location='cpu', weights_only=False)
        assert state['completed_epoch'] == 35
        assert {key: state['identity'][key] for key in self.base_identity} == self.base_identity
        assert state['identity']['config'] == canonical_hash(self.archive_config)
        assert state['identity']['fold'] == 'zara2' and state['identity']['experiment'] == 'continuous_joint_flow'
        assert state['recipe'] == self.archive_training['recipe']
        dependence = EdgeDependence()
        dependence.load_state_dict(state['model'], strict=True)
        for module in (model, provider, dependence):
            module.to(device).eval().requires_grad_(False)
            assert not any(part.training for part in module.modules())
            assert not any(parameter.requires_grad for parameter in module.parameters())
        return model, provider, dependence

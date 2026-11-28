"""Read identity-checked S0 assets and recover only missing baseline paths."""

from dataclasses import dataclass
from pathlib import Path
import csv
import time

import numpy as np
import torch

from bcep.contracts import canonical_hash, require
from bcep_formal.data import FormalData
from joint_flow_frozen_test.run import forbid_heldout_truth
from .common import ASSETS, ATOL, DOC as ORIGINAL_DOC, RTOL, RUN as ORIGINAL_RUN, SEEDS, SOURCE
from .common import digest, immutable, read, window_key
from .evaluate import _read_prediction
from .group_slot_recombination.data import DevelopmentCache, dev_reference
from .interaction_diagnosis.assets import read_arrays, save_npz
from .interaction_run import _fold_identity, _load
from .model import decode_student, student_noise
from .train import _window_goals


BASE = 'b03cd0271fa7453fdf794c0cd1422c86bf85e363'
RUN = ASSETS/'fixed_k20_cv_20261128'
DOC = ORIGINAL_DOC.parent/'fixed_k20_cv_20261128'
FOLDS = ('hotel','univ')
DEV = 'selection_dev'
TEST = 'heldout_confirmation'
S0_NATIVE_SOURCE = SOURCE.parent/'RSJG_direct_k20_20261126'


def fixed_inputs():
    table = ORIGINAL_DOC/'SELECTION.csv'
    with table.open(newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['fold'] in FOLDS and r['selected']=='True']
    require(len(rows)==len(FOLDS) and {r['fold'] for r in rows}==set(FOLDS),
            'missing unique S0 in original five-fold selection')
    selections = {row['fold']:_fold_identity(row) for row in rows}
    refs = {fold:dev_reference(fold,RUN) for fold in FOLDS}
    dev = {}
    for fold,ref in refs.items():
        require(all(selections[fold][key]==ref.identity[key] for key in
                    ('S0_sha256','S0_epoch','split_sha256','dev_manifest_sha256')),
                'the original development trajectory identity differs from selected S0')
        dev[fold] = DevelopmentCache(ref,RUN)
        selections[fold].update(dev_cache=dev[fold].identity,
            dev_max_people=max(len(ref.split['members'][index]['agent_ids']) for index in ref.indices))
    return selections,refs,dev


@dataclass
class FrozenTestBaseline:
    fold: str
    purpose: str
    split: dict
    indices: list
    seeds: list
    rows: dict
    identity: dict
    source_hashes: dict
    folder: Path
    recovery_root: Path = RUN

    def _source(self,index,seed):
        entry = self.rows[index,seed]
        return self.folder/entry['file'],entry

    def _repair_path(self,index,seed):
        _,entry = self._source(index,seed)
        return self.recovery_root/'recovered_test_S0'/self.fold/entry['file']

    def missing_pairs(self):
        absent = []
        for (index,seed),entry in self.rows.items():
            original = self.folder/entry['file']
            if not original.is_file() or digest(original)!=entry['sha256']:
                absent.append((index,seed))
        return absent

    def read_path(self,index,seed):
        original,entry = self._source(index,seed)
        if original.is_file() and digest(original)==entry['sha256']:
            return _read_prediction(original,entry['n'])
        path = self._repair_path(index,seed)
        side = read(path.with_suffix('.json'))
        require(side['sha256']==digest(path) and side['source_sha256']==entry['sha256']
                and side['member_sha256']==entry['member_sha256'] and side['seed']==seed
                and side['S0_sha256']==self.identity['S0_sha256'] and side['ddim_calls']==1,
                'recovered baseline trajectory differs from the authorized scope')
        return read_arrays(path,{'paths':(20,entry['n'],12,2)})['paths']

    @torch.no_grad()
    def repair(self,absent):
        pending = [(i,s) for i,s in absent if not self._repair_path(i,s).with_suffix('.json').is_file()]
        if pending:
            assets,student = _load(self.fold,self.identity,torch.device('cuda'))
            reader = FormalData(self.split,phase='export',verify=False)
            with forbid_heldout_truth():
                for index,seed in pending:
                    started = time.monotonic()
                    _,entry = self._source(index,seed)
                    record = self.split['records'][index]
                    require(digest(record['path'])==record['sha256'],'heldout record changed')
                    batch = reader.batch([index],TEST,truth=False).to(assets.device)
                    require(not bool(batch.future.count_nonzero()) and bool(batch.valid.all()),
                            'native replay requires observed-only packed input')
                    goals,key = _window_goals(assets,student,index,batch,input_grad=False)
                    require(key==entry['member_sha256'],'repair would change the native noise key')
                    bridge = assets.model.encode_context(batch.observed,batch.semantic_maps,batch.valid)
                    noise = student_noise(key,batch.agent_ids[0],seed,assets.device)
                    paths = decode_student(assets.model,bridge,goals,noise)[0].cpu().numpy()
                    path = self._repair_path(index,seed)
                    save_npz(path,paths=paths)
                    immutable(path.with_suffix('.json'),dict(
                        sha256=digest(path), source_sha256=entry['sha256'], member_sha256=key, seed=seed,
                        S0_sha256=self.identity['S0_sha256'], ddim_calls=1, seconds=time.monotonic()-started))
        times = []
        for index,seed in absent:
            self.read_path(index,seed)
            times.append(read(self._repair_path(index,seed).with_suffix('.json'))['seconds'])
        return dict(new_DDIM_calls=len(absent),seconds=sum(times),windows=len({i for i,_ in absent}))


def open_test_baseline(fold,dev_ref,identity,freeze):
    """Open only the already-qualified fold's old five-seed seal."""
    path = RUN/'FROZEN_TEST.json'
    require(freeze==read(path) and fold in freeze['folds'] and freeze['folds'][fold]==identity
            and identity['phase']=='test','legacy test trajectories are gated by a frozen passing fold')
    phase = identity['dev_score_phase']
    require(phase in ('cv_development','center_development'),'invalid development gate')
    dev_score = RUN/phase/'SCORE_RECEIPT.json'
    require(digest(dev_score)==identity['dev_score_sha256']
            and read(dev_score)['decision'][fold]['passed'],
            'development gate receipt changed or did not pass')
    return _open_archived_test_baseline(fold,dev_ref,identity,RUN)


def open_exploratory_test_baseline(dev_ref,identity,run):
    """Replay only the archived UNIV S0 under its explicit follow-up freeze."""
    from .fixed_k20_eval import EXPLORATORY_SCOPE, _test_freeze

    require(dev_ref.fold=='univ' and identity['phase']=='test'
            and identity['test_scope']==EXPLORATORY_SCOPE,
            'invalid baseline reuse for the exploratory test')
    require(read(run/'FROZEN_TEST.json')['test_scope']==EXPLORATORY_SCOPE,
            'exploratory test freeze missing')
    _test_freeze(run,dev_ref.fold,identity)
    return _open_archived_test_baseline(dev_ref.fold,dev_ref,identity,run)


def _open_archived_test_baseline(fold,dev_ref,identity,recovery_root):
    original_freeze = ORIGINAL_RUN/'FROZEN_STUDENTS.json'
    original_config = ORIGINAL_RUN/'RUN_CONFIG.json'
    original_seal = ORIGINAL_RUN/'ALL_PREDICTIONS_SEALED.json'
    original_score = ORIGINAL_RUN/'SCORE_RECEIPT.json'
    for p in (original_config,original_seal,original_score,ORIGINAL_RUN/'PER_SEED.csv',
              ORIGINAL_RUN/'PER_WINDOW.csv'):
        require(digest(p)==digest(ORIGINAL_DOC/p.name),'original test archive changed')
    frozen = read(original_freeze)
    selected = frozen['folds'][fold]
    require(selected['selected_sha256']==identity['S0_sha256']
            and selected['selected_epoch']==identity['S0_epoch']
            and frozen['config_sha256']==digest(original_config)
            and frozen['seeds']==list(SEEDS),'original test S0 identity differs')
    # Authenticate the code that generated the old cache on its original
    # worktree. The new worktree descends from the group-query implementation.
    source = read(original_config)['source_sha256']
    require(all(digest(S0_NATIVE_SOURCE/name)==value for name,value in source.items()),
            'historical S0 inference sources changed')
    scoring = read(original_score)
    entire = read(original_seal)
    require(scoring['global_prediction_seal_sha256']==digest(original_seal)
            and entire['freeze_barrier_sha256']==digest(original_freeze)
            and scoring['selection_seal_unix']<=scoring['first_truth_read_unix'],
            'original frozen test provenance differs')
    split = dev_ref.split
    indices = split[TEST+'_indices']
    folder = ORIGINAL_RUN/fold/'test'
    seal_path = folder/'PREDICTIONS_FROZEN.json'
    require(digest(seal_path)==entire['folds'][fold]['sha256'],'original fold prediction manifest changed')
    seal = read(seal_path)
    expected = dict(fold=fold,selected_sha256=identity['S0_sha256'],selected_epoch=identity['S0_epoch'],
        ordered_members_sha256=canonical_hash([split['members'][i] for i in indices]),
        freeze_barrier_sha256=digest(original_freeze),seeds=list(SEEDS),K=20,
        ddim_steps=20,eta=0.,prediction_frames=12)
    require(seal['identity']==expected and seal['truth_reads']==0
            and seal['ddim_calls']==len(indices)*len(SEEDS)==len(seal['rows']),
            'original heldout K20 protocol differs')
    rows = {}
    for ordinal,index in enumerate(indices,1):
        for seed in SEEDS:
            offset = (ordinal-1)*len(SEEDS)+SEEDS.index(seed)
            row = seal['rows'][offset]
            require(row['index']==index and row['seed']==seed
                    and row['window']==f'test_{ordinal:04d}'
                    and row['file']==f'seed_{seed}/{ordinal:04d}.npz'
                    and row['n']==len(split['members'][index]['agent_ids'])
                    and row['member_sha256']==window_key(split,index)
                    and row['export_identity_sha256']==canonical_hash(expected),
                    'native cached seed, person order or window differs')
            rows[index,seed] = row
    return FrozenTestBaseline(fold,TEST,split,indices,list(SEEDS),rows,identity,dict(
        S0_test_seal_sha256=digest(seal_path),S0_test_score_sha256=digest(original_score),
        S0_global_seal_sha256=digest(original_seal)),folder,recovery_root)

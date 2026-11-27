"""Build train losses from frozen endpoints and read sealed development paths."""

from dataclasses import dataclass
from pathlib import Path
import argparse
import fcntl
import time

import numpy as np
import torch

from bcep.contracts import canonical_hash, require
from bcep_formal.data import FormalData
from bcep_v2.runtime import configure_cuda
from direct_k20.common import ASSETS, ATOL, RTOL, SOURCE, digest, immutable, read, status
from direct_k20.interaction_diagnosis.assets import read_arrays, reference_fold, save_npz
from direct_k20.interaction_diagnosis.statistics import greedy_observed_pairs
from direct_k20.interaction_run import _load
from direct_k20.model import decode_student, student_noise
from direct_k20.train import _window_goals
from joint_flow_fourfold.inputs import state_hash


FOLDS = ('hotel', 'univ')
DEV = 'selection_dev'
SEED = 2035
BASE = '2e712d5225de03cb1e346c2f04fae5085095386b'
RUN = ASSETS / 'group_slot_recombination_20261127'
DOC = SOURCE / 'docs/research/continuous_joint_flow/group_slot_recombination_20261127'
DIAG = ASSETS / 'endpoint_diagnosis_20261127'
DIAG_DOC = DOC.parent / 'endpoint_diagnosis_20261127'
EPOCHS = {'hotel': 24, 'univ': 16}


def dev_reference(fold, run=RUN):
    ref = reference_fold(fold, DEV, run)
    require(ref.purpose == DEV and ref.seeds == [SEED] and ref.identity['S0_epoch'] == EPOCHS[fold],
            'unexpected frozen dev identity')
    return ref


def queries(ref):
    ident = ref.identity
    require(digest(ident['S0_checkpoint']) == ident['S0_sha256'], 'S0 checkpoint hash mismatch')
    saved = torch.load(ident['S0_checkpoint'], map_location='cpu', weights_only=False)
    q = saved['student']['queries']
    require(saved['completed_epoch'] == ident['S0_epoch']
            and q.shape == (20, 32) and q.dtype == torch.float32 and bool(torch.isfinite(q).all()),
            'native S0 queries did not come from the selected checkpoint')
    return q.detach().clone()


def _group_index(observed):
    pairs = greedy_observed_pairs(observed.astype(np.float64))
    paired = set(pairs.ravel())
    groups = [list(pair) for pair in pairs] + [[person] for person in range(len(observed)) if person not in paired]
    ids = np.empty(len(observed), dtype=np.int64)
    for group, members in enumerate(sorted(groups, key=lambda members: members[0])):
        ids[members] = group
    return ids


def _sample(path, row):
    require(digest(path) == row['sha256'], 'frozen train cache shard hash changed')
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == {'observed','g0','groups','E'}, 'unexpected train cache fields')
        arrays = {name: saved[name] for name in saved.files}
    n, g = row['n'], row['group_count']
    require(arrays['observed'].shape == (n,8,2) and arrays['g0'].shape == (20,n,2)
            and arrays['E'].shape == (g,20) and arrays['groups'].shape == (n,)
            and arrays['groups'].dtype == np.int64 and np.array_equal(np.unique(arrays['groups']), np.arange(g))
            and all(arrays[key].dtype == np.float32 and np.isfinite(arrays[key]).all()
                    for key in ('observed','g0','E')), 'invalid training sample dimensions or dtype')
    return arrays


@dataclass
class TrainingCache:
    directory: Path
    indices: tuple
    rows: dict
    identity: dict

    def load(self, index):
        row = self.rows[index]
        return _sample(self.directory / row['file'], row)


def _train_identity(ref):
    split = ref.split
    indices = split['train_indices']
    return dict(fold=ref.fold, purpose='train', model='S0', S0_sha256=ref.identity['S0_sha256'],
        S0_epoch=ref.identity['S0_epoch'], split_sha256=ref.identity['split_sha256'],
        train_member_sha256=canonical_hash([split['members'][i] for i in indices]),
        K=20, frames=12, dt=.4, unit='m', observation_grouping='greedy mean of final three frame distances <= 2m',
        code_sha256=digest(__file__))


def open_training_cache(ref, run=RUN):
    folder = run / ref.fold / 'train_cache'
    path = folder / 'SEALED.json'
    frozen = read(path)
    require(frozen['identity'] == _train_identity(ref) and frozen['training_updates'] == frozen['DDIM_calls'] == 0,
            'wrong train endpoint cache')
    rows = {row['index']: row for row in frozen['rows']}
    require(tuple(rows) == tuple(ref.split['train_indices']) and len(rows) == len(frozen['rows']),
            'train cache membership or ordering changed')
    identity = frozen['identity'] | dict(cache_sha256=digest(path))
    return TrainingCache(folder, tuple(rows), rows, identity)


@torch.no_grad()
def prepare_training(ref, run, device):
    folder = run / ref.fold / 'train_cache'
    folder.mkdir(parents=True, exist_ok=True)
    seal_path = folder / 'SEALED.json'
    if seal_path.exists():
        open_training_cache(ref, run)
        return read(seal_path)
    identity = _train_identity(ref)
    immutable(folder / 'IDENTITY.json', identity)
    assets, student = _load(ref.fold, ref.identity, torch.device(device))
    observed_only = FormalData(ref.split, phase='export', verify=False)
    training = FormalData(ref.split, phase='training', verify=False)
    frozen = assets.frozen_states() | dict(student=state_hash(student))
    indices = ref.split['train_indices']
    identities = ref.split['members']
    source = ref.split['records']
    rows, started = [], time.monotonic()
    for ordinal, index in enumerate(indices, 1):
        n = len(identities[index]['agent_ids'])
        expected = dict(index=index, window=f'train_{ordinal:04d}', n=n, file=f'{ordinal:04d}.npz',
                        member_sha256=canonical_hash(identities[index]),
                        original_record_sha256=source[index]['sha256'], identity_sha256=canonical_hash(identity))
        path = folder / expected['file']
        side = path.with_suffix('.json')
        if side.exists():
            row = read(side)
            require(all(row[key] == value for key,value in expected.items()), 'train endpoint row changed')
        else:
            obs = observed_only.batch([index], 'train', truth=False)
            require(obs.valid.shape == (1,n) and bool(obs.valid.all()) and not bool(obs.future.count_nonzero()),
                    'frozen endpoint generation accepted a future or invalid mask')
            observed = obs.observed[0].numpy()
            g0, _ = _window_goals(assets, student, index, obs.to(assets.device), input_grad=False)
            g0 = g0.cpu()
            labels = training.batch([index], 'train', truth=True)
            require(labels.agent_ids == obs.agent_ids and torch.equal(labels.valid, obs.valid)
                    and torch.equal(labels.observed, obs.observed), 'train targets do not match native agent order')
            err = torch.linalg.vector_norm(g0-labels.future[0,None,:,-1], dim=-1).numpy()
            group = _group_index(observed)
            g = int(group.max())+1
            E = np.stack([err[:, group == i].sum(axis=1) for i in range(g)])
            save_npz(path, observed=observed, g0=g0.numpy(), groups=group, E=E)
            row = expected | dict(group_count=g, sha256=digest(path))
            immutable(side, row)
        _sample(path, row)
        rows.append(row)
        if ordinal % 250 == 0 or ordinal == len(indices):
            status(run, 'CACHING_FROZEN_TRAIN_ENDPOINTS', fold=ref.fold, windows=ordinal, total=len(indices),
                   elapsed_seconds=time.monotonic()-started)
    require(assets.frozen_states() | dict(student=state_hash(student)) == frozen, 'frozen train assets changed')
    receipt = dict(identity=identity, rows=rows, frozen_before=frozen, frozen_after=frozen,
                   windows=len(rows), agents=sum(x['n'] for x in rows), DDIM_calls=0,
                   training_updates=0, seconds=time.monotonic()-started)
    immutable(seal_path, receipt)
    return receipt


class DevelopmentCache:
    """Source S0 endpoints and full paths by the exact old packed member order."""

    def __init__(self, ref, run=RUN):
        require(ref.purpose == DEV and ref.seeds == [SEED], 'development cache is selection_dev only')
        self.ref = ref
        self.output = run / ref.fold / 'recovered_dev_S0'
        global_path = DIAG / 'ENDPOINTS_EXPORTED.json'
        prior = read(DIAG_DOC / 'RECEIPT.json')
        require(prior['endpoint_seal_sha256'] == digest(global_path), 'diagnosis endpoint seal changed')
        global_seal = read(global_path)
        source = global_seal['folds'][f'{DEV}/{ref.fold}']
        self.endpoint_folder = DIAG / 'endpoints' / DEV / ref.fold
        path = self.endpoint_folder / 'ENDPOINTS_SEALED.json'
        require(digest(path) == source['endpoint_seal_sha256'], 'locked dev endpoint fold changed')
        sealed = read(path)
        require(sealed['truth_reads'] == 0 and sealed['windows'] == len(ref.indices) == len(sealed['rows']),
                'frozen endpoint membership differs')
        self.row = {}
        for ordinal, (index, row) in enumerate(zip(ref.indices, sealed['rows']), 1):
            original = ref.rows[index,SEED]
            expected = {key: original[key] for key in ('window','member_sha256','n')}
            expected.update(file=f'{ordinal:04d}.npz', source_pair_seal_sha256=ref.source_hashes['pair_seal_sha256'])
            require(all(row[key] == value for key, value in expected.items()), 'native dev endpoint row changed')
            require(not original['S0_reused'], 'diagnostic dev S0 must have its own sealed path')
            self.row[index] = row
        self.identity = dict(dev_endpoint_seal_sha256=digest(path),
                             dev_path_seal_sha256=ref.source_hashes['pair_seal_sha256'],
                             archive_score_sha256=ref.source_hashes['score_receipt_sha256'])

    def read_endpoints(self, index):
        row = self.row[index]
        path = self.endpoint_folder / row['file']
        require(digest(path) == row['sha256'], 'frozen dev endpoint array changed')
        fields = ('raw','g0','g1','unary_world','social_world')
        return read_arrays(path, dict.fromkeys(fields,(20,row['n'],2)))

    def _repair_identity(self, index):
        row = self.ref.rows[index,SEED]
        payload = self.output / f'{row["window"]}.npz'
        expected = dict(window=row['window'], n=row['n'], seed=SEED, S0_sha256=self.ref.identity['S0_sha256'],
            original_pair_sha256=row['sha256'], member_sha256=row['member_sha256'])
        return payload, expected

    def _repair_path(self, index):
        payload, expected = self._repair_identity(index)
        record = read(payload.with_suffix('.json'))
        require(all(record[key] == value for key,value in expected.items())
                and record['sha256'] == digest(payload) and record['DDIM_calls'] == 1,
                'repaired S0 path lacks native identity')
        return read_arrays(payload,{'S0':(20,expected['n'],12,2)})['S0']

    def read_path(self, index):
        row = self.ref.rows[index,SEED]
        path = self.ref.folder / row['file']
        if path.is_file() and digest(path) == row['sha256']:
            shape = (20,row['n'],12,2)
            return read_arrays(path,{'S0':shape,'R12':shape})['S0']
        return self._repair_path(index)

    @torch.no_grad()
    def repair_if_needed(self, device):
        invalid = [index for index in self.ref.indices if self._source_invalid(index)]
        if not invalid:
            return dict(new_DDIM_calls=0, repaired_windows=0, seconds=0.)
        missing = [index for index in invalid if not self._repair_identity(index)[0].with_suffix('.json').exists()]
        if missing:
            assets, student = _load(self.ref.fold,self.ref.identity,torch.device(device))
            reader = FormalData(self.ref.split,phase='export',verify=False)
            for index in missing:
                started = time.monotonic()
                payload, record = self._repair_identity(index)
                n = record['n']
                batch = reader.batch([index],DEV,truth=False).to(assets.device)
                require(batch.valid.shape == (1,n) and bool(batch.valid.all())
                        and not bool(batch.future.count_nonzero()), 'repair read an invalid observed-only batch')
                endpoints, key = _window_goals(assets,student,index,batch,input_grad=False)
                np.testing.assert_allclose(endpoints.cpu().numpy(), self.read_endpoints(index)['g0'],
                                           atol=ATOL,rtol=RTOL)
                require(key == record['member_sha256'], 'repaired proposal noise key changed')
                context = assets.model.encode_context(batch.observed,batch.semantic_maps,batch.valid)
                noise = student_noise(key,batch.agent_ids[0],SEED,assets.device)
                path = decode_student(assets.model,context,endpoints,noise)[0].cpu().numpy()
                save_npz(payload,S0=path)
                immutable(payload.with_suffix('.json'),record | dict(
                    sha256=digest(payload),DDIM_calls=1,seconds=time.monotonic()-started))
        times = []
        for index in invalid:
            self._repair_path(index)
            path, _ = self._repair_identity(index)
            times.append(read(path.with_suffix('.json'))['seconds'])
        return dict(new_DDIM_calls=len(invalid),repaired_windows=len(invalid),seconds=sum(times))

    def _source_invalid(self, index):
        row = self.ref.rows[index,SEED]
        path = self.ref.folder / row['file']
        return not path.is_file() or digest(path) != row['sha256']


def setup_device(device):
    torch.set_num_threads(2)
    if device == 'cuda':
        configure_cuda()
        torch.cuda.set_per_process_memory_fraction(.45)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cuda','cpu'),default='cuda')
    args = parser.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    with (ASSETS/'auxiliary_gpu.lock').open('a') as lock:
        if args.device == 'cuda':
            fcntl.flock(lock,fcntl.LOCK_EX)
        setup_device(args.device)
        for fold in FOLDS:
            ref = dev_reference(fold)
            prepare_training(ref,RUN,args.device)
            if args.device == 'cuda':
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

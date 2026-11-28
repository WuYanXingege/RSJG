"""Choose one frozen S0 slot using only the original training endpoints."""

from pathlib import Path
import time

import numpy as np
import torch

from bcep.contracts import require
from joint_flow_fourfold.report import csvwrite

from .common import SOURCE, canonical_hash, digest, immutable, read
from .group_slot_recombination import data as regroup


K = 20
FOLDS = ('hotel', 'univ')
METRICS = ('JFDE', 'minFDE')
_ORDER = torch.arange(K)
_KEEP = _ORDER.expand(K, K)[~torch.eye(K, dtype=torch.bool)].reshape(K, K-1)


def _fde_sums(errors):
    """Use the native hard FDE reductions on packed valid people."""
    return torch.stack((errors.sum(1).div(errors.shape[1]).min(1).values.double(),
                        errors.min(2).values.double().sum(1)), dim=1)


def _window_totals(goals, target):
    # Match the native Euclidean endpoint errors and FP32 window reduction.
    errors = torch.linalg.vector_norm(goals.float() - target[None].float(), dim=-1).T.contiguous()
    kept = errors[:, _KEEP].permute(1, 0, 2).contiguous()
    return torch.cat((_fde_sums(errors[None]), _fde_sums(kept)), dim=0)


@torch.no_grad()
def _rank(windows):
    sums = torch.zeros((K+1, len(METRICS)), dtype=torch.float64)
    people = count = 0
    for goals, target in windows:
        n = len(target)
        require(goals.shape == (K,n,2) and target.shape == (n,2) and n > 0,
                'slot ranking requires one K20 window with valid training endpoints')
        sums += _window_totals(goals,target)
        people += n
        count += 1
    require(count > 0 and people > 0, 'empty train-only slot ranking population')
    values = sums / torch.tensor((count,people), dtype=sums.dtype)
    require(bool(torch.isfinite(values).all()) and bool((values[0] > 0).all()),
            'slot ranking requires positive finite baseline train metrics')
    baseline = dict(zip(METRICS,values[0].tolist()))
    rows = []
    for j, after in enumerate(values[1:].tolist()):
        delta = {metric: value-baseline[metric] for metric,value in zip(METRICS,after)}
        relative = {metric: delta[metric]/baseline[metric] for metric in METRICS}
        require(all(value >= -1e-7 for value in delta.values()), 'deleting a slot improved a minimum')
        rows.append(dict(slot=j, JFDE_m=after[0], minFDE_m=after[1],
                         JFDE_increase_m=delta['JFDE'], minFDE_increase_m=delta['minFDE'],
                         JFDE_relative_increase=relative['JFDE'],
                         minFDE_relative_increase=relative['minFDE'],
                         max_relative_increase=max(relative.values())))
    ranked = sorted(rows,key=lambda row:(row['max_relative_increase'],row['slot']))
    for rank,row in enumerate(ranked,1):
        row.update(rank=rank, selected=rank==1)
    return dict(j_drop=ranked[0]['slot'], baseline=baseline, rows=rows,
                train_windows=count, train_agents=people)


def _train_windows(cache, split):
    for index in cache.indices:
        member = split['members'][index]
        record = split['records'][index]
        row = cache.rows[index]
        require(row['n'] == len(member['agent_ids'])
                and row['member_sha256'] == canonical_hash(member)
                and row['original_record_sha256'] == record['sha256'] == digest(record['path']),
                'train endpoint shard is not aligned to the original train record')
        sample = cache.load(index)
        with np.load(record['path'],allow_pickle=False) as original:
            future = original['future']
            require(future.shape == (row['n'],12,2) and future.dtype == np.float32
                    and np.array_equal(original['observed'],sample['observed']), 'training members differ')
        yield torch.from_numpy(sample['g0']),torch.from_numpy(future[:,-1])


def rank_fold(ref, output: Path, *, cache_root=regroup.RUN):
    """Seal one zero-based drop index and all 20 scores for a single fold."""
    require(ref.fold in FOLDS and ref.split['fold']==ref.fold, 'unexpected fold for slot ranking')
    require(digest(ref.identity['S0_checkpoint'])==ref.identity['S0_sha256'],
            'selected frozen S0 checkpoint changed')
    cache_root = Path(cache_root)
    cache = regroup.open_training_cache(ref,cache_root)
    if cache_root.resolve()==regroup.RUN.resolve():
        prior = read(regroup.DOC/ref.fold/'TRAIN_CACHE_RECEIPT.json')
        require(prior['sha256']==cache.identity['cache_sha256'],
                'the reused train cache differs from its original GitHub receipt')
    require(tuple(ref.split['train_indices']) == cache.indices
            and not set(cache.indices).intersection(ref.split['selection_dev_indices'])
            and not set(cache.indices).intersection(ref.split['heldout_confirmation_indices']),
            'train cache overlaps a validation or test window')
    identity = dict(fold=ref.fold,purpose='train',cache_identity=cache.identity,
                    S0_epoch=ref.identity['S0_epoch'],S0_sha256=ref.identity['S0_sha256'],
                    native_score_source_sha256=digest(SOURCE/'ebjd/metrics.py'),
                    source_sha256=digest(__file__),selection='minimize max relative FDE increase, ties lowest slot')
    folder = Path(output)/ref.fold
    folder.mkdir(parents=True,exist_ok=True)
    target = folder/'DELETION_RANKING.json'
    table = folder/'DELETION_RANKING.csv'
    if target.exists():
        receipt = read(target)
        require(receipt['identity']==identity and receipt['csv_sha256']==digest(table),
                'frozen training deletion ranking has changed')
        return receipt
    start = time.monotonic()
    ranking = _rank(_train_windows(cache,ref.split))
    require(ranking['train_windows']==len(cache.indices),'train slot scoring coverage incomplete')
    csvwrite(table,ranking['rows'])
    receipt = ranking | dict(fold=ref.fold,identity=identity,cache_identity=cache.identity,
        source_sha256=identity['source_sha256'], seconds=time.monotonic()-start,
        csv_sha256=digest(table),training_updates=0,DDIM_calls=0,
        development_truth_reads=0,test_truth_reads=0)
    immutable(target,receipt)
    return receipt

"""Rechoose HOTEL's single CV replacement slot on original train, then verify it once."""

import fcntl
import time

import numpy as np
import torch

from bcep.contracts import require
from ebjd.metrics import hard_metric_statistics
from joint_flow_fourfold.report import csvwrite

from .common import ASSETS, METRICS, SOURCE, canonical_hash, digest, immutable, read
from .fixed_k20_center import constant_velocity
from .fixed_k20_eval import _locked_dev_receipt, export_fold, score_phase
from .fixed_k20_inputs import DOC as PRIOR_DOC, RUN as PRIOR_RUN, fixed_inputs
from .fixed_k20_slot import K, _fde_sums
from .group_slot_recombination import data as regroup


BASE = 'fea283129b483d552ca6c7b977413bc92b848876'
RUN = ASSETS / 'fixed_k20_cv_followup_20261128' / 'hotel_reselection'
FOLD = 'hotel'
PHASE = 'cv_development'
PRIOR_SLOT = 7
SEED = 2035


def _train_windows(cache, split):
    for index in cache.indices:
        member = split['members'][index]
        record = split['records'][index]
        row = cache.rows[index]
        require(row['n'] == len(member['agent_ids'])
                and row['member_sha256'] == canonical_hash(member)
                and row['original_record_sha256'] == record['sha256'] == digest(record['path']),
                'HOTEL cached endpoint or agent order differs from the original train record')
        sample = cache.load(index)
        with np.load(record['path'], allow_pickle=False) as original:
            future = original['future']
            require(future.shape == (row['n'], 12, 2) and future.dtype == np.float32
                    and np.array_equal(original['observed'], sample['observed']),
                    'HOTEL train observation or target differs from cached endpoint membership')
        yield (torch.from_numpy(sample['g0']), torch.from_numpy(sample['observed']),
               torch.from_numpy(future[:, -1]))


def _totals(goals, cv, target):
    errors = torch.linalg.vector_norm(goals - target[None], dim=-1).T.contiguous()
    cv_errors = torch.linalg.vector_norm(cv - target, dim=-1)
    substitutions = errors.expand(K, -1, -1).clone()
    slots = torch.arange(K)
    substitutions[slots, :, slots] = cv_errors
    return _fde_sums(torch.cat((errors[None], substitutions)))


def _verify_native(goals, cv, target, totals):
    paths = goals[None, :, :, None, :].expand(K+1, -1, -1, 12, -1).clone()
    slots = torch.arange(K)
    paths[slots+1, slots, :, -1] = cv
    truth = target[None, :, None].expand(K+1, -1, 12, -1)
    valid = torch.ones((K+1, len(target)), dtype=torch.bool)
    native = hard_metric_statistics(paths, truth, valid)
    expected = torch.stack((native.scene_jfde.double(), native.agent_minfde.double().sum(1)), dim=1)
    return float((totals-expected).abs().max())


@torch.no_grad()
def _rank_train(ref, config_sha):
    cache_root = regroup.RUN
    if not (cache_root / FOLD / 'train_cache' / 'SEALED.json').is_file():
        cache_root = RUN / 'recovered_train_endpoints'
        with (ASSETS / 'auxiliary_gpu.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            regroup.setup_device('cuda')
            regroup.prepare_training(ref, cache_root, 'cuda')
    cache = regroup.open_training_cache(ref, cache_root)
    if cache_root == regroup.RUN:
        original = read(regroup.DOC / FOLD / 'TRAIN_CACHE_RECEIPT.json')
        require(original['sha256'] == cache.identity['cache_sha256'], 'HOTEL train seal no longer matches archive')
    require(tuple(ref.split['train_indices']) == cache.indices
            and not set(cache.indices).intersection(ref.split['selection_dev_indices'])
            and not set(cache.indices).intersection(ref.split['heldout_confirmation_indices']),
            'HOTEL train slot selector must not access development or test windows')
    identity = dict(config_sha256=config_sha, train_cache=cache.identity, S0_sha256=ref.identity['S0_sha256'])
    ranking_path, table_path = RUN / 'REPLACEMENT_RANKING.json', RUN / 'REPLACEMENT_RANKING.csv'
    if ranking_path.exists():
        ranking = read(ranking_path)
        require(ranking['identity'] == identity and ranking['table_sha256'] == digest(table_path),
                'sealed HOTEL train ranking differs')
        return ranking

    start = time.monotonic()
    sums = torch.zeros((K+1, 2), dtype=torch.float64)
    windows = people = 0
    native_error = None
    for goals, observed, target in _train_windows(cache, ref.split):
        n = len(target)
        require(n > 0 and goals.shape == (K, n, 2) and observed.shape == (n, 8, 2)
                and target.shape == (n, 2) and bool(torch.isfinite(target).all()), 'invalid packed train window')
        cv = constant_velocity(observed, ref.split['dt'])[:, -1]
        totals = _totals(goals, cv, target)
        if native_error is None:
            native_error = _verify_native(goals, cv, target, totals)
            require(native_error <= 1e-7, 'HOTEL endpoint ranking disagrees with native scorer')
        sums += totals
        windows += 1
        people += n
    require(windows == len(cache.indices) and people > 0 and bool(torch.isfinite(sums).all()),
            'incomplete HOTEL train endpoint computation')
    values = (sums / torch.tensor((windows, people), dtype=torch.float64)).tolist()
    j_base, f_base = values[0]
    prior_train = read(PRIOR_DOC / FOLD / 'DELETION_RANKING.json')
    require(windows == prior_train['train_windows'] and people == prior_train['train_agents']
            and abs(j_base - prior_train['baseline']['JFDE']) <= 1e-12
            and abs(f_base - prior_train['baseline']['minFDE']) <= 1e-12,
            'HOTEL train S0 baseline does not match the original train ranking')
    rows = [dict(slot=slot, JFDE_m=j, minFDE_m=f, delta_JFDE_m=j-j_base,
                 delta_minFDE_m=f-f_base, feasible=f <= f_base)
            for slot, (j, f) in enumerate(values[1:])]
    ordered = sorted(rows, key=lambda row: (not row['feasible'], row['JFDE_m'], row['slot']))
    selected = ordered[0]['slot'] if ordered[0]['feasible'] else None
    for rank, row in enumerate(ordered, 1):
        row.update(rank=rank, selected=row['slot'] == selected)
    csvwrite(table_path, ordered)
    result = dict(
        identity=identity, windows=windows, agents=people, F_base_m=f_base, J_base_m=j_base,
        rows=ordered, selected_slot=selected, feasible_slots=[row['slot'] for row in ordered if row['feasible']],
        table_sha256=digest(table_path), native_preflight_max_abs_m=native_error,
        reused_endpoint_cache=cache_root == regroup.RUN, training_updates=0, new_DDIM_calls=0,
        development_truth_reads=0, test_truth_reads=0, seconds=time.monotonic()-start,
    )
    immutable(ranking_path, result)
    return result


def _development(ref, cache, freeze, selection):
    if any(cache._source_invalid(index) for index in ref.indices):
        with (ASSETS / 'auxiliary_gpu.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            regroup.setup_device('cuda')
            replay = cache.repair_if_needed('cuda')
    else:
        replay = cache.repair_if_needed('cpu')
    immutable(RUN / 'S0_REPLAY.json', replay)
    identity = selection | dict(
        phase=PHASE, method='cv', j_drop=freeze['selected_slot'], seed=SEED,
        config_sha256=freeze['config_sha256'], candidate_checkpoint=None, checkpoint_sha256=None,
        frozen_slot_sha256=digest(RUN / 'SELECTION_FROZEN.json'),
    )

    def read_base(index, seed):
        require(seed == SEED, 'HOTEL development uses the single locked seed')
        return cache.read_path(index)

    export = export_fold(ref, RUN, identity, read_base)
    immutable(RUN / PHASE / FOLD / 'EXPORT_COMPLETE.json', export)
    result = score_phase(RUN, {FOLD: (ref, identity, read_base)}, phase=PHASE, development_folds=(FOLD,))
    require(result['groups'][FOLD]['all']['agents'] == 5203, 'changed HOTEL development agent population')
    return result, replay['new_DDIM_calls']


def main():
    started = time.monotonic()
    torch.set_num_threads(2)
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selections, refs, caches = fixed_inputs()
        ref, cache = refs[FOLD], caches[FOLD]
        cache.output = RUN / 'recovered_dev_S0'
        config = dict(
            base_commit=BASE, fold=FOLD, purpose='one train-ranked HOTEL CV replacement on original development',
            selection=selections[FOLD], seed=SEED, K=20, diagnostic_K=21, dt=.4, steps=12, precision='FP32',
            train_feasibility='minFDE_slot <= original S0 minFDE', train_selector='ascending (JFDE, slot)',
            cv='use original constant_velocity(observed, dt): final three observation intervals',
            dev_gate=dict(all_JFDE_max_m=-.01, remaining_max_m=1e-6, groups=['all', 'N2_8']),
            training_updates=0, test_accesses=0, original_slot=PRIOR_SLOT,
            source_sha256={name: digest(SOURCE / name) for name in (
                'direct_k20/fixed_k20_hotel_reselection.py', 'direct_k20/fixed_k20_center.py',
                'direct_k20/fixed_k20_slot.py', 'direct_k20/fixed_k20_inputs.py',
                'direct_k20/fixed_k20_eval.py', 'ebjd/metrics.py')},
        )
        immutable(RUN / 'RUN_CONFIG.json', config)
        result_path = RUN / 'RESULT.json'
        if result_path.exists():
            result = read(result_path)
            require(result['config_sha256'] == digest(RUN / 'RUN_CONFIG.json')
                    and result['selection_sha256'] == digest(RUN / 'SELECTION_FROZEN.json'),
                    'previous HOTEL output belongs to a different frozen protocol')
            print(result['outcome'], result['selected_slot'], flush=True)
            return
        ranking = _rank_train(ref, digest(RUN / 'RUN_CONFIG.json'))
        selected = ranking['selected_slot']
        freeze_path = RUN / 'SELECTION_FROZEN.json'
        frozen = dict(selected_slot=selected, config_sha256=digest(RUN / 'RUN_CONFIG.json'),
                      train_ranking_sha256=digest(RUN / 'REPLACEMENT_RANKING.json'),
                      source='fixed train-only K20 minFDE constraint, then JFDE and native slot',
                      training_updates=0)
        if freeze_path.exists():
            require(all(read(freeze_path)[key] == value for key, value in frozen.items()),
                    'frozen HOTEL selected slot changed')
        else:
            immutable(freeze_path, frozen | dict(frozen_unix=time.time()))
        print('HOTEL frozen slot', selected, 'feasible slots', ranking['feasible_slots'], flush=True)
        prior_path, prior = _locked_dev_receipt(PRIOR_RUN, PHASE)
        require(prior['identities'][FOLD]['j_drop'] == PRIOR_SLOT and not prior['decision'][FOLD]['passed'],
                'original HOTEL slot-7 gate must remain failed')
        prior_groups = prior['groups'][FOLD]
        decision = groups = comparison = None
        new_ddim = 0
        if selected is None:
            outcome = 'no_feasible_train_slot'
        elif selected == PRIOR_SLOT:
            outcome = 'reused_archived_slot_7'
            decision, groups = prior['decision'][FOLD], prior_groups
        else:
            score, new_ddim = _development(ref, cache, frozen, selections[FOLD])
            require(score['first_truth_read_unix'] >= read(freeze_path)['frozen_unix'],
                    'HOTEL development truth was read before training slot freeze')
            decision, groups = score['decision'][FOLD], score['groups'][FOLD]
            outcome = 'qualified_for_five_seed_test' if decision['passed'] else 'development_failed_preserve_S24'
        if groups is not None:
            comparison = {group: {metric: dict(
                slot_7_m=prior_groups[group][f'final_{metric}_m'],
                new_m=groups[group][f'final_{metric}_m'],
                improvement_m=prior_groups[group][f'final_{metric}_m']-groups[group][f'final_{metric}_m'],
                deletion_cost_reduction_m=prior_groups[group][f'D_{metric}_m']-groups[group][f'D_{metric}_m'],
            ) for metric in METRICS} for group in groups if groups[group]['windows']}
            require(all(abs(value['improvement_m']-value['deletion_cost_reduction_m']) <= 2e-12
                        for metrics in comparison.values() for value in metrics.values()),
                    'slot-7 replay changed K21 instead of the deleted slot')
        result = dict(
            status='COMPLETE', selected_slot=selected, original_slot=PRIOR_SLOT, outcome=outcome,
            decision=decision, groups=groups, comparison_to_slot_7=comparison,
            prior_groups=prior_groups, prior_decision=prior['decision'][FOLD],
            prior_score_sha256=digest(prior_path), selection_sha256=digest(freeze_path),
            score_receipt_sha256=digest(RUN / PHASE / 'SCORE_RECEIPT.json') if selected not in (None, PRIOR_SLOT) else None,
            config_sha256=frozen['config_sha256'], training_updates=0, test_truth_reads=0,
            new_DDIM_calls=new_ddim, wall_seconds=time.monotonic()-started,
        )
        immutable(result_path, result)
        print(outcome, selected, flush=True)


if __name__ == '__main__':
    main()

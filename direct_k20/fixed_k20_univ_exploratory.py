"""Replay the unchanged UNIV CV candidate for five heldout inference seeds."""

import fcntl
import os
import time

import torch

from bcep.contracts import require
from .common import ASSETS, METRICS, SEEDS, SOURCE, digest, immutable, read, status, write
from .fixed_k20_eval import EXPLORATORY_SCOPE, _locked_dev_receipt, export_fold, score_phase
from .fixed_k20_inputs import RUN as DEV_RUN, fixed_inputs, open_exploratory_test_baseline


BASE = 'fea283129b483d552ca6c7b977413bc92b848876'
RUN = ASSETS / 'fixed_k20_cv_followup_20261128' / 'univ_exploratory'
SOURCES = (
    'direct_k20/fixed_k20_univ_exploratory.py', 'direct_k20/fixed_k20_inputs.py',
    'direct_k20/fixed_k20_eval.py', 'direct_k20/fixed_k20_center.py',
    'direct_k20/fixed_k20_slot.py', 'direct_k20/fixed_k20_run.py', 'ebjd/metrics.py',
)


def _start():
    selections, refs, _ = fixed_inputs()
    selection, ref = selections['univ'], refs['univ']
    cv_path, previous = _locked_dev_receipt(DEV_RUN, 'cv_development')
    original = previous['identities']['univ']
    require(not previous['decision']['univ']['passed'] and original['j_drop'] == 3
            and original['method'] == 'cv' and original['S0_epoch'] == 16
            and original['S0_sha256'] == selection['S0_sha256'], 'unexpected original UNIV CV identity')
    config_path = RUN / 'RUN_CONFIG.json'
    config = dict(base_commit=BASE, fold='univ', purpose='heldout_confirmation',
        test_scope=EXPLORATORY_SCOPE, development_gate='FAIL', previous_CV_score_sha256=digest(cv_path),
        previous_development_dense={key:previous['groups']['univ']['N9plus'][key] for key in
                                    ('windows', 'agents', 'delta_JFDE_m', 'delta_minFDE_m')},
        S0_epoch=16, S0_sha256=selection['S0_sha256'], precision='FP32', j_drop=3, dt=.4,
        seeds=list(SEEDS), K=20, diagnostic_K=21, observed_frames=8, prediction_frames=12,
        formula='obs[-1] + t * (obs[-1] - obs[-4]) / 3, t=1..12; original fixed CV implementation',
        source_sha256={name:digest(SOURCE / name) for name in SOURCES},
        unchanged_S0_slots=19, source_trajectories_per_window=20, train_updates=0,
        windows_per_seed=947, people_per_seed=24334,
        dense_windows_per_seed=907, dense_people_per_seed=24099,
        command='python -m direct_k20.fixed_k20_univ_exploratory')
    immutable(config_path, config)
    identity = original | dict(phase='test', test_scope=EXPLORATORY_SCOPE,
        dev_score_phase='cv_development', dev_score_sha256=digest(cv_path),
        config_sha256=digest(config_path), seeds=list(SEEDS))
    freeze_path = RUN / 'FROZEN_TEST.json'
    freeze = dict(test_scope=EXPLORATORY_SCOPE, development_gate='FAIL',
                  original_base_commit=BASE, config_sha256=digest(config_path),
                  folds={'univ':identity}, seeds=list(SEEDS))
    if freeze_path.is_file():
        recorded = read(freeze_path)
        require(all(recorded[key] == value for key, value in freeze.items()), 'UNIV test freeze changed')
    else:
        immutable(freeze_path, freeze | dict(freeze_unix=time.time()))
    return ref, identity, config


def _recover_only_missing(ref):
    absent = ref.missing_pairs()
    if not absent:
        return dict(new_DDIM_calls=0, seconds=0., windows=0, reused_pairs=len(ref.rows))
    from bcep_v2.runtime import configure_cuda

    with (ASSETS/'auxiliary_gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        configure_cuda()
        torch.cuda.set_per_process_memory_fraction(.45)
        return ref.repair(absent) | dict(reused_pairs=len(ref.rows)-len(absent))


def _run():
    if not (RUN / 'START.json').is_file():
        immutable(RUN / 'START.json', dict(unix=time.time(), pid=os.getpid()))
    status(RUN, 'FREEZING_UNIV_FIXED_CV')
    dev_ref, identity, config = _start()
    ref = open_exploratory_test_baseline(dev_ref, identity, RUN)
    split = ref.split
    counts = [len(split['members'][index]['agent_ids']) for index in ref.indices]
    dense = [n for n in counts if n >= 9]
    require((len(counts), sum(counts), len(dense), sum(dense)) == (947, 24334, 907, 24099)
            and len(ref.rows) == 4735, 'UNIV test population differs before export')
    replay = _recover_only_missing(ref)
    folder = RUN / 'test' / 'univ'
    folder.mkdir(parents=True, exist_ok=True)
    immutable(folder / 'S0_REPLAY.json', replay)
    status(RUN, 'EXPORTING_UNIV_FIXED_CV', reused_S0_pairs=replay['reused_pairs'])
    sealed = export_fold(ref, RUN, identity, ref.read_path)
    require(sealed['prediction_pairs'] == 4735 and sealed['truth_reads'] == 0,
            'UNIV CV predictions must be fully sealed before test scoring')
    immutable(folder / 'EXPORT_COMPLETE.json', sealed)
    status(RUN, 'SCORING_UNIV_FIXED_CV', prediction_pairs=sealed['prediction_pairs'])
    score = score_phase(RUN, {'univ':(ref, identity, ref.read_path)}, phase='test')
    require(score['prediction_pairs'] == 4735, 'incomplete exploratory test score')
    previous = config['previous_development_dense']['delta_JFDE_m']
    dense_result = score['groups']['univ']['N9plus']
    test_delta = dense_result['delta_JFDE_m']
    result = dict(fold='univ', status='COMPLETE', development_gate='FAIL',
        scope=EXPLORATORY_SCOPE, j_drop=3, S0_sha256=config['S0_sha256'], S0_epoch=16,
        config_sha256=digest(RUN / 'RUN_CONFIG.json'), test_freeze_sha256=digest(RUN / 'FROZEN_TEST.json'),
        score_receipt_sha256=digest(RUN / 'test' / 'SCORE_RECEIPT.json'),
        prediction_pairs=score['prediction_pairs'],
        development_dense_JFDE_delta_m=previous, test_dense_JFDE_delta_m=test_delta,
        test_dense_minFDE_delta_m=dense_result['delta_minFDE_m'],
        development_dense_JFDE_gain_retention=test_delta / previous,
        S0_replay=replay, added_candidate_DDIM_calls=sealed['new_candidate_ddim_calls'],
        training_updates=0, native_replay_max_abs_m=score['native_replay_max_abs_m']['univ'],
        dense_test_metrics={metric:{'delta_m':dense_result[f'delta_{metric}_m'],
            'sample_sd_m':dense_result[f'delta_{metric}_m_sample_sd']} for metric in METRICS})
    end = RUN / 'COMPLETE.json'
    if end.is_file():
        require(all(read(end)[key] == value for key, value in result.items()), 'UNIV completion receipt changed')
    else:
        elapsed = time.time() - read(RUN / 'START.json')['unix']
        immutable(end, result | dict(wall_seconds=elapsed))
    status(RUN, 'COMPLETE', test_dense_JFDE_delta_m=test_delta,
           test_dense_minFDE_delta_m=dense_result['delta_minFDE_m'])


def main():
    torch.set_num_threads(2)
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _run()


if __name__ == '__main__':
    main()

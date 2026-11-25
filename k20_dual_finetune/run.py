"""Fit two fold-local students, evaluate their fixed epoch eight, and archive."""
import argparse
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from bcep.contracts import digest
from bcep_joint.common import read, write
from joint_flow_fourfold.common import immutable, source_hashes, status
from joint_flow_fourfold.controller import check_protection

SOURCE = Path(__file__).resolve().parents[1]
ASSETS = SOURCE.parent / 'RSJG_EBJD_assets/experiments'
PRIOR = ASSETS / 'joint_flow_fourfold_confirmation_20261122'
RUN = ASSETS / 'k20_dual_finetune_20261125'
DOC = SOURCE / 'docs/research/continuous_joint_flow/k20_dual_finetune_20261125'
BASE = 'e520cb01bfc11d546d868e0a82c25b210301b4da'
BRANCH = 'experiment/k20-dual-finetune-20261125'
FOLDS = ('hotel', 'univ')
YIELD_TO_MAIN = 75
EXPECTED = {
    'hotel': (19, '1806aabf478641021584aca02f4040e8e9c2431fe80c457d5adabba259d073d0', 445),
    'univ': (33, '776235267e5cf4f1be2a9ba80cd5eaf8c44cdda5a8e94b6cde3ae3f78a24681a', 947),
}
HISTORICAL = {
    'hotel': ASSETS / 'zara1_hotel_frozen_test_20261124/hotel/test',
    'univ': ASSETS / 'univ_frozen_test_20261125/univ/test',
}


def sources():
    local_sources = {
        str(path.relative_to(SOURCE)): digest(path)
        for path in (SOURCE / 'k20_dual_finetune').glob('*.py')
    }
    metric_source = 'hotel_endpoint_diagnostic/math.py'
    return source_hashes() | local_sources | {metric_source: digest(SOURCE / metric_source)}


def protect(config):
    check_protection(read(PRIOR / 'RUN_CONFIG.json'))
    assert digest(PRIOR / 'RUN_CONFIG.json') == config['original_controller_config_sha256']
    assert sources() == config['source_sha256']


def prepare():
    from .train import RECIPE

    identities = {}
    for fold in FOLDS:
        upstream_path = PRIOR / fold / 'UPSTREAM_FROZEN.json'
        selected_path = PRIOR / fold / 'B_FROZEN.json'
        upstream, selected = read(upstream_path), read(selected_path)
        epoch, sha, windows = EXPECTED[fold]
        assert selected['fold'] == upstream['fold'] == fold
        assert (selected['rank1_epoch'], selected['rank1_sha256']) == (epoch, sha)
        assert digest(selected['rank1_checkpoint']) == sha
        assert selected['test_access'] is False
        for stage in ('density', 'head', 'joint'):
            asset = upstream[stage]
            assert digest(asset['checkpoint']) == asset['sha256']
        split = read(upstream['split_path'])
        assert digest(upstream['split_path']) == upstream['split_sha256']
        assert len(split['heldout_confirmation_indices']) == windows
        assert set(split['train_indices']).isdisjoint(split['heldout_confirmation_indices'])
        prior_score = read(HISTORICAL[fold] / 'SCORE_RECEIPT.json')
        assert prior_score['status'] == 'COMPLETE' and prior_score['windows'] == windows
        assert digest(HISTORICAL[fold] / 'PER_SEED.csv') == prior_score['output_sha256']['PER_SEED.csv']
        identities[fold] = dict(
            teacher_epoch=epoch, teacher_sha256=sha,
            checkpoint=selected['rank1_checkpoint'],
            upstream_sha256=digest(upstream_path),
            selection_sha256=digest(selected_path),
            upstream_weights={
                stage: dict(epoch=upstream[stage]['frozen']['epoch'], sha256=upstream[stage]['sha256'])
                for stage in ('density', 'head', 'joint')
            },
            split_sha256=upstream['split_sha256'],
            train_windows=len(split['train_indices']), exploratory_windows=windows,
            historic_per_seed_sha256=prior_score['output_sha256']['PER_SEED.csv'],
            historic_score_receipt_sha256=digest(HISTORICAL[fold] / 'SCORE_RECEIPT.json'),
        )
    config = dict(
        base_commit=BASE, folds=list(FOLDS), identities=identities, recipe=RECIPE,
        source_sha256=sources(),
        original_controller_config_sha256=digest(PRIOR / 'RUN_CONFIG.json'),
        training_split='train', evaluation_split='heldout_confirmation',
        report_scope='原测试集上的单种子探索性开发评估',
        evaluation_seed=2035, evaluated_epoch=8, M=256, K=20,
        objective='agent-micro M/M0 + window-macro J/J0 + mean(relu(J-JB))/J0 + 0.1*mean(native_delta_nll)',
        inference='original B medoids and bounded CandidateHead; conditional 20-step DDIM, eta 0, raw-ID keyed noise',
        endpoint_gate=dict(delta_minFDE_le=-0.020, delta_JFDE_lt=0.),
        retention='all eight complete epoch checkpoints, last, fixed selected epoch 8; no validation ranking before epoch 8',
        cuda_allocator_fraction=.35, min_free_gpu_mib=6144,
        environment=dict(CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2',
                         MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='0'),
    )
    protect(config)
    immutable(RUN / 'RUN_CONFIG.json', config)
    return config


def main_waiting_for_gpu():
    return read(PRIOR / 'STATUS.json')['status'] == 'WAITING_FOR_GPU'


def worker(fold):
    import torch
    from bcep_v2.runtime import configure_cuda
    from joint_flow_dependency.model import EdgeDependence
    from joint_flow_fourfold.inputs import Inputs
    from .train import train_fold
    from .evaluate import evaluate

    config = read(RUN / 'RUN_CONFIG.json')
    protect(config)
    torch.set_num_threads(2)
    configure_cuda()
    torch.cuda.set_per_process_memory_fraction(config['cuda_allocator_fraction'])
    folder = RUN / fold
    folder.mkdir(exist_ok=True)
    expected = config['identities'][fold]
    manifest = PRIOR / fold / 'UPSTREAM_FROZEN.json'
    assert digest(manifest) == expected['upstream_sha256']
    assert digest(expected['checkpoint']) == expected['teacher_sha256']
    inputs = Inputs(manifest, folder, 'cuda')
    model, provider = inputs.load_frozen('cuda')
    teacher = EdgeDependence()
    saved = torch.load(expected['checkpoint'], map_location='cpu', weights_only=False)
    assert saved['completed_epoch'] == expected['teacher_epoch']
    teacher.load_state_dict(saved['model'], strict=True)
    teacher.to('cuda').eval().requires_grad_(False)
    result = train_fold(inputs, model, provider, teacher, folder, device='cuda',
                        should_yield=main_waiting_for_gpu)
    if result is None or main_waiting_for_gpu():
        return YIELD_TO_MAIN
    path = Path(result['final_checkpoint'])
    assert digest(path) == result['final_checkpoint_sha256']
    final = torch.load(path, map_location='cpu', weights_only=False)
    assert final['completed_epoch'] == 8
    student_provider = deepcopy(provider)
    student_dependence = deepcopy(teacher)
    student_provider.head.readout[-1].load_state_dict(final['readout'], strict=True)
    student_dependence.load_state_dict(final['dependence'], strict=True)
    evaluate(inputs, model, student_provider, student_dependence, folder,
             result['final_checkpoint_sha256'], HISTORICAL[fold], device='cuda')
    protect(config)
    write(folder / 'COMPLETE.json', dict(fold=fold, epoch=8,
        evaluation_sha256=digest(folder / 'evaluation/SUMMARY.json'), unix=time.time()))
    return 0


def gpu_free():
    return int(subprocess.check_output([
        'nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'
    ], text=True).splitlines()[0])


def execute_fold(fold, config):
    folder = RUN / fold
    if (folder / 'COMPLETE.json').exists():
        assert read(folder / 'COMPLETE.json')['evaluation_sha256'] == digest(folder / 'evaluation/SUMMARY.json')
        return
    log_path = RUN / f'{fold}.log'
    cmd = [sys.executable, '-m', 'k20_dual_finetune.run', '--worker', fold]
    failures = 0
    while failures < 3:
        protect(config)
        while main_waiting_for_gpu() or gpu_free() < config['min_free_gpu_mib']:
            status(RUN, 'WAITING_FOR_GPU', fold=fold, main_stage_has_priority=True)
            time.sleep(30)
        assert shutil.disk_usage(RUN).free > 6 * 1024**3
        failures_before = failures
        with (ASSETS / 'auxiliary_gpu.lock').open('a') as gpu_lock:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX)
            if main_waiting_for_gpu() or gpu_free() < config['min_free_gpu_mib']:
                continue
            status(RUN, 'FOLD_RUNNING', fold=fold, attempt=failures + 1)
            with log_path.open('a', buffering=1) as log:
                child = subprocess.Popen(cmd, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT)
                write(RUN / 'ACTIVE_CHILD.json', dict(fold=fold, pid=child.pid, unix=time.time()))
                code = child.wait()
            if code == 0:
                assert (folder / 'COMPLETE.json').exists()
                return
            if code != YIELD_TO_MAIN:
                failures += 1
        if failures == failures_before:
            status(RUN, 'YIELDED_TO_MAIN_STAGE', fold=fold)
        elif failures < 3:
            status(RUN, 'RESUMING_SAME_RUN', fold=fold, exit_code=code, failures=failures)
        time.sleep(30)
    raise RuntimeError(f'{fold} failed three attempts in the same run. See {log_path}')


def archive():
    from .report import build_report

    build_report()
    assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == BRANCH
    assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], text=True).strip()
    subprocess.run(['git', 'add', str(DOC.relative_to(SOURCE))], check=True)
    subprocess.run(['git', 'diff', '--cached', '--check'], check=True)
    if subprocess.call(['git', 'diff', '--cached', '--quiet']):
        subprocess.run(['git', 'commit', '-m', 'Archive HOTEL and UNIV K20 dual-objective epoch-eight evaluation'], check=True)
    subprocess.run(['git', 'push', '-u', 'origin', BRANCH], check=True)
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/' + BRANCH],
                                     text=True).split()[0]
    assert sha == remote
    write(RUN / 'DELIVERY.json', dict(sha=sha, verified_remote_sha=remote, branch=BRANCH))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=FOLDS)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = prepare()
        if args.prepare_only:
            status(RUN, 'PREPARED')
            return 0
        if not (RUN / 'START.json').exists():
            write(RUN / 'START.json', dict(unix=time.time(),
                command=[sys.executable, '-m', 'k20_dual_finetune.run'],
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()))
        for fold in FOLDS:
            execute_fold(fold, config)
        protect(config)
        write(RUN / 'WALL_TIME.json', dict(seconds=time.time() - read(RUN / 'START.json')['unix']))
        status(RUN, 'ARCHIVING')
        archive()
        status(RUN, 'COMPLETE', folds=list(FOLDS))
    return 0


if __name__ == '__main__':
    try:
        exit_code = main()
    except Exception:
        if '--worker' not in sys.argv:
            status(RUN, 'INTERRUPTED_RESUME_SAME_RUN', traceback=traceback.format_exc())
        raise
    raise SystemExit(exit_code)

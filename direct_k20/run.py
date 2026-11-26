"""Run fixed five-fold direct-K20 training, freeze, export, score and publish."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from bcep.contracts import require
from .common import (
    ASSETS, BASE, BRANCH, DOC, ENVIRONMENT, FOLDS, MIX_REFERENCE, RUN, SEEDS, SOURCE,
    digest, immutable, read, runtime_sources, status, write,
)
from .references import discover

GPU_FRACTION = .82
MINIMUM_FREE_MIB = 12500


def prepare():
    from .train import RECIPE

    refs = discover()
    require(all(not ref['missing'] for ref in refs.values()), 'missing original A/B trajectory cache')
    config = dict(base_commit=BASE, earlier_mix_reference=MIX_REFERENCE,
        source_sha256=runtime_sources(),
        reference={fold: refs[fold]['metadata'] for fold in FOLDS},
        folds=list(FOLDS), inference_seeds=list(SEEDS), teacher_seed=2035, validation_seed=2035,
        evaluation_scope='历史测试结果启发的五折五种子探索性对照',
        K=20, prediction_frames=12, dt=.4, unit='m', ddim_steps=20, proposal_eta=0.,
        optimizer=RECIPE,
        architecture=dict(queries=20, query_width=32, personal_width=64,
                          message_width=64, MLP_hidden_width=64, latent_width=2,
                          neighbor_aggregation='mean directed messages; zero when isolated'),
        objectives=dict(
            epochs_1_to_12='MF/bMF + JF/bJF + 0.25*(TA/bMF + TB/bJF)',
            epochs_13_to_24='L_end + 0.5*(MA_hat/bMA + JA_hat/bJA)',
            temporal_window_sampling='uniform without replacement; 2/16, tail approximately 1/8 at least one'),
        selectors=dict(epochs=[12, 16, 20, 24], order='minimum max of four B ratios, then mean ratio, then epoch'),
        retention='complete snapshots for all 24 epochs; last recovery; selected/best; Top5 lists all four evaluated epochs',
        same_device_timing='fixed 8 hashed heldout windows per fold, seed 2035, resident observations through full trajectories',
        gpu_allocator_fraction=GPU_FRACTION, minimum_free_gpu_mib=MINIMUM_FREE_MIB,
        environment=ENVIRONMENT, reproduce='python -m direct_k20.run with recorded environment',
    )
    immutable(RUN / 'RUN_CONFIG.json', config)
    return refs, config


def _gpu_free():
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    return int(lines[0])


def _setup_worker():
    import torch
    from bcep_v2.runtime import configure_cuda

    config = read(RUN / 'RUN_CONFIG.json')
    require(config['source_sha256'] == runtime_sources(), 'fixed training/inference source changed')
    torch.set_num_threads(2)
    configure_cuda()
    torch.cuda.set_per_process_memory_fraction(config['gpu_allocator_fraction'])


def worker(stage, fold):
    import torch
    from bcep_joint.common import seed_all
    from .assets import load_fold, prepare_training
    from .model import Student

    _setup_worker()
    folder = RUN / fold
    folder.mkdir(exist_ok=True)
    assets = load_fold(fold, folder, 'cuda')
    # All trained tensors start with the fixed initialization, then resume from last.
    seed_all(3201)
    student = Student().to('cuda')
    if stage == 'train':
        from .train import train_fold
        status(RUN, 'PREPARING_TRAIN_TEACHERS', fold=fold)
        cache = prepare_training(assets, folder / 'cache')
        status(RUN, 'FITTING_WITH_FIXED_SCHEDULE', fold=fold)
        result = train_fold(assets, student, cache, folder / 'training', device='cuda')
        require(result['completed_epochs'] == 24 and result['selected_epoch'] in (12, 16, 20, 24)
                and digest(result['selected_checkpoint']) == result['selected_sha256'],
                'training result was not fixed by the required selector')
        immutable(folder / 'TRAINING_COMPLETE.json', result)
    else:
        from .evaluate import export_fold
        barrier = RUN / 'FROZEN_STUDENTS.json'
        selected = read(barrier)['folds'][fold]
        checkpoint = Path(selected['selected_checkpoint'])
        require(digest(checkpoint) == selected['selected_sha256'], 'selected checkpoint changed')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        require(saved['completed_epoch'] == selected['selected_epoch'], 'wrong selected student epoch')
        student.load_state_dict(saved['student'], strict=True)
        student.eval().requires_grad_(False)
        result = export_fold(assets, student, selected, barrier, folder / 'test')
        immutable(folder / 'TEST_EXPORT_COMPLETE.json', result)


def execute(stage, fold, config):
    filename = 'TRAINING_COMPLETE.json' if stage == 'train' else 'TEST_EXPORT_COMPLETE.json'
    output = RUN/fold/filename
    if output.exists():
        return
    cmd = [sys.executable, '-m', 'direct_k20.run', '--worker', stage, fold]
    with (ASSETS / 'auxiliary_gpu.lock').open('a') as device:
        fcntl.flock(device, fcntl.LOCK_EX)
        for attempt in range(1, 4):
            while _gpu_free() < config['minimum_free_gpu_mib']:
                status(RUN, 'WAITING_FOR_GPU', stage=stage, fold=fold)
                time.sleep(30)
            require(shutil.disk_usage(RUN).free > 5*1024**3, 'less than five GiB free')
            require(config['source_sha256'] == runtime_sources(), 'run source changed')
            with (RUN/f'{fold}_{stage}.log').open('a', buffering=1) as log:
                child = subprocess.Popen(cmd, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT)
                write(RUN/'ACTIVE_CHILD.json', dict(
                    pid=child.pid, stage=stage, fold=fold, attempt=attempt, unix=time.time(), command=cmd))
                code = child.wait()
            if code == 0:
                require(output.exists(), f'{fold} {stage} did not publish a completion receipt')
                return
            status(RUN, 'RECOVERING_SAME_RUN', stage=stage, fold=fold, attempt=attempt, exit_code=code)
            if attempt < 3:
                time.sleep(30)
    raise RuntimeError(f'{fold} {stage} interrupted three times; see {fold}_{stage}.log')


def freeze(config):
    target = RUN / 'FROZEN_STUDENTS.json'
    folds = {}
    for fold in FOLDS:
        done = RUN / fold / 'TRAINING_COMPLETE.json'
        result = read(done)
        folds[fold] = {k: result[k] for k in ('selected_epoch', 'selected_checkpoint', 'selected_sha256')}
        require(digest(folds[fold]['selected_checkpoint']) == folds[fold]['selected_sha256'],
                'selected student changed before freeze')
        folds[fold]['training_receipt_sha256'] = digest(done)
    core = dict(status='ALL_FIVE_STUDENTS_FROZEN', folds=folds,
                seeds=list(SEEDS), config_sha256=digest(RUN/'RUN_CONFIG.json'))
    if target.exists():
        saved = read(target)
        require(all(saved[key] == value for key, value in core.items()), 'five-fold freeze barrier changed')
    else:
        immutable(target, dict(core, freeze_unix=time.time()))
    return target


def archive():
    from .report import build_report

    build_report()
    assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == BRANCH
    assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], text=True).strip()
    subprocess.run(['git', 'add', str(DOC.relative_to(SOURCE))], check=True)
    subprocess.run(['git', 'diff', '--cached', '--check'], check=True)
    if subprocess.call(['git', 'diff', '--cached', '--quiet']):
        subprocess.run(['git', 'commit', '-m', 'Archive five-fold direct K20 frozen student experiment'], check=True)
    subprocess.run(['git', 'push', '-u', 'origin', BRANCH], check=True)
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/' + BRANCH],
                                     text=True).split()[0]
    require(sha == remote, 'remote experiment commit differs')
    write(RUN/'DELIVERY.json', dict(sha=sha, verified_remote_sha=remote, branch=BRANCH))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', nargs=2, metavar=('STAGE', 'FOLD'))
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.worker:
        stage, fold = args.worker
        require(stage in ('train','test') and fold in FOLDS, 'invalid worker')
        worker(stage, fold)
        return
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        refs, config = prepare()
        if args.prepare_only:
            status(RUN, 'PREPARED')
            return
        require(all(os.environ.get(key) == value for key, value in ENVIRONMENT.items()),
                'run with the recorded environment')
        if not (RUN/'START.json').exists():
            write(RUN/'START.json', dict(unix=time.time(), command=[sys.executable, '-m', 'direct_k20.run'],
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()))
        for fold in FOLDS:
            execute('train', fold, config)
        barrier = freeze(config)
        for fold in FOLDS:
            execute('test', fold, config)
        require(config['source_sha256'] == runtime_sources(), 'frozen source changed')
        import torch
        from .evaluate import score_all
        torch.set_num_threads(1)
        score_all(refs, barrier)
        write(RUN/'WALL_TIME.json', dict(seconds=time.time() - read(RUN/'START.json')['unix']))
        status(RUN, 'ARCHIVING')
        archive()
        status(RUN, 'COMPLETE', folds=list(FOLDS), seeds=list(SEEDS))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if '--worker' not in sys.argv:
            status(RUN, 'INTERRUPTED_RESUME_SAME_RUN', traceback=traceback.format_exc())
        raise

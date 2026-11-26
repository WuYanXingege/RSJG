"""Fixed checkpoint UNIV test, isolated from the running five-fold job."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from bcep.contracts import require
from .common import (
    ASSETS, ENVIRONMENT, RUN as TRAIN_RUN, SEEDS, SOURCE,
    digest, immutable, read, runtime_sources, status, write,
)
from .references import discover

RUN = ASSETS / 'direct_k20_univ_test_20261126'
DOC = SOURCE / 'docs/research/continuous_joint_flow/direct_k20_univ_test_20261126'
BRANCH = 'experiment/direct-k20-univ-test-20261126'
TRAIN_SOURCE = SOURCE.parent / 'RSJG_direct_k20_20261126'
BASE = '1a7444b0ec2f2b0bf1c9e31c8e7f04decddf2a91'
FOLD = 'univ'
GPU_ALLOCATOR_FRACTION = .24
MINIMUM_FREE_MIB = 7168


def prepare():
    source_config = read(TRAIN_RUN / 'RUN_CONFIG.json')
    for name, sha in source_config['source_sha256'].items():
        require(digest(TRAIN_SOURCE/name) == sha, 'existing training source changed')
    result_path = TRAIN_RUN / FOLD / 'TRAINING_COMPLETE.json'
    result = read(result_path)
    require(result['status'] == 'COMPLETE' and result['completed_epochs'] == 24
            and result['selected_epoch'] in (12, 16, 20, 24), 'UNIV has not completed fixed selection')
    require(result['selected_sha256'] == digest(result['selected_checkpoint']),
            'selected UNIV checkpoint has changed')
    ordered = sorted(result['dev_scores'], key=lambda item: (
        item['max_ratio'], item['mean_ratio'], item['epoch']))
    require(len(ordered) == 4 and {x['epoch'] for x in ordered} == {12,16,20,24}
            and ordered[0]['epoch'] == result['selected_epoch'],
            'fixed ratio selector does not match locked checkpoint')
    reference = discover()[FOLD]
    require(not reference['missing'] and len(reference['indices']) == 947
            and reference['metadata']['agents'] == 24334, 'official UNIV reference incomplete')
    config = dict(base_commit=BASE, fold=FOLD, windows=947, agents=24334,
        scope='用户追加授权的 UNIV 单折五种子探索性测试', seeds=list(SEEDS), K=20, frames=12,
        ddim_steps=20, proposal_eta=0., training_updates=0, baseline_reinference=0,
        training_config_sha256=digest(TRAIN_RUN/'RUN_CONFIG.json'),
        training_completion_sha256=digest(result_path), selected_epoch=result['selected_epoch'],
        selected_checkpoint=result['selected_checkpoint'], selected_sha256=result['selected_sha256'],
        reference=reference['metadata'], source_sha256=runtime_sources(),
        resources='one independent CUDA exporter concurrently; training process and settings unchanged',
        cuda_allocator_fraction=GPU_ALLOCATOR_FRACTION, min_free_gpu_mib=MINIMUM_FREE_MIB,
        environment=ENVIRONMENT,
        reproduce='python -m direct_k20.early_univ with recorded environment')
    immutable(RUN/'RUN_CONFIG.json', config)
    path = RUN/'FROZEN_STUDENTS.json'
    core = dict(status='USER_AUTHORIZED_SINGLE_FOLD_FREEZE', seeds=list(SEEDS),
        config_sha256=digest(RUN/'RUN_CONFIG.json'), folds={
            FOLD: {key: config[key] for key in ('selected_epoch','selected_checkpoint','selected_sha256')}
        })
    if path.exists():
        prior = read(path)
        require(all(prior[k] == v for k,v in core.items()), 'already frozen single-fold barrier changed')
    else:
        immutable(path, dict(core, freeze_unix=time.time()))
    return reference, config


def _free_memory():
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
        '--format=csv,noheader,nounits'], text=True)
    return int(out.splitlines()[0])


def export():
    import torch
    from bcep_v2.runtime import configure_cuda
    from .assets import load_fold
    from .evaluate import export_fold
    from .model import Student

    config = read(RUN/'RUN_CONFIG.json')
    require(runtime_sources() == config['source_sha256'], 'UNIV protocol source changed')
    torch.set_num_threads(1)
    configure_cuda()
    torch.cuda.set_per_process_memory_fraction(config['cuda_allocator_fraction'])
    folder = RUN / FOLD
    folder.mkdir(exist_ok=True)
    assets = load_fold(FOLD, folder, 'cuda')
    checkpoint = Path(config['selected_checkpoint'])
    require(digest(checkpoint) == config['selected_sha256'], 'selected checkpoint changed before inference')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    require(saved['completed_epoch'] == config['selected_epoch'], 'incorrect selected epoch')
    student = Student().to('cuda').eval().requires_grad_(False)
    student.load_state_dict(saved['student'], strict=True)
    barrier = RUN/'FROZEN_STUDENTS.json'
    selected = read(barrier)['folds'][FOLD]
    result = export_fold(assets, student, selected, barrier, folder/'test',
                         progress_root=RUN, benchmark=False)
    require(result['ddim_calls'] == 4735, 'incorrect test DDIM count')
    immutable(RUN/'EXPORT_COMPLETE.json', result)


def _publish():
    from .early_univ_report import build_report

    build_report(RUN, DOC)
    assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == BRANCH
    assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], text=True).strip()
    subprocess.run(['git', 'add', str(DOC.relative_to(SOURCE))], check=True)
    subprocess.run(['git', 'diff', '--cached', '--check'], check=True)
    if subprocess.call(['git', 'diff', '--cached', '--quiet']):
        subprocess.run(['git', 'commit', '-m', 'Archive locked UNIV five-seed student test'], check=True)
    subprocess.run(['git', 'push', '-u', 'origin', BRANCH], check=True)
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/'+BRANCH],
                                     text=True).split()[0]
    require(sha == remote, 'remote UNIV archive hash differs')
    write(RUN/'DELIVERY.json', dict(sha=sha, verified_remote_sha=remote, branch=BRANCH))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-worker', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.export_worker:
        export()
        return
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        training_complete = TRAIN_RUN/FOLD/'TRAINING_COMPLETE.json'
        while not training_complete.exists():
            main_run = read(TRAIN_RUN/'STATUS.json')
            status(RUN, 'WAITING_FOR_UNIV_FINAL_SELECTION',
                   main_fold=main_run.get('fold'), main_status=main_run['status'])
            if args.prepare_only:
                return
            time.sleep(15)
        reference, config = prepare()
        if args.prepare_only:
            status(RUN, 'PREPARED')
            return
        require(all(os.environ.get(k) == v for k,v in ENVIRONMENT.items()), 'use recorded CUDA environment')
        if not (RUN/'START.json').exists():
            write(RUN/'START.json', dict(unix=time.time(), command=[sys.executable,'-m','direct_k20.early_univ']))
        for attempt in range(1,4):
            if (RUN/'EXPORT_COMPLETE.json').exists():
                break
            while _free_memory() < config['min_free_gpu_mib']:
                status(RUN, 'WAITING_FOR_SHARED_GPU_MEMORY')
                time.sleep(30)
            status(RUN, 'STUDENT_TEST_EXPORT', fold=FOLD, attempt=attempt)
            cmd = [sys.executable, '-m', 'direct_k20.early_univ', '--export-worker']
            with (RUN/'export.log').open('a', buffering=1) as log:
                p = subprocess.Popen(cmd, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT)
                write(RUN/'ACTIVE_CHILD.json', dict(pid=p.pid, attempt=attempt, unix=time.time()))
                code = p.wait()
            if code == 0:
                require((RUN/'EXPORT_COMPLETE.json').exists(), 'missing exporter receipt')
                break
            status(RUN, 'RESUMING_SAME_EXPORT', attempt=attempt, exit_code=code)
            time.sleep(30)
        require((RUN/'EXPORT_COMPLETE.json').exists(), 'export interrupted after three attempts')
        import torch
        from .evaluate import score_all
        torch.set_num_threads(1)
        score_all({FOLD: reference}, RUN/'FROZEN_STUDENTS.json', run=RUN, folds=(FOLD,))
        write(RUN/'WALL_TIME.json', dict(seconds=time.time()-read(RUN/'START.json')['unix']))
        status(RUN, 'ARCHIVING_UNIV')
        _publish()
        status(RUN, 'COMPLETE', fold=FOLD, seeds=list(SEEDS), training_updates=0)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if '--export-worker' not in sys.argv:
            status(RUN, 'INTERRUPTED_RESUME_SAME_UNIV', traceback=traceback.format_exc())
        raise

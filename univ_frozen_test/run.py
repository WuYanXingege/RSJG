"""Run the existing five-seed native test on the already frozen UNIV fold."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from bcep.contracts import canonical_hash, digest
from bcep_joint.common import read, write
from joint_flow_fourfold.common import immutable, source_hashes, status
from joint_flow_fourfold.controller import check_protection

SOURCE = Path(__file__).resolve().parents[1]
ROOT = SOURCE.parent
PRIOR = ROOT/'RSJG_EBJD_assets/experiments/joint_flow_fourfold_confirmation_20261122'
RUN = ROOT/'RSJG_EBJD_assets/experiments/univ_frozen_test_20261125'
DOC = SOURCE/'docs/research/continuous_joint_flow/univ_frozen_test_20261125'
BASE = 'e20579aae45fbcf1484ec7f7665b0758de375de6'
BRANCH = 'experiment/univ-frozen-test-20261125'
FOLDS = ('univ',)
SEEDS = tuple(range(2035, 2040))
EXPECTED = {'univ': dict(A_epoch=9, A_sha256='7b381c1e8a11080ba4adce3f4459e4d493d1eb3b5690e890ac5551e0a68f9168',
    B_epoch=33, B_sha256='776235267e5cf4f1be2a9ba80cd5eaf8c44cdda5a8e94b6cde3ae3f78a24681a',
    windows=947, agents=24334, dense_windows=907, dense_agents=24099)}


def sources():
    return dict(native=source_hashes(), launcher={str(p.relative_to(SOURCE)): digest(p)
        for p in sorted((SOURCE/'univ_frozen_test').glob('*.py'))})


def protect(config=None):
    # The bound training source and its immutable configuration remain untouched.
    prior = read(PRIOR/'RUN_CONFIG.json')
    check_protection(prior)
    if config is not None:
        assert sources() == config['source_sha256'], 'test runtime source changed'
        assert digest(PRIOR/'RUN_CONFIG.json') == config['original_controller_config_sha256']
        for fold in FOLDS:
            for name in ('UPSTREAM_FROZEN.json', 'B_FROZEN.json'):
                assert digest(PRIOR/fold/name) == digest(RUN/fold/name), 'selected weights changed'


def prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    protect()
    admitted = {}
    for fold in FOLDS:
        folder = RUN/fold; folder.mkdir(exist_ok=True)
        upstream = read(PRIOR/fold/'UPSTREAM_FROZEN.json')
        selected = read(PRIOR/fold/'B_FROZEN.json')
        expected = EXPECTED[fold]
        assert upstream['fold'] == selected['fold'] == fold
        assert upstream['joint']['frozen']['epoch'] == expected['A_epoch']
        assert selected['rank1_epoch'] == expected['B_epoch']
        assert upstream['joint']['sha256'] == expected['A_sha256']
        assert selected['rank1_sha256'] == expected['B_sha256']
        assert selected['test_access'] is False and selected['completed_epochs'] >= 30
        for stage in ('density', 'head', 'joint'):
            asset = upstream[stage]
            assert asset['frozen']['trained'] and asset['frozen']['frozen']
            assert digest(asset['checkpoint']) == asset['sha256']
        assert digest(selected['rank1_checkpoint']) == expected['B_sha256']
        split = read(upstream['split_path'])
        assert digest(upstream['split_path']) == upstream['split_sha256']
        ids = split['heldout_confirmation_indices']
        counts = [len(split['members'][i]['agent_ids']) for i in ids]
        assert len(counts) == expected['windows'] and sum(counts) == expected['agents']
        assert sum(n >= 9 for n in counts) == expected['dense_windows']
        assert sum(n for n in counts if n >= 9) == expected['dense_agents']
        # Integrity only: hashing compressed files does not materialize future arrays.
        for index in ids:
            record = split['records'][index]
            assert digest(record['path']) == record['sha256']
        immutable(folder/'UPSTREAM_FROZEN.json', upstream)
        immutable(folder/'B_FROZEN.json', selected)
        admitted[fold] = dict(A_checkpoint=upstream['joint']['checkpoint'], A_sha256=expected['A_sha256'],
            A_epoch=expected['A_epoch'], A_selection='rank1 dev H-JADE',
            B_checkpoint=selected['rank1_checkpoint'], B_sha256=expected['B_sha256'],
            B_epoch=expected['B_epoch'], B_selection='rank1 dev JADE seed2035',
            density_sha256=upstream['density']['sha256'], head_sha256=upstream['head']['sha256'],
            upstream_manifest_sha256=digest(folder/'UPSTREAM_FROZEN.json'),
            B_manifest_sha256=digest(folder/'B_FROZEN.json'), counts=expected)
    amendment = dict(date='2026-11-25', authorized_folds=list(FOLDS),
        user_instruction='开启UNIV的测试集测评',
        previous_order='wait for all four new-fold A/B weights to freeze before any new-fold test',
        effective_order='test the already frozen UNIV A/B now; ETH training continues with its original fixed recipe and selector',
        original_controller_source_or_config_modified=False, additional_training_updates=0,
        predictions='per fold, seal all five seeds A/B before reading any heldout future',
        later_controller_reuse='independent test directory; original controller remains unchanged')
    config = dict(schema='univ-frozen-five-seed-test-v1', base_commit=BASE,
        folds=list(FOLDS), seeds=list(SEEDS), models=EXPECTED,
        original_controller_config_sha256=digest(PRIOR/'RUN_CONFIG.json'),
        source_sha256=sources(), authorization_amendment=amendment,
        dataset='official heldout_confirmation; 8 observed / 12 predicted frames, dt=0.4s, meters',
        inference=dict(M=256, K=20, ddim_steps=20, eta=0., calibration=False, scene_swap=False, trajectory_repair=False),
        A='unchanged joint_flow_fourfold.test native H0 entry', B='unchanged joint_flow_dependency.inference.generate',
        device='cuda', cuda_execution='bcep_v2.runtime.configure_cuda deterministic FP32',
        torch_threads=1, cuda_allocator_fraction=0.35, concurrency='one test worker under auxiliary_gpu.lock; share existing GPU; >=6GiB free at start',
        training_updates=0, model_selection_on_test=False,
        joint_aggregation='equal windows', marginal_aggregation='valid-agent weighted',
        seed_statistics='five independent K20 evaluations; mean and sample standard deviation, ddof=1; paired B-minus-A SD',
        empty_dense_group='N/A; never replace with zero')
    immutable(RUN/'RUN_CONFIG.json', config)
    immutable(RUN/'AUTHORIZATION_AMENDMENT.json', amendment)
    barrier = dict(status='ALL_AVAILABLE_MODELS_FROZEN_BEFORE_TEST', requested_folds=list(FOLDS),
        folds=admitted, missing_folds={}, completed_fold_count=1, scope='UNIV authorized for immediate testing',
        source_config_sha256=digest(RUN/'RUN_CONFIG.json'), authorization_amendment_sha256=digest(RUN/'AUTHORIZATION_AMENDMENT.json'))
    if (RUN/'FROZEN_MODELS.json').exists():
        old = read(RUN/'FROZEN_MODELS.json')
        assert {k:v for k,v in old.items() if k != 'freeze_unix'} == barrier
        barrier = old
    else:
        barrier['freeze_unix'] = time.time()
        immutable(RUN/'FROZEN_MODELS.json', barrier)
    protect(config)
    return config, barrier


def worker(fold):
    import torch
    from bcep_v2.runtime import configure_cuda
    from joint_flow_fourfold.test import run_test
    config = read(RUN/'RUN_CONFIG.json'); protect(config)
    torch.set_num_threads(config['torch_threads']); configure_cuda()
    torch.cuda.set_per_process_memory_fraction(config['cuda_allocator_fraction'])
    with (RUN.parent/'auxiliary_gpu.lock').open('a') as auxiliary_lock:
        fcntl.flock(auxiliary_lock,fcntl.LOCK_EX)
        while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0]) < 6144:
            status(RUN,'WAITING_AUXILIARY_MEMORY',fold=fold);time.sleep(45)
        result = run_test(read(RUN/fold/'UPSTREAM_FROZEN.json'), read(RUN/fold/'B_FROZEN.json'),
                      read(RUN/'FROZEN_MODELS.json'), RUN/fold/'test', 'cuda')
    protect(config)
    write(RUN/fold/'TEST_RESULT.json', result)


def archive(config, push):
    from .report import build_report
    protect(config); build_report(RUN, DOC)
    if not push:return
    assert subprocess.check_output(['git','branch','--show-current'],cwd=SOURCE,text=True).strip() == BRANCH
    assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=SOURCE,text=True).strip()
    subprocess.run(['git','add',str(DOC.relative_to(SOURCE))],cwd=SOURCE,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=SOURCE,check=True)
    if subprocess.call(['git','diff','--cached','--quiet'],cwd=SOURCE):
        subprocess.run(['git','commit','-m','Archive frozen UNIV five-seed paired test results'],cwd=SOURCE,check=True)
    subprocess.run(['git','push','-u','origin',BRANCH],cwd=SOURCE,check=True)
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
    remote = subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=SOURCE,text=True).split()[0]
    assert head == remote
    write(RUN/'DELIVERY.json',dict(sha=head,branch=BRANCH,verified_remote_sha=head,
        report=f'https://github.com/WuYanXingege/RSJG/blob/{head}/{DOC.relative_to(SOURCE)}/README.md',
        summary=f'https://github.com/WuYanXingege/RSJG/blob/{head}/{DOC.relative_to(SOURCE)}/SUMMARY.csv',unix=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--worker',choices=FOLDS)
    parser.add_argument('--push-on-complete',action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker); return
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config, barrier = prepare()
        if args.prepare_only:
            status(RUN,'PREPARED',models=EXPECTED); return
        (RUN/'logs').mkdir(exist_ok=True)
        for fold in FOLDS:
            if (RUN/fold/'TEST_RESULT.json').exists():continue
            for attempt in range(1,4):
                protect(config)
                assert shutil.disk_usage(RUN).free > 5*1024**3
                status(RUN,'TESTING_FROZEN_MODELS',fold=fold,attempt=attempt)
                with (RUN/'logs'/f'{fold}.log').open('a',buffering=1) as log:
                    log.write(f'\nATTEMPT {attempt} same frozen test unix={time.time()}\n')
                    p = subprocess.Popen([sys.executable,'-m','univ_frozen_test.run','--worker',fold],
                        cwd=SOURCE,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                    write(RUN/'ACTIVE_CHILD.json',dict(pid=p.pid,fold=fold,attempt=attempt,unix=time.time()))
                    code = p.wait()
                if code == 0 and (RUN/fold/'TEST_RESULT.json').exists():break
                if attempt == 3:raise RuntimeError(f'{fold} test failed three same-configuration attempts')
                status(RUN,'RECOVERING_SAME_TEST',fold=fold,attempt=attempt,exit_code=code); time.sleep(30)
        status(RUN,'ARCHIVING_COMPLETE_TESTS')
        archive(config,args.push_on_complete)
        status(RUN,'COMPLETE',folds=list(FOLDS),seeds=list(SEEDS),training_updates=0)


if __name__ == '__main__':
    try:main()
    except BaseException:
        RUN.mkdir(parents=True,exist_ok=True)
        status(RUN,'INTERRUPTED_RESUME_SAME_TEST',traceback=traceback.format_exc())
        raise

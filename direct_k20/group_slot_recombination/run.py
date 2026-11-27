"""Run the locked HOTEL/UNIV train and development slot recombination study."""

import argparse
import csv
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from bcep.contracts import require
from direct_k20.common import ASSETS, ENVIRONMENT, METRICS, SOURCE, digest, immutable, read, runtime_sources, status, write

from .data import BASE, DEV, DOC, FOLDS, RUN, SEED


BRANCH = 'experiment/group-slot-recombination-20261127'
YIELD_CODE = 75


def _sources():
    return runtime_sources() | {str(p.relative_to(SOURCE)):digest(p)
                                for p in sorted(Path(__file__).parent.glob('*.py'))}


def _refs_and_cache():
    from .data import DevelopmentCache, dev_reference

    refs = {fold:dev_reference(fold) for fold in FOLDS}
    return refs, {fold:DevelopmentCache(ref,RUN) for fold,ref in refs.items()}


def prepare():
    from .data import open_training_cache
    from .train import RECIPE

    refs, dev = _refs_and_cache()
    folds = {}
    for fold, ref in refs.items():
        train = open_training_cache(ref,RUN)
        identity = ref.identity
        folds[fold] = dict(
            fold=fold, S0_epoch=identity['S0_epoch'], S0_sha256=identity['S0_sha256'],
            S0_checkpoint=identity['S0_checkpoint'], split_sha256=identity['split_sha256'],
            development_seed=SEED, train=train.identity, dev=dev[fold].identity,
            train_windows=len(train.indices), dev_windows=len(ref.indices), dev_agents=identity['dev_agents'],
            S0_dev_reference=identity['S0_dev_reference'], train_recipe=RECIPE,
        )
    config = dict(
        base_commit=BASE, folds=list(FOLDS), purpose=DEV, test_access=False,
        seeds=[SEED], freeze_s0=True, S0_updates=0, A_updates=0, B_updates=0,
        K=20, prediction_frames=12, dt=.4,
        groups='original diagnostic greedy disjoint pairs at <=2 m mean distance over last 3 observations',
        matcher=dict(hidden_width=64, final_weight_and_bias=0., diagonal_score_bias=.1,
                     inference='maximum Hungarian, one permutation per group, intact 12-frame gather'),
        training=RECIPE,
        retention='every completed epoch and last; only epoch 12 is eligible for one fixed development score',
        gates=dict(UNIV_N9plus_min_JFDE_reduction_m=.010, no_regression_tolerance_m=1e-6,
                   checked_JADE_JFDE_groups=['univ/all','hotel/all','hotel/N2_8'],
                   exact_per_person_cover=True),
        inputs=folds, source_sha256=_sources(), environment=ENVIRONMENT,
        reproduce='python -m direct_k20.group_slot_recombination.data && python -m direct_k20.group_slot_recombination.run',
    )
    immutable(RUN/'RUN_CONFIG.json',config)
    return config,refs,dev


def _device():
    lines = subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,memory.free,memory.total',
                                    '--format=csv,noheader,nounits'],text=True).splitlines()
    [(uuid, free, total)] = csv.reader(lines,skipinitialspace=True)
    return dict(uuid=uuid,free_mib=int(free),total_mib=int(total))


def _train_worker(fold):
    from .data import dev_reference, open_training_cache, queries, setup_device
    from .train import train

    stop_requested = False
    parent = os.getppid()

    def stop(_signal, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGUSR1,stop)
    config = read(RUN/'RUN_CONFIG.json')
    require(config['source_sha256'] == _sources(), 'locked train source changed')
    ref = dev_reference(fold)
    cache = open_training_cache(ref,RUN)
    require(cache.identity == config['inputs'][fold]['train'], 'frozen train cache changed')
    setup_device('cuda')
    result = train(cache,queries(ref),RUN/fold/'training',config['inputs'][fold],device='cuda',
                   should_yield=lambda:stop_requested or os.getppid()!=parent)
    if result is None:
        return YIELD_CODE
    require(result['completed_epochs'] == 12
            and digest(result['final_checkpoint']) == result['final_sha256'], 'final matcher is incomplete')
    immutable(RUN/fold/'TRAINING_COMPLETE.json',result)
    return 0


def _execute_train(fold):
    path = RUN/fold/'TRAINING_COMPLETE.json'
    if path.exists():
        return
    command = [sys.executable,'-m','direct_k20.group_slot_recombination.run','--worker',fold]
    with (ASSETS/'auxiliary_gpu.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            status(RUN,'WAITING_FOR_GPU_LOCK',fold=fold)
            fcntl.flock(lock,fcntl.LOCK_EX)
        for attempt in range(1,4):
            gpu = _device()
            while gpu['free_mib'] < 4000:
                status(RUN,'WAITING_FOR_GPU_MEMORY',fold=fold,gpu=gpu)
                time.sleep(30)
                gpu = _device()
            started = time.time()
            with (RUN/f'{fold}_train.log').open('a',buffering=1) as log:
                child = subprocess.Popen(command,cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT,
                                         pass_fds=(lock.fileno(),))
                write(RUN/'ACTIVE_CHILD.json',dict(fold=fold,pid=child.pid,gpu=gpu,attempt=attempt,unix=started,
                                                  command=command))
                status(RUN,'TRAINING_MATCHER',fold=fold,pid=child.pid,gpu=gpu)
                code = child.wait()
            if code == 0:
                require(path.exists(),f'{fold} worker omitted final receipt')
                return
            status(RUN,'RECOVERING_SAME_TRAINING',fold=fold,attempt=attempt,exit_code=code)
            if attempt < 3:
                time.sleep(15)
    raise RuntimeError(f'{fold} matching training interrupted; see {fold}_train.log')


def _fixed_matches(config,refs,dev):
    import torch

    from .data import queries,setup_device
    from .evaluate import export_fold
    from .matcher import Matcher

    repair = {}
    for fold in FOLDS:
        ref = refs[fold]
        invalid = any(dev[fold]._source_invalid(i) for i in ref.indices)
        if invalid:
            with (ASSETS/'auxiliary_gpu.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                setup_device('cuda')
                repair[fold] = dev[fold].repair_if_needed('cuda')
                torch.cuda.empty_cache()
        else:
            repair[fold] = dev[fold].repair_if_needed('cpu')
    immutable(RUN/'DEV_PATH_REPLAY.json',repair)
    for fold in FOLDS:
        trained = read(RUN/fold/'TRAINING_COMPLETE.json')
        path = trained['final_checkpoint']
        require(trained['completed_epochs'] == 12 and digest(path) == trained['final_sha256'],
                'only the last complete checkpoint can produce dev results')
        saved = torch.load(path,map_location='cpu',weights_only=False)
        require(saved['completed_epoch'] == 12, 'wrong matching epoch')
        model = Matcher()
        model.load_state_dict(saved['matcher'],strict=True)
        model.eval().requires_grad_(False)
        identity = config['inputs'][fold] | dict(
            final_checkpoint_sha256=trained['final_sha256'], final_epoch=12,
            config_sha256=digest(RUN/'RUN_CONFIG.json'), device='cpu', test_access=False)
        immutable(RUN/fold/'DEVELOPMENT_IDENTITY.json',identity)
        status(RUN,'SEALING_DEVELOPMENT_MATCHES',fold=fold)
        result = export_fold(fold=fold,device='cpu',matcher=model,queries=queries(refs[fold]),
                             dev_reference=refs[fold],read_endpoints=dev[fold].read_endpoints,
                             read_path=dev[fold].read_path,output=RUN,identity=identity)
        immutable(RUN/fold/'DEV_EXPORT_COMPLETE.json',result)
    return repair


def _publish():
    from joint_flow_fourfold.report import public_metadata
    from .data import open_training_cache

    DOC.mkdir(parents=True,exist_ok=True)
    refs,_ = _refs_and_cache()
    for fold in FOLDS:
        folder = DOC/fold
        folder.mkdir(exist_ok=True)
        for path in (RUN/fold/'TRAINING_COMPLETE.json',RUN/fold/'training/checkpoints/TOP5.json',
                     RUN/fold/'DEVELOPMENT_IDENTITY.json', RUN/fold/'DEV_EXPORT_COMPLETE.json'):
            write(folder/path.name,public_metadata(read(path)))
        ready = RUN/fold/'train_cache/SEALED.json'
        cache = read(ready)
        public = {k:v for k,v in cache.items() if k!='rows'}
        write(folder/'TRAIN_CACHE_RECEIPT.json',public|dict(sha256=digest(ready),
              members=len(open_training_cache(refs[fold],RUN).indices)))
    for name in ('RUN_CONFIG.json','WALL_TIME.json','START.json','DEV_PATH_REPLAY.json'):
        write(DOC/name,public_metadata(read(RUN/name)))
    write(DOC/'ARTIFACT_SHA256.json',{str(path.relative_to(DOC)):digest(path)
          for path in sorted(DOC.rglob('*')) if path.is_file() and path.name!='ARTIFACT_SHA256.json'})
    require(subprocess.check_output(['git','branch','--show-current'],text=True).strip()==BRANCH,
            'archive is outside the experiment branch')
    require(not subprocess.check_output(['git','diff','--cached','--name-only'],text=True).strip(),
            'unexpected staged files')
    subprocess.run(['git','add',str(DOC.relative_to(SOURCE))],check=True)
    subprocess.run(['git','diff','--cached','--check'],check=True)
    if subprocess.call(['git','diff','--cached','--quiet']):
        subprocess.run(['git','commit','-m','Archive train-only two-fold slot recombination'],check=True)
    subprocess.run(['git','push','-u','origin',BRANCH],check=True)
    sha = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    remote = subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],text=True).split()[0]
    require(sha==remote,'GitHub branch differs from local archive')
    write(RUN/'DELIVERY.json',dict(sha=sha,verified_remote_sha=remote,branch=BRANCH))
    status(RUN,'COMPLETE',folds=list(FOLDS),test_access=False,sha=sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',choices=FOLDS)
    parser.add_argument('--prepare-only',action='store_true')
    args = parser.parse_args()
    if args.worker:
        return _train_worker(args.worker)
    require(all(os.environ.get(key)==value for key,value in ENVIRONMENT.items()),'wrong launch environment')
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config,refs,dev = prepare()
        if args.prepare_only:
            return 0
        if (RUN/'DELIVERY.json').exists():
            return 0
        if not (RUN/'START.json').exists():
            write(RUN/'START.json',dict(unix=time.time(),command=[sys.executable,'-m',
                  'direct_k20.group_slot_recombination.run'],
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
        for fold in FOLDS:
            _execute_train(fold)
        import torch
        from .evaluate import score_all
        from .train import record_final_metrics

        torch.set_num_threads(1)
        require(_sources() == config['source_sha256'], 'execution source changed')
        started = time.monotonic()
        repair = _fixed_matches(config,refs,dev)
        status(RUN,'SCORING_SEALED_DEVELOPMENT')
        runtime = dict(development_path_replay=repair,
                       training={fold:read(RUN/fold/'TRAINING_COMPLETE.json') for fold in FOLDS},
                       total_started_unix=read(RUN/'START.json')['unix'],
                       dev_started_monotonic=started)
        score = score_all(refs=refs,output=RUN,doc=DOC,
                          read_paths={fold:dev[fold].read_path for fold in FOLDS},runtime=runtime)
        final = {row['fold']:row for row in score['groups'] if row['group']=='all'}
        for fold in FOLDS:
            record_final_metrics(RUN/fold/'training',{metric:final[fold][f'new_{metric}_m'] for metric in METRICS})
        write(RUN/'WALL_TIME.json',dict(seconds=time.time()-read(RUN/'START.json')['unix'],
              dev_seconds=time.monotonic()-started,
              endpoint_cache_seconds={fold:read(RUN/fold/'train_cache/SEALED.json')['seconds'] for fold in FOLDS}))
        status(RUN,'ARCHIVING')
        _publish()
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        if '--worker' not in sys.argv:
            status(RUN,'INTERRUPTED_RESUME_SAME_RUN',traceback=traceback.format_exc())
        raise

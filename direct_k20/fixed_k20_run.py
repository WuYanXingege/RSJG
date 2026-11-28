"""Run the fixed single-candidate K20 experiment and its per-fold test gates."""

import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from bcep.contracts import require
from .common import ASSETS, ENVIRONMENT, METRICS, SEEDS, SOURCE, digest, immutable, read, runtime_sources, status, write
from .fixed_k20_inputs import BASE, DOC, FOLDS, RUN, fixed_inputs, open_test_baseline
from .interaction_run import _gpu, _load


BRANCH = 'experiment/fixed-k20-single-candidate-20261128'
CV_DEV = 'cv_development'
CENTER_DEV = 'center_development'
SEED = 2035
YIELD = 75


def _prepare():
    from .fixed_k20_center import RECIPE

    selections, refs, dev = fixed_inputs()
    config = dict(
        base_commit=BASE, folds=list(FOLDS), inputs=selections, source_sha256=runtime_sources(),
        precision='FP32', K=20, original_slots_per_predict=20, retained_S0_slots=19,
        new_slots=1, diagnostics_K=21, frames=12, observed_frames=8, dt=.4,
        role='19 unchanged complete S0 trajectories plus one complete deterministic candidate',
        slot_selection='minimize max of original train relative JFDE and minFDE deletion increases; ties earlier slot',
        CV='mean velocity between observations -4 and -1, extrapolated for the full 12 frames',
        train_center_only_if_univ_cv_fails=True, center_recipe=RECIPE,
        center_scales_m={name:selections['univ']['scales_m'][name] for name in ('minADE','minFDE')},
        development_seed=SEED, test_seeds=list(SEEDS), tolerance_m=1e-6,
        thresholds=dict(HOTEL_target_all_JFDE_delta_m=-.01,
            HOTEL_guards=['all/JADE','N2_8/JADE','N2_8/JFDE','all/minADE','all/minFDE'],
            UNIV_target_N9plus_JFDE_delta_m=-.01,
            UNIV_guards=['all/JADE','all/JFDE','N9plus/JADE','all/minADE','all/minFDE','N9plus/minADE','N9plus/minFDE']),
        test='exploratory frozen paired replay only for each passed fold',
        device=dict(center_train_and_inference='cuda',CV_and_scoring='cpu',single_worker=True),
        environment=ENVIRONMENT, reproduce='python -m direct_k20.fixed_k20_run',
    )
    immutable(RUN/'RUN_CONFIG.json',config)
    return config,refs,dev


def _rank(refs):
    from .fixed_k20_slot import rank_fold
    from .group_slot_recombination.data import RUN as REGROUP, prepare_training

    choices = {}
    for fold,ref in refs.items():
        rank = RUN/fold/'DELETION_RANKING.json'
        if not rank.exists():
            status(RUN,'RANKING_TRAIN_DELETION_SLOTS',fold=fold)
            original = REGROUP/fold/'train_cache/SEALED.json'
            cache_root = REGROUP if original.exists() else RUN/'native_endpoint_cache'
            if not original.exists():
                from .group_slot_recombination.data import setup_device
                with (ASSETS/'auxiliary_gpu.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX)
                    setup_device('cuda')
                    prepare_training(ref,cache_root,'cuda')
            rank_fold(ref,RUN,cache_root=cache_root)
        result = read(rank)
        require(result['fold']==fold and 0<=result['j_drop']<20
                and result['train_windows']==len(ref.split['train_indices'])
                and result['cache_identity']['S0_sha256']==ref.identity['S0_sha256']
                and result['development_truth_reads']==result['test_truth_reads']==0
                and result['csv_sha256']==digest(rank.with_suffix('.csv')),
                'invalid deleted slot receipt')
        choices[fold] = dict(j_drop=result['j_drop'],train_deletion_sha256=digest(rank))
    immutable(RUN/'DELETIONS_FROZEN.json',dict(purpose='train',folds=choices,tie_rule='smallest native slot index'))
    return choices


def _make_dev_identity(config, choices, fold, method, phase):
    source = config['inputs'][fold]
    checkpoint = None
    checkpoint_sha = None
    if method=='center':
        fixed = read(RUN/'univ'/'CENTER_TRAINING_COMPLETE.json')
        checkpoint = fixed['final_checkpoint']
        checkpoint_sha = fixed['final_sha256']
        require(fold=='univ' and fixed['completed_epochs']==12 and digest(checkpoint)==checkpoint_sha,
                'center final checkpoint does not match the fixed candidate')
    return source | choices[fold] | dict(
        phase=phase, method=method, candidate_checkpoint=checkpoint, checkpoint_sha256=checkpoint_sha,
        config_sha256=digest(RUN/'RUN_CONFIG.json'), frozen_deleted_slots_sha256=digest(RUN/'DELETIONS_FROZEN.json'),
        seed=SEED,
    )


def _center_training_identity(config, choices):
    candidate = RUN/CV_DEV/'SCORE_RECEIPT.json'
    result = read(candidate)
    require(result['decision']['univ']['passed'] is False,'center training is only for failed UNIV CV')
    return config['inputs']['univ'] | choices['univ'] | dict(
        cv_receipt_sha256=digest(candidate),cv_passed=False,
        config_sha256=digest(RUN/'RUN_CONFIG.json'))


def _dev_path(dev,index,seed):
    require(seed==SEED,'development inference seed changed')
    return dev.read_path(index)


def _export(ref,dev,identity,predict_new=None):
    from .fixed_k20_eval import export_fold

    if identity['phase']=='test':
        read_base = ref.read_path
    else:
        def read_base(index,seed):
            return _dev_path(dev,index,seed)
    status(RUN,'SEALING_K20_PREDICTIONS',phase=identity['phase'],fold=ref.fold,method=identity['method'])
    outcome = export_fold(ref,RUN,identity,read_base,predict_new=predict_new)
    immutable(RUN/identity['phase']/ref.fold/'EXPORT_COMPLETE.json',outcome)
    return outcome


def _worker(task):
    import torch
    from bcep_v2.runtime import configure_cuda
    from .fixed_k20_center import load_center, predict_center, train_center

    config = read(RUN/'RUN_CONFIG.json')
    require(config['source_sha256']==runtime_sources(),'fixed experiment source changed')
    torch.set_num_threads(2)
    configure_cuda()
    torch.cuda.set_per_process_memory_fraction(.45)
    device = torch.device('cuda')
    _,refs,dev = fixed_inputs()
    choices = read(RUN/'DELETIONS_FROZEN.json')['folds']
    parent = os.getppid()
    yielding = False

    def yield_worker(_sig,_frame):
        nonlocal yielding
        yielding = True

    signal.signal(signal.SIGUSR1,yield_worker)
    if task=='center_train':
        identity = _center_training_identity(config,choices)
        assets,_ = _load('univ',identity,device)
        receipt = train_center(assets,RUN/'univ'/'center_training',identity,config['center_scales_m'],
                      device='cuda',should_yield=lambda:yielding or os.getppid()!=parent)
        if receipt is None:
            return YIELD
        require(receipt['completed_epochs']==12 and digest(receipt['final_checkpoint'])==receipt['final_sha256'],
                'partial center run cannot complete')
        immutable(RUN/'univ'/'CENTER_TRAINING_COMPLETE.json',receipt)
    elif task=='repair_dev':
        repair = {fold:dev[fold].repair_if_needed('cuda') for fold in FOLDS}
        immutable(RUN/'DEV_S0_REPLAY.json',repair)
    elif task=='center_dev':
        identity = _make_dev_identity(config,choices,'univ','center',CENTER_DEV)
        assets,_ = _load('univ',identity,device)
        trained = load_center(identity['candidate_checkpoint'],identity['checkpoint_sha256'],device)
        _export(refs['univ'],dev['univ'],identity,
                predict_new=lambda batch:predict_center(assets,trained,batch.to(device)))
    else:
        freeze = read(RUN/'FROZEN_TEST.json')
        require(task.startswith('test_') and task[5:] in freeze['folds'],'invalid frozen test worker')
        fold = task[5:]
        identity = freeze['folds'][fold]
        ref = open_test_baseline(fold,refs[fold],identity,freeze)
        absent = ref.missing_pairs()
        repaired = ref.repair(absent)
        immutable(RUN/'test'/fold/'S0_REPLAY.json',repaired)
        predict = None
        if identity['method']=='center':
            assets,_ = _load(fold,identity,device)
            trained = load_center(identity['candidate_checkpoint'],identity['checkpoint_sha256'],device)
            def predict(batch):
                return predict_center(assets,trained,batch.to(device))
        _export(ref,None,identity,predict_new=predict)
    return 0


def _execute(task,receipt):
    if receipt.is_file():
        return
    command = [sys.executable,'-m','direct_k20.fixed_k20_run','--worker',task]
    with (ASSETS/'auxiliary_gpu.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            status(RUN,'WAITING_FOR_GPU_LOCK',task=task)
            fcntl.flock(lock,fcntl.LOCK_EX)
        for attempt in range(1,4):
            device = _gpu()
            while device['free_mib'] < 6000:
                status(RUN,'WAITING_FOR_GPU_MEMORY',task=task,gpu=device)
                time.sleep(30)
                device = _gpu()
            started = time.time()
            with (RUN/f'{task}.log').open('a',buffering=1) as log:
                child = subprocess.Popen(command,cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),))
                write(RUN/'ACTIVE_CHILD.json',dict(task=task,pid=child.pid,attempt=attempt,gpu=device,unix=started))
                status(RUN,'FIXED_K20_GPU_WORKER_RUNNING',task=task,pid=child.pid,gpu=device)
                code = child.wait()
            if code==0:
                require(receipt.is_file(),f'{task} did not report completion')
                return
            status(RUN,'RECOVERING_FIXED_K20_STAGE',task=task,attempt=attempt,exit_code=code)
            if attempt<3:
                time.sleep(10)
    raise RuntimeError(f'{task} stopped repeatedly; see {task}.log')


def _dev_and_test(config,refs,dev,choices):
    from .fixed_k20_center import record_final_metrics
    from .fixed_k20_eval import score_phase

    missing = any(cache._source_invalid(index) for cache in dev.values() for index in cache.ref.indices)
    if missing:
        _execute('repair_dev',RUN/'DEV_S0_REPLAY.json')
    else:
        immutable(RUN/'DEV_S0_REPLAY.json',{fold:dev[fold].repair_if_needed('cpu') for fold in FOLDS})
    identities = {fold:_make_dev_identity(config,choices,fold,'cv',CV_DEV) for fold in FOLDS}
    readers = {}
    for fold in FOLDS:
        if not (RUN/CV_DEV/fold/'EXPORT_COMPLETE.json').exists():
            _export(refs[fold],dev[fold],identities[fold])
        readers[fold] = lambda index,seed,current=dev[fold]:_dev_path(current,index,seed)
    status(RUN,'SCORING_CV_DEVELOPMENT')
    cv = score_phase(RUN,{fold:(refs[fold],identities[fold],readers[fold]) for fold in FOLDS},phase=CV_DEV)
    eligible = {}
    for fold in FOLDS:
        if cv['decision'][fold]['passed']:
            eligible[fold] = identities[fold]
    if not cv['decision']['univ']['passed']:
        _execute('center_train',RUN/'univ'/'CENTER_TRAINING_COMPLETE.json')
        _execute('center_dev',RUN/CENTER_DEV/'univ'/'EXPORT_COMPLETE.json')
        center_id = _make_dev_identity(config,choices,'univ','center',CENTER_DEV)
        status(RUN,'SCORING_CENTER_DEVELOPMENT')
        center = score_phase(RUN,{'univ':(refs['univ'],center_id,readers['univ'])},phase=CENTER_DEV)
        summary = center['groups']['univ']['all']
        record_final_metrics(RUN/'univ'/'center_training',{m:summary[f'final_{m}_m'] for m in METRICS})
        if center['decision']['univ']['passed']:
            eligible['univ'] = center_id
    if not eligible:
        return []
    frozen = {fold:value | dict(phase='test', dev_score_phase=value['phase'],
        dev_score_sha256=digest(RUN/value['phase']/'SCORE_RECEIPT.json'),seeds=list(SEEDS))
        for fold,value in eligible.items()}
    freeze_path = RUN/'FROZEN_TEST.json'
    core = dict(folds=frozen,seeds=list(SEEDS),config_sha256=digest(RUN/'RUN_CONFIG.json'),
                source_sha256=config['source_sha256'],test_scope='exploratory frozen paired replay')
    if freeze_path.exists():
        hold = read(freeze_path)
        require(all(hold[key]==value for key,value in core.items()),'previous test decision changed')
    else:
        hold = core | dict(freeze_unix=time.time())
        immutable(freeze_path,hold)
    entries = {}
    for fold,identity in frozen.items():
        ref = open_test_baseline(fold,refs[fold],identity,hold)
        directory = RUN/'test'/fold
        if identity['method']=='center' or ref.missing_pairs():
            _execute(f'test_{fold}',directory/'EXPORT_COMPLETE.json')
        else:
            directory.mkdir(parents=True,exist_ok=True)
            immutable(directory/'S0_REPLAY.json',dict(new_DDIM_calls=0,seconds=0.,windows=0))
            if not (directory/'EXPORT_COMPLETE.json').exists():
                _export(ref,None,identity)
        entries[fold] = ref,identity,ref.read_path
    status(RUN,'SCORING_FROZEN_TEST',folds=list(frozen))
    score_phase(RUN,entries,phase='test')
    return list(frozen)


def _publish(tested):
    from joint_flow_fourfold.report import public_metadata
    from .fixed_k20_eval import build_report

    DOC.mkdir(parents=True,exist_ok=True)
    build_report(RUN,DOC)
    for fold in FOLDS:
        folder = DOC/fold
        folder.mkdir(exist_ok=True)
        for name in ('DELETION_RANKING.json','DELETION_RANKING.csv'):
            path = RUN/fold/name
            if path.suffix=='.csv':
                (folder/name).write_bytes(path.read_bytes())
            else:
                write(folder/name,public_metadata(read(path)))
    if (RUN/'univ'/'CENTER_TRAINING_COMPLETE.json').exists():
        folder = DOC/'univ'/'center_training'
        folder.mkdir(parents=True,exist_ok=True)
        for path in (
            RUN/'univ'/'CENTER_TRAINING_COMPLETE.json',
            RUN/'univ'/'center_training'/'checkpoints'/'TOP5.json',
            RUN/'univ'/'center_training'/'EPOCH_METRICS.json',
            RUN/'univ'/'center_training'/'PREFLIGHT.json',
        ):
            if path.is_file():
                write(folder/path.name,public_metadata(read(path)))
        cached = RUN/'univ'/'center_training'/'frozen_train'/'COMPLETE.json'
        if cached.exists():
            cache = read(cached)
            public = {k:v for k,v in cache.items() if k!='rows'}
            write(folder/'FROZEN_TRAIN_FEATURES.json',public_metadata(public|dict(manifest_sha256=digest(cached))))
    for name in ('RUN_CONFIG.json','WALL_TIME.json','START.json','DEV_S0_REPLAY.json',
                 'FROZEN_TEST.json','DELETIONS_FROZEN.json'):
        if (RUN/name).exists():
            write(DOC/name,public_metadata(read(RUN/name)))
    write(DOC/'ARTIFACT_SHA256.json',{str(p.relative_to(DOC)):digest(p)
          for p in sorted(DOC.rglob('*')) if p.is_file() and p.name!='ARTIFACT_SHA256.json'})
    require(subprocess.check_output(['git','branch','--show-current'],text=True).strip()==BRANCH,
            'cannot archive outside the fixed K20 experiment branch')
    require(not subprocess.check_output(['git','diff','--cached','--name-only'],text=True).strip(),
            'unexpected staged source files')
    subprocess.run(['git','add',str(DOC.relative_to(SOURCE))],check=True)
    subprocess.run(['git','diff','--cached','--check'],check=True)
    if subprocess.call(['git','diff','--cached','--quiet']):
        subprocess.run(['git','commit','-m','Archive fixed K20 single-candidate development and frozen tests'],check=True)
    subprocess.run(['git','push','-u','origin',BRANCH],check=True)
    sha = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    remote = subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],text=True).split()[0]
    require(sha==remote,'fixed K20 archive remote SHA differs')
    write(RUN/'DELIVERY.json',dict(sha=sha,verified_remote_sha=remote,branch=BRANCH))
    status(RUN,'COMPLETE',tested_folds=tested,sha=sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker')
    args = parser.parse_args()
    if args.worker:
        return _worker(args.worker)
    require(all(os.environ.get(k)==v for k,v in ENVIRONMENT.items()),'wrong fixed K20 environment')
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config,refs,dev = _prepare()
        if (RUN/'DELIVERY.json').exists():
            return 0
        if not (RUN/'START.json').exists():
            write(RUN/'START.json',dict(unix=time.time(),pid=os.getpid(),
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                command=[sys.executable,'-m','direct_k20.fixed_k20_run']))
        import torch
        torch.set_num_threads(1)
        choices = _rank(refs)
        tested = _dev_and_test(config,refs,dev,choices)
        require(config['source_sha256']==runtime_sources(),'fixed experiment source changed')
        write(RUN/'WALL_TIME.json',dict(seconds=time.time()-read(RUN/'START.json')['unix']))
        status(RUN,'ARCHIVING_FIXED_K20_RESULTS',tested_folds=tested)
        _publish(tested)
    return 0


if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception:
        if '--worker' not in sys.argv:
            status(RUN,'INTERRUPTED_RESUME_SAME_RUN',traceback=traceback.format_exc())
        raise

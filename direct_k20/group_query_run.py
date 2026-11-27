"""Train the fixed group-query variants and stop after paired development scoring."""

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

from bcep.contracts import canonical_hash, digest, require
from .common import ASSETS, DOC as ORIGINAL_DOC, ENVIRONMENT, METRICS, RUN as ORIGINAL_RUN, SOURCE
from .common import immutable, read, runtime_sources, status, write
from .group_slot_recombination.data import DevelopmentCache, dev_reference
from .interaction_run import _fold_identity, _gpu, _load


BASE = 'b0c2b2bc3ad10d1ae8786db516fc43d64dd38467'
RUN = ASSETS / 'group_query_20261127'
DOC = ORIGINAL_DOC.parent / 'group_query_20261127'
BRANCH = 'experiment/group-query-generation-20261127'
FOLDS = ('hotel', 'univ')
SEED = 2035
GPU_FRACTION = .82
GPU_WAIT_MIB = 12500
OOM_CODE = 76
YIELD_CODE = 75


def _fold_inputs():
    selections = ORIGINAL_DOC / 'SELECTION.csv'
    with selections.open(newline='') as stream:
        rows = [row for row in csv.DictReader(stream) if row['fold'] in FOLDS and row['selected']=='True']
    require(len(rows)==len(FOLDS) and {row['fold'] for row in rows}==set(FOLDS),
            'missing or ambiguous selected S0 in the five-fold archive')
    return {row['fold']:_fold_identity(row) for row in rows}


def _teacher_cache(ref, identity):
    import torch
    from .assets import TeacherCache

    folder = ORIGINAL_RUN/ref.fold/'cache/teacher_cache'
    manifest = read(folder/'MANIFEST.json')
    scales = read(folder/'TRAIN_B_SCALES.json')
    cache = TeacherCache(folder,manifest,scales)
    saved = torch.load(identity['S0_checkpoint'],map_location='cpu',weights_only=False)
    original = saved['identity']
    indices = tuple(ref.split['train_indices'])
    members = ref.split['members']
    require(saved['completed_epoch']==identity['S0_epoch']
            and cache.identity==original['cache']
            and cache.indices==indices
            and original['train_indices_sha256']==canonical_hash(list(indices))
            and manifest['identity']['train_members_sha256']==identity['train_manifest_sha256']
            and manifest['identity']['purpose']==scales['purpose']=='train'
            and manifest['identity']['seed']==scales['seed']==2035
            and scales['scales']==identity['scales_m']
            and manifest['target_reads_during_export']==0,'original train-only A/B caches differ')
    for index in indices:
        item = cache.rows[index]
        require(item['n']==len(members[index]['agent_ids'])
                and item['member_sha256']==canonical_hash(members[index])
                and item['record_sha256']==ref.split['records'][index]['sha256'],
                'teacher cache window is out of order or changed')
    return cache


def _refs(inputs):
    refs = {fold:dev_reference(fold,RUN) for fold in FOLDS}
    caches = {}
    for fold,ref in refs.items():
        fixed = inputs[fold]
        require(all(fixed[name]==ref.identity[name] for name in
                    ('S0_epoch','S0_sha256','split_sha256','dev_manifest_sha256')),
                'S0 cache and selected checkpoint disagree')
        caches[fold] = DevelopmentCache(ref,RUN)
    return refs,caches


def prepare():
    from .group_query_train import RECIPE

    inputs = _fold_inputs()
    refs,dev = _refs(inputs)
    for fold in FOLDS:
        locked = inputs[fold]
        cache = _teacher_cache(refs[fold],locked)
        dev_max = max(dev[fold].row[index]['n'] for index in refs[fold].indices)
        locked.update(teachers=cache.identity, frozen_dev_cache=dev[fold].identity,
                      maximum_dev_people=dev_max)
    config = dict(
        base_commit=BASE, baseline_selection_sha256=digest(ORIGINAL_DOC/'SELECTION.csv'),
        folds=list(FOLDS), inputs=inputs, source_sha256=runtime_sources(), test_access=False,
        inference_seed=SEED, K=20, frames=12, observed_frames=8, dt=.4, DDIM_steps=20, eta=0.,
        only_final_epoch=12, precision='FP32', training=RECIPE,
        architecture=dict(width=64, activation='SiLU', query_width=32, group_motion_width=4,
            output_layer_zero=True, single_sync_pre_personal_update=True,
            grouping='same original greedy 2m mean last-three-position pairing; unmatched people form singletons'),
        validation=dict(purpose='selection_dev', UNIV_dense_JFDE_max_delta_m=-.01,
            joint_tolerance_m=1e-6,
            joint_fold_groups=['hotel/all','hotel/N2_8','univ/all'],
            marginal_fold_groups=['hotel/all','hotel/N9plus','univ/all','univ/N9plus'],
            test=False),
        optimization_scopes=dict(
            base_trainable=['queries','personal','edge','latent'],
            added_trainable=['group_query'],
            frozen=['S0','spatial encoder','Flow','CandidateHead','20-step bridge','A/B teachers']),
        retention='all 12 completed epochs; only R12 receives fixed final development score',
        resources=dict(mode='one GPU, folds serial', allocator_fraction=GPU_FRACTION,
            gpu_wait_free_mib=GPU_WAIT_MIB, recovery='same batch from atomic last, dense 5-slot bridge VJP if OOM'),
        environment=ENVIRONMENT, reproduce='python -m direct_k20.group_query_run',
    )
    immutable(RUN/'RUN_CONFIG.json',config)
    return config,refs,dev


def _eval_identity(fold, config):
    folder = RUN/fold
    receipt = folder/'TRAINING_COMPLETE.json'
    training = read(receipt)
    require(training['completed_epochs']==12 and
            digest(training['final_checkpoint'])==training['final_sha256'],
            'group query selected checkpoint is not the final epoch')
    identity = config['inputs'][fold] | dict(
        R12_checkpoint=training['final_checkpoint'], R12_sha256=training['final_sha256'],
        final_epoch=12, config_sha256=digest(RUN/'RUN_CONFIG.json'),
        training_receipt_sha256=digest(receipt), test_access=False,
    )
    immutable(folder/'EVALUATION_IDENTITY.json',identity)
    return identity


def _worker(fold, stage):
    import torch
    from bcep_joint.common import seed_all
    from bcep_v2.runtime import configure_cuda
    from .group_query import GroupQueryStudent

    parent = os.getppid()
    interrupted = False

    def stop(_signal, _frame):
        nonlocal interrupted
        interrupted = True

    def should_yield():
        return interrupted or os.getppid()!=parent

    signal.signal(signal.SIGUSR1,stop)
    config = read(RUN/'RUN_CONFIG.json')
    require(runtime_sources()==config['source_sha256'],'locked group-query source changed')
    folder = RUN/fold
    identity = config['inputs'][fold]
    torch.set_num_threads(2)
    configure_cuda()
    torch.cuda.set_per_process_memory_fraction(config['resources']['allocator_fraction'])
    device = torch.device('cuda')
    try:
        ref = dev_reference(fold,RUN)
        assets, original = _load(fold,identity,device)
        seed_all(config['training']['seed'])
        student = GroupQueryStudent()
        if stage=='train':
            from .group_query_train import train

            student.load_s0_state_dict(original.state_dict())
            del original
            cache = _teacher_cache(ref,identity)
            require(cache.identity==identity['teachers'],'original teacher cache changed')
            recover = (folder/'DDIM_MEMORY_RECOVERY.json').exists()
            result = train(assets,student.to(device),cache,folder/'training',identity,
                           device='cuda',slot_chunk=5 if recover else 0,dense_agents=24,
                           should_yield=should_yield)
        else:
            from .group_query_eval import export_fold

            del original
            identity = _eval_identity(fold,config)
            saved = torch.load(identity['R12_checkpoint'],map_location='cpu',weights_only=False)
            require(saved['completed_epoch']==12,'wrong dev model epoch')
            student.load_state_dict(saved['student'],strict=True)
            student.to(device=device,dtype=torch.float32).eval().requires_grad_(False)
            dev = DevelopmentCache(ref,RUN)
            require(dev.identity==identity['frozen_dev_cache'],'original dev prediction seal changed')
            repair = dev.repair_if_needed('cuda')
            immutable(folder/'DEV_PATH_REPLAY.json',repair)
            result = export_fold(assets,student,RUN,identity,dev,should_yield=should_yield)
        if result is None:
            return YIELD_CODE
        immutable(folder/_done(stage),result)
    except torch.OutOfMemoryError as error:
        write(folder/'GPU_OOM.json',dict(stage=stage,unix=time.time(),error=str(error),
              peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
              peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
        traceback.print_exc()
        return OOM_CODE
    return 0


def _done(stage):
    return 'TRAINING_COMPLETE.json' if stage=='train' else 'DEV_EXPORT_COMPLETE.json'


def _execute(fold, stage):
    folder = RUN/fold
    folder.mkdir(exist_ok=True)
    complete = folder/_done(stage)
    if complete.exists():
        return
    command = [sys.executable,'-m','direct_k20.group_query_run','--worker',fold,stage]
    with (ASSETS/'auxiliary_gpu.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            status(RUN,'WAITING_FOR_GPU_LOCK',fold=fold,stage=stage)
            fcntl.flock(lock,fcntl.LOCK_EX)
        for attempt in range(1,4):
            free = _gpu()
            while free['free_mib'] < GPU_WAIT_MIB:
                status(RUN,'WAITING_FOR_GPU_MEMORY',fold=fold,stage=stage,gpu=free)
                time.sleep(30)
                free = _gpu()
            started = time.time()
            with (RUN/f'{fold}_{stage}.log').open('a',buffering=1) as log:
                child = subprocess.Popen(command,cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT,
                                         pass_fds=(lock.fileno(),))
                write(RUN/'ACTIVE_CHILD.json',dict(pid=child.pid,fold=fold,stage=stage,gpu=free,
                      attempt=attempt,unix=started,command=command))
                status(RUN,'GROUP_QUERY_WORKER_RUNNING',fold=fold,stage=stage,pid=child.pid,gpu=free)
                code = child.wait()
            if code==0:
                require(complete.exists(),f'{fold} {stage} completed without receipt')
                return
            marker = folder/'DDIM_MEMORY_RECOVERY.json'
            if stage=='train' and code==OOM_CODE and not marker.exists():
                immutable(marker,dict(slot_chunk=5,dense_agents=24,oom=read(folder/'GPU_OOM.json')))
            status(RUN,'RECOVERING_SAME_GROUP_QUERY_RUN',fold=fold,stage=stage,attempt=attempt,exit_code=code)
            if attempt<3:
                time.sleep(20)
    raise RuntimeError(f'{fold} {stage} stopped three times; see {fold}_{stage}.log')


def _publish():
    from joint_flow_fourfold.report import public_metadata
    from .group_query_eval import build_report

    DOC.mkdir(parents=True,exist_ok=True)
    build_report(RUN,DOC)
    for fold in FOLDS:
        out = DOC/fold
        out.mkdir(exist_ok=True)
        source = RUN/fold
        for path in (source/'training/checkpoints/TOP5.json', source/'EVALUATION_IDENTITY.json',
                     source/'DEV_EXPORT_COMPLETE.json', source/'DDIM_MEMORY_RECOVERY.json'):
            if path.exists():
                write(out/path.name,public_metadata(read(path)))
    for name in ('RUN_CONFIG.json','START.json','WALL_TIME.json','DEV_PATH_REPLAY.json'):
        write(DOC/name,public_metadata(read(RUN/name)))
    write(DOC/'ARTIFACT_SHA256.json',{str(path.relative_to(DOC)):digest(path)
          for path in sorted(DOC.rglob('*')) if path.is_file() and path.name!='ARTIFACT_SHA256.json'})
    require(subprocess.check_output(['git','branch','--show-current'],text=True).strip()==BRANCH,
            'wrong group-query branch for archive')
    require(not subprocess.check_output(['git','diff','--cached','--name-only'],text=True).strip(),
            'unexpected staged files')
    subprocess.run(['git','add',str(DOC.relative_to(SOURCE))],check=True)
    subprocess.run(['git','diff','--cached','--check'],check=True)
    if subprocess.call(['git','diff','--cached','--quiet']):
        subprocess.run(['git','commit','-m','Archive fixed HOTEL and UNIV group-query development evaluation'],check=True)
    subprocess.run(['git','push','-u','origin',BRANCH],check=True)
    sha = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    remote = subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],text=True).split()[0]
    require(sha==remote,'archived remote group-query ref differs')
    write(RUN/'DELIVERY.json',dict(sha=sha,verified_remote_sha=remote,branch=BRANCH))
    status(RUN,'COMPLETE',folds=list(FOLDS),test_access=False,sha=sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',nargs=2,metavar=('FOLD','STAGE'))
    parser.add_argument('--prepare-only',action='store_true')
    args = parser.parse_args()
    if args.worker:
        fold,stage = args.worker
        require(fold in FOLDS and stage in ('train','dev'),'invalid worker scope')
        return _worker(fold,stage)
    require(all(os.environ.get(k)==v for k,v in ENVIRONMENT.items()),'wrong launch environment')
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config,refs,dev_caches = prepare()
        if args.prepare_only or (RUN/'DELIVERY.json').exists():
            return 0
        if not (RUN/'START.json').exists():
            write(RUN/'START.json',dict(unix=time.time(),pid=os.getpid(),gpu=_gpu(),
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                command=[sys.executable,'-m','direct_k20.group_query_run']))
        for fold in FOLDS:
            _execute(fold,'train')
        identities = {fold:_eval_identity(fold,config) for fold in FOLDS}
        dev_start = time.monotonic()
        for fold in FOLDS:
            _execute(fold,'dev')
        require(runtime_sources()==config['source_sha256'],'group-query source changed during execution')

        import torch
        from .group_query_eval import score_all
        from .group_query_train import record_final_metrics

        torch.set_num_threads(1)
        write(RUN/'DEV_PATH_REPLAY.json',{fold:read(RUN/fold/'DEV_PATH_REPLAY.json') for fold in FOLDS})
        status(RUN,'SCORING_SEALED_DEVELOPMENT')
        score = score_all(refs,RUN,identities,dev_caches)
        for fold in FOLDS:
            final = score['groups'][fold]['all']
            record_final_metrics(RUN/fold/'training',{m:final[f'R12_{m}_m'] for m in METRICS})
        wall = time.time()-read(RUN/'START.json')['unix']
        write(RUN/'WALL_TIME.json',dict(seconds=wall,dev_seconds=time.monotonic()-dev_start))
        status(RUN,'ARCHIVING',decision=score['decision'])
        _publish()
    return 0


if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception:
        if '--worker' not in sys.argv:
            status(RUN,'INTERRUPTED_RESUME_SAME_RUN',traceback=traceback.format_exc())
        raise

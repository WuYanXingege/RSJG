"""python -m residual_frozen_test.run --push-on-complete; resumable, no training."""
import argparse
from contextlib import contextmanager
import fcntl
import os
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from bcep.contracts import canonical_hash, digest
from bcep_joint.common import read, write
from joint_flow_fourfold.common import immutable
from joint_flow_fourfold.inputs import state_hash
from joint_flow_frozen_test.inputs import ObservedOnlyHeldoutData
from joint_flow_frozen_test.run import forbid_heldout_truth
from joint_marginal_residual.assets import load_teacher, load_inputs
from joint_marginal_residual.inference import generate_student
from joint_marginal_residual.model import ResidualHead
from joint_marginal_residual.run import protect
from .common import *


def status(value, **extra):
    row = dict(status=value, unix=time.time(), **extra)
    write(RUN/'STATUS.json', row); print(row, flush=True)


@contextmanager
def observation_guard():
    # Guard both the formal reader API and accidental direct NPZ future access.
    original = np.lib.npyio.NpzFile.__getitem__
    def guarded(self, key):
        if key in ('future', 'future.npy'):
            raise RuntimeError('all three folds must be sealed before reading any future')
        return original(self, key)
    with forbid_heldout_truth():
        np.lib.npyio.NpzFile.__getitem__ = guarded
        try:yield
        finally:np.lib.npyio.NpzFile.__getitem__ = original


def load_models(fold, device):
    archived = read(DEV_DOC/fold/'RESULT.json')
    local = read(PRIOR/fold/'head/RESULT.json')
    assert archived['rank1_epoch'] == local['rank1_epoch'] == EXPECTED[fold][1]
    path = local['checkpoint']
    assert digest(path) == local['checkpoint_sha256'] == archived['checkpoint_sha256']
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['completed_epoch'] == archived['rank1_epoch'] and not saved['ablation']
    os.environ['RSJG_DISABLE_OPTIMIZATIONS'] = saved['native_disabled_optimizations']
    inputs, model, provider, dependence, identity = load_teacher(fold, device)
    assert identity == saved['teacher_identity']
    assert identity['teacher_epoch'] == EXPECTED[fold][0]
    head = ResidualHead().cpu().eval().requires_grad_(False)
    head.load_state_dict(saved['model'], strict=True)
    assert head.parameter_count == 27332
    modules = dict(backbone=model, provider=provider, dependence=dependence, head=head)
    assert all(not p.requires_grad for m in modules.values() for p in m.parameters())
    assert all(not sub.training for m in modules.values() for sub in m.modules())
    model_identity = dict(teacher_identity=identity, head_epoch=archived['rank1_epoch'],
        head_sha256=archived['checkpoint_sha256'], head_checkpoint=path,
        head_state_sha256=state_hash(head), ablation=False,
        native_disabled_optimizations=saved['native_disabled_optimizations'])
    return inputs, modules, model_identity


def prepare():
    RUN.mkdir(parents=True, exist_ok=True); protect(); identities={}; manifests={}
    with observation_guard():
        for fold in FOLDS:
            inputs, modules, identity = load_models(fold, 'cpu')
            split=inputs.split; ids=split[PURPOSE+'_indices'];ns=[len(split['members'][i]['agent_ids']) for i in ids]
            assert (len(ns),sum(ns),sum(n>=9 for n in ns),sum(n for n in ns if n>=9)) == EXPECTED[fold][2:]
            rows=[]
            for pos,index in enumerate(ids,1):
                record=split['records'][index];assert digest(record['path'])==record['sha256']
                rows.append(dict(window=f'test_{pos:04d}', index=index, n=ns[pos-1],
                    member_sha256=canonical_hash(split['members'][index]), record_sha256=record['sha256']))
            manifest=dict(fold=fold, split_sha256=canonical_hash(split), rows=rows)
            immutable(RUN/fold/'TEST_WINDOWS.json',manifest)
            identities[fold]=identity;manifests[fold]=canonical_hash(manifest)
            del modules,inputs
    config=dict(schema='residual-threefold-frozen-test-v1', base_commit=BASE, folds=list(FOLDS), seed=SEED,
        expected_counts=EXPECTED, identities=identities, window_manifests=manifests, source_sha256=sources(),
        protocol=dict(purpose=PURPOSE,M=256,K=20,prediction_frames=12,DDIM=20,eta=0.,ablation=False,
                      calibration=False,scene_swap=False,training_updates=0,weight_selection_on_test=False),
        scoring=dict(native='ebjd.metrics.hard_metric_statistics + MetricAccumulator',dtype='CPU FP32; FP64 accumulator',
                     joint='window equal weight',marginal='valid-agent weighted',four_independent_argmin=True,
                     fold_summary='equal folds; dense includes two available folds only',empty_group='N/A'),
        historical_B={fold:dict(path=str(historical(fold).relative_to(SOURCE)),sha256=digest(historical(fold))) for fold in FOLDS},
        replay_tolerance=dict(atol=1e-6,rtol=1e-6,reference='joint_flow_frozen_test/run.py replay_dev existing FP32 tolerance'),
        runtime=dict(threads=2,device='cuda',head_device='cpu',allocator_fraction=.4,minimum_free_MiB=6144,
                     auxiliary_workers=1,resource_lock='controller.lock plus auxiliary_gpu.lock; original queue unchanged'),
        seal_rule='All 2148 paired predictions and membership/hash/shape checks complete before ANY test future read')
    # JSON normalizes EXPECTED tuples to lists; canonical comparison for restarts.
    config=__import__('json').loads(__import__('json').dumps(config))
    immutable(RUN/'RUN_CONFIG.json', config)
    return config


def verify_sources(config):
    protect();assert sources()==config['source_sha256'], 'bound test source changed'


@torch.no_grad()
def export_fold(fold):
    from bcep_v2.runtime import configure_cuda
    config=read(RUN/'RUN_CONFIG.json');verify_sources(config)
    torch.set_num_threads(2);configure_cuda();torch.cuda.set_per_process_memory_fraction(.4)
    folder=RUN/fold;folder.mkdir(exist_ok=True)
    with observation_guard():
        inputs,modules,identity=load_models(fold,'cuda')
        assert identity==config['identities'][fold]
        before={k:state_hash(v) for k,v in modules.items()}
        reader=ObservedOnlyHeldoutData(inputs.split);indices=inputs.split[PURPOSE+'_indices']
        rows=[];start=time.monotonic();new=0;reused=0
        for pos,index in enumerate(indices,1):
            expected=expected_row(fold,pos,index,inputs.split,config)
            path=RUN/expected['file'];side=path.with_suffix('.json');path.parent.mkdir(exist_ok=True)
            if side.exists():
                row=read(side);validate_row(row,expected);reused+=1
            else:
                host=reader.batch([index],PURPOSE,truth=False)
                assert host.valid.all() and int(host.valid.sum())==expected['n'] and not host.future.count_nonzero()
                assert canonical_hash(inputs.split['members'][index])==expected['member_sha256']
                batch=host.to('cuda');tick=time.monotonic()
                student,detail=generate_student(modules['backbone'],modules['provider'],modules['dependence'],
                                               modules['head'],batch,seed=SEED,ablation=False)
                torch.cuda.synchronize()
                assert detail['ddim_calls']==1
                payload=dict(B=detail['B'].cpu().numpy(),Bstar=student.cpu().numpy(),
                             raw_ids=detail['raw_ids'].cpu().numpy().astype(np.int64),valid=host.valid[0].numpy())
                if path.exists():
                    previous=arrays(path,expected['n'])
                    assert all(np.array_equal(previous[k],v) for k,v in payload.items()), 'orphan prediction differs on replay'
                else:
                    tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**payload);tmp.replace(path)
                row=dict(expected,sha256=digest(path),ddim_calls=1,ablation=False,seconds=time.monotonic()-tick,
                         teacher_seconds=detail['teacher_seconds'],feature_seconds=detail['feature_seconds'],
                         head_seconds=detail['head_seconds'])
                validate_row(row,expected);write(side,row);new+=1
            rows.append(row)
            if pos%25==0 or pos==len(indices):
                progress=dict(fold=fold,windows=pos,total=len(indices),seconds=time.monotonic()-start,
                              new_windows=new,reused_windows=reused,unix=time.time())
                write(folder/'PROGRESS.json',progress);print(progress,flush=True)
        after={k:state_hash(v) for k,v in modules.items()};assert before==after
        immutable(folder/'EXPORT_COMPLETE.json',dict(fold=fold,source_config_sha256=canonical_hash(config),
            rows=rows,states=before,exact_weights_unchanged=True,training_updates=0,heldout_future_reads=0))
        write(folder/'EXPORT_TIMING.json',dict(wall_seconds=time.monotonic()-start,new_windows=new,reused_windows=reused,
            prediction_seconds=sum(r['seconds'] for r in rows),complete_unix=time.time()))
    verify_sources(config)


def verify_all(config):
    rows=[]
    with observation_guard():
        for fold in FOLDS:
            inputs,_,_=load_inputs(fold)
            manifest=read(RUN/fold/'TEST_WINDOWS.json')
            assert canonical_hash(manifest)==config['window_manifests'][fold]
            complete=read(RUN/fold/'EXPORT_COMPLETE.json')
            assert complete['source_config_sha256']==canonical_hash(config)
            assert complete['exact_weights_unchanged'] and complete['heldout_future_reads']==0
            indices=inputs.split[PURPOSE+'_indices'];assert len(complete['rows'])==len(indices)
            for pos,(index,row) in enumerate(zip(indices,complete['rows']),1):
                validate_row(row,expected_row(fold,pos,index,inputs.split,config));rows.append(row)
    assert len(rows)==2148
    return rows


def seal_all(config):
    rows=verify_all(config);path=RUN/'PREDICTIONS_FROZEN.json'
    if path.exists():
        seal=read(path);assert seal['rows']==rows and seal['config_sha256']==digest(RUN/'RUN_CONFIG.json')
    else:
        seal=dict(schema='threefold-paired-predictions-seal-v1',rows=rows,windows=2148,seed=SEED,
                  config_sha256=digest(RUN/'RUN_CONFIG.json'),heldout_future_reads=0,seal_unix=time.time())
        immutable(path,seal)
    return seal


def archive(push):
    from .report import build_report
    build_report()
    if not push:return
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()==BRANCH
    subprocess.run(['git','add','residual_frozen_test',str(DOC.relative_to(SOURCE))],check=True)
    subprocess.run(['git','diff','--cached','--check'],check=True)
    if subprocess.call(['git','diff','--cached','--quiet']):
        subprocess.run(['git','commit','-m','Archive three-fold seed2035 frozen residual-head test'],check=True)
    subprocess.run(['git','push','-u','origin',BRANCH],check=True)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],text=True).split()[0]
    assert sha==remote
    write(RUN/'DELIVERY.json',dict(sha=sha,remote_verified_sha=remote,branch=BRANCH,
        report=f'https://github.com/WuYanXingege/RSJG/blob/{sha}/{DOC.relative_to(SOURCE)}/README.md'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',choices=FOLDS)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--push-on-complete',action='store_true')
    args=p.parse_args();os.chdir(SOURCE)
    if args.worker:export_fold(args.worker);return
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2);config=prepare()
        if args.prepare_only:status('PREPARED');return
        commands=[];start=time.time();(RUN/'logs').mkdir(exist_ok=True)
        for fold in FOLDS:
            if (RUN/fold/'EXPORT_COMPLETE.json').exists():continue
            with (RUN.parent/'auxiliary_gpu.lock').open('a') as resource:
                fcntl.flock(resource,fcntl.LOCK_EX)
                for attempt in range(1,4):
                    while True:
                        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
                        if free>=6144:break
                        status('WAITING_SHARED_GPU_HEADROOM',free_MiB=free);time.sleep(45)
                    verify_sources(config);status('EXPORT_OBSERVATIONS_ONLY',fold=fold,attempt=attempt)
                    command=[sys.executable,'-m','residual_frozen_test.run','--worker',fold];tick=time.time()
                    with (RUN/'logs'/f'{fold}.log').open('a',buffering=1) as log:
                        child=subprocess.Popen(command,cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
                        write(RUN/'ACTIVE_CHILD.json',dict(pid=child.pid,fold=fold));code=child.wait()
                    commands.append(dict(command=command,fold=fold,start_unix=tick,wall_seconds=time.time()-tick,exit_code=code))
                    write(RUN/'COMMANDS.json',commands)
                    if code==0:break
                    if attempt==3:raise RuntimeError(f'{fold} failed; resume same frozen configuration')
                    time.sleep(15)
        status('VERIFYING_ALL_THREE_FOLDS_BEFORE_TRUTH');seal_all(config)
        from .scoring import score
        status('SCORING_ALL_SEALED_PREDICTIONS');score(config)
        verify_sources(config)
        write(RUN/'WALL_TIME.json',dict(start_unix=start,finished_unix=time.time(),wall_seconds=time.time()-start))
        status('ARCHIVING');archive(args.push_on_complete)
        status('COMPLETE',folds=list(FOLDS),seed=SEED,training_updates=0)


if __name__=='__main__':
    try:main()
    except BaseException:
        RUN.mkdir(parents=True,exist_ok=True);status('INTERRUPTED_RESUME_SAME_TEST',traceback=traceback.format_exc());raise

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


def verify_old():
    from joint_flow_fourfold.report import public_metadata
    old=read(OLD_RUN/'RUN_CONFIG.json');receipt=read(OLD_RUN/'SCORE_RECEIPT.json')
    assert old['seed']==2035 and old['protocol']['training_updates']==0
    assert public_metadata(old)==read(OLD_DOC/'RUN_CONFIG.json')
    assert receipt==read(OLD_DOC/'SCORE_RECEIPT.json') and receipt['status']=='COMPLETE'
    assert receipt['source_config_sha256']==digest(OLD_RUN/'RUN_CONFIG.json')
    assert receipt['first_truth_read_unix']>receipt['global_verification_unix']>receipt['global_seal_unix']
    for name,sha in receipt['outputs'].items():
        assert digest(OLD_RUN/name)==sha and digest(OLD_DOC/name)==sha
    seal=read(OLD_RUN/'PREDICTIONS_FROZEN.json')
    assert digest(OLD_RUN/'PREDICTIONS_FROZEN.json')==receipt['prediction_manifest_sha256']
    assert len(seal['rows'])==2148 and read(OLD_RUN/'BASELINE_REPLAY.json')['passed']
    for row in seal['rows']:
        assert row['seed']==2035 and row['source_config_sha256']==canonical_hash(old)
        validate_row(row,{k:v for k,v in row.items() if k not in ('sha256',)},root=OLD_RUN)
    return old,receipt


def prepare():
    RUN.mkdir(parents=True,exist_ok=True);protect();identities={};manifests={};audits={}
    with observation_guard():
        old,receipt=verify_old()
        for fold in FOLDS:
            inputs,modules,identity=load_models(fold,'cpu')
            assert identity==old['identities'][fold]
            split=inputs.split;ids=split[PURPOSE+'_indices'];ns=[len(split['members'][i]['agent_ids']) for i in ids]
            assert (len(ns),sum(ns),sum(n>=9 for n in ns),sum(n for n in ns if n>=9))==EXPECTED[fold][2:]
            manifest=read(OLD_RUN/fold/'TEST_WINDOWS.json')
            assert canonical_hash(manifest)==old['window_manifests'][fold]
            assert manifest['split_sha256']==canonical_hash(split)
            for pos,(index,row) in enumerate(zip(ids,manifest['rows']),1):
                assert row['index']==index and row['n']==ns[pos-1]
                assert row['member_sha256']==canonical_hash(split['members'][index])
                assert digest(split['records'][index]['path'])==row['record_sha256']
            immutable(RUN/fold/'TEST_WINDOWS.json',manifest)
            identities[fold]=identity;manifests[fold]=canonical_hash(manifest)
            audits[fold]=[next(i for i,n in zip(ids,ns) if n==1),next(i for i,n in zip(ids,ns) if 2<=n<=8)]
            if any(n>=9 for n in ns):audits[fold].append(next(i for i,n in zip(ids,ns) if n>=9))
            del modules,inputs
    config=dict(schema='residual-threefold-five-seed-v1',base_commit=BASE,folds=list(FOLDS),
        seeds=list(SEEDS),new_seeds=list(NEW_SEEDS),reused_seed=2035,expected_counts=EXPECTED,
        identities=identities,window_manifests=manifests,source_sha256=sources(),random_audit_indices=audits,
        protocol=old['protocol'],scoring=dict(old['scoring'],seed_summary='five paired deltas mean and sample SD ddof=1',
            direction='unrounded delta <0 / ==0 / >0; no tolerance for direction; same-seed four metrics'),
        historical_B=old['historical_B'],replay_tolerance=old['replay_tolerance'],runtime=old['runtime'],
        prior2035=dict(config_sha256=digest(OLD_RUN/'RUN_CONFIG.json'),receipt_sha256=digest(OLD_RUN/'SCORE_RECEIPT.json'),
            prediction_manifest_sha256=digest(OLD_RUN/'PREDICTIONS_FROZEN.json'),outputs=receipt['outputs']),
        disclosure='Seeds 2036-2039 were authorized AFTER observing seed2035: ZARA1 improved, ZARA2/HOTEL worsened. Frozen weights and settings unchanged.',
        random_audit='fixed singleton/small/dense windows; actual epsilon/xi/bridge fingerprints, exact repeat regeneration, cross-seed distinct nonempty streams; no extra DDIM',
        seal_rule='All 8592 new pairs across four seeds and three folds sealed before ANY new truth read; reuse original 2035 scores without re-inference/re-scoring')
    config=__import__('json').loads(__import__('json').dumps(config))
    immutable(RUN/'RUN_CONFIG.json',config)
    immutable(RUN/'REUSE_2035_RECEIPT.json',dict(verified=True,source_commit=BASE,**config['prior2035'],
        reused_pairs=2148,additional_DDIM_calls=0,additional_truth_reads=0))
    return config


def verify_sources(config):
    protect();assert sources()==config['source_sha256'], 'bound test source changed'


@torch.no_grad()
def export_fold(fold, seed):
    assert seed in NEW_SEEDS
    from bcep_v2.runtime import configure_cuda
    config=read(RUN/'RUN_CONFIG.json');verify_sources(config)
    torch.set_num_threads(2);configure_cuda();torch.cuda.set_per_process_memory_fraction(.4)
    folder=RUN/fold/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    with observation_guard():
        inputs,modules,identity=load_models(fold,'cuda')
        assert identity==config['identities'][fold]
        before={k:state_hash(v) for k,v in modules.items()}
        reader=ObservedOnlyHeldoutData(inputs.split);indices=inputs.split[PURPOSE+'_indices']
        audit_indices=set(config['random_audit_indices'][fold])
        rows=[];start=time.monotonic();new=0;reused=0
        for pos,index in enumerate(indices,1):
            expected=expected_row(fold,seed,pos,index,inputs.split,config)
            path=RUN/expected['file'];side=path.with_suffix('.json');path.parent.mkdir(exist_ok=True)
            if side.exists():
                row=read(side);validate_row(row,expected);reused+=1
                if index in audit_indices:
                    from .random_audit import validate_resume
                    host=reader.batch([index],PURPOSE,truth=False).to('cuda')
                    ids=torch.from_numpy(arrays(path,expected['n'])['raw_ids'])
                    validate_resume(host,ids,seed,row['random_inputs'])
            else:
                host=reader.batch([index],PURPOSE,truth=False)
                assert host.valid.all() and int(host.valid.sum())==expected['n'] and not host.future.count_nonzero()
                assert canonical_hash(inputs.split['members'][index])==expected['member_sha256']
                batch=host.to('cuda');tick=time.monotonic()
                student,detail=generate_student(modules['backbone'],modules['provider'],modules['dependence'],
                                               modules['head'],batch,seed=seed,ablation=False,audit_random_inputs=index in audit_indices)
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
                         head_seconds=detail['head_seconds'],random_inputs=detail['random_inputs'])
                validate_row(row,expected);write(side,row);new+=1
            rows.append(row)
            if pos%25==0 or pos==len(indices):
                progress=dict(fold=fold,seed=seed,windows=pos,total=len(indices),seconds=time.monotonic()-start,
                              new_windows=new,reused_windows=reused,unix=time.time())
                write(folder/'PROGRESS.json',progress);print(progress,flush=True)
        after={k:state_hash(v) for k,v in modules.items()};assert before==after
        immutable(folder/'EXPORT_COMPLETE.json',dict(fold=fold,seed=seed,source_config_sha256=canonical_hash(config),
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
            indices=inputs.split[PURPOSE+'_indices']
            for seed in NEW_SEEDS:
                complete=read(RUN/fold/f'seed_{seed}/EXPORT_COMPLETE.json')
                assert complete['source_config_sha256']==canonical_hash(config) and complete['seed']==seed
                assert complete['exact_weights_unchanged'] and complete['heldout_future_reads']==0
                assert len(complete['rows'])==len(indices)
                for pos,(index,row) in enumerate(zip(indices,complete['rows']),1):
                    validate_row(row,expected_row(fold,seed,pos,index,inputs.split,config));rows.append(row)
    assert len(rows)==8592 and len({(r['fold'],r['seed'],r['window']) for r in rows})==8592
    return rows


def seal_all(config):
    from .random_audit import validate_cross_seed
    rows=verify_all(config);path=RUN/'PREDICTIONS_FROZEN.json'
    audit=validate_cross_seed(rows);immutable(RUN/'RANDOM_INPUT_AUDIT.json',audit)
    if path.exists():
        seal=read(path);assert seal['rows']==rows and seal['config_sha256']==digest(RUN/'RUN_CONFIG.json')
    else:
        seal=dict(schema='threefold-four-new-seeds-seal-v1',rows=rows,new_pairs=8592,seeds=list(NEW_SEEDS),
            reused2035_manifest_sha256=config['prior2035']['prediction_manifest_sha256'],
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
        subprocess.run(['git','commit','-m','Archive three-fold five-seed frozen residual-head replication'],check=True)
    subprocess.run(['git','push','-u','origin',BRANCH],check=True)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],text=True).split()[0]
    assert sha==remote
    write(RUN/'DELIVERY.json',dict(sha=sha,remote_verified_sha=remote,branch=BRANCH,
        report=f'https://github.com/WuYanXingege/RSJG/blob/{sha}/{DOC.relative_to(SOURCE)}/README.md'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',choices=FOLDS)
    p.add_argument('--seed',type=int,choices=NEW_SEEDS)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--push-on-complete',action='store_true')
    args=p.parse_args();os.chdir(SOURCE)
    if args.worker:
        if args.seed is None:p.error('--worker requires --seed')
        export_fold(args.worker,args.seed);return
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2);config=prepare()
        if args.prepare_only:status('PREPARED');return
        commands=read(RUN/'COMMANDS.json') if (RUN/'COMMANDS.json').exists() else []
        if not (RUN/'START.json').exists():immutable(RUN/'START.json',dict(unix=time.time()))
        start=read(RUN/'START.json')['unix'];(RUN/'logs').mkdir(exist_ok=True)
        for seed in NEW_SEEDS:
            for fold in FOLDS:
                if (RUN/fold/f'seed_{seed}/EXPORT_COMPLETE.json').exists():continue
                with (RUN.parent/'auxiliary_gpu.lock').open('a') as resource:
                    fcntl.flock(resource,fcntl.LOCK_EX)
                    for attempt in range(1,4):
                        while True:
                            free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
                            if free>=6144:break
                            status('WAITING_SHARED_GPU_HEADROOM',free_MiB=free);time.sleep(45)
                        verify_sources(config);status('EXPORT_OBSERVATIONS_ONLY',fold=fold,seed=seed,attempt=attempt)
                        command=[sys.executable,'-m','residual_frozen_test.run','--worker',fold,'--seed',str(seed)];tick=time.time()
                        with (RUN/'logs'/f'{fold}_{seed}.log').open('a',buffering=1) as log:
                            child=subprocess.Popen(command,cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
                            write(RUN/'ACTIVE_CHILD.json',dict(pid=child.pid,fold=fold,seed=seed));code=child.wait()
                        commands.append(dict(command=command,fold=fold,seed=seed,start_unix=tick,wall_seconds=time.time()-tick,exit_code=code))
                        write(RUN/'COMMANDS.json',commands)
                        if code==0:break
                        if attempt==3:raise RuntimeError(f'{fold}/{seed} failed; resume same frozen configuration')
                        time.sleep(15)
        status('VERIFYING_ALL_NEW_PAIRS_BEFORE_TRUTH');seal_all(config)
        from .scoring import score
        status('SCORING_ALL_SEALED_PREDICTIONS');score(config)
        verify_sources(config)
        write(RUN/'WALL_TIME.json',dict(start_unix=start,finished_unix=time.time(),wall_seconds=time.time()-start))
        status('ARCHIVING');archive(args.push_on_complete)
        status('COMPLETE',folds=list(FOLDS),seeds=list(SEEDS),new_pairs=8592,reused_pairs=2148,training_updates=0)


if __name__=='__main__':
    try:main()
    except BaseException:
        RUN.mkdir(parents=True,exist_ok=True);status('INTERRUPTED_RESUME_SAME_TEST',traceback=traceback.format_exc());raise

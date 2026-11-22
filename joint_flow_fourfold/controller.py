"""Durable fourfold fit -> cross-fold freeze -> test -> archive controller."""
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
from .common import (SOURCE, ROOT, RUN, DOC, BASE_COMMIT, BRANCH, FOLDS,
                     SEEDS, immutable, source_hashes, status, wait_gpu)


def protected_paths():
    assets = ROOT/'RSJG_EBJD_assets/experiments'
    return dict(old_queue_status=assets/'bcep_ebjd_v2_integrated/STATUS.json',
        old_queue_contract=assets/'bcep_ebjd_v2_integrated/FORMAL_RUN.json',
        old_queue_authorization=assets/'bcep_ebjd_v2_integrated/AUTHORIZED.yaml',
        SSEC=assets/'zara2_ssec_energy_stage2_20261121/SUPERVISOR_STATUS.json')


def check_protection(config):
    assert {key: digest(path) for key, path in protected_paths().items()} == config['protected_sha256']
    assert read(protected_paths()['SSEC'])['status'] == 'PAUSED_NO_AUTOMATIC_RESUME'
    assert source_hashes() == config['source_sha256'], 'runtime source changed; do not hot-edit a bound run'


def prepare(run):
    from .upstream import inspect_fold, MissingUpstreamAssets
    run.mkdir(parents=True, exist_ok=True)
    inventory = {}
    for fold in FOLDS:
        try:
            inventory[fold] = inspect_fold(fold)
        except (FileNotFoundError, MissingUpstreamAssets) as error:
            inventory[fold] = dict(status='MISSING_ORIGINAL_ASSETS', error=repr(error),
                                   missing=getattr(error,'missing',[str(error)]))
    immutable(run/'ASSET_INVENTORY.json', inventory)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip()
    assert subprocess.check_output(['git','branch','--show-current'],cwd=SOURCE,text=True).strip() == BRANCH
    subprocess.run(['git','merge-base','--is-ancestor',BASE_COMMIT,head],cwd=SOURCE,check=True)
    config = dict(schema='fourfold-continuous-joint-confirmation-v1', base_commit=BASE_COMMIT,
        runtime_source_commit=head, source_sha256=source_hashes(), folds=list(FOLDS),
        execution_order=list(FOLDS), seeds=list(SEEDS), training_seed=3201,
        provider_training_seed=3102, head_training_seed=3102,
        dataset=dict(observed_frames=8,predicted_frames=12,dt_seconds=.4,unit='m',
                     fit='train',selection='selection_dev',test='heldout_confirmation'),
        upstream='native original fivefold contract; reuse matching assets; copy ZARA1 head before resume',
        A_selector='full-dev H-JADE rank1; exact ties earlier completed epoch',
        B_selector='full-dev real K20 trajectory JADE seed2035 rank1; exact ties earlier completed epoch',
        dependency_parameters=16964, dependency_recipe=dict(seed=3201,precision='FP32',optimizer='AdamW',
            lr=1e-4,weight_decay=1e-4,clip=5,logical_scenes=16,warmup_epochs=5,
            cosine_horizon=100,minimum=30,patience=15,max_epochs=100),
        inference=dict(M=256,K=20,ddim_steps=20,eta=0.,B='joint compression then one frozen correction'),
        retention='all distinct completed epoch snapshots, best, last and machine-readable Top5; no asset deletion',
        freeze_barrier='all available new-fold A/B rank1 weights selected and SHA256-recorded before any new-fold test',
        test_truth_barrier='within each fold all five seeds A/B predictions sealed before loading heldout future',
        aggregation='per-seed equal folds then five-seed mean and sample standard deviation; dense empty folds N/A',
        zara2=dict(reuse_commit=BASE_COMMIT, archive='docs/research/continuous_joint_flow/zara2_frozen_test_20261122',
            A_epoch=5,A_historical_dev_rank=2,B_epoch=35,no_new_inference=True),
        missing_assets='mark exact missing item; continue remaining runnable folds; disclose actual completed fold count',
        failures='resume same last checkpoint; never introduce a new model, seed, recipe or selector',
        protected_sha256={key:digest(path) for key,path in protected_paths().items()})
    if (run/'RUN_CONFIG.json').exists():
        previous = read(run/'RUN_CONFIG.json')
        # Report commits do not change the code provenance or identity of this run.
        config['runtime_source_commit'] = previous['runtime_source_commit']
    immutable(run/'RUN_CONFIG.json',config)
    check_protection(config)
    return config, inventory


def execute_stage(run, config, fold, stage):
    result_name = dict(upstream='UPSTREAM_FROZEN.json',dependency='B_FROZEN.json',test='TEST_RESULT.json')[stage]
    result = run/fold/result_name
    if result.exists():
        return read(result)
    (run/fold).mkdir(parents=True,exist_ok=True)
    logs = run/'logs'
    logs.mkdir(exist_ok=True)
    # Recovery never changes seeds, outputs, model selection or source identity.
    for attempt in range(1,4):
        check_protection(config)
        wait_gpu(run)
        assert shutil.disk_usage(run).free > 2*1024**3, 'less than 2 GiB free; preserve all existing assets'
        status(run,'RUNNING',fold=fold,stage=stage,attempt=attempt,
               log=str(logs/f'{fold}_{stage}.log'))
        with (logs/f'{fold}_{stage}.log').open('a',buffering=1) as log:
            log.write(f'\nATTEMPT {attempt} unix={time.time()} same-run resume\n')
            proc = subprocess.Popen([sys.executable,'-m','joint_flow_fourfold.worker',stage,
                '--fold',fold,'--run',str(run)],cwd=SOURCE,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
            write(run/'ACTIVE_CHILD.json',dict(pid=proc.pid,fold=fold,stage=stage,attempt=attempt,unix=time.time()))
            code = proc.wait()
        if code == 0 and result.exists():
            check_protection(config)
            return read(result)
        if code == 3 and (run/fold/'MISSING_ASSETS.json').exists():
            return read(run/fold/'MISSING_ASSETS.json')
        status(run,'RECOVERING_SAME_RUN',fold=fold,stage=stage,attempt=attempt,exit_code=code)
        if attempt < 3:
            time.sleep(30)
    raise RuntimeError(f'{fold}/{stage} failed three same-run attempts; inspect {logs/f"{fold}_{stage}.log"}')


def freeze_all(run, config, inventory):
    admitted, missing = {}, {}
    for fold in FOLDS:
        if inventory[fold].get('status') == 'MISSING_ORIGINAL_ASSETS':
            missing[fold] = inventory[fold]
            continue
        upstream = read(run/fold/'UPSTREAM_FROZEN.json')
        selected = read(run/fold/'B_FROZEN.json')
        assert digest(upstream['joint']['checkpoint']) == upstream['joint']['sha256']
        assert digest(selected['rank1_checkpoint']) == selected['rank1_sha256']
        admitted[fold] = dict(A_checkpoint=upstream['joint']['checkpoint'],A_sha256=upstream['joint']['sha256'],
            A_epoch=upstream['joint']['frozen']['epoch'],A_selection='rank1 dev H-JADE',
            B_checkpoint=selected['rank1_checkpoint'],B_sha256=selected['rank1_sha256'],
            B_epoch=selected['rank1_epoch'],B_selection='rank1 dev JADE seed2035',
            density_sha256=upstream['density']['sha256'],head_sha256=upstream['head']['sha256'],
            upstream_manifest_sha256=digest(run/fold/'UPSTREAM_FROZEN.json'),
            B_manifest_sha256=digest(run/fold/'B_FROZEN.json'))
    path = run/'FROZEN_MODELS.json'
    barrier = dict(status='ALL_AVAILABLE_MODELS_FROZEN_BEFORE_TEST',requested_folds=list(FOLDS),
        folds=admitted,missing_folds=missing,completed_fold_count=len(admitted),
        source_config_sha256=digest(run/'RUN_CONFIG.json'))
    if path.exists():
        previous=read(path)
        assert {k:v for k,v in previous.items() if k!='freeze_unix'} == barrier
        barrier=previous
    else:
        barrier['freeze_unix']=time.time()
        immutable(path,barrier)
    return barrier


def archive(run, config, push):
    from .report import build_report
    check_protection(config)
    result=build_report(run,DOC)
    if not push:
        return result
    assert subprocess.check_output(['git','branch','--show-current'],cwd=SOURCE,text=True).strip()==BRANCH
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=SOURCE,text=True).strip()
    assert not staged, 'unexpected staged user changes'
    subprocess.run(['git','add',str(DOC.relative_to(SOURCE))],cwd=SOURCE,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=SOURCE,check=True)
    if subprocess.call(['git','diff','--cached','--quiet'],cwd=SOURCE)!=0:
        subprocess.run(['git','commit','-m','Archive fourfold frozen confirmation and fivefold paired results'],cwd=SOURCE,check=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
    subprocess.run(['git','push','-u','origin',BRANCH],cwd=SOURCE,check=True)
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=SOURCE,text=True).split()[0]
    assert remote==head
    receipt=dict(status='DELIVERED',remote_sha256_commit=head,branch=BRANCH,
        report=f'https://github.com/WuYanXingege/RSJG/blob/{head}/{DOC.relative_to(SOURCE)}/README.md',
        summary=f'https://github.com/WuYanXingege/RSJG/blob/{head}/{DOC.relative_to(SOURCE)}/SUMMARY.csv',
        verified_remote_sha=head,unix=time.time())
    write(run/'DELIVERY.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=RUN)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--push-on-complete',action='store_true')
    args=parser.parse_args()
    run=args.run.resolve()
    run.mkdir(parents=True,exist_ok=True)
    with (run/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            config,inventory=prepare(run)
            if args.prepare_only:
                status(run,'PREPARED',inventory_sha256=digest(run/'ASSET_INVENTORY.json'))
                return
            for fold in FOLDS:
                if inventory[fold].get('status')=='MISSING_ORIGINAL_ASSETS':
                    status(run/fold,'MISSING_ORIGINAL_ASSETS',detail=inventory[fold])
                    continue
                upstream = execute_stage(run,config,fold,'upstream')
                if upstream.get('status') == 'MISSING_ORIGINAL_ASSETS':
                    inventory[fold] = upstream
                    continue
                execute_stage(run,config,fold,'dependency')
            barrier=freeze_all(run,config,inventory)
            for fold in FOLDS:
                if fold in barrier['folds']:
                    execute_stage(run,config,fold,'test')
            status(run,'COMPLETE' if len(barrier['folds'])==4 else 'PARTIAL_MISSING_ASSETS',
                   completed_new_folds=len(barrier['folds']),requested_new_folds=4)
            receipt=archive(run,config,args.push_on_complete)
            write(run/'ARCHIVE_RESULT.json',receipt)
        except BaseException as error:
            status(run,'INTERRUPTED_RESUME_SAME_RUN',error=repr(error),traceback=traceback.format_exc())
            raise


if __name__=='__main__':
    main()

"""Isolated native v2 upstream stages, writing only the new four-fold output.

The native source identity is retained: a fresh subprocess imports the exact
original workspace, while this adapter directs its existing workflow/state
writer to the new output. No original controller or queue is invoked.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid


SOURCE = Path(__file__).resolve().parents[1]
ROOT = SOURCE.parent
FORMAL = ROOT/'RSJG_EBJD_assets/experiments/bcep_ebjd_v2_integrated'
OUTPUT_ROOT = ROOT/'RSJG_EBJD_assets/experiments/joint_flow_fourfold_confirmation_20261122'
FOLDS = ('zara1', 'hotel', 'univ', 'eth')
STAGES = ('density', 'base-cache', 'head', 'banks', 'joint')
NATIVE_COMMIT = '468c5eddde48bdeec47c4bb3c39418352be16bb9'


class MissingUpstreamAssets(RuntimeError):
    """A fold may be skipped while the independent remaining folds continue."""
    def __init__(self, fold, missing):
        self.fold, self.missing = fold, list(missing)
        super().__init__(f'{fold}: missing upstream assets: '+', '.join(self.missing))


class UpstreamPaused(RuntimeError):
    """The same stage has a resumable last checkpoint/export, never a new run."""


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def _output(fold, output):
    if fold not in FOLDS:
        raise ValueError('only the four additional folds are authorized')
    output = Path(output).absolute()
    expected = OUTPUT_ROOT/fold/'upstream'
    if output.resolve() != expected.resolve() or output.is_symlink():
        raise ValueError('upstream output must be the isolated four-fold directory')
    return output


def _launch(fold, output, device, mode):
    output = _output(fold, output)
    required = [FORMAL/'FORMAL_RUN.json', FORMAL/'AUTHORIZED.yaml']
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise MissingUpstreamAssets(fold, missing)
    native = Path(_read(FORMAL/'FORMAL_RUN.json')['workspace'])
    if not native.is_dir():
        raise MissingUpstreamAssets(fold, [str(native)])
    command = [sys.executable, str(Path(__file__).resolve()), '--native', mode,
               '--fold', fold, '--output', str(output), '--device', str(device)]
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    if mode == 'inspect':
        result = subprocess.run(command, cwd=native, env=environment, text=True, capture_output=True)
        if result.returncode:
            if result.returncode == 3:
                error = json.loads(result.stdout)
                raise MissingUpstreamAssets(fold, error['missing'])
            raise RuntimeError(result.stderr or result.stdout)
        return json.loads(result.stdout)
    output.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, cwd=native, env=environment)
    status = _read(output/'UPSTREAM_STATUS.json')
    if result.returncode == 3:
        raise MissingUpstreamAssets(fold, status['missing'])
    if result.returncode == 75:
        raise UpstreamPaused(f'{fold}: '+status.get('stage', 'upstream'))
    if result.returncode:
        raise RuntimeError(f'{fold}: native upstream failed: {status}')
    return _read(output/'UPSTREAM_FROZEN.json')


def inspect_fold(fold):
    """Read-only metadata/asset inventory; creates no model, optimizer or output."""
    return _launch(fold, OUTPUT_ROOT/fold/'upstream', 'cpu', 'inspect')


def prepare_and_run_fold(fold, output, device):
    """Run/resume the one native sequence and return frozen paths and identities.

    ``output`` is exactly OUTPUT_ROOT/fold/upstream. MissingUpstreamAssets is
    distinguishable from execution failures, so callers can continue other folds.
    The caller owns GPU scheduling; this adapter never starts the original queue.
    """
    return _launch(fold, output, device, 'run')


def _native_contract(fold):
    # Imports must occur only after sys.path is pinned to the native workspace.
    from bcep.contracts import digest, require
    from bcep_v2.workflow import load_contract, recipe
    from bcep_v2.formal_manifest import budgets
    run = _read(FORMAL/'FORMAL_RUN.json')
    required = [Path(run['input_plan']), Path(run['workspace'])/'bcep_v2/workflow.py']
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise MissingUpstreamAssets(fold, missing)
    require(run['commit'] == NATIVE_COMMIT, 'original formal source identity changed')
    require(digest(FORMAL/'AUTHORIZED.yaml') == run['config_sha256'], 'original config changed')
    require(digest(run['input_plan']) == run['input_plan_sha256'], 'original input plan changed')
    plan = _read(run['input_plan'])
    split_path = Path(plan['folds'][fold]['split']['path'])
    missing = [str(p) for p in (split_path, Path(plan['split_lock'])) if not p.is_file()]
    if missing:
        raise MissingUpstreamAssets(fold, missing)
    ctx = load_contract(FORMAL/'AUTHORIZED.yaml', run['input_plan'], fold)
    cfg, split, folder, geometry, identity = ctx
    require(identity == run['folds'][fold]['identity'], 'native workspace no longer matches original sources')
    require(folder == FORMAL/fold, 'unexpected original asset directory')
    require(budgets(cfg, split) == run['folds'][fold]['caps'], 'native stage caps changed')
    recipes = {stage: recipe(cfg, stage) for stage in ('density', 'head', 'joint')}
    require([recipes[s]['seed'] for s in recipes] == [3102, 3102, 3201], 'training seeds changed')
    require([(recipes[s]['minimum'], recipes[s]['patience'], recipes[s]['max_epochs']) for s in recipes]
            == [(20, 15, 100), (5, 5, 20), (30, 15, 100)], 'native early-stop contract changed')
    missing = [str(row['path']) for row in split['records'] if not Path(row['path']).is_file()]
    if missing:
        raise MissingUpstreamAssets(fold, missing)
    return ctx, run, split_path


def _inspect_native(fold, ctx, run, split_path):
    import torch
    from bcep_v2.workflow import recipe, METRIC
    cfg, split, folder, geometry, identity = ctx
    stages = {}
    for stage in ('density', 'head', 'joint'):
        path = folder/stage
        result = dict(available=(path/'last.pt').is_file(),
                      complete_epoch_files=len(list(path.glob('epoch_*.pt'))),
                      recipe=recipe(cfg, stage), selector=METRIC[stage])
        if result['available']:
            state = torch.load(path/'last.pt', map_location='cpu', weights_only=False)
            result.update({k: state.get(k) for k in ('status', 'completed_epoch', 'epoch', 'update', 'cursor')})
            result['frozen'] = _read(path/'FROZEN.json') if (path/'FROZEN.json').is_file() else None
        stages[stage] = result
    return dict(fold=fold, original_folder=str(folder), native_workspace=run['workspace'],
        native_commit=run['commit'], split_path=str(split_path), identity=identity,
        train_windows=len(split['train_indices']), dev_windows=len(split['selection_dev_indices']),
        raw_record_files=len(split['records']), missing_assets=[], stages=stages,
        complete_base_caches=len(list((folder/'base_cache').glob('*/COMPLETE.json'))),
        complete_refined_banks=len(list((folder/'refined_banks').glob('*/COMPLETE.json'))),
        caps=run['folds'][fold]['caps'], heldout_future_materialized=False)


def _immutable(path, value):
    if Path(path).exists():
        if _read(path) != value:
            raise ValueError('immutable upstream configuration changed: '+str(path))
    else:
        _write(path, value)


def _copy_stage(source, target, identity, cfg, stage):
    """Copy all extant snapshots and partial state; never mutate the source."""
    import torch
    from bcep.contracts import digest, require
    from bcep_v2.workflow import recipe
    if target.exists() or not (source/'last.pt').is_file():
        return False
    state = torch.load(source/'last.pt', map_location='cpu', weights_only=False)
    require(state['identity'] == identity and state['recipe'] == recipe(cfg, stage),
            'existing original stage is not compatible with this same-run migration')
    require({'model', 'optimizer', 'scheduler', 'rng', 'cursor', 'order', 'history',
             'epoch_sums', 'epoch_steps', 'epoch_raw_sums', 'epoch_counts'} <= state.keys(),
            'original last is not a complete recovery state')
    require(all((source/row['file']).is_file() for row in state['history']),
            'original completed epoch files are missing; cannot claim full retention')
    files = sorted(p for p in source.rglob('*') if p.is_file())
    before = {str(p.relative_to(source)): digest(p) for p in files}
    temporary = target.with_name('.'+target.name+'.copy-'+uuid.uuid4().hex)
    shutil.copytree(source, temporary)
    require(before == {str(p.relative_to(source)): digest(p) for p in files},
            'original source changed during stage copy')
    require(before == {str(p.relative_to(temporary)): digest(p) for p in temporary.rglob('*') if p.is_file()},
            'migrated stage copy differs from original')
    temporary.replace(target)
    _write(target.parent/('IMPORTED_'+stage.upper()+'.json'), dict(
        source=str(source), source_identity=identity, file_sha256=before,
        completed_epoch=state['completed_epoch'], next_epoch=state['epoch'],
        update=state['update'], cursor=state['cursor'], independent_copies=True,
        original_unmodified=True, imported_unix=time.time()))
    return True


def _reuse_banks(ctx, original, previous, refined):
    from bcep.contracts import require
    from bcep_joint.common import local_seed
    from bcep_v2.bank import Bank, bank_identity
    from bcep_v2.workflow import SEEDS
    cfg, split, output, geometry, identity = ctx
    parent = 'refined_banks' if refined else 'base_cache'
    items = ([('train', 2035, None), ('selection_dev', 2035, None)] if refined else
             [('train', local_seed('bcep-v2-train-view', cfg['provider_seed'], view), view) for view in range(2)]
             + [('selection_dev', seed, None) for seed in SEEDS])
    for purpose, seed, view in items:
        source = original/parent/f'{purpose}_{seed}'
        target = output/parent/source.name
        if target.exists() or not (source/'COMPLETE.json').is_file():
            continue
        expected = bank_identity(split, purpose, seed, geometry, identity['sources'],
            previous['density']['selected'], previous['head']['selected'] if refined else None, view=view)
        # A complete, identity-matching bank is only read through its symlink.
        require(_read(source/'IDENTITY.json') == expected, 'existing original bank identity mismatch')
        Bank(source, expected, split)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source, target_is_directory=True)


def _run_native(fold, output, device, original_ctx, run, split_path):
    import torch
    from bcep.contracts import digest, require
    from bcep_formal.control import install
    from bcep_formal.data import FormalData
    from bcep_v2 import workflow
    from bcep_v2.bank import Bank, ExportPaused
    from bcep_v2.formal_manifest import stage_identity, stage_proof
    cfg, split, original, geometry, identity = original_ctx
    ctx = cfg, split, output, geometry, identity
    output.mkdir(parents=True, exist_ok=True)
    inventory = _inspect_native(fold, original_ctx, run, split_path)
    if not (output/'ORIGINAL_ASSET_INVENTORY.json').exists():
        _write(output/'ORIGINAL_ASSET_INVENTORY.json', inventory)
    _immutable(output/'ADAPTER_CONFIG.json', dict(fold=fold, original_folder=str(original),
        native_workspace=run['workspace'], native_commit=NATIVE_COMMIT,
        native_identity=identity, config_sha256=digest(FORMAL/'AUTHORIZED.yaml'),
        split_path=str(split_path), split_sha256=digest(split_path),
        stage_caps=run['folds'][fold]['caps'], stages=list(STAGES),
        recipes={s: workflow.recipe(cfg, s) for s in workflow.METRIC},
        selectors=workflow.METRIC, tie_break='strict earlier completed epoch',
        retention='ALL completed epochs + original rank1/best + last + machine-readable Top5',
        native_state_writer_sha256=digest(Path(run['workspace'])/'bcep_v2/state.py'),
        native_workflow_sha256=digest(Path(run['workspace'])/'bcep_v2/workflow.py'),
        migrated_stage='copy complete surviving files; preserve original state/identity/RNG',
        original_assets_read_only=True, heldout_future_access=False))
    install()
    # Independent process-local guard. No heldout item is needed in these stages.
    original_item = FormalData.item
    def guarded_item(self, index, purpose, truth=True):
        if purpose == 'heldout_confirmation':
            raise RuntimeError('upstream stages cannot access heldout items')
        return original_item(self, index, purpose, truth)
    FormalData.item = guarded_item
    native_run_stage = workflow.run_stage
    def progress_run_stage(*args, **kwargs):
        kwargs['record_progress'] = True
        return native_run_stage(*args, **kwargs)
    workflow.run_stage = progress_run_stage
    previous = {}
    reuse = {}
    for stage in STAGES:
        _write(output/'UPSTREAM_STATUS.json', dict(status='RUNNING', fold=fold, stage=stage,
            phase='prepare', native_workspace=run['workspace'], unix=time.time(), pid=os.getpid()))
        if stage in workflow.METRIC:
            expected = stage_identity(ctx, stage, previous)
            target, source = output/stage, original/stage
            if not target.exists() and (source/'last.pt').is_file():
                state = torch.load(source/'last.pt', map_location='cpu', weights_only=False)
                require(state['identity'] == expected, 'original upstream dependency identity mismatch')
                if state['status'] == 'COMPLETE' and (source/'FROZEN.json').is_file():
                    stage_proof(original_ctx, stage, previous)
                    target.symlink_to(source, target_is_directory=True)
                    reuse[stage] = 'complete native stage, read-only symlink'
                else:
                    _copy_stage(source, target, expected, cfg, stage)
                    reuse[stage] = 'complete surviving native stage copied; same last resumed'
            complete = False
            if (target/'last.pt').is_file():
                state = torch.load(target/'last.pt', map_location='cpu', weights_only=False)
                require(state['identity'] == expected, 'new output stage identity mismatch')
                complete = state['status'] == 'COMPLETE' and (target/'FROZEN.json').is_file()
            if target.is_symlink():
                require(complete and (target/'FROZEN.json').is_file(), 'refusing to train through original symlink')
            if complete:
                result = dict(stage=stage, status='COMPLETE', reused=True)
            else:
                _write(output/'UPSTREAM_STATUS.json', dict(status='RUNNING', fold=fold, stage=stage,
                    phase='native_training', unix=time.time(), pid=os.getpid()))
                args = SimpleNamespace(stage=stage, device=str(device),
                    until_update=run['folds'][fold]['caps'][stage], resume=(target/'last.pt').is_file())
                result = workflow.execute(args, cfg, split, output, geometry, identity)
                if result['status'] != 'COMPLETE':
                    _write(output/'UPSTREAM_STATUS.json', dict(status='PAUSED_RESUMABLE', fold=fold,
                        stage=stage, result=result, unix=time.time()))
                    return 75
        else:
            _reuse_banks(ctx, original, previous, refined=stage == 'banks')
            _write(output/'UPSTREAM_STATUS.json', dict(status='RUNNING', fold=fold, stage=stage,
                phase='native_observed_only_export', unix=time.time(), pid=os.getpid()))
            try:
                result = workflow.execute(SimpleNamespace(stage=stage, device=str(device),
                    until_update=None, resume=False), cfg, split, output, geometry, identity)
            except ExportPaused as exc:
                _write(output/'UPSTREAM_STATUS.json', dict(status='PAUSED_RESUMABLE', fold=fold,
                    stage=stage, bank=str(exc.folder), next_index=exc.next_index, unix=time.time()))
                return 75
        proof = stage_proof(ctx, stage, previous)
        previous[stage] = proof
        _immutable(output/'receipts'/f'{stage}.json', proof)
        _write(output/'UPSTREAM_STATUS.json', dict(status='STAGE_COMPLETE', fold=fold,
            stage=stage, result=result, unix=time.time(), completed_stages=list(previous)))
        print(json.dumps(dict(fold=fold, upstream_stage=stage, status='COMPLETE')), flush=True)
        if str(device).startswith('cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
    reuse = {stage: ('complete native stage, read-only symlink' if (output/stage).is_symlink()
                     else 'complete surviving native stage copied; same last resumed')
             for stage in workflow.METRIC
             if (output/stage).is_symlink() or (output/('IMPORTED_'+stage.upper()+'.json')).is_file()}
    paths = {'split': str(split_path)}
    frozen = {}
    for stage in workflow.METRIC:
        rank1 = previous[stage]['ranks'][0]
        checkpoint = output/stage/rank1['file']
        paths[stage] = str(checkpoint)
        frozen[stage] = dict(checkpoint=str(checkpoint), sha256=digest(checkpoint),
            frozen=previous[stage]['selected'], stage_directory=str(output/stage))
    banks = {}
    for purpose, key in [('train', 'train_bank'), ('selection_dev', 'dev_bank')]:
        folder = output/'refined_banks'/f'{purpose}_2035'
        bank_identity = _read(folder/'IDENTITY.json')
        bank = Bank(folder, bank_identity, split)
        paths[key] = str(folder)
        banks[purpose] = dict(path=str(folder), sha256=bank.hash, identity=bank_identity)
    manifest = dict(status='COMPLETE', fold=fold, output=str(output), paths=paths,
        split_path=str(split_path), split_sha256=digest(split_path),
        config_path=str(FORMAL/'AUTHORIZED.yaml'), config_sha256=digest(FORMAL/'AUTHORIZED.yaml'),
        native_workspace=run['workspace'], native_commit=NATIVE_COMMIT,
        identity=identity, geometry=geometry, banks=banks, reuse=reuse, **frozen)
    # The freeze identity is deterministic; recovery must never change rank1.
    _immutable(output/'UPSTREAM_FROZEN.json', manifest)
    _write(output/'UPSTREAM_STATUS.json', dict(status='COMPLETE', fold=fold,
        stage='joint', frozen_manifest_sha256=digest(output/'UPSTREAM_FROZEN.json'), unix=time.time()))
    return 0


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', choices=('inspect', 'run'), required=True)
    parser.add_argument('--fold', choices=FOLDS, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    output = _output(args.fold, args.output)
    run = _read(FORMAL/'FORMAL_RUN.json')
    native = Path(run['workspace'])
    sys.path.insert(0, str(native))
    try:
        ctx, run, split_path = _native_contract(args.fold)
        if args.native == 'inspect':
            print(json.dumps(_inspect_native(args.fold, ctx, run, split_path)))
            return 0
        output.mkdir(parents=True, exist_ok=True)
        with (output/'.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return _run_native(args.fold, output, args.device, ctx, run, split_path)
    except MissingUpstreamAssets as exc:
        status = dict(status='MISSING_ASSETS', fold=args.fold, missing=exc.missing, unix=time.time())
        if args.native == 'inspect':
            print(json.dumps(status))
        else:
            _write(output/'UPSTREAM_STATUS.json', status)
        return 3
    except Exception as exc:
        if args.native == 'run':
            prior = _read(output/'UPSTREAM_STATUS.json') if (output/'UPSTREAM_STATUS.json').exists() else {}
            _write(output/'UPSTREAM_STATUS.json', dict(prior, status='FAILED', fold=args.fold,
                                                     error=repr(exc), unix=time.time()))
        raise


if __name__ == '__main__':
    raise SystemExit(_main())

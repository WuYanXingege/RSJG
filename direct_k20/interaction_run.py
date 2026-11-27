"""Train two frozen-student endpoint refiners and enforce the joint test gate."""

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from bcep.contracts import canonical_hash, digest, require

from .common import ASSETS, DOC as ORIGINAL_DOC, ENVIRONMENT, METRICS, RUN as ORIGINAL_RUN
from .common import SEEDS, SOURCE, immutable, read, runtime_sources, status, write


BASE = '03849f58392865ccad823e4cb0b58e3ef0778132'
RUN = ASSETS / 'direct_k20_interaction_20261127'
DOC = ORIGINAL_DOC.parent / 'direct_k20_interaction_20261127'
BRANCH = 'experiment/direct-k20-endpoint-interaction-20261127'
FOLDS = ('hotel', 'univ')
EPOCHS = {'hotel': 24, 'univ': 16}
GPU_FRACTION = .82
GPU_MINIMUM_FREE_MIB = 12500
OOM = 76
YIELD = 75


def _split(fold):
    manifest = ASSETS / 'joint_flow_fourfold_confirmation_20261122' / fold / 'UPSTREAM_FROZEN.json'
    values = read(manifest)
    path = Path(values.get('split_path') or values['paths']['split'])
    split = read(path)
    require(split['fold'] == fold and split['unit'] == 'm'
            and split['pred_length'] == 12 and split['dt'] == .4, 'wrong fold or geometry')
    return split, digest(manifest)


def _population(split, purpose):
    indices = split[purpose + '_indices']
    require(indices and len(set(indices)) == len(indices), f'invalid {purpose} index')
    members = [split['members'][index] for index in indices]
    return dict(windows=len(members), agents=sum(len(m['agent_ids']) for m in members),
                member_sha256=canonical_hash(members))


def _fold_identity(row):
    fold = row['fold']
    frozen = read(ORIGINAL_DOC / 'FROZEN_STUDENTS.json')['folds'][fold]
    receipt_path = ORIGINAL_RUN / fold / 'TRAINING_COMPLETE.json'
    receipt = read(receipt_path)
    checkpoint = Path(receipt['selected_checkpoint'])
    epoch = EPOCHS[fold]
    require(int(row['epoch']) == frozen['selected_epoch'] == receipt['selected_epoch'] == epoch
            and int(row['rank']) == 1 and receipt['status'] == 'COMPLETE'
            and frozen['training_receipt_sha256'] == digest(receipt_path)
            and digest(checkpoint) == row['checkpoint_sha256'] == frozen['selected_sha256']
            == receipt['selected_sha256'], 'S0 checkpoint does not match the fixed archive')
    selected = next(item for item in receipt['dev_scores'] if item['epoch'] == epoch)
    reference = {metric: float(row[metric+'_m']) for metric in METRICS}
    require(selected['seed'] == 2035
            and all(reference[metric] == selected['metrics'][metric] for metric in METRICS),
            'S0 reference scores differ')
    scales_path = ORIGINAL_RUN / fold / 'cache/teacher_cache/TRAIN_B_SCALES.json'
    values = read(scales_path)
    require(values['purpose'] == 'train' and values['seed'] == 2035
            and values['scales'] == receipt['identity']['scales_m']
            and digest(scales_path) == receipt['identity']['cache']['scale_receipt_sha256'],
            'training metric normalizers differ from S0')
    split, upstream_sha = _split(fold)
    train = _population(split, 'train')
    dev = _population(split, 'selection_dev')
    require(dev['windows'] == selected['metrics']['scene_count']
            and dev['agents'] == selected['metrics']['agent_count'], 'dev membership differs')
    return dict(
        fold=fold, S0_checkpoint=str(checkpoint), S0_sha256=frozen['selected_sha256'], S0_epoch=epoch,
        S0_training_receipt_sha256=digest(receipt_path), S0_dev_reference=reference,
        upstream_sha256=upstream_sha, scales_m=values['scales'], scales_sha256=digest(scales_path),
        split_sha256=canonical_hash(split), train_manifest_sha256=train['member_sha256'],
        train_windows=train['windows'], train_agents=train['agents'],
        dev_manifest_sha256=dev['member_sha256'], dev_windows=dev['windows'], dev_agents=dev['agents'],
    )


def prepare():
    table = ORIGINAL_DOC / 'SELECTION.csv'
    with table.open(newline='') as stream:
        picked = [row for row in csv.DictReader(stream)
                  if row['fold'] in FOLDS and row['selected'] == 'True']
    require(len(picked) == len(FOLDS) and {row['fold'] for row in picked} == set(FOLDS),
            'missing unique selected S0 per fold')
    identities = {row['fold']: _fold_identity(row) for row in picked}
    config = dict(
        base_commit=BASE, source_sha256=runtime_sources(), baseline_selection_sha256=digest(table),
        folds=list(FOLDS), inputs=identities,
        architecture=dict(max_observed_neighbors=4, max_cpa_neighbors=4, width=64, activation='SiLU',
            synchronous_update=True, zero_output_initialization=True, zero_attention_option=True,
            closest_time_epsilon=1e-8, relative_stop_speed_mps=1e-3,
            trend='difference of adjacent three-step velocity means divided by 3*dt',
            neighbor_ties='stable geometric keys, with no agent identifiers'),
        recipe=dict(epochs=12, only_evaluated_epoch=12, seed=3201, trajectory_sampler_seed=3202,
            optimizer='AdamW', lr=3e-4, weight_decay=1e-4, grad_clip=5.,
            precision='FP32', logical_windows=16, microbatch_windows=1, K=20, frames=12, dt=.4,
            DDIM_steps=20, eta=0., checkpoint_steps=True,
            loss='JFDE/bJF + minFDE/bMF + 0.5*(JADE_HT/bJA + minADE_HT/bMA) + guard/bMF',
            guard='valid-person mean of ReLU(new endpoint minimum - detached S0 endpoint minimum)',
            trajectory_sample='uniform without replacement, max(1,(logical_windows+4)//8) draws',
            dense_recovery='native K20 forward minima and 5-slot VJP replay on N>=24 after capacity OOM'),
        development=dict(purpose='selection_dev', seed=2035, tolerance_m=1e-6,
            gate='each fold strictly lower all JADE/JFDE and targeted JFDE by >1e-6 m; all minADE/minFDE rises <=1e-6 m',
            targets={'hotel': 'N2_8', 'univ': 'N9plus'}),
        test=dict(purpose='heldout_confirmation', seeds=list(SEEDS),
            require_both_folds_pass=True, exploratory=True, allow_identity_checked_baseline_cache=True),
        retention='keep all 12 complete epochs and last; sole epoch 12 gets a final dev score',
        device=dict(mode='serial on one available GPU', allocator_fraction=GPU_FRACTION,
                    minimum_free_mib=GPU_MINIMUM_FREE_MIB),
        environment=ENVIRONMENT, reproduce='python -m direct_k20.interaction_run',
    )
    immutable(RUN / 'RUN_CONFIG.json', config)
    return config


def _load(fold, identity, device):
    import torch

    from .assets import BASE_COMMIT, TEACHER_SEED, FoldAssets, _regular
    from .model import Student

    folder = RUN / fold
    split, upstream_sha = _split(fold)
    require(upstream_sha == identity['upstream_sha256']
            and canonical_hash(split) == identity['split_sha256'], 'locked fold manifest changed')
    native, model, provider, dependence, upstream, baseline, geometry = _regular(fold, folder, device)
    require(digest(identity['S0_checkpoint']) == identity['S0_sha256'], 'S0 weights changed')
    saved = torch.load(identity['S0_checkpoint'], map_location='cpu', weights_only=False)
    require(saved['completed_epoch'] == identity['S0_epoch'], 'wrong S0 epoch')
    baseline = {metric: float(baseline[metric]) for metric in METRICS}
    upstream.update(fold=fold, source_report_commit=BASE_COMMIT, split_sha256=identity['split_sha256'],
                    dev_B_seed=TEACHER_SEED, dev_B=baseline)
    recorded = saved['identity']['upstream']
    require(upstream == {key: value for key, value in recorded.items() if key != 'historical'},
            'S0 shared frozen assets differ')
    for module in (model, provider, dependence):
        module.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    student = Student()
    student.load_state_dict(saved['student'], strict=True)
    student.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    return FoldAssets(fold, native.split, native.data, native.banks['train'], model, provider,
                      dependence, recorded, baseline, None, device, geometry), student


def _worker(stage, fold):
    import torch
    from bcep_joint.common import seed_all
    from bcep_v2.runtime import configure_cuda

    from .endpoint_interaction import EndpointInteraction

    parent = os.getppid()
    yielded = False

    def stop(_signal, _frame):
        nonlocal yielded
        yielded = True

    def should_yield():
        return yielded or os.getppid() != parent

    signal.signal(signal.SIGUSR1, stop)
    config = read(RUN / 'RUN_CONFIG.json')
    require(runtime_sources() == config['source_sha256'], 'execution source changed')
    device = torch.device('cuda')
    torch.set_num_threads(2)
    configure_cuda()
    torch.cuda.set_per_process_memory_fraction(config['device']['allocator_fraction'])
    seed_all(config['recipe']['seed'])
    folder = RUN / fold
    identity = config['inputs'][fold]
    try:
        assets, student = _load(fold, identity, device)
        # Set this explicitly so fixture-independent upstream module construction
        # cannot shift the new refiner initialization.
        seed_all(config['recipe']['seed'])
        refiner = EndpointInteraction().to(device=device, dtype=torch.float32)
        if stage == 'train':
            from .interaction_train import train

            recover = (folder / 'DDIM_MEMORY_RECOVERY.json').is_file()
            result = train(assets, student, refiner, folder / 'training', identity,
                           dense_agents=24, slot_chunk=5 if recover else 0, should_yield=should_yield)
        else:
            from .interaction_eval import export_fold

            identity = read(folder / 'R12_IDENTITY.json')
            require(digest(identity['R12_checkpoint']) == identity['R12_sha256'], 'R12 weights changed')
            saved = torch.load(identity['R12_checkpoint'], map_location='cpu', weights_only=False)
            require(saved['completed_epoch'] == 12, 'only fixed R12 may be evaluated')
            refiner.load_state_dict(saved['interaction'], strict=True)
            refiner.eval().requires_grad_(False)
            purpose = 'selection_dev' if stage == 'dev' else 'heldout_confirmation'
            result = export_fold(assets, student, refiner, RUN, identity, purpose=purpose,
                                 should_yield=should_yield)
        if result is None:
            return YIELD
        immutable(folder / _output(stage), result)
    except torch.OutOfMemoryError as error:
        write(folder / 'GPU_OOM.json', dict(stage=stage, unix=time.time(), error=str(error),
              peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
              peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
        traceback.print_exc()
        return OOM
    return 0


def _output(stage):
    return {'train': 'TRAINING_COMPLETE.json', 'dev': 'DEV_EXPORT_COMPLETE.json',
            'test': 'TEST_EXPORT_COMPLETE.json'}[stage]


def _gpu():
    result = subprocess.check_output(['nvidia-smi', '-i', '0',
        '--query-gpu=uuid,memory.total,memory.free', '--format=csv,noheader,nounits'], text=True)
    [(uuid, total, free)] = csv.reader(result.splitlines(), skipinitialspace=True)
    return dict(uuid=uuid, total_mib=int(total), free_mib=int(free))


def execute(stage, fold, config):
    folder = RUN / fold
    folder.mkdir(exist_ok=True)
    receipt = folder / _output(stage)
    if receipt.exists():
        return
    cmd = [sys.executable, '-m', 'direct_k20.interaction_run', '--worker', stage, fold]
    with (ASSETS / 'auxiliary_gpu.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            status(RUN, 'WAITING_FOR_GPU_LOCK', fold=fold, stage=stage)
            fcntl.flock(lock, fcntl.LOCK_EX)
        for attempt in range(1, 4):
            gpu = _gpu()
            while gpu['free_mib'] < config['device']['minimum_free_mib']:
                status(RUN, 'WAITING_FOR_GPU_MEMORY', fold=fold, stage=stage, gpu=gpu)
                time.sleep(30)
                gpu = _gpu()
            status(RUN, 'STARTING_WORKER', stage=stage, fold=fold, attempt=attempt, gpu=gpu)
            with (RUN / f'{fold}_{stage}.log').open('a', buffering=1) as log:
                child = subprocess.Popen(cmd, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT,
                                         pass_fds=(lock.fileno(),))
                write(RUN / 'ACTIVE_CHILD.json', dict(pid=child.pid, stage=stage, fold=fold,
                      attempt=attempt, unix=time.time(), gpu=gpu, command=cmd))
                status(RUN, 'WORKER_RUNNING', fold=fold, stage=stage, pid=child.pid, gpu=gpu)
                code = child.wait()
            if code == 0:
                require(receipt.exists(), f'{fold} {stage} worker omitted its completion receipt')
                return
            if code == OOM and stage == 'train':
                marker = folder / 'DDIM_MEMORY_RECOVERY.json'
                if not marker.exists():
                    immutable(marker, dict(window_agents_ge=24, slot_chunk=5,
                              oom=read(folder / 'GPU_OOM.json')))
            status(RUN, 'RECOVERING_SAME_RUN', stage=stage, fold=fold, attempt=attempt, exit_code=code)
            time.sleep(20)
    raise RuntimeError(f'{fold} {stage} stopped after three attempts, see {fold}_{stage}.log')


def _identities(config):
    result = {}
    for fold in FOLDS:
        folder = RUN / fold
        receipt = folder / 'TRAINING_COMPLETE.json'
        trained = read(receipt)
        require(trained['completed_epochs'] == 12
                and digest(trained['R12_checkpoint']) == trained['R12_sha256'],
                'R12 training or checkpoint incomplete')
        identity = config['inputs'][fold] | {
            name: trained[name] for name in ('R12_checkpoint', 'R12_sha256')
        } | dict(config_sha256=digest(RUN / 'RUN_CONFIG.json'), training_receipt_sha256=digest(receipt))
        immutable(folder / 'R12_IDENTITY.json', identity)
        result[fold] = identity
    return result


def _freeze_test(config, identities):
    dev = RUN / 'dev_evaluation/SCORE_RECEIPT.json'
    require(read(dev)['decision']['test_authorized'], 'both development gates must pass before test')
    core = dict(status='BOTH_DEV_GATES_PASSED', folds=identities, seeds=list(SEEDS),
                dev_score_sha256=digest(dev), config_sha256=digest(RUN / 'RUN_CONFIG.json'),
                source_sha256=config['source_sha256'])
    path = RUN / 'FROZEN_R12_TEST.json'
    if path.exists():
        require(all(read(path)[key] == value for key, value in core.items()), 'test freeze changed')
    else:
        immutable(path, dict(core, freeze_unix=time.time()))


def _publish():
    from joint_flow_fourfold.report import public_metadata
    from .interaction_eval import build_report

    DOC.mkdir(parents=True, exist_ok=True)
    for fold in FOLDS:
        (DOC / fold).mkdir(exist_ok=True)
        for source, name in (
            (RUN / fold / 'training/checkpoints/TOP5.json', 'TOP5.json'),
            (RUN / fold / 'R12_IDENTITY.json', 'R12_IDENTITY.json'),
        ):
            write(DOC / fold / name, public_metadata(read(source)))
        recovery = RUN / fold / 'DDIM_MEMORY_RECOVERY.json'
        if recovery.exists():
            write(DOC / fold / recovery.name, public_metadata(read(recovery)))
    write(DOC / 'START.json', public_metadata(read(RUN / 'START.json')))
    build_report(RUN, DOC)
    require(subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == BRANCH,
            'wrong archive branch')
    require(not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], text=True).strip(),
            'unexpected staged changes')
    subprocess.run(['git', 'add', str(DOC.relative_to(SOURCE))], check=True)
    subprocess.run(['git', 'diff', '--cached', '--check'], check=True)
    if subprocess.call(['git', 'diff', '--cached', '--quiet']):
        subprocess.run(['git', 'commit', '-m', 'Archive fixed HOTEL and UNIV endpoint interaction results'], check=True)
    subprocess.run(['git', 'push', '-u', 'origin', BRANCH], check=True)
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/' + BRANCH], text=True).split()[0]
    require(sha == remote, 'published archive hash differs')
    write(RUN / 'DELIVERY.json', dict(sha=sha, verified_remote_sha=remote, branch=BRANCH))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', nargs=2, metavar=('STAGE', 'FOLD'))
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.worker:
        stage, fold = args.worker
        require(stage in ('train', 'dev', 'test') and fold in FOLDS, 'invalid worker')
        return _worker(stage, fold)
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = prepare()
        if args.prepare_only:
            status(RUN, 'PREPARED')
            return 0
        if (RUN / 'DELIVERY.json').exists():
            return 0
        require(all(os.environ.get(key) == value for key, value in ENVIRONMENT.items()), 'wrong launch environment')
        if not (RUN / 'START.json').exists():
            write(RUN / 'START.json', dict(unix=time.time(), pid=os.getpid(), gpu=_gpu(),
                  command=[sys.executable, '-m', 'direct_k20.interaction_run'],
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()))
        for fold in FOLDS:
            execute('train', fold, config)
        identities = _identities(config)
        for fold in FOLDS:
            execute('dev', fold, config)

        import torch
        from .interaction_eval import score_all
        from .interaction_train import record_r12_metrics

        torch.set_num_threads(1)
        splits = {fold: _split(fold)[0] for fold in FOLDS}
        status(RUN, 'SCORING_DEVELOPMENT')
        dev = score_all(splits, RUN, identities, purpose='selection_dev')
        for fold in FOLDS:
            final = dev['groups'][fold]['all']
            record_r12_metrics(RUN / fold / 'training', {m: final[f'R12_{m}_m'] for m in METRICS})
        if dev['decision']['test_authorized']:
            require(runtime_sources() == config['source_sha256'], 'source changed before test freeze')
            _freeze_test(config, identities)
            for fold in FOLDS:
                execute('test', fold, config)
            status(RUN, 'SCORING_TEST')
            score_all(splits, RUN, identities, purpose='heldout_confirmation')
        require(runtime_sources() == config['source_sha256'], 'source changed before report')
        write(RUN / 'WALL_TIME.json', dict(seconds=time.time()-read(RUN / 'START.json')['unix']))
        status(RUN, 'ARCHIVING', decision=dev['decision'])
        _publish()
        status(RUN, 'COMPLETE', folds=list(FOLDS), test_executed=dev['decision']['test_authorized'])
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        if '--worker' not in sys.argv:
            status(RUN, 'INTERRUPTED_RESUME_SAME_RUN', traceback=traceback.format_exc())
        raise

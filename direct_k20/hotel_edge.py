"""Run HOTEL S24 to S28, yielding the GPU to the original Direct-K20 job."""

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from bcep.contracts import canonical_hash, digest, require

from .common import ASSETS, ENVIRONMENT, METRICS, RUN as MAIN, SOURCE, immutable, read, runtime_sources, status, write


BASE = '52547bb2acbc3a8ab8f52edffeac274e30ae238b'
INITIAL_RUN_SOURCE = '0f843f9c1649633f0369de3c48b85d8a4009e0ad'
BRANCH = 'experiment/hotel-direct-k20-edge-s28-20261126'
RUN = ASSETS / 'hotel_edge_s28_20261126'
DOC = SOURCE / 'docs/research/continuous_joint_flow/hotel_edge_s28_20261126'
MAIN_SOURCE = SOURCE.parent / 'RSJG_direct_k20_20261126'
ORIGINAL = MAIN / 'hotel'
TEACHER_CACHE = ORIGINAL / 'cache/teacher_cache'
UPSTREAM = ASSETS / 'joint_flow_fourfold_confirmation_20261122/hotel/UPSTREAM_FROZEN.json'
S24_SHA256 = '43be503d852c9d713f464810f0cc9263e08d1b5d0310a5f9b99ff28872deb87a'
GPU_FRACTION = .18
START_FREE_MIB = 6144
MAIN_START_MIB = 12500
YIELD_FREE_MIB = 2048
YIELD_CODE = 75
OOM_CODE = 76
POLL_SECONDS = 2


def _optional(path):
    return read(path) if path.exists() else {}


def _main_source_identity():
    config = read(MAIN / 'RUN_CONFIG.json')
    for relative, sha in config['source_sha256'].items():
        require(digest(MAIN_SOURCE / relative) == sha, f'main source changed: {relative}')
    require(config['minimum_free_gpu_mib'] == MAIN_START_MIB, 'main GPU launch threshold changed')
    return digest(MAIN / 'RUN_CONFIG.json')


def _members(split, purpose):
    indices = split[purpose + '_indices']
    require(bool(indices) and len(set(indices)) == len(indices), f'invalid {purpose} population')
    return [split['members'][index] for index in indices]


def _check_run_source(config):
    recovery = _optional(RUN / 'RECOVERY_CONFIG.json')
    if recovery:
        require(recovery['original_config_sha256'] == digest(RUN / 'RUN_CONFIG.json'),
                'the training identity cannot change during GPU recovery')
    expected = recovery.get('source_sha256', config['source_sha256'])
    require(runtime_sources() == expected, 'HOTEL edge source differs from the pinned execution')
    return recovery


def _pin_slot_replay(config):
    original = config['source_sha256']
    current = runtime_sources()
    edited = {path for path in original.keys() | current.keys() if original.get(path) != current.get(path)}
    require(edited == {'direct_k20/hotel_edge.py', 'direct_k20/hotel_edge_train.py',
                       'direct_k20/hotel_edge_eval.py'},
            'GPU recovery may only change the isolated HOTEL experiment modules')
    for path in sorted(edited):
        initial = subprocess.check_output(['git', 'show', INITIAL_RUN_SOURCE + ':' + path])
        require(hashlib.sha256(initial).hexdigest() == original[path], 'initial execution source differs')
    rollback = read(RUN / 'training/ROLLBACK.json')
    require(rollback['reason'] == 'cuda_oom' and rollback['activation_offload']
            and rollback['sampled_for_ddim'] and rollback['interrupted_window_agents'] >= 24,
            'no dense DDIM allocator failure to recover')
    resume_path = RUN / 'training/checkpoints/last.pt'
    immutable(RUN / 'RECOVERY_CONFIG.json', dict(
        original_config_sha256=digest(RUN / 'RUN_CONFIG.json'),
        original_source_commit=INITIAL_RUN_SOURCE,
        recovery_source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256=current, changed_sources=sorted(edited),
        retained_resume_sha256=digest(resume_path), failed_batch=rollback,
        training_replay=dict(slot_chunk=5, minimum_agents=24, global_worlds=20, full_ddim_steps=20,
            objective='differentiable hard reductions over the native full K20 path; VJP recomputed in slot chunks',
            path_atol=1e-5, path_rtol=1e-6),
        activation_offload='off initially after chunk recovery; enabled only on further OOM',
        reproduce='python -m direct_k20.hotel_edge --resume-sliced-ddim',
    ))


def prepare(resume_sliced_ddim=False):
    receipt = ORIGINAL / 'TRAINING_COMPLETE.json'
    finished = read(receipt)
    checkpoint = Path(finished['selected_checkpoint'])
    require(finished['completed_epochs'] == finished['selected_epoch'] == 24
            and finished['selected_sha256'] == S24_SHA256 == digest(checkpoint),
            'the original HOTEL S24 identity differs')
    candidate = ORIGINAL / 'training/DEV_EPOCH_024.json'
    dev = read(candidate)
    ranked = [row for row in finished['dev_scores'] if row['epoch'] == 24]
    require(len(ranked) == 1 and ranked[0] == dev and dev['seed'] == 2035,
            'archived S24 selection_dev score differs')
    upstream = read(UPSTREAM)
    split_path = Path(upstream.get('split_path') or upstream['paths']['split'])
    split = read(split_path)
    require(split['fold'] == 'hotel' and split['unit'] == 'm' and split['pred_length'] == 12,
            'unexpected HOTEL split or trajectory units')
    train, development = (_members(split, purpose) for purpose in ('train', 'selection_dev'))
    counts = [len(member['agent_ids']) for member in development]
    n_train = [len(member['agent_ids']) for member in train]
    require(len(development) == dev['windows'] and sum(counts) == dev['metrics']['agent_count'],
            'S24 receipt does not cover this development split')
    manifest = read(TEACHER_CACHE / 'MANIFEST.json')
    scales_path = TEACHER_CACHE / 'TRAIN_B_SCALES.json'
    scales = read(scales_path)['scales']
    train_sha = canonical_hash(train)
    require(manifest['identity']['fold'] == 'hotel'
            and manifest['identity']['purpose'] == 'train'
            and manifest['identity']['seed'] == 2035
            and manifest['identity']['train_members_sha256'] == train_sha
            and [row['index'] for row in manifest['rows']] == split['train_indices'],
            'frozen teacher cache is not the official HOTEL train population')
    prior = _optional(RUN / 'RUN_CONFIG.json')
    config = dict(
        code_base=BASE, source_sha256=prior.get('source_sha256') or runtime_sources(), fold='hotel',
        main_config_sha256=_main_source_identity(), main_source_commit='1a7444b0ec2f2b0bf1c9e31c8e7f04decddf2a91',
        S24_checkpoint=str(checkpoint), S24_sha256=S24_SHA256,
        S24_training_receipt_sha256=digest(receipt), S24_dev_score_sha256=digest(candidate),
        S24_dev_reference={metric: dev['metrics'][metric] for metric in METRICS},
        upstream_sha256=digest(UPSTREAM), split_sha256=canonical_hash(split),
        train_manifest_sha256=train_sha, dev_manifest_sha256=canonical_hash(development),
        train_windows=len(train), train_agents=sum(n_train),
        eligible_train_windows=sum(2 <= n <= 8 for n in n_train),
        dev_windows=len(development), dev_agents=sum(counts),
        teacher_manifest_sha256=digest(TEACHER_CACHE / 'MANIFEST.json'),
        teacher_scales_sha256=digest(scales_path), scales_m=scales,
        recipe=dict(initial_epoch=24, completed_epochs=[25, 26, 27, 28], edge_only=True,
            optimizer='AdamW', lr=1e-4, weight_decay=1e-4, clip_grad_norm=5.,
            training_seed=3201, trajectory_sampler_seed=3202, logical_windows=16,
            gradient_accumulation='one full window', precision='FP32',
            trajectory_sampling='uniform without replacement, max(1,(batch_windows+4)//8)',
            K=20, frames=12, ddim_steps=20, eta=0., checkpoint_steps=True,
            objective='original epochs 13–24 loss + mean eligible ReLU(student JFDE - cached B JFDE) / train bJF',
            extra_N_min=2, extra_N_max=8, extra_weight=1.,
            singleton='all original denominators; no edge backward; all-singleton batches skip AdamW'),
        evaluation=dict(purpose='selection_dev', seed=2035, pair=['S24', 'S28'],
            criterion='S28-S24 JFDE 2–8 < -1e-6 m and all four S28-S24 all <= 1e-6 m',
            historical_test_inspired=True, test_windows_read=0),
        retention=dict(epoch_snapshots='all four new completed epochs, plus protected S24 parent',
            ranking='fixed S28 evaluation only; intermediate epochs retained without a selector score',
            recovery='atomic after each complete logical batch, including skipped all-singleton batches'),
        GPU=dict(initial_allocator_fraction=GPU_FRACTION, start_minimum_free_mib=START_FREE_MIB,
            go_idle_below_free_mib=YIELD_FREE_MIB, main_start_minimum_free_mib=MAIN_START_MIB,
            max_auxiliary_workers=1, oom_recovery='same durable update; CPU activation offload when needed'),
        environment=ENVIRONMENT, reproduce='python -m direct_k20.hotel_edge',
    )
    immutable(RUN / 'RUN_CONFIG.json', config)
    if resume_sliced_ddim and not (RUN / 'RECOVERY_CONFIG.json').exists():
        require(bool(prior), 'DDIM recovery is only for an existing experiment')
        _pin_slot_replay(config)
    _check_run_source(config)
    return config


def _assets(config, device):
    import torch
    from .assets import BASE_COMMIT, TEACHER_SEED, FoldAssets, _regular

    require(digest(UPSTREAM) == config['upstream_sha256'] and
            digest(TEACHER_CACHE / 'MANIFEST.json') == config['teacher_manifest_sha256'] and
            digest(TEACHER_CACHE / 'TRAIN_B_SCALES.json') == config['teacher_scales_sha256'],
            'frozen HOTEL input identity changed')
    device = torch.device(device)
    native, model, provider, dependence, identity, baseline, geometry = _regular('hotel', RUN, device)
    require(canonical_hash(native.split) == config['split_sha256'], 'HOTEL split changed')
    identity.update(fold='hotel', source_report_commit=BASE_COMMIT, split_sha256=config['split_sha256'],
                    dev_B_seed=TEACHER_SEED, dev_B={key: float(baseline[key]) for key in METRICS})
    expected = read(TEACHER_CACHE / 'MANIFEST.json')['identity']['teachers']
    # Preserve the existing cache identity without reopening any heldout files.
    require(identity == {key: value for key, value in expected.items() if key != 'historical'},
            'teacher cache has different frozen weights or preprocessing')
    for module in (model, provider, dependence):
        module.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    return FoldAssets('hotel', native.split, native.data, native.banks['train'], model, provider,
                      dependence, expected, identity['dev_B'], None, device, geometry)


def _student(path, sha, epoch):
    import torch
    from .model import Student

    require(digest(path) == sha, f'S{epoch} checkpoint changed')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    require(saved['completed_epoch'] == epoch, 'wrong student checkpoint epoch')
    student = Student()
    student.load_state_dict(saved['student'], strict=True)
    return student.to(device='cuda', dtype=torch.float32).eval().requires_grad_(False)


def _verify_slot_replay(assets, student, replay):
    import torch

    from .hotel_edge_train import _trajectory_slot_vjp
    from .model import trajectory_terms
    from .train import _trajectory, _window_goals

    path = RUN / 'RECOVERY_PARITY.json'
    recovery_sha = digest(RUN / 'RECOVERY_CONFIG.json')
    if path.exists():
        require(read(path)['recovery_config_sha256'] == recovery_sha, 'GPU parity receipt changed')
        return
    index = next(index for index in assets.split['train_indices']
                 if len(assets.split['members'][index]['agent_ids']) == 9)
    batch = assets.batch(index, 'train', truth=True)
    with torch.no_grad():
        goals, key = _window_goals(assets, student, index, batch, input_grad=False)
    goals = goals.detach().requires_grad_(True)
    native = _trajectory(assets, batch, goals, key, seed=3201, input_grad=True)
    native_loss = sum(trajectory_terms(native, batch.future, batch.valid))
    reference, = torch.autograd.grad(native_loss, goals)
    value, gradient, drift = _trajectory_slot_vjp(assets, batch, goals, key, {'MA': 1., 'JA': 1.},
                                                replay['slot_chunk'])
    torch.testing.assert_close(native_loss.detach(), value, atol=1e-5, rtol=1e-6)
    torch.testing.assert_close(reference, gradient, atol=2e-6, rtol=1e-4)
    immutable(path, dict(recovery_config_sha256=recovery_sha, agent_count=9, world_count=20,
                         slot_chunk=replay['slot_chunk'], updates=0,
                         path_max_abs_m=float(drift),
                         loss_abs_diff=float((native_loss.detach()-value).abs()),
                         gradient_max_abs_diff=float((reference-gradient).abs().max())))
    torch.cuda.empty_cache()


def _worker(stage, activation_offload):
    yielded = False
    controller_pid = os.getppid()

    def request_yield(_signum, _frame):
        nonlocal yielded
        yielded = True

    def should_yield():
        return yielded or os.getppid() != controller_pid

    signal.signal(signal.SIGUSR1, request_yield)
    write(RUN / 'WORKER_READY.json', dict(pid=os.getpid(), stage=stage, unix=time.time()))
    import torch
    from bcep_v2.runtime import configure_cuda

    config = read(RUN / 'RUN_CONFIG.json')
    recovery = _check_run_source(config)
    torch.set_num_threads(1)
    configure_cuda()
    torch.cuda.set_per_process_memory_fraction(config['GPU']['initial_allocator_fraction'])
    try:
        assets = _assets(config, 'cuda')
        s24 = _student(config['S24_checkpoint'], config['S24_sha256'], 24)
        if should_yield():
            return YIELD_CODE
        if stage == 'train':
            from .hotel_edge_train import train

            replay = recovery.get('training_replay', {})
            if replay:
                _verify_slot_replay(assets, s24, replay)
            outcome = train(assets, s24, TEACHER_CACHE, RUN, config,
                            should_yield=should_yield, activation_offload=activation_offload,
                            dense_slot_chunk=replay.get('slot_chunk', 0))
            if outcome is None:
                return YIELD_CODE
            immutable(RUN / 'TRAINING_COMPLETE.json', outcome)
        else:
            from .hotel_edge_eval import export

            identity = read(RUN / 'EVALUATION_IDENTITY.json')
            s28 = _student(identity['S28_checkpoint'], identity['S28_sha256'], 28)
            outcome = export(assets, s24, s28, RUN, identity, should_yield=should_yield)
            if outcome is None:
                return YIELD_CODE
            immutable(RUN / 'EXPORT_COMPLETE.json', outcome)
    except torch.OutOfMemoryError as error:
        write(RUN / 'GPU_OOM.json', dict(stage=stage, pid=os.getpid(), unix=time.time(),
              activation_offload=activation_offload,
              peak_allocated_bytes=torch.cuda.max_memory_allocated(),
              peak_reserved_bytes=torch.cuda.max_memory_reserved(), message=str(error),
              recovery_cursor=_optional(RUN / 'training/PROGRESS.json')))
        traceback.print_exc()
        return OOM_CODE
    return 0


def _query(args):
    output = subprocess.check_output(['nvidia-smi', '-i', '0', *args, '--format=csv,noheader,nounits'],
                                     text=True, timeout=10)
    return list(csv.reader(output.splitlines(), skipinitialspace=True))


def _gpu():
    [(uuid, total, free)] = _query(['--query-gpu=uuid,memory.total,memory.free'])
    processes = {int(pid): int(memory) for pid, memory in _query(['--query-compute-apps=pid,used_memory'])
                 if pid.isdigit() and memory.isdigit()}
    return dict(uuid=uuid, total_mib=int(total), free_mib=int(free), compute_mib=processes)


def _live(pid, worker=False):
    if not pid:
        return False
    try:
        args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError, ProcessLookupError):
        return False
    return b'direct_k20.run' in args and (b'--worker' in args) == worker


def _main(gpu):
    controller = _optional(MAIN / 'LAUNCH.json').get('pid')
    child = _optional(MAIN / 'ACTIVE_CHILD.json')
    main_status = _optional(MAIN / 'STATUS.json')
    pid = child.get('pid')
    live = _live(controller)
    child_live = _live(pid, worker=True)
    in_gpu = child_live and pid in gpu['compute_mib']
    fold = child.get('fold')
    progress = _optional(MAIN / fold / 'training/PROGRESS.json') if fold else {}
    return dict(controller_pid=controller, worker_pid=pid if child_live else None,
                worker_stage=child.get('stage'), fold=fold, status=main_status.get('status'),
                controller_alive=live, worker_on_gpu=in_gpu, progress=progress,
                changing_worker=live and not in_gpu)


def _event(event, **details):
    row = dict(event=event, unix=time.time(), **details)
    with (RUN / 'RESOURCE_EVENTS.jsonl').open('a') as out:
        out.write(json.dumps(row, ensure_ascii=False) + '\n')
    return row


def _admission(gpu, main):
    if main['changing_worker']:
        return 'MAIN_WORKER_TRANSITION'
    if gpu['free_mib'] < START_FREE_MIB:
        return 'FREE_BELOW_6144_MIB'
    return None


def _execute(stage, receipt):
    if receipt.exists():
        return
    failures = 0
    marker = ('RECOVERY_ACTIVATION_OFFLOAD.json' if (RUN / 'RECOVERY_CONFIG.json').exists()
              else 'ACTIVATION_OFFLOAD.json')
    offload = stage == 'train' and (RUN / marker).exists()
    waits = None
    while not receipt.exists():
        with (RUN / 'worker.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                time.sleep(5)
                continue
        gpu = _gpu()
        main = _main(gpu)
        reason = _admission(gpu, main)
        if reason:
            if waits != reason:
                _event('WAITING_FOR_RESOURCES', stage=stage, reason=reason, gpu=gpu, main=main)
                status(RUN, 'WAITING_FOR_GPU', stage=stage, reason=reason, gpu=gpu, main=main)
            waits = reason
            time.sleep(15)
            continue
        if waits:
            _event('RESOURCES_AVAILABLE', stage=stage, previous_reason=waits, gpu=gpu, main=main)
            waits = None
        command = [sys.executable, '-m', 'direct_k20.hotel_edge', '--worker', stage]
        if offload:
            command.append('--activation-offload')
        started = time.time()
        with (RUN / f'{stage}.log').open('a', buffering=1) as log:
            child = subprocess.Popen(command, cwd=SOURCE, stdout=log, stderr=subprocess.STDOUT)
            write(RUN / 'ACTIVE_CHILD.json', dict(pid=child.pid, stage=stage, command=command,
                  unix=started, activation_offload=offload, gpu_uuid=gpu['uuid']))
            _event('WORKER_STARTED', stage=stage, pid=child.pid, gpu=gpu, main=main,
                   activation_offload=offload)
            status(RUN, 'AUXILIARY_WORKER_STARTED', stage=stage, pid=child.pid, main=main, gpu=gpu)
            peak_mib, minimum_free = 0, gpu['free_mib']
            yield_reason, requested = None, None
            probe_errors = 0
            while child.poll() is None:
                time.sleep(POLL_SECONDS)
                try:
                    sample = _gpu()
                    probe_errors = 0
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                    probe_errors += 1
                    if probe_errors < 3:
                        continue
                    _event('RESOURCE_PROBE_FAILED', stage=stage, pid=child.pid, error=str(error))
                    sample = gpu
                    yield_reason = 'RESOURCE_PROBE_FAILED'
                current_main = _main(sample)
                peak_mib = max(peak_mib, sample['compute_mib'].get(child.pid, 0))
                minimum_free = min(minimum_free, sample['free_mib'])
                if child.poll() is not None:
                    break
                if requested is not None:
                    if time.monotonic() - requested > 45:
                        child.terminate()
                        _event('WORKER_TERMINATED_AT_RECOVERY_BOUNDARY', stage=stage, pid=child.pid)
                        break
                    continue
                if current_main['changing_worker']:
                    yield_reason = 'MAIN_WORKER_TRANSITION'
                elif sample['free_mib'] < YIELD_FREE_MIB:
                    yield_reason = 'FREE_BELOW_2048_MIB'
                if yield_reason and _optional(RUN / 'WORKER_READY.json').get('pid') == child.pid:
                    try:
                        child.send_signal(signal.SIGUSR1)
                    except ProcessLookupError:
                        break
                    requested = time.monotonic()
                    _event('YIELD_REQUESTED', stage=stage, pid=child.pid, reason=yield_reason,
                           gpu=sample, main=current_main)
                    status(RUN, 'YIELDING_GPU', stage=stage, pid=child.pid, reason=yield_reason)
            code = child.wait()
        _event('WORKER_EXITED', stage=stage, pid=child.pid, exit_code=code, seconds=time.time()-started,
               peak_observed_process_mib=peak_mib, minimum_free_mib=minimum_free,
               yield_reason=yield_reason, activation_offload=offload)
        if code == 0:
            require(receipt.exists(), f'{stage} worker exited without completion receipt')
            return
        if code == YIELD_CODE or (requested is not None and code == -signal.SIGTERM):
            _event('RESUME_FROM_LAST_DURABLE_STATE', stage=stage, reason=yield_reason)
            time.sleep(15)
            continue
        failures += 1
        if code == OOM_CODE:
            _event('GPU_OOM_ROLLBACK', oom=read(RUN / 'GPU_OOM.json'))
            if stage == 'train' and not offload:
                offload = True
                write(RUN / marker, dict(enabled=True, unix=time.time()))
        require(failures < 3, f'{stage} failed repeatedly at the same shared GPU budget; see {stage}.log')
        status(RUN, 'RECOVERING_SAME_RUN', stage=stage, exit_code=code, retry=failures)
        time.sleep(20)


def _evaluation_identity(config):
    training = read(RUN / 'TRAINING_COMPLETE.json')
    keys = ('S24_checkpoint', 'S24_sha256', 'dev_manifest_sha256', 'S24_dev_reference',
            'dev_windows', 'dev_agents', 'split_sha256', 'upstream_sha256')
    identity = {key: config[key] for key in keys} | dict(
        S28_checkpoint=training['S28_checkpoint'], S28_sha256=training['S28_sha256'],
        fold='hotel', purpose='selection_dev', seed=2035, K=20, frames=12, ddim_steps=20,
        eta=0., config_sha256=digest(RUN / 'RUN_CONFIG.json'),
        training_receipt_sha256=digest(RUN / 'TRAINING_COMPLETE.json'),
    )
    require(digest(identity['S28_checkpoint']) == identity['S28_sha256'], 'S28 checkpoint changed')
    immutable(RUN / 'EVALUATION_IDENTITY.json', identity)
    return identity


def _summarize_resources():
    rows = [json.loads(line) for line in (RUN / 'RESOURCE_EVENTS.jsonl').read_text().splitlines()]
    starts = [row for row in rows if row['event'] == 'WORKER_STARTED']
    exits = [row for row in rows if row['event'] == 'WORKER_EXITED']
    gpu = _gpu()
    write(RUN / 'RESOURCE_SUMMARY.json', dict(
        gpu_uuid=gpu['uuid'], gpu_total_mib=gpu['total_mib'], controller_pid=os.getpid(),
        aux_worker_pids=[row['pid'] for row in starts],
        main_worker_pids=sorted({row['main']['worker_pid'] for row in starts if row['main']['worker_pid']}),
        shared_launches=sum(row['main']['worker_on_gpu'] for row in starts),
        yield_requests=sum(row['event'] == 'YIELD_REQUESTED' for row in rows),
        oom_rollbacks=sum(row['event'] == 'GPU_OOM_ROLLBACK' for row in rows),
        resource_wait_periods=sum(row['event'] == 'WAITING_FOR_RESOURCES' for row in rows),
        peak_observed_process_mib=max((row['peak_observed_process_mib'] for row in exits), default=0),
        workers=exits,
    ))
    write(RUN / 'MAIN_PROGRESS_AT_COMPLETION.json', _main(gpu))


def _record_checkpoint_scores(identity, score):
    path = RUN / 'training/checkpoints/TOP5.json'
    index = read(path)
    require([row['epoch'] for row in index['entries']] == [25, 26, 27, 28],
            'a complete retained epoch is missing')
    for row in index['entries']:
        require(digest(path.parent / row['file']) == row['sha256'], 'retained checkpoint changed')
    final = index['entries'][-1]
    require(final['sha256'] == identity['S28_sha256'], 'scored checkpoint was not the fixed S28')
    final.update(rank=1, metrics={metric: score['groups']['all'][f'S28_{metric}_m']
                                  for metric in METRICS})
    index.update(ranked_epoch_count=1, validation_purpose='selection_dev',
                 retention_decision=score['decision']['retained'],
                 parent=dict(epoch=24, sha256=identity['S24_sha256']))
    write(path, index)


def _publish():
    from .hotel_edge_eval import build_report

    DOC.mkdir(parents=True, exist_ok=True)
    write(DOC / 'TOP5.json', read(RUN / 'training/checkpoints/TOP5.json'))
    for name in ('RECOVERY_CONFIG.json', 'RECOVERY_PARITY.json'):
        if (RUN / name).exists():
            write(DOC / name, read(RUN / name))
    build_report(RUN, DOC)
    require(subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == BRANCH,
            'archive attempted outside the HOTEL auxiliary branch')
    require(not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], text=True).strip(),
            'unexpected staged changes')
    subprocess.run(['git', 'add', str(DOC.relative_to(SOURCE))], check=True)
    subprocess.run(['git', 'diff', '--cached', '--check'], check=True)
    if subprocess.call(['git', 'diff', '--cached', '--quiet']):
        subprocess.run(['git', 'commit', '-m', 'Archive HOTEL S24 to S28 interaction-only study'], check=True)
    subprocess.run(['git', 'push', '-u', 'origin', BRANCH], check=True)
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/' + BRANCH],
                                     text=True).split()[0]
    require(sha == remote, 'HOTEL S28 archive remote SHA differs')
    write(RUN / 'DELIVERY.json', dict(sha=sha, verified_remote_sha=remote, branch=BRANCH))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=('train', 'export'))
    parser.add_argument('--activation-offload', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume-sliced-ddim', action='store_true')
    args = parser.parse_args()
    if args.worker:
        with (RUN / 'worker.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return _worker(args.worker, args.activation_offload)
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = prepare(resume_sliced_ddim=args.resume_sliced_ddim)
        if args.prepare_only:
            status(RUN, 'PREPARED')
            return 0
        if (RUN / 'DELIVERY.json').exists():
            status(RUN, 'COMPLETE', fold='hotel')
            return 0
        require(all(os.environ.get(key) == value for key, value in ENVIRONMENT.items()),
                'run with the recorded CUDA environment')
        if not (RUN / 'START.json').exists():
            write(RUN / 'START.json', dict(unix=time.time(), command=[sys.executable, '-m', 'direct_k20.hotel_edge'],
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()))
        write(RUN / 'CONTROLLER.json', dict(pid=os.getpid(), gpu_uuid=_gpu()['uuid'], unix=time.time()))
        _execute('train', RUN / 'TRAINING_COMPLETE.json')
        identity = _evaluation_identity(config)
        _execute('export', RUN / 'EXPORT_COMPLETE.json')
        _check_run_source(config)
        require(_main_source_identity() == config['main_config_sha256'], 'main run identity changed')
        status(RUN, 'SCORING_SELECTION_DEV', fold='hotel')
        import torch
        from .hotel_edge_eval import score

        torch.set_num_threads(1)
        development = score(_assets(config, 'cpu'), RUN, identity)
        _record_checkpoint_scores(identity, development)
        _summarize_resources()
        write(RUN / 'WALL_TIME.json', dict(seconds=time.time() - read(RUN / 'START.json')['unix']))
        status(RUN, 'ARCHIVING', fold='hotel')
        _publish()
        status(RUN, 'COMPLETE', fold='hotel')
        return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        if '--worker' not in sys.argv:
            status(RUN, 'INTERRUPTED_RESUME_SAME_HOTEL', traceback=traceback.format_exc())
        raise

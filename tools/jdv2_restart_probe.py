#!/usr/bin/env python3
"""One reserved P0..P3 prefix; no retry, training, region replay or overwrite."""
from __future__ import annotations
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.jdv2_stage_a_bank import REPO, CONFIG, CHECKPOINT, require, write_json, write_npz, atomic_new, sha256
from tools.jdv2_export_stage_a_bank import preflight, gpu_snapshot, require_idle, capture, DESKTOP_EXECUTABLES

SOURCE_FILES = ['tools/jdv2_restart_probe.py', 'tools/jdv2_restart_observer.py', 'tools/jdv2_restart_cpu_checks.py']
ROOT = REPO / 'outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/restart_trace_ea846fc'


def save_torch(path, value):
    import torch
    stream = io.BytesIO()
    torch.save(value, stream)
    atomic_new(path, stream.getvalue())


def execution_settings(torch):
    return {'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(),
            'driver': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip(),
            'device': torch.cuda.get_device_name(),
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'matmul_precision': torch.get_float32_matmul_precision(),
            'environment': {k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'CUDA_LAUNCH_BLOCKING', 'CUBLAS_WORKSPACE_CONFIG', 'PYTHONHASHSEED', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}}


def run(args):
    output = ROOT / args.process
    output.mkdir(parents=True, exist_ok=False)  # exclusive persistent slot, even failed initialization consumes it
    counts = {'processes_started': 1, 'initialization_attempted': 0, 'initialization_completed': 0,
              'full_forwards_attempted': 0, 'full_forwards_completed': 0,
              'full_forwards_failed': 0, 'in_flight_unknown': 0, 'region_replays': 0}
    write_json(output / 'STARTED.json', {'pid': os.getpid(), 'argv': sys.argv,
               'slot': args.process, 'trace': args.process in ('P2', 'P3'), 'time': time.time(),
               'hard_budget': {'process_slots': ['P0', 'P1', 'P2', 'P3'], 'windows_per_slot': 14, 'total': 56, 'region': 4}})
    try:
        resource = gpu_snapshot()
        write_json(output / 'RESOURCE_INITIAL.json', resource)
        require_idle(resource, args.gpu_uuid, allow_desktop_graphics=True)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid
        provenance = preflight()
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        for path in SOURCE_FILES:
            committed = subprocess.check_output(['git', 'show', f'{commit}:{path}'])
            require(committed == (REPO / path).read_bytes(), 'Probe source must be committed before GPU initialization', 'BLOCKED_INPUT')
        provenance.update(probe_source_commit=commit, probe_files={p: sha256(REPO / p) for p in SOURCE_FILES},
                          desktop_allowlist=sorted(DESKTOP_EXECUTABLES))
        write_json(output / 'PROVENANCE.json', provenance)
        import torch
        from src.metrics import compute_metric_mask
        from src.utils import isolated_random_seed
        from tools.audit_jdv2_stage_a import _active_evaluator
        from tools.jdv2_stage_b_v1_failure_mechanism_audit import rng_snapshot, velocities_to_predictions
        from tools.jdv2_stage_b_identity_contract_audit import _rng_equal
        from tools.jdv2_restart_observer import BoundaryObserver, Recorder, state_fingerprint
        counts['initialization_attempted'] = 1
        write_json(output / 'INITIALIZING.json', counts)
        evaluator, epoch = _active_evaluator(str(REPO / CONFIG), str(REPO / CHECKPOINT), output / 'runtime', 'cuda:0')
        counts['initialization_completed'] = 1
        net = evaluator.net
        require(epoch == 13 and net.args.training_stage == 'joint_goal' and
                net.args.use_dependency_corrector is True and net.strict_no_z and
                net.args.amp_enabled and net.args.amp_dtype == 'bf16', 'Canonical route mismatch', 'BLOCKED_INPUT')
        head = net.jdv2_corrector.output[2]
        require(torch.count_nonzero(head.weight).item() == 0 and torch.count_nonzero(head.bias).item() == 0 and not net.training,
                'Stage-A head/eval mismatch', 'BLOCKED_INPUT')
        dataset = evaluator.data_loaders['valid'].dataset
        require(dataset.ids == provenance['physical_validation_filenames'] and len(dataset) == 139, 'Loader mismatch', 'BLOCKED_INPUT')
        reference_args = json.loads((REPO / 'docs/joint_dependency_v2/reviews/2026-10-03_aa7022e_stage_a_bank_export_pairing/RESOLVED_ARGS.json').read_text())
        clean = lambda d: {k: v for k, v in d.items() if k not in ('model_dir', 'save_dir')}
        require(clean(reference_args) == clean(vars(net.args)), 'Resolved args changed', 'BLOCKED_INPUT')
        write_json(output / 'RESOLVED_ARGS.json', vars(net.args))
        write_json(output / 'INITIALIZED.json', {**counts, 'strict_checkpoint_load': True, 'production_cache_validation': True})
        with torch.no_grad(), isolated_random_seed(2036, use_cuda=True):
            # Enumerate production loader in order. Stop after 13 without fetching window14.
            for index, (batch_data, batch_id) in enumerate(evaluator.data_loaders['valid']):
                snapshot = gpu_snapshot()
                write_json(output / f'resources/{index:02d}.json', snapshot)
                require_idle(snapshot, args.gpu_uuid, allow_self=True, allow_desktop_graphics=True)
                inputs, sequence = net.prepare_inputs(batch_data, batch_id)
                mask = compute_metric_mask(sequence)
                net.jdv2_sampler.set_sampling_context(2036, index)
                target = index == 13
                if target:
                    state_before = state_fingerprint(net)
                    rng_before = rng_snapshot(True)
                    settings = execution_settings(torch)
                    observer = BoundaryObserver(net, enabled=args.process in ('P2', 'P3'))
                    prepared = Recorder()
                    prepared.take('inputs', inputs)
                    prepared.take('sequence', sequence)
                    prepared.take('metric_mask', mask)
                    context_before = net.jdv2_sampler._sampling_context
                counts['full_forwards_attempted'] += 1
                counts['in_flight_unknown'] = 1
                write_json(output / f'budget/{index:02d}_attempt.json', dict(counts))
                import contextlib
                with observer if target else contextlib.nullcontext():
                    with evaluator._autocast_context():
                        contexts, auxiliary = net.encode(inputs, if_test=True)
                    if target:
                        rng_after_encode = rng_snapshot(True)
                    with evaluator._autocast_context():
                        velocity = net.ts_sample(contexts, auxiliary['dependency_state'])
                counts['full_forwards_completed'] += 1
                counts['in_flight_unknown'] = 0
                write_json(output / f'budget/{index:02d}_complete.json', dict(counts))
                before_capture = rng_snapshot(True)
                prediction = velocities_to_predictions(net, inputs, velocity)
                arrays, row = capture(evaluator, prediction, auxiliary, inputs, sequence, mask, batch_id, 2036, index, provenance)
                write_npz(output / f'windows/{index:02d}.npz', arrays)
                write_json(output / f'windows/{index:02d}.json', row)
                require(_rng_equal(before_capture, rng_snapshot(True)), 'Output capture consumed RNG')
                if target:
                    prepared.take('contexts', contexts)
                    prepared.take('velocity', velocity)
                    prepared.take('auxiliary', {k: v for k, v in auxiliary.items() if k not in ('goal_logit_map',)})
                    captured, captured_meta = prepared.finish()
                    save_torch(output / 'TARGET.pt', captured)
                    write_json(output / 'TARGET_METADATA.json', captured_meta)
                    if observer.enabled:
                        traced, trace_meta = observer.recorder.finish()
                        save_torch(output / 'TRACE.pt', traced)
                        write_json(output / 'TRACE_METADATA.json', trace_meta)
                    save_torch(output / 'RNG.pt', {'before13': rng_before, 'after_encode13': rng_after_encode, 'after_diffusion13': before_capture})
                    after = state_fingerprint(net)
                    write_json(output / 'STATE.json', {'before': state_before, 'after': after, 'unchanged': state_before == after,
                        'all_modules_eval': all(not m.training for m in net.modules()), 'settings': settings,
                        'sampler_context_before': context_before, 'sampler_context_after': net.jdv2_sampler._sampling_context,
                        'observation_cost': 'All arms: pre13/post13 state CPU copies/hash; RNG states before13/after encode/after diffusion. Traced arms: held references and function observation. No historical allocator restoration claimed.',
                        'untraced_explicit_generators': 'NOT_CAPTURED_NO_INTERNAL_HOOKS' if not observer.enabled else 'TRACE.pt',
                        'strict_load': True, 'cache_validation': True})
                    require(_rng_equal(before_capture, rng_snapshot(True)), 'Trace persistence consumed RNG')
                print(f'{args.process} complete window={index} forwards={counts["full_forwards_completed"]}', flush=True)
                del contexts, velocity, prediction, auxiliary, inputs, batch_data, arrays
                if target:
                    break
        require(counts['full_forwards_completed'] == 14, 'Incomplete prefix')
        write_json(output / 'COMPLETE.json', {'status': 'COMPLETE', **counts, 'source_commit': commit})
    except BaseException as exc:
        if counts['in_flight_unknown']:
            counts['full_forwards_failed'] += 1
            counts['in_flight_unknown'] = 0  # caught failure; abrupt death retains attempt marker instead
        write_json(output / 'FAILED.json', {'status': getattr(exc, 'status', 'FAILED'), 'error': repr(exc), **counts})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--process', choices=['P0', 'P1', 'P2', 'P3'], required=True)
    parser.add_argument('--gpu-uuid', required=True)
    run(parser.parse_args())

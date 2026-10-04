#!/usr/bin/env python3
"""Bounded original SocialMotionEncoder calls; no GDTS/evaluator initialization."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tools.jdv2_stage_a_bank import REPO, CHECKPOINT, CHECKPOINT_SHA, CONFIG, CONFIG_SHA, sha256, write_json, require, code_provenance
from tools.jdv2_restart_observer import Recorder, raw_bytes, compare_tensor, state_fingerprint
from tools.jdv2_restart_probe import save_torch, execution_settings

BASE = '634a467868c3d6a488837be4edc18ccff52db864'
PRIOR = REPO / 'docs/joint_dependency_v2/reviews/2026-10-04_80ee0e2_restart_trace'
OLD = REPO / 'outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/restart_trace_ea846fc'
ROOT = REPO / 'outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/social_encoder_local_634a467'
FIELDS = ('obs_traj', 'edge_index', 'edge_feat', 'edge_weight', 'scene_index')
SHAPES = ((9, 8, 2), (2, 36), (36, 14), (36,), (9,))
PREFIX = 'SocialMotionEncoder.forward/000/input/'
SOURCES = ('tools/jdv2_social_local.py', 'tools/jdv2_social_local_cpu.py', 'tools/jdv2_social_local_analyze.py')


def read(path):
    return json.loads(path.read_text())


def fingerprint(value):
    return {'dtype': str(value.dtype), 'shape': list(value.shape), 'stride': list(value.stride()),
            'storage_offset': value.storage_offset(), 'raw_sha256': hashlib.sha256(raw_bytes(value)).hexdigest()}


class ModuleObserver:
    """Only temporal GRU input/final hidden and whole message-layer boundaries."""
    def __init__(self, encoder):
        self.encoder, self.recorder, self.handles = encoder, Recorder(), []

    def pre(self, label, names):
        def hook(module, args):
            self.recorder.events.append({'boundary': label, 'event': 'input'})
            for name, value in zip(names, args):
                self.recorder.take(label + '/input/' + name, value)
            # None: leave the original arguments untouched.
        return hook

    def post(self, label, gru=False):
        def hook(module, args, output):
            self.recorder.events.append({'boundary': label, 'event': 'output'})
            self.recorder.take(label + ('/final_hidden' if gru else '/output'), output[1] if gru else output)
            # Do not retain output[0] (unused full GRU sequence). Return None.
        return hook

    def __enter__(self):
        require(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.encoder.modules()), 'Existing module hooks', 'BLOCKED_INPUT')
        try:
            modules = [('temporal_encoder', self.encoder.temporal_encoder, ('temporal_input',), True)]
            modules += [(f'message_layer_{i}', m, ('agent_feat', 'edge_index', 'edge_feat', 'edge_weight'), False)
                        for i, m in enumerate(self.encoder.message_layers)]
            for name, module, names, gru in modules:
                self.handles.append(module.register_forward_pre_hook(self.pre(name, names)))
                self.handles.append(module.register_forward_hook(self.post(name, gru)))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False


def authenticate():
    # Metadata-only sources plus CPU checkpoint extraction. No forward of any kind.
    require(not subprocess.check_output(['git', 'diff', BASE, '--', 'src', 'configs']), 'Production/config changed', 'BLOCKED_INPUT')
    code = code_provenance()
    require(sha256(REPO / CHECKPOINT) == CHECKPOINT_SHA, 'Checkpoint hash', 'BLOCKED_INPUT')
    require(sha256(REPO / CONFIG) == CONFIG_SHA, 'Config hash', 'BLOCKED_INPUT')
    archive_hashes = read(PRIOR / 'ARTIFACT_HASHES.json')['files']
    for name, digest in archive_hashes.items():
        require(sha256(PRIOR / name) == digest, 'Prior archive hash: ' + name, 'BLOCKED_INPUT')
    manifest = read(PRIOR / 'TRACE_MANIFEST.json')['files']
    authenticated = {}
    for process in ('P2', 'P3'):
        for name in ('TRACE.pt', 'TRACE_METADATA.json', 'STATE.json', 'PROVENANCE.json', 'RNG.pt', 'RESOLVED_ARGS.json'):
            rel = process + '/' + name
            require(sha256(OLD / rel) == manifest[rel]['sha256'], 'Prior trace hash: ' + rel, 'BLOCKED_INPUT')
            authenticated[rel] = manifest[rel]
    tapes = {p: torch.load(OLD / p / 'TRACE.pt', map_location='cpu', weights_only=True) for p in ('P2', 'P3')}
    metadata = {p: read(OLD / p / 'TRACE_METADATA.json')['tensors'] for p in tapes}
    contract = {}
    for name, shape in zip(FIELDS, SHAPES):
        key = PREFIX + name
        dtype = torch.int64 if name in ('edge_index', 'scene_index') else torch.float32
        for process in tapes:
            value, row = tapes[process][key], metadata[process][key]
            require(row['valid_snapshot'] and row['version_at_capture'] == row['version_at_dump'], 'Invalid saved snapshot', 'BLOCKED_INPUT')
            require(value.dtype == dtype and tuple(value.shape) == shape, 'Input dtype/shape', 'BLOCKED_INPUT')
            require(all(fingerprint(value)[f] == row[f] for f in ('dtype', 'shape', 'stride', 'storage_offset', 'raw_sha256')), 'Saved layout needs unsupported reconstruction', 'BLOCKED_INPUT')
        require(compare_tensor(tapes['P2'][key], tapes['P3'][key])['bitwise_equal'], 'P2/P3 input differs', 'BLOCKED_INPUT')
        contract[name] = {'P2': metadata['P2'][key], 'P3': metadata['P3'][key], 'saved_cpu': fingerprint(tapes['P2'][key]), 'P2_P3_bitwise_equal': True}
    # These five actual source inputs have independent storage. Refuse silently breaking aliasing.
    for process in tapes:
        require(len({metadata[process][PREFIX + f]['alias_group'] for f in FIELDS}) == 5, 'Shared source storage requires explicit reconstruction', 'BLOCKED_INPUT')
        require(len({tapes[process][PREFIX + f].untyped_storage()._cdata for f in FIELDS}) == 5, 'Unexpected saved aliases', 'BLOCKED_INPUT')
    args = read(OLD / 'P2/RESOLVED_ARGS.json')
    require(args['social_attention_layers'] == 1 and args['jdv2_active'] and args['social_feature_dim'] == 128, 'Encoder construction args', 'BLOCKED_INPUT')
    from src.models.social_encoder import SocialMotionEncoder
    from src.models.interaction_graph import EDGE_FEATURE_DIM, DEFAULT_OBSERVATION_DT
    constructor = {'hidden_dim': 128, 'output_dim': 128, 'edge_dim': EDGE_FEATURE_DIM,
                   'num_message_layers': args['social_attention_layers'], 'dropout': 0.0, 'dt': DEFAULT_OBSERVATION_DT}
    encoder = SocialMotionEncoder(**constructor)
    checkpoint = torch.load(REPO / CHECKPOINT, map_location='cpu', weights_only=False)
    require(checkpoint['epoch'] == 13 and checkpoint['training_stage'] == 'joint_goal' and checkpoint['latent_objective'] == 'strict_no_z', 'Checkpoint route metadata', 'BLOCKED_INPUT')
    prefix = 'social_encoder.'
    extracted = {k[len(prefix):]: v for k, v in checkpoint['model_state_dict'].items() if k.startswith(prefix)}
    encoder.load_state_dict(extracted, strict=True)
    encoder.eval()
    loaded = state_fingerprint(encoder)
    for process in tapes:
        state = read(OLD / process / 'STATE.json')
        reference = {k[len(prefix):]: v for k, v in state['before'].items() if k.startswith(prefix)}
        require(set(reference) == set(loaded), 'Encoder state keys differ', 'BLOCKED_INPUT')
        for key, row in loaded.items():
            require(all(row[f] == reference[key][f] for f in ('dtype', 'shape', 'sha256')), 'Encoder state bytes mismatch', 'BLOCKED_INPUT')
    # Nothing from the unused trace tensors is retained into the GPU batch.
    inputs = {name: tapes['P2'][PREFIX + name] for name in FIELDS}
    info = {'base_commit': BASE, 'production_src_tree': subprocess.check_output(['git', 'rev-parse', 'HEAD:src'], text=True).strip(),
        'code': code, 'checkpoint': CHECKPOINT, 'checkpoint_sha256': CHECKPOINT_SHA,
        'config_sha256': CONFIG_SHA, 'source_files_authenticated': authenticated, 'inputs': contract,
        'selected_input_source': 'P2 only; P3 authentication only', 'layout_reconstruction': 'none needed: native contiguous stride, offset0; five distinct storage groups',
        'constructor': constructor, 'constructor_evidence': 'model.py JDV2 construction lines423-425; social_encoder.py defaults dropout0 and DEFAULT_OBSERVATION_DT=0.4',
        'strict_encoder_load': True, 'encoder_state_matches_P2_and_P3': True, 'encoder_state': loaded,
        'settings_reference': read(OLD / 'P2/STATE.json')['settings'],
        'unrecoverable': ['CUDA addresses', 'allocator', 'stream scheduling', 'GRU workspace history', 'full prior run', 'exact historical encoder-entry RNG snapshot']}
    return inputs, encoder, info


def run(process, uuid):
    from tools.jdv2_export_stage_a_bank import gpu_snapshot, require_idle, DESKTOP_EXECUTABLES
    from tools.jdv2_stage_b_v1_failure_mechanism_audit import rng_snapshot, rng_restore
    from tools.jdv2_stage_b_identity_contract_audit import _rng_equal
    out = ROOT / process
    out.mkdir(parents=True, exist_ok=False)
    counts = {'processes_started': 1, 'initialization_attempted': 0, 'initialization_completed': 0,
              'encoder_attempted': 0, 'encoder_completed': 0, 'encoder_failed': 0, 'unknown_inflight': 0,
              'full_A_forwards': 0, 'net_encode_calls': 0, 'sampler_calls': 0, 'diffusion_calls': 0, 'independent_submodule_replays': 0}
    write_json(out / 'STARTED.json', {'pid': os.getpid(), 'argv': sys.argv, 'time': time.time(), 'budget': 'four exclusive process slots U0/U1/T0/T1, at most four encoder attempts per slot; no retry'})
    try:
        resource = gpu_snapshot()
        write_json(out / 'RESOURCE_INITIAL.json', resource)
        require_idle(resource, uuid, allow_desktop_graphics=True)
        os.environ['CUDA_VISIBLE_DEVICES'] = uuid
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        for name in SOURCES:
            require(subprocess.check_output(['git', 'show', commit + ':' + name]) == (REPO / name).read_bytes(), 'Uncommitted probe source', 'BLOCKED_INPUT')
        counts['initialization_attempted'] = 1
        write_json(out / 'INITIALIZING.json', counts)
        inputs, encoder, info = authenticate()
        write_json(out / 'INPUT_MANIFEST.json', info)
        ref = info['settings_reference']
        # Restore the original isolated_random_seed context, NOT a deterministic intervention.
        defaults = {'cudnn_deterministic': torch.backends.cudnn.deterministic, 'cudnn_benchmark': torch.backends.cudnn.benchmark}
        torch.backends.cudnn.deterministic = ref['cudnn_deterministic']
        torch.backends.cudnn.benchmark = ref['cudnn_benchmark']
        encoder = encoder.to('cuda:0').eval()
        inputs = {k: v.to('cuda:0') for k, v in inputs.items()}
        settings = execution_settings(torch)
        differences = {k: [ref[k], settings[k]] for k in ref if ref[k] != settings[k]}
        require(not differences, 'Critical execution settings mismatch: ' + repr(differences), 'BLOCKED_INPUT')
        before_inputs = {k: fingerprint(v) for k, v in inputs.items()}
        require(all(all(before_inputs[k][f] == info['inputs'][k]['P2'][f] for f in before_inputs[k]) for k in FIELDS), 'GPU input layout/bytes mismatch', 'BLOCKED_INPUT')
        before_model = state_fingerprint(encoder)
        require(all(not m.training for m in encoder.modules()), 'Non-eval module', 'BLOCKED_INPUT')
        require(sys.getprofile() is None and sys.gettrace() is None, 'Active profiler/tracer', 'BLOCKED_INPUT')
        saved_rng = torch.load(OLD / 'P2/RNG.pt', map_location='cpu', weights_only=False)['before13']
        rng_restore(saved_rng)
        before_rng = rng_snapshot(True)
        require(_rng_equal(saved_rng, before_rng), 'RNG restore failed', 'BLOCKED_INPUT')
        counts['initialization_completed'] = 1
        write_json(out / 'INITIALIZED.json', {**counts, 'source_commit': commit, 'settings': settings,
                   'original_process_defaults': defaults, 'restored_baseline_flags': ['cudnn_deterministic', 'cudnn_benchmark'],
                   'precision_context': 'original FP32 island, torch.autocast(cuda, enabled=False); no precision intervention',
                   'RNG_source': 'P2/RNG.pt before13; not claimed encoder-entry state', 'desktop_allowlist': sorted(DESKTOP_EXECUTABLES)})
        resource = gpu_snapshot()
        write_json(out / 'RESOURCE_BEFORE_BATCH.json', resource)
        require_idle(resource, uuid, allow_self=True, allow_desktop_graphics=True)
        outputs = []
        observer = ModuleObserver(encoder)  # no hooks until explicit __enter__ on fourth traced call
        with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False):
            for index in range(4):
                counts['encoder_attempted'] += 1
                counts['unknown_inflight'] = 1
                write_json(out / f'budget/{index+1}_attempt.json', dict(counts))
                traced = process.startswith('T') and index == 3
                with observer if traced else contextlib.nullcontext():
                    h = encoder(**inputs)
                outputs.append(h)
                counts['encoder_completed'] += 1
                counts['unknown_inflight'] = 0
                write_json(out / f'budget/{index+1}_complete.json', dict(counts))
        after_rng = rng_snapshot(True)
        rng_equal = _rng_equal(before_rng, after_rng)
        save_torch(out / 'RNG.pt', {'before': before_rng, 'after': after_rng})
        after_inputs = {k: fingerprint(v) for k, v in inputs.items()}
        after_model = state_fingerprint(encoder)
        hooks_removed = all(not m._forward_hooks and not m._forward_pre_hooks for m in encoder.modules())
        common = Recorder()
        for index, h in enumerate(outputs, 1):
            common.take(f'call{index}/h', h)
        cpu, metadata = common.finish()
        metadata['observer_cost'] = 'All arms retain four returned h tensors and defer transfer until batch end; input/model/RNG checks outside batch; per-call CPU ledger writes'
        save_torch(out / 'OUTPUTS.pt', cpu)
        write_json(out / 'OUTPUT_CONTRACTS.json', metadata)
        trace_meta = None
        if process.startswith('T'):
            observer.recorder.take('encoder/final_h', outputs[-1])
            trace, trace_meta = observer.recorder.finish()
            trace_meta['observer_cost'] = 'Only GRU/message module pre/post hooks on call4; no profiler, torch wrappers, dispatch, clone/cpu/item/synchronize/hash in hooks. References change lifetimes/allocator; module hooks cost CPU scheduling.'
            save_torch(out / 'TRACE.pt', trace)
            write_json(out / 'TRACE_CONTRACTS.json', trace_meta)
        validation = {'input_unchanged': before_inputs == after_inputs, 'model_unchanged': before_model == after_model,
            'global_RNG_unchanged': rng_equal, 'hooks_removed': hooks_removed, 'inputs': before_inputs,
            'model_before': before_model, 'model_after': after_model,
            'snapshots_valid': not metadata['invalid_snapshots'] and (trace_meta is None or not trace_meta['invalid_snapshots']),
            'precision': 'FP32', 'settings': settings}
        write_json(out / 'VALIDATION.json', validation)
        require(rng_equal, 'Encoder consumed RNG: stop, do not reset and retry', 'RNG_CHANGED')
        require(all(validation[k] for k in ('input_unchanged','model_unchanged','hooks_removed','snapshots_valid')), 'Observation/input/state contract failed', 'INVALID_SNAPSHOT_OR_STATE')
        resource = gpu_snapshot()
        write_json(out / 'RESOURCE_AFTER_BATCH.json', resource)
        require_idle(resource, uuid, allow_self=True, allow_desktop_graphics=True)
        write_json(out / 'COMPLETE.json', {'status': 'COMPLETE', **counts, 'source_commit': commit})
        print(json.dumps({'process': process, **counts, 'validation': 'PASS_CHECKED_PROPERTIES_ONLY'}), flush=True)
    except BaseException as exc:
        if counts['unknown_inflight']:
            counts['encoder_failed'] += 1
            counts['unknown_inflight'] = 0
        write_json(out / 'FAILED.json', {'status': getattr(exc, 'status', 'FAILED'), 'error': repr(exc), **counts})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--process', choices=['U0','U1','T0','T1'], required=True)
    parser.add_argument('--gpu-uuid', required=True)
    cli = parser.parse_args()
    run(cli.process, cli.gpu_uuid)

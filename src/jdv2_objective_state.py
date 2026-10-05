"""Independent CPU sampling state with explicit loss compute backends.

Resume covers fixed-input optimizer boundaries, not data/worker/global RNG.
"""
import hashlib
import json
import math
import os
import tempfile
import subprocess
from functools import lru_cache
from pathlib import Path

import torch

MC = 'expected_conditional_mc'
FIELD = 'jdv2_goal_objective_state'
NAMESPACE = 'jdv2.expected-conditional.cpu.v1'
SCOPE = 'fixed-input objective RNG + optimizer/scheduler/progress; epoch boundary'
CPU_BACKEND = 'cpu_v1'
CUDA_BACKEND = 'cuda_fp32_bf16_v1'


def compute_backend(args):
    return getattr(args, 'jdv2_mc_backend', CPU_BACKEND)


def validate_compute_context(backend):
    if backend not in {CPU_BACKEND, CUDA_BACKEND}:
        raise ValueError('Unknown MC compute backend')
    if backend == CPU_BACKEND and torch.is_autocast_enabled('cpu'):
        raise ValueError('CPU MC AMP is not supported')
    if (backend == CUDA_BACKEND and torch.is_autocast_enabled('cuda') and
            torch.get_autocast_dtype('cuda') != torch.bfloat16):
        raise ValueError('CUDA MC only permits BF16 autocast, not FP16')


@lru_cache(maxsize=1)
def implementation_source_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def objective_mode(args):
    return getattr(args, 'jdv2_goal_objective', 'mean_energy')


def validate_mode(args, device=None):
    mode = objective_mode(args)
    if mode not in {'mean_energy', MC}:
        raise ValueError('Unknown JDV2 goal objective')
    if mode == 'mean_energy':
        return
    if (getattr(args, 'goal_model_type', None) != 'joint_dependency_v2' or
            not getattr(args, 'jdv2_active', False) or
            getattr(args, 'jdv2_latent_objective', None) != 'strict_no_z' or
            getattr(args, 'training_stage', None) != 'joint_goal' or
            getattr(args, 'use_scene_latent', False) or
            str(getattr(args, 'dataset', '')).lower() == 'sdd' or
            getattr(args, 'trajectory_coupling', 'none') != 'none'):
        raise ValueError('MC requires active strict_no_z Stage-A (not SDD/V4)')
    if getattr(args, 'jdv2_neighbor_draws', 4) != 4:
        raise ValueError('MC neighbor_draws must be exactly 4')
    seed = getattr(args, 'jdv2_neighbor_seed', None)
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**63):
        raise ValueError('neighbor seed must be integer in [0,2**63)')
    if getattr(args, 'jdv2_edge_chunk_size', 1) <= 0:
        raise ValueError('MC edge chunk must be positive')
    backend = compute_backend(args)
    if backend not in {CPU_BACKEND, CUDA_BACKEND}:
        raise ValueError('Unknown MC compute backend')
    kind = torch.device(device or getattr(args, 'device', 'cpu')).type
    amp, dtype = getattr(args, 'amp_enabled', False), getattr(args, 'amp_dtype', 'fp32')
    if backend == CPU_BACKEND:
        if kind != 'cpu' or amp or dtype != 'fp32':
            raise ValueError('cpu_v1 requires CPU FP32/FP64 without AMP')
    elif kind != 'cuda' or (amp, dtype) not in {(False, 'fp32'), (True, 'bf16')}:
        raise ValueError('CUDA MC requires explicit CUDA device, FP32 or BF16 autocast')


def semantic_fingerprint(args):
    config = dict(version=NAMESPACE, objective=MC, S=4,
                  sigma=args.goal_soft_sigma,
                  progress='successful-updates/num_epochs*train_batches; accumulation=1',
                  optimizer=getattr(args, 'optimizer', None),
                  scheduler=getattr(args, 'scheduler', None),
                  clip=getattr(args, 'clip', None),
                  learning_rate=getattr(args, 'learning_rate', None),
                  structured_lr_scale=getattr(args, 'structured_lr_scale', None))
    for key in ('jdv2_cache_manifest_hash', 'jdv2_source_checkpoint_hash',
                'clean_split_manifest_hash', 'dataset', 'test_set'):
        config[key] = getattr(args, key, None)
    if compute_backend(args) == CUDA_BACKEND:
        config.update(loss_compute_backend=CUDA_BACKEND, effective_dtype='float32',
                      amp_policy='bf16' if getattr(args, 'amp_enabled', False) else 'off')
    return hashlib.sha256(json.dumps(config, sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def validate_request(u, q, scenes, edges, mask, backend=CPU_BACKEND):
    """Reject invalid labels/geometry BEFORE consuming the objective generator."""
    validate_compute_context(backend)
    expected_kind = 'cpu' if backend == CPU_BACKEND else 'cuda'
    if not isinstance(u, torch.Tensor):
        raise TypeError('MC unary tensor required')
    for value in (u, q, scenes, edges, mask):
        if not isinstance(value, torch.Tensor):
            raise TypeError('MC request tensors required')
        if (value.device.type != expected_kind or value.device != u.device or
                value.layout != torch.strided):
            raise ValueError('MC request requires dense tensors on one backend device')
    if backend == CUDA_BACKEND and u.dtype != torch.float32:
        raise TypeError('CUDA MC effective inputs must be FP32')
    if u.dtype not in (torch.float32, torch.float64) or q.dtype != u.dtype:
        raise TypeError('MC u/q must share FP32/FP64 dtype')
    if u.ndim != 2 or min(u.shape) <= 0 or q.shape != u.shape:
        raise ValueError('MC u/q shape must be nonempty [N,K]')
    n, k = u.shape
    if q.requires_grad:
        raise ValueError('endpoint q must be fixed, not trainable')
    if mask.dtype != torch.bool or mask.shape != u.shape or not mask.any(-1).all():
        raise ValueError('invalid MC candidate support')
    if (not torch.isfinite(u).all() or not torch.isfinite(q).all() or
            (q < 0).any() or (q[~mask] != 0).any()):
        raise ValueError('MC u/q must be finite; q nonnegative and zero off support')
    tol = 1e-12 if q.dtype == torch.float64 else 1e-6
    if not torch.allclose(q.sum(-1), torch.ones(n, dtype=q.dtype, device=q.device), atol=tol, rtol=0):
        raise ValueError('MC endpoint q must be normalized')
    if scenes.dtype != torch.int64 or scenes.shape != (n,):
        raise ValueError('MC scene_index must be int64 [N]')
    if edges.dtype != torch.int64 or edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError('MC edge_index must be int64 [2,E]')
    if ((edges < 0) | (edges >= n)).any():
        raise ValueError('MC edge out of range')
    i, j = edges
    if ((i >= j).any() or torch.unique(edges, dim=1).shape[1] != edges.shape[1]
            or (scenes[i] != scenes[j]).any()):
        raise ValueError('MC edges must be unique canonical within-scene pairs')


class ObjectiveRNG:
    def __init__(self, args):
        validate_mode(args)
        if objective_mode(args) != MC:
            raise ValueError('Do not create MC RNG for legacy mode')
        seed = getattr(args, 'jdv2_neighbor_seed', None)
        if seed is None:
            seed = int.from_bytes(hashlib.sha256(
                f'{NAMESPACE}:{args.seed}'.encode()).digest()[:8], 'big') % (2**63)
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('neighbor seed must be integer in [0,2**63)')
        self.initialization_seed = seed
        self.backend = compute_backend(args)
        self.amp_policy = 'bf16' if getattr(args, 'amp_enabled', False) else 'off'
        self.generator = torch.Generator(device='cpu').manual_seed(seed)
        self.fingerprint = semantic_fingerprint(args)
        self.source_commit = implementation_source_commit()
        self.draw_calls = self.sampled_agent_draws = 0
        self.successful_optimizer_updates = 0
        self.pending_backward = False

    def draw(self, u, q, scenes, edges, mask):
        if self.pending_backward:
            raise RuntimeError('Previous MC request unfinished; stop, do not retry')
        validate_request(u, q, scenes, edges, mask, self.backend)
        # Validate on the compute device first. ONLY fixed endpoint labels cross
        # to CPU; unary/cost/upstream tensors never leave their gradient device.
        fixed_q = q.detach().to(device='cpu')
        ids = torch.multinomial(fixed_q, 4, replacement=True,
                                generator=self.generator).T.contiguous()
        self.draw_calls += 1
        self.sampled_agent_draws += 4 * q.shape[0]
        # A failure after this point retains consumed state and forbids saves/retry.
        self.pending_backward = True
        return ids.to(device=u.device)

    def finish_update(self, successful):
        if not self.pending_backward:
            raise RuntimeError('No pending MC request')
        if successful:
            self.successful_optimizer_updates += 1
        self.pending_backward = False

    def snapshot(self, total_steps, role):
        if self.pending_backward:
            raise RuntimeError('Cannot save pending backward/request')
        if role not in {'resume_last', 'weights_best', 'weights_numbered'}:
            raise ValueError('Unknown MC checkpoint role')
        if type(total_steps) is not int or total_steps <= 0:
            raise ValueError('Positive total_steps required')
        if not 0 <= self.successful_optimizer_updates <= total_steps:
            raise ValueError('MC updates exceed fixed stage budget')
        state = dict(format_version=1, objective=MC, neighbor_draws=4,
                    generator_device='cpu', generator_algorithm='torch.cpu.MT19937',
                    sampling_semantics_version='multinomial-replacement-Nx4-transpose-v1',
                    torch_version=str(torch.__version__), namespace=NAMESPACE,
                    initialization_seed=self.initialization_seed,
                    generator_state=self.generator.get_state().clone(),
                    draw_calls=self.draw_calls, sampled_agent_draws=self.sampled_agent_draws,
                    successful_optimizer_updates=self.successful_optimizer_updates,
                    stage_total_optimizer_steps=total_steps,
                    stage_progress=self.successful_optimizer_updates / total_steps,
                    checkpoint_role=role, pending_backward=False,
                    objective_semantic_fingerprint=self.fingerprint,
                    source_commit=self.source_commit, resume_scope=SCOPE)
        if self.backend == CUDA_BACKEND:
            state.update(format_version=2, loss_compute_backend=CUDA_BACKEND,
                         loss_effective_dtype='float32', amp_policy=self.amp_policy,
                         resume_scope=SCOPE + '; CUDA exact resume NOT certified')
        return state

    def validated_state(self, state, total_steps):
        if not isinstance(state, dict):
            raise ValueError('Missing MC objective state; same-stage warm start forbidden')
        expected = self.snapshot(total_steps, 'resume_last')
        for key in ('loss_compute_backend', 'loss_effective_dtype', 'amp_policy'):
            if key in expected and state.get(key) != expected[key]:
                raise ValueError(f'MC checkpoint mismatch: {key}')
        for key in ('format_version', 'objective', 'neighbor_draws', 'generator_device',
                    'generator_algorithm', 'sampling_semantics_version', 'torch_version',
                    'namespace', 'stage_total_optimizer_steps', 'checkpoint_role',
                    'pending_backward', 'objective_semantic_fingerprint', 'resume_scope'):
            if type(state.get(key)) is not type(expected[key]) or state[key] != expected[key]:
                raise ValueError(f'MC checkpoint mismatch: {key}')
        for key in ('draw_calls', 'sampled_agent_draws', 'successful_optimizer_updates',
                    'initialization_seed'):
            if type(state.get(key)) is not int or state[key] < 0:
                raise ValueError(f'Invalid MC counter/seed: {key}')
        d, rows, steps = (state[k] for k in ('draw_calls', 'sampled_agent_draws',
                                             'successful_optimizer_updates'))
        if (steps > d or steps > total_steps or rows % 4 or rows < 4*d or
                (d == 0 and rows != 0) or state['initialization_seed'] >= 2**63 or
                type(state.get('stage_progress')) is not float or
                not math.isfinite(state['stage_progress']) or
                state['stage_progress'] != steps / total_steps):
            raise ValueError('Inconsistent MC counters/progress')
        value = state.get('generator_state')
        if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or
                value.dtype != torch.uint8 or value.ndim != 1):
            raise ValueError('Invalid MC generator_state')
        probe = torch.Generator(device='cpu')
        try:
            probe.set_state(value.clone())
        except RuntimeError as error:
            raise ValueError('Corrupt MC generator_state') from error
        return probe

    def restore(self, state, total_steps):
        probe = self.validated_state(state, total_steps)
        self.generator = probe
        for key in ('initialization_seed', 'draw_calls', 'sampled_agent_draws',
                    'successful_optimizer_updates'):
            setattr(self, key, state[key])


class CheckpointDurabilityError(OSError):
    """Rename committed the new file; directory durability is unknown."""
    committed = True


def atomic_save(payload, path):
    """MC resumable payload only. Pre-rename failures preserve old bytes."""
    directory = os.path.dirname(os.path.abspath(path))
    fd, temporary = tempfile.mkstemp(prefix='.mc-checkpoint-', suffix='.tmp', dir=directory)
    committed = False
    try:
        with os.fdopen(fd, 'wb') as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        committed = True
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as error:
        if committed:
            raise CheckpointDurabilityError(
                'MC checkpoint committed; directory durability unknown') from error
        raise
    finally:
        if not committed and os.path.exists(temporary):
            os.unlink(temporary)

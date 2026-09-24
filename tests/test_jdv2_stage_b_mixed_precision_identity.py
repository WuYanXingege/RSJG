"""Numerical identity contracts for Stage-B dependency routing."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.models.joint_dependency_v2 import DependencyCorrector
from src.models.model import GDTS
from tools.jdv2_stage_b_identity_contract_audit import (
    _rng_equal,
    legacy_replay_branches,
)
from tools.jdv2_stage_b_v1_failure_mechanism_audit import (
    make_noise_tape,
    replay_branches,
    rng_snapshot,
    trunk_from_tape,
)


ROOT = Path(__file__).resolve().parents[1]
STAGE_A = ROOT / ("outputs/joint_dependency_v2/eth/joint_dependency_v2/"
                  "runs/jdv2_stage_a_no_z_full_seed2035/saved_models/"
                  "best_model.pt")
STAGE_B = ROOT / ("outputs/joint_dependency_v2/eth/joint_dependency_v2/"
                  "runs/jdv2_stage_b_v1_eth_seed2035/saved_models/"
                  "best_model.pt")


class _Scene:
    @staticmethod
    def make_world_coord_torch(value):
        return value.float()


class _Schedule:
    def __init__(self, device):
        self.num_steps = 4
        self.alphas = torch.tensor(
            [1.0, .99, .98, .97, .96], device=device)
        self.alpha_bars = torch.tensor(
            [1.0, .99, .97, .94, .90], device=device)
        self.betas = 1 - self.alphas


class _Diffnet:
    def __init__(self, dtype):
        self.dtype = dtype
        self.calls = 0

    def __call__(self, x, beta, context):
        del beta, context
        self.calls += 1
        return (x * .125).to(self.dtype)


class _Corrector:
    def __init__(self):
        self.calls = 0

    def __call__(self, velocity, last, edge, relation, timestep,
                 edge_weight=None):
        del last, relation, timestep, edge_weight
        self.calls += 1
        result = torch.zeros_like(velocity, dtype=torch.float32)
        if edge.shape[1]:
            active = torch.unique(edge)
            result[active] = .25
        return result


class _Harness:
    ts_sample = GDTS.ts_sample

    def __init__(self, device, denoiser_dtype=torch.float32):
        self.args = SimpleNamespace(
            pred_length=12, trunk_stage_step=2, ddim_step=2,
            branch_stage_step=1, num_samples=2, dataset="eth5",
            use_dependency_corrector=True, down_factor=1,
            trajectory_dt=.4)
        self.var_sched = _Schedule(device)
        self.diffnet = _Diffnet(denoiser_dtype)
        self.jdv2_corrector = _Corrector()


def _contexts(device, n=3):
    return [torch.zeros(n, 1, 4, device=device) for _ in range(3)]


def _state(device, mixed):
    edge = (torch.tensor([[0], [1]], device=device) if mixed else
            torch.empty(2, 0, dtype=torch.long, device=device))
    return {
        "edge_index": edge,
        "edge_weight": torch.ones(edge.shape[1], device=device),
        "relation_embedding": torch.zeros(
            edge.shape[1], 2, 16, device=device),
        "last_position_map": torch.zeros(3, 2, device=device),
        "last_position_world": torch.zeros(3, 2, device=device),
        "scene": _Scene(),
    }


def _paired_production(harness, contexts, state, seed=9):
    torch.manual_seed(seed)
    fixed = harness.ts_sample(contexts, state)
    post_fixed = rng_snapshot(use_cuda=contexts[0].is_cuda)
    calls = harness.jdv2_corrector.calls
    harness.args.use_dependency_corrector = False
    torch.manual_seed(seed)
    stage_a = harness.ts_sample(contexts, state)
    post_stage_a = rng_snapshot(use_cuda=contexts[0].is_cuda)
    harness.args.use_dependency_corrector = True
    return fixed, stage_a, calls, post_fixed, post_stage_a


def test_cpu_fp32_e0_is_exact_and_bypasses_corrector():
    harness = _Harness(torch.device("cpu"))
    fixed, stage_a, calls, post_fixed, post_stage_a = _paired_production(
        harness, _contexts("cpu"), _state("cpu", mixed=False))
    assert torch.equal(fixed, stage_a)
    assert calls == 0
    assert _rng_equal(post_fixed, post_stage_a)


def test_mixed_degree_zero_is_exact_and_active_agent_is_corrected():
    harness = _Harness(torch.device("cpu"))
    fixed, stage_a, calls, post_fixed, post_stage_a = _paired_production(
        harness, _contexts("cpu"), _state("cpu", mixed=True))
    assert calls == 2  # one active branch step for each of two worlds
    assert torch.equal(fixed[:, 2], stage_a[:, 2])
    assert not torch.equal(fixed[:, :2], stage_a[:, :2])
    assert _rng_equal(post_fixed, post_stage_a)


def test_active_agent_matches_prefixed_v1_arithmetic():
    device = torch.device("cpu")
    harness = _Harness(device)
    contexts, state = _contexts(device), _state(device, mixed=True)
    torch.manual_seed(33)
    tape = make_noise_tape(harness, contexts[-1])
    middle = trunk_from_tape(harness, contexts, tape)
    fixed, _ = replay_branches(
        harness, contexts, state, tape, "full_v1",
        torch.zeros(3, dtype=torch.long), None, 33, middle_result=middle)
    legacy = legacy_replay_branches(
        harness, contexts, state, tape, middle_result=middle)
    assert torch.equal(fixed[:, :2], legacy[:, :2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_bf16_e0_and_mixed_identity():
    device = torch.device("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        e0 = _Harness(device, torch.bfloat16)
        fixed, stage_a, calls, fixed_rng, stage_a_rng = _paired_production(
            e0, _contexts(device), _state(device, mixed=False))
        assert torch.equal(fixed, stage_a)
        assert calls == 0
        assert _rng_equal(fixed_rng, stage_a_rng)

        mixed = _Harness(device, torch.bfloat16)
        fixed, stage_a, calls, fixed_rng, stage_a_rng = _paired_production(
            mixed, _contexts(device), _state(device, mixed=True))
        assert torch.equal(fixed[:, 2], stage_a[:, 2])
        assert not torch.equal(fixed[:, :2], stage_a[:, :2])
        assert calls == 2
        assert _rng_equal(fixed_rng, stage_a_rng)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_dependency_corrector_architecture_and_source_unchanged():
    assert sum(parameter.numel()
               for parameter in DependencyCorrector().parameters()) == 30851
    assert _sha256(ROOT / "src/models/joint_dependency_v2/"
                   "dependency_corrector.py") == \
        "0ff28eb7107e49a0ad6f2bf339a65d8847f83c4a7cf315faa5a80cac4dc00522"


@pytest.mark.skipif(not STAGE_A.exists() or not STAGE_B.exists(),
                    reason="protected checkpoints unavailable")
def test_protected_checkpoint_hashes_unchanged():
    assert _sha256(STAGE_A) == \
        "699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb"
    assert _sha256(STAGE_B) == \
        "e24a1cbfc5760db57bcb5cacd64c66a4afd1aed0f11bb1efefa8549933d21c1a"

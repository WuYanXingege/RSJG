"""Contracts for the canonical Stage-A exact-zero inference bypass."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.models.joint_dependency_v2 import DependencyCorrector
from src.models.model import GDTS, _jdv2_output_layer_is_exact_zero
from tools.jdv2_stage_b_identity_contract_audit import _rng_equal
from tools.jdv2_stage_b_v1_failure_mechanism_audit import rng_snapshot


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

    def __call__(self, value, beta, context):
        del beta, context
        return (value * .125).to(self.dtype)


class _CountingCorrector(DependencyCorrector):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, *args, **kwargs):
        self.calls += 1
        return super().forward(*args, **kwargs)


class _Harness:
    ts_sample = GDTS.ts_sample

    def __init__(self, device, dtype=torch.float32, stage="joint_goal",
                 training=False):
        self.args = SimpleNamespace(
            pred_length=12, trunk_stage_step=2, ddim_step=2,
            branch_stage_step=1, num_samples=2, dataset="eth5",
            use_dependency_corrector=True, down_factor=1,
            trajectory_dt=.4, jdv2_residual_projection="none",
            training_stage=stage)
        self.training = training
        self.var_sched = _Schedule(device)
        self.diffnet = _Diffnet(dtype)
        self.jdv2_corrector = _CountingCorrector().to(device)


def _contexts(device, n):
    return [torch.zeros(n, 1, 4, device=device) for _ in range(3)]


def _state(device, n, mixed):
    if mixed:
        edge = torch.tensor([[0], [1]], dtype=torch.long, device=device)
    else:
        edge = torch.stack((
            torch.arange(n - 1, device=device),
            torch.arange(1, n, device=device)))
    return {
        "edge_index": edge,
        "edge_weight": torch.ones(edge.shape[1], device=device),
        "relation_embedding": torch.zeros(
            edge.shape[1], 2, 16, device=device),
        "last_position_map": torch.zeros(n, 2, device=device),
        "last_position_world": torch.zeros(n, 2, device=device),
        "scene": _Scene(),
    }


def _paired(harness, contexts, state, seed=71):
    torch.manual_seed(seed)
    if contexts[0].is_cuda:
        torch.cuda.manual_seed_all(seed)
    canonical = harness.ts_sample(contexts, state)
    canonical_rng = rng_snapshot(contexts[0].is_cuda)
    calls = harness.jdv2_corrector.calls
    harness.args.use_dependency_corrector = False
    torch.manual_seed(seed)
    if contexts[0].is_cuda:
        torch.cuda.manual_seed_all(seed)
    literal = harness.ts_sample(contexts, state)
    literal_rng = rng_snapshot(contexts[0].is_cuda)
    harness.args.use_dependency_corrector = True
    return canonical, literal, calls, canonical_rng, literal_rng


@pytest.mark.parametrize("mixed", [True, False])
def test_stage_a_fp32_exact_zero_bypasses_mixed_and_all_active(mixed):
    harness = _Harness(torch.device("cpu"))
    canonical, literal, calls, left_rng, right_rng = _paired(
        harness, _contexts("cpu", 3), _state("cpu", 3, mixed))

    assert _jdv2_output_layer_is_exact_zero(harness.jdv2_corrector)
    assert calls == 0
    assert canonical.dtype == literal.dtype
    assert torch.equal(canonical, literal)
    assert _rng_equal(left_rng, right_rng)


def test_stage_a_nonzero_corrector_preserves_existing_active_semantics():
    harness = _Harness(torch.device("cpu"))
    with torch.no_grad():
        harness.jdv2_corrector.output[-1].bias.fill_(.25)
    canonical, literal, calls, left_rng, right_rng = _paired(
        harness, _contexts("cpu", 3), _state("cpu", 3, mixed=True))

    assert not _jdv2_output_layer_is_exact_zero(harness.jdv2_corrector)
    assert calls == 2
    assert not torch.equal(canonical[:, :2], literal[:, :2])
    assert torch.equal(canonical[:, 2], literal[:, 2])
    assert _rng_equal(left_rng, right_rng)


def test_stage_b_training_route_does_not_use_stage_a_zero_bypass():
    harness = _Harness(
        torch.device("cpu"), stage="joint_trajectory", training=True)
    canonical, _literal, calls, left_rng, right_rng = _paired(
        harness, _contexts("cpu", 3), _state("cpu", 3, mixed=True))

    assert canonical.shape == (2, 3, 12, 2)
    assert calls == 2
    assert _rng_equal(left_rng, right_rng)


def test_exact_zero_contract_survives_checkpoint_reload(tmp_path):
    source = _Harness(torch.device("cpu"))
    checkpoint = tmp_path / "corrector.pt"
    torch.save(source.jdv2_corrector.state_dict(), checkpoint)
    reloaded = _Harness(torch.device("cpu"))
    reloaded.jdv2_corrector.load_state_dict(
        torch.load(checkpoint, map_location="cpu"))

    canonical, literal, calls, left_rng, right_rng = _paired(
        reloaded, _contexts("cpu", 3), _state("cpu", 3, mixed=False))
    assert _jdv2_output_layer_is_exact_zero(reloaded.jdv2_corrector)
    assert calls == 0
    assert torch.equal(canonical, literal)
    assert _rng_equal(left_rng, right_rng)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("mixed", [True, False])
def test_stage_a_cuda_bf16_exact_zero_is_tensor_exact(mixed):
    device = torch.device("cuda")
    harness = _Harness(device, torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        canonical, literal, calls, left_rng, right_rng = _paired(
            harness, _contexts(device, 3), _state(device, 3, mixed))

    assert calls == 0
    assert canonical.dtype == literal.dtype
    assert torch.equal(canonical, literal)
    assert _rng_equal(left_rng, right_rng)

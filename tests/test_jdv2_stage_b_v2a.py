"""Contracts for Stage-B V2-A component-relative residual projection."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from src import parser as parser_module
from src.models.joint_dependency_v2 import (
    DependencyCorrector,
    build_component_metadata,
    component_zero_mean_projection,
)
from src.models.model import GDTS
from tools.jdv2_stage_b_v1_failure_mechanism_audit import (
    decompose_component_residual,
    rng_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
V1_CONFIG = ROOT / "configs/joint_dependency_v2/jdv2_stage_b_v1_eth.yaml"
V2A_CONFIG = ROOT / "configs/joint_dependency_v2/jdv2_stage_b_v2a_eth.yaml"
STAGE_A = ROOT / (
    "outputs/joint_dependency_v2/eth/joint_dependency_v2/runs/"
    "jdv2_stage_a_no_z_full_seed2035/saved_models/best_model.pt")
STAGE_B = ROOT / (
    "outputs/joint_dependency_v2/eth/joint_dependency_v2/runs/"
    "jdv2_stage_b_v1_eth_seed2035/saved_models/best_model.pt")


def _graph(device="cpu"):
    return torch.tensor([[0, 1, 3], [1, 2, 4]], device=device)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse(extra):
    values = [
        "--device", "cpu", "--data_augmentation", "False",
        "--goal_model_type", "joint_dependency_v2",
        "--training_stage", "joint_trajectory",
        "--use_scene_latent", "False",
        "--jdv2_latent_objective", "strict_no_z",
        "--jdv2_refinement_policy", "exact_lexicographic_persistent_tie",
    ] + extra
    args = parser_module.get_parser().parse_args(values)
    return parser_module.check_and_add_additional_args(args)


def test_metadata_is_canonical_and_excludes_isolates():
    scene = torch.tensor([0, 0, 0, 1, 1, 1])
    metadata = build_component_metadata(_graph(), 6, scene)
    assert metadata.degree.tolist() == [1, 2, 1, 1, 1, 0]
    assert metadata.active_mask.tolist() == [True] * 5 + [False]
    assert metadata.active_index.tolist() == [0, 1, 2, 3, 4]
    assert metadata.component_id.tolist() == [0, 0, 0, 1, 1, -1]
    assert metadata.component_count.tolist() == [3, 2]
    assert metadata.num_components == 2
    reversed_metadata = build_component_metadata(
        _graph().flip(1).flip(0), 6, scene)
    assert torch.equal(metadata.component_id,
                       reversed_metadata.component_id)


def test_projection_contracts_and_audit_counterfactual_parity():
    torch.manual_seed(7)
    raw = torch.randn(6, 12, 2, requires_grad=True)
    metadata = build_component_metadata(
        _graph(), 6, torch.tensor([0, 0, 0, 1, 1, 1]))
    projected = component_zero_mean_projection(raw, metadata)
    assert projected.dtype == torch.float32
    assert torch.allclose(projected[:3].sum(0), torch.zeros(12, 2), atol=1e-6)
    assert torch.allclose(projected[3:5].sum(0), torch.zeros(12, 2), atol=1e-6)
    assert torch.equal(projected[5], torch.zeros_like(projected[5]))
    src, dst = _graph()
    assert torch.allclose(projected[dst] - projected[src],
                          raw[dst] - raw[src], atol=1e-6)
    audited = decompose_component_residual(raw.detach(), _graph())["centered"]
    assert torch.equal(projected.detach(), audited)
    projected.square().mean().backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()


def test_e0_projection_is_exact_zero_and_does_not_consume_rng():
    edge = torch.empty(2, 0, dtype=torch.long)
    metadata = build_component_metadata(edge, 3, torch.zeros(3, dtype=torch.long))
    raw = torch.randn(3, 12, 2)
    before = rng_snapshot(use_cuda=False)
    result = component_zero_mean_projection(raw, metadata)
    after = rng_snapshot(use_cuda=False)
    assert torch.equal(result, torch.zeros_like(raw))
    assert torch.equal(before["torch"], after["torch"])
    assert before["python"] == after["python"]
    assert np.array_equal(before["numpy"][1], after["numpy"][1])


def test_metadata_rejects_invalid_graph_contracts():
    with pytest.raises(ValueError, match="cross-scene"):
        build_component_metadata(
            torch.tensor([[0], [1]]), 2, torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="out-of-range"):
        build_component_metadata(
            torch.tensor([[0], [2]]), 2, torch.tensor([0, 0]))
    with pytest.raises(ValueError, match="self edges"):
        build_component_metadata(
            torch.tensor([[0], [0]]), 1, torch.tensor([0]))


def test_v1_and_v2a_parser_and_config_contracts():
    v1 = yaml.safe_load(V1_CONFIG.read_text())
    v2a = yaml.safe_load(V2A_CONFIG.read_text())
    assert v1.get("jdv2_residual_projection", "none") == "none"
    assert v2a["stage_b_architecture_version"] == "jdv2-stage-b-v2a"
    assert v2a["jdv2_residual_projection"] == "component_zero_mean"
    args = _parse([
        "--stage_b_architecture_version", "jdv2-stage-b-v2a",
        "--jdv2_residual_projection", "component_zero_mean"])
    assert args.jdv2_residual_projection == "component_zero_mean"
    with pytest.raises(ValueError, match="requires jdv2_residual_projection"):
        _parse([
            "--stage_b_architecture_version", "jdv2-stage-b-v1",
            "--jdv2_residual_projection", "component_zero_mean"])
    with pytest.raises(ValueError, match="requires jdv2_residual_projection"):
        _parse(["--stage_b_architecture_version", "jdv2-stage-b-v2a"])


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

    def __call__(self, value, beta, context):
        del beta, context
        self.calls += 1
        return (value * .125).to(self.dtype)


class _Harness:
    ts_sample = GDTS.ts_sample

    def __init__(self, device, dtype=torch.float32):
        self.args = SimpleNamespace(
            pred_length=12, trunk_stage_step=2, ddim_step=2,
            branch_stage_step=1, num_samples=2, dataset="eth5",
            use_dependency_corrector=True, down_factor=1,
            trajectory_dt=.4,
            jdv2_residual_projection="component_zero_mean")
        self.var_sched = _Schedule(device)
        self.diffnet = _Diffnet(dtype)
        self.jdv2_corrector = DependencyCorrector().to(device)


def _contexts(device, n=3):
    return [torch.zeros(n, 1, 4, device=device) for _ in range(3)]


def _state(device, mixed=True):
    edge = (torch.tensor([[0], [1]], device=device) if mixed else
            torch.empty(2, 0, dtype=torch.long, device=device))
    scene = torch.zeros(3, dtype=torch.long, device=device)
    return {
        "edge_index": edge,
        "edge_weight": torch.ones(edge.shape[1], device=device),
        "relation_embedding": torch.zeros(
            edge.shape[1], 2, 16, device=device),
        "last_position_map": torch.zeros(3, 2, device=device),
        "last_position_world": torch.zeros(3, 2, device=device),
        "scene": _Scene(),
        "component_metadata": build_component_metadata(edge, 3, scene),
    }


def _paired_zero_init(device, dtype):
    harness = _Harness(device, dtype)
    contexts = _contexts(device)
    state = _state(device)
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(17)
    projected = harness.ts_sample(contexts, state)
    projected_rng = rng_snapshot(use_cuda=device.type == "cuda")
    corrector_calls = harness.diffnet.calls
    harness.args.use_dependency_corrector = False
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(17)
    stage_a = harness.ts_sample(contexts, state)
    stage_a_rng = rng_snapshot(use_cuda=device.type == "cuda")
    return projected, stage_a, projected_rng, stage_a_rng, corrector_calls


def test_cpu_zero_init_v2a_is_tensor_exact_stage_a():
    projected, stage_a, projected_rng, stage_a_rng, _ = _paired_zero_init(
        torch.device("cpu"), torch.float32)
    assert torch.equal(projected, stage_a)
    assert torch.equal(projected_rng["torch"], stage_a_rng["torch"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_bf16_projection_and_zero_init_identity():
    device = torch.device("cuda")
    raw = torch.randn(6, 12, 2, device=device, dtype=torch.bfloat16)
    metadata = build_component_metadata(
        _graph(device), 6,
        torch.tensor([0, 0, 0, 1, 1, 1], device=device))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        projected = component_zero_mean_projection(raw, metadata)
        v2a, stage_a, v2a_rng, stage_a_rng, _ = _paired_zero_init(
            device, torch.bfloat16)
    assert projected.dtype == torch.float32
    assert torch.isfinite(projected).all()
    assert torch.allclose(projected[:3].sum(0), torch.zeros(
        12, 2, device=device), atol=2e-5)
    assert torch.equal(v2a, stage_a)
    assert torch.equal(v2a_rng["torch"], stage_a_rng["torch"])
    assert all(torch.equal(left, right) for left, right in zip(
        v2a_rng["cuda"], stage_a_rng["cuda"]))


def test_dependency_corrector_and_protected_artifacts_unchanged():
    assert sum(parameter.numel()
               for parameter in DependencyCorrector().parameters()) == 30851
    assert _sha256(ROOT / "src/models/joint_dependency_v2/"
                   "dependency_corrector.py") == \
        "0ff28eb7107e49a0ad6f2bf339a65d8847f83c4a7cf315faa5a80cac4dc00522"
    if STAGE_A.exists():
        assert _sha256(STAGE_A) == \
            "699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb"
    if STAGE_B.exists():
        assert _sha256(STAGE_B) == \
            "e24a1cbfc5760db57bcb5cacd64c66a4afd1aed0f11bb1efefa8549933d21c1a"

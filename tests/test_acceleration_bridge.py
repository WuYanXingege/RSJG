"""Boundary and checkpoint contracts for the optional acceleration prior."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from ebjd.config import build_model, semantic_config_hash, validate_supported_config
from ebjd.model import EBJDModel
from ebjd.representation import (
    EndpointBridgeDecomposition, conditional_bridge_statistics, cv_baseline,
)

from conftest import load_paper_config


ACCELERATION = "integrated_acceleration_v1"


def test_default_brownian_buffers_remain_exactly_the_archived_values():
    times = torch.arange(1, 12, dtype=torch.float64) / 12
    covariance = torch.minimum(times[:, None], times[None, :])
    covariance = covariance - times[:, None] * times[None, :]
    representation = EndpointBridgeDecomposition()
    assert torch.equal(representation.a, times.float())
    assert torch.equal(
        representation.bridge_cholesky, torch.linalg.cholesky(covariance).float())
    explicit = EndpointBridgeDecomposition(bridge_prior="brownian_position")
    assert representation.state_dict().keys() == explicit.state_dict().keys()
    for key, value in representation.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[key])


def test_acceleration_statistics_match_independent_white_acceleration_integrals():
    # Four-node Gauss integration exactly integrates this quadratic integrand;
    # this deliberately does not reuse the closed-form covariance expression.
    times = torch.tensor([0.0, 0.03, 0.2, 0.5, 0.83, 1.0], dtype=torch.float64)
    nodes, weights = np.polynomial.legendre.leggauss(4)

    def integral(left, right):
        upper = min(float(left), float(right))
        points = (nodes + 1) * (upper / 2)
        return np.sum(weights * (float(left) - points) *
                      (float(right) - points)) * (upper / 2)

    raw = torch.tensor(
        [[integral(left, right) for right in times] for left in times],
        dtype=torch.float64)
    cross = torch.tensor([integral(value, 1.0) for value in times])
    terminal = integral(1.0, 1.0)
    expected_mean = cross / terminal
    expected_covariance = raw - cross[:, None] * cross[None, :] / terminal
    mean, covariance = conditional_bridge_statistics(times, ACCELERATION)
    assert mean.device == times.device and covariance.device == times.device
    assert mean.dtype == torch.float64 and covariance.dtype == torch.float64
    torch.testing.assert_close(mean, expected_mean, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(
        covariance, expected_covariance, atol=1e-14, rtol=1e-14)
    assert torch.linalg.eigvalsh(covariance).min() >= -1e-14
    torch.testing.assert_close(
        covariance[[0, -1]], torch.zeros(2, len(times), dtype=torch.float64),
        atol=1e-14, rtol=0)
    interior = torch.arange(1, 12, dtype=torch.float64) / 12
    _, interior_covariance = conditional_bridge_statistics(interior, ACCELERATION)
    assert torch.linalg.eigvalsh(interior_covariance).min() > 0
    torch.linalg.cholesky(interior_covariance)


def test_acceleration_reference_has_exact_endpoint_and_zero_initial_derivative():
    times = torch.tensor([0.0, 0.1, 0.5, 1.0], dtype=torch.float64,
                         requires_grad=True)
    mean, covariance = conditional_bridge_statistics(times, ACCELERATION)
    torch.testing.assert_close(mean, 0.5 * times.square() * (3 - times),
                               atol=1e-14, rtol=1e-14)
    derivative, = torch.autograd.grad(mean.sum(), times)
    assert mean[0].item() == 0 and mean[-1].item() == 1
    assert derivative[0].item() == 0
    # This is a continuous-time reference-process boundary, not an assertion
    # that the first discrete predicted velocity equals observed velocity.
    assert covariance[0].abs().max().item() < 1e-14


@pytest.mark.parametrize("eta", [0.0, 1.0])
def test_conditioned_acceleration_roundtrip_endpoint_and_gradient(eta):
    torch.manual_seed(802)
    observed = torch.randn(2, 3, 8, 2)
    baseline, _ = cv_baseline(observed)
    target = (baseline + 0.2 * torch.randn_like(baseline)).requires_grad_()
    proposal = target.detach()[..., -1, :] + torch.randn(2, 3, 2)
    representation = EndpointBridgeDecomposition(
        goal_scale=1.7, bridge_scale=0.8, bridge_prior=ACCELERATION)
    latent = representation.encode_target_conditioned(
        target, baseline, proposal, eta=eta)
    decoded, goal = representation.decode_conditioned(
        latent, baseline, proposal, eta=eta)
    torch.testing.assert_close(decoded[..., :11, :], target[..., :11, :],
                               atol=2e-5, rtol=2e-5)
    expected_goal = proposal if eta == 0 else target[..., -1, :]
    torch.testing.assert_close(goal, expected_goal, atol=2e-5, rtol=2e-5)
    assert torch.equal(decoded[..., -1, :], goal)
    decoded.square().mean().backward()
    assert target.grad is not None and torch.isfinite(target.grad).all()
    assert target.grad[..., :11, :].abs().sum() > 0
    if eta == 0:
        assert torch.count_nonzero(target.grad[..., -1, :]) == 0


def test_acceleration_unconditioned_roundtrip_and_world_agent_equivariance():
    torch.manual_seed(803)
    observed = torch.randn(2, 3, 8, 2)
    baseline, _ = cv_baseline(observed)
    target = baseline + 0.2 * torch.randn_like(baseline)
    representation = EndpointBridgeDecomposition(bridge_prior=ACCELERATION)
    encoded = representation.encode_target(target, baseline)
    decoded, goal = representation.decode(encoded, baseline)
    torch.testing.assert_close(decoded, target, atol=2e-5, rtol=2e-5)
    assert torch.equal(decoded[..., -1, :], goal)
    latent = torch.randn(2, 4, 3, 12, 2, requires_grad=True)
    paths, _ = representation.decode(latent, baseline)
    world_order = torch.tensor([2, 0, 3, 1])
    agent_order = torch.tensor([1, 2, 0])
    permuted, _ = representation.decode(
        latent[:, world_order][:, :, agent_order], baseline[:, agent_order])
    torch.testing.assert_close(
        permuted, paths[:, world_order][:, :, agent_order], atol=0, rtol=0)
    paths.square().mean().backward()
    assert latent.grad is not None and torch.isfinite(latent.grad).all()
    assert latent.grad[..., :11, :].abs().sum() > 0


@pytest.mark.parametrize("prior", ["brownian_position", ACCELERATION])
def test_strict_checkpoint_loading_preserves_prior_family(prior):
    model = EBJDModel(hidden_dim=32, ff_dim=64, bridge_prior=prior)
    restored = EBJDModel(hidden_dim=32, ff_dim=64, bridge_prior=prior)
    restored.load_state_dict(model.state_dict(), strict=True)
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    other = (ACCELERATION if prior == "brownian_position" else "brownian_position")
    incompatible = EBJDModel(hidden_dim=32, ff_dim=64, bridge_prior=other)
    with pytest.raises(RuntimeError, match="bridge prior"):
        incompatible.load_state_dict(model.state_dict(), strict=True)


def test_acceleration_is_explicit_research_semantics_not_runtime_optimization():
    legacy = load_paper_config()
    candidate = deepcopy(legacy)
    candidate["representation"]["type"] = "invertible_endpoint_acceleration_bridge_v1"
    validate_supported_config(candidate)
    assert semantic_config_hash(candidate) != semantic_config_hash(legacy)
    model = build_model(candidate, proposal_conditioned=True)
    expected, _ = conditional_bridge_statistics(
        torch.arange(1, 12, dtype=torch.float64) / 12, ACCELERATION)
    torch.testing.assert_close(model.representation.a, expected.float(), atol=0, rtol=0)


def test_research_arm_preparation_keeps_interventions_separate(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from prepare_acceleration_bridge_research import prepare

    base = load_paper_config()
    base["training"]["early_stopping"] = {
        "monitor": "H_JADE", "max_epochs": base["training"]["epochs"],
        "earliest_stop_epoch": 10, "patience_validations": 8,
        "significant_relative_improvement": 0.001,
    }
    before = deepcopy(base)
    candidate = prepare(base, {
        "representation_type": "invertible_endpoint_acceleration_bridge_v1",
        "coupling_singleton_backward": "legacy",
    }, tmp_path / "new", "0" * 40)
    assert base == before
    assert candidate["training"]["epochs"] == base["training"]["epochs"]
    assert candidate["training"]["coupling_singleton_backward"] == "legacy"
    assert candidate["training"]["rollout_training_enabled"] is False
    assert candidate["training"]["paired_marginal_actual_step_projection_enabled"] is False
    assert candidate["denoiser"]["ctf_feedback_basis"] == "legacy"
    singleton = prepare(base, {
        "representation_type": "invertible_endpoint_brownian_bridge",
        "coupling_singleton_backward": "identity",
    }, tmp_path / "other", "0" * 40)
    assert singleton["representation"]["type"] == base["representation"]["type"]
    assert semantic_config_hash(singleton) != semantic_config_hash(candidate)


def test_research_output_rejects_existing_training_state(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from prepare_acceleration_bridge_research import prepare

    (tmp_path / "last.pt").touch()
    with pytest.raises(ValueError, match="already contains a training checkpoint"):
        prepare(load_paper_config(), {
            "representation_type": "invertible_endpoint_acceleration_bridge_v1",
            "coupling_singleton_backward": "legacy",
        }, tmp_path, "0" * 40)


def test_research_preparation_rejects_incomplete_historical_contract(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from prepare_acceleration_bridge_research import prepare

    with pytest.raises(ValueError, match="complete unified"):
        prepare(load_paper_config(), {
            "representation_type": "invertible_endpoint_acceleration_bridge_v1",
            "coupling_singleton_backward": "legacy",
        }, tmp_path / "fresh", "0" * 40)

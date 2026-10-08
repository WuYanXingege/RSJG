import torch

from ebjd.representation import (
    CartesianVelocityRepresentation, EndpointBridgeRepresentation, cosine_vp,
    cv_baseline,
)


def test_endpoint_bridge_roundtrip_and_literal_endpoint():
    torch.manual_seed(2)
    observed = torch.randn(2, 3, 8, 2)
    baseline, _ = cv_baseline(observed)
    target = baseline + torch.randn_like(baseline)
    representation = EndpointBridgeRepresentation(1.7, 0.8)
    latent = representation.encode_target(target, baseline)
    decoded, goal = representation.decode(latent, baseline)
    torch.testing.assert_close(decoded, target, atol=2e-5, rtol=2e-5)
    assert torch.equal(decoded[..., -1, :], goal)


def test_cosine_terminal_and_near_zero_are_finite():
    time = torch.tensor([0.0, 1e-8, 0.5, 1.0])
    alpha, sigma = cosine_vp(time)
    assert alpha[0] == 1 and sigma[0] == 0
    assert alpha[-1] == 0 and sigma[-1] == 1
    assert torch.isfinite(alpha).all() and torch.isfinite(sigma).all()


def test_cartesian_velocity_ablation_is_also_invertible():
    observed = torch.randn(1, 2, 8, 2)
    baseline, _ = cv_baseline(observed)
    target = baseline + 0.2 * torch.randn_like(baseline)
    representation = CartesianVelocityRepresentation(0.7)
    decoded, goal = representation.decode(
        representation.encode_target(target, baseline), baseline)
    torch.testing.assert_close(decoded, target, atol=2e-5, rtol=2e-5)
    assert torch.equal(decoded[..., -1, :], goal)

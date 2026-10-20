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


def test_anchor_residual_roundtrip_world_broadcast_and_literal_endpoint():
    torch.manual_seed(21)
    observed = torch.randn(2, 3, 8, 2)
    baseline, _ = cv_baseline(observed)
    target = baseline + torch.randn_like(baseline)
    anchors = target[..., -1, :] + 0.3 * torch.randn(2, 3, 2)
    representation = EndpointBridgeRepresentation(
        1.7, 0.8, anchor_residual=True)
    latent = representation.encode_target(target, baseline, anchors)
    decoded, goal = representation.decode(latent, baseline, anchors)
    torch.testing.assert_close(decoded, target, atol=2e-5, rtol=2e-5)
    assert torch.equal(decoded[..., -1, :], goal)

    world_latent = latent[:, None].expand(2, 4, 3, 12, 2).clone()
    world_anchors = anchors[:, None].expand(2, 4, 3, 2).clone()
    world_decoded, world_goal = representation.decode(
        world_latent, baseline, world_anchors)
    torch.testing.assert_close(
        world_decoded, target[:, None].expand_as(world_decoded),
        atol=2e-5, rtol=2e-5)
    assert torch.equal(world_decoded[..., -1, :], world_goal)


def test_anchor_residual_requires_anchor_and_jacobian_matches_finite_difference():
    representation = EndpointBridgeRepresentation(
        1.4, 0.3, anchor_residual=True)
    baseline = torch.zeros(1, 1, 12, 2)
    latent = torch.zeros(1, 1, 1, 12, 2)
    anchors = torch.zeros(1, 1, 1, 2)
    try:
        representation.decode(latent, baseline)
    except ValueError as error:
        assert "requires world anchors" in str(error)
    else:
        raise AssertionError("anchor-residual decode silently accepted no anchor")

    weights = representation.decoder_jacobian_weights()
    torch.testing.assert_close(weights.sum(-1), torch.ones(12))
    epsilon = 1e-4
    for token in (0, 5, 10, 11):
        perturbed = latent.clone()
        perturbed[..., token, 0] = epsilon
        decoded, _ = representation.decode(perturbed, baseline, anchors)
        numerical = decoded[0, 0, 0, :, 0] / epsilon
        if token < 11:
            expected = torch.zeros(12)
            expected[:11] = (
                representation.bridge_scale
                * representation.bridge_cholesky[:, token])
        else:
            expected = representation.goal_scale * torch.arange(1, 13) / 12
        torch.testing.assert_close(numerical, expected, atol=2e-3, rtol=2e-3)


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

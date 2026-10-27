import copy

import torch

from ebjd.encoders import EBJDContext
from ebjd.model import EBJDModel
from ebjd.data import (
    SceneBatch, augment_batch, augment_batch_with_gt_endpoint_condition,
    augment_batch_with_points,
)
from ebjd.proposals import NativeProposalBank
from ebjd.representation import EndpointBridgeDecomposition


def test_conditioned_representation_definitions_are_exact():
    torch.manual_seed(4)
    rep = EndpointBridgeDecomposition(goal_scale=1.7, bridge_scale=0.8)
    baseline = torch.randn(2, 3, 12, 2)
    target = baseline + torch.randn_like(baseline) * 0.2
    proposal = target[..., -1, :] + torch.randn(2, 3, 2) * 0.3

    control = rep.encode_target_conditioned(target, baseline, proposal, eta=0)
    control_path, control_goal = rep.decode_conditioned(
        control, baseline, proposal, eta=0)
    assert torch.equal(control[..., 11, :], torch.zeros_like(control[..., 11, :]))
    torch.testing.assert_close(control_path[..., :11, :], target[..., :11, :])
    torch.testing.assert_close(control_goal, proposal)

    candidate = rep.encode_target_conditioned(target, baseline, proposal, eta=1)
    candidate_path, candidate_goal = rep.decode_conditioned(
        candidate, baseline, proposal, eta=1)
    torch.testing.assert_close(candidate_path, target)
    torch.testing.assert_close(candidate_goal, target[..., -1, :])


def test_proposal_model_adds_only_embedding_parameters_and_zero_starts():
    torch.manual_seed(5)
    legacy = EBJDModel(hidden_dim=32, attention_heads=4, ff_dim=64)
    conditioned = EBJDModel(
        hidden_dim=32, attention_heads=4, ff_dim=64,
        proposal_conditioned=True)
    legacy_state = legacy.state_dict()
    candidate_state = conditioned.state_dict()
    additions = set(candidate_state) - set(legacy_state)
    assert additions == {
        "denoiser.proposal_embedding.0.weight",
        "denoiser.proposal_embedding.0.bias",
        "denoiser.proposal_embedding.2.weight",
        "denoiser.proposal_embedding.2.bias",
    }
    for name, value in legacy_state.items():
        candidate_state[name].copy_(value)
    conditioned.load_state_dict(candidate_state, strict=True)
    assert torch.count_nonzero(conditioned.denoiser.proposal_embedding[-1].weight) == 0
    assert torch.count_nonzero(conditioned.denoiser.proposal_embedding[-1].bias) == 0


def test_eta_zero_endpoint_is_literal_proposal_and_has_finite_gradients():
    torch.manual_seed(6)
    model = EBJDModel(
        hidden_dim=32, attention_heads=4, ff_dim=64,
        proposal_conditioned=True)
    batch, worlds, agents = 1, 2, 2
    valid = torch.ones(batch, agents, dtype=torch.bool)
    context = EBJDContext(
        observed=torch.randn(batch, agents, 8, 2),
        baseline=torch.randn(batch, agents, 12, 2),
        mean_velocity=torch.randn(batch, agents, 2),
        social_h=torch.randn(batch, agents, 32),
        memory=torch.randn(batch, agents, 72, 32),
        map_features=torch.randn(batch, agents, 32, 8, 8),
        map_logits=torch.randn(batch, agents, 12, 8, 8),
        map_center=torch.randn(batch, agents, 2),
        observed_bias=torch.zeros(batch, 4, agents, agents),
        valid=valid, crop_width_m=32.0,
    )
    latent = torch.randn(batch, worlds, agents, 12, 2, requires_grad=True)
    proposal = torch.randn(batch, worlds, agents, 2)
    final, _, coarse_path, coarse_goal = model.denoiser(
        latent, torch.tensor([0.5]), context, checkpoint_blocks=False,
        proposal_goals=proposal, proposal_eta=0)
    torch.testing.assert_close(coarse_goal, proposal)
    torch.testing.assert_close(coarse_path[..., -1, :], proposal)
    final.square().mean().backward()
    assert torch.isfinite(latent.grad).all()


def test_training_world_selection_is_scene_level_and_stable():
    proposals = torch.tensor([[[[1.0, 0.0], [9.0, 0.0]],
                               [[2.0, 0.0], [2.0, 0.0]],
                               [[9.0, 0.0], [1.0, 0.0]]]])
    future = torch.zeros(1, 2, 12, 2)
    valid = torch.ones(1, 2, dtype=torch.bool)
    selected, index = NativeProposalBank.select_training_world(
        proposals, future, valid)
    assert index.tolist() == [1]
    torch.testing.assert_close(selected, proposals[:, 1])


def test_joint_augmentation_preserves_legacy_and_point_offsets():
    observed = torch.zeros(1, 2, 8, 2)
    observed[0, 0, -1] = torch.tensor([1.0, 0.0])
    observed[0, 1, -1] = torch.tensor([3.0, 0.0])
    batch = SceneBatch(
        observed, torch.ones(1, 2, 12, 2),
        torch.zeros(1, 2, 6, 8, 8), torch.ones(1, 2, dtype=torch.bool),
        ["s"], [["1", "2"]], [list(range(20))],
        [[0.4 * i for i in range(20)]], ["x"], {})
    points = observed[:, None, :, -1, :] + torch.tensor([0.5, -0.25])
    left_generator = torch.Generator().manual_seed(17)
    right_generator = torch.Generator().manual_seed(17)
    legacy = augment_batch(batch, left_generator)
    paired, transformed = augment_batch_with_points(
        batch, points, right_generator)
    torch.testing.assert_close(legacy.observed, paired.observed)
    torch.testing.assert_close(legacy.future, paired.future)
    torch.testing.assert_close(legacy.semantic_maps, paired.semantic_maps)
    original_distance = torch.linalg.vector_norm(
        points[:, 0] - observed[:, :, -1, :], dim=-1)
    transformed_distance = torch.linalg.vector_norm(
        transformed[:, 0] - paired.observed[:, :, -1, :], dim=-1)
    torch.testing.assert_close(original_distance, transformed_distance)


def test_gt_condition_roundtrip_matches_original_bridge_and_zeroes_padding():
    torch.manual_seed(23)
    rep = EndpointBridgeDecomposition(goal_scale=1.7, bridge_scale=0.8)
    baseline = torch.randn(2, 3, 12, 2)
    target = baseline + torch.randn_like(baseline) * 0.2
    goal = target[..., -1, :]
    conditioned = rep.encode_target_conditioned(target, baseline, goal, eta=0.0)
    original = rep.encode_target(target, baseline)
    decoded, decoded_goal = rep.decode_conditioned(
        conditioned, baseline, goal, eta=0.0)
    torch.testing.assert_close(decoded, target)
    torch.testing.assert_close(decoded_goal, goal)
    torch.testing.assert_close(conditioned[..., :11, :], original[..., :11, :])
    assert torch.equal(
        conditioned[..., 11, :], torch.zeros_like(conditioned[..., 11, :]))

    batch = SceneBatch(
        observed=torch.randn(1, 3, 8, 2),
        future=torch.randn(1, 3, 12, 2),
        semantic_maps=torch.zeros(1, 3, 6, 8, 8),
        valid=torch.tensor([[True, True, False]]),
        scene_ids=["s"], agent_ids=[["a", "b"]],
        frame_ids=[list(range(20))],
        timestamps=[[0.4 * i for i in range(20)]],
        source_sequences=["x"], metadata={})
    augmented, condition = augment_batch_with_gt_endpoint_condition(
        batch, torch.Generator().manual_seed(31))
    torch.testing.assert_close(
        condition[augmented.valid], augmented.future[..., -1, :][augmented.valid])
    assert torch.equal(condition[~augmented.valid], torch.zeros_like(condition[~augmented.valid]))


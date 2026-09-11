import torch

from src.models.joint_sampler import (
    JointGoalSampler,
    estimate_joint_goal_complexity,
)


def test_map_refinement_uses_pair_energy_and_updates_both_edge_ends():
    candidates = torch.tensor([
        [[3.0, 0.0], [3.0, 3.0]],
        [[3.0, 1.0], [3.0, -3.0]],
    ])
    # A initially chooses 0; B initially chooses 1. Only pair (0,0) receives
    # strong learned compatibility, so one blocked update coordinates both.
    conditional_logits = torch.tensor([[[2.0, 0.0]], [[0.0, 2.0]]])
    left_factor = torch.tensor([[[[10.0], [0.0]]]])
    right_factor = torch.tensor([[[[10.0], [0.0]]]])
    energy_output = {
        "edge_index": torch.tensor([[0], [1]]),
        "left_factor": left_factor,
        "right_factor": right_factor,
        "relation_prob": torch.ones(1, 1),
        "edge_weight": torch.ones(1),
    }
    sampler = JointGoalSampler(
        num_samples=2, num_refinement_steps=1,
        energy_weight=1.0, sampling_mode="map")
    details = sampler(
        candidates,
        mode_log_prob=torch.zeros(1, 1),
        conditional_goal_log_prob=conditional_logits,
        scene_index=torch.tensor([7, 7]),
        energy_output=energy_output,
        return_details=True,
    )

    assert details["joint_goal_points"].shape == (2, 2, 2)
    assert details["candidate_index"].tolist() == [[0, 0], [0, 0]]
    torch.testing.assert_close(
        details["joint_goal_points"][:, 0], candidates[:, 0])


def test_scene_modes_are_shared_within_each_scene_sample():
    candidates = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0]],
        [[0.0, 1.0], [1.0, 1.0]],
        [[5.0, 0.0], [6.0, 0.0]],
    ])
    # Scene 10 selects mode 0 -> candidate 0. Scene 90 selects mode 1 -> cand 1.
    mode_logits = torch.tensor([[10.0, -10.0], [-10.0, 10.0]])
    conditional_logits = torch.tensor([
        [[10.0, -10.0], [-10.0, 10.0]],
        [[10.0, -10.0], [-10.0, 10.0]],
        [[10.0, -10.0], [-10.0, 10.0]],
    ])
    sampler = JointGoalSampler(
        num_samples=3, num_refinement_steps=0, sampling_mode="map")
    details = sampler(
        candidates, mode_logits, conditional_logits,
        scene_index=torch.tensor([10, 10, 90]), return_details=True)

    assert details["social_mode_index"].tolist() == [[0, 0, 0], [1, 1, 1]]
    assert details["candidate_index"].tolist() == [[0, 0, 0],
                                                   [0, 0, 0],
                                                   [1, 1, 1]]
    # One sample column contains every agent and hence both complete scenes.
    assert details["joint_goal_points"][:, 0].shape == (3, 2)


def test_sampling_is_reproducible_and_handles_single_agent_no_edge():
    candidates = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    mode_logits = torch.tensor([[0.0, 0.0]])
    conditional_logits = torch.zeros(1, 2, 3)
    sampler = JointGoalSampler(
        num_samples=8, num_refinement_steps=4, sampling_mode="sample")

    generator_a = torch.Generator().manual_seed(123)
    generator_b = torch.Generator().manual_seed(123)
    output_a = sampler.sample(
        candidates, mode_logits, conditional_logits,
        scene_index=torch.tensor([4]), generator=generator_a)
    output_b = sampler.sample(
        candidates, mode_logits, conditional_logits,
        scene_index=torch.tensor([4]), generator=generator_b)
    assert output_a.shape == (1, 8, 2)
    assert torch.isfinite(output_a).all()
    torch.testing.assert_close(output_a, output_b)


def test_coverage_strategy_uses_every_allowed_candidate_before_repeating():
    candidates = torch.tensor([[
        [0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0],
    ]])
    # A very peaked unary would collapse iid sampling to candidate zero.  The
    # coverage strategy still retains all three TTST branch representatives;
    # candidate three is reserved for the tree trunk by the mask.
    conditional_logits = torch.tensor([[[20.0, 2.0, 1.0, 0.0]]])
    sampler = JointGoalSampler(
        num_samples=3, num_refinement_steps=0,
        sampling_mode="map", sampling_strategy="coverage")
    details = sampler(
        candidates,
        mode_log_prob=torch.zeros(1, 1),
        conditional_goal_log_prob=conditional_logits,
        candidate_mask=torch.tensor([[True, True, True, False]]),
        return_details=True,
    )

    assert sorted(details["candidate_index"][0].tolist()) == [0, 1, 2]


def test_coverage_strategy_preserves_diversity_after_pair_refinement():
    candidates = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
    ])
    energy_output = {
        "edge_index": torch.tensor([[0], [1]]),
        "left_factor": torch.ones(1, 1, 3, 1),
        "right_factor": torch.ones(1, 1, 3, 1),
        "relation_prob": torch.ones(1, 1),
        "edge_weight": torch.ones(1),
    }
    sampler = JointGoalSampler(
        num_samples=3, num_refinement_steps=2,
        sampling_mode="sample", sampling_strategy="coverage")
    details = sampler(
        candidates, torch.zeros(1, 1), torch.zeros(2, 1, 3),
        scene_index=torch.zeros(2, dtype=torch.long),
        energy_output=energy_output,
        generator=torch.Generator().manual_seed(9),
        return_details=True,
    )

    for row in details["candidate_index"]:
        assert sorted(row.tolist()) == [0, 1, 2]


def test_sampler_rejects_cross_scene_edges():
    candidates = torch.randn(2, 2, 2)
    energy_output = {
        "edge_index": torch.tensor([[0], [1]]),
        "left_factor": torch.randn(1, 1, 2, 2),
        "right_factor": torch.randn(1, 1, 2, 2),
        "relation_prob": torch.ones(1, 1),
    }
    sampler = JointGoalSampler(num_samples=1, sampling_mode="map")
    try:
        sampler(
            candidates, torch.zeros(2, 1), torch.zeros(2, 1, 2),
            scene_index=torch.tensor([0, 1]), energy_output=energy_output)
    except ValueError as error:
        assert "cross scene" in str(error)
    else:
        raise AssertionError("Cross-scene interaction edge was not rejected.")


def test_complexity_diagnostic_matches_structured_formula():
    diagnostic = estimate_joint_goal_complexity(
        num_agents=6,
        num_candidates=20,
        num_social_modes=4,
        num_relation_modes=5,
        pair_rank=8,
        num_edges=7,
        num_refinement_steps=3,
    )
    assert diagnostic["naive_joint_states"] == 20 ** 6
    expected = 4 * 6 * 20 + 3 * 7 * 20 * 8 * 5
    assert diagnostic["structured_computation_estimate"] == expected
    assert diagnostic["structured_computation_estimate"] < \
        diagnostic["naive_joint_states"]

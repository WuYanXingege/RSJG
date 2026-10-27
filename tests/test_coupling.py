import numpy as np
import torch

from ebjd.coupling import couple_scene, gather_worlds, relative_pairing_change


def _endpoint_metrics(points, truth, valid):
    error = np.linalg.norm(points - truth[None], axis=-1)
    best = error.min(0)[valid]
    joint = error[:, valid].mean(1).min()
    return best.mean(), np.mean(best <= 0.5), joint


def test_exact_assignment_preserves_indexed_multiset_and_repeated_points():
    proposals = np.array([
        [[0., 0.], [0., 0.]], [[0., 0.], [2., 0.]],
        [[2., 0.], [1., 0.]], [[3., 0.], [3., 0.]],
    ], dtype=np.float32)
    reference = proposals[[2, 0, 3, 1]].copy()
    valid = np.array([True, True])
    coupled, permutation = couple_scene(proposals, reference, valid)
    for agent in range(2):
        assert np.array_equal(np.sort(permutation[:, agent]), np.arange(4))
        # Compare indexed rows, not unique coordinates: duplicate candidates
        # remain distinct members of the exact assignment.
        np.testing.assert_array_equal(coupled[:, agent], proposals[permutation[:, agent], agent])
        np.testing.assert_array_equal(
            np.sort(coupled[:, agent], axis=0), np.sort(proposals[:, agent], axis=0))


def test_marginals_stay_exact_and_pairing_can_change_joint_metric():
    proposals = np.array([
        [[0., 0.], [2., 0.]], [[2., 0.], [0., 0.]],
        [[4., 0.], [6., 0.]], [[6., 0.], [4., 0.]],
    ], dtype=np.float32)
    truth = np.array([[0., 0.], [0., 0.]], dtype=np.float32)
    valid = np.array([True, True])
    global_reference = proposals[[1, 0, 3, 2]]
    globally, global_pi = couple_scene(proposals, global_reference, valid)
    before = _endpoint_metrics(proposals, truth, valid)
    after_global = _endpoint_metrics(globally, truth, valid)
    np.testing.assert_allclose(before, after_global)
    assert relative_pairing_change(global_pi, valid)["mean_relative_change"] == 0.0

    reference = proposals.copy()
    reference[:, 1] = proposals[[1, 0, 3, 2], 1]
    coupled, pi = couple_scene(proposals, reference, valid)
    after = _endpoint_metrics(coupled, truth, valid)
    np.testing.assert_allclose(before[:2], after[:2])
    assert before[2] != after[2]
    assert relative_pairing_change(pi, valid)["mean_relative_change"] > 0.0


def test_noise_and_proposals_use_identical_gather_and_single_agent_is_identity():
    generator = torch.Generator().manual_seed(7)
    values = torch.randn(1, 4, 2, 12, 2, generator=generator)
    permutation = torch.tensor([[[2, 1], [0, 3], [3, 0], [1, 2]]])
    gathered = gather_worlds(values, permutation)
    for world in range(4):
        for agent in range(2):
            torch.testing.assert_close(
                gathered[0, world, agent],
                values[0, permutation[0, world, agent], agent])

    proposal = np.random.default_rng(9).normal(size=(4, 2, 2)).astype(np.float32)
    reference = proposal[::-1].copy()
    coupled, identity = couple_scene(proposal, reference, np.array([True, False]))
    np.testing.assert_array_equal(coupled, proposal)
    np.testing.assert_array_equal(
        identity, np.broadcast_to(np.arange(4)[:, None], (4, 2)))

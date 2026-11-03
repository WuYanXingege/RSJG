from __future__ import annotations

import numpy as np
import torch

from ebjd.collision_metrics import (
    collision_flags,
    ground_truth_collision_rate,
    scene_collision_statistics,
)


def _prediction(*agents: list[tuple[float, float]]) -> torch.Tensor:
    return torch.tensor([[agents]], dtype=torch.float32).squeeze(0)


def test_safe_distance_and_single_agent_are_zero() -> None:
    safe = _prediction(
        [(0.0, 0.0), (1.0, 0.0)],
        [(0.0, 1.0), (1.0, 1.0)],
    )
    assert not collision_flags(safe).any()
    single = _prediction([(0.0, 0.0), (1.0, 0.0)])
    stats = scene_collision_statistics(single, single[0])
    assert stats.cr_jade == stats.cr_mean == 0.0


def test_mid_interval_collision_with_safe_discrete_endpoints() -> None:
    crossing = _prediction(
        [(-1.0, 0.0), (1.0, 0.0)],
        [(1.0, 0.0), (-1.0, 0.0)],
    )
    assert collision_flags(crossing).tolist() == [[True, True]]


def test_spatial_crossing_at_different_times_is_not_collision() -> None:
    # The spatial segments cross, but the two synchronized moving points never
    # approach within 0.2 m during the interval.
    asynchronous = _prediction(
        [(-1.0, 0.0), (3.0, 0.0)],
        [(0.0, -3.0), (0.0, 1.0)],
    )
    assert not collision_flags(asynchronous).any()


def test_stationary_zero_length_segment_and_strict_threshold() -> None:
    touching_threshold = _prediction(
        [(0.0, 0.0), (0.0, 0.0)],
        [(0.2, 0.0), (0.2, 0.0)],
    )
    assert not collision_flags(touching_threshold).any()
    inside = touching_threshold.clone()
    inside[:, 1, :, 0] = 0.199
    assert collision_flags(inside).all()


def test_multiple_partners_count_once_and_self_is_excluded() -> None:
    crowded = _prediction(
        [(0.0, 0.0), (0.0, 0.0)],
        [(0.05, 0.0), (0.05, 0.0)],
        [(-0.05, 0.0), (-0.05, 0.0)],
    )
    flags = collision_flags(crowded)
    assert flags.tolist() == [[True, True, True]]
    stats = scene_collision_statistics(crowded, crowded[0])
    assert stats.collision_numerator_jade == 3


def test_padding_and_agent_permutation_do_not_change_rates() -> None:
    base = _prediction(
        [(-1.0, 0.0), (1.0, 0.0)],
        [(1.0, 0.0), (-1.0, 0.0)],
    )
    padded = torch.cat((base, torch.full((1, 1, 2, 2), 99.0)), dim=1)
    flags = collision_flags(padded, torch.tensor([True, True, False]))
    assert flags.tolist() == [[True, True, False]]
    permuted = base[:, torch.tensor([1, 0])]
    assert collision_flags(permuted).sum() == collision_flags(base).sum()


def test_world_reordering_preserves_crmean_and_gt_only_changes_crjade_choice() -> None:
    safe = _prediction(
        [(0.0, 0.0), (1.0, 0.0)],
        [(0.0, 1.0), (1.0, 1.0)],
    )
    colliding = _prediction(
        [(-1.0, 0.0), (1.0, 0.0)],
        [(1.0, 0.0), (-1.0, 0.0)],
    )
    worlds = torch.cat((safe, colliding), dim=0)
    target_safe = safe[0]
    target_colliding = colliding[0]
    first = scene_collision_statistics(worlds, target_safe)
    reordered = scene_collision_statistics(worlds.flip(0), target_safe)
    changed_gt = scene_collision_statistics(worlds, target_colliding)
    assert first.cr_mean == reordered.cr_mean == changed_gt.cr_mean
    assert first.cr_jade == 0.0
    assert changed_gt.cr_jade == 1.0


def test_ground_truth_collision_rate() -> None:
    target = torch.tensor([
        [[-1.0, 0.0], [1.0, 0.0]],
        [[1.0, 0.0], [-1.0, 0.0]],
    ])
    rate, numerator, denominator = ground_truth_collision_rate(target)
    assert (rate, numerator, denominator) == (1.0, 2, 2)


def test_released_jmm_numpy_semantics_match_on_fixed_input() -> None:
    # Direct transliteration of official commit 894182b5 `_lineseg_dist` and
    # `_check_collision_in_scene_fast`, kept tiny to make parity auditable.
    rng = np.random.default_rng(3101)
    sample = rng.normal(size=(20, 7, 12, 2)).astype(np.float32)
    ours = collision_flags(torch.from_numpy(sample)).numpy()
    expected = np.zeros((20, 7), dtype=bool)
    for k in range(20):
        for i in range(7):
            for j in range(i + 1, 7):
                rel = sample[k, i] - sample[k, j]
                hit = np.linalg.norm(rel[0]) < 0.2
                for start, end in zip(rel[:-1], rel[1:]):
                    delta = end - start
                    denom = float(delta @ delta)
                    fraction = 0.0 if denom == 0 else np.clip(
                        -float(start @ delta) / denom, 0.0, 1.0)
                    hit |= np.linalg.norm(start + fraction * delta) < 0.2
                if hit:
                    expected[k, i] = expected[k, j] = True
    np.testing.assert_array_equal(ours, expected)

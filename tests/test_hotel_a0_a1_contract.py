from types import SimpleNamespace

import numpy as np
import torch

from src.trainer import trainer
from src.models.joint_dependency_v2.joint_sampler import (
    ParallelConditionalSampler,
)


class _Loader:
    def __init__(self, values=()):
        self.values = list(values)
        self.generator = torch.Generator().manual_seed(17)

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)


def _owner():
    item = trainer.__new__(trainer)
    item.args = SimpleNamespace(
        jdv2_step_cap=0, jdv2_arm_time_limit_seconds=0.0,
        jdv2_queue_deadline_utc=None, training_stage='joint_goal')
    item.net = SimpleNamespace(jdv2_active=True, best_valid_metric=lambda: 'JADE')
    item.data_loaders = {'train': _Loader([1, 2, 3])}
    item._optimizer_attempts = 0
    item._arm_started_monotonic = 0.0
    return item


def test_stage_a_selection_uses_jfde_then_earlier_epoch():
    item = _owner()
    assert item._selection_tuple({
        'valid_JADE': 0.2, 'valid_JFDE': 0.3}) == (0.2, 0.3)
    assert item._selection_is_better((0.2, 0.29), (0.2, 0.3))
    assert not item._selection_is_better((0.2, 0.3), (0.2, 0.3))


def test_step_cap_stops_before_fetching_next_window():
    item = _owner()
    item.args.jdv2_step_cap = 2
    iterator = item._budgeted_train_batches(item.data_loaders['train'])
    assert next(iterator) == 1
    item._optimizer_attempts = 1
    assert next(iterator) == 2
    item._optimizer_attempts = 2
    assert list(iterator) == []
    assert item._step_cap_reached


def test_runtime_rng_roundtrip_includes_loader_generator():
    item = _owner()
    torch.manual_seed(3)
    np.random.seed(5)
    state = item._runtime_rng_state()
    expected = (
        torch.rand(4), np.random.rand(4),
        torch.rand(4, generator=item.data_loaders['train'].generator))
    item._restore_runtime_rng_state(state)
    actual = (
        torch.rand(4), np.random.rand(4),
        torch.rand(4, generator=item.data_loaders['train'].generator))
    assert torch.equal(expected[0], actual[0])
    assert np.array_equal(expected[1], actual[1])
    assert torch.equal(expected[2], actual[2])


def test_explicit_test_seeds_are_distinct_parser_contract():
    from src.parser import get_parser, check_and_add_additional_args

    args = get_parser().parse_args([
        '--dataset', 'eth5', '--test_set', 'hotel',
        '--goal_model_type', 'joint_dependency_v2',
        '--training_stage', 'joint_goal', '--use_scene_latent', 'false',
        '--jdv2_latent_objective', 'strict_no_z',
        '--jdv2_evaluation_seeds', '2035', '2036', '2037', '2038', '2039',
    ])
    resolved = check_and_add_additional_args(args)
    assert resolved.jdv2_evaluation_seeds == [2035, 2036, 2037, 2038, 2039]


def test_pure_interaction_off_preserves_only_additive_gauge():
    row = torch.tensor([[[1.0], [3.0], [5.0]]])
    column = torch.tensor([[[2.0, 7.0, 11.0]]])
    additive = row + column
    selected = torch.tensor([[2, 0]])
    gathered = ParallelConditionalSampler._gather_additive_pair_cost(
        additive, selected)
    assert torch.allclose(gathered[0, 0], additive[0, 2])
    assert torch.allclose(gathered[0, 1], additive[0, 0])
    interaction = torch.tensor([[[0.0, 1.0], [2.0, 0.0]]])
    projected = ParallelConditionalSampler._gather_additive_pair_cost(
        interaction, torch.tensor([[0, 1]]))
    assert projected.shape == interaction.shape


def test_physical_cost_is_zero_when_paths_stay_far_and_one_on_collision():
    goals = torch.tensor([
        [[1.0, 0.0], [2.0, 0.0]],
        [[1.0, 2.0], [2.0, 2.0]],
    ])
    last = torch.tensor([[0.0, 0.0], [0.0, 2.0]])
    edges = torch.tensor([[0], [1]])
    selected = torch.tensor([[0, 1], [0, 1]])
    far = ParallelConditionalSampler._physical_selected_cost(
        goals, last, edges, selected, 'source')
    assert torch.count_nonzero(far).item() == 0
    goals[1, :, 1] = 0.0
    last[1, 1] = 0.0
    collision = ParallelConditionalSampler._physical_selected_cost(
        goals, last, edges, selected, 'source')
    assert torch.allclose(collision, torch.ones_like(collision))

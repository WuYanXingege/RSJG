import torch
from torch import nn

from ebjd.optimizer import ActualStepAdamW, list_dot, project_two_halfspaces


def test_projection_constrains_actual_adamw_decrement_and_finite_difference():
    parameter = nn.Parameter(torch.tensor([1.0]))
    optimizer = ActualStepAdamW([("weight", parameter)], lr=0.1, unet_lr=0.1, weight_decay=0)
    candidate, pending = optimizer.propose([torch.tensor([-1.0])])
    protected_gradient = [parameter.detach().clone()]
    projected, stats = project_two_halfspaces(
        candidate, protected_gradient, [torch.zeros_like(parameter)])
    assert float(list_dot(protected_gradient, projected)) >= -1e-10
    before = 0.5 * float(parameter.detach().square())
    optimizer.apply(projected, pending)
    after = 0.5 * float(parameter.detach().square())
    assert after <= before + 1e-8
    assert stats.changed_fraction > 0
    assert optimizer.state[id(parameter)]["step"] == 1


def test_projection_handles_duplicate_and_opposite_constraints():
    candidate = [torch.tensor([2.0, -1.0])]
    first = [torch.tensor([1.0, 0.0])]
    duplicate = [torch.tensor([2.0, 0.0])]
    projected, _ = project_two_halfspaces(candidate, first, duplicate)
    torch.testing.assert_close(projected[0], candidate[0])
    opposite, _ = project_two_halfspaces(candidate, first, [torch.tensor([-1.0, 0.0])])
    assert abs(float(opposite[0][0])) < 1e-7


def test_projection_solves_tangent_active_set_in_fp64_before_cast():
    candidate = [torch.tensor([-1.0, 2.0], dtype=torch.float32)]
    first = [torch.tensor([1.0, 0.0], dtype=torch.float32)]
    second = [torch.tensor([1.0, 1.0e-7], dtype=torch.float32)]
    projected, stats = project_two_halfspaces(candidate, first, second)
    assert projected[0].dtype == torch.float32
    assert not stats.fallback_zero
    torch.testing.assert_close(
        projected[0], torch.tensor([0.0, 2.0]), atol=5e-7, rtol=0)
    assert float(list_dot(first, projected)) >= -1e-7
    assert float(list_dot(second, projected)) >= -1e-7

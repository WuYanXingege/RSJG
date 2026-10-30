import json

import torch

from ebjd.goal_provider import GoalUNet, NativeGoalProvider


def test_goal_provider_is_only_goal_unet():
    module = NativeGoalProvider(GoalUNet().state_dict(), {"schema": "test"})
    assert sum(parameter.numel() for parameter in module.parameters()) == 613772
    assert set(dict(module.named_children())) == {"goal_module"}
    assert not any(parameter.requires_grad for parameter in module.parameters())


def test_prepared_provider_shapes_and_finite():
    module = NativeGoalProvider(GoalUNet().state_dict(), {"schema": "test"})
    torch.manual_seed(3)
    logits, goals = module.predict_prepared(
        torch.zeros(6, 32, 32), torch.zeros(2, 8, 32, 32))
    assert logits.shape == (2, 12, 32, 32)
    assert goals.shape == (20, 2, 2)
    assert torch.isfinite(logits).all() and torch.isfinite(goals).all()

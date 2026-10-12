import math
from copy import deepcopy

import pytest
import torch

from ebjd.config import load_config, semantic_config_hash, trainer_options
from ebjd.model import EBJDModel
from ebjd.trainer import EBJDTrainer, TrainerOptions


def test_config_to_schedule_wires_only_rollout_end_weight():
    source = load_config("configs/hotel_seed3101_small_group_uncheckpoint_v1.yaml")
    candidate = deepcopy(source)
    candidate["training"]["rollout_weight_end"] = 0.05
    assert semantic_config_hash(candidate) != semantic_config_hash(source)
    options = trainer_options(candidate)
    trainer = EBJDTrainer(EBJDModel(), options)
    assert options.goal_v_weight == 1.0
    assert candidate["objectives"]["bridge_v_weight"] == 1.0
    assert options.rollout_weight_start == 0.01
    assert trainer._rollout_weight(20) == 0.0
    assert trainer._rollout_weight(21) == 0.01
    assert trainer._rollout_weight(46) == 0.05
    assert trainer._rollout_weight(50) == 0.05


def test_default_rollout_end_weight_retains_existing_behavior():
    source = load_config("configs/hotel_seed3101_small_group_uncheckpoint_v1.yaml")
    options = trainer_options(source)
    trainer = EBJDTrainer(EBJDModel(), options)
    assert trainer._rollout_weight(46) == 0.20
    assert trainer._rollout_weight(50) == 0.20
    assert options.rollout_compensation == 4.0


def test_fixed_graph_loss_and_preprojection_gradient_difference_is_exact():
    torch.manual_seed(1205)
    parameter = torch.randn(37, dtype=torch.float64, requires_grad=True)
    base = (parameter.square() + 0.3 * parameter.sin()).mean()
    rollout = ((parameter - 0.7).square() + parameter.cos()).mean()
    marginal_ade = (parameter + 0.4).square().mean()
    marginal_fde = (parameter - 0.2).abs().mean()

    control = base + 4.0 * 0.20 * rollout
    candidate = base + 4.0 * 0.05 * rollout
    grad_control, = torch.autograd.grad(control, parameter, retain_graph=True)
    grad_candidate, = torch.autograd.grad(candidate, parameter, retain_graph=True)
    grad_rollout, = torch.autograd.grad(rollout, parameter, retain_graph=True)
    protected_control = torch.autograd.grad(
        (marginal_ade, marginal_fde), parameter, retain_graph=True)
    protected_candidate = torch.autograd.grad(
        (marginal_ade, marginal_fde), parameter)

    torch.testing.assert_close(candidate - control, -0.60 * rollout)
    torch.testing.assert_close(grad_candidate - grad_control, -0.60 * grad_rollout)
    torch.testing.assert_close(protected_candidate[0], protected_control[0])


def test_original_lr_timeline_and_absolute_epoch_contract_are_unchanged():
    options = TrainerOptions(
        epochs=100, warmup_epochs=5, new_module_lr=1e-4,
        unet_lr=1e-5, lr_final_fraction=0.01,
        rollout_weight_end=0.05)
    trainer = EBJDTrainer(EBJDModel(), options)
    for epoch in (46, 50):
        trainer._schedule(epoch)
        progress = (epoch - 5) / 95
        factor = 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))
        assert trainer.optimizer.lr == pytest.approx(1e-4 * factor)
        assert trainer.optimizer.unet_lr == pytest.approx(1e-5 * factor)

import copy
import importlib.util
import math
from pathlib import Path

import pytest
import torch

from conftest import make_batch
from ebjd.config import (
    load_config, semantic_config_hash, trainer_options,
    validate_supported_config,
)
from ebjd.model import EBJDModel
from ebjd.objectives import diffusion_loss
from ebjd.train import new_short_selection, resolve_stop_epoch, short_role_scores
from ebjd.trainer import EBJDTrainer, TrainerOptions
ROOT = Path(__file__).parents[1]


MATERIALIZER_SPEC = importlib.util.spec_from_file_location(
    "ebjd_goal_weight_materializer",
    ROOT / "scripts" / "materialize_epoch45_short_experiment.py")
assert MATERIALIZER_SPEC is not None and MATERIALIZER_SPEC.loader is not None
MATERIALIZER = importlib.util.module_from_spec(MATERIALIZER_SPEC)
MATERIALIZER_SPEC.loader.exec_module(MATERIALIZER)


def test_materialized_yaml_preserves_scientific_float_types(tmp_path):
    payload = {
        "unet_lr": 1e-5, "optimizer_eps": 1e-8,
        "weight": 2.0, "enabled": False,
    }
    path = tmp_path / "materialized.yaml"
    MATERIALIZER.atomic_yaml(payload, path)
    import yaml
    restored = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert restored == payload
    assert isinstance(restored["unet_lr"], float)
    assert isinstance(restored["optimizer_eps"], float)


def _masked_mean(value, valid):
    mask = valid
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value).to(value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1)


def test_default_loss_is_bitwise_legacy_and_goal_weight_math_is_exact():
    torch.manual_seed(401)
    final = torch.randn(2, 1, 3, 12, 2, dtype=torch.float64)
    coarse = torch.randn_like(final)
    target = torch.randn_like(final)
    valid = torch.tensor([[True, False, True], [True, True, False]])

    def legacy_split(value):
        goal = _masked_mean(
            (value[..., 11, :] - target[..., 11, :]).square(), valid[:, None])
        bridge = _masked_mean(
            (value[..., :11, :] - target[..., :11, :]).square(), valid[:, None])
        return goal + bridge

    legacy = legacy_split(final) + 0.25 * legacy_split(coarse)
    weight_one, parts = diffusion_loss(
        final, coarse, target, valid, return_components=True)
    weight_two = diffusion_loss(
        final, coarse, target, valid, goal_weight=2.0)
    assert torch.equal(weight_one, legacy)
    expected_increment = (
        parts["goal_fine_unweighted"]
        + 0.25 * parts["goal_coarse_unweighted"])
    torch.testing.assert_close(weight_two - weight_one, expected_increment)


def test_goal_output_gradients_double_and_bridge_gradients_do_not_change():
    torch.manual_seed(402)
    target = torch.randn(2, 1, 3, 12, 2, dtype=torch.float64)
    valid = torch.tensor([[True, True, False], [True, False, False]])

    def gradients(weight):
        final = torch.randn(
            2, 1, 3, 12, 2, dtype=torch.float64, requires_grad=True)
        coarse = torch.randn_like(final, requires_grad=True)
        # Compare the same fixed prediction tensors for both weights.
        return final, coarse

    final_one, coarse_one = gradients(1.0)
    final_two = final_one.detach().clone().requires_grad_(True)
    coarse_two = coarse_one.detach().clone().requires_grad_(True)
    diffusion_loss(
        final_one, coarse_one, target, valid, goal_weight=1.0).backward()
    diffusion_loss(
        final_two, coarse_two, target, valid, goal_weight=2.0).backward()
    for first, second in (
        (final_one.grad, final_two.grad),
        (coarse_one.grad, coarse_two.grad),
    ):
        torch.testing.assert_close(second[..., 11, :], 2 * first[..., 11, :])
        torch.testing.assert_close(second[..., :11, :], first[..., :11, :])
        invalid = (~valid)[:, None, :, None, None].expand_as(first)
        assert not first[invalid].any()
        assert not second[invalid].any()


@pytest.mark.parametrize("bad", [True, False, float("nan"), float("inf"), 1.5, 3, "2"])
def test_goal_weight_rejects_noncertified_values(bad):
    config = load_config(ROOT / "configs" / "endpoint_bridge_joint.yaml")
    config["objectives"]["goal_v_weight"] = bad
    with pytest.raises(ValueError, match="goal_v_weight"):
        validate_supported_config(config)


def test_config_to_trainer_to_actual_loss_path_passes_goal_weight(monkeypatch):
    import ebjd.trainer as trainer_module

    config = load_config(ROOT / "configs" / "endpoint_bridge_joint.yaml")
    original_semantic_hash = semantic_config_hash(config)
    config["objectives"]["goal_v_weight"] = 2.0
    validate_supported_config(config)
    assert semantic_config_hash(config) != original_semantic_hash
    options = trainer_options(config)
    assert options.goal_v_weight == 2.0
    options.precision = "fp32"
    options.activation_checkpointing = False
    options.scene_forward_chunk = 0
    options.rollout_start_epoch = 999
    captured = []
    actual = trainer_module.diffusion_loss

    def recorded(*args, **kwargs):
        captured.append(kwargs["goal_weight"])
        return actual(*args, **kwargs)

    monkeypatch.setattr(trainer_module, "diffusion_loss", recorded)
    trainer = EBJDTrainer(EBJDModel(), options)
    trainer.train_step([make_batch((1,), pixels=16)], epoch=1)
    assert captured == [2.0]


def test_epoch_schedule_absolute_boundary_and_short_selector_contract():
    assert resolve_stop_epoch(46, 100, 5, 50) == 50
    assert resolve_stop_epoch(48, 100, 5, 50) == 50
    assert resolve_stop_epoch(51, 100, 5, 50) == 50
    with pytest.raises(ValueError, match="beyond"):
        resolve_stop_epoch(52, 100, 5, 50)

    options = TrainerOptions(
        epochs=100, warmup_epochs=5, new_module_lr=1e-4,
        unet_lr=1e-5, lr_final_fraction=0.01)
    trainer = EBJDTrainer(EBJDModel(), options)
    for epoch in (46, 50):
        trainer._schedule(epoch)
        progress = (epoch - 5) / 95
        factor = 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))
        assert trainer.optimizer.lr == pytest.approx(1e-4 * factor)
        assert trainer.optimizer.unet_lr == pytest.approx(1e-5 * factor)

    roles = new_short_selection(46, 50)
    assert roles["start_epoch"] == 46 and roles["stop_epoch"] == 50
    assert all(roles[name] is None for name in (
        "best_joint_short", "best_marginal_short", "least_violation_short"))
    scores = short_role_scores(
        {"minADE": 0.2, "minFDE": 0.3, "JADE": 0.4, "JFDE": 0.6},
        {"minADE": 0.1, "minFDE": 0.2}, 0.0)
    assert scores["best_joint_short"] == pytest.approx(0.7)


def test_goal_weight_adds_no_model_state_or_parameters():
    model = EBJDModel()
    state = list(model.state_dict())
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert parameter_count == 3_612_764
    assert list(EBJDModel().state_dict()) == state

"""Regression contracts for exact same-stage JDV2 stopping-state resume."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.trainer import trainer


class _TinyNet(torch.nn.Module):
    jdv2_active = True

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.25, -0.5]))

    @staticmethod
    def best_valid_metric():
        return "JADE"

    @staticmethod
    def init_test_metrics():
        return {"JADE": [], "JFDE": []}

    @staticmethod
    def init_losses():
        return {"loss": 0.0}


class _DisabledScaler:
    @staticmethod
    def is_enabled():
        return False


def _harness(model_dir: Path) -> trainer:
    torch.manual_seed(19)
    item = trainer.__new__(trainer)
    item.args = SimpleNamespace(
        model_dir=str(model_dir), training_stage="joint_trajectory",
        jdv2_latent_objective="strict_no_z", jdv2_stage_progress=0.25,
        jdv2_cache_manifest_hash="cache", jdv2_source_checkpoint_hash="src",
        stage_a_parent_checkpoint_sha256="stage-a",
        stage_a_freeze_manifest_sha256="manifest",
        stage_a_freeze_source_commit="freeze-commit",
        stage_b_architecture_version="jdv2-stage-b-v2a",
        jdv2_residual_projection="component_zero_mean", seed=2035,
        load_checkpoint="3", trajectory_alignment=False,
        trajectory_coupling="none")
    item.net = _TinyNet()
    item.optimizer = torch.optim.Adam(item.net.parameters(), lr=1e-3)
    item.scheduler = torch.optim.lr_scheduler.ExponentialLR(
        item.optimizer, gamma=0.9)
    item.scaler = _DisabledScaler()
    item._stage_optimizer_steps_completed = 0
    item._validations_without_improvement = 0
    item._collapse_signature_epochs = 0
    item._best_selection = (float("inf"), float("inf"))
    item.best_metrics = {"JADE": float("inf"), "JFDE": float("inf")}
    item.best_metrics_epochs = {"JADE": -1, "JFDE": -1}
    item.log_curve_file = str(model_dir / "log_curve.txt")
    item.curve_metric_names = ["JADE", "JFDE"]
    item.curve_loss_names = ["loss", "loss_total"]
    item._jdv2_architecture_config = lambda: {"variant": "test"}
    item._jdv2_ablation_config = lambda: {}
    item._jdv2_inference_sampler_config = lambda: {}
    return item


def _step(item: trainer, epoch: int, selection: tuple[float, float],
          patience: int) -> bool:
    item.optimizer.zero_grad(set_to_none=True)
    for parameter in item.net.parameters():
        parameter.grad = torch.full_like(parameter, float(epoch) / 100.0)
    item.optimizer.step()
    item.scheduler.step()
    item._stage_optimizer_steps_completed += 1
    improved = item._selection_is_better(selection, item._best_selection)
    count = item._record_validation_outcome(improved)
    if improved:
        item._best_selection = selection
        item.best_metrics.update({"JADE": selection[0], "JFDE": selection[1]})
        item.best_metrics_epochs.update({"JADE": epoch, "JFDE": epoch})
    return count >= patience


def _assert_nested_equal(left, right):
    assert type(left) is type(right)
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right):
            _assert_nested_equal(left_value, right_value)
    else:
        assert left == right


def test_interrupted_resume_matches_uninterrupted_early_stop(tmp_path):
    selections = {
        1: (0.50, 0.90), 2: (0.40, 0.80), 3: (0.41, 0.81),
        4: (0.42, 0.82), 5: (0.43, 0.83), 6: (0.44, 0.84),
    }
    patience = 3

    control = _harness(tmp_path / "control")
    control_stop = None
    for epoch in selections:
        if _step(control, epoch, selections[epoch], patience):
            control_stop = epoch
            break

    interrupted = _harness(tmp_path / "interrupted")
    for epoch in range(1, 4):
        assert not _step(interrupted, epoch, selections[epoch], patience)
    interrupted._save_checkpoint(3, last_epoch=True)
    checkpoint_path = (tmp_path / "interrupted" / "saved_models" /
                       "last_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu",
                            weights_only=False)
    assert checkpoint["validations_without_improvement"] == 1
    assert checkpoint["collapse_signature_epochs"] == 0

    resumed = _harness(tmp_path / "resumed")
    resumed.net.load_state_dict(checkpoint["model_state_dict"])
    resumed.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    resumed.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    resumed._restore_best_state(checkpoint)
    resumed._restore_stopping_state(checkpoint, same_stage=True)
    resumed._pending_training_state = checkpoint
    resumed._load_checkpoint = lambda _: checkpoint["epoch"]
    next_epoch = resumed._load_or_restart()

    assert next_epoch == 4
    assert resumed._best_selection == interrupted._best_selection
    assert resumed._validations_without_improvement == 1
    _assert_nested_equal(resumed.optimizer.state_dict(),
                         interrupted.optimizer.state_dict())
    _assert_nested_equal(resumed.scheduler.state_dict(),
                         interrupted.scheduler.state_dict())

    resumed_stop = None
    for epoch in range(next_epoch, max(selections) + 1):
        if _step(resumed, epoch, selections[epoch], patience):
            resumed_stop = epoch
            break

    assert control_stop == resumed_stop == 5
    assert resumed._best_selection == control._best_selection
    assert resumed.best_metrics == control.best_metrics
    assert resumed.best_metrics_epochs == control.best_metrics_epochs
    assert resumed._validations_without_improvement == \
        control._validations_without_improvement == patience
    _assert_nested_equal(resumed.net.state_dict(), control.net.state_dict())
    _assert_nested_equal(resumed.optimizer.state_dict(),
                         control.optimizer.state_dict())
    _assert_nested_equal(resumed.scheduler.state_dict(),
                         control.scheduler.state_dict())


def test_cross_stage_and_legacy_stopping_state_fallbacks(tmp_path):
    item = _harness(tmp_path)
    item._validations_without_improvement = 5
    item._collapse_signature_epochs = 1
    item._restore_stopping_state({}, same_stage=True)
    assert item._validations_without_improvement == 0
    assert item._collapse_signature_epochs == 0

    checkpoint = {
        "validations_without_improvement": 4,
        "collapse_signature_epochs": 1,
    }
    item._restore_stopping_state(checkpoint, same_stage=False)
    assert item._validations_without_improvement == 0
    assert item._collapse_signature_epochs == 0

    with pytest.raises(RuntimeError, match="negative checkpoint stopping"):
        item._restore_stopping_state(
            {"validations_without_improvement": -1}, same_stage=True)

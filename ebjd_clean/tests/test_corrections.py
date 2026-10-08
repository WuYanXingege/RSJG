import copy
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from ebjd.config import build_model, load_config, trainer_options, validate_supported_config
from ebjd.model import EBJDModel
from ebjd.objectives import gated_geometry_mean
from ebjd.representation import EndpointBridgeRepresentation, cv_baseline
from ebjd.selection import ConstrainedSelector
from ebjd.trainer import EBJDTrainer


def metrics(minade, minfde, jade, jfde):
    return {"minADE": minade, "minFDE": minfde, "JADE": jade, "JFDE": jfde}


def test_per_scene_alpha_squared_geometry_gate():
    loss = torch.tensor([10.0, 1.0])
    alpha = torch.tensor([0.0, 1.0]).view(2, 1, 1, 1, 1)
    assert gated_geometry_mean(loss, alpha) == 0.5


def test_selector_rejects_joint_only_gain_and_resume_preserves_best():
    selector = ConstrainedSelector(0.20, 0.30)
    accepted = selector.consider(1, metrics(0.19, 0.29, 0.40, 0.50))
    rejected = selector.consider(2, metrics(0.25, 0.36, 0.30, 0.40))
    assert selector.best_eligible is accepted
    assert rejected["simultaneous_improvement"] is False
    assert len(selector.state_dict()["pareto_candidates"]) == 2

    resumed = ConstrainedSelector.from_state_dict(selector.state_dict())
    resumed.consider(3, metrics(0.20, 0.30, 0.60, 0.60))
    assert resumed.best_eligible["epoch"] == 1
    assert resumed.state_dict()["simultaneous_improvement"] is True


def test_config_values_are_wired_or_rejected():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs" / "endpoint_bridge_joint.yaml")
    changed = copy.deepcopy(config)
    changed["training"]["new_module_lr"] = 7e-5
    changed["objectives"]["relative_motion_weight"] = 0.17
    options = trainer_options(changed)
    assert options.new_module_lr == 7e-5
    assert options.geometry_weight == 0.17

    unsupported = copy.deepcopy(config)
    unsupported["history_encoder"]["hidden_dim"] = 64
    with pytest.raises(ValueError, match="hidden_dim"):
        validate_supported_config(unsupported)
    unknown = copy.deepcopy(config)
    unknown["training"]["silent_option"] = True
    with pytest.raises(ValueError, match="silent_option"):
        validate_supported_config(unknown)


def test_future_time_is_fixed_and_parameter_count_matches_design():
    model = EBJDModel()
    assert "denoiser.future_time" not in dict(model.named_parameters())
    assert "denoiser.future_time" in dict(model.named_buffers())
    assert sum(parameter.numel() for parameter in model.parameters()) == 3_612_764


def test_bridge_decode_stays_fp32_inside_outer_autocast():
    torch.manual_seed(17)
    observed = torch.randn(1, 2, 8, 2)
    baseline, _ = cv_baseline(observed)
    latent = torch.randn(1, 3, 2, 12, 2)
    representation = EndpointBridgeRepresentation(1.3, 0.7)
    reference, reference_goal = representation.decode(latent, baseline)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual, actual_goal = representation.decode(latent, baseline)
    assert actual.dtype == torch.float32 and actual_goal.dtype == torch.float32
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    torch.testing.assert_close(actual_goal, reference_goal, atol=0, rtol=0)


def test_checkpoint_restores_rng_identity_and_selection(tmp_path):
    identity = {"fold": "hotel", "seed": 3101, "ablation": "none", "run_id": "unit"}
    model = EBJDModel()
    trainer = EBJDTrainer(model, resolved_config={"unit": True}, run_identity=identity)
    trainer.update_index = 9
    selector = ConstrainedSelector(0.2, 0.3)
    selector.consider(4, metrics(0.19, 0.29, 0.4, 0.5))
    generator = torch.Generator().manual_seed(91)
    random.seed(81)
    np.random.seed(82)
    torch.manual_seed(83)
    checkpoint = tmp_path / "last.pt"
    trainer.save_checkpoint(checkpoint, 4, selector.state_dict(), generator)
    expected = (
        random.random(), float(np.random.rand()), float(torch.rand(())),
        float(torch.rand((), generator=generator)),
    )

    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    generator.manual_seed(4)
    restored = EBJDTrainer(
        EBJDModel(), resolved_config={"unit": True}, run_identity=identity)
    payload = restored.load_checkpoint(checkpoint, generator)
    actual = (
        random.random(), float(np.random.rand()), float(torch.rand(())),
        float(torch.rand((), generator=generator)),
    )
    assert actual == expected
    assert restored.update_index == 9
    assert payload["selection_state"]["best_eligible"]["epoch"] == 4
    assert payload["resolved_config"] == {"unit": True}


def test_all_ablations_keep_distinct_identity_values():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs" / "endpoint_bridge_joint.yaml")
    config["paths"]["accepted_fold_matched_gdts_unet"] = None
    identities = set()
    for ablation in (
        "none", "future_social_off", "noisy_geometry", "cartesian_velocity",
        "geometry_loss_off", "rollout_loss_off", "actual_step_constraint_off",
    ):
        build_model(config, ablation)
        identities.add((config["experiment"]["fold"], config["seed"], ablation, "unit"))
    assert len(identities) == 7

from pathlib import Path

import torch

from ebjd.config import build_model, load_config
from ebjd.representation import CartesianVelocityRepresentation


def test_all_registered_ablations_build_and_source_loads():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs" / "endpoint_bridge_joint.yaml")
    if Path(config["paths"]["accepted_fold_matched_gdts_unet"]).exists():
        source_model = build_model(config)
        assert torch.isfinite(source_model.encoder.map_unet.encoders[0].net[0].weight).all()
    config["paths"]["accepted_fold_matched_gdts_unet"] = None
    for ablation in (
        "future_social_off", "noisy_geometry", "cartesian_velocity",
        "geometry_loss_off", "rollout_loss_off", "actual_step_constraint_off",
    ):
        model = build_model(config, ablation)
        if ablation == "cartesian_velocity":
            assert isinstance(model.representation, CartesianVelocityRepresentation)

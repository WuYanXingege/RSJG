"""Strict YAML loading and model/ablation construction."""

from __future__ import annotations

from pathlib import Path
import hashlib

import yaml
import torch

from .model import EBJDModel
from .trainer import TrainerOptions


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")
    return payload


def build_model(config: dict, ablation: str = "none") -> EBJDModel:
    allowed = {
        "none", "future_social_off", "noisy_geometry", "cartesian_velocity",
        "geometry_loss_off", "rollout_loss_off", "actual_step_constraint_off",
    }
    if ablation not in allowed:
        raise ValueError(f"unsupported ablation {ablation!r}; expected {sorted(allowed)}")
    rep = config.get("representation", {})
    map_config = config.get("map_encoder", {})
    model = EBJDModel(
        goal_scale=float(rep.get("goal_scale_m", 1.0)),
        bridge_scale=float(rep.get("bridge_scale_m", 1.0)),
        crop_width_m=float(map_config.get("crop_width_m", 32.0)),
        representation="cartesian_velocity" if ablation == "cartesian_velocity" else "endpoint_bridge",
        future_social=ablation != "future_social_off",
        clean_geometry=ablation != "noisy_geometry",
    )
    source = config.get("paths", {}).get("accepted_fold_matched_gdts_unet")
    if source:
        source_path = Path(source)
        expected = config.get("paths", {}).get("accepted_fold_matched_gdts_unet_sha256")
        if expected:
            actual = hashlib.sha256(source_path.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"U-Net checkpoint SHA256 mismatch: {actual}")
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict", payload)
        translated = {}
        for index in range(5):
            for convolution in (0, 2):
                old = f"goal_module.encoder.enc_blocks.{index}.block.{convolution}"
                new = f"encoders.{index}.net.{convolution}"
                translated[new + ".weight"] = state[old + ".weight"]
                translated[new + ".bias"] = state[old + ".bias"]
        for index in range(4):
            old_up = f"goal_module.decoder.upconvs.{index}.up.1"
            new_up = f"up_blocks.{index}.up"
            translated[new_up + ".weight"] = state[old_up + ".weight"]
            translated[new_up + ".bias"] = state[old_up + ".bias"]
            for convolution in (0, 2):
                old = f"goal_module.decoder.dec_blocks.{index}.block.{convolution}"
                new = f"up_blocks.{index}.conv.net.{convolution}"
                translated[new + ".weight"] = state[old + ".weight"]
                translated[new + ".bias"] = state[old + ".bias"]
        translated["future_logits.weight"] = state["goal_module.head.out_layer.weight"]
        translated["future_logits.bias"] = state["goal_module.head.out_layer.bias"]
        model.encoder.map_unet.load_state_dict(translated, strict=True)
    return model


def trainer_options(config: dict, ablation: str = "none") -> TrainerOptions:
    training = config.get("training", {})
    return TrainerOptions(
        epochs=int(training.get("epochs", 100)),
        accumulation=int(training.get("gradient_accumulation", 4)),
        rollout_start_epoch=int(training.get("rollout_start_epoch", 21)),
        rollout_every=int(training.get("rollout_every_optimizer_updates", 4)),
        rollout_worlds=int(training.get("rollout_worlds", 4)),
        rollout_steps=int(training.get("rollout_steps", 20)),
        use_rollout_loss=ablation != "rollout_loss_off",
        use_geometry_loss=ablation != "geometry_loss_off",
        use_actual_step_constraint=ablation != "actual_step_constraint_off",
        activation_checkpointing=bool(training.get("activation_checkpointing", True)),
    )

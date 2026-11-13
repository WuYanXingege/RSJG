"""Strict YAML loading, validation and model/trainer construction."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import torch
import yaml

from .model import EBJDModel


ABLATIONS = {"none"}

ALLOWED_KEYS = {
    "experiment": {"fold", "source_commit", "implementation_base"},
    "paths": {
        "train_manifest", "validation_manifest", "test_manifest", "output_root",
        "accepted_fold_matched_gdts_unet",
        "accepted_fold_matched_gdts_unet_sha256", "initialization_role",
        "matched_gdts_validation_checkpoint",
        "matched_gdts_validation_checkpoint_sha256",
        "matched_gdts_validation_result",
        "matched_gdts_validation_result_sha256",
    },
    "data": {
        "obs_length", "pred_length", "dt_seconds", "coordinates",
        "complete_horizon_required", "scene_agent_padding",
        "do_not_truncate_scene_agents",
    },
    "map_encoder": {
        "semantic_channels", "history_heatmap_channels", "crop_width_m",
        "crop_pixels", "enc_channels", "dec_channels",
        "future_heatmap_channels", "heatmap_sigma_m", "memory_grid", "trainable",
        "agent_chunk",
    },
    "history_encoder": {
        "input_dim", "hidden_dim", "social_blocks", "social_heads",
        "social_ff_dim", "observed_geometry_dim", "observed_bias_bound", "dropout",
    },
    "representation": {
        "type", "goal_tokens", "bridge_tokens", "goal_scale_m", "bridge_scale_m",
        "train_fold_rms_scaling", "goal_rms_floor_m", "bridge_rms_floor_m",
        "literal_endpoint_concatenation",
    },
    "denoiser": {
        "hidden_dim", "heads", "ff_dim", "coarse_blocks", "refinement_blocks",
        "sublayers", "memory_tokens_per_agent", "cross_world_attention",
        "target_parameterization", "geometry_gate", "clean_future_detach",
        "ctf_feedback_basis",
        "block_type",
    },
    "diffusion": {
        "schedule", "exact_terminal_snr_zero", "training_noise_bins",
        "shared_time_within_scene", "sampler", "ddim_eta",
    },
    "objectives": {
        "goal_v_weight", "bridge_v_weight", "coarse_v_weight",
        "relative_motion_weight", "map_bce_weight", "relative_velocity_factor",
        "rollout_joint_ade_weight", "rollout_joint_fde_weight",
        "rollout_marginal_ade_weight", "rollout_marginal_fde_weight",
        "supervision_edge_radius_m", "supervision_edge_cv_cpa_radius_m",
        "supervision_edge_cv_horizon_s",
    },
    "training": {
        "epochs", "batch_scenes", "gradient_accumulation", "new_module_lr",
        "unet_lr", "lr_warmup_epochs", "lr_final_fraction", "optimizer",
        "optimizer_betas", "optimizer_eps", "weight_decay",
        "total_gradient_clip_norm", "exclude_bias_and_layernorm_from_decay",
        "gradient_constraint_uses_unclipped_marginal_gradients",
        "rollout_start_epoch", "rollout_ramp_end_epoch", "rollout_weight_start",
        "rollout_weight_end", "rollout_every_optimizer_updates",
        "rollout_frequency_loss_compensation", "rollout_worlds", "rollout_steps",
        "rollout_detach_between_steps", "softmin_temperature_start_m",
        "softmin_temperature_end_m", "protected_losses", "activation_checkpointing",
        "checkpoint_use_reentrant", "mixed_precision", "ema", "train_all_modules",
        "scene_forward_chunk", "execution_strategy", "scene_compaction",
        "physical_group_max_scenes", "physical_group_max_agent_slots",
        "physical_group_max_padding_ratio", "rollout_block_checkpointing",
        "rollout_uncheckpointed_max_agent_slots",
        "training_runtime_optimizations",
        "data_loader_workers", "data_loader_prefetch_factor",
        "rollout_training_enabled",
        "paired_marginal_actual_step_projection_enabled",
        "ctf_feedback_transition_start_update", "ctf_feedback_transition_updates",
        "early_stopping", "ebjd_input_pipeline",
        "ebjd_input_pipeline_receipt",
        "coupling_singleton_backward",
    },
    "evaluation": {
        "worlds", "steps", "selection_seed", "inference_seeds",
        "minimum_training_seeds", "metrics",
        "per_agent_reordering", "final_test_used_for_checkpoint_selection",
        "validation_test_identity", "marginal_tolerance", "matched_gdts_validation",
        "unit",
    },
    "ablations": {"available"},
}

# Added execution-only settings must remain optional so archived v2 configs and
# their historical config hashes continue to validate unchanged.
OPTIONAL_KEYS = {"training": {
    "scene_forward_chunk", "execution_strategy", "scene_compaction",
    "physical_group_max_scenes", "physical_group_max_agent_slots",
    "physical_group_max_padding_ratio", "rollout_block_checkpointing",
    "rollout_uncheckpointed_max_agent_slots",
    "training_runtime_optimizations",
    "data_loader_workers", "data_loader_prefetch_factor",
    "rollout_training_enabled",
    "paired_marginal_actual_step_projection_enabled",
    "ctf_feedback_transition_start_update", "ctf_feedback_transition_updates",
    "early_stopping", "ebjd_input_pipeline",
    "ebjd_input_pipeline_receipt",
    "coupling_singleton_backward",
}, "denoiser": {"ctf_feedback_basis", "block_type"}, "evaluation": {"unit"}}

SEMANTIC_CONFIG_EXCLUSIONS = {
    ("experiment", "source_commit"),
    ("paths", "output_root"),
    ("training", "scene_forward_chunk"),
    ("training", "execution_strategy"),
    ("training", "scene_compaction"),
    ("training", "physical_group_max_scenes"),
    ("training", "physical_group_max_agent_slots"),
    ("training", "physical_group_max_padding_ratio"),
    ("training", "rollout_block_checkpointing"),
    ("training", "rollout_uncheckpointed_max_agent_slots"),
    ("training", "training_runtime_optimizations"),
    ("training", "data_loader_workers"),
    ("training", "data_loader_prefetch_factor"),
    ("training", "ebjd_input_pipeline"),
    ("training", "ebjd_input_pipeline_receipt"),
}


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle) if Path(path).suffix.lower() == ".json" else yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")
    validate_supported_config(payload)
    return payload


def config_hash(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def semantic_config(config: dict) -> dict:
    """Remove explicitly execution-only fields for controlled checkpoint migration."""
    payload = deepcopy(config)
    for section, key in SEMANTIC_CONFIG_EXCLUSIONS:
        payload.get(section, {}).pop(key, None)
    return payload


def semantic_config_hash(config: dict) -> str:
    return config_hash(semantic_config(config))


def config_differences(left: dict, right: dict) -> dict[str, dict]:
    """Return leaf differences with a sentinel for missing optional fields."""
    missing = object()
    differences: dict[str, dict] = {}

    def walk(a, b, prefix: tuple[str, ...]) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                walk(a.get(key, missing), b.get(key, missing), (*prefix, key))
            return
        if a is missing or b is missing or a != b:
            differences[".".join(prefix)] = {
                "old": "<MISSING>" if a is missing else a,
                "new": "<MISSING>" if b is missing else b,
            }

    walk(left, right, ())
    return differences


def _require(config: dict, section: str, key: str, expected) -> None:
    actual = config.get(section, {}).get(key)
    if actual != expected:
        raise ValueError(
            f"unsupported {section}.{key}={actual!r}; EBJD-clean-v1 requires {expected!r}")


def validate_supported_config(config: dict) -> None:
    expected_root = {"seed", *ALLOWED_KEYS}
    unknown_root = set(config) - expected_root
    missing_root = expected_root - set(config)
    if unknown_root or missing_root:
        raise ValueError(
            f"configuration root fields differ: unknown={sorted(unknown_root)}, "
            f"missing={sorted(missing_root)}")
    for section, allowed in ALLOWED_KEYS.items():
        value = config.get(section)
        if not isinstance(value, dict):
            raise ValueError(f"configuration section {section} must be a mapping")
        optional = OPTIONAL_KEYS.get(section, set())
        unknown = set(value) - allowed
        missing = (allowed - optional) - set(value)
        if unknown or missing:
            raise ValueError(
                f"configuration fields in {section} differ: unknown={sorted(unknown)}, "
                f"missing={sorted(missing)}")
    for section, key, expected in (
        ("data", "obs_length", 8), ("data", "pred_length", 12),
        ("data", "dt_seconds", 0.4), ("data", "coordinates", "metres"),
        ("data", "complete_horizon_required", True),
        ("data", "scene_agent_padding", True),
        ("data", "do_not_truncate_scene_agents", True),
        ("map_encoder", "semantic_channels", 6),
        ("map_encoder", "history_heatmap_channels", 8),
        ("map_encoder", "enc_channels", [14, 32, 32, 64, 64, 64]),
        ("map_encoder", "dec_channels", [64, 64, 64, 32, 32]),
        ("map_encoder", "future_heatmap_channels", 12),
        ("map_encoder", "memory_grid", [8, 8]),
        ("map_encoder", "trainable", True),
        ("map_encoder", "crop_width_m", 32.0),
        ("history_encoder", "input_dim", 6),
        ("history_encoder", "social_blocks", 2),
        ("history_encoder", "social_heads", 4),
        ("history_encoder", "observed_geometry_dim", 10),
        ("history_encoder", "observed_bias_bound", 2.0),
        ("history_encoder", "dropout", 0.0),
        ("representation", "goal_tokens", 1),
        ("representation", "bridge_tokens", 11),
        ("representation", "train_fold_rms_scaling", True),
        ("representation", "goal_rms_floor_m", 0.5),
        ("representation", "bridge_rms_floor_m", 0.1),
        ("representation", "literal_endpoint_concatenation", True),
        ("denoiser", "heads", 4),
        ("denoiser", "memory_tokens_per_agent", 72),
        ("denoiser", "cross_world_attention", False),
        ("denoiser", "target_parameterization", "v"),
        ("denoiser", "geometry_gate", "alpha_squared"),
        ("denoiser", "clean_future_detach", False),
        ("denoiser", "sublayers", [
            "temporal_attention", "social_attention", "context_cross_attention", "ffn"]),
        ("diffusion", "schedule", "cosine_vp_alpha_cos_sigma_sin"),
        ("diffusion", "exact_terminal_snr_zero", True),
        ("diffusion", "shared_time_within_scene", True),
        ("diffusion", "sampler", "ddim"), ("diffusion", "ddim_eta", 0.0),
        ("objectives", "goal_v_weight", 1.0),
        ("objectives", "bridge_v_weight", 1.0),
        ("objectives", "relative_velocity_factor", 0.5),
        ("objectives", "rollout_joint_ade_weight", 1.0),
        ("objectives", "rollout_joint_fde_weight", 0.5),
        ("objectives", "rollout_marginal_ade_weight", 1.0),
        ("objectives", "rollout_marginal_fde_weight", 0.5),
        ("training", "optimizer", "adamw"),
        ("training", "exclude_bias_and_layernorm_from_decay", True),
        ("training", "gradient_constraint_uses_unclipped_marginal_gradients", True),
        ("training", "rollout_detach_between_steps", False),
        ("training", "checkpoint_use_reentrant", False),
        ("training", "protected_losses", ["soft_marginal_ade", "soft_marginal_fde"]),
        ("training", "ema", False), ("training", "train_all_modules", True),
        ("evaluation", "per_agent_reordering", False),
        ("evaluation", "final_test_used_for_checkpoint_selection", False),
        ("evaluation", "validation_test_identity",
         "mirrored_official_package_internal_only"),
        ("evaluation", "minimum_training_seeds", 3),
        ("evaluation", "metrics", ["minADE", "minFDE", "JADE", "JFDE"]),
        ("ablations", "available", []),
    ):
        _require(config, section, key, expected)
    denoiser = config["denoiser"]
    block_type = denoiser.get("block_type", "legacy_fjab")
    depths = (denoiser["coarse_blocks"], denoiser["refinement_blocks"])
    allowed_depths = ({(3, 3)} if block_type == "legacy_fjab" else {(3, 3), (4, 4)})
    if (block_type not in {"legacy_fjab", "endpoint_gated_dit_v1"} or
            any(type(value) is not int for value in depths) or depths not in allowed_depths):
        raise ValueError("unsupported denoiser block_type/coarse/refinement depth")
    if (block_type == "endpoint_gated_dit_v1" and
            denoiser.get("ctf_feedback_basis", "legacy") != "legacy"):
        raise ValueError("endpoint-gated DiT requires legacy CTF feedback")
    if config["representation"]["type"] not in {
        "invertible_endpoint_brownian_bridge",
        "invertible_endpoint_acceleration_bridge_v1",
    }:
        raise ValueError("unsupported representation.type")
    if config["representation"]["type"] == "invertible_endpoint_acceleration_bridge_v1":
        if config["denoiser"].get("ctf_feedback_basis", "legacy") != "legacy":
            raise ValueError("acceleration bridge requires legacy CTF feedback")
    width_profile = (
        int(config["history_encoder"]["hidden_dim"]),
        int(config["history_encoder"]["social_ff_dim"]),
        int(config["denoiser"]["hidden_dim"]),
        int(config["denoiser"]["ff_dim"]),
    )
    if width_profile not in {(128, 512, 128, 512), (192, 768, 192, 768)}:
        raise ValueError(
            "unsupported coherent hidden_dim/social_ff_dim width profile; "
            "expected legacy 128/512 or "
            f"wide 192/768, got {width_profile}")
    precision = config["training"].get("mixed_precision")
    if precision not in ("fp32", "bf16_with_fp32_geometry_loss_optimizer"):
        raise ValueError(f"unsupported training.mixed_precision={precision!r}")
    if int(config["map_encoder"].get("crop_pixels", 0)) != 256:
        raise ValueError("formal EBJD config requires 256x256 map crops")
    ctf_basis = config["denoiser"].get("ctf_feedback_basis", "legacy")
    if ctf_basis not in {"legacy", "decoder_energy"}:
        raise ValueError(
            "denoiser.ctf_feedback_basis must be legacy or decoder_energy")
    if "scene_forward_chunk" in config["training"]:
        chunk = config["training"]["scene_forward_chunk"]
        if isinstance(chunk, bool) or int(chunk) != chunk or int(chunk) != 1:
            raise ValueError(
                "training.scene_forward_chunk currently supports only the OOM-safe value 1")
    training = config["training"]
    if training.get("coupling_singleton_backward", "legacy") not in {
        "legacy", "identity",
    }:
        raise ValueError("unsupported training.coupling_singleton_backward")
    strategy = training.get("execution_strategy", "legacy_or_oom_safe_v1")
    if strategy not in {
        "legacy_or_oom_safe_v1", "compact_grouped_v1",
        "small_group_uncheckpoint_v1", "paper_training_v1",
        "prefetch_runtime_v1",
    }:
        raise ValueError(f"unsupported training.execution_strategy={strategy!r}")
    for key in (
        "scene_compaction", "rollout_block_checkpointing",
        "training_runtime_optimizations", "rollout_training_enabled",
        "paired_marginal_actual_step_projection_enabled",
    ):
        if key in training and not isinstance(training[key], bool):
            raise ValueError(f"training.{key} must be boolean")
    input_pipeline = training.get("ebjd_input_pipeline", "legacy_cpu_rigid")
    if input_pipeline not in {
        "legacy_cpu_rigid", "compact_uint8_gpu_rigid_v1",
    }:
        raise ValueError(
            f"unsupported training.ebjd_input_pipeline={input_pipeline!r}")
    if "ebjd_input_pipeline_receipt" in training and not isinstance(
            training["ebjd_input_pipeline_receipt"], str):
        raise ValueError("training.ebjd_input_pipeline_receipt must be a string")
    if "early_stopping" in training and not isinstance(
            training["early_stopping"], dict):
        raise ValueError("training.early_stopping must be a mapping")
    for key in ("physical_group_max_scenes", "physical_group_max_agent_slots"):
        if key in training:
            value = training[key]
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError(f"training.{key} must be a positive integer")
    for key in ("data_loader_workers", "data_loader_prefetch_factor"):
        if key in training:
            value = training[key]
            if isinstance(value, bool) or int(value) != value or int(value) < 0:
                raise ValueError(f"training.{key} must be a non-negative integer")
    for key in (
        "ctf_feedback_transition_start_update",
        "ctf_feedback_transition_updates",
    ):
        if key in training:
            value = training[key]
            if isinstance(value, bool) or int(value) != value or int(value) < 0:
                raise ValueError(f"training.{key} must be a non-negative integer")
    if ctf_basis == "decoder_energy":
        if int(training.get("ctf_feedback_transition_updates", 0)) < 1:
            raise ValueError(
                "decoder_energy requires a positive feedback transition length")
    if "physical_group_max_padding_ratio" in training:
        ratio = training["physical_group_max_padding_ratio"]
        if isinstance(ratio, bool) or float(ratio) < 1.0:
            raise ValueError(
                "training.physical_group_max_padding_ratio must be at least one")
    if "rollout_uncheckpointed_max_agent_slots" in training:
        value = training["rollout_uncheckpointed_max_agent_slots"]
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise ValueError(
                "training.rollout_uncheckpointed_max_agent_slots must be a "
                "non-negative integer")
    if strategy in {
        "compact_grouped_v1", "small_group_uncheckpoint_v1", "paper_training_v1",
        "prefetch_runtime_v1",
    }:
        required_execution = {
            "scene_forward_chunk", "scene_compaction",
            "physical_group_max_scenes", "physical_group_max_agent_slots",
            "physical_group_max_padding_ratio", "rollout_block_checkpointing",
        }
        missing_execution = required_execution - set(training)
        if missing_execution:
            raise ValueError(
                "compact_grouped_v1 execution settings are incomplete: "
                f"missing={sorted(missing_execution)}")
        if training["scene_forward_chunk"] != 1 or training["scene_compaction"] is not True:
            raise ValueError(
                "compact_grouped_v1 requires scene_forward_chunk=1 and scene_compaction=true")
    if strategy == "small_group_uncheckpoint_v1":
        if training.get("rollout_uncheckpointed_max_agent_slots") != 32:
            raise ValueError(
                "small_group_uncheckpoint_v1 requires "
                "rollout_uncheckpointed_max_agent_slots=32")
    if strategy == "paper_training_v1":
        if training.get("rollout_uncheckpointed_max_agent_slots") != 0:
            raise ValueError(
                "paper_training_v1 requires rollout_uncheckpointed_max_agent_slots=0")
        if training.get("training_runtime_optimizations") is not True:
            raise ValueError(
                "paper_training_v1 requires training_runtime_optimizations=true")
    if strategy == "prefetch_runtime_v1":
        expected_runtime = {
            "training_runtime_optimizations": True,
            "data_loader_workers": 4,
            "data_loader_prefetch_factor": 2,
        }
        actual_runtime = {key: training.get(key) for key in expected_runtime}
        if actual_runtime != expected_runtime:
            raise ValueError(
                "prefetch_runtime_v1 requires exact settings "
                f"{expected_runtime}, got {actual_runtime}")
    if config["paths"].get("initialization_role") != "trainable_goal_unet_only":
        raise ValueError("paths.initialization_role must be trainable_goal_unet_only")
    baseline = config["evaluation"].get("matched_gdts_validation")
    if not isinstance(baseline, dict):
        raise ValueError("evaluation.matched_gdts_validation must be a mapping")
    allowed_baseline = {
        "status", "minADE", "minFDE", "JADE", "JFDE", "source",
        "reference_limitation",
        "manifest_sha256", "checkpoint_sha256", "ebjd_initialization_sha256",
        "fold", "unit", "worlds", "steps", "evaluation_seed",
        "scene_count", "agent_occurrence_count", "result_sha256",
    }
    unknown_baseline = set(baseline) - allowed_baseline
    required_baseline = {"status", "minADE", "minFDE", "source"}
    if unknown_baseline or not required_baseline.issubset(baseline):
        raise ValueError(
            "matched baseline fields differ: "
            f"unknown={sorted(unknown_baseline)}, "
            f"missing={sorted(required_baseline - set(baseline))}")
    if baseline.get("status") == "bound_to_manifest":
        required_bound = {
            "manifest_sha256", "checkpoint_sha256",
            "ebjd_initialization_sha256", "fold", "unit", "worlds", "steps",
            "evaluation_seed", "scene_count", "agent_occurrence_count",
            "result_sha256",
        }
        missing_bound = required_bound - set(baseline)
        if missing_bound:
            raise ValueError(
                "bound matched baseline is incomplete: "
                f"missing={sorted(missing_bound)}")
        for path_key, hash_key in (
            ("matched_gdts_validation_checkpoint",
             "matched_gdts_validation_checkpoint_sha256"),
            ("matched_gdts_validation_result",
             "matched_gdts_validation_result_sha256"),
        ):
            if not config["paths"].get(path_key) or not config["paths"].get(hash_key):
                raise ValueError(f"paths.{path_key} and paths.{hash_key} are required")


def build_model(
    config: dict, ablation: str = "none", *, proposal_conditioned: bool = False,
) -> EBJDModel:
    if ablation not in ABLATIONS:
        raise ValueError(f"unsupported ablation {ablation!r}; expected {sorted(ABLATIONS)}")
    validate_supported_config(config)
    rep = config["representation"]
    map_config = config["map_encoder"]
    model = EBJDModel(
        goal_scale=float(rep["goal_scale_m"]),
        bridge_scale=float(rep["bridge_scale_m"]),
        crop_width_m=float(map_config["crop_width_m"]),
        map_agent_chunk=int(map_config["agent_chunk"]),
        hidden_dim=int(config["denoiser"]["hidden_dim"]),
        attention_heads=int(config["denoiser"]["heads"]),
        ff_dim=int(config["denoiser"]["ff_dim"]),
        history_social_blocks=int(config["history_encoder"]["social_blocks"]),
        ctf_feedback_basis=str(
            config["denoiser"].get("ctf_feedback_basis", "legacy")),
        proposal_conditioned=proposal_conditioned,
        block_type=str(config["denoiser"].get("block_type", "legacy_fjab")),
        coarse_blocks=int(config["denoiser"]["coarse_blocks"]),
        refinement_blocks=int(config["denoiser"]["refinement_blocks"]),
        bridge_prior=(
            "integrated_acceleration_v1"
            if rep["type"] == "invertible_endpoint_acceleration_bridge_v1"
            else "brownian_position"),
    )
    model.denoiser.activation_checkpointing = bool(
        config["training"]["activation_checkpointing"])
    model.encoder.activation_checkpointing = bool(
        config["training"]["activation_checkpointing"])
    preserve_checkpoint_rng = not bool(
        config["training"].get("training_runtime_optimizations", False))
    model.encoder.checkpoint_preserve_rng_state = preserve_checkpoint_rng
    model.denoiser.checkpoint_preserve_rng_state = preserve_checkpoint_rng
    source = config["paths"].get("accepted_fold_matched_gdts_unet")
    if source:
        source_path = Path(source)
        expected = config["paths"].get("accepted_fold_matched_gdts_unet_sha256")
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



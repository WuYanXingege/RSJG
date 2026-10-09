import copy

import torch

from conftest import make_batch
from ebjd.model import EBJDModel
from ebjd.config import load_config, semantic_config_hash, validate_supported_config
from ebjd.optimizer import list_norm
from ebjd.train import validate_execution_migration_settings
from ebjd.trainer import (
    BatchRandomDraws, EBJDTrainer, TrainerOptions, contiguous_scene_groups,
)


def _fix_identities(batch):
    batch.agent_ids = [
        identities[:int(valid.sum())]
        for identities, valid in zip(batch.agent_ids, batch.valid, strict=True)
    ]
    return batch


def _options(**overrides):
    values = dict(
        precision="fp32", activation_checkpointing=False,
        scene_forward_chunk=1, rollout_worlds=2, rollout_steps=2,
        rollout_start_epoch=1, rollout_ramp_end_epoch=2,
    )
    values.update(overrides)
    return TrainerOptions(**values)


def test_compaction_preserves_nonprefix_valid_order_identity_and_draw_mapping():
    batch = _fix_identities(make_batch((2, 3), pixels=8))
    # Relocate the second valid agent of scene zero to a non-prefix slot.
    for tensor in (batch.observed, batch.future, batch.semantic_maps):
        tensor[0, 2] = tensor[0, 1]
        tensor[0, 1].zero_()
    batch.valid[0] = torch.tensor([True, False, True])
    compact = batch.compact_scenes(0, 2)
    assert compact.valid.tolist() == [[True, True, False], [True, True, True]]
    torch.testing.assert_close(compact.observed[0, :2], batch.observed[0, [0, 2]])
    assert compact.agent_ids == batch.agent_ids
    assert compact.scene_ids == batch.scene_ids

    diffusion = torch.arange(2 * 1 * 3 * 12 * 2).reshape(2, 1, 3, 12, 2).float()
    rollout = torch.arange(2 * 2 * 3 * 12 * 2).reshape(2, 2, 3, 12, 2).float()
    draws = BatchRandomDraws(torch.tensor([0.2, 0.8]), diffusion, rollout)
    packed = draws.compact_scenes(batch.valid, 0, 2)
    torch.testing.assert_close(
        packed.diffusion_noise[0, :, :2], diffusion[0, :, [0, 2]])
    torch.testing.assert_close(
        packed.rollout_initial[0, :, :2], rollout[0, :, [0, 2]])
    assert not packed.diffusion_noise[0, :, 2].any()


def test_contiguous_adaptive_groups_are_deterministic_and_dense_safe():
    counts = [3, 6, 30, 1, 57, 1, 7, 8]
    valid = torch.zeros(len(counts), max(counts), dtype=torch.bool)
    for scene, count in enumerate(counts):
        valid[scene, :count] = True
    groups = contiguous_scene_groups(
        valid, max_scenes=4, max_agent_slots=96, max_padding_ratio=1.6)
    assert groups == [(0, 2), (2, 3), (3, 4), (4, 5), (5, 8)]
    assert groups == contiguous_scene_groups(valid, 4, 96, 1.6)
    assert [index for start, stop in groups for index in range(start, stop)] == list(
        range(len(counts)))


def test_compacted_grouped_train_step_matches_certified_scene_execution():
    torch.manual_seed(123)
    base = EBJDModel()
    batch = _fix_identities(make_batch((1, 2), pixels=16))
    certified = EBJDTrainer(copy.deepcopy(base), _options())
    optimized = EBJDTrainer(copy.deepcopy(base), _options(
        scene_compaction=True,
        physical_group_max_scenes=2,
        physical_group_max_agent_slots=8,
        physical_group_max_padding_ratio=2.0,
        rollout_block_checkpointing=False,
    ))

    torch.manual_seed(999)
    reference = certified.train_step([batch], epoch=1)
    torch.manual_seed(999)
    candidate = optimized.train_step([batch], epoch=1)

    for name in (
        "loss", "diffusion", "geometry", "map", "rollout",
        "soft_marginal_ade", "soft_marginal_fde", "minimum_time",
        "maximum_time", "candidate_dot_marginal_ade",
        "candidate_dot_marginal_fde", "applied_dot_marginal_ade",
        "applied_dot_marginal_fde", "candidate_decrement_norm",
        "applied_decrement_norm", "projection_changed_fraction",
    ):
        assert abs(float(reference[name]) - float(candidate[name])) < 8e-6, name
    assert candidate["physical_group_count"] == 1
    assert candidate["maximum_physical_group_scenes"] == 2
    assert candidate["maximum_physical_agent_slots"] == 4
    difference = [
        left.detach() - right.detach()
        for left, right in zip(
            certified.parameters, optimized.parameters, strict=True)
    ]
    assert float(list_norm(difference)) < 8e-6
    assert max(float(value.abs().max()) for value in difference) < 2e-6


def test_speed_runtime_config_is_semantic_hash_neutral_and_migration_exact():
    archived = load_config("configs/endpoint_bridge_joint.yaml")
    optimized = copy.deepcopy(archived)
    optimized["training"].update({
        "execution_strategy": "compact_grouped_v1",
        "scene_compaction": True,
        "physical_group_max_scenes": 4,
        "physical_group_max_agent_slots": 96,
        "physical_group_max_padding_ratio": 1.5,
        "rollout_block_checkpointing": False,
    })
    validate_supported_config(optimized)
    assert semantic_config_hash(optimized) == semantic_config_hash(archived)
    profile = validate_execution_migration_settings(optimized)
    assert profile["strategy"] == "compact_grouped_v1"
    assert profile["physical_group_max_agent_slots"] == 96

    unsupported = copy.deepcopy(optimized)
    unsupported["training"]["physical_group_max_agent_slots"] = 97
    try:
        validate_execution_migration_settings(unsupported)
    except ValueError as error:
        assert "requires exact settings" in str(error)
    else:
        raise AssertionError("an uncertified migration budget was accepted")


def test_architecture_identity_and_rollout_off_path_are_preserved():
    torch.manual_seed(71)
    model = EBJDModel()
    assert sum(parameter.numel() for parameter in model.parameters()) == 3_612_764
    state_names = list(model.state_dict())
    trainer = EBJDTrainer(model, _options(
        rollout_start_epoch=99,
        scene_compaction=True,
        physical_group_max_scenes=4,
        physical_group_max_agent_slots=96,
        physical_group_max_padding_ratio=1.5,
        rollout_block_checkpointing=False,
    ))
    batch = _fix_identities(make_batch((1, 2, 3), pixels=16))
    record = trainer.train_step([batch], epoch=1)
    assert record["rollout_update"] == 0
    assert record["physical_group_count"] == 1
    assert list(model.state_dict()) == state_names
    assert sum(parameter.numel() for parameter in model.parameters()) == 3_612_764


def test_denoiser_checkpoint_policy_is_explicit_per_call(monkeypatch):
    import ebjd.denoiser as denoiser_module

    torch.manual_seed(19)
    model = EBJDModel()
    model.train()
    batch = _fix_identities(make_batch((2,), pixels=16))
    context = model.encode_context(
        batch.observed, batch.semantic_maps, batch.valid)
    latent = torch.randn(1, 1, 2, 12, 2)
    calls = []

    def counted(function, *args, **kwargs):
        calls.append(kwargs.get("use_reentrant"))
        return function(*args)

    monkeypatch.setattr(denoiser_module, "checkpoint", counted)
    model.denoiser(latent, torch.tensor([0.5]), context, checkpoint_blocks=False)
    assert calls == []
    model.denoiser(latent, torch.tensor([0.5]), context, checkpoint_blocks=True)
    assert calls == [False] * 6

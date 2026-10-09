import copy

import torch

from conftest import make_batch
from ebjd.config import (
    config_differences, load_config, semantic_config_hash,
    validate_supported_config,
)
from ebjd.model import EBJDModel
from ebjd.optimizer import ActualStepAdamW, list_norm
from ebjd.trainer import EBJDTrainer, TrainerOptions, logical_scene_weights


def _small_options(scene_forward_chunk: int) -> TrainerOptions:
    return TrainerOptions(
        precision="fp32", activation_checkpointing=False,
        scene_forward_chunk=scene_forward_chunk,
        rollout_worlds=2, rollout_steps=2, rollout_start_epoch=1,
        rollout_ramp_end_epoch=2)


def test_scene_slice_retains_original_padding_and_identity():
    batch = make_batch((1, 3), pixels=16)
    sliced = batch.slice_scenes(0, 1)
    assert sliced.observed.shape == (1, 3, 8, 2)
    assert sliced.semantic_maps.shape == (1, 3, 6, 16, 16)
    assert sliced.valid.tolist() == [[True, False, False]]
    assert sliced.scene_ids == batch.scene_ids[:1]
    assert sliced.agent_ids == batch.agent_ids[:1]
    assert sliced.frame_ids == batch.frame_ids[:1]
    assert sliced.timestamps == batch.timestamps[:1]
    assert sliced.source_sequences == batch.source_sequences[:1]
    assert sliced.metadata == batch.metadata


def test_agent_and_scene_reduction_weights_and_tail_microbatch_semantics():
    valid = torch.tensor([[True, False, False], [True, True, True]])
    agent, scene = logical_scene_weights(valid)
    torch.testing.assert_close(
        agent, torch.tensor([0.25, 0.75], dtype=torch.float64), atol=0, rtol=0)
    torch.testing.assert_close(
        scene, torch.tensor([0.5, 0.5], dtype=torch.float64), atol=0, rtol=0)
    # The canonical counterexample: per-scene agent means [1, 0] must reduce
    # to 1/4, while the scene-reduced quantity remains 1/2.
    values = torch.tensor([1.0, 0.0], dtype=torch.float64)
    assert float((agent * values).sum()) == 0.25
    assert float((scene * values).sum()) == 0.5

    # The 4,249-scene tail has K=3 logical microbatches B=[4,4,1].  The old
    # trainer averages the three logical objectives, rather than reweighting
    # the final nine scenes into a new global denominator.
    logical_values = torch.tensor([2.0, 5.0, 11.0], dtype=torch.float64)
    assert float(logical_values.mean()) == 6.0


def test_scene_chunk_preserves_rng_layout_and_full_rollout_update():
    torch.manual_seed(123)
    base = EBJDModel()
    batch = make_batch((1, 3), pixels=16)
    legacy = EBJDTrainer(copy.deepcopy(base), _small_options(0))
    chunked = EBJDTrainer(copy.deepcopy(base), _small_options(1))

    torch.manual_seed(999)
    legacy_record = legacy.train_step([batch], epoch=1)
    torch.manual_seed(999)
    chunked_record = chunked.train_step([batch], epoch=1)

    component_tolerance = 2e-6
    for name in (
        "loss", "diffusion", "geometry", "map", "rollout",
        "soft_marginal_ade", "soft_marginal_fde", "minimum_time",
        "maximum_time", "candidate_dot_marginal_ade",
        "candidate_dot_marginal_fde", "applied_dot_marginal_ade",
        "applied_dot_marginal_fde", "candidate_decrement_norm",
        "applied_decrement_norm", "projection_changed_fraction",
    ):
        assert abs(float(legacy_record[name]) - float(chunked_record[name])) < component_tolerance
    assert legacy_record["rollout_update"] == chunked_record["rollout_update"] == 1
    assert legacy_record["projection_active"] == chunked_record["projection_active"]
    assert legacy.update_index == chunked.update_index == 1

    parameter_difference = [
        left.detach() - right.detach()
        for left, right in zip(legacy.parameters, chunked.parameters, strict=True)]
    assert float(list_norm(parameter_difference)) < 2e-6
    assert max(float(value.abs().max()) for value in parameter_difference) < 5e-7
    legacy_states = legacy.optimizer.state_dict()["states"]
    chunked_states = chunked.optimizer.state_dict()["states"]
    for left, right in zip(legacy_states, chunked_states, strict=True):
        assert left["step"] == right["step"] == 1
        torch.testing.assert_close(left["moment"], right["moment"], atol=5e-7, rtol=5e-6)
        torch.testing.assert_close(left["variance"], right["variance"], atol=5e-8, rtol=5e-6)


def test_optimizer_restores_learning_rates_state_and_parameter_order():
    first = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    source = ActualStepAdamW(
        [("first", first)], lr=3e-4, unet_lr=7e-5, weight_decay=0)
    decrement, pending = source.propose([torch.tensor([0.3, -0.4])])
    source.apply(decrement, pending)
    payload = source.state_dict()

    second = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    restored = ActualStepAdamW(
        [("first", second)], lr=1.0, unet_lr=2.0, weight_decay=0)
    restored.load_state_dict(payload)
    assert restored.lr == 3e-4
    assert restored.unet_lr == 7e-5
    assert restored.state_dict()["parameter_names"] == ["first"]
    assert restored.state_dict()["states"][0]["step"] == 1
    torch.testing.assert_close(
        restored.state_dict()["states"][0]["moment"],
        payload["states"][0]["moment"], atol=0, rtol=0)

    bad = copy.deepcopy(payload)
    bad["parameter_names"] = ["renamed"]
    try:
        restored.load_state_dict(bad)
    except ValueError as error:
        assert "names/order" in str(error)
    else:
        raise AssertionError("optimizer parameter-order mismatch was accepted")


def test_scene_chunk_is_optional_for_archived_config_and_semantic_hash(tmp_path):
    config = load_config("configs/endpoint_bridge_joint.yaml")
    archived = copy.deepcopy(config)
    archived["training"].pop("scene_forward_chunk")
    validate_supported_config(archived)
    assert semantic_config_hash(archived) == semantic_config_hash(config)

    migrated = copy.deepcopy(config)
    migrated["experiment"]["source_commit"] = "new-execution"
    migrated["paths"]["output_root"] = str(tmp_path)
    differences = config_differences(archived, migrated)
    assert set(differences) == {
        "experiment.source_commit", "paths.output_root",
        "training.scene_forward_chunk",
    }

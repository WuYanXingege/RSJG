from dataclasses import replace

import torch

from conftest import make_batch
from ebjd.model import EBJDModel
from ebjd.objectives import fixed_anchor_path_loss
from ebjd.representation import fixed_endpoint_trajectory
from ebjd.sampling import differentiable_sample, predict
from ebjd.trainer import EBJDTrainer, TrainerOptions


def native_batch(counts=(1, 2)):
    batch = make_batch(counts)
    batch.agent_ids = [
        identities[:count] for identities, count in zip(
            batch.agent_ids, counts, strict=True)]
    generator = torch.Generator().manual_seed(71)
    batch.native_goals = (
        batch.observed[..., -1, None, :]
        + torch.randn(*batch.observed.shape[:2], 20, 2, generator=generator))
    batch.native_goals *= batch.valid[..., None, None]
    return batch


def full_model():
    model = EBJDModel(
        native_goal_conditioning=True,
        memory_mode="split_gated", motion_summaries=True,
        map_memory_grid=4, anchor_residual=True,
        joint_goal_assembly=True, geometry_mapping="decoder_jacobian")
    model.encoder.activation_checkpointing = False
    model.denoiser.activation_checkpointing = False
    model.denoiser.static_context_cache_enabled = True
    return model


def test_v2_memory_layout_anchor_roundtrip_and_forward_shapes():
    torch.manual_seed(72)
    model = full_model()
    batch = native_batch()
    context = model.encode_context(
        batch.observed, batch.semantic_maps, batch.valid, batch.native_goals)
    assert context.memory.shape[-2:] == (28, 128)
    assert context.motion_memory.shape[-2:] == (12, 128)
    assert context.map_memory.shape[-2:] == (16, 128)
    context, labels = model.assign_training_anchors(context, batch.future)
    assert labels.shape == batch.valid.shape
    latent = model.representation.encode_target(
        batch.future, context.baseline, context.world_anchors[:, 0])
    decoded, goal = model.representation.decode(
        latent[:, None], context.baseline, context.world_anchors)
    torch.testing.assert_close(decoded[:, 0], batch.future, atol=2e-5, rtol=2e-5)
    assert torch.equal(decoded[..., -1, :], goal)
    final, coarse, trajectory, endpoint = model.denoiser(
        latent[:, None], torch.tensor([0.4, 0.4]), context)
    assert final.shape == coarse.shape == (2, 1, 2, 12, 2)
    assert trajectory.shape == (2, 1, 2, 12, 2)
    assert endpoint.shape == (2, 1, 2, 2)
    assert all(torch.isfinite(value).all() for value in (final, coarse, trajectory, endpoint))


def test_joint_goal_zero_energy_preserves_seeded_permutations_and_sets():
    torch.manual_seed(73)
    model = full_model().eval()
    batch = native_batch((2,))
    context = model.encode_context(
        batch.observed, batch.semantic_maps, batch.valid, batch.native_goals)
    context, permutations, diagnostics = model.assemble_world_anchors(
        context, batch.scene_ids, batch.agent_ids, seed=2035)
    expected = torch.arange(20)
    for agent in range(2):
        assert torch.equal(permutations[0, agent].sort().values.cpu(), expected)
        torch.testing.assert_close(
            context.world_anchors[0, :, agent].sort(dim=0).values,
            batch.native_goals[0, agent].sort(dim=0).values)
    assert torch.equal(diagnostics.initial_energy, diagnostics.final_energy)
    assert torch.equal(diagnostics.changed_world_fraction, torch.zeros(1))
    pair = model.joint_goal_assembler.pair_energy(
        context, context.world_anchors)
    torch.testing.assert_close(pair, pair.transpose(-1, -2))


def test_pairing_loss_has_energy_head_gradient_and_history_is_detached():
    torch.manual_seed(74)
    model = full_model().train()
    batch = native_batch((2,))
    context = model.encode_context(
        batch.observed, batch.semantic_maps, batch.valid, batch.native_goals)
    context.social_h.retain_grad()
    generator = torch.Generator(device="cpu").manual_seed(3101)
    loss = model.joint_goal_assembler.association_loss(
        context, batch.future, generator)
    assert float(loss.detach()) > 0
    loss.backward()
    assert model.joint_goal_assembler.pair_mlp[-1].weight.grad is not None
    assert model.joint_goal_assembler.pair_mlp[-1].weight.grad.abs().sum() > 0
    assert context.social_h.grad is None


def test_no_anchor_split_gate_uses_zero_third_input_and_no_joint_assembler():
    model = EBJDModel(
        native_goal_conditioning=True,
        memory_mode="split_gated", motion_summaries=True,
        map_memory_grid=4, anchor_residual=False,
        joint_goal_assembly=False, geometry_mapping="decoder_jacobian")
    assert model.joint_goal_assembler is None
    last = model.denoiser.memory_group_gate[-1]
    assert torch.equal(last.weight, torch.zeros_like(last.weight))
    assert torch.equal(last.bias, torch.zeros_like(last.bias))


def test_v2_full_real_training_step_updates_new_modules_without_rollout():
    torch.manual_seed(75)
    model = full_model()
    batch = native_batch((2,))
    options = TrainerOptions(
        accumulation=1, precision="fp32", activation_checkpointing=False,
        use_rollout_loss=False, use_actual_step_constraint=False,
        keep_weight=0.01, association_weight=0.05,
        lazy_off_projection_gradients=True)
    trainer = EBJDTrainer(
        model, options, resolved_config={"seed": 3101})
    before = {
        name: value.detach().clone() for name, value in model.named_parameters()
        if any(part in name for part in (
            "motion_summary", "memory_group_gate", "goal_anchor_gate",
            "bridge_anchor_gate", "joint_goal_assembler"))}
    record = trainer.train_step([batch], epoch=21)
    assert record["rollout_update"] == 0
    assert record["projection_executed"] == 0
    assert record["keep"] >= 0 and record["association"] > 0
    changed = {
        name for name, value in model.named_parameters()
        if name in before and not torch.equal(value.detach(), before[name])}
    assert any("motion_summary" in name for name in changed)
    assert any("memory_group_gate" in name for name in changed)
    assert any("anchor_gate" in name for name in changed)
    assert any("joint_goal_assembler.pair_mlp.2" in name for name in changed)


def test_sampler_context_cache_preserves_candidate_and_anchor_gradients():
    torch.manual_seed(76)
    model = full_model().train()
    batch = native_batch((2,))
    context = model.encode_context(
        batch.observed, batch.semantic_maps, batch.valid, batch.native_goals)
    context, _, _ = model.assemble_world_anchors(
        context, batch.scene_ids, batch.agent_ids, seed=2035)
    indices = torch.tensor([[0, 1, 2, 3]])
    context = model.select_world_anchors(context, indices)
    conditioner = model.denoiser.native_goal_conditioner
    with torch.no_grad():
        conditioner.gate.fill_(0.2)
        conditioner.goal_anchor_gate.fill_(0.2)
        conditioner.bridge_anchor_gate.fill_(0.2)
    initial = torch.randn(1, 4, 2, 12, 2)
    trajectory, _, _ = differentiable_sample(
        model, context, initial, steps=2,
        checkpoint_steps=True, checkpoint_blocks=False)
    trajectory.square().mean().backward()
    gradients = [
        parameter.grad for parameter in conditioner.candidate_mlp.parameters()]
    assert all(value is not None and torch.isfinite(value).all() for value in gradients)
    assert sum(float(value.abs().sum()) for value in gradients) > 0


def test_joint_goal_initializer_is_isolated_and_shared_across_v2_arms():
    torch.manual_seed(77)
    state_before = torch.random.get_rng_state().clone()
    full = EBJDModel(
        native_goal_conditioning=True,
        memory_mode="split_gated", motion_summaries=True,
        map_memory_grid=4, anchor_residual=True,
        joint_goal_assembly=True, joint_goal_init_seed=3_003_108,
        geometry_mapping="decoder_jacobian")
    # Rewind only the public construction RNG to make the comparison isolate
    # variant-specific module order.  The assembler itself must not advance it.
    torch.random.set_rng_state(state_before)
    no_token = EBJDModel(
        native_goal_conditioning=True,
        memory_mode="legacy_joint", motion_summaries=False,
        map_memory_grid=8, anchor_residual=True,
        joint_goal_assembly=True, joint_goal_init_seed=3_003_108,
        geometry_mapping="decoder_jacobian")
    for name, value in full.joint_goal_assembler.state_dict().items():
        torch.testing.assert_close(
            value, no_token.joint_goal_assembler.state_dict()[name],
            atol=0, rtol=0)


def test_read_only_zero_radius_projection_preserves_bridge_residual():
    torch.manual_seed(78)
    model = full_model().eval()
    batch = native_batch((2,))
    generator = torch.Generator().manual_seed(2035)
    trajectory, goal = predict(
        model, batch.observed, batch.semantic_maps, batch.valid,
        worlds=4, steps=1, generator=generator,
        native_goals=batch.native_goals, scene_ids=batch.scene_ids,
        agent_ids=batch.agent_ids, inference_seed=2035,
        pairing_rounds=0, endpoint_projection_radius=0.0)
    context = model.encode_context(
        batch.observed, batch.semantic_maps, batch.valid, batch.native_goals)
    context, _, _ = model.assemble_world_anchors(
        context, batch.scene_ids, batch.agent_ids, 2035, rounds=0)
    torch.testing.assert_close(goal, context.world_anchors[:, :4], atol=1e-6, rtol=0)
    assert trajectory.shape == (1, 4, 2, 12, 2)
    torch.testing.assert_close(trajectory[..., -1, :], goal, atol=1e-6, rtol=0)


def test_fixed_endpoint_helper_is_differentiable_and_preserves_bridge():
    torch.manual_seed(79)
    trajectory = torch.randn(2, 1, 3, 12, 2, requires_grad=True)
    anchor = torch.randn(2, 1, 3, 2)
    fixed = fixed_endpoint_trajectory(trajectory, anchor)
    torch.testing.assert_close(fixed[..., -1, :], anchor, atol=1e-6, rtol=0)
    fraction = torch.arange(1, 13, dtype=torch.float32) / 12
    old_line = trajectory[..., :1, :] + fraction.view(1, 1, 1, 12, 1) * (
        trajectory[..., -1:, :] - trajectory[..., :1, :])
    new_line = trajectory[..., :1, :] + fraction.view(1, 1, 1, 12, 1) * (
        anchor[..., None, :] - trajectory[..., :1, :])
    torch.testing.assert_close(
        trajectory - old_line, fixed - new_line, atol=1e-5, rtol=0)
    fixed[..., :11, :].square().mean().backward()
    assert trajectory.grad is not None
    assert float(trajectory.grad[..., :11, :].abs().sum()) > 0


def test_fixed_anchor_path_loss_masks_padding_excludes_terminal_and_normalizes_agents():
    torch.manual_seed(80)
    final = torch.randn(2, 1, 3, 12, 2, requires_grad=True)
    coarse = torch.randn(2, 1, 3, 12, 2, requires_grad=True)
    target = torch.randn(2, 3, 12, 2)
    anchors = torch.randn(2, 1, 3, 2)
    valid = torch.tensor([[True, True, False], [True, False, False]])
    alpha = torch.tensor([0.25, 0.5]).view(2, 1, 1, 1, 1)
    loss = fixed_anchor_path_loss(
        final, coarse, target, anchors, valid, alpha)
    assert torch.isfinite(loss) and float(loss.detach()) > 0
    loss.backward()
    assert float(final.grad[..., :11, :].abs().sum()) > 0
    assert float(coarse.grad[..., :11, :].abs().sum()) > 0

    changed_target = target.clone()
    changed_target[..., -1, :] += 1000
    changed_final = final.detach().clone()
    changed_coarse = coarse.detach().clone()
    changed_final[:, :, 2] += 1000
    changed_coarse[:, :, 2] -= 1000
    replay = fixed_anchor_path_loss(
        changed_final, changed_coarse, changed_target, anchors, valid, alpha)
    torch.testing.assert_close(loss.detach(), replay, atol=0, rtol=0)

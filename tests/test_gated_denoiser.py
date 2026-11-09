"""Backward compatibility and invariants of the optional gated DiT backbone."""

from copy import deepcopy

import pytest
import torch

from conftest import load_paper_config, make_batch
from ebjd.config import build_model, semantic_config_hash, validate_supported_config
from ebjd.model import EBJDModel
from ebjd.objectives import diffusion_loss, map_loss
from ebjd.sampling import differentiable_sample


MODERN = "endpoint_gated_dit_v1"


def model_for(depth=4, *, training=True):
    model = EBJDModel(
        hidden_dim=32, ff_dim=64, proposal_conditioned=True,
        block_type=MODERN, coarse_blocks=depth, refinement_blocks=depth)
    model.encoder.activation_checkpointing = False
    model.denoiser.activation_checkpointing = False
    return model.train(training)


def open_gates(model):
    # Nonzero modulation makes the invariance tests exercise learned attention
    # rather than pass trivially because adaLN-Zero starts at the identity.
    count = 0
    with torch.no_grad():
        for block in model.denoiser.blocks:
            for module in block.modules():
                if (isinstance(module, torch.nn.Linear) and
                        module.out_features == 12 * model.denoiser.dim):
                    module.bias.fill_(0.1)
                    count += 1
    assert count == len(model.denoiser.blocks)


def forward(model, batch, latent, goals, checkpoint_blocks=False):
    context = model.encode_context(batch.observed, batch.semantic_maps, batch.valid)
    times = torch.full((batch.observed.shape[0],), 0.4)
    outputs = model.denoiser(
        latent, times, context, proposal_goals=goals, proposal_eta=0.0,
        checkpoint_blocks=checkpoint_blocks)
    return outputs, context


def test_implicit_legacy_preserves_explicit_six_block_state_forward_and_gradients():
    torch.manual_seed(881)
    implicit = EBJDModel(hidden_dim=32, ff_dim=64, proposal_conditioned=True)
    torch.manual_seed(881)
    explicit = EBJDModel(
        hidden_dim=32, ff_dim=64, proposal_conditioned=True,
        block_type="legacy_fjab", coarse_blocks=3, refinement_blocks=3)
    for key, value in implicit.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[key])
    batch = make_batch((1, 2))
    latent = torch.randn(2, 2, 2, 12, 2)
    goals = batch.future[:, None, :, -1].expand(-1, 2, -1, -1)
    left, _ = forward(implicit, batch, latent, goals)
    right, _ = forward(explicit, batch, latent, goals)
    assert all(torch.equal(a, b) for a, b in zip(left, right, strict=True))
    left[0].square().sum().backward()
    right[0].square().sum().backward()
    for (ka, a), (kb, b) in zip(
            implicit.named_parameters(), explicit.named_parameters(), strict=True):
        assert ka == kb
        if a.grad is None or b.grad is None:
            assert a.grad is b.grad
        else:
            assert torch.equal(a.grad, b.grad)


@pytest.mark.parametrize("depth", [3, 4])
def test_modern_six_and_eight_blocks_complete_real_optimizer_updates(depth):
    torch.manual_seed(882 + depth)
    model = model_for(depth)
    assert len(model.denoiser.blocks) == 2 * depth
    assert model.denoiser.coarse_blocks == depth
    batch = make_batch((2,))
    goals = batch.future[:, None, :, -1]
    latent = torch.randn(1, 1, 2, 12, 2)
    target = torch.randn_like(latent)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    endpoint_condition_gradient_seen = False
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        outputs, context = forward(model, batch, latent, goals)
        final, coarse, coarse_path, _ = outputs
        loss = (diffusion_loss(final, coarse, target, batch.valid)
                + 0.1 * map_loss(batch.future, context)
                + 0.01 * coarse_path.square().mean())
        assert torch.isfinite(loss)
        loss.backward()
        endpoint_condition_gradient_seen |= any(
            name.startswith("denoiser.endpoint_condition") and p.grad is not None
            and p.grad.abs().sum() > 0 for name, p in model.named_parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all()
                   for p in model.parameters())
        optimizer.step()
    required = ["encoder.map_unet", "encoder.history_gru",
                "denoiser.blocks.0", f"denoiser.blocks.{2 * depth - 1}",
                "denoiser.clean_injection", "denoiser.map_injection",
                "denoiser.future_geometry_mlp", "denoiser.proposal_embedding",
                "denoiser.endpoint_condition"]
    changed = {name for name, p in model.named_parameters()
               if not torch.equal(p.detach(), initial[name])}
    assert all(any(name.startswith(prefix) for name in changed)
               for prefix in required)
    assert endpoint_condition_gradient_seen


def test_modern_open_gates_preserve_agent_world_scene_and_padding_invariants():
    torch.manual_seed(889)
    model = model_for(training=False)
    open_gates(model)
    batch = make_batch((2, 2))
    latent = torch.randn(2, 3, 2, 12, 2)
    goals = batch.future[:, None, :, -1].expand(-1, 3, -1, -1).clone()
    goals = goals + torch.randn_like(goals) * 0.2
    reference, _ = forward(model, batch, latent, goals)

    agent_order = torch.tensor([1, 0])
    world_order = torch.tensor([2, 0, 1])
    permuted_batch = deepcopy(batch)
    for name in ("observed", "future", "semantic_maps", "valid"):
        setattr(permuted_batch, name, getattr(batch, name)[:, agent_order])
    permuted, _ = forward(
        model, permuted_batch, latent[:, world_order][:, :, agent_order],
        goals[:, world_order][:, :, agent_order])
    torch.testing.assert_close(
        permuted[0], reference[0][:, world_order][:, :, agent_order],
        atol=2e-5, rtol=2e-5)
    isolated, _ = forward(model, batch.slice_scenes(0, 1), latent[:1], goals[:1])
    torch.testing.assert_close(isolated[0], reference[0][:1], atol=2e-5, rtol=2e-5)
    changed_latent = latent.clone()
    changed_goals = goals.clone()
    changed_latent[:, 1] += 4
    changed_goals[:, 1] += 3
    changed, _ = forward(model, batch, changed_latent, changed_goals)
    assert torch.equal(changed[0][:, 0], reference[0][:, 0])

    padded = make_batch((1, 2))
    padded_goals = padded.future[:, None, :, -1].expand(-1, 3, -1, -1)
    output, _ = forward(model, padded, latent, padded_goals)
    assert torch.equal(output[0][0, :, 1], torch.zeros_like(output[0][0, :, 1]))
    assert all(torch.isfinite(value).all() for value in output)


def test_modern_block_checkpoint_recomputation_keeps_outputs_and_gradients():
    torch.manual_seed(890)
    plain = model_for(depth=3)
    open_gates(plain)
    checked = deepcopy(plain)
    batch = make_batch((2,))
    latent = torch.randn(1, 2, 2, 12, 2)
    goals = batch.future[:, None, :, -1].expand(-1, 2, -1, -1)
    left, _ = forward(plain, batch, latent, goals, checkpoint_blocks=False)
    right, _ = forward(checked, batch, latent, goals, checkpoint_blocks=True)
    torch.testing.assert_close(left[0], right[0], atol=0, rtol=0)
    left[0].square().sum().backward()
    right[0].square().sum().backward()
    for (ka, a), (kb, b) in zip(
            plain.named_parameters(), checked.named_parameters(), strict=True):
        assert ka == kb
        if a.grad is None or b.grad is None:
            assert a.grad is b.grad
        else:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)


def test_modern_and_legacy_checkpoint_families_cannot_silently_mix():
    legacy = EBJDModel(hidden_dim=32, ff_dim=64, proposal_conditioned=True)
    restored = EBJDModel(hidden_dim=32, ff_dim=64, proposal_conditioned=True)
    restored.load_state_dict(legacy.state_dict(), strict=True)
    modern = model_for(depth=3)
    modern_restored = model_for(depth=3)
    modern_restored.load_state_dict(modern.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        modern.load_state_dict(legacy.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        model_for(depth=4).load_state_dict(modern_restored.state_dict(), strict=True)


def test_modern_config_records_architecture_and_depth_as_research_semantics():
    legacy = load_paper_config()
    modern = deepcopy(legacy)
    modern["denoiser"].update(block_type=MODERN, coarse_blocks=4, refinement_blocks=4)
    validate_supported_config(modern)
    assert semantic_config_hash(legacy) != semantic_config_hash(modern)
    model = build_model(modern, proposal_conditioned=True)
    assert len(model.denoiser.blocks) == 8
    assert model.denoiser.coarse_blocks == 4
    invalid = deepcopy(legacy)
    invalid["denoiser"].update(block_type="legacy_fjab", coarse_blocks=4, refinement_blocks=4)
    with pytest.raises(ValueError):
        validate_supported_config(invalid)


def test_modern_ddim_keeps_literal_endpoints_and_is_differentiable():
    torch.manual_seed(891)
    model = model_for(depth=3)
    open_gates(model)
    batch = make_batch((2,))
    context = model.encode_context(batch.observed, batch.semantic_maps, batch.valid)
    initial = torch.randn(1, 2, 2, 12, 2, requires_grad=True)
    goals = batch.future[:, None, :, -1].expand(-1, 2, -1, -1)
    paths, endpoint, _ = differentiable_sample(
        model, context, initial, steps=2,
        proposal_goals=goals, proposal_eta=0.0)
    assert torch.equal(endpoint, goals)
    assert torch.equal(paths[..., -1, :], goals)
    paths.square().mean().backward()
    assert initial.grad is not None and torch.isfinite(initial.grad).all()
    assert initial.grad[..., :11, :].abs().sum() > 0

import torch

from ebjd.model import EBJDModel
from ebjd.objectives import diffusion_loss, map_loss

from conftest import make_batch


def test_forward_n1_n2_multiple_scenes_padding_and_finite():
    torch.manual_seed(3)
    model = EBJDModel()
    batch = make_batch((1, 2))
    context = model.encode_context(batch.observed, batch.semantic_maps, batch.valid)
    latent = torch.randn(2, 2, 2, 12, 2)
    for time in (torch.tensor([1.0, 1.0]), torch.tensor([1e-7, 1e-7])):
        outputs = model.denoiser(latent, time, context)
        assert outputs[0].shape == (2, 2, 2, 12, 2)
        assert all(torch.isfinite(item).all() for item in outputs)
        assert torch.equal(outputs[0][0, :, 1], torch.zeros_like(outputs[0][0, :, 1]))


def test_agent_permutation_equivariance_and_scene_isolation():
    torch.manual_seed(4)
    model = EBJDModel().eval()
    batch = make_batch((2, 2))
    latent = torch.randn(2, 1, 2, 12, 2)
    context = model.encode_context(batch.observed, batch.semantic_maps, batch.valid)
    reference = model.denoiser(latent, torch.tensor([0.4, 0.4]), context)[0]
    permutation = torch.tensor([1, 0])
    perm_context = model.encode_context(
        batch.observed[:, permutation], batch.semantic_maps[:, permutation],
        batch.valid[:, permutation])
    permuted = model.denoiser(
        latent[:, :, permutation], torch.tensor([0.4, 0.4]), perm_context)[0]
    torch.testing.assert_close(permuted, reference[:, :, permutation], atol=2e-5, rtol=2e-5)
    one_context = model.encode_context(
        batch.observed[:1], batch.semantic_maps[:1], batch.valid[:1])
    isolated = model.denoiser(latent[:1], torch.tensor([0.4]), one_context)[0]
    torch.testing.assert_close(isolated, reference[:1], atol=2e-5, rtol=2e-5)


def test_all_required_modules_receive_gradients():
    torch.manual_seed(5)
    model = EBJDModel()
    batch = make_batch((2,))
    context = model.encode_context(batch.observed, batch.semantic_maps, batch.valid)
    latent = torch.randn(1, 1, 2, 12, 2)
    final, coarse, pre_trajectory, _ = model.denoiser(latent, torch.tensor([0.5]), context)
    loss = diffusion_loss(final, coarse, torch.randn_like(final), batch.valid)
    loss = loss + pre_trajectory.square().mean() + map_loss(batch.future, context)
    loss.backward()
    names = {name for name, parameter in model.named_parameters() if parameter.grad is not None}
    required = [
        "encoder.map_unet", "encoder.history_gru", "encoder.history_social.0",
        "encoder.history_social.1", "denoiser.blocks.0", "denoiser.blocks.5",
        "denoiser.pre_bridge_head", "denoiser.pre_goal_head",
        "denoiser.final_bridge_head", "denoiser.final_goal_head",
    ]
    assert all(any(name.startswith(prefix) for name in names) for prefix in required)
    assert all(parameter.requires_grad for parameter in model.parameters())

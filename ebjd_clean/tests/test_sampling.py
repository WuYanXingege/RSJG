import torch

from ebjd.model import EBJDModel
from ebjd.sampling import differentiable_sample

from conftest import make_batch


def test_rollout_is_differentiable_and_world_chunks_match():
    torch.manual_seed(8)
    model = EBJDModel().eval()
    batch = make_batch((2,))
    context = model.encode_context(batch.observed, batch.semantic_maps, batch.valid)
    noise = torch.randn(1, 2, 2, 12, 2)
    complete, goal, _ = differentiable_sample(model, context, noise, steps=2)
    chunks = [differentiable_sample(model, context, noise[:, index:index + 1], steps=2)[0]
              for index in range(2)]
    torch.testing.assert_close(complete, torch.cat(chunks, 1), atol=2e-5, rtol=2e-5)
    loss = complete.square().mean() + goal.square().mean()
    loss.backward()
    assert model.denoiser.pre_goal_head[-1].weight.grad is not None
    assert model.denoiser.blocks[0].temporal.in_proj_weight.grad is not None

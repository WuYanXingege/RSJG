"""CPU checks for opt-in current-seed execution optimizations."""

from dataclasses import replace

import torch

from ebjd.encoders import EBJDContext
from ebjd.optimizer import batched_gradients, gradients


def test_batched_gradients_preserves_independent_vjps_and_unused_zeros():
    first = torch.nn.Parameter(torch.tensor([1.5, -0.5]))
    unused = torch.nn.Parameter(torch.tensor([2.0]))

    def losses():
        return [first.square().sum(), (3.0 * first).sum(), first.prod()]

    sequential_losses = losses()
    expected = [
        gradients(loss, [first, unused], retain_graph=index < 2)
        for index, loss in enumerate(sequential_losses)
    ]
    actual = batched_gradients(losses(), [first, unused])
    for expected_row, actual_row in zip(expected, actual, strict=True):
        for expected_value, actual_value in zip(
            expected_row, actual_row, strict=True,
        ):
            torch.testing.assert_close(actual_value, expected_value)


def test_context_fp32_cache_remains_attached_to_source_graph():
    source = torch.randn(
        1, 1, 32, 3, 3, dtype=torch.bfloat16, requires_grad=True)
    zeros = torch.zeros(1)
    context = EBJDContext(
        memory=zeros, social_h=zeros, observed_bias=zeros, baseline=zeros,
        mean_velocity=zeros, map_features=source, map_logits=zeros,
        map_center=zeros, valid=torch.ones(1, 1, dtype=torch.bool),
        observed=zeros, crop_width_m=32.0)
    cached = replace(context, map_features_fp32=context.map_features.float())
    assert cached.map_features_fp32 is not None
    assert cached.map_features_fp32.dtype == torch.float32
    assert cached.map_features_fp32.grad_fn is not None
    cached.map_features_fp32.sum().backward()
    torch.testing.assert_close(source.grad, torch.ones_like(source))

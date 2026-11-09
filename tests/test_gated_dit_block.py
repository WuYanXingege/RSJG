"""Targeted contracts for the opt-in endpoint-gated factorized block."""

from copy import deepcopy

import pytest
import torch

from ebjd.gated_dit import EndpointGatedJointBlock, NormalizedAttention


def inputs(batch=2, worlds=3, agents=3, times=12, dim=32):
    valid = torch.tensor([[True, True, False], [True, False, False]])[:batch, :agents]
    return (
        torch.randn(batch, worlds, agents, times, dim),
        torch.randn(batch, agents, 72, dim),
        torch.randn(batch, agents, dim),
        torch.randn(batch, dim),
        torch.randn(batch, worlds, times, 4, agents, agents),
        valid,
        torch.randn(batch, worlds, agents, dim),
    )


def open_gates(block):
    with torch.no_grad():
        block.modulation[-1].bias.view(4, 3, block.dim)[:, 2].fill_(0.2)


def test_zero_initialized_block_is_identity_and_gate_can_learn():
    torch.manual_seed(51)
    block = EndpointGatedJointBlock(32, 4, 64)
    values = inputs()
    output = block(*values)
    expected = values[0] * values[5][:, None, :, None, None]
    assert torch.equal(output, expected)
    (output * torch.randn_like(output)).sum().backward()
    gradient = block.modulation[-1].weight.grad.view(4, 3, 32, 32)
    assert torch.isfinite(gradient).all()
    assert gradient[:, 2].abs().sum() > 0
    # Exact adaLN-Zero starts attention/FFN weights behind closed gates.
    assert block.temporal.query.weight.grad.abs().sum() == 0
    assert block.ff[0].weight.grad.abs().sum() == 0


def test_open_block_is_agent_and_world_equivariant_padding_safe():
    torch.manual_seed(52)
    block = EndpointGatedJointBlock(32, 4, 64).eval()
    open_gates(block)
    values = inputs()
    reference = block(*values)
    people = torch.tensor([2, 0, 1])
    worlds = torch.tensor([2, 0, 1])
    h, memory, social, time, bias, valid, goal = values
    permuted = block(
        h[:, worlds][:, :, people], memory[:, people], social[:, people], time,
        bias[:, worlds][:, :, :, :, people][:, :, :, :, :, people],
        valid[:, people], goal[:, worlds][:, :, people],
    )
    torch.testing.assert_close(
        permuted, reference[:, worlds][:, :, people], atol=2e-5, rtol=2e-5)
    assert torch.equal(reference[0, :, 2], torch.zeros_like(reference[0, :, 2]))
    assert torch.equal(reference[1, :, 1:], torch.zeros_like(reference[1, :, 1:]))


def test_scene_and_world_never_mix_and_endpoint_condition_is_active():
    torch.manual_seed(53)
    block = EndpointGatedJointBlock(32, 4, 64).eval()
    with torch.no_grad():
        torch.nn.init.normal_(block.modulation[-1].weight, std=0.01)
    values = inputs()
    reference = block(*values)
    isolated = block(*(value[:1] for value in values))
    torch.testing.assert_close(isolated, reference[:1], atol=2e-5, rtol=2e-5)
    changed = list(values)
    changed[0] = values[0].clone()
    changed[0][:, 0] += 3
    modified = block(*changed)
    assert torch.equal(modified[:, 1:], reference[:, 1:])
    changed = list(values)
    changed[-1] = values[-1].clone()
    changed[-1][:, 0] += 2
    modified = block(*changed)
    assert not torch.equal(modified[:, 0], reference[:, 0])
    assert torch.equal(modified[:, 1:], reference[:, 1:])


def test_fully_masked_attention_and_block_have_finite_backward():
    torch.manual_seed(54)
    attention = NormalizedAttention(32, 4)
    query = torch.randn(2, 3, 32, requires_grad=True)
    valid = torch.zeros(2, 3, dtype=torch.bool)
    output = attention(query, valid=valid)
    assert torch.equal(output, torch.zeros_like(output))
    output.square().sum().backward()
    assert torch.isfinite(query.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in attention.parameters())
    block = EndpointGatedJointBlock(32, 4, 64)
    open_gates(block)
    values = list(inputs())
    values[5] = torch.zeros_like(values[5])
    output = block(*values)
    assert torch.equal(output, torch.zeros_like(output))
    output.square().sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in block.parameters())


def test_training_opens_gates_and_reaches_attention_and_geometry_gradients():
    torch.manual_seed(55)
    block = EndpointGatedJointBlock(32, 4, 64)
    optimizer = torch.optim.AdamW(block.parameters(), lr=1e-3)
    values = list(inputs(batch=1))
    values[4].requires_grad_(True)
    target = torch.randn_like(values[0])
    observed = {name: False for name in (
        "temporal.query.weight", "social.query.weight", "cross.query.weight", "ff.0.weight")}
    geometry_gradient_seen = False
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        values[4].grad = None
        output = block(*values)
        loss = (output - target).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        parameters = dict(block.named_parameters())
        for name in observed:
            gradient = parameters[name].grad
            observed[name] |= gradient is not None and float(gradient.abs().sum()) > 0
        geometry_gradient_seen |= (
            values[4].grad is not None and float(values[4].grad.abs().sum()) > 0)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in block.parameters())
        optimizer.step()
    assert all(observed.values()) and geometry_gradient_seen


def test_normalized_attention_uses_geometry_bias_and_restores_state_strictly():
    torch.manual_seed(56)
    attention = NormalizedAttention(32, 4).eval()
    query = torch.randn(1, 3, 32)
    reference = attention(query)
    bias = torch.zeros(1, 4, 3, 3)
    bias[..., 0] = 4
    assert not torch.equal(attention(query, bias=bias), reference)
    restored = NormalizedAttention(32, 4).eval()
    restored.load_state_dict(deepcopy(attention.state_dict()), strict=True)
    assert torch.equal(restored(query, bias=bias), attention(query, bias=bias))
    with pytest.raises(ValueError, match="divisible"):
        NormalizedAttention(31, 4)
    with pytest.raises(ValueError, match="valid"):
        attention(query, valid=torch.ones(1, 4, dtype=torch.bool))

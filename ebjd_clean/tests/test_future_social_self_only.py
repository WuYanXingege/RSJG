import copy

import torch

from ebjd.denoiser import FactorizedFutureBlock
from ebjd.model import EBJDModel


def block_inputs(agents=3, valid=None):
    torch.manual_seed(71)
    valid = torch.tensor(
        [[True] + [False] * (agents - 1)] if valid is None else [valid])
    return {
        "hidden": torch.randn(1, 2, agents, 12, 128),
        "memory": torch.randn(1, agents, 128),
        "social_h": torch.randn(1, agents, 128),
        "time_embedding": torch.randn(1, 128),
        "bias": torch.randn(1, 2, 12, 4, agents, agents),
        "valid": valid,
    }


def test_default_full_path_is_explicit_full_path():
    block = FactorizedFutureBlock().eval()
    values = block_inputs(valid=[True, True, False])
    implicit = block(**values)
    explicit = block(**values, social_self_only=False)
    torch.testing.assert_close(implicit, explicit, atol=0, rtol=0)


def test_gamma_endpoints_are_existing_full_and_self_only_paths():
    block = FactorizedFutureBlock().eval()
    values = block_inputs(valid=[True, True, False])
    full = block(**values)
    gamma_one = block(**values, social_gamma=1.0)
    self_only = block(**values, social_self_only=True)
    gamma_zero = block(**values, social_gamma=0.0)
    torch.testing.assert_close(gamma_one, full, atol=0, rtol=0)
    torch.testing.assert_close(gamma_zero, self_only, atol=0, rtol=0)


def test_gamma_half_interpolates_same_layer_projected_attention_outputs():
    block = FactorizedFutureBlock().eval()
    torch.manual_seed(72)
    query = torch.randn(5, 3, 128)
    bias = torch.randn(5, 4, 3, 3)
    valid = torch.tensor([[True, True, False]]).expand(5, 3)
    full = block._social_output(query, bias, valid, 1.0)
    self_only = block._social_output(query, bias, valid, 0.0)
    half = block._social_output(query, bias, valid, 0.5)
    torch.testing.assert_close(half, self_only + 0.5 * (full - self_only))


def test_self_only_mask_keeps_diagonal_and_blocks_cross_agent_logits():
    bias = torch.arange(2 * 4 * 3 * 3, dtype=torch.float32).reshape(2, 4, 3, 3)
    masked = FactorizedFutureBlock._self_only_bias(bias)
    diagonal = torch.eye(3, dtype=torch.bool).view(1, 1, 3, 3)
    torch.testing.assert_close(masked[diagonal.expand_as(masked)], bias[diagonal.expand_as(bias)])
    assert torch.isneginf(masked[~diagonal.expand_as(masked)]).all()


def test_single_valid_agent_full_and_self_only_are_identical():
    block = FactorizedFutureBlock().eval()
    values = block_inputs()
    full = block(**values)
    self_only = block(**values, social_self_only=True)
    half = block(**values, social_gamma=0.5)
    torch.testing.assert_close(full, self_only, atol=0, rtol=0)
    torch.testing.assert_close(full, half, atol=1e-6, rtol=1e-6)


def test_self_only_mixed_padding_is_finite_and_cross_agent_invariant():
    block = FactorizedFutureBlock().eval()
    values = block_inputs(valid=[True, True, False])
    changed = copy.deepcopy(values)
    changed["hidden"][:, :, 1] += 100.0
    original = block(**values, social_self_only=True)
    intervened = block(**changed, social_self_only=True)
    assert torch.isfinite(original).all()
    assert torch.isfinite(intervened).all()
    torch.testing.assert_close(original[:, :, 0], intervened[:, :, 0], atol=0, rtol=0)
    assert torch.count_nonzero(original[:, :, 2]) == 0


def test_gamma_half_mixed_padding_is_finite():
    block = FactorizedFutureBlock().eval()
    output = block(**block_inputs(valid=[True, True, False]), social_gamma=0.5)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[:, :, 2]) == 0


def test_runtime_mode_keeps_checkpoint_parameters_identical():
    model = EBJDModel()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.denoiser.future_social_self_only = True
    after = model.state_dict()
    assert before.keys() == after.keys()
    for name in before:
        assert before[name].shape == after[name].shape
        torch.testing.assert_close(before[name], after[name], atol=0, rtol=0)

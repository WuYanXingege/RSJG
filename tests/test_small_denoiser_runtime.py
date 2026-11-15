from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from conftest import load_paper_config, make_batch
from ebjd.config import build_model, semantic_config, validate_supported_config
from ebjd.denoiser import GeometryConditionedJointDenoiser
from ebjd.model import EBJDModel


def test_policy_bounds_and_explicit_override():
    model = SimpleNamespace(training=True, proposal_conditioned=True,
        block_type="endpoint_gated_dit_v1", coarse_blocks=4, refinement_blocks=4,
        uncheckpointed_max_agent_slots=16, activation_checkpointing=True)
    policy = GeometryConditionedJointDenoiser._use_block_checkpoints
    assert not policy(model, 1, 1, 16, None, 0)
    assert policy(model, 1, 1, 17, None, 0)
    assert policy(model, 4, 1, 5, None, 0)  # padding counts toward memory bound
    assert policy(model, 1, 20, 1, None, 0)  # multi-world rollout not affected
    assert policy(model, 1, 1, 1, None, 1)  # unfixed endpoints not affected
    assert policy(model, 1, 1, 1, True, 0)
    assert not policy(model, 1, 1, 20, False, 0)
    model.uncheckpointed_max_agent_slots = 32
    assert not policy(model, 1, 1, 32, None, 0)
    assert policy(model, 1, 1, 33, None, 0)
    assert policy(model, 1, 20, 1, None, 0)
    model.uncheckpointed_max_agent_slots = 0
    assert policy(model, 1, 1, 1, None, 0)


def test_config_and_state_dict_compatibility():
    config = load_paper_config()
    config["denoiser"].update(block_type="endpoint_gated_dit_v1", coarse_blocks=4, refinement_blocks=4)
    before = deepcopy(config)
    config["training"]["denoiser_uncheckpointed_max_agent_slots"] = 16
    assert semantic_config(config) == semantic_config(before)
    # Asset loading is not relevant to runtime/key-space wiring.
    config["paths"]["accepted_fold_matched_gdts_unet"] = None
    before["paths"]["accepted_fold_matched_gdts_unet"] = None
    m = build_model(config, proposal_conditioned=True)
    assert m.denoiser.uncheckpointed_max_agent_slots == 16
    assert m.encoder.activation_checkpointing is True
    state = deepcopy(m.state_dict())
    m.denoiser.uncheckpointed_max_agent_slots = 0
    m.load_state_dict(state, strict=True)
    config["training"]["denoiser_uncheckpointed_max_agent_slots"] = 32
    validate_supported_config(config)
    assert semantic_config(config) == semantic_config(before)
    config["training"]["denoiser_uncheckpointed_max_agent_slots"] = 64
    with pytest.raises(ValueError, match="0, 16 or 32"):
        validate_supported_config(config)


def test_cpu_forward_backward_matches_checkpointed_path():
    torch.set_num_threads(2)
    torch.manual_seed(412)
    model = EBJDModel(hidden_dim=192, attention_heads=4, ff_dim=768,
        proposal_conditioned=True, block_type="endpoint_gated_dit_v1",
        coarse_blocks=4, refinement_blocks=4).train()
    other = deepcopy(model)
    other.denoiser.uncheckpointed_max_agent_slots = 16
    batch = make_batch((2,), pixels=32)
    latent = torch.randn(1, 1, 2, 12, 2)
    outputs = []
    for m in (model, other):
        context = m.encode_context(batch.observed, batch.semantic_maps, batch.valid)
        result = m.denoiser(latent, torch.tensor([.63]), context,
                           proposal_goals=batch.future[:, None, :, -1], proposal_eta=0.)
        loss = result[0].square().mean() + .25 * result[1].square().mean()
        loss.backward()
        outputs.append(result)
    for a, b in zip(*outputs):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for (name, a), (_, b) in zip(model.named_parameters(), other.named_parameters()):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

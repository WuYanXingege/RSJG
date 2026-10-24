import copy

import torch

from conftest import make_batch
from ebjd.model import EBJDModel
from ebjd.trainer import ctf_feedback_transition_eta


def _forward(model, batch, latent):
    context = model.encode_context(
        batch.observed, batch.semantic_maps, batch.valid)
    return model.denoiser(latent, torch.tensor([0.4, 0.4]), context)[0]


def test_decoder_energy_weights_follow_ebd_sensitivity_contract():
    model = EBJDModel(ctf_feedback_basis="decoder_energy")
    weights = model.denoiser.ctf_decoder_energy_weights()
    assert weights.shape == (12, 12)
    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights.sum(1), torch.ones(12))
    assert torch.equal(weights[:11, 11], torch.zeros(11))
    assert torch.all(weights[11, 1:] > weights[11, :-1])


def test_legacy_and_zero_eta_are_exact_with_unchanged_state_dict():
    torch.manual_seed(41)
    legacy = EBJDModel()
    candidate = EBJDModel(ctf_feedback_basis="decoder_energy")
    candidate.load_state_dict(copy.deepcopy(legacy.state_dict()), strict=True)
    candidate.denoiser.set_ctf_feedback_eta(0.0)
    assert list(legacy.state_dict()) == list(candidate.state_dict())
    batch = make_batch((1, 2))
    latent = torch.randn(2, 2, 2, 12, 2)
    left = _forward(legacy, batch, latent)
    right = _forward(candidate, batch, latent)
    assert torch.equal(left, right)
    left.square().sum().backward()
    right.square().sum().backward()
    for (_, a), (_, b) in zip(
        legacy.named_parameters(), candidate.named_parameters(), strict=True,
    ):
        if a.grad is None or b.grad is None:
            assert a.grad is b.grad
        else:
            assert torch.equal(a.grad, b.grad)


def test_decoder_energy_is_finite_padding_safe_and_permutation_equivariant():
    torch.manual_seed(42)
    model = EBJDModel(ctf_feedback_basis="decoder_energy").eval()
    model.denoiser.set_ctf_feedback_eta(1.0)
    batch = make_batch((1, 2))
    latent = torch.randn(2, 2, 2, 12, 2)
    output = _forward(model, batch, latent)
    assert torch.isfinite(output).all()
    assert torch.equal(output[0, :, 1], torch.zeros_like(output[0, :, 1]))

    dense = make_batch((2, 2))
    dense_latent = torch.randn(2, 1, 2, 12, 2)
    reference = _forward(model, dense, dense_latent)
    permutation = torch.tensor([1, 0])
    permuted_batch = copy.deepcopy(dense)
    permuted_batch.observed = dense.observed[:, permutation]
    permuted_batch.future = dense.future[:, permutation]
    permuted_batch.semantic_maps = dense.semantic_maps[:, permutation]
    permuted_batch.valid = dense.valid[:, permutation]
    permuted_batch.agent_ids = [
        [ids[index] for index in permutation.tolist()]
        for ids in dense.agent_ids]
    permuted = _forward(
        model, permuted_batch, dense_latent[:, :, permutation])
    torch.testing.assert_close(
        permuted, reference[:, :, permutation], atol=2e-5, rtol=2e-5)


def test_candidate_changes_output_and_transition_hits_fixed_endpoints():
    torch.manual_seed(43)
    model = EBJDModel(ctf_feedback_basis="decoder_energy").eval()
    batch = make_batch((2, 2))
    latent = torch.randn(2, 1, 2, 12, 2)
    model.denoiser.set_ctf_feedback_eta(0.0)
    legacy = _forward(model, batch, latent)
    model.denoiser.set_ctf_feedback_eta(1.0)
    aligned = _forward(model, batch, latent)
    assert not torch.equal(legacy, aligned)
    assert torch.isfinite(aligned).all()
    assert ctf_feedback_transition_eta(25920, 25920, 540) == 0.0
    assert 0.49 < ctf_feedback_transition_eta(26190, 25920, 540) < 0.51
    assert ctf_feedback_transition_eta(26459, 25920, 540) == 1.0
    assert ctf_feedback_transition_eta(27000, 25920, 540) == 1.0

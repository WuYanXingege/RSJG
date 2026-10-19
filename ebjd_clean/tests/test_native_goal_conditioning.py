import copy

import torch

from ebjd.model import EBJDModel
from ebjd.trainer import EBJDTrainer, TrainerOptions


def inputs():
    torch.manual_seed(17)
    return (
        torch.randn(1, 2, 8, 2), torch.zeros(1, 2, 6, 32, 32),
        torch.ones(1, 2, dtype=torch.bool), torch.randn(1, 3, 2, 12, 2),
        torch.tensor([0.4]), torch.randn(1, 2, 20, 2),
    )


def paired_models():
    torch.manual_seed(1)
    legacy = EBJDModel(native_goal_conditioning=False)
    torch.manual_seed(2)
    candidate = EBJDModel(native_goal_conditioning=True)
    missing, unexpected = candidate.load_state_dict(legacy.state_dict(), strict=False)
    assert missing and all(
        name.startswith("denoiser.native_goal_conditioner.") for name in missing)
    assert not unexpected
    return legacy, candidate


def test_zero_gate_is_bitwise_legacy_and_candidate_order_invariant():
    legacy, candidate = paired_models()
    legacy.eval(); candidate.eval()
    observed, maps, valid, latent, time, goals = inputs()
    with torch.no_grad():
        expected = legacy(observed, maps, valid, latent, time)[0]
        actual = candidate(observed, maps, valid, latent, time, goals)[0]
    assert torch.equal(expected, actual)
    with torch.no_grad():
        candidate.denoiser.native_goal_conditioner.gate.fill_(0.2)
        normal = candidate(observed, maps, valid, latent, time, goals)[0]
        permuted = candidate(observed, maps, valid, latent, time, goals.flip(2))[0]
    assert torch.allclose(normal, permuted, atol=2e-6, rtol=2e-6)


def test_gate_has_gradient_updates_and_is_candidate_sensitive():
    _, candidate = paired_models()
    candidate.train()
    observed, maps, valid, latent, time, goals = inputs()
    conditioner = candidate.denoiser.native_goal_conditioner
    assert sum(parameter.numel() for parameter in conditioner.parameters()) == 74945
    assert sum(int(torch.count_nonzero(parameter)) for name, parameter in
               conditioner.named_parameters() if name != "gate") > 0
    loss = candidate(observed, maps, valid, latent, time, goals)[0].square().mean()
    loss.backward()
    assert conditioner.gate.grad is not None
    assert torch.isfinite(conditioner.gate.grad) and conditioner.gate.grad.abs() > 0
    before = conditioner.gate.detach().clone()
    with torch.no_grad():
        conditioner.gate.sub_(1e-3 * conditioner.gate.grad)
    assert not torch.equal(before, conditioner.gate)
    with torch.no_grad():
        first = candidate(observed, maps, valid, latent, time, goals)[0]
        changed = candidate(observed, maps, valid, latent, time, goals + 3.0)[0]
    assert not torch.equal(first, changed)


def test_explicit_model_extension_preserves_old_state_and_rng():
    legacy, candidate = paired_models()
    old = EBJDTrainer(legacy, TrainerOptions())
    loader_generator = torch.Generator().manual_seed(91)
    payload = {
        "schema": "ebjd-checkpoint-v2", "epoch_boundary_only": True,
        "model": copy.deepcopy(legacy.state_dict()),
        "optimizer": copy.deepcopy(old.optimizer.state_dict()),
        "update_index": 1285,
        "rng": old._rng_state(loader_generator),
    }
    restored = EBJDTrainer(candidate, TrainerOptions())
    receipt = restored.restore_model_extension_payload(payload, loader_generator)
    assert receipt["old_model_state_exact"]
    assert receipt["old_optimizer_state_by_name_exact"]
    assert receipt["new_optimizer_state_empty"]
    assert restored.update_index == 1285

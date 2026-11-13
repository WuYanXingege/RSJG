"""Small public API for the standalone, proposal-conditioned EBJD snapshot."""
from __future__ import annotations

from pathlib import Path

import torch
import yaml

from ebjd.model import EBJDModel
from ebjd.sampling import differentiable_sample
from ebjd.unified import UnifiedCouplingBridge, gather_candidates


def profiles() -> dict:
    return yaml.safe_load((Path(__file__).parent / "configs/profiles.yaml").read_text())


def build_network(profile: str = "gated8", *, goal_scale: float = 1.0,
                  bridge_scale: float = 1.0) -> UnifiedCouplingBridge:
    """Fresh model for interfaces/research; verified fold initialization is external.

    Formal runs use prepare.py/train.py to enforce asset identity and fit scales.
    This function deliberately never guesses asset paths or loads old weights.
    """
    plan = profiles()
    if profile not in plan["profiles"]:
        raise ValueError(f"unknown network profile: {profile}")
    bridge = EBJDModel(goal_scale=goal_scale, bridge_scale=bridge_scale,
                       **plan["shared"], **plan["profiles"][profile])
    model = UnifiedCouplingBridge(
        bridge, worlds=20, singleton_backward=plan["coupling"]["singleton_backward"])
    model.training_options = {"coarse": 0.25, "geometry": 0.05, "map": 0.1}
    return model


@torch.no_grad()
def predict_g_h(model, batch, proposals, initial_noise):
    """Paired 20-world/20-step Raw inference; inputs and outputs are metres.

    Caller supplies identity-bound proposals and explicit initial noise.
    SDD paper metrics need the scene's paper_metric_scale after this call.
    """
    if model.training:
        raise ValueError("call model.eval() before frozen inference")
    b, n = batch.valid.shape
    if proposals.shape != (b, 20, n, 2) or initial_noise.shape != (b, 20, n, 12, 2):
        raise ValueError("expected 20 proposals and 12-slot noise with matching scene/agent axes")
    if not torch.isfinite(proposals).all() or not torch.isfinite(initial_noise).all():
        raise ValueError("non-finite proposal/noise")
    context = model.encode_context(batch.observed, batch.semantic_maps, batch.valid)
    coupled, permutation, _ = model.coupling.assign(
        context, proposals, model.bridge.representation.goal_scale, straight_through=False)
    g, _, _ = differentiable_sample(
        model.bridge, context, initial_noise, steps=20,
        proposal_goals=proposals, proposal_eta=0.0)
    h, _, _ = differentiable_sample(
        model.bridge, context, gather_candidates(initial_noise, permutation), steps=20,
        proposal_goals=coupled, proposal_eta=0.0)
    if not torch.equal(g[..., -1, :], proposals) or not torch.equal(h[..., -1, :], coupled):
        raise RuntimeError("literal endpoint invariant failed")
    return {"G": g, "H": h, "G_goals": proposals, "H_goals": coupled,
            "permutation": permutation}

"""Occurrence-aware thin adapter over unchanged native G/H DDIM inference."""
import torch
from ebjd.unified import gather_candidates
from bcep_joint.inference import generate as native_generate
from .identity import canonical_order


class _CouplingView:
    def __init__(self, coupling, ids):
        self.original, self.ids = coupling, ids

    def assign(self, *args, **kwargs):
        return self.original.assign(*args, **kwargs, candidate_ids=self.ids)

    def __getattr__(self, name):
        return getattr(self.original, name)


class IdentityBoundModel:
    """No parameters/copies: bind occurrence metadata to one native objective call.

    This view can also be passed to bcep_v2.objective.bridge_and_pair_loss. The
    immutable IDs must already have followed input gathers and augmentation.
    """
    def __init__(self, model, ids):
        self.original = model
        self.coupling = _CouplingView(model.coupling, ids)

    def __getattr__(self, name):
        return getattr(self.original, name)

    def eval(self):
        self.original.eval()
        return self


@torch.no_grad()
def generate(model, observed, maps, valid, scene_ids, agent_ids, goals, seed,
             *, candidate_ids, mode='GH'):
    """Recover ORIGINAL bank slots before native G grouping/keyed-noise creation.

    G is the original bank combination, NOT combinations of independently
    relabelled array slots. H indices are returned in the caller's input order.
    Neither the original key(scene, agent, slot, seed) nor DDIM is changed.
    """
    canonical_order(goals, valid, candidate_ids)  # shape/finite/identity validation
    original_to_input = candidate_ids.argsort(-1)
    restored = gather_candidates(goals, original_to_input.transpose(1, 2))
    slots = torch.arange(20, device=goals.device).expand_as(candidate_ids)
    view = IdentityBoundModel(model, slots)
    g, h, pi, checks = native_generate(view, observed, maps, valid, scene_ids,
                                     agent_ids, restored, seed, mode=mode)
    input_pi = original_to_input.gather(2, pi.transpose(1, 2)).transpose(1, 2)
    assert torch.equal(gather_candidates(goals, input_pi), gather_candidates(restored, pi))
    return g, h, input_pi, dict(checks, original_bank_G_restored=True,
                                occurrence_noise_keys_unchanged=True)

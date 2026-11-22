"""Choose one whole-window permutation before exactly one native DDIM call."""
import torch
from bcep_joint.inference import keyed_noise, check_permutation, differentiable_sample
from ebjd.unified import gather_candidates
from .model import choose_initial


@torch.no_grad()
def paired_context(model, observed, maps, valid, goals):
    model.eval()
    context = model.encode_context(observed, maps, valid)
    h, hp, _, ip = model.coupling.assign(context, goals, model.bridge.representation.goal_scale,
                                        straight_through=False, return_initial=True)
    check_permutation(ip, valid)
    check_permutation(hp, valid)
    assert torch.equal(h, gather_candidates(goals, hp))
    return context, ip, hp


@torch.no_grad()
def decode(model, context, goals, noise, permutation):
    endpoints = gather_candidates(goals, permutation)
    paired_noise = gather_candidates(noise, permutation)
    # This checks whole trajectory-noise vectors, not merely their marginal moments.
    inverse = permutation.argsort(1)
    assert torch.equal(gather_candidates(endpoints, inverse), goals)
    assert torch.equal(gather_candidates(paired_noise, inverse), noise)
    path, _, _ = differentiable_sample(model.bridge, context, paired_noise, 20,
                                       proposal_goals=endpoints, proposal_eta=0.)
    assert torch.isfinite(path).all()
    torch.testing.assert_close(path[..., -1, :], endpoints, rtol=1e-6, atol=1e-6)
    return path


@torch.no_grad()
def generate(model, selector, observed, maps, valid, scene_ids, agent_ids, goals, seed=2035, force=None):
    """No future argument exists. force is an offline control, never a learned input."""
    assert seed == 2035 and force in (None, 'I', 'H0')
    context, ip, hp = paired_context(model, observed, maps, valid, goals)
    if force is None:
        selector.eval()
        delta = selector.difference(observed, context.social_h, gather_candidates(goals, ip),
                                    gather_candidates(goals, hp), valid)
        switch = choose_initial(delta, ip, hp, valid)
    else:
        delta = observed.new_zeros(len(valid), 2)
        switch = torch.full((len(valid),), force == 'I', device=valid.device, dtype=torch.bool)
    pi = torch.where(switch[:, None, None], ip, hp)
    # Noise is keyed once in the original bank namespace, then gathered with endpoints.
    noise = keyed_noise(scene_ids, agent_ids, valid, seed).to(observed.device)
    path = decode(model, context, goals, noise, pi)
    return path, dict(switch=switch, delta=delta, initial_pi=ip, feedback_pi=hp, selected_pi=pi,
                      ddim_calls=1, ddim_steps=20)

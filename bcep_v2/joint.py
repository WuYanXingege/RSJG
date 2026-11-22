"""One assignment-conditioned interaction round, followed by the original bridge."""

import math
import torch
from torch import nn
from torch.nn import functional as F
from ebjd.unified import (
    LearnableJointCouplingHead,
    gather_candidates,
    hard_assignment_indices,
    log_sinkhorn,
    permutation_matrix,
)
from bcep_joint.model import build as build_legacy
from .runtime import bilinear_zeros, pool_map


def neighbors(observed, valid):
    if not valid.any(1).all():
        raise ValueError("empty scene")
    position = observed[:, :, -1].float()
    distance = (position[:, :, None] - position[:, None, :]).square().sum(-1)
    legal = (
        valid[:, :, None]
        & valid[:, None, :]
        & ~torch.eye(valid.shape[1], device=valid.device, dtype=torch.bool)
    )
    distance = distance.masked_fill(~legal, torch.inf)
    # Only the distance at the boundary matters, including every tied agent.
    # CUDA kthvalue also returns an index and is unavailable in strict deterministic mode.
    k = min(8, valid.shape[1])
    boundary = distance.sort(dim=-1).values[..., k - 1 : k]
    return (legal & (distance <= boundary)).nonzero().unbind(1)


def assignment(scores, valid, training):
    indices = hard_assignment_indices(scores, valid)
    hard = permutation_matrix(indices, scores.shape[-1])
    if not training:
        return indices, hard, hard
    identity = torch.eye(scores.shape[-1], device=scores.device)
    relaxed = log_sinkhorn(scores.float(), 20)
    movable = valid & (valid.sum(1, keepdim=True) > 1)
    soft = torch.where(movable[..., None, None], relaxed, identity)
    return indices, hard + (soft - soft.detach()), soft


class FeedbackCoupling(LearnableJointCouplingHead):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(192, 24)
        self.compatibility = nn.Sequential(nn.Linear(54, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.compatibility[-1].weight)
        nn.init.zeros_(self.compatibility[-1].bias)

    def messages(self, features, observed, proposals, valid, initial):
        batch, agents, worlds, _ = features.shape
        scene, receiver, sender = neighbors(observed, valid)
        output = features.new_zeros(batch * agents, worlds, worlds, dtype=torch.float32)
        self.last_edge_count = scene.numel()
        compact = self.project(features).float()
        positions = proposals.transpose(1, 2).float()
        last = observed[:, :, -1]
        velocity = (last - observed[:, :, -4]) / 1.2
        degree = torch.bincount(scene * agents + receiver, minlength=batch * agents)
        for start in range(0, scene.numel(), 128):
            edge = slice(start, start + 128)
            b, i, j = scene[edge], receiver[edge], sender[edge]
            relative = torch.cat((last[b, j] - last[b, i], velocity[b, j] - velocity[b, i]), -1)
            pair = torch.cat(
                (
                    compact[b, i, :, None].expand(-1, -1, worlds, -1),
                    compact[b, j, None, :].expand(-1, worlds, -1, -1),
                    relative[:, None, None].expand(-1, worlds, worlds, -1),
                    positions[b, j, None, :] - positions[b, i, :, None],
                ),
                -1,
            )
            compatibility = self.compatibility(pair.float()).squeeze(-1)
            msg = torch.bmm(initial[b, j].float(), compatibility.transpose(1, 2).float())
            output = output.index_add(0, b * agents + i, msg)
        return (output / degree.clamp_min(1)[:, None, None]).reshape(batch, agents, worlds, worlds)

    def assign(self, context, proposals, goal_scale, *, straight_through, return_initial=False):
        valid = context.valid
        if not valid.any(1).all():
            raise ValueError("empty scene")
        proposals = proposals.detach()
        safe = torch.where(valid[:, None, :, None], proposals, 0.0)
        features = []
        handle = self.candidate_norm.register_forward_hook(
            lambda mod, args, value: features.append(value)
        )
        try:
            scores = super().forward(context, safe, goal_scale)
        finally:
            handle.remove()
        initial_indices, initial, _ = assignment(scores, valid, straight_through)
        scores = scores + self.messages(features[0], context.observed, safe, valid, initial)
        indices, _, soft = assignment(scores, valid, straight_through)
        hard = gather_candidates(proposals, indices)
        if straight_through:
            surrogate = torch.einsum("bnkj,bnjd->bknd", soft.float(), safe.transpose(1, 2).float())
            hard = hard + (surrogate - surrogate.detach()).to(hard)
        if return_initial:
            # Both permutations belong to this exact forward. G is not I.
            return hard, indices, scores, initial_indices
        return hard, indices, scores


def build_joint(config, legacy_state=None):
    model = build_legacy(config)
    model.bridge.denoiser.map_sampler = bilinear_zeros
    model.bridge.encoder.map_pool = pool_map
    if legacy_state is not None:
        model.load_state_dict(legacy_state, strict=True)
    original = model.coupling
    model.coupling = FeedbackCoupling()
    for name, module in original.named_children():
        getattr(model.coupling, name).load_state_dict(module.state_dict(), strict=True)
    with torch.no_grad():
        model.coupling.world_queries.copy_(original.world_queries)
    return model


def pair_loss(coupled, truth, valid):
    if not valid.any(1).all():
        raise ValueError("empty scene")
    difference = torch.where(valid[:, None, :, None], coupled.float() - truth[:, None].float(), 0.0)
    world_error = difference.norm(dim=-1).sum(-1) / valid.sum(1)[:, None]
    logits = -world_error / 0.1
    loss = (-0.1 * (logits.logsumexp(1) - math.log(coupled.shape[1]))).mean()
    log_weights = logits.detach().log_softmax(1)
    return loss, dict(
        pair_softmin_m=loss.detach(),
        pair_exact_hard_jfde_m=world_error.detach().amin(1).mean(),
        world_weight_entropy=-(log_weights.exp() * log_weights).sum(1).mean(),
        assignment_temperature=1.0,
        pair_temperature_m=0.1,
        pair_weight=1.0,
    )


def clean_path_terms(path, future, valid, signal_squared):
    if signal_squared.numel() != len(valid) or not valid.any(1).all():
        raise ValueError("one signal weight required per nonempty scene")
    delta = torch.where(
        valid[:, :, None, None], path[:, 0, :, :-1].float() - future.detach()[:, :, :-1].float(), 0.0
    )
    error = delta.norm(dim=-1)
    penalty = F.smooth_l1_loss(error, torch.zeros_like(error), reduction="none", beta=0.5)
    per_scene = penalty.sum((1, 2)) / (valid.sum(1) * error.shape[-1])
    weighted = signal_squared.float().reshape_as(per_scene) * per_scene
    return weighted.mean(), per_scene.detach().sum(), weighted.detach().sum()


def clean_path_loss(path, future, valid, signal_squared):
    return clean_path_terms(path, future, valid, signal_squared)[0]

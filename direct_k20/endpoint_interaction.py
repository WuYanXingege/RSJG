"""One synchronous interaction update of native, ordered K20 endpoints."""

import torch
from torch import nn

from bcep.motion_frame import frame
from .model import FRAMES, K, encode_observed, student_endpoints


NEIGHBORS = 4
WIDTH = 64
CONTEXT_WIDTH = 128
TC_EPS = 1e-8
RELATIVE_STOP_SPEED = 1e-3


def _nearest(distance, sender_order, count):
    # A stable sort resolves equal distances using geometric sender content.
    # Exact duplicate keys have identical messages. The shared secondary
    # order makes both selections take prefixes of any such tied group.
    order = sender_order[:, None].expand_as(distance)
    positions = distance.gather(-1, order).argsort(dim=-1, stable=True)
    return order.gather(-1, positions[..., :count])


def _sender_order(position, velocity, goals):
    """Break distance ties with sender geometry from the current slot."""
    known = torch.cat((position, velocity), dim=-1).expand(K, -1, -1)
    keys = torch.cat((known, goals), dim=-1)
    order = torch.arange(len(position), device=position.device).expand(K, -1)
    for coordinate in reversed(keys.unbind(dim=-1)):
        positions = coordinate.gather(-1, order).argsort(dim=-1, stable=True)
        order = order.gather(-1, positions)
    return order


class EndpointInteraction(nn.Module):
    """Refine each joint proposal with sparse, slot-local social messages."""

    def __init__(self):
        super().__init__()
        self.motion = nn.Sequential(nn.Linear(CONTEXT_WIDTH + 6, WIDTH), nn.SiLU(),
                                    nn.Linear(WIDTH, 2))
        self.pair = nn.Sequential(nn.Linear(8, WIDTH), nn.SiLU())
        self.pair_displacement = nn.Linear(WIDTH, 2)
        self.pair_logit = nn.Linear(WIDTH, 1)
        for head in (self.motion[-1], self.pair_displacement):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _social(self, goals, position, velocity, rotation, valid, duration):
        n = len(position)
        count = min(NEIGHBORS, n - 1)
        if not count:
            return torch.zeros_like(goals)

        position_delta = position[None, :] - position[:, None]
        proposal_velocity = (goals - position[None]) / duration
        proposal_delta = proposal_velocity[:, None, :] - proposal_velocity[:, :, None]
        speed_squared = proposal_delta.square().sum(dim=-1)
        tc = -(position_delta * proposal_delta).sum(dim=-1) / (speed_squared + TC_EPS)
        tc = torch.where(speed_squared > RELATIVE_STOP_SPEED**2, tc.clamp(0., duration), 0.)
        dc = (position_delta + tc.unsqueeze(-1) * proposal_delta).norm(dim=-1)

        permitted = valid[:, None] & valid[None, :] & ~torch.eye(n, dtype=torch.bool, device=valid.device)
        with torch.no_grad():
            sender_order = _sender_order(position, velocity, goals)
            current_distance = position_delta.norm(dim=-1).masked_fill(~permitted, torch.inf)
            current = _nearest(current_distance.expand(K, -1, -1), sender_order, count)
            closest = _nearest(dc.masked_fill(~permitted, torch.inf), sender_order, count)
            duplicates = (closest.unsqueeze(-1) == current.unsqueeze(-2)).any(dim=-1)
            candidates = torch.cat((current, closest), dim=-1)
            allowed = permitted.expand(K, -1, -1).gather(-1, candidates)
            keep = allowed & torch.cat((torch.ones_like(duplicates), ~duplicates), dim=-1)
            slot, receiver, rank = keep.nonzero(as_tuple=True)
            sender = candidates[slot, receiver, rank]

        # Learn embeddings only for retained edges, including in dense scenes.
        relative = torch.stack((
            position_delta[receiver, sender],
            velocity[sender] - velocity[receiver],
            goals[slot, sender] - goals[slot, receiver],
        ), dim=1)
        local = (relative @ rotation[receiver]).flatten(1)
        features = torch.cat((local, tc[slot, receiver, sender, None] / duration,
                              dc[slot, receiver, sender, None]), dim=-1)
        hidden = self.pair(features)
        messages = self.pair_displacement(hidden)
        logits = self.pair_logit(hidden).squeeze(-1)
        slots = candidates.shape[-1]
        flat_edge = (slot * n + receiver) * slots + rank
        dense_messages = goals.new_zeros((K * n * slots, 2)).index_copy(0, flat_edge, messages)
        dense_logits = goals.new_full((K * n * slots,), -torch.inf).index_copy(0, flat_edge, logits)
        dense_logits = dense_logits.reshape(K, n, slots)
        # The extra attention entry has fixed logit zero and displacement zero.
        null = goals.new_zeros((K, n, 1))
        weight = torch.cat((dense_logits, null), dim=-1).softmax(dim=-1)[..., :slots]
        return (weight[..., None] * dense_messages.reshape(K, n, slots, 2)).sum(dim=2)

    def forward(self, goals, observed, context, valid, *, dt=.4, return_components=False):
        n = len(observed)
        if (observed.shape != (n, 8, 2) or goals.shape != (K, n, 2)
                or context.shape != (n, CONTEXT_WIDTH) or valid.shape != (n,)
                or valid.dtype != torch.bool or n < 1 or dt <= 0):
            raise ValueError('expected K20 goals, eight observed positions and aligned valid agents')
        observation = torch.where(valid[:, None, None], observed, 0.)
        safe_goals = torch.where(valid[None, :, None], goals, 0.)
        context = torch.where(valid[:, None], context, 0.)
        _, rotation, _ = frame(observation, FRAMES, dt, observed.new_ones(()))
        velocity_history = observation.diff(dim=1) / dt
        velocity = velocity_history[:, -3:].mean(dim=1)
        trend = (velocity - velocity_history[:, -6:-3].mean(dim=1)) / (3 * dt)
        position = observation[:, -1]
        motion = (torch.stack((velocity, trend), dim=1) @ rotation).flatten(1)
        proposal = ((safe_goals - position).unsqueeze(-2) @ rotation).squeeze(-2)
        history = torch.cat((context, motion), dim=-1).expand(K, -1, -1)
        unary = self.motion(torch.cat((history, proposal), dim=-1))
        unary = torch.where(valid[None, :, None], unary, 0.)
        social = self._social(safe_goals, position, velocity, rotation, valid, FRAMES * dt)
        update = (rotation @ (unary + social).unsqueeze(-1)).squeeze(-1)
        refined = goals + update
        if return_components:
            return refined, unary, social
        return refined


def interaction_goals(assets, student, refiner, batch):
    """Return native and refined goals with gradients confined to the refiner."""
    observed, maps, valid = batch.observed[0], batch.semantic_maps[0], batch.valid[0]
    with torch.no_grad():
        context, features = encode_observed(assets.provider, batch)
        original = student_endpoints(student, assets.provider, observed, maps, valid, context, features)
    refined = (original if refiner is None else
               refiner(original, observed, context, valid, dt=float(assets.split['dt'])))
    return original, refined

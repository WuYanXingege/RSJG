"""GT-blind joint endpoint assembly for the EBJD V2 anchor representation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from .encoders import pairwise_observed_geometry

if TYPE_CHECKING:
    from .encoders import EBJDContext


@dataclass
class JointGoalDiagnostics:
    initial_energy: torch.Tensor
    final_energy: torch.Tensor
    changed_world_fraction: torch.Tensor
    edge_count: torch.Tensor


def _safe_norm(value: torch.Tensor) -> torch.Tensor:
    return value.square().sum(dim=-1).add(1e-6).sqrt()


class JointGoalAssembler(nn.Module):
    """Learn a symmetric pair energy and assemble one permutation per agent.

    Candidate coordinates are inputs, never trainable parameters.  The
    auxiliary association loss detaches the history/social summary so its
    gradients update only this module's projection and energy head.
    """

    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.social_projection = nn.Linear(dim, 16)
        self.pair_mlp = nn.Sequential(
            nn.Linear(50, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.pair_mlp[-1].weight)
        nn.init.zeros_(self.pair_mlp[-1].bias)

    @staticmethod
    def edge_mask(context: "EBJDContext") -> torch.Tensor:
        position = context.observed.float()[..., -1, :]
        velocity = context.mean_velocity.float()
        rel_p = position[:, :, None] - position[:, None, :]
        rel_v = velocity[:, :, None] - velocity[:, None, :]
        distance = _safe_norm(rel_p)
        cpa_time = (-(rel_p * rel_v).sum(-1) /
                    rel_v.square().sum(-1).clamp_min(1e-6)).clamp(0, 4.8)
        cpa_distance = _safe_norm(rel_p + cpa_time[..., None] * rel_v)
        valid = context.valid[:, :, None] & context.valid[:, None, :]
        eye = torch.eye(
            valid.shape[-1], dtype=torch.bool, device=valid.device)[None]
        return valid & ~eye & ((distance <= 6.0) | (cpa_distance <= 2.0))

    @staticmethod
    def _target_geometry(
        goals: torch.Tensor, origins: torch.Tensor, scale: torch.Tensor,
    ) -> torch.Tensor:
        """Return ordered target relation features ``[B,P,N,N,8]``."""
        goal_i = goals[:, :, :, None, :]
        goal_j = goals[:, :, None, :, :]
        rel_goal = goal_i - goal_j
        goal_distance = _safe_norm(rel_goal)
        origin_i = origins[:, None, :, None, :]
        origin_j = origins[:, None, None, :, :]
        direction_i = goal_i - origin_i
        direction_j = goal_j - origin_j
        rel_origin = origin_i - origin_j
        rel_direction = direction_i - direction_j
        closest_fraction = (-(rel_origin * rel_direction).sum(-1) /
                            rel_direction.square().sum(-1).clamp_min(1e-6)).clamp(0, 1)
        closest = rel_origin + closest_fraction[..., None] * rel_direction
        closest_distance = _safe_norm(closest)
        length_i = _safe_norm(direction_i).expand_as(goal_distance)
        length_j = _safe_norm(direction_j).expand_as(goal_distance)
        cosine = ((direction_i * direction_j).sum(-1) /
                  (length_i * length_j).clamp_min(1e-6))
        scale = scale.float().clamp_min(0.5)
        speed_scale = (scale / 4.8).clamp_min(1e-6)
        return torch.cat((
            rel_goal / scale,
            (goal_distance / scale)[..., None],
            (closest_distance / scale)[..., None],
            closest_fraction[..., None],
            (length_i / 4.8 / speed_scale)[..., None],
            (length_j / 4.8 / speed_scale)[..., None],
            cosine[..., None],
        ), dim=-1).clamp(-20, 20)

    def pair_energy(
        self, context: "EBJDContext", goals: torch.Tensor,
    ) -> torch.Tensor:
        """Return symmetric ordered energy ``[B,P,N,N]``."""
        expected = (
            context.observed.shape[0], context.observed.shape[1], 2)
        if goals.ndim != 4 or goals.shape[0] != expected[0] or \
                goals.shape[2:] != expected[1:]:
            raise ValueError("goals must have shape [B,P,N,2]")
        batch, worlds, agents = goals.shape[:3]
        social = self.social_projection(context.social_h.detach())
        si = social[:, None, :, None, :].expand(batch, worlds, agents, agents, 16)
        sj = social[:, None, None, :, :].expand_as(si)
        observed = pairwise_observed_geometry(
            context.observed, context.mean_velocity,
            context.goal_scale).detach()
        observed = observed[:, None].expand(batch, worlds, agents, agents, 10)
        target = self._target_geometry(
            goals.float(), context.observed.float()[..., -1, :],
            context.goal_scale)
        directed = self.pair_mlp(torch.cat((si, sj, observed, target), -1)).squeeze(-1)
        return 0.5 * (directed + directed.transpose(-1, -2))

    def scene_energy(
        self, context: "EBJDContext", goals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        energy = self.pair_energy(context, goals)
        edges = self.edge_mask(context)
        # Count each undirected edge once.
        upper = torch.triu(edges, diagonal=1)
        count = upper.sum(dim=(-1, -2))
        total = (energy * upper[:, None]).sum(dim=(-1, -2))
        mean = total / count[:, None].clamp_min(1)
        mean = torch.where(count[:, None] > 0, mean, torch.zeros_like(mean))
        return mean, count

    @staticmethod
    def training_anchor_labels(
        native_goals: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
    ) -> torch.Tensor:
        if native_goals.shape[:2] != target.shape[:2] or native_goals.shape[-2:] != (20, 2):
            raise ValueError("native goals and targets have incompatible shapes")
        distance = torch.linalg.vector_norm(
            native_goals.float() - target.float()[..., -1, None, :], dim=-1)
        labels = distance.argmin(-1)
        return torch.where(valid, labels, torch.zeros_like(labels))

    @staticmethod
    def anchors_from_labels(
        native_goals: torch.Tensor, labels: torch.Tensor,
    ) -> torch.Tensor:
        return native_goals.gather(
            2, labels[..., None, None].expand(*labels.shape, 1, 2)).squeeze(2)

    def association_loss(
        self, context: "EBJDContext", target: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if context.native_goals is None:
            raise ValueError("association loss requires native goals")
        labels = self.training_anchor_labels(
            context.native_goals, target, context.valid)
        batch, agents = labels.shape
        combinations = labels[:, None].expand(batch, 20, agents).clone()
        # The dedicated CPU generator is independent from data/diffusion RNG.
        for scene in range(batch):
            valid_indices = context.valid[scene].nonzero(as_tuple=False).flatten().cpu()
            for negative in range(1, 20):
                while True:
                    proposal = torch.randint(
                        0, 20, (len(valid_indices),), generator=generator)
                    positive = labels[scene, valid_indices.to(labels.device)].cpu()
                    if not torch.equal(proposal, positive):
                        break
                combinations[scene, negative, valid_indices.to(labels.device)] = proposal.to(
                    labels.device)
        candidates = context.native_goals[:, None].expand(batch, 20, agents, 20, 2)
        anchors = candidates.gather(
            3, combinations[..., None, None].expand(batch, 20, agents, 1, 2)).squeeze(3)
        score, edge_count = self.scene_energy(context, anchors)
        loss = F.cross_entropy(-score, torch.zeros(
            batch, dtype=torch.long, device=score.device), reduction="none")
        active = edge_count > 0
        return loss[active].mean() if active.any() else score.sum() * 0

    @staticmethod
    def _permutation(seed: int, scene_id: str, agent_id: str) -> torch.Tensor:
        digest = hashlib.sha256(
            f"{seed}|{scene_id}|{agent_id}".encode("utf-8")).digest()
        generator = torch.Generator(device="cpu").manual_seed(
            int.from_bytes(digest[:8], "little") % (2**63 - 1))
        return torch.randperm(20, generator=generator)

    @torch.no_grad()
    def assemble(
        self, context: "EBJDContext", scene_ids: list[str],
        agent_ids: list[list[str]], seed: int, rounds: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor, JointGoalDiagnostics]:
        if context.native_goals is None:
            raise ValueError("joint goal assembly requires native goals")
        batch, agents, candidates, _ = context.native_goals.shape
        if candidates != 20 or len(scene_ids) != batch or len(agent_ids) != batch:
            raise ValueError("joint goal assembly identity/candidate dimensions differ")
        permutations = torch.zeros(
            batch, agents, 20, dtype=torch.long,
            device=context.native_goals.device)
        for scene in range(batch):
            valid_count = int(context.valid[scene].sum())
            if len(agent_ids[scene]) != valid_count:
                raise ValueError("agent identity count differs from valid mask")
            for agent in range(valid_count):
                permutations[scene, agent] = self._permutation(
                    seed, scene_ids[scene], agent_ids[scene][agent]).to(
                        permutations.device)
        def materialize() -> torch.Tensor:
            goals = context.native_goals[:, None].expand(batch, 20, agents, 20, 2)
            index = permutations.permute(0, 2, 1)[..., None, None].expand(
                batch, 20, agents, 1, 2)
            return goals.gather(3, index).squeeze(3)

        initial = materialize()
        initial_energy, edge_count = self.scene_energy(context, initial)
        edge_mask = self.edge_mask(context)
        for _ in range(rounds):
            for scene in range(batch):
                valid_count = int(context.valid[scene].sum())
                if valid_count <= 1 or int(edge_count[scene]) == 0:
                    continue
                for agent in range(valid_count):
                    neighbors = edge_mask[scene, agent, :valid_count].nonzero(
                        as_tuple=False).flatten()
                    if not len(neighbors):
                        continue
                    current = materialize()[scene:scene + 1]
                    # [world, choice, agent, xy].  Other agents retain their
                    # current world assignments while the selected agent tries
                    # each of its 20 candidates in each world.
                    proposal = current[0, :, None].expand(
                        20, 20, agents, 2).clone()
                    proposal[:, :, agent] = context.native_goals[
                        scene, agent][None].expand(20, 20, 2)
                    pair = self.pair_energy(
                        _slice_context(context, scene),
                        proposal.reshape(1, 400, agents, 2))
                    cost = pair[0, :, agent, neighbors].sum(-1).reshape(20, 20)
                    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
                    candidate_perm = torch.empty(20, dtype=torch.long, device=cost.device)
                    candidate_perm[torch.as_tensor(rows, device=cost.device)] = torch.as_tensor(
                        cols, device=cost.device)
                    old = permutations[scene, agent].clone()
                    permutations[scene, agent] = candidate_perm
                    before = self.scene_energy(
                        _slice_context(context, scene), current)[0].sum()
                    after_goals = materialize()[scene:scene + 1]
                    after = self.scene_energy(
                        _slice_context(context, scene), after_goals)[0].sum()
                    if not bool(after < before - 1e-12):
                        permutations[scene, agent] = old
        final = materialize()
        final_energy, _ = self.scene_energy(context, final)
        changed = (final != initial).any(-1).float()
        changed = (changed * context.valid[:, None]).sum((1, 2)) / (
            20 * context.valid.sum(1).clamp_min(1))
        diagnostics = JointGoalDiagnostics(
            initial_energy.mean(1), final_energy.mean(1), changed, edge_count)
        return final, permutations, diagnostics


def _slice_context(context: "EBJDContext", index: int) -> "EBJDContext":
    """Slice a context without importing its class at module import time."""
    from dataclasses import replace
    values = {}
    for name in context.__dataclass_fields__:
        value = getattr(context, name)
        if torch.is_tensor(value) and value.ndim and value.shape[0] == context.valid.shape[0]:
            values[name] = value[index:index + 1]
    return replace(context, **values)

"""Single-stage frozen-prior, learnable-coupling conditional EBJD."""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from .encoders import EBJDContext
from .model import EBJDModel


class FrozenNativeGoalProvider(nn.Module):
    """Parameter-free identity-bound provider wrapper kept permanently in eval."""

    def __init__(self, bank) -> None:
        super().__init__()
        self.bank = bank
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)

    def forward(self, batch) -> torch.Tensor:
        return self.bank.batch(batch)


def log_sinkhorn(scores: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    """Return a doubly stochastic relaxation of [...,K,K] scores."""
    if scores.ndim < 2 or scores.shape[-1] != scores.shape[-2]:
        raise ValueError("scores must end in a square assignment matrix")
    log_p = scores.float()
    for _ in range(iterations):
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
        log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
    return log_p.exp()


def hard_assignment_indices(scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Maximum-score Hungarian indices [B,K,N], identity for N=1/padding."""
    if scores.ndim != 4 or scores.shape[1] != valid.shape[1]:
        raise ValueError("scores/valid dimensions differ")
    batch, agents, worlds, candidates = scores.shape
    if worlds != candidates:
        raise ValueError("assignment must be square")
    identity = torch.arange(worlds, device=scores.device)
    result = identity.view(1, worlds, 1).expand(batch, worlds, agents).clone()
    detached = scores.detach().float().cpu().numpy()
    for scene in range(batch):
        active = valid[scene].nonzero(as_tuple=False).flatten().tolist()
        if len(active) <= 1:
            continue
        for agent in active:
            row, column = linear_sum_assignment(-detached[scene, agent])
            ordered = np.empty(worlds, dtype=np.int64)
            ordered[row] = column
            result[scene, :, agent] = torch.as_tensor(ordered, device=scores.device)
    return result


def permutation_matrix(indices: torch.Tensor, candidates: int) -> torch.Tensor:
    """Convert [B,K,N] gather indices to [B,N,K,K]."""
    return torch.nn.functional.one_hot(
        indices.permute(0, 2, 1), num_classes=candidates).float()


def gather_candidates(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather a [B,K,N,...] tensor independently for every agent."""
    if values.shape[:3] != indices.shape:
        raise ValueError("values and assignment indices must share [B,K,N]")
    view = indices
    for _ in values.shape[3:]:
        view = view.unsqueeze(-1)
    view = view.expand(*indices.shape, *values.shape[3:])
    return torch.gather(values, 1, view)


class LearnableJointCouplingHead(nn.Module):
    """Candidate-order-equivariant scene coupling with shared world queries."""

    def __init__(self, dim: int = 192, heads: int = 4, worlds: int = 20) -> None:
        super().__init__()
        self.worlds = int(worlds)
        self.endpoint = nn.Sequential(
            nn.Linear(2, dim), nn.LayerNorm(dim), nn.SiLU(), nn.Linear(dim, dim))
        self.candidate_norm = nn.LayerNorm(dim)
        self.world_queries = nn.Parameter(torch.empty(worlds, dim))
        nn.init.normal_(self.world_queries, std=0.02)
        self.scene_cross_attention = nn.MultiheadAttention(
            dim, heads, batch_first=True, dropout=0.0)
        self.score = nn.Sequential(
            nn.Linear(3 * dim, dim), nn.SiLU(), nn.Linear(dim, 1))

    def forward(
        self, context: EBJDContext, proposals: torch.Tensor, goal_scale: torch.Tensor,
    ) -> torch.Tensor:
        if proposals.ndim != 4 or proposals.shape[1] != self.worlds:
            raise ValueError("proposals must be [B,20,N,2]")
        batch, worlds, agents, _ = proposals.shape
        if context.social_h.shape[:2] != (batch, agents):
            raise ValueError("proposal/context agent dimensions differ")
        relative = (
            proposals - context.observed[:, None, :, -1, :]
        ) / goal_scale.float().clamp_min(0.5)
        candidate = self.endpoint(relative.float()).permute(0, 2, 1, 3)
        candidate = self.candidate_norm(candidate + context.social_h[:, :, None])
        tokens = candidate.reshape(batch, agents * worlds, -1)
        token_valid = context.valid[:, :, None].expand(batch, agents, worlds).reshape(batch, -1)
        query = self.world_queries[None].expand(batch, -1, -1)
        world, _ = self.scene_cross_attention(
            query, tokens, tokens, key_padding_mask=~token_valid, need_weights=False)
        world = world[:, None, :, None].expand(-1, agents, -1, worlds, -1)
        cand = candidate[:, :, None].expand(-1, -1, worlds, -1, -1)
        person = context.social_h[:, :, None, None].expand(
            -1, -1, worlds, worlds, -1)
        scores = self.score(torch.cat((world, cand, person), dim=-1)).squeeze(-1)
        return scores.masked_fill(~context.valid[:, :, None, None], 0.0)

    def assign(
        self, context: EBJDContext, proposals: torch.Tensor,
        goal_scale: torch.Tensor, *, straight_through: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores = self(context, proposals, goal_scale)
        indices = hard_assignment_indices(scores, context.valid)
        if not straight_through:
            return gather_candidates(proposals, indices), indices, scores
        hard = permutation_matrix(indices, self.worlds).to(scores)
        soft = log_sinkhorn(scores, iterations=20)
        valid = context.valid[:, :, None, None]
        identity = torch.eye(self.worlds, device=scores.device)[None, None]
        soft = torch.where(valid, soft, identity)
        st = hard + soft - soft.detach()
        candidate = proposals.permute(0, 2, 1, 3)
        coupled = torch.einsum("bnkj,bnjd->bknd", st, candidate)
        return coupled, indices, scores


def joint_pair_loss(
    coupled: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor,
    goal_scale: torch.Tensor, temperature: float = 0.1,
) -> torch.Tensor:
    distance = torch.linalg.vector_norm(
        coupled.float() - truth[:, None].float(), dim=-1)
    distance = distance / goal_scale.float().clamp_min(0.5)
    mask = valid[:, None].float()
    world_error = (distance * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    worlds = coupled.shape[1]
    return (-temperature * (
        torch.logsumexp(-world_error / temperature, dim=1)
        - torch.log(torch.tensor(float(worlds), device=coupled.device))
    )).mean()


class UnifiedCouplingBridge(nn.Module):
    """One checkpoint containing shared MSCE, coupling and conditional bridge."""

    def __init__(self, bridge: EBJDModel, worlds: int = 20) -> None:
        super().__init__()
        if not bridge.denoiser.proposal_conditioned:
            raise ValueError("unified bridge must use proposal conditioning")
        self.bridge = bridge
        self.coupling = LearnableJointCouplingHead(
            dim=bridge.encoder.history_stem[0].out_features,
            heads=bridge.encoder.history_social[0].attn.heads,
            worlds=worlds)

    def encode_context(self, *args, **kwargs) -> EBJDContext:
        return self.bridge.encode_context(*args, **kwargs)

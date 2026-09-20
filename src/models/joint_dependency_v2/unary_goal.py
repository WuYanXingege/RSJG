"""Zero-initialized unary goal residual."""

from typing import Dict, Optional

import torch
from torch import nn

from src.models.joint_goal import sanitize_candidate_mask


class UnaryGoalResidual(nn.Module):
    """Score candidate endpoints while preserving the frozen prior at init."""

    def __init__(self, agent_dim: int = 128,
                 prior_temperature: float = 1.0,
                 use_scene_latent: bool = True) -> None:
        super().__init__()
        if agent_dim != 128:
            raise ValueError("JDV2 requires agent_dim=128")
        if prior_temperature <= 0:
            raise ValueError("prior_temperature must be positive")
        self.prior_temperature = float(prior_temperature)
        self.use_scene_latent = bool(use_scene_latent)
        self.goal_encoder = nn.Sequential(
            nn.Linear(4, 64), nn.SiLU(), nn.Linear(64, 64),
            nn.LayerNorm(64))
        self.agent_projection = nn.Linear(128, 64)
        if self.use_scene_latent:
            self.scene_embedding = nn.Embedding(4, 64)
        self.context_norm = nn.LayerNorm(64)
        self.residual_head = nn.Sequential(
            nn.Linear(192, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(
        self,
        agent_feat: torch.Tensor,
        goal_candidates: torch.Tensor,
        last_position: torch.Tensor,
        candidate_log_prior: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        scene_mode: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return unary score ``[N,K]`` or mode-specific score ``[N,Z,K]``."""
        num_agents, num_candidates, coordinate_dim = goal_candidates.shape
        if coordinate_dim != 2 or agent_feat.shape != (num_agents, 128):
            raise ValueError("Expected agent [N,128] and goals [N,K,2]")
        if last_position.shape != (num_agents, 2) or \
                candidate_log_prior.shape != (num_agents, num_candidates):
            raise ValueError("last_position/log_prior shape mismatch")
        mask = sanitize_candidate_mask(
            candidate_mask, num_agents, num_candidates,
            goal_candidates.device)
        displacement = goal_candidates - last_position[:, None]
        distance = torch.linalg.vector_norm(displacement.float(), dim=-1)
        candidate_feature = torch.cat((
            displacement, distance.to(displacement.dtype)[..., None],
            candidate_log_prior[..., None]), dim=-1)
        goal_hidden = self.goal_encoder(candidate_feature)
        agent_hidden = self.agent_projection(agent_feat)
        if scene_mode is None:
            # The canonical unary is z-independent.  A zero scene embedding
            # keeps this exact while retaining the approved context algebra.
            context = self.context_norm(agent_hidden)[:, None, :].expand(
                -1, num_candidates, -1)
            interaction = torch.cat(
                (context, goal_hidden, context * goal_hidden), dim=-1)
            residual = self.residual_head(interaction).squeeze(-1)
        else:
            if not self.use_scene_latent:
                raise RuntimeError(
                    "Strict no-z unary has no scene embedding")
            if scene_mode.ndim != 1 or scene_mode.shape[0] != num_agents:
                raise ValueError("scene_mode must have shape [N]")
            context = self.context_norm(
                agent_hidden + self.scene_embedding(scene_mode.long()))
            context = context[:, None, :].expand(-1, num_candidates, -1)
            interaction = torch.cat(
                (context, goal_hidden, context * goal_hidden), dim=-1)
            residual = self.residual_head(interaction).squeeze(-1)
        score = candidate_log_prior / self.prior_temperature + residual
        score = score.masked_fill(~mask, float("-inf"))
        return {"score": score, "residual": residual,
                "candidate_hidden": goal_hidden, "candidate_mask": mask}


__all__ = ["UnaryGoalResidual"]

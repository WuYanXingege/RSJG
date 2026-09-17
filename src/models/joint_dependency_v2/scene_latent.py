"""Permutation-invariant scene-latent prior ``p(z | X)``."""

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from src.models.joint_goal import canonicalize_scene_index


def segment_softmax(
    logits: torch.Tensor,
    compact_index: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Numerically stable softmax independently within every scene."""
    if logits.ndim != 1 or compact_index.shape != logits.shape:
        raise ValueError("logits and compact_index must both have shape [N]")
    if num_segments <= 0:
        raise ValueError("num_segments must be positive")
    logits32 = logits.float()
    maxima = logits32.new_full((num_segments,), float("-inf"))
    maxima.scatter_reduce_(0, compact_index, logits32, reduce="amax",
                           include_self=True)
    weights = torch.exp(logits32 - maxima[compact_index])
    denominator = logits32.new_zeros((num_segments,))
    denominator.index_add_(0, compact_index, weights)
    return (weights / denominator[compact_index].clamp_min(1e-12)).to(
        logits.dtype)


class SceneLatentPrior(nn.Module):
    """Attention-pool agent features and predict four scene modes."""

    def __init__(self, agent_dim: int = 128, num_modes: int = 4) -> None:
        super().__init__()
        if agent_dim != 128 or num_modes != 4:
            raise ValueError("JDV2 requires agent_dim=128 and num_modes=4")
        self.agent_dim = agent_dim
        self.num_modes = num_modes
        self.attention = nn.Sequential(
            nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        self.scene_encoder = nn.Sequential(
            nn.Linear(129, 128), nn.SiLU(), nn.Linear(128, 128),
            nn.LayerNorm(128))
        self.prior_head = nn.Sequential(
            nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 4))

    def pool(
        self,
        agent_feat: torch.Tensor,
        scene_index: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return scene ids, compact indices, pooled features and attention."""
        if agent_feat.ndim != 2 or agent_feat.shape[1] != self.agent_dim:
            raise ValueError("agent_feat must have shape [N,128]")
        scene_ids, compact = canonicalize_scene_index(
            scene_index, agent_feat.shape[0], agent_feat.device)
        score = self.attention(agent_feat).squeeze(-1)
        attention = segment_softmax(score, compact, scene_ids.numel())
        pooled = agent_feat.new_zeros((scene_ids.numel(), self.agent_dim))
        pooled.index_add_(0, compact, agent_feat * attention[:, None])
        return scene_ids, compact, pooled, attention

    def forward(
        self,
        agent_feat: torch.Tensor,
        scene_index: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        scene_ids, compact, pooled, attention = self.pool(
            agent_feat, scene_index)
        counts = torch.bincount(
            compact, minlength=scene_ids.numel()).to(
                device=agent_feat.device, dtype=agent_feat.dtype)
        count_feature = torch.log1p(counts).unsqueeze(-1)
        history_scene = self.scene_encoder(
            torch.cat((pooled, count_feature), dim=-1))
        logits = self.prior_head(history_scene)
        log_prob = F.log_softmax(logits.float(), dim=-1)
        return {
            "scene_ids": scene_ids,
            "compact_scene_index": compact,
            "history_scene": history_scene,
            "attention": attention,
            "logits": logits,
            "log_prob": log_prob,
            "prob": log_prob.exp(),
        }


__all__ = ["SceneLatentPrior", "segment_softmax"]

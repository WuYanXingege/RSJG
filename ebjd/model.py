"""Top-level independent EBJD network."""

from __future__ import annotations

import torch
from torch import nn

from .denoiser import GeometryConditionedJointDenoiser
from .encoders import EBJDContext, MotionSceneContextEncoder
from .representation import EndpointBridgeDecomposition


class EBJDModel(nn.Module):
    def __init__(
        self,
        goal_scale: float = 1.0,
        bridge_scale: float = 1.0,
        crop_width_m: float = 32.0,
        map_agent_chunk: int = 8,
        hidden_dim: int = 128,
        attention_heads: int = 4,
        ff_dim: int = 512,
        history_social_blocks: int = 2,
        ctf_feedback_basis: str = "legacy",
    ) -> None:
        super().__init__()
        self.representation = EndpointBridgeDecomposition(goal_scale, bridge_scale)
        self.encoder = MotionSceneContextEncoder(
            goal_scale, crop_width_m, dim=hidden_dim,
            map_agent_chunk=map_agent_chunk,
            social_blocks=history_social_blocks,
            social_heads=attention_heads, social_ff_dim=ff_dim)
        self.denoiser = GeometryConditionedJointDenoiser(
            self.representation, dim=hidden_dim, heads=attention_heads,
            ff_dim=ff_dim, ctf_feedback_basis=ctf_feedback_basis)

    def encode_context(
        self, observed: torch.Tensor, semantic_maps: torch.Tensor, valid: torch.Tensor
    ) -> EBJDContext:
        return self.encoder(observed, semantic_maps, valid)

    def forward(
        self,
        observed: torch.Tensor,
        semantic_maps: torch.Tensor,
        valid: torch.Tensor,
        latent: torch.Tensor,
        time: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.denoiser(latent, time, self.encode_context(observed, semantic_maps, valid))

"""Top-level independent EBJD network."""

from __future__ import annotations

import torch
from torch import nn

from .denoiser import JointEndpointBridgeDenoiser
from .encoders import EBJDContext, HistoryMapEncoder
from .representation import CartesianVelocityRepresentation, EndpointBridgeRepresentation


class EBJDModel(nn.Module):
    def __init__(
        self,
        goal_scale: float = 1.0,
        bridge_scale: float = 1.0,
        crop_width_m: float = 32.0,
        map_agent_chunk: int = 8,
        representation: str = "endpoint_bridge",
        future_social: bool = True,
        clean_geometry: bool = True,
    ) -> None:
        super().__init__()
        if representation == "endpoint_bridge":
            self.representation = EndpointBridgeRepresentation(goal_scale, bridge_scale)
        elif representation == "cartesian_velocity":
            self.representation = CartesianVelocityRepresentation(goal_scale)
        else:
            raise ValueError(f"unknown representation: {representation}")
        self.encoder = HistoryMapEncoder(
            goal_scale, crop_width_m, map_agent_chunk=map_agent_chunk)
        self.denoiser = JointEndpointBridgeDenoiser(
            self.representation, future_social=future_social, clean_geometry=clean_geometry)

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

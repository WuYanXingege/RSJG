"""Endpoint-Bridge Joint Diffusion (EBJD), isolated from legacy JDV2."""

from .model import EBJDModel
from .representation import EndpointBridgeRepresentation

__all__ = [
    "EBJDModel",
    "EndpointBridgeRepresentation",
]

__version__ = "0.1.0"

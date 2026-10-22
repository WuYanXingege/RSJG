"""Endpoint--Bridge Joint Diffusion (EBJD)."""

from .denoiser import FactorizedJointAttentionBlock, GeometryConditionedJointDenoiser
from .encoders import MotionSceneContextEncoder
from .model import EBJDModel
from .representation import EndpointBridgeDecomposition, EndpointBridgeRepresentation

__all__ = [
    "EBJDModel",
    "MotionSceneContextEncoder",
    "EndpointBridgeDecomposition",
    "EndpointBridgeRepresentation",
    "GeometryConditionedJointDenoiser",
    "FactorizedJointAttentionBlock",
]

__version__ = "0.1.0"

"""Top-level independent EBJD network."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from .denoiser import JointEndpointBridgeDenoiser
from .encoders import EBJDContext, HistoryMapEncoder
from .joint_goal import JointGoalAssembler, JointGoalDiagnostics
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
        native_goal_conditioning: bool = False,
        memory_mode: str = "legacy_joint",
        motion_summaries: bool = False,
        map_memory_grid: int = 8,
        anchor_residual: bool = False,
        joint_goal_assembly: bool = False,
        joint_goal_init_seed: int = 3_003_108,
        geometry_mapping: str = "direct_index",
    ) -> None:
        super().__init__()
        if representation == "endpoint_bridge":
            self.representation = EndpointBridgeRepresentation(
                goal_scale, bridge_scale, anchor_residual=anchor_residual)
        elif representation == "cartesian_velocity":
            self.representation = CartesianVelocityRepresentation(goal_scale)
        else:
            raise ValueError(f"unknown representation: {representation}")
        self.encoder = HistoryMapEncoder(
            goal_scale, crop_width_m, map_agent_chunk=map_agent_chunk,
            memory_mode=memory_mode, motion_summaries=motion_summaries,
            map_memory_grid=map_memory_grid)
        self.denoiser = JointEndpointBridgeDenoiser(
            self.representation, future_social=future_social,
            clean_geometry=clean_geometry,
            native_goal_conditioning=native_goal_conditioning,
            memory_mode=memory_mode, anchor_conditioning=anchor_residual,
            geometry_mapping=geometry_mapping)
        if joint_goal_assembly:
            # The assembler is absent from NO_ANCHOR and is constructed after
            # variant-specific modules.  Isolate its initialization so FULL
            # and NO_TOKEN receive an identical named-parameter snapshot
            # without advancing the shared training/diffusion RNG.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(joint_goal_init_seed))
                self.joint_goal_assembler = JointGoalAssembler()
        else:
            self.joint_goal_assembler = None

    def encode_context(
        self, observed: torch.Tensor, semantic_maps: torch.Tensor, valid: torch.Tensor,
        native_goals: torch.Tensor | None = None,
    ) -> EBJDContext:
        return self.encoder(observed, semantic_maps, valid, native_goals)

    def assign_training_anchors(
        self, context: EBJDContext, target: torch.Tensor,
    ) -> tuple[EBJDContext, torch.Tensor]:
        if self.joint_goal_assembler is None or context.native_goals is None:
            raise ValueError("training-anchor assignment requires V2 joint goals")
        labels = self.joint_goal_assembler.training_anchor_labels(
            context.native_goals, target, context.valid)
        anchors = self.joint_goal_assembler.anchors_from_labels(
            context.native_goals, labels)
        return replace(context, world_anchors=anchors[:, None]), labels

    @torch.no_grad()
    def assemble_world_anchors(
        self, context: EBJDContext, scene_ids: list[str],
        agent_ids: list[list[str]], seed: int, rounds: int = 2,
    ) -> tuple[EBJDContext, torch.Tensor, JointGoalDiagnostics]:
        if self.joint_goal_assembler is None:
            raise ValueError("world-anchor assembly requires V2 joint goals")
        anchors, permutations, diagnostics = self.joint_goal_assembler.assemble(
            context, scene_ids, agent_ids, seed, rounds=rounds)
        return replace(context, world_anchors=anchors), permutations, diagnostics

    @staticmethod
    def select_world_anchors(
        context: EBJDContext, indices: torch.Tensor,
    ) -> EBJDContext:
        if context.world_anchors is None:
            raise ValueError("cannot select worlds without assembled anchors")
        if indices.ndim != 2 or indices.shape[0] != context.world_anchors.shape[0]:
            raise ValueError("world indices must have shape [B,P]")
        gather = indices[..., None, None].expand(
            indices.shape[0], indices.shape[1],
            context.world_anchors.shape[2], 2)
        return replace(
            context, world_anchors=context.world_anchors.gather(1, gather))

    def forward(
        self,
        observed: torch.Tensor,
        semantic_maps: torch.Tensor,
        valid: torch.Tensor,
        latent: torch.Tensor,
        time: torch.Tensor | float,
        native_goals: torch.Tensor | None = None,
        world_anchors: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.encode_context(
            observed, semantic_maps, valid, native_goals)
        if world_anchors is not None:
            context = replace(context, world_anchors=world_anchors)
        return self.denoiser(
            latent, time, context)

"""Sparse relation-specific low-rank joint goal energy."""

from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn

from src.models.interaction_graph import EDGE_FEATURE_DIM, reverse_edge_features


class RelationSpecificJointEnergy(nn.Module):
    """Produce rank-eight factors without reusing unary/prior features."""

    def __init__(self, agent_dim: int = 128, num_scene_modes: int = 4,
                 num_relation_modes: int = 4, rank: int = 8) -> None:
        super().__init__()
        if (agent_dim, num_scene_modes, num_relation_modes, rank) != \
                (128, 4, 4, 8):
            raise ValueError("JDV2 requires D=128, Z=M=4, rank=8")
        self.rank = rank
        self.candidate_encoder = nn.Sequential(
            nn.Linear(4, 64), nn.SiLU(), nn.Linear(64, 64))
        self.pair_encoder = nn.Sequential(
            nn.Linear(2 * agent_dim + EDGE_FEATURE_DIM, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.LayerNorm(64))
        self.scene_embedding = nn.Embedding(4, 64)
        self.relation_embedding = nn.Embedding(4, 64)
        self.fusion_norm = nn.LayerNorm(64)
        self.factor_head = nn.Sequential(nn.SiLU(), nn.Linear(64, rank))

    def _candidate_hidden(
        self,
        goal_candidates: torch.Tensor,
        last_position: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index.long()
        source_input = torch.cat((
            goal_candidates[src] - last_position[src, None],
            goal_candidates[src] - last_position[dst, None]), dim=-1)
        destination_input = torch.cat((
            goal_candidates[dst] - last_position[dst, None],
            goal_candidates[dst] - last_position[src, None]), dim=-1)
        return (self.candidate_encoder(source_input),
                self.candidate_encoder(destination_input))

    def _pair_hidden(
        self,
        agent_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index.long()
        forward = self.pair_encoder(torch.cat((
            agent_feat[src], agent_feat[dst], edge_feat), dim=-1))
        reverse = self.pair_encoder(torch.cat((
            agent_feat[dst], agent_feat[src],
            reverse_edge_features(edge_feat)), dim=-1))
        return forward, reverse

    def factors(
        self,
        agent_feat: torch.Tensor,
        goal_candidates: torch.Tensor,
        last_position: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        mode_enabled: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Return left/right factors ``[E,Z,M,K,8]``."""
        edge_count = edge_index.shape[1]
        num_candidates = goal_candidates.shape[1]
        num_modes = 4 if mode_enabled else 1
        if edge_count == 0:
            empty = goal_candidates.new_empty(
                (0, num_modes, 4, num_candidates, self.rank))
            return {"left_factor": empty, "right_factor": empty}
        source, destination = self._candidate_hidden(
            goal_candidates, last_position, edge_index)
        forward, reverse = self._pair_hidden(
            agent_feat, edge_index, edge_feat)
        if mode_enabled:
            scene = self.scene_embedding.weight
        else:
            scene = self.scene_embedding.weight.new_zeros((1, 64))
        relation = self.relation_embedding.weight
        left_hidden = (
            forward[:, None, None, None, :] +
            source[:, None, None, :, :] +
            scene[None, :, None, None, :] +
            relation[None, None, :, None, :])
        right_hidden = (
            reverse[:, None, None, None, :] +
            destination[:, None, None, :, :] +
            scene[None, :, None, None, :] +
            relation[None, None, :, None, :])
        return {
            "left_factor": self.factor_head(self.fusion_norm(left_hidden)),
            "right_factor": self.factor_head(self.fusion_norm(right_hidden)),
        }

    @staticmethod
    def relation_energy(
        left_factor: torch.Tensor,
        right_factor: torch.Tensor,
    ) -> torch.Tensor:
        """Return exact relation-specific energy ``[E,Z,K,K,M]`` in FP32."""
        if left_factor.shape != right_factor.shape or left_factor.ndim != 5:
            raise ValueError("factors must share shape [E,Z,M,K,R]")
        rank = left_factor.shape[-1]
        with torch.autocast(
                device_type=left_factor.device.type, enabled=False):
            energy = -torch.einsum(
                "ezmkr,ezmlr->ezklm", left_factor.float(),
                right_factor.float()) / math.sqrt(float(rank))
        return energy

    @staticmethod
    def effective_energy(
        relation_energy: torch.Tensor,
        relation_log_prob: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``-logsumexp_m(log p(r)-E_r)`` in FP32."""
        if relation_energy.shape != relation_log_prob.shape:
            raise ValueError("energy and relation_log_prob shapes must match")
        with torch.autocast(
                device_type=relation_energy.device.type, enabled=False):
            return -torch.logsumexp(
                relation_log_prob.float() - relation_energy.float(), dim=-1)

    def selected_effective_energy(
        self,
        agent_feat: torch.Tensor,
        goal_candidates: torch.Tensor,
        last_position: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        selected_candidate: torch.Tensor,
        edge_scene_mode: torch.Tensor,
        relation_log_prob: torch.Tensor,
        conditioned_side: str,
        mode_enabled: bool = True,
    ) -> torch.Tensor:
        """Evaluate ``[E,P,K]`` energies against selected neighbours."""
        if conditioned_side not in {"source", "destination"}:
            raise ValueError("conditioned_side must be source or destination")
        edge_count = edge_index.shape[1]
        num_samples = selected_candidate.shape[1]
        num_candidates = goal_candidates.shape[1]
        if relation_log_prob.shape != (edge_count, num_samples,
                                       num_candidates, 4):
            raise ValueError("relation_log_prob must have shape [E,P,K,4]")
        if edge_count == 0:
            return relation_log_prob.new_empty(
                (0, num_samples, num_candidates))
        source, destination = self._candidate_hidden(
            goal_candidates, last_position, edge_index)
        forward, reverse = self._pair_hidden(
            agent_feat, edge_index, edge_feat)
        src, dst = edge_index.long()
        source_index = selected_candidate[src]
        destination_index = selected_candidate[dst]
        source_fixed = source.gather(
            1, source_index[:, :, None].expand(-1, -1, 64))
        destination_fixed = destination.gather(
            1, destination_index[:, :, None].expand(-1, -1, 64))
        if mode_enabled:
            scene = self.scene_embedding(edge_scene_mode.long())
        else:
            scene = forward.new_zeros((edge_count, num_samples, 64))
        relation = self.relation_embedding.weight
        common = scene[:, :, None, :] + relation[None, None, :, :]
        if conditioned_side == "source":
            variable_hidden = (
                forward[:, None, None, None, :] +
                source[:, None, None, :, :] + common[:, :, :, None, :])
            fixed_hidden = (
                reverse[:, None, None, :] + destination_fixed[:, :, None, :] +
                common)
            variable_factor = self.factor_head(
                self.fusion_norm(variable_hidden))
            fixed_factor = self.factor_head(self.fusion_norm(fixed_hidden))
            with torch.autocast(
                    device_type=variable_factor.device.type, enabled=False):
                energy = -torch.einsum(
                    "epmkr,epmr->epkm", variable_factor.float(),
                    fixed_factor.float()) / math.sqrt(float(self.rank))
        else:
            fixed_hidden = (
                forward[:, None, None, :] + source_fixed[:, :, None, :] +
                common)
            variable_hidden = (
                reverse[:, None, None, None, :] +
                destination[:, None, None, :, :] +
                common[:, :, :, None, :])
            fixed_factor = self.factor_head(self.fusion_norm(fixed_hidden))
            variable_factor = self.factor_head(
                self.fusion_norm(variable_hidden))
            with torch.autocast(
                    device_type=fixed_factor.device.type, enabled=False):
                energy = -torch.einsum(
                    "epmr,epmkr->epkm", fixed_factor.float(),
                    variable_factor.float()) / math.sqrt(float(self.rank))
        return self.effective_energy(energy, relation_log_prob)


__all__ = ["RelationSpecificJointEnergy"]

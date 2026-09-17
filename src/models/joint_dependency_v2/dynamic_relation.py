"""Hypothesis-conditioned sparse relation distribution."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


class DynamicHypothesisRelation(nn.Module):
    """Model ``p(r_ij | X,z,g_i,g_j)`` with the frozen four scalars."""

    def __init__(self, graph_radius: float, num_scene_modes: int = 4,
                 num_relation_modes: int = 4, eps: float = 1e-8) -> None:
        super().__init__()
        if graph_radius <= 0:
            raise ValueError("graph_radius must be positive")
        if num_scene_modes != 4 or num_relation_modes != 4:
            raise ValueError("JDV2 requires Z=M=4")
        self.graph_radius = float(graph_radius)
        self.num_scene_modes = num_scene_modes
        self.num_relation_modes = num_relation_modes
        self.eps = float(eps)
        self.geometry_encoder = nn.Sequential(
            nn.Linear(4, 32), nn.SiLU(), nn.Linear(32, 16))
        self.query = nn.Parameter(torch.empty(4, 4, 16))
        self.bias = nn.Parameter(torch.zeros(4, 4))
        self.relation_embedding = nn.Parameter(torch.empty(4, 16))
        nn.init.normal_(self.query, std=16 ** -0.5)
        nn.init.normal_(self.relation_embedding, std=16 ** -0.5)

    def geometry(
        self,
        source_goal: torch.Tensor,
        destination_goal: torch.Tensor,
        source_last: torch.Tensor,
        destination_last: torch.Tensor,
    ) -> torch.Tensor:
        """Return endpoint-reversal-invariant geometry ``[...,4]`` in FP32."""
        tensors = (source_goal, destination_goal, source_last, destination_last)
        if any(value.shape[-1] != 2 for value in tensors):
            raise ValueError("all goal/position tensors must end in coordinate 2")
        source_goal = source_goal.float()
        destination_goal = destination_goal.float()
        source_last = source_last.float()
        destination_last = destination_last.float()
        source_displacement = source_goal - source_last
        destination_displacement = destination_goal - destination_last
        endpoint_relative = destination_goal - source_goal
        history_relative = destination_last - source_last
        source_norm = torch.linalg.vector_norm(source_displacement, dim=-1)
        destination_norm = torch.linalg.vector_norm(
            destination_displacement, dim=-1)
        denominator = source_norm * destination_norm
        cosine = (source_displacement * destination_displacement).sum(dim=-1)
        cosine = cosine / denominator.clamp_min(self.eps)
        cosine = torch.where(
            denominator > self.eps, cosine.clamp(-1, 1),
            torch.zeros_like(cosine))
        return torch.stack((
            torch.linalg.vector_norm(endpoint_relative, dim=-1) /
            self.graph_radius,
            cosine,
            torch.linalg.vector_norm(
                endpoint_relative - history_relative, dim=-1) /
            self.graph_radius,
            (source_norm - destination_norm).abs() / self.graph_radius,
        ), dim=-1)

    def _mode_query(self, mode_enabled: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if mode_enabled:
            return self.query, self.bias
        return self.query.mean(dim=0, keepdim=True), \
            self.bias.mean(dim=0, keepdim=True)

    def full_pair_relation(
        self,
        base_relation_logits: torch.Tensor,
        goal_candidates: torch.Tensor,
        last_position: torch.Tensor,
        edge_index: torch.Tensor,
        mode_enabled: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Return full probabilities ``[E,Z,K,K,M]`` for loss/reference use."""
        if base_relation_logits.ndim != 2 or \
                base_relation_logits.shape[1] != 4:
            raise ValueError("base_relation_logits must have shape [E,4]")
        if edge_index.shape != (2, base_relation_logits.shape[0]):
            raise ValueError("edge_index/base relation edge mismatch")
        src, dst = edge_index.long()
        source_goal = goal_candidates[src, :, None, :]
        destination_goal = goal_candidates[dst, None, :, :]
        source_last = last_position[src, None, None, :]
        destination_last = last_position[dst, None, None, :]
        geometry = self.geometry(
            source_goal, destination_goal, source_last, destination_last)
        embedding = self.geometry_encoder(geometry)
        query, bias = self._mode_query(mode_enabled)
        residual = torch.einsum("eklh,zmh->ezklm", embedding, query.float())
        residual = residual / math.sqrt(16.0) + bias.float()[None, :, None, None]
        logits = base_relation_logits.float()[:, None, None, None, :] + residual
        log_prob = F.log_softmax(logits, dim=-1)
        return {"geometry": geometry, "geometry_embedding": embedding,
                "logits": logits, "log_prob": log_prob,
                "prob": log_prob.exp()}

    @staticmethod
    def _selected_goals(
        goal_candidates: torch.Tensor,
        selected_candidate: torch.Tensor,
        node_index: torch.Tensor,
    ) -> torch.Tensor:
        edge_candidates = goal_candidates[node_index]
        gather = selected_candidate[node_index].unsqueeze(-1).expand(-1, -1, 2)
        return edge_candidates.gather(1, gather)

    def selected_neighbor_relation(
        self,
        base_relation_logits: torch.Tensor,
        goal_candidates: torch.Tensor,
        last_position: torch.Tensor,
        edge_index: torch.Tensor,
        selected_candidate: torch.Tensor,
        edge_scene_mode: torch.Tensor,
        conditioned_side: str,
        dynamic_enabled: bool = True,
        mode_enabled: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Return selected-neighbour probabilities ``[E,P,K,M]``.

        ``conditioned_side='source'`` varies source candidate ``k`` against a
        fixed destination choice; ``'destination'`` performs the reverse.
        """
        if conditioned_side not in {"source", "destination"}:
            raise ValueError("conditioned_side must be source or destination")
        edge_count = edge_index.shape[1]
        if base_relation_logits.shape != (edge_count, 4):
            raise ValueError("base_relation_logits must be [E,4]")
        if selected_candidate.ndim != 2:
            raise ValueError("selected_candidate must be [N,P]")
        if edge_scene_mode.shape != (edge_count, selected_candidate.shape[1]):
            raise ValueError("edge_scene_mode must be [E,P]")
        num_samples = selected_candidate.shape[1]
        num_candidates = goal_candidates.shape[1]
        if edge_count == 0:
            shape = (0, num_samples, num_candidates, 4)
            empty = base_relation_logits.new_empty(shape, dtype=torch.float32)
            return {"logits": empty, "log_prob": empty, "prob": empty}
        src, dst = edge_index.long()
        if not dynamic_enabled:
            logits = base_relation_logits.float()[:, None, None, :].expand(
                -1, num_samples, num_candidates, -1)
            log_prob = F.log_softmax(logits, dim=-1)
            return {"logits": logits, "log_prob": log_prob,
                    "prob": log_prob.exp()}

        fixed_source = self._selected_goals(
            goal_candidates, selected_candidate, src)
        fixed_destination = self._selected_goals(
            goal_candidates, selected_candidate, dst)
        if conditioned_side == "source":
            source_goal = goal_candidates[src, None, :, :]
            destination_goal = fixed_destination[:, :, None, :]
        else:
            source_goal = fixed_source[:, :, None, :]
            destination_goal = goal_candidates[dst, None, :, :]
        geometry = self.geometry(
            source_goal, destination_goal,
            last_position[src, None, None, :],
            last_position[dst, None, None, :])
        embedding = self.geometry_encoder(geometry)
        if mode_enabled:
            query = self.query[edge_scene_mode.long()].float()  # [E,P,M,16]
            bias = self.bias[edge_scene_mode.long()].float()    # [E,P,M]
        else:
            query = self.query.mean(dim=0)[None, None].expand(
                edge_count, num_samples, -1, -1).float()
            bias = self.bias.mean(dim=0)[None, None].expand(
                edge_count, num_samples, -1).float()
        residual = torch.einsum("epkh,epmh->epkm", embedding, query)
        residual = residual / math.sqrt(16.0) + bias[:, :, None, :]
        logits = base_relation_logits.float()[:, None, None, :] + residual
        log_prob = F.log_softmax(logits, dim=-1)
        return {"geometry": geometry, "geometry_embedding": embedding,
                "logits": logits, "log_prob": log_prob,
                "prob": log_prob.exp()}

    def selected_joint_relation(
        self,
        base_relation_logits: torch.Tensor,
        goal_candidates: torch.Tensor,
        last_position: torch.Tensor,
        edge_index: torch.Tensor,
        selected_candidate: torch.Tensor,
        edge_scene_mode: torch.Tensor,
        dynamic_enabled: bool = True,
        mode_enabled: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Return relation distribution and expected embedding ``[E,P,*]``."""
        source_view = self.selected_neighbor_relation(
            base_relation_logits, goal_candidates, last_position, edge_index,
            selected_candidate, edge_scene_mode, "source",
            dynamic_enabled=dynamic_enabled, mode_enabled=mode_enabled)
        if edge_index.shape[1] == 0:
            probability = source_view["prob"].new_empty(
                (0, selected_candidate.shape[1], 4))
        else:
            src = edge_index[0].long()
            source_index = selected_candidate[src]
            probability = source_view["prob"].gather(
                2, source_index[:, :, None, None].expand(-1, -1, 1, 4)
            ).squeeze(2)
        expected_embedding = torch.einsum(
            "epm,mh->eph", probability.float(),
            self.relation_embedding.float())
        return {"prob": probability,
                "expected_embedding": expected_embedding}


__all__ = ["DynamicHypothesisRelation"]

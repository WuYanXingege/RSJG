"""Graph-wide differentiable synchronization of trajectory permutations.

V4 keeps every agent's marginal bank of ``K`` trajectories fixed and learns
only its coupling to common joint-sample slots.  This module solves the soft
part of that problem over every edge in a connected component.  It never
changes trajectory coordinates and never constructs a ``K ** N`` table.

Conventions
-----------
``pair_score[e, a, b]`` is the compatibility between source candidate ``a``
and destination candidate ``b`` for ``edge_index[:, e] = (source, dest)``.
``P[i, a, s]`` assigns candidate ``a`` of agent ``i`` to joint slot ``s``.
The edge objective is ``<C_ij, P_i @ P_j.T>``.  Its two oriented messages are
therefore exactly ``C_ij @ P_j`` and ``C_ij.T @ P_i``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

from src.models.hungarian_projection import project_permutations


def _connected_components(
    num_agents: int,
    edge_index: torch.Tensor,
    scene_index: torch.Tensor,
) -> Tuple[List[List[int]], List[int]]:
    """Return deterministic scene-local components and their scene labels."""
    adjacency: List[List[int]] = [[] for _ in range(num_agents)]
    for source, target in edge_index.detach().t().cpu().tolist():
        source, target = int(source), int(target)
        adjacency[source].append(target)
        adjacency[target].append(source)

    scene_cpu = scene_index.detach().cpu().tolist()
    scene_to_agents: Dict[int, List[int]] = {}
    for agent, scene in enumerate(scene_cpu):
        scene_to_agents.setdefault(int(scene), []).append(agent)

    components: List[List[int]] = []
    component_scenes: List[int] = []
    for scene in sorted(scene_to_agents):
        unseen = set(scene_to_agents[scene])
        while unseen:
            root = min(unseen)
            unseen.remove(root)
            stack = [root]
            component: List[int] = []
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbour in sorted(adjacency[current], reverse=True):
                    if neighbour in unseen:
                        unseen.remove(neighbour)
                        stack.append(neighbour)
            components.append(sorted(component))
            component_scenes.append(scene)
    return components, component_scenes


def log_sinkhorn(
    logits: torch.Tensor,
    iterations: int = 8,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Log-space Sinkhorn for ``[..., K, K]`` finite square matrices."""
    if logits.ndim < 2 or logits.shape[-2] != logits.shape[-1]:
        raise ValueError("logits must have shape [...,K,K]")
    if logits.shape[-1] <= 0:
        raise ValueError("Sinkhorn matrices must be non-empty")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not torch.is_floating_point(logits):
        raise TypeError("logits must be floating point")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits contain NaN or Inf")

    log_assignment = logits / float(temperature)
    # Removing one constant per matrix is not mathematically required, but it
    # keeps the first normalization comfortably scaled for extreme scores.
    maximum = log_assignment.amax(dim=(-2, -1), keepdim=True).detach()
    log_assignment = log_assignment - maximum
    for _ in range(int(iterations)):
        log_assignment = log_assignment - torch.logsumexp(
            log_assignment, dim=-1, keepdim=True
        )
        log_assignment = log_assignment - torch.logsumexp(
            log_assignment, dim=-2, keepdim=True
        )
    assignment = log_assignment.exp()
    if not bool(torch.isfinite(assignment).all()):
        raise FloatingPointError("Sinkhorn produced NaN or Inf")
    return assignment


class MultiwayPermutationSynchronizer(nn.Module):
    """Synchronize candidate-to-slot assignments over an entire sparse graph.

    An anchor is selected per connected component by weighted degree, but it
    only fixes the global permutation gauge (``P_anchor = I``).  Every graph
    edge contributes in both directions during every synchronous update.
    Optional edge top-k is applied only in inference mode.
    """

    def __init__(
        self,
        sync_iterations: int = 4,
        sinkhorn_iterations: int = 8,
        tau_sync: float = 0.2,
        alpha_msg: float = 1.0,
        alpha_prev: float = 0.1,
        alpha_identity: float = 0.1,
        eps: float = 1e-8,
        inference_edge_topk: int = 4,
        use_inference_edge_topk: bool = True,
        keep_hidden_dim: int = 32,
        initial_keep_confidence: float = 0.1,
        use_soft_confidence_mixing: bool = True,
        hard_projection: str = "hungarian",
        use_identity_bypass: bool = True,
        keep_threshold: float = 0.6,
    ) -> None:
        super().__init__()
        if sync_iterations <= 0 or sinkhorn_iterations <= 0:
            raise ValueError("sync_iterations and sinkhorn_iterations must be positive")
        if tau_sync <= 0 or eps <= 0:
            raise ValueError("tau_sync and eps must be positive")
        if min(alpha_msg, alpha_prev, alpha_identity) < 0:
            raise ValueError("synchronization alpha coefficients must be non-negative")
        if inference_edge_topk < 0:
            raise ValueError("inference_edge_topk cannot be negative")
        if keep_hidden_dim <= 0:
            raise ValueError("keep_hidden_dim must be positive")
        if not 0 < initial_keep_confidence < 1:
            raise ValueError("initial_keep_confidence must lie strictly in (0,1)")
        if hard_projection not in {"hungarian", "greedy"}:
            raise ValueError("hard_projection must be 'hungarian' or 'greedy'")
        if not 0 <= keep_threshold <= 1:
            raise ValueError("keep_threshold must lie in [0,1]")

        self.sync_iterations = int(sync_iterations)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.tau_sync = float(tau_sync)
        self.alpha_msg = float(alpha_msg)
        self.alpha_prev = float(alpha_prev)
        self.alpha_identity = float(alpha_identity)
        self.eps = float(eps)
        self.inference_edge_topk = int(inference_edge_topk)
        self.use_inference_edge_topk = bool(use_inference_edge_topk)
        self.use_soft_confidence_mixing = bool(use_soft_confidence_mixing)
        self.hard_projection = hard_projection
        self.use_identity_bypass = bool(use_identity_bypass)
        self.keep_threshold = float(keep_threshold)

        # gain, normalized entropy, mean gate/strength, log component size,
        # log score dispersion, graph density and cycle error.
        self.keep_feature_dim = 7
        self.keep_head = nn.Sequential(
            nn.Linear(self.keep_feature_dim, keep_hidden_dim),
            nn.SiLU(),
            nn.Linear(keep_hidden_dim, 1),
        )
        # A fresh model conservatively favours the identity bypass.  The head
        # can subsequently learn from task/evidence supervision.
        nn.init.zeros_(self.keep_head[-1].weight)
        initial_logit = math.log(initial_keep_confidence / (1.0 - initial_keep_confidence))
        nn.init.constant_(self.keep_head[-1].bias, initial_logit)

    def sinkhorn(self, logits: torch.Tensor) -> torch.Tensor:
        return log_sinkhorn(
            logits,
            iterations=self.sinkhorn_iterations,
            temperature=self.tau_sync,
        )

    @staticmethod
    def _validate_inputs(
        pair_score: torch.Tensor,
        edge_index: torch.Tensor,
        edge_strength: Optional[torch.Tensor],
        scene_index: Optional[torch.Tensor],
        num_agents: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if pair_score.ndim != 3 or pair_score.shape[1] != pair_score.shape[2]:
            raise ValueError("pair_score must have shape [E,K,K]")
        if pair_score.shape[1] <= 0:
            raise ValueError("pair_score must use positive K")
        if not torch.is_floating_point(pair_score):
            raise TypeError("pair_score must be floating point")
        if not bool(torch.isfinite(pair_score).all()):
            raise ValueError("pair_score contains NaN or Inf")
        if edge_index.ndim != 2 or edge_index.shape != (2, pair_score.shape[0]):
            raise ValueError("edge_index must have shape [2,E]")

        if num_agents is None:
            if scene_index is not None:
                num_agents = int(scene_index.numel())
            elif edge_index.numel():
                num_agents = int(edge_index.max().item()) + 1
            else:
                raise ValueError("num_agents is required when both graph and scene_index are empty")
        num_agents = int(num_agents)
        if num_agents <= 0:
            raise ValueError("num_agents must be positive")

        device = pair_score.device
        edge_index = edge_index.to(device=device, dtype=torch.long)
        if edge_index.numel():
            source, target = edge_index
            if bool((edge_index < 0).any()) or bool((edge_index >= num_agents).any()):
                raise ValueError("edge_index contains an out-of-range agent")
            if bool((source == target).any()):
                raise ValueError("self edges are not supported")

        if edge_strength is None:
            edge_strength = pair_score.new_ones((pair_score.shape[0],))
        else:
            if edge_strength.shape != (pair_score.shape[0],):
                raise ValueError("edge_strength must have shape [E]")
            edge_strength = edge_strength.to(device=device, dtype=pair_score.dtype)
        if not bool(torch.isfinite(edge_strength).all()):
            raise ValueError("edge_strength contains NaN or Inf")
        if bool((edge_strength < 0).any()):
            raise ValueError("edge_strength must be non-negative")

        if scene_index is None:
            scene_index = torch.zeros(num_agents, dtype=torch.long, device=device)
        else:
            if scene_index.ndim != 1 or scene_index.shape[0] != num_agents:
                raise ValueError("scene_index must have shape [num_agents]")
            scene_index = scene_index.to(device=device, dtype=torch.long)
        if edge_index.numel():
            source, target = edge_index
            if bool((scene_index[source] != scene_index[target]).any()):
                raise ValueError("edges cannot cross scene boundaries")
        return edge_index, edge_strength, num_agents

    def _active_edge_mask(
        self,
        edge_index: torch.Tensor,
        edge_strength: torch.Tensor,
        num_agents: int,
        inference: bool,
    ) -> torch.Tensor:
        edge_count = edge_index.shape[1]
        keep = torch.ones(edge_count, dtype=torch.bool, device=edge_index.device)
        if (
            not inference
            or not self.use_inference_edge_topk
            or self.inference_edge_topk <= 0
            or edge_count == 0
        ):
            return keep
        keep.zero_()
        source, target = edge_index
        detached_strength = edge_strength.detach()
        for agent in range(num_agents):
            incident = ((source == agent) | (target == agent)).nonzero(
                as_tuple=False
            ).flatten()
            if incident.numel() <= self.inference_edge_topk:
                keep[incident] = True
            else:
                chosen = torch.topk(
                    detached_strength[incident],
                    k=self.inference_edge_topk,
                    sorted=False,
                ).indices
                keep[incident[chosen]] = True
        return keep

    @staticmethod
    def _anchors(
        components: Sequence[Sequence[int]],
        weighted_degree: torch.Tensor,
    ) -> List[int]:
        anchors: List[int] = []
        for component in components:
            nodes = torch.as_tensor(
                component, dtype=torch.long, device=weighted_degree.device
            )
            # Components/nodes are sorted, so argmax resolves exact ties by
            # the smallest global agent index without adding feature noise.
            anchors.append(int(nodes[weighted_degree[nodes].argmax()].item()))
        return anchors

    @staticmethod
    def _component_cycle_error(
        assignment: torch.Tensor,
        component: Sequence[int],
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """Mean triangle error for relative assignments induced by global P."""
        zero = assignment.sum() * 0.0
        if len(component) < 3 or edge_index.shape[1] < 3:
            return zero, 0
        component_set = set(int(node) for node in component)
        adjacency = {node: set() for node in component_set}
        for source, target in edge_index.detach().t().cpu().tolist():
            source, target = int(source), int(target)
            if source in component_set and target in component_set:
                adjacency[source].add(target)
                adjacency[target].add(source)
        errors = []
        nodes = sorted(component_set)
        for position, first in enumerate(nodes):
            for second in nodes[position + 1:]:
                if second not in adjacency[first]:
                    continue
                for third in nodes:
                    if third <= second:
                        continue
                    if third not in adjacency[first] or third not in adjacency[second]:
                        continue
                    q_first_second = assignment[first] @ assignment[second].transpose(0, 1)
                    q_second_third = assignment[second] @ assignment[third].transpose(0, 1)
                    q_first_third = assignment[first] @ assignment[third].transpose(0, 1)
                    errors.append(
                        (q_first_second @ q_second_third - q_first_third).abs().mean()
                    )
        if not errors:
            return zero, 0
        return torch.stack(errors).mean(), len(errors)

    def _component_confidence(
        self,
        assignment: torch.Tensor,
        pair_score: torch.Tensor,
        edge_index: torch.Tensor,
        edge_strength: torch.Tensor,
        components: Sequence[Sequence[int]],
        component_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return confidence and evidence once per connected component."""
        identity = torch.eye(
            assignment.shape[-1], dtype=assignment.dtype, device=assignment.device
        )
        confidence = assignment.new_zeros((len(components),))
        features = assignment.new_zeros((len(components), self.keep_feature_dim))
        cycle_errors = assignment.new_zeros((len(components),))
        triangle_counts = torch.zeros(
            len(components), dtype=torch.long, device=assignment.device
        )
        if edge_index.shape[1]:
            source, target = edge_index

        for component_id, component in enumerate(components):
            cycle_error, triangle_count = self._component_cycle_error(
                assignment, component, edge_index
            )
            cycle_errors[component_id] = cycle_error
            triangle_counts[component_id] = triangle_count
            if len(component) <= 1:
                # Singleton output is always identity, hence bypass confidence
                # is deliberately zero instead of a learned arbitrary value.
                continue

            edge_mask = component_ids[source] == component_id
            edge_ids = edge_mask.nonzero(as_tuple=False).flatten()
            local_score = pair_score[edge_ids]
            local_strength = edge_strength[edge_ids]
            local_source = source[edge_ids]
            local_target = target[edge_ids]
            relative_assignment = assignment[local_source] @ assignment[
                local_target
            ].transpose(-1, -2)
            identity_objective = (
                local_strength[:, None, None] * local_score * identity
            ).sum()
            sync_objective = (
                local_strength[:, None, None] * local_score * relative_assignment
            ).sum()
            normalizer = (
                local_strength.sum().clamp_min(self.eps) * assignment.shape[-1]
            )
            gain = (sync_objective - identity_objective) / normalizer

            nodes = torch.as_tensor(
                component, dtype=torch.long, device=assignment.device
            )
            probability = assignment[nodes].clamp_min(self.eps)
            entropy = -(probability * probability.log()).sum(dim=-2).mean()
            if assignment.shape[-1] > 1:
                entropy = entropy / math.log(assignment.shape[-1])
            else:
                entropy = entropy * 0.0
            mean_strength = local_strength.mean()
            size_feature = assignment.new_tensor(math.log1p(len(component)))
            dispersion = torch.log1p(local_score.std(unbiased=False).clamp_min(0))
            max_edges = len(component) * (len(component) - 1) / 2
            density = assignment.new_tensor(float(edge_ids.numel()) / max(max_edges, 1.0))
            evidence = torch.stack(
                (
                    gain,
                    entropy,
                    mean_strength,
                    size_feature,
                    dispersion,
                    density,
                    cycle_error,
                )
            )
            features[component_id] = evidence
            confidence[component_id] = torch.sigmoid(
                self.keep_head(evidence).squeeze(-1)
            )
        return confidence, features, cycle_errors, triangle_counts

    def forward(
        self,
        pair_score: torch.Tensor,
        edge_index: torch.Tensor,
        edge_strength: Optional[torch.Tensor] = None,
        scene_index: Optional[torch.Tensor] = None,
        num_agents: Optional[int] = None,
        *,
        inference: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Run graph-wide synchronization and return assignments/diagnostics.

        ``inference=None`` follows ``not self.training``.  Training therefore
        retains every candidate edge and task gradients can reach all scores
        and strengths.  Setting inference explicitly is useful for controlled
        top-k ablations.
        """
        edge_index, edge_strength, num_agents = self._validate_inputs(
            pair_score, edge_index, edge_strength, scene_index, num_agents
        )
        if scene_index is None:
            scene_index = torch.zeros(
                num_agents, dtype=torch.long, device=pair_score.device
            )
        else:
            scene_index = scene_index.to(device=pair_score.device, dtype=torch.long)
        inference = (not self.training) if inference is None else bool(inference)
        active_mask = self._active_edge_mask(
            edge_index, edge_strength, num_agents, inference
        )
        active_index = edge_index[:, active_mask]
        active_score = pair_score[active_mask]
        active_strength = edge_strength[active_mask]

        components, component_scenes = _connected_components(
            num_agents, active_index, scene_index
        )
        component_ids = torch.empty(
            num_agents, dtype=torch.long, device=pair_score.device
        )
        for component_id, component in enumerate(components):
            component_ids[torch.as_tensor(
                component, dtype=torch.long, device=pair_score.device
            )] = component_id

        weighted_degree = pair_score.new_zeros((num_agents,))
        if active_index.shape[1]:
            source, target = active_index
            weighted_degree.index_add_(0, source, active_strength)
            weighted_degree.index_add_(0, target, active_strength)
        anchors = self._anchors(components, weighted_degree)
        anchor_mask = torch.zeros(
            num_agents, dtype=torch.bool, device=pair_score.device
        )
        anchor_mask[torch.as_tensor(
            anchors, dtype=torch.long, device=pair_score.device
        )] = True

        num_samples = pair_score.shape[-1]
        identity = torch.eye(
            num_samples, dtype=pair_score.dtype, device=pair_score.device
        )
        assignment = identity.unsqueeze(0).expand(num_agents, -1, -1).clone()
        if active_index.shape[1]:
            source, target = active_index
            update_mask = weighted_degree > self.eps
            update_mask = update_mask & ~anchor_mask
            for _ in range(self.sync_iterations):
                source_message = torch.bmm(active_score, assignment[target])
                target_message = torch.bmm(
                    active_score.transpose(-1, -2), assignment[source]
                )
                weighted_source = active_strength[:, None, None] * source_message
                weighted_target = active_strength[:, None, None] * target_message
                message = pair_score.new_zeros(
                    (num_agents, num_samples, num_samples)
                )
                message.index_add_(0, source, weighted_source)
                message.index_add_(0, target, weighted_target)
                message = message / weighted_degree.clamp_min(self.eps)[:, None, None]
                logits = (
                    self.alpha_msg * message
                    + self.alpha_prev * assignment.clamp_min(self.eps).log()
                    + self.alpha_identity * identity.unsqueeze(0)
                )
                proposal = self.sinkhorn(logits)
                assignment = torch.where(
                    update_mask[:, None, None], proposal, assignment
                )
                assignment = torch.where(
                    anchor_mask[:, None, None], identity.unsqueeze(0), assignment
                )

        component_confidence, component_features, component_cycle_error, triangle_count = (
            self._component_confidence(
                assignment,
                active_score,
                active_index,
                active_strength,
                components,
                component_ids,
            )
        )
        confidence = component_confidence[component_ids]
        # This closes the learning loop for the keep/confidence head without
        # inventing a separate binary label.  Task and no-harm losses consume
        # P_eff, so beneficial synchronization raises confidence while harmful
        # synchronization is softly routed back to identity.  A convex mixture
        # of two doubly-stochastic matrices remains doubly stochastic.  The raw
        # graph solution is retained separately for diagnostics and hard
        # projection.  Hard inference never projects this soft mixture.
        if self.use_soft_confidence_mixing and not inference:
            effective_assignment = (
                confidence[:, None, None] * assignment
                + (1.0 - confidence[:, None, None]) * identity.unsqueeze(0)
            )
        else:
            effective_assignment = assignment
        component_sizes = torch.as_tensor(
            [len(component) for component in components],
            dtype=torch.long,
            device=pair_score.device,
        )
        component_anchors = torch.as_tensor(
            anchors, dtype=torch.long, device=pair_score.device
        )
        component_scene_ids = torch.as_tensor(
            component_scenes, dtype=torch.long, device=pair_score.device
        )
        row_error_per_agent = (
            effective_assignment.sum(dim=-1) - 1.0
        ).abs().amax(dim=-1)
        column_error_per_agent = (
            effective_assignment.sum(dim=-2) - 1.0
        ).abs().amax(dim=-1)
        clamped = effective_assignment.clamp_min(self.eps)
        entropy_per_agent = -(clamped * clamped.log()).sum(dim=-2).mean(dim=-1)
        if num_samples > 1:
            normalized_entropy_per_agent = entropy_per_agent / math.log(num_samples)
        else:
            normalized_entropy_per_agent = entropy_per_agent * 0.0
        if int(triangle_count.sum().item()):
            cycle_error = (
                component_cycle_error * triangle_count.to(pair_score.dtype)
            ).sum() / triangle_count.sum().to(pair_score.dtype)
        else:
            cycle_error = assignment.sum() * 0.0

        effective_component_cycle_error = effective_assignment.new_zeros(
            (len(components),)
        )
        for component_id, component in enumerate(components):
            effective_component_cycle_error[component_id], _ = (
                self._component_cycle_error(
                    effective_assignment, component, active_index
                )
            )
        if int(triangle_count.sum().item()):
            effective_cycle_error = (
                effective_component_cycle_error * triangle_count.to(pair_score.dtype)
            ).sum() / triangle_count.sum().to(pair_score.dtype)
        else:
            effective_cycle_error = effective_assignment.sum() * 0.0

        if (
            not bool(torch.isfinite(assignment).all())
            or not bool(torch.isfinite(effective_assignment).all())
            or not bool(torch.isfinite(component_confidence).all())
        ):
            raise FloatingPointError("Permutation synchronization produced NaN or Inf")
        result: Dict[str, Any] = {
            "synchronized_permutation": assignment,
            "raw_synchronized_permutation": assignment,
            "soft_permutation": effective_assignment,
            "soft_permutations": effective_assignment,
            "soft_assignment": effective_assignment,
            "assignment": effective_assignment,
            "P": effective_assignment,
            "component_ids": component_ids,
            "component_id": component_ids,
            "component_sizes": component_sizes,
            "component_size_per_agent": component_sizes[component_ids],
            "component_anchors": component_anchors,
            "anchors": component_anchors,
            "anchor_per_agent": component_anchors[component_ids],
            "component_scene_ids": component_scene_ids,
            "component_confidence": component_confidence,
            "confidence": confidence,
            "keep_confidence": confidence,
            "component_confidence_features": component_features,
            "weighted_degree": weighted_degree,
            "active_edge_mask": active_mask,
            "sinkhorn_row_error": row_error_per_agent.mean(),
            "sinkhorn_col_error": column_error_per_agent.mean(),
            "sinkhorn_row_error_max": row_error_per_agent.max(),
            "sinkhorn_col_error_max": column_error_per_agent.max(),
            "sinkhorn_row_error_per_agent": row_error_per_agent,
            "sinkhorn_col_error_per_agent": column_error_per_agent,
            "permutation_entropy": normalized_entropy_per_agent.mean(),
            "permutation_entropy_per_agent": normalized_entropy_per_agent,
            "cycle_consistency_error": effective_cycle_error,
            "component_cycle_error": effective_component_cycle_error,
            "synchronized_cycle_consistency_error": cycle_error,
            "component_synchronized_cycle_error": component_cycle_error,
            "triangle_count": triangle_count.sum(),
            "component_triangle_count": triangle_count,
        }
        # Hard projection is deliberately outside the differentiable training
        # path.  Inference exposes both the raw Hungarian/greedy projection and
        # the conservative component-level identity-bypassed result so that
        # changed-fraction and projection-gap diagnostics remain auditable.
        if inference:
            projection = project_permutations(
                assignment, method=self.hard_projection
            )
            projected_permutation = projection["permutation"]
            projected_assignment = projection["hard_assignment"]
            component_bypass = component_confidence < self.keep_threshold
            if not self.use_identity_bypass:
                component_bypass = torch.zeros_like(component_bypass, dtype=torch.bool)
            agent_bypass = component_bypass[component_ids]
            identity_permutation = torch.arange(
                num_samples, dtype=torch.long, device=pair_score.device
            ).expand(num_agents, -1)
            hard_permutation = torch.where(
                agent_bypass[:, None], identity_permutation, projected_permutation
            )
            hard_assignment = torch.nn.functional.one_hot(
                hard_permutation, num_classes=num_samples
            ).transpose(-1, -2).to(dtype=pair_score.dtype)
            result.update({
                "projected_permutation": projected_permutation,
                "projected_assignment": projected_assignment,
                "hard_permutation": hard_permutation,
                "permutation": hard_permutation,
                "hard_assignment": hard_assignment,
                "assignment": hard_assignment,
                "component_identity_bypass": component_bypass,
                "identity_bypass": agent_bypass,
                "projection_metadata": projection["metadata"],
            })
        return result


__all__ = ["MultiwayPermutationSynchronizer", "log_sinkhorn"]

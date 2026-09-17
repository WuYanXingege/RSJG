"""Numerically stable objectives for low-rank, sparse joint goal models."""

from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from src.models.joint_goal import (
    canonicalize_scene_index,
    masked_log_softmax,
    sanitize_candidate_mask,
)


def _normalize_soft_target(
        soft_target: torch.Tensor,
        eps: float = 1e-8,
) -> torch.Tensor:
    """Validate and normalize a non-negative ``[N,K]`` target distribution."""
    if soft_target.ndim != 2 or soft_target.shape[1] <= 0:
        raise ValueError("soft_target must have shape [N,K] with K > 0.")
    if not torch.isfinite(soft_target).all() or (soft_target < 0).any():
        raise ValueError("soft_target must be finite and non-negative.")
    normalizer = soft_target.sum(dim=-1, keepdim=True)
    if (normalizer <= eps).any():
        raise ValueError("Every soft-target row must have positive mass.")
    return soft_target / normalizer


def _reduce_loss(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    """Apply a standard reduction to a per-example loss tensor."""
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'.")


def build_soft_goal_target(
        goal_candidates: torch.Tensor,
        ground_truth_goal: torch.Tensor,
        sigma_goal: float,
        candidate_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Softly assign a ground-truth endpoint to per-agent candidates.

    ``q_i(k)`` is proportional to
    ``exp(-||g_i^k-g_i_GT||^2 / (2*sigma_goal^2))``.

    Parameters
    ----------
    goal_candidates:
        Candidate endpoints ``[N,K,2]``.
    ground_truth_goal:
        Ground-truth endpoints ``[N,2]`` in the same coordinate system.
    sigma_goal:
        Positive assignment bandwidth in those coordinate units.
    candidate_mask:
        Optional validity mask ``[N,K]``. An all-false row uses candidate zero
        as a deterministic finite fallback.
    """
    if goal_candidates.ndim != 3:
        raise ValueError("goal_candidates must have shape [N,K,D].")
    num_agents, num_candidates, goal_dim = goal_candidates.shape
    if ground_truth_goal.shape != (num_agents, goal_dim):
        raise ValueError("ground_truth_goal must have shape [N,D].")
    if sigma_goal <= 0:
        raise ValueError("sigma_goal must be positive.")
    if goal_candidates.device != ground_truth_goal.device:
        raise ValueError("Candidates and ground truth must share a device.")
    if not torch.isfinite(goal_candidates).all() or \
            not torch.isfinite(ground_truth_goal).all():
        raise ValueError("Candidates and ground truth must be finite.")

    valid_mask = sanitize_candidate_mask(
        candidate_mask, num_agents, num_candidates, goal_candidates.device)
    squared_distance = (
        (goal_candidates - ground_truth_goal[:, None, :]).square().sum(dim=-1))
    target_logits = -squared_distance / (2.0 * float(sigma_goal) ** 2)
    return masked_log_softmax(target_logits, valid_mask).exp()


def low_rank_mode_log_likelihood(
        mode_log_prob: torch.Tensor,
        conditional_goal_log_prob: torch.Tensor,
        soft_goal_target: torch.Tensor,
        scene_index: Optional[torch.Tensor] = None,
        normalize_by_num_agents: bool = False,
) -> torch.Tensor:
    """Return stable low-rank joint goal log likelihood for each scene.

    Implements

    ``logsumexp_r(log p(z=r|X) + sum_i sum_k q_i(k) log p_i(k|r,X))``.
    """
    if conditional_goal_log_prob.ndim != 3:
        raise ValueError(
            "conditional_goal_log_prob must have shape [N,R,K].")
    num_agents, num_modes, num_candidates = \
        conditional_goal_log_prob.shape
    target = _normalize_soft_target(soft_goal_target)
    if target.shape != (num_agents, num_candidates):
        raise ValueError("soft_goal_target must have shape [N,K].")

    _, compact_scene_index = canonicalize_scene_index(
        scene_index, num_agents, conditional_goal_log_prob.device)
    num_scenes = int(compact_scene_index.max().item()) + 1
    if mode_log_prob.ndim == 1:
        mode_log_prob = mode_log_prob.unsqueeze(0)
    if mode_log_prob.shape != (num_scenes, num_modes):
        raise ValueError("mode_log_prob must have shape [num_scenes,R].")
    if torch.isnan(mode_log_prob).any() or torch.isposinf(mode_log_prob).any():
        raise ValueError("mode_log_prob contains invalid values.")
    if torch.isnan(conditional_goal_log_prob).any() or \
            torch.isposinf(conditional_goal_log_prob).any():
        raise ValueError("conditional_goal_log_prob contains invalid values.")

    mode_log_prob = F.log_softmax(mode_log_prob, dim=-1)
    conditional_goal_log_prob = F.log_softmax(
        conditional_goal_log_prob, dim=-1)
    # Avoid the undefined 0 * -inf product for masked-out candidates.
    weighted_log_prob = torch.where(
        target[:, None, :] > 0,
        target[:, None, :] * conditional_goal_log_prob,
        torch.zeros_like(conditional_goal_log_prob))
    agent_mode_log_likelihood = weighted_log_prob.sum(dim=-1)  # [N,R]

    scene_mode_log_likelihood = conditional_goal_log_prob.new_zeros(
        (num_scenes, num_modes))
    scene_mode_log_likelihood.index_add_(
        0, compact_scene_index, agent_mode_log_likelihood)
    if normalize_by_num_agents:
        scene_agent_count = torch.bincount(
            compact_scene_index, minlength=num_scenes).to(
                device=conditional_goal_log_prob.device,
                dtype=conditional_goal_log_prob.dtype)
        scene_mode_log_likelihood = scene_mode_log_likelihood / \
            scene_agent_count.clamp_min(1.0).unsqueeze(-1)
    return torch.logsumexp(
        mode_log_prob + scene_mode_log_likelihood, dim=-1)      # [S]


def low_rank_social_mode_loss(
        mode_log_prob: torch.Tensor,
        conditional_goal_log_prob: torch.Tensor,
        soft_goal_target: torch.Tensor,
        scene_index: Optional[torch.Tensor] = None,
        normalize_by_num_agents: bool = False,
        reduction: str = "mean",
) -> torch.Tensor:
    """Negative low-rank scene-goal log likelihood ``L_mode``."""
    negative_log_likelihood = -low_rank_mode_log_likelihood(
        mode_log_prob=mode_log_prob,
        conditional_goal_log_prob=conditional_goal_log_prob,
        soft_goal_target=soft_goal_target,
        scene_index=scene_index,
        normalize_by_num_agents=normalize_by_num_agents,
    )
    return _reduce_loss(negative_log_likelihood, reduction)


def scene_joint_ranking_loss(
    mode_log_prob: torch.Tensor,
    conditional_goal_log_prob: torch.Tensor,
    goal_candidates: torch.Tensor,
    ground_truth_goal: torch.Tensor,
    scene_index: Optional[torch.Tensor] = None,
    edge_index: Optional[torch.Tensor] = None,
    effective_pair_energy: Optional[torch.Tensor] = None,
    candidate_mask: Optional[torch.Tensor] = None,
    energy_weight: float = 1.0,
    target_temperature: float = 0.5,
    score_temperature: float = 1.0,
    normalize_pair_energy: bool = True,
    reduction: str = "mean",
    return_details: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    """Listwise rank the tractable scene-level latent goal proposals.

    Each latent mode supplies one complete joint proposal by choosing the
    highest-probability candidate for every agent.  Its oracle target quality
    is the scene-mean endpoint error in metres, a direct differentiable-scale
    surrogate for joint FDE.  The predicted proposal score combines scene-mode
    probability, selected unary log probability, and (when available) sparse
    pair energy.  No ``K**N`` joint table or sampled future trajectory is
    constructed.

    Candidate argmax indices and target ranks are intentionally detached: the
    loss trains the probability/energy assigned to each current proposal,
    rather than differentiating through a discrete selection operation.
    """
    if conditional_goal_log_prob.ndim != 3:
        raise ValueError(
            "conditional_goal_log_prob must have shape [N,R,K].")
    num_agents, num_modes, num_candidates = \
        conditional_goal_log_prob.shape
    if goal_candidates.shape != (num_agents, num_candidates, 2):
        raise ValueError("goal_candidates must have shape [N,K,2].")
    if ground_truth_goal.shape != (num_agents, 2):
        raise ValueError("ground_truth_goal must have shape [N,2].")
    if target_temperature <= 0 or score_temperature <= 0:
        raise ValueError("ranking temperatures must be positive.")
    if energy_weight < 0:
        raise ValueError("energy_weight cannot be negative.")
    if not torch.isfinite(goal_candidates).all() or \
            not torch.isfinite(ground_truth_goal).all():
        raise ValueError("Goal candidates and ground truth must be finite.")

    scene_ids, compact_scene = canonicalize_scene_index(
        scene_index, num_agents, goal_candidates.device)
    num_scenes = int(scene_ids.numel())
    if mode_log_prob.ndim == 1:
        mode_log_prob = mode_log_prob.unsqueeze(0)
    if mode_log_prob.shape != (num_scenes, num_modes):
        raise ValueError("mode_log_prob must have shape [num_scenes,R].")
    valid_mask = sanitize_candidate_mask(
        candidate_mask, num_agents, num_candidates, goal_candidates.device)
    conditional_log_prob = masked_log_softmax(
        conditional_goal_log_prob, valid_mask)
    normalized_mode_log_prob = F.log_softmax(mode_log_prob, dim=-1)

    # One deterministic, coherent scene proposal per latent social mode.
    selected_index = conditional_log_prob.detach().argmax(dim=-1)  # [N,R]
    selected_unary = conditional_log_prob.gather(
        -1, selected_index.unsqueeze(-1)).squeeze(-1)              # [N,R]
    endpoint_distance = torch.linalg.vector_norm(
        goal_candidates - ground_truth_goal[:, None, :], dim=-1)  # [N,K]
    selected_error = endpoint_distance.gather(1, selected_index)   # [N,R]

    scene_count = torch.bincount(
        compact_scene, minlength=num_scenes).to(
            device=goal_candidates.device, dtype=goal_candidates.dtype)
    scene_count = scene_count.clamp_min(1.0).unsqueeze(-1)
    scene_error = goal_candidates.new_zeros((num_scenes, num_modes))
    scene_error.index_add_(0, compact_scene, selected_error)
    scene_error = scene_error / scene_count
    scene_unary = goal_candidates.new_zeros((num_scenes, num_modes))
    scene_unary.index_add_(0, compact_scene, selected_unary)
    scene_unary = scene_unary / scene_count
    predicted_score = normalized_mode_log_prob + scene_unary

    scene_pair_energy = goal_candidates.new_zeros(
        (num_scenes, num_modes))
    if edge_index is not None:
        edge_index = edge_index.to(
            device=goal_candidates.device, dtype=torch.long)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2,E].")
        edge_count = edge_index.shape[1]
        if edge_count:
            if effective_pair_energy is None:
                raise ValueError(
                    "effective_pair_energy is required for non-empty edges.")
            if effective_pair_energy.shape != (
                    edge_count, num_candidates, num_candidates):
                raise ValueError(
                    "effective_pair_energy must have shape [E,K,K].")
            effective_pair_energy = effective_pair_energy.to(
                device=goal_candidates.device, dtype=goal_candidates.dtype)
            if not torch.isfinite(effective_pair_energy).all():
                raise ValueError("effective_pair_energy must be finite.")
            src, dst = edge_index
            if ((edge_index < 0).any() or
                    (edge_index >= num_agents).any() or
                    not bool((src < dst).all())):
                raise ValueError("edge_index must contain canonical valid pairs.")
            if (compact_scene[src] != compact_scene[dst]).any():
                raise ValueError("Interaction edges cannot cross scenes.")
            edge_rows = torch.arange(
                edge_count, device=goal_candidates.device)[:, None]
            selected_pair_energy = effective_pair_energy[
                edge_rows,
                selected_index[src],
                selected_index[dst],
            ]                                                     # [E,R]
            edge_scene = compact_scene[src]
            scene_pair_energy.index_add_(
                0, edge_scene, selected_pair_energy)
            if normalize_pair_energy:
                pair_count = torch.bincount(
                    edge_scene, minlength=num_scenes).to(
                        dtype=goal_candidates.dtype,
                        device=goal_candidates.device)
                scene_pair_energy = scene_pair_energy / \
                    pair_count.clamp_min(1.0).unsqueeze(-1)
            predicted_score = predicted_score - \
                float(energy_weight) * scene_pair_energy
        elif effective_pair_energy is not None and \
                effective_pair_energy.shape != (
                    0, num_candidates, num_candidates):
            raise ValueError(
                "Empty edge_index requires effective_pair_energy [0,K,K].")

    target_prob = torch.softmax(
        -scene_error.detach() / float(target_temperature), dim=-1)
    predicted_log_prob = F.log_softmax(
        predicted_score / float(score_temperature), dim=-1)
    per_scene_loss = -(target_prob * predicted_log_prob).sum(dim=-1)
    loss = _reduce_loss(per_scene_loss, reduction)
    if return_details:
        details = {
            "selected_candidate_index": selected_index,
            "scene_endpoint_error": scene_error,
            "target_proposal_prob": target_prob,
            "predicted_proposal_score": predicted_score,
            "predicted_proposal_log_prob": predicted_log_prob,
            "scene_pair_energy": scene_pair_energy,
        }
        return loss, details
    return loss


def best_joint_continuous_refinement_loss(
    anchor_goal: torch.Tensor,
    refined_goal: torch.Tensor,
    ground_truth_goal: torch.Tensor,
    scene_index: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> Dict[str, torch.Tensor]:
    """Supervise the best discrete joint proposal after bounded refinement.

    ``anchor_goal`` and ``refined_goal`` are ``[N,P,2]`` joint proposals.  The
    best anchor is selected per scene without gradient and only that proposal
    is pulled toward the true endpoints.  This avoids forcing every mode to
    the same goal and preserves multi-modality.  A separate mean displacement
    term is returned so callers can regularize overly aggressive corrections.
    """
    if anchor_goal.ndim != 3 or anchor_goal.shape[-1] != 2:
        raise ValueError("anchor_goal must have shape [N,P,2].")
    if refined_goal.shape != anchor_goal.shape:
        raise ValueError("refined_goal must match anchor_goal.")
    num_agents, num_proposals, _ = anchor_goal.shape
    if num_proposals <= 0:
        raise ValueError("At least one joint proposal is required.")
    if ground_truth_goal.shape != (num_agents, 2):
        raise ValueError("ground_truth_goal must have shape [N,2].")
    if not torch.isfinite(anchor_goal).all() or \
            not torch.isfinite(refined_goal).all() or \
            not torch.isfinite(ground_truth_goal).all():
        raise ValueError("Continuous refinement tensors must be finite.")

    scene_ids, compact_scene = canonicalize_scene_index(
        scene_index, num_agents, anchor_goal.device)
    num_scenes = int(scene_ids.numel())
    count = torch.bincount(compact_scene, minlength=num_scenes).to(
        device=anchor_goal.device, dtype=anchor_goal.dtype)
    count = count.clamp_min(1.0).unsqueeze(-1)

    anchor_error = torch.linalg.vector_norm(
        anchor_goal - ground_truth_goal[:, None, :], dim=-1)
    refined_error = torch.linalg.vector_norm(
        refined_goal - ground_truth_goal[:, None, :], dim=-1)
    scene_anchor_error = anchor_goal.new_zeros(
        (num_scenes, num_proposals))
    scene_refined_error = anchor_goal.new_zeros(
        (num_scenes, num_proposals))
    scene_anchor_error.index_add_(0, compact_scene, anchor_error)
    scene_refined_error.index_add_(0, compact_scene, refined_error)
    scene_anchor_error = scene_anchor_error / count
    scene_refined_error = scene_refined_error / count
    best_anchor_index = scene_anchor_error.detach().argmin(dim=-1)
    selected_refined_error = scene_refined_error.gather(
        1, best_anchor_index[:, None]).squeeze(1)
    refinement_loss = _reduce_loss(selected_refined_error, reduction)
    delta_magnitude = torch.linalg.vector_norm(
        refined_goal - anchor_goal, dim=-1).mean()
    return {
        "continuous_refinement_loss": refinement_loss,
        "refinement_delta_loss": delta_magnitude,
        "best_anchor_index": best_anchor_index,
        "scene_anchor_error": scene_anchor_error,
        "scene_refined_error": scene_refined_error,
    }


def marginal_goal_log_prob(
        mode_log_prob: torch.Tensor,
        conditional_goal_log_prob: torch.Tensor,
        scene_index: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Marginalize scene mode to get normalized per-agent logits ``[N,K]``."""
    if conditional_goal_log_prob.ndim != 3:
        raise ValueError(
            "conditional_goal_log_prob must have shape [N,R,K].")
    num_agents, num_modes, _ = conditional_goal_log_prob.shape
    _, compact_scene_index = canonicalize_scene_index(
        scene_index, num_agents, conditional_goal_log_prob.device)
    num_scenes = int(compact_scene_index.max().item()) + 1
    if mode_log_prob.ndim == 1:
        mode_log_prob = mode_log_prob.unsqueeze(0)
    if mode_log_prob.shape != (num_scenes, num_modes):
        raise ValueError("mode_log_prob must have shape [num_scenes,R].")
    mode_log_prob = F.log_softmax(mode_log_prob, dim=-1)
    conditional_goal_log_prob = F.log_softmax(
        conditional_goal_log_prob, dim=-1)
    return torch.logsumexp(
        mode_log_prob[compact_scene_index, :, None] +
        conditional_goal_log_prob,
        dim=1)


def pseudo_likelihood_loss(
        unary_log_prob: torch.Tensor,
        soft_goal_target: torch.Tensor,
        edge_index: torch.Tensor,
        effective_pair_energy: torch.Tensor,
        energy_weight: float = 1.0,
        neighbor_mode: str = "soft",
        scene_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        normalize_by_degree: bool = False,
        reduction: str = "mean",
        return_conditional_log_prob: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Compute sparse pairwise pseudo-likelihood ``L_PL``.

    ``effective_pair_energy[e,k,l]`` must already be the relation-mixture
    effective energy (computed via logsumexp in compatibility space).  In soft
    mode, the neighboring GT candidate is integrated as an expectation *after*
    this nonlinear relation mixture. It never substitutes ``sum_m q_m E_m``.

    Each canonical undirected edge ``(i,j)`` is supplied once and contributes
    ``E_ij(k,g_j)`` to i and ``E_ij(g_i,l)`` to j.
    """
    if unary_log_prob.ndim != 2:
        raise ValueError("unary_log_prob must have shape [N,K].")
    num_agents, num_candidates = unary_log_prob.shape
    target = _normalize_soft_target(soft_goal_target)
    if target.shape != (num_agents, num_candidates):
        raise ValueError("soft_goal_target must have shape [N,K].")
    if energy_weight < 0:
        raise ValueError("energy_weight cannot be negative.")
    if neighbor_mode not in ("soft", "hard", "nearest"):
        raise ValueError("neighbor_mode must be 'soft' or 'hard'.")

    edge_index = edge_index.to(
        device=unary_log_prob.device, dtype=torch.long)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2,E].")
    edge_count = edge_index.shape[1]
    if effective_pair_energy.shape != (
            edge_count, num_candidates, num_candidates):
        raise ValueError("effective_pair_energy must have shape [E,K,K].")
    effective_pair_energy = effective_pair_energy.to(
        device=unary_log_prob.device, dtype=unary_log_prob.dtype)
    if not torch.isfinite(effective_pair_energy).all():
        raise ValueError("effective_pair_energy must be finite.")
    if edge_count and ((edge_index < 0).any() or
                       (edge_index >= num_agents).any()):
        raise IndexError("edge_index contains an invalid agent id.")
    src, dst = edge_index
    if edge_count and (src == dst).any():
        raise ValueError("Self edges are not valid interaction pairs.")
    if edge_count and not bool((src < dst).all()):
        raise ValueError(
            "Each canonical undirected edge must satisfy source < destination.")
    if scene_index is not None:
        _, compact_scene = canonicalize_scene_index(
            scene_index, num_agents, unary_log_prob.device)
        if edge_count and (compact_scene[src] != compact_scene[dst]).any():
            raise ValueError("Interaction edges cannot cross scene windows.")

    if edge_weight is None:
        edge_weight = unary_log_prob.new_ones((edge_count,))
    else:
        edge_weight = edge_weight.to(
            device=unary_log_prob.device, dtype=unary_log_prob.dtype)
        if edge_weight.shape != (edge_count,):
            raise ValueError("edge_weight must have shape [E].")
        if not torch.isfinite(edge_weight).all() or (edge_weight < 0).any():
            raise ValueError("edge_weight must be finite and non-negative.")

    local_energy = unary_log_prob.new_zeros((num_agents, num_candidates))
    if edge_count:
        edge_rows = torch.arange(edge_count, device=unary_log_prob.device)
        if neighbor_mode == "soft":
            source_energy = torch.einsum(
                "ekl,el->ek", effective_pair_energy, target[dst])
            destination_energy = torch.einsum(
                "ekl,ek->el", effective_pair_energy, target[src])
        else:
            hard_target = target.argmax(dim=-1)
            source_energy = effective_pair_energy[
                edge_rows, :, hard_target[dst]]
            destination_energy = effective_pair_energy[
                edge_rows, hard_target[src], :]
        local_energy.index_add_(0, src, source_energy)
        local_energy.index_add_(0, dst, destination_energy)
        if normalize_by_degree:
            degree = unary_log_prob.new_zeros((num_agents,))
            degree.index_add_(0, src, edge_weight)
            degree.index_add_(0, dst, edge_weight)
            local_energy = local_energy / degree.clamp_min(1.0).unsqueeze(-1)

    unary_log_prob = F.log_softmax(unary_log_prob, dim=-1)
    conditional_log_prob = F.log_softmax(
        unary_log_prob - float(energy_weight) * local_energy, dim=-1)
    weighted = torch.where(
        target > 0,
        target * conditional_log_prob,
        torch.zeros_like(conditional_log_prob))
    per_agent_loss = -weighted.sum(dim=-1)
    loss = _reduce_loss(per_agent_loss, reduction)
    if return_conditional_log_prob:
        return loss, conditional_log_prob
    return loss


class JointGoalLoss(nn.Module):
    """Convenience wrapper that builds targets and returns ``L_mode``/``L_PL``."""

    def __init__(
            self,
            sigma_goal: float = 1.0,
            energy_weight: float = 1.0,
            neighbor_mode: str = "soft",
            normalize_mode_loss_by_agents: bool = False,
            normalize_pair_energy: bool = False,
    ) -> None:
        super().__init__()
        if sigma_goal <= 0:
            raise ValueError("sigma_goal must be positive.")
        if energy_weight < 0:
            raise ValueError("energy_weight cannot be negative.")
        if neighbor_mode not in ("soft", "hard", "nearest"):
            raise ValueError("neighbor_mode must be 'soft' or 'hard'.")
        self.sigma_goal = float(sigma_goal)
        self.energy_weight = float(energy_weight)
        self.neighbor_mode = neighbor_mode
        self.normalize_mode_loss_by_agents = bool(
            normalize_mode_loss_by_agents)
        self.normalize_pair_energy = bool(normalize_pair_energy)

    def forward(
            self,
            goal_candidates: torch.Tensor,
            ground_truth_goal: torch.Tensor,
            mode_log_prob: torch.Tensor,
            conditional_goal_log_prob: torch.Tensor,
            scene_index: Optional[torch.Tensor] = None,
            edge_index: Optional[torch.Tensor] = None,
            effective_pair_energy: Optional[torch.Tensor] = None,
            edge_weight: Optional[torch.Tensor] = None,
            candidate_mask: Optional[torch.Tensor] = None,
            return_details: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Compute joint-goal losses for a packed variable-agent batch."""
        target = build_soft_goal_target(
            goal_candidates, ground_truth_goal, self.sigma_goal,
            candidate_mask=candidate_mask)
        mode_loss = low_rank_social_mode_loss(
            mode_log_prob, conditional_goal_log_prob, target,
            scene_index=scene_index,
            normalize_by_num_agents=self.normalize_mode_loss_by_agents)
        unary_log_prob = marginal_goal_log_prob(
            mode_log_prob, conditional_goal_log_prob,
            scene_index=scene_index)

        num_candidates = goal_candidates.shape[1]
        if edge_index is None:
            edge_index = torch.empty(
                (2, 0), dtype=torch.long, device=goal_candidates.device)
        if effective_pair_energy is None:
            if edge_index.shape[1] != 0:
                raise ValueError(
                    "effective_pair_energy is required for non-empty edges.")
            effective_pair_energy = goal_candidates.new_empty(
                (0, num_candidates, num_candidates))
        pseudo_loss, conditional_log_prob = pseudo_likelihood_loss(
            unary_log_prob=unary_log_prob,
            soft_goal_target=target,
            edge_index=edge_index,
            effective_pair_energy=effective_pair_energy,
            energy_weight=self.energy_weight,
            neighbor_mode=self.neighbor_mode,
            scene_index=scene_index,
            edge_weight=edge_weight,
            normalize_by_degree=self.normalize_pair_energy,
            return_conditional_log_prob=True,
        )
        result = {
            "mode_loss": mode_loss,
            "pseudo_likelihood_loss": pseudo_loss,
        }
        if return_details:
            result.update({
                "soft_goal_target": target,
                "marginal_goal_log_prob": unary_log_prob,
                "pseudo_conditional_log_prob": conditional_log_prob,
            })
        return result


def scene_balanced_mean(
        per_agent_value: torch.Tensor,
        scene_index: Optional[torch.Tensor],
) -> torch.Tensor:
    """Average agents within scenes, then average equally across scenes."""
    if per_agent_value.ndim != 1:
        raise ValueError("per_agent_value must have shape [N]")
    _, compact = canonicalize_scene_index(
        scene_index, per_agent_value.shape[0], per_agent_value.device)
    scene_count = int(compact.max().item()) + 1
    total = per_agent_value.new_zeros((scene_count,), dtype=torch.float32)
    total.index_add_(0, compact, per_agent_value.float())
    count = torch.bincount(compact, minlength=scene_count).float().to(
        per_agent_value.device)
    return (total / count.clamp_min(1)).mean()


def jdv2_pseudo_likelihood(
        unary_score: torch.Tensor,
        soft_goal_target: torch.Tensor,
        scene_mode_probability: torch.Tensor,
        scene_index: torch.Tensor,
        edge_index: torch.Tensor,
        effective_pair_energy: torch.Tensor,
) -> torch.Tensor:
    """Structured/composite pseudo-likelihood for one relation path.

    Parameters use ``N`` agents, ``Z`` modes, ``K`` candidates and ``E``
    canonical edges.  The energy has shape ``[E,Z,K,K]``.  Each edge adds an
    expected neighbour energy to both endpoints, without degree averaging.
    """
    if unary_score.ndim != 2:
        raise ValueError("unary_score must have shape [N,K]")
    num_agents, num_candidates = unary_score.shape
    target = _normalize_soft_target(soft_goal_target.float())
    _, compact = canonicalize_scene_index(
        scene_index, num_agents, unary_score.device)
    scene_count = int(compact.max().item()) + 1
    if scene_mode_probability.ndim != 2 or \
            scene_mode_probability.shape[0] != scene_count:
        raise ValueError("scene_mode_probability must have shape [C,Z]")
    num_modes = scene_mode_probability.shape[1]
    edge_count = edge_index.shape[1]
    if effective_pair_energy.shape != (
            edge_count, num_modes, num_candidates, num_candidates):
        raise ValueError("effective_pair_energy must be [E,Z,K,K]")
    local_energy = unary_score.new_zeros(
        (num_agents, num_modes, num_candidates), dtype=torch.float32)
    if edge_count:
        src, dst = edge_index.long()
        source_energy = torch.einsum(
            "ezkl,el->ezk", effective_pair_energy.float(), target[dst])
        destination_energy = torch.einsum(
            "ezkl,ek->ezl", effective_pair_energy.float(), target[src])
        local_energy.index_add_(0, src, source_energy)
        local_energy.index_add_(0, dst, destination_energy)
    conditional_log_prob = F.log_softmax(
        unary_score.float()[:, None, :] - local_energy, dim=-1)
    per_agent_mode = -torch.einsum(
        "nk,nzk->nz", target, conditional_log_prob)
    mode_probability = scene_mode_probability.float()
    mode_probability = mode_probability / mode_probability.sum(
        dim=-1, keepdim=True).clamp_min(1e-12)
    per_agent = (per_agent_mode * mode_probability[compact]).sum(dim=-1)
    return scene_balanced_mean(per_agent, compact)


def jdv2_scene_kl(
        posterior_log_prob: torch.Tensor,
        prior_log_prob: torch.Tensor,
) -> torch.Tensor:
    """Return scene-mean ``KL(q_z || p_z)`` in FP32."""
    if posterior_log_prob.shape != prior_log_prob.shape or \
            posterior_log_prob.ndim != 2:
        raise ValueError("scene log probabilities must share shape [C,Z]")
    q_log = F.log_softmax(posterior_log_prob.float(), dim=-1)
    p_log = F.log_softmax(prior_log_prob.float(), dim=-1)
    return (q_log.exp() * (q_log - p_log)).sum(dim=-1).mean()


def jdv2_relation_kl(
        teacher_relation_log_prob: torch.Tensor,
        deployable_relation_log_prob: torch.Tensor,
        scene_posterior_probability: torch.Tensor,
        soft_goal_target: torch.Tensor,
        edge_index: torch.Tensor,
        scene_index: torch.Tensor,
) -> torch.Tensor:
    """Exact soft-target relation distillation from the frozen specification."""
    edge_count = edge_index.shape[1]
    if teacher_relation_log_prob.shape != (edge_count, 4):
        raise ValueError("teacher relation must have shape [E,4]")
    if deployable_relation_log_prob.ndim != 5 or \
            deployable_relation_log_prob.shape[0] != edge_count or \
            deployable_relation_log_prob.shape[-1] != 4:
        raise ValueError("deployable relation must have shape [E,Z,K,K,4]")
    if edge_count == 0:
        return deployable_relation_log_prob.sum() * 0.0
    target = _normalize_soft_target(soft_goal_target.float())
    _, compact = canonicalize_scene_index(
        scene_index, target.shape[0], target.device)
    src, dst = edge_index.long()
    edge_scene = compact[src]
    q_log = F.log_softmax(teacher_relation_log_prob.float(), dim=-1)
    p_log = F.log_softmax(deployable_relation_log_prob.float(), dim=-1)
    # KL per edge/mode/candidate pair: [E,Z,K,K].
    kl = torch.sum(q_log[:, None, None, None, :].exp() * (
        q_log[:, None, None, None, :] - p_log), dim=-1)
    weighted_candidate = torch.einsum(
        "ek,el,ezkl->ez", target[src], target[dst], kl)
    qz = scene_posterior_probability.float()
    qz = qz / qz.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return (weighted_candidate * qz[edge_scene]).sum(dim=-1).mean()


def jdv2_warmup_beta(stage_progress: float) -> float:
    """Linear 0 -> 0.1 KL warm-up over the first 20% of Stage A."""
    if not 0.0 <= float(stage_progress) <= 1.0:
        raise ValueError("stage_progress must lie in [0,1]")
    return 0.1 * min(float(stage_progress) / 0.20, 1.0)


def jdv2_teacher_probability(stage_progress: float) -> float:
    """Frozen Stage-B teacher/deployable curriculum."""
    progress = float(stage_progress)
    if not 0.0 <= progress <= 1.0:
        raise ValueError("stage_progress must lie in [0,1]")
    if progress < 0.20:
        return 1.0
    if progress < 0.60:
        return 1.0 - (progress - 0.20) / 0.40
    return 0.0


def sparse_relative_motion_loss(
        predicted_position: torch.Tensor,
        target_position: torch.Tensor,
        edge_index: torch.Tensor,
) -> torch.Tensor:
    """MSE of sparse-edge relative future displacement in world units."""
    if predicted_position.shape != target_position.shape or \
            predicted_position.ndim != 3 or \
            predicted_position.shape[-1] != 2:
        raise ValueError("positions must share shape [N,T,2]")
    if edge_index.shape[1] == 0:
        return predicted_position.sum() * 0.0
    src, dst = edge_index.long()
    predicted_relative = predicted_position[dst] - predicted_position[src]
    target_relative = target_position[dst] - target_position[src]
    return F.mse_loss(
        predicted_relative.float(), target_relative.float(), reduction="mean")


# Compatibility aliases for concise integration code.
build_soft_goal_targets = build_soft_goal_target
low_rank_mode_nll = low_rank_social_mode_loss
pseudo_likelihood_goal_loss = pseudo_likelihood_loss


__all__ = [
    "build_soft_goal_target",
    "build_soft_goal_targets",
    "low_rank_mode_log_likelihood",
    "low_rank_social_mode_loss",
    "low_rank_mode_nll",
    "scene_joint_ranking_loss",
    "best_joint_continuous_refinement_loss",
    "marginal_goal_log_prob",
    "pseudo_likelihood_loss",
    "pseudo_likelihood_goal_loss",
    "JointGoalLoss",
    "jdv2_pseudo_likelihood",
    "jdv2_relation_kl",
    "jdv2_scene_kl",
    "jdv2_teacher_probability",
    "jdv2_warmup_beta",
    "scene_balanced_mean",
    "sparse_relative_motion_loss",
]

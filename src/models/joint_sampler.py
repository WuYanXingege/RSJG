"""Tractable scene-level sampling for socially coupled candidate goals."""

import math
from typing import Dict, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F

from src.models.goal_energy import selected_effective_energy
from src.models.joint_goal import (
    canonicalize_scene_index,
    masked_log_softmax,
    sanitize_candidate_mask,
)


def estimate_joint_goal_complexity(
        num_agents: int,
        num_candidates: int,
        num_social_modes: int,
        num_relation_modes: int,
        pair_rank: int,
        num_edges: int,
        num_refinement_steps: int,
) -> Dict[str, Union[int, float]]:
    """Report naive state count and the structured computation estimate.

    The estimate follows

    ``R*N*K + L*E*K*r_e*M``

    where ``N`` is agents, ``K`` candidates, ``R`` global modes, ``M`` local
    relation modes, ``r_e`` factor rank, ``E`` sparse edges, and ``L`` blocked
    refinement rounds.
    """
    values = (
        num_agents, num_candidates, num_social_modes, num_relation_modes,
        pair_rank, num_edges, num_refinement_steps)
    if any(int(value) != value or value < 0 for value in values):
        raise ValueError("Complexity dimensions must be non-negative integers.")
    if num_agents <= 0 or num_candidates <= 0 or \
            num_social_modes <= 0 or num_relation_modes <= 0 or pair_rank <= 0:
        raise ValueError("N, K, R, M, and rank must be positive.")

    naive_states = int(num_candidates) ** int(num_agents)
    low_rank_terms = (
        int(num_social_modes) * int(num_agents) * int(num_candidates))
    refinement_terms = (
        int(num_refinement_steps) * int(num_edges) * int(num_candidates) *
        int(pair_rank) * int(num_relation_modes))
    return {
        "num_agents": int(num_agents),
        "num_candidates": int(num_candidates),
        "num_social_modes": int(num_social_modes),
        "num_relation_modes": int(num_relation_modes),
        "pair_rank": int(pair_rank),
        "num_edges": int(num_edges),
        "num_refinement_steps": int(num_refinement_steps),
        "naive_joint_states": naive_states,
        "naive_log10_states": (
            float(num_agents) * math.log10(float(num_candidates))),
        "low_rank_unary_terms": low_rank_terms,
        "refinement_terms": refinement_terms,
        "structured_computation_estimate": (
            low_rank_terms + refinement_terms),
    }


def _categorical_choice(
        scores: torch.Tensor,
        mode: str,
        temperature: float,
        generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Choose one category per row of a ``[...,K]`` score tensor."""
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if mode in ("map", "greedy", "argmax"):
        return scores.argmax(dim=-1)
    if mode not in ("sample", "sampling", "stochastic"):
        raise ValueError("sampling_mode must be 'sample' or 'map'.")

    flat_scores = (scores / temperature).reshape(-1, scores.shape[-1])
    probabilities = F.softmax(flat_scores, dim=-1)
    if not torch.isfinite(probabilities).all():
        raise ValueError("Non-finite categorical probabilities encountered.")
    choices = torch.multinomial(
        probabilities, num_samples=1, replacement=True,
        generator=generator)
    return choices.reshape(scores.shape[:-1])


def _coverage_choice(
        scores: torch.Tensor,
        candidate_mask: torch.Tensor,
        mode: str,
        temperature: float,
        generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Assign diverse candidates to sample columns for every agent.

    ``scores`` is ``[N,S,K]``.  The greedy assignment maximizes the perturbed
    categorical scores while preventing a candidate from being reused for an
    agent until all of that agent's valid candidates have been used once.  In
    stochastic mode, Gumbel perturbations retain probability-aware variation;
    MAP mode is deterministic.

    This is deliberately an inference-time diversity constraint across the
    returned sample set, not an attempt to draw independent MCMC samples.  It
    is particularly important for TTST: sampling its cluster representatives
    with replacement discards the spatial coverage that TTST was built to
    provide.
    """
    if scores.ndim != 3:
        raise ValueError("coverage scores must have shape [N,S,K].")
    num_agents, sample_count, num_candidates = scores.shape
    if candidate_mask.shape != (num_agents, num_candidates):
        raise ValueError("candidate_mask is incompatible with coverage scores.")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    if mode not in (
            "sample", "sampling", "stochastic",
            "map", "greedy", "argmax"):
        raise ValueError("sampling_mode must be 'sample' or 'map'.")

    working_score = scores / temperature
    if mode in ("sample", "sampling", "stochastic"):
        uniform = torch.rand(
            scores.shape, dtype=scores.dtype, device=scores.device,
            generator=generator)
        tiny = torch.finfo(scores.dtype).tiny
        uniform = uniform.clamp(min=tiny, max=1.0 - torch.finfo(
            scores.dtype).eps)
        working_score = working_score - torch.log(-torch.log(uniform))

    result = torch.empty(
        (num_agents, sample_count), dtype=torch.long,
        device=scores.device)
    available_candidate = candidate_mask.clone()
    unassigned_sample = torch.ones(
        (num_agents, sample_count), dtype=torch.bool,
        device=scores.device)
    agent_rows = torch.arange(num_agents, device=scores.device)

    # One vectorized greedy matching step assigns one (sample,candidate) pair
    # per agent.  If S>K_valid, availability is reset only after a complete
    # coverage cycle, so duplicates occur no earlier than necessary.
    for _ in range(sample_count):
        exhausted = ~available_candidate.any(dim=-1)
        if exhausted.any():
            available_candidate[exhausted] = candidate_mask[exhausted]
        allowed = (
            unassigned_sample.unsqueeze(-1) &
            available_candidate.unsqueeze(1))
        flat_score = working_score.masked_fill(
            ~allowed, float("-inf")).reshape(num_agents, -1)
        flat_choice = flat_score.argmax(dim=-1)
        sample_choice = torch.div(
            flat_choice, num_candidates, rounding_mode="floor")
        candidate_choice = flat_choice.remainder(num_candidates)
        result[agent_rows, sample_choice] = candidate_choice
        unassigned_sample[agent_rows, sample_choice] = False
        available_candidate[agent_rows, candidate_choice] = False
    return result


class JointGoalSampler(nn.Module):
    """Latent-mode initialization followed by blocked Gibbs-like refinement.

    Every column in the returned ``[N,S,2]`` tensor is a complete joint future
    for all agents.  Refinement is synchronous: all agents score candidates
    against the previous round's neighbor choices and are then updated as one
    block.  Each canonical undirected edge is supplied once and contributes to
    both endpoints.
    """

    def __init__(
            self,
            num_samples: int = 20,
            num_refinement_steps: int = 2,
            temperature: float = 1.0,
            energy_weight: float = 1.0,
            sampling_mode: str = "sample",
            sampling_strategy: str = "iid",
            normalize_pair_energy: bool = False,
            eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if num_refinement_steps < 0:
            raise ValueError("num_refinement_steps cannot be negative.")
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        if energy_weight < 0:
            raise ValueError("energy_weight cannot be negative.")
        if sampling_mode not in (
                "sample", "sampling", "stochastic",
                "map", "greedy", "argmax"):
            raise ValueError("sampling_mode must be 'sample' or 'map'.")
        if sampling_strategy not in ("iid", "coverage"):
            raise ValueError("sampling_strategy must be 'iid' or 'coverage'.")

        self.num_samples = int(num_samples)
        self.num_refinement_steps = int(num_refinement_steps)
        self.temperature = float(temperature)
        self.energy_weight = float(energy_weight)
        self.sampling_mode = sampling_mode
        self.sampling_strategy = sampling_strategy
        self.normalize_pair_energy = bool(normalize_pair_energy)
        self.eps = float(eps)

    @staticmethod
    def _validate_energy_output(
            energy_output: Dict[str, torch.Tensor],
            num_agents: int,
            num_candidates: int,
            compact_scene_index: torch.Tensor,
            device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Validate sparse factors and canonical undirected graph metadata."""
        required = {
            "edge_index", "left_factor", "right_factor", "relation_prob"}
        missing = required.difference(energy_output)
        if missing:
            raise KeyError(
                "energy_output is missing keys: {}".format(sorted(missing)))

        edge_index = energy_output["edge_index"].to(
            device=device, dtype=torch.long)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("energy edge_index must have shape [2,E].")
        edge_count = edge_index.shape[1]
        if edge_count and ((edge_index < 0).any() or
                           (edge_index >= num_agents).any()):
            raise IndexError("edge_index contains an invalid agent id.")
        src, dst = edge_index
        if edge_count and (src == dst).any():
            raise ValueError("Self edges are not valid interaction pairs.")
        if edge_count and not bool((src < dst).all()):
            raise ValueError(
                "Each canonical undirected edge must satisfy source < destination.")
        if edge_count and \
                (compact_scene_index[src] != compact_scene_index[dst]).any():
            raise ValueError("Interaction edges cannot cross scene windows.")

        left_factor = energy_output["left_factor"].to(device=device)
        right_factor = energy_output["right_factor"].to(device=device)
        relation_prob = energy_output["relation_prob"].to(device=device)
        if left_factor.ndim != 4 or right_factor.shape != left_factor.shape:
            raise ValueError(
                "Energy factors must share shape [E,M,K,rank].")
        if left_factor.shape[0] != edge_count or \
                left_factor.shape[2] != num_candidates:
            raise ValueError("Energy factors disagree with E or K.")
        if relation_prob.shape != left_factor.shape[:2]:
            raise ValueError("relation_prob must have shape [E,M].")

        edge_weight = energy_output.get("edge_weight")
        if edge_weight is not None:
            edge_weight = edge_weight.to(device=device)
        return {
            "edge_index": edge_index,
            "left_factor": left_factor,
            "right_factor": right_factor,
            "relation_prob": relation_prob,
            "edge_weight": edge_weight,
        }

    def forward(
            self,
            goal_candidates: torch.Tensor,
            mode_log_prob: torch.Tensor,
            conditional_goal_log_prob: torch.Tensor,
            scene_index: Optional[torch.Tensor] = None,
            energy_output: Optional[Dict[str, torch.Tensor]] = None,
            candidate_mask: Optional[torch.Tensor] = None,
            num_samples: Optional[int] = None,
            num_refinement_steps: Optional[int] = None,
            temperature: Optional[float] = None,
            energy_weight: Optional[float] = None,
            sampling_mode: Optional[str] = None,
            sampling_strategy: Optional[str] = None,
            normalize_pair_energy: Optional[bool] = None,
            generator: Optional[torch.Generator] = None,
            return_details: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Draw coherent joint candidate goals for one or more scene windows.

        ``mode_log_prob`` may be normalized log probabilities or raw logits and
        has shape ``[S_scene,R]`` (or ``[R]`` for a single scene).
        ``conditional_goal_log_prob`` similarly accepts log probabilities or
        logits with shape ``[N,R,K]``.  Applying log-softmax makes either form
        safe and normalized.
        """
        if goal_candidates.ndim != 3 or goal_candidates.shape[-1] != 2:
            raise ValueError("goal_candidates must have shape [N,K,2].")
        num_agents, num_candidates, _ = goal_candidates.shape
        if not torch.isfinite(goal_candidates).all():
            raise ValueError("goal_candidates must be finite.")
        device = goal_candidates.device

        scene_ids, compact_scene_index = canonicalize_scene_index(
            scene_index, num_agents, device)
        num_scenes = scene_ids.numel()
        if mode_log_prob.ndim == 1:
            mode_log_prob = mode_log_prob.unsqueeze(0)
        mode_log_prob = mode_log_prob.to(device=device)
        if mode_log_prob.ndim != 2 or mode_log_prob.shape[0] != num_scenes:
            raise ValueError("mode_log_prob must have shape [num_scenes,R].")
        num_social_modes = mode_log_prob.shape[1]
        if num_social_modes <= 0:
            raise ValueError("At least one social mode is required.")
        if torch.isnan(mode_log_prob).any() or torch.isposinf(mode_log_prob).any():
            raise ValueError("mode_log_prob contains invalid values.")
        mode_log_prob = F.log_softmax(mode_log_prob, dim=-1)

        conditional_goal_log_prob = conditional_goal_log_prob.to(device=device)
        if conditional_goal_log_prob.shape != (
                num_agents, num_social_modes, num_candidates):
            raise ValueError(
                "conditional_goal_log_prob must have shape [N,R,K].")
        if torch.isnan(conditional_goal_log_prob).any() or \
                torch.isposinf(conditional_goal_log_prob).any():
            raise ValueError("conditional_goal_log_prob contains invalid values.")
        valid_mask = sanitize_candidate_mask(
            candidate_mask, num_agents, num_candidates, device)
        conditional_goal_log_prob = masked_log_softmax(
            conditional_goal_log_prob, valid_mask)

        sample_count = self.num_samples if num_samples is None else int(num_samples)
        refinement_count = (
            self.num_refinement_steps if num_refinement_steps is None
            else int(num_refinement_steps))
        sample_temperature = (
            self.temperature if temperature is None else float(temperature))
        pair_energy_weight = (
            self.energy_weight if energy_weight is None else float(energy_weight))
        choice_mode = self.sampling_mode if sampling_mode is None else sampling_mode
        choice_strategy = (self.sampling_strategy if sampling_strategy is None
                           else sampling_strategy)
        normalize_energy = (
            self.normalize_pair_energy if normalize_pair_energy is None
            else bool(normalize_pair_energy))
        if sample_count <= 0 or refinement_count < 0:
            raise ValueError("Invalid sample or refinement count.")
        if sample_temperature <= 0 or pair_energy_weight < 0:
            raise ValueError("Invalid temperature or energy weight.")
        if choice_strategy not in ("iid", "coverage"):
            raise ValueError("sampling_strategy must be 'iid' or 'coverage'.")

        # One z is chosen per scene and sample. All agents mapped to that scene
        # therefore share the same high-level joint intention.
        expanded_mode_score = mode_log_prob[:, None, :].expand(
            -1, sample_count, -1)
        sampled_mode = _categorical_choice(
            expanded_mode_score, choice_mode, sample_temperature,
            generator)                                           # [S_scene,S]

        agent_sample_mode = sampled_mode[compact_scene_index]     # [N,S]
        unary_by_sample = conditional_goal_log_prob.gather(
            1, agent_sample_mode.unsqueeze(-1).expand(
                -1, -1, num_candidates))                         # [N,S,K]
        if choice_strategy == "coverage":
            candidate_index = _coverage_choice(
                unary_by_sample, valid_mask, choice_mode,
                sample_temperature, generator)
        else:
            candidate_index = _categorical_choice(
                unary_by_sample, choice_mode, sample_temperature,
                generator)                                      # [N,S]

        sparse_energy = None
        if energy_output is not None:
            sparse_energy = self._validate_energy_output(
                energy_output, num_agents, num_candidates,
                compact_scene_index, device)

        # Blocked/Gibbs-like local refinement. Each operation below is sparse:
        # O(E*M*K*rank) per scene sample and refinement step.
        if sparse_energy is not None and \
                sparse_energy["edge_index"].shape[1] > 0 and \
                refinement_count > 0 and pair_energy_weight > 0:
            edge_index = sparse_energy["edge_index"]
            src, dst = edge_index
            left_factor = sparse_energy["left_factor"]
            right_factor = sparse_energy["right_factor"]
            relation_prob = sparse_energy["relation_prob"]
            edge_weight_tensor = sparse_energy["edge_weight"]

            degree = None
            if normalize_energy:
                degree = goal_candidates.new_zeros((num_agents,))
                degree_weight = (edge_weight_tensor
                                 if edge_weight_tensor is not None
                                 else goal_candidates.new_ones((src.numel(),)))
                degree.index_add_(0, src, degree_weight)
                degree.index_add_(0, dst, degree_weight)
                degree = degree.clamp_min(1.0).unsqueeze(-1)

            for _ in range(refinement_count):
                previous_index = candidate_index.clone()
                refined_score = []
                for sample_idx in range(sample_count):
                    # Vary source k while destination l is fixed.
                    source_energy = selected_effective_energy(
                        left_factor, right_factor, relation_prob,
                        previous_index[dst, sample_idx],
                        conditioned_side="source",
                        edge_weight=edge_weight_tensor,
                        eps=self.eps)                              # [E,K]
                    # Vary destination l while source k is fixed. Orientation
                    # remains A_src(k_fixed) dot B_dst(l).
                    destination_energy = selected_effective_energy(
                        left_factor, right_factor, relation_prob,
                        previous_index[src, sample_idx],
                        conditioned_side="destination",
                        edge_weight=edge_weight_tensor,
                        eps=self.eps)                              # [E,K]

                    local_energy = goal_candidates.new_zeros(
                        (num_agents, num_candidates))
                    local_energy.index_add_(0, src, source_energy)
                    local_energy.index_add_(0, dst, destination_energy)
                    if degree is not None:
                        local_energy = local_energy / degree

                    local_score = (
                        unary_by_sample[:, sample_idx] -
                        pair_energy_weight * local_energy)
                    local_score = local_score.masked_fill(
                        ~valid_mask, float("-inf"))
                    refined_score.append(local_score)
                refined_score = torch.stack(refined_score, dim=1)  # [N,S,K]
                if choice_strategy == "coverage":
                    candidate_index = _coverage_choice(
                        refined_score, valid_mask, choice_mode,
                        sample_temperature, generator)
                else:
                    candidate_index = _categorical_choice(
                        refined_score, choice_mode, sample_temperature,
                        generator)

        gather_index = candidate_index.unsqueeze(-1).expand(-1, -1, 2)
        joint_goal_points = goal_candidates.gather(
            dim=1, index=gather_index)                            # [N,S,2]

        if return_details:
            return {
                "joint_goal_points": joint_goal_points,
                "candidate_index": candidate_index,
                "social_mode_index": sampled_mode,
                "scene_ids": scene_ids,
                "compact_scene_index": compact_scene_index,
            }
        return joint_goal_points

    def sample(self, *args, **kwargs):
        """Explicit sampling alias for integration sites that do not call modules."""
        return self.forward(*args, **kwargs)


__all__ = ["JointGoalSampler", "estimate_joint_goal_complexity"]

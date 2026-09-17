"""Vectorized two-round Parallel Conditional Refinement."""

from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from src.models.joint_goal import canonicalize_scene_index, sanitize_candidate_mask


def mode_stratified_allocation(
    scene_probability: torch.Tensor,
    num_samples: int,
    minimum_active_mode: bool = False,
) -> torch.Tensor:
    """Allocate ``P`` columns per scene using floor plus largest remainder."""
    if scene_probability.ndim != 2 or num_samples <= 0:
        raise ValueError("scene_probability must be [C,Z] and P positive")
    probability = scene_probability.float()
    if not torch.isfinite(probability).all() or (probability < 0).any():
        raise ValueError("scene_probability contains invalid values")
    probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(
        1e-12)
    expected = probability * float(num_samples)
    counts = expected.floor().long()
    if minimum_active_mode:
        active = probability > 0
        # This option is valid only when the sample budget can cover modes.
        if bool((active.sum(dim=-1) <= num_samples).all()):
            counts = torch.where(active & counts.eq(0), 1, counts)
    difference = num_samples - counts.sum(dim=-1)
    remainder = expected - expected.floor()
    result = torch.empty(
        (probability.shape[0], num_samples), dtype=torch.long,
        device=probability.device)
    for scene in range(probability.shape[0]):
        local = counts[scene].clone()
        delta = int(difference[scene].item())
        if delta > 0:
            order = torch.argsort(remainder[scene], descending=True,
                                  stable=True)
            local[order[:delta]] += 1
        elif delta < 0:
            # Minimum-one allocation may overfill. Remove from the smallest
            # remainders while never taking the sole active allocation.
            order = torch.argsort(remainder[scene], stable=True)
            for mode in order.tolist():
                removable = max(int(local[mode].item()) - 1, 0)
                take = min(removable, -delta)
                local[mode] -= take
                delta += take
                if delta == 0:
                    break
        modes = torch.repeat_interleave(
            torch.arange(probability.shape[1], device=probability.device),
            local)
        if modes.numel() != num_samples:
            raise RuntimeError("mode allocation did not produce P samples")
        result[scene] = modes
    return result


def _categorical(
    score: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
    sampling_mode: str,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    score = score.float().masked_fill(~mask, float("-inf"))
    if sampling_mode == "map":
        return score.argmax(dim=-1)
    probability = F.softmax(score / temperature, dim=-1)
    if not torch.isfinite(probability).all():
        raise FloatingPointError(
            f"sampler probability is non-finite: shape={tuple(score.shape)}")
    choices = torch.multinomial(
        probability.reshape(-1, probability.shape[-1]), 1,
        replacement=True, generator=generator)
    return choices.reshape(score.shape[:-1])


class ParallelConditionalSampler(nn.Module):
    """Sample coherent scene columns without constructing ``K**N`` states."""

    def __init__(self, num_samples: int = 20, num_refinement_steps: int = 2,
                 temperature: float = 1.0,
                 minimum_active_mode: bool = False) -> None:
        super().__init__()
        if num_samples != 20 or num_refinement_steps != 2:
            raise ValueError("canonical JDV2 requires P=20 and two rounds")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.num_samples = num_samples
        self.num_refinement_steps = num_refinement_steps
        self.temperature = float(temperature)
        self.minimum_active_mode = bool(minimum_active_mode)

    def forward(
        self,
        unary_score: torch.Tensor,
        goal_candidates: torch.Tensor,
        scene_probability: torch.Tensor,
        scene_index: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        agent_feat: torch.Tensor,
        last_position: torch.Tensor,
        base_relation_logits: torch.Tensor,
        dynamic_relation,
        joint_energy,
        candidate_mask: Optional[torch.Tensor] = None,
        sampling_mode: str = "sample",
        generator: Optional[torch.Generator] = None,
        use_scene_latent: bool = True,
        use_dynamic_relation: bool = True,
        use_joint_energy: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run mode allocation, unary initialization and two sync rounds."""
        if sampling_mode not in {"sample", "map"}:
            raise ValueError("sampling_mode must be sample or map")
        num_agents, num_candidates = unary_score.shape
        if goal_candidates.shape != (num_agents, num_candidates, 2):
            raise ValueError("goal_candidates must have shape [N,K,2]")
        mask = sanitize_candidate_mask(
            candidate_mask, num_agents, num_candidates, unary_score.device)
        scene_ids, compact = canonicalize_scene_index(
            scene_index, num_agents, unary_score.device)
        if scene_probability.shape[0] != scene_ids.numel():
            raise ValueError("scene_probability has wrong C axis")
        scene_mode = mode_stratified_allocation(
            scene_probability, self.num_samples,
            minimum_active_mode=self.minimum_active_mode)
        if not use_scene_latent:
            scene_mode.zero_()
        agent_mode = scene_mode[compact]
        score = unary_score[:, None, :].expand(
            num_agents, self.num_samples, num_candidates)
        expanded_mask = mask[:, None, :].expand_as(score)
        selected = _categorical(
            score, expanded_mask, self.temperature, sampling_mode, generator)

        edge_count = edge_index.shape[1]
        src, dst = edge_index.long()
        edge_mode = agent_mode[src] if edge_count else scene_mode.new_empty(
            (0, self.num_samples))
        if use_joint_energy and edge_count:
            # The sole Python loop is the frozen two-round algorithm.  Agents,
            # samples and candidates remain tensor axes throughout.
            for _ in range(self.num_refinement_steps):
                source_relation = dynamic_relation.selected_neighbor_relation(
                    base_relation_logits, goal_candidates, last_position,
                    edge_index, selected, edge_mode, "source",
                    dynamic_enabled=use_dynamic_relation,
                    mode_enabled=use_scene_latent)
                destination_relation = \
                    dynamic_relation.selected_neighbor_relation(
                        base_relation_logits, goal_candidates, last_position,
                        edge_index, selected, edge_mode, "destination",
                        dynamic_enabled=use_dynamic_relation,
                        mode_enabled=use_scene_latent)
                source_energy = joint_energy.selected_effective_energy(
                    agent_feat, goal_candidates, last_position, edge_index,
                    edge_feat, selected, edge_mode,
                    source_relation["log_prob"], "source",
                    mode_enabled=use_scene_latent)
                destination_energy = joint_energy.selected_effective_energy(
                    agent_feat, goal_candidates, last_position, edge_index,
                    edge_feat, selected, edge_mode,
                    destination_relation["log_prob"], "destination",
                    mode_enabled=use_scene_latent)
                accumulated = unary_score.new_zeros(
                    (num_agents, self.num_samples, num_candidates),
                    dtype=torch.float32)
                accumulated.index_add_(0, src, source_energy.float())
                accumulated.index_add_(0, dst, destination_energy.float())
                conditional_score = unary_score.float()[:, None, :] - accumulated
                selected = _categorical(
                    conditional_score, expanded_mask, self.temperature,
                    sampling_mode, generator)

        gather = selected.unsqueeze(-1).expand(-1, -1, 2)
        goals = goal_candidates.gather(1, gather)
        relation = dynamic_relation.selected_joint_relation(
            base_relation_logits, goal_candidates, last_position, edge_index,
            selected, edge_mode, dynamic_enabled=use_dynamic_relation,
            mode_enabled=use_scene_latent)
        return {
            "candidate_index": selected,
            "goals": goals,
            "scene_mode": scene_mode,
            "agent_scene_mode": agent_mode,
            "edge_scene_mode": edge_mode,
            "relation_prob": relation["prob"],
            "relation_embedding": relation["expected_embedding"],
        }


__all__ = ["ParallelConditionalSampler", "mode_stratified_allocation"]

"""Vectorized two-round Parallel Conditional Refinement."""

from typing import Callable, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from src.models.joint_goal import canonicalize_scene_index, sanitize_candidate_mask


REFINEMENT_POLICIES = (
    "categorical",
    "structured_gumbel_assignment",
)


def _stable_stream_seed(base_seed: int, window_index: int,
                        stream_index: int) -> int:
    """Derive a stable positive Torch seed without Python's random hash."""
    modulus = 2 ** 63 - 1
    value = (
        int(base_seed) * 1_000_003 +
        int(window_index) * 97_409 +
        int(stream_index) * 65_537 +
        0x5A17
    ) % modulus
    return int(value)


def make_sampling_generators(
    base_seed: int,
    window_index: int,
    device: torch.device,
    num_refinement_steps: int = 2,
) -> Dict[str, torch.Generator]:
    """Create independent initialization/refinement RNG streams."""
    result = {}
    names = ["initial"] + [
        f"round_{index}" for index in range(1, num_refinement_steps + 1)]
    for stream_index, name in enumerate(names):
        generator = torch.Generator(device=device)
        generator.manual_seed(_stable_stream_seed(
            base_seed, window_index, stream_index))
        result[name] = generator
    return result


def _unit_gumbel(
    shape: Sequence[int],
    device: torch.device,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    uniform = torch.rand(
        tuple(shape), dtype=torch.float32, device=device,
        generator=generator)
    finfo = torch.finfo(uniform.dtype)
    uniform = uniform.clamp(min=finfo.tiny, max=1.0 - finfo.eps)
    return -torch.log(-torch.log(uniform))


def _slot_invariant_agent_mask(mask: torch.Tensor,
                               num_slots: int) -> torch.Tensor:
    """Validate CPSR V1's agent-level, slot-invariant mask contract."""
    if mask.ndim != 3 or mask.shape[1] != num_slots:
        raise ValueError("CPSR mask must have shape [N,P,K]")
    local = mask[:, 0].bool()
    if not torch.equal(mask.bool(), local[:, None, :].expand_as(mask)):
        raise ValueError(
            "CPSR V1 requires an agent-level slot-invariant candidate mask")
    if bool((local.sum(-1) < num_slots).any()):
        raise ValueError("CPSR requires valid_K >= P for every agent")
    return local


def weighted_gumbel_top_p(
    score: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
    generator: Optional[torch.Generator],
    gumbel_noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Weighted Gumbel-Top-P initialization without replacement."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if score.ndim != 3 or mask.shape != score.shape:
        raise ValueError("score/mask must have shape [N,P,K]")
    num_agents, num_slots, num_candidates = score.shape
    if num_slots > num_candidates:
        raise ValueError(
            f"CPSR requires P<=K; got P={num_slots}, K={num_candidates}")
    local_mask = _slot_invariant_agent_mask(mask, num_slots)
    if not torch.equal(score, score[:, :1].expand_as(score)):
        raise ValueError("CPSR initialization logits must be slot-invariant")
    logits = score[:, 0].float()
    if not torch.isfinite(logits[local_mask]).all():
        raise FloatingPointError("valid CPSR initialization logits are non-finite")
    if gumbel_noise is None:
        gumbel_noise = _unit_gumbel(
            (num_agents, num_candidates), score.device, generator)
    elif gumbel_noise.shape != (num_agents, num_candidates):
        raise ValueError("initial Gumbel noise must have shape [N,K]")
    gumbel_noise = gumbel_noise.to(
        device=score.device, dtype=torch.float32)
    if not torch.isfinite(gumbel_noise).all():
        raise FloatingPointError("initial Gumbel noise is non-finite")
    perturbed = (logits / float(temperature) + gumbel_noise).masked_fill(
        ~local_mask, float("-inf"))
    selected = torch.topk(
        perturbed, k=num_slots, dim=-1, largest=True, sorted=True).indices
    if not local_mask.gather(1, selected).all():
        raise RuntimeError("CPSR initialization selected an invalid candidate")
    if num_slots > 1 and bool((
            selected.sort(-1).values[:, 1:] ==
            selected.sort(-1).values[:, :-1]).any()):
        raise RuntimeError("CPSR initialization produced duplicate candidates")
    return selected


def maximum_weight_injective_assignment(
    score: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Allocate each agent's slots injectively using an exact CPU solver."""
    if score.ndim != 3 or mask.shape != score.shape:
        raise ValueError("score/mask must have shape [N,P,K]")
    num_agents, num_slots, num_candidates = score.shape
    if num_slots > num_candidates:
        raise ValueError(
            f"CPSR requires P<=K; got P={num_slots}, K={num_candidates}")
    local_mask = _slot_invariant_agent_mask(mask, num_slots)
    expanded_mask = local_mask[:, None, :].expand_as(mask)
    if not torch.isfinite(score[expanded_mask]).all():
        raise FloatingPointError("valid CPSR assignment scores are non-finite")

    result = torch.empty(
        (num_agents, num_slots), dtype=torch.long, device=score.device)
    score_cpu = score.detach().float().cpu().numpy()
    mask_cpu = local_mask.detach().cpu().numpy().astype(bool)
    for agent in range(num_agents):
        cost = -score_cpu[agent].astype(np.float64, copy=True)
        valid_cost = cost[:, mask_cpu[agent]]
        finite_scale = max(float(np.abs(valid_cost).max()), 1.0)
        cost[:, ~mask_cpu[agent]] = finite_scale * 1e9
        rows, columns = linear_sum_assignment(cost)
        if rows.size != num_slots or not np.array_equal(
                np.sort(rows), np.arange(num_slots)):
            raise RuntimeError("CPSR solver returned an incomplete assignment")
        ordered = np.empty(num_slots, dtype=np.int64)
        ordered[rows] = columns
        if not mask_cpu[agent, ordered].all():
            raise RuntimeError("CPSR solver selected an invalid candidate")
        if np.unique(ordered).size != num_slots:
            raise RuntimeError("CPSR assignment is not injective")
        result[agent] = torch.from_numpy(ordered).to(result.device)
    return result


def structured_gumbel_assignment(
    score: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
    generator: Optional[torch.Generator],
    gumbel_noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Stochastic perturb-and-MAP allocation of conditional scores.

    This performs sample-set-level structured allocation of existing
    neighbor-conditioned conditional compatibility scores.  It does not
    globally maximize a full multi-agent joint compatibility objective.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if score.ndim != 3 or mask.shape != score.shape:
        raise ValueError("score/mask must have shape [N,P,K]")
    if score.shape[1] > score.shape[2]:
        raise ValueError(
            f"CPSR requires P<=K; got P={score.shape[1]}, K={score.shape[2]}")
    _slot_invariant_agent_mask(mask, score.shape[1])
    valid = mask.bool()
    score = score.float()
    if not torch.isfinite(score[valid]).all():
        raise FloatingPointError("valid CPSR conditional scores are non-finite")
    if gumbel_noise is None:
        gumbel_noise = _unit_gumbel(score.shape, score.device, generator)
    elif gumbel_noise.shape != score.shape:
        raise ValueError("refinement Gumbel noise must have shape [N,P,K]")
    gumbel_noise = gumbel_noise.to(
        device=score.device, dtype=torch.float32)
    if not torch.isfinite(gumbel_noise).all():
        raise FloatingPointError("refinement Gumbel noise is non-finite")
    perturbed = score / float(temperature) + gumbel_noise
    return maximum_weight_injective_assignment(perturbed, valid)


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
                 minimum_active_mode: bool = False,
                 strict_no_z: bool = False,
                 refinement_policy: str = "categorical") -> None:
        super().__init__()
        if num_samples != 20 or num_refinement_steps != 2:
            raise ValueError("canonical JDV2 requires P=20 and two rounds")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if refinement_policy not in REFINEMENT_POLICIES:
            raise ValueError(
                f"unknown JDV2 refinement policy: {refinement_policy}")
        if (refinement_policy == "structured_gumbel_assignment" and
                not strict_no_z):
            raise ValueError("CPSR V1 requires strict_no_z")
        self.num_samples = num_samples
        self.num_refinement_steps = num_refinement_steps
        self.temperature = float(temperature)
        self.minimum_active_mode = bool(minimum_active_mode)
        self.strict_no_z = bool(strict_no_z)
        self.refinement_policy = refinement_policy
        self.diagnostic_callback: Optional[Callable] = None
        self._sampling_context: Optional[tuple[int, int]] = None

    def set_sampling_context(self, seed: int, window_index: int) -> None:
        """Set the explicit per-window CPSR RNG context for one forward."""
        self._sampling_context = (int(seed), int(window_index))

    def _resolve_sampling_generators(
        self,
        device: torch.device,
        sampling_generators: Optional[
            Mapping[str, torch.Generator]],
    ) -> Mapping[str, torch.Generator]:
        if sampling_generators is not None:
            result = sampling_generators
        else:
            if self._sampling_context is None:
                raise RuntimeError(
                    "CPSR requires an explicit (seed, window) sampling context")
            seed, window_index = self._sampling_context
            result = make_sampling_generators(
                seed, window_index, device, self.num_refinement_steps)
        required = {"initial"} | {
            f"round_{index}"
            for index in range(1, self.num_refinement_steps + 1)}
        missing = required - set(result)
        if missing:
            raise ValueError(
                f"CPSR sampling generators are missing: {sorted(missing)}")
        self._sampling_context = None
        return result

    def _emit_diagnostic(self, round_index, score, mask, selected,
                         previous) -> None:
        callback = self.diagnostic_callback
        if callback is not None:
            callback(
                round_index=int(round_index),
                score=score,
                mask=mask,
                candidate_index=selected,
                previous_candidate_index=previous)

    def forward(
        self,
        unary_score: torch.Tensor,
        goal_candidates: torch.Tensor,
        scene_probability: Optional[torch.Tensor],
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
        sampling_generators: Optional[
            Mapping[str, torch.Generator]] = None,
        refinement_policy: Optional[str] = None,
        use_scene_latent: bool = True,
        use_dynamic_relation: bool = True,
        use_joint_energy: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run mode allocation, unary initialization and two sync rounds."""
        if sampling_mode not in {"sample", "map"}:
            raise ValueError("sampling_mode must be sample or map")
        policy = (self.refinement_policy if refinement_policy is None
                  else refinement_policy)
        if policy not in REFINEMENT_POLICIES:
            raise ValueError(f"unknown JDV2 refinement policy: {policy}")
        if policy == "structured_gumbel_assignment":
            if not self.strict_no_z:
                raise ValueError("CPSR V1 requires strict_no_z")
            if sampling_mode != "sample":
                raise ValueError("CPSR requires stochastic sampling_mode=sample")
        num_agents, num_candidates = unary_score.shape
        if goal_candidates.shape != (num_agents, num_candidates, 2):
            raise ValueError("goal_candidates must have shape [N,K,2]")
        mask = sanitize_candidate_mask(
            candidate_mask, num_agents, num_candidates, unary_score.device)
        scene_mode = None
        agent_mode = None
        edge_mode = None
        if self.strict_no_z:
            if use_scene_latent or scene_probability is not None:
                raise ValueError(
                    "Strict no-z sampler accepts no scene probability/mode")
        else:
            scene_ids, compact = canonicalize_scene_index(
                scene_index, num_agents, unary_score.device)
            if scene_probability is None or \
                    scene_probability.shape[0] != scene_ids.numel():
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
        structured_generators = None
        if policy == "structured_gumbel_assignment":
            structured_generators = self._resolve_sampling_generators(
                unary_score.device, sampling_generators)
            selected = weighted_gumbel_top_p(
                score, expanded_mask, self.temperature,
                structured_generators["initial"])
        else:
            selected = _categorical(
                score, expanded_mask, self.temperature, sampling_mode,
                generator)
        initial_selected = selected.clone()
        self._emit_diagnostic(
            0, score, expanded_mask, selected, previous=None)

        edge_count = edge_index.shape[1]
        src, dst = edge_index.long()
        if not self.strict_no_z:
            edge_mode = agent_mode[src] if edge_count else \
                scene_mode.new_empty((0, self.num_samples))
        if use_joint_energy and edge_count:
            # The sole Python loop is the frozen two-round algorithm.  Agents,
            # samples and candidates remain tensor axes throughout.
            for round_index in range(1, self.num_refinement_steps + 1):
                previous_selected = selected
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
                if policy == "structured_gumbel_assignment":
                    selected = structured_gumbel_assignment(
                        conditional_score, expanded_mask, self.temperature,
                        structured_generators[f"round_{round_index}"])
                else:
                    selected = _categorical(
                        conditional_score, expanded_mask, self.temperature,
                        sampling_mode, generator)
                self._emit_diagnostic(
                    round_index, conditional_score, expanded_mask, selected,
                    previous=previous_selected)

        gather = selected.unsqueeze(-1).expand(-1, -1, 2)
        goals = goal_candidates.gather(1, gather)
        relation = dynamic_relation.selected_joint_relation(
            base_relation_logits, goal_candidates, last_position, edge_index,
            selected, edge_mode, dynamic_enabled=use_dynamic_relation,
            mode_enabled=use_scene_latent)
        output = {
            "candidate_index": selected,
            "initial_candidate_index": initial_selected,
            "goals": goals,
            "relation_prob": relation["prob"],
            "relation_embedding": relation["expected_embedding"],
        }
        if not self.strict_no_z:
            output.update({
                "scene_mode": scene_mode,
                "agent_scene_mode": agent_mode,
                "edge_scene_mode": edge_mode,
            })
        return output


__all__ = [
    "ParallelConditionalSampler",
    "REFINEMENT_POLICIES",
    "make_sampling_generators",
    "maximum_weight_injective_assignment",
    "mode_stratified_allocation",
    "structured_gumbel_assignment",
    "weighted_gumbel_top_p",
]

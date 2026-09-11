"""Losses and diagnostics for V4 multiway trajectory coupling.

The convention in this module is intentionally explicit and never inferred:

``P[n, a, s]`` maps original candidate ``a`` of agent ``n`` to joint slot
``s``.  Consequently, the relative assignment on an oriented edge ``i -> j``
is ``Q_ij = P_i @ P_j.T``.  Training losses operate on expected *costs* under
``P``; trajectory coordinates are never averaged or interpolated here.

All public helpers are stateless.  Empty graphs, singleton scenes and fully
masked inputs produce finite zero losses, which keeps the training loop free
of special cases while preserving autograd connections where possible.
"""

from collections import OrderedDict
import math
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


Tensor = torch.Tensor


def _finite_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _check_floating_finite(tensor: Tensor, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or Inf")


def _zero(*tensors: Optional[Tensor]) -> Tensor:
    """Return a differentiable scalar zero on the first tensor's device."""
    available = [value for value in tensors if isinstance(value, torch.Tensor)]
    if not available:
        return torch.tensor(0.0)
    floating = [value for value in available if value.is_floating_point()]
    reference = floating[0] if floating else available[0]
    zero = reference.new_zeros(())
    for value in floating:
        zero = zero + value.sum() * 0.0
    return zero


def _bool_mask(mask: Optional[Tensor], shape: Tuple[int, ...], device,
               name: str) -> Tensor:
    if mask is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    mask = torch.as_tensor(mask, device=device)
    if tuple(mask.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(mask.shape)}")
    if (mask.is_floating_point() or mask.is_complex()) and not \
            torch.isfinite(mask).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return mask.bool()


def _canonical_groups(group_index: Optional[Tensor], num_agents: int,
                      device) -> Tuple[Tensor, Tensor]:
    """Return sorted group ids and compact ``[N]`` group indices."""
    if group_index is None:
        if num_agents == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty
        return (torch.zeros(1, dtype=torch.long, device=device),
                torch.zeros(num_agents, dtype=torch.long, device=device))
    group_index = torch.as_tensor(group_index, device=device)
    if group_index.ndim != 1 or group_index.shape[0] != num_agents:
        raise ValueError(f"group_index must have shape ({num_agents},)")
    if (group_index.is_floating_point() or group_index.is_complex()):
        if not torch.isfinite(group_index).all():
            raise ValueError("group_index contains NaN or Inf")
        if not torch.equal(group_index, group_index.round()):
            raise ValueError("group_index must contain integer-valued ids")
    group_index = group_index.to(dtype=torch.long)
    if num_agents == 0:
        return group_index, group_index
    group_ids, compact = torch.unique(
        group_index, sorted=True, return_inverse=True)
    return group_ids, compact


def _validate_trajectories(raw_future: Tensor, ground_truth: Tensor) -> None:
    _check_floating_finite(raw_future, "raw_future")
    _check_floating_finite(ground_truth, "ground_truth")
    if raw_future.ndim != 4:
        raise ValueError("raw_future must have shape [N,K,T,D]")
    num_agents, num_samples, num_steps, coord_dim = raw_future.shape
    if num_samples <= 0 or num_steps <= 0 or coord_dim <= 0:
        raise ValueError("raw_future requires K, T and D greater than zero")
    if ground_truth.shape != (num_agents, num_steps, coord_dim):
        raise ValueError(
            "ground_truth must have shape [N,T,D] matching raw_future")
    if raw_future.device != ground_truth.device:
        raise ValueError("raw_future and ground_truth must share a device")
    if raw_future.dtype != ground_truth.dtype:
        raise ValueError("raw_future and ground_truth must share a dtype")


def trajectory_gt_costs(
        raw_future: Tensor,
        ground_truth: Tensor,
        future_mask: Optional[Tensor] = None,
        lambda_fde: float = 1.0,
        agent_mask: Optional[Tensor] = None,
        return_details: bool = False,
) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
    """Compute masked per-trajectory ``ADE + lambda_fde * FDE``.

    Parameters use the V4 layout ``raw_future=[N,K,T,D]`` and
    ``ground_truth=[N,T,D]``.  FDE is measured at each agent's final valid
    timestep, not blindly at ``T-1``.  Agents with no valid future timestep
    have zero costs and ``valid_agent_mask=False`` so downstream reductions
    can exclude them safely.
    """
    _validate_trajectories(raw_future, ground_truth)
    lambda_fde = _finite_nonnegative(lambda_fde, "lambda_fde")
    num_agents, _, num_steps, _ = raw_future.shape
    temporal_mask = _bool_mask(
        future_mask, (num_agents, num_steps), raw_future.device, "future_mask")
    selected_agents = _bool_mask(
        agent_mask, (num_agents,), raw_future.device, "agent_mask")
    valid_agent = selected_agents & temporal_mask.any(dim=-1)

    distance = torch.linalg.vector_norm(
        raw_future - ground_truth[:, None, :, :], dim=-1)  # [N,K,T]
    weights = temporal_mask[:, None, :].to(dtype=raw_future.dtype)
    valid_count = weights.sum(dim=-1).clamp_min(1.0)
    ade = (distance * weights).sum(dim=-1) / valid_count

    timestep = torch.arange(num_steps, device=raw_future.device)
    last_valid = torch.where(
        temporal_mask, timestep[None, :], -torch.ones_like(timestep)[None, :]
    ).amax(dim=-1).clamp_min(0)
    final_index = last_valid[:, None, None].expand(
        num_agents, raw_future.shape[1], 1)
    fde = distance.gather(dim=2, index=final_index).squeeze(-1)

    valid_float = valid_agent[:, None].to(dtype=raw_future.dtype)
    ade = ade * valid_float
    fde = fde * valid_float
    cost = ade + lambda_fde * fde
    if return_details:
        return cost, {
            "cost": cost,
            "ade": ade,
            "fde": fde,
            "valid_agent_mask": valid_agent,
            "last_valid_index": last_valid,
        }
    return cost


def softmin_cost(values: Tensor, temperature: float = 0.5, dim: int = -1,
                 normalize: bool = True) -> Tensor:
    """Numerically stable differentiable soft minimum.

    With ``normalize=True`` (the project default), ``softmin([c,...,c]) == c``
    by subtracting ``log(K)`` inside the log-sum-exp.  Setting it to ``False``
    recovers the literal ``-tau*logsumexp(-x/tau)`` formula.
    """
    _check_floating_finite(values, "values")
    temperature = _positive(temperature, "temperature")
    if values.ndim == 0:
        return values
    dim = dim if dim >= 0 else values.ndim + dim
    if dim < 0 or dim >= values.ndim:
        raise ValueError("dim is out of range")
    width = values.shape[dim]
    if width <= 0:
        raise ValueError("softmin dimension must be non-empty")
    result = -temperature * torch.logsumexp(
        -values / temperature, dim=dim)
    if normalize:
        result = result + temperature * math.log(width)
    return result


def _validate_permutations(permutations: Tensor, num_agents: int,
                           num_samples: int) -> None:
    _check_floating_finite(permutations, "permutations")
    if permutations.shape != (num_agents, num_samples, num_samples):
        raise ValueError("permutations must have shape [N,K,K]")
    if (permutations < 0).any():
        raise ValueError("permutations must be non-negative")


def expected_joint_slot_costs(
        permutations: Tensor,
        trajectory_cost: Tensor,
        agent_mask: Optional[Tensor] = None,
        scene_index: Optional[Tensor] = None,
        return_metadata: bool = False,
) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
    """Return expected cost for each joint slot and packed scene.

    ``J[c,s] = mean_i sum_a P[i,a,s] d[i,a]``.  The returned shape is always
    ``[num_scenes,K]`` (including ``[1,K]`` for an unpacked non-empty input).
    """
    _check_floating_finite(trajectory_cost, "trajectory_cost")
    if trajectory_cost.ndim != 2 or trajectory_cost.shape[1] <= 0:
        raise ValueError("trajectory_cost must have shape [N,K], K > 0")
    num_agents, num_samples = trajectory_cost.shape
    _validate_permutations(permutations, num_agents, num_samples)
    if permutations.device != trajectory_cost.device or \
            permutations.dtype != trajectory_cost.dtype:
        raise ValueError("permutations and trajectory_cost must match device/dtype")
    valid_agent = _bool_mask(
        agent_mask, (num_agents,), trajectory_cost.device, "agent_mask")
    scene_ids, compact_scene = _canonical_groups(
        scene_index, num_agents, trajectory_cost.device)
    num_scenes = scene_ids.numel()

    per_agent_slot = torch.einsum(
        "nas,na->ns", permutations, trajectory_cost)
    per_agent_slot = per_agent_slot * valid_agent[:, None].to(
        dtype=trajectory_cost.dtype)
    scene_cost = trajectory_cost.new_zeros((num_scenes, num_samples))
    if num_agents:
        scene_cost.index_add_(0, compact_scene, per_agent_slot)
        counts = torch.bincount(
            compact_scene[valid_agent], minlength=num_scenes).to(
                device=trajectory_cost.device, dtype=trajectory_cost.dtype)
        scene_cost = scene_cost / counts.clamp_min(1.0)[:, None]
    else:
        counts = trajectory_cost.new_zeros((num_scenes,))
    valid_scene = counts > 0
    if return_metadata:
        return scene_cost, {
            "scene_ids": scene_ids,
            "scene_agent_count": counts,
            "valid_scene_mask": valid_scene,
            "per_agent_slot_cost": per_agent_slot,
        }
    return scene_cost


def _mean_valid(values: Tensor, valid: Tensor, *gradient_sources: Tensor) -> Tensor:
    if values.ndim != 1 or valid.shape != values.shape:
        raise ValueError("values and valid must be same-shaped vectors")
    if valid.any():
        return values[valid].mean()
    return _zero(values, *gradient_sources)


def alignment_losses(
        permutations: Tensor,
        trajectory_cost: Tensor,
        agent_mask: Optional[Tensor] = None,
        scene_index: Optional[Tensor] = None,
        temperature: float = 0.5,
        no_harm_margin: float = 0.0,
        normalized_softmin: bool = True,
) -> Dict[str, Tensor]:
    """Compute V4 alignment, identity and joint no-harm losses."""
    temperature = _positive(temperature, "temperature")
    no_harm_margin = _finite_nonnegative(no_harm_margin, "no_harm_margin")
    expected, metadata = expected_joint_slot_costs(
        permutations, trajectory_cost, agent_mask, scene_index,
        return_metadata=True)
    num_agents, num_samples = trajectory_cost.shape
    identity = torch.eye(
        num_samples, device=trajectory_cost.device, dtype=trajectory_cost.dtype
    ).expand(num_agents, -1, -1)
    identity_cost, identity_metadata = expected_joint_slot_costs(
        identity, trajectory_cost, agent_mask, scene_index,
        return_metadata=True)
    per_scene_alignment = softmin_cost(
        expected, temperature, dim=-1, normalize=normalized_softmin
    ) if expected.shape[0] else trajectory_cost.new_empty((0,))
    per_scene_identity = softmin_cost(
        identity_cost, temperature, dim=-1, normalize=normalized_softmin
    ) if identity_cost.shape[0] else trajectory_cost.new_empty((0,))
    valid_scene = metadata["valid_scene_mask"]
    loss_alignment = _mean_valid(
        per_scene_alignment, valid_scene, permutations)
    loss_identity = _mean_valid(
        per_scene_identity, valid_scene, trajectory_cost)
    per_scene_no_harm = F.relu(
        per_scene_alignment - per_scene_identity + no_harm_margin)
    loss_no_harm = _mean_valid(
        per_scene_no_harm, valid_scene, permutations)
    return {
        "loss_alignment": loss_alignment,
        "loss_identity": loss_identity,
        "loss_no_harm": loss_no_harm,
        "expected_slot_cost": expected,
        "identity_slot_cost": identity_cost,
        "per_scene_alignment": per_scene_alignment,
        "per_scene_identity": per_scene_identity,
        "per_scene_no_harm": per_scene_no_harm,
        "scene_ids": metadata["scene_ids"],
        "scene_agent_count": identity_metadata["scene_agent_count"],
        "valid_scene_mask": valid_scene,
    }


def _validate_edges(edge_index: Tensor, num_agents: int, device,
                    scene_index: Optional[Tensor] = None) -> Tensor:
    edge_index = torch.as_tensor(edge_index, device=device)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2,E]")
    edge_index = edge_index.to(dtype=torch.long)
    if edge_index.numel() and (
            (edge_index < 0).any() or (edge_index >= num_agents).any()):
        raise ValueError("edge_index contains an invalid agent index")
    if scene_index is not None and edge_index.shape[1]:
        scene_index = torch.as_tensor(scene_index, device=device)
        if scene_index.ndim != 1 or scene_index.shape[0] != num_agents:
            raise ValueError(f"scene_index must have shape ({num_agents},)")
        if not torch.equal(
                scene_index[edge_index[0]], scene_index[edge_index[1]]):
            raise ValueError("edge_index must not connect different scenes")
    return edge_index


def pair_oracle_costs(
        raw_future: Tensor,
        ground_truth: Tensor,
        edge_index: Tensor,
        trajectory_cost: Optional[Tensor] = None,
        future_mask: Optional[Tensor] = None,
        agent_mask: Optional[Tensor] = None,
        lambda_fde: float = 1.0,
        lambda_rel_geom: float = 1.0,
        lambda_rel_end: float = 1.0,
        scene_index: Optional[Tensor] = None,
        return_details: bool = False,
) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
    """Construct the candidate-pair oracle cost ``[E,K,K]``.

    The geometry terms compare predicted and ground-truth *relative paths*;
    they do not encode a universal near-is-bad repulsion rule.
    """
    _validate_trajectories(raw_future, ground_truth)
    lambda_rel_geom = _finite_nonnegative(
        lambda_rel_geom, "lambda_rel_geom")
    lambda_rel_end = _finite_nonnegative(lambda_rel_end, "lambda_rel_end")
    num_agents, num_samples, num_steps, _ = raw_future.shape
    edge_index = _validate_edges(
        edge_index, num_agents, raw_future.device, scene_index)
    temporal_mask = _bool_mask(
        future_mask, (num_agents, num_steps), raw_future.device, "future_mask")
    selected_agent = _bool_mask(
        agent_mask, (num_agents,), raw_future.device, "agent_mask")
    valid_agent = selected_agent & temporal_mask.any(dim=-1)
    if trajectory_cost is None:
        trajectory_cost = trajectory_gt_costs(
            raw_future, ground_truth, temporal_mask, lambda_fde,
            selected_agent)
    else:
        _check_floating_finite(trajectory_cost, "trajectory_cost")
        if trajectory_cost.shape != (num_agents, num_samples):
            raise ValueError("trajectory_cost must have shape [N,K]")
        if trajectory_cost.device != raw_future.device or \
                trajectory_cost.dtype != raw_future.dtype:
            raise ValueError("trajectory_cost must match raw_future device/dtype")

    edge_count = edge_index.shape[1]
    if edge_count == 0:
        empty_pair = raw_future.new_empty((0, num_samples, num_samples))
        empty_edge = torch.empty(0, dtype=torch.bool, device=raw_future.device)
        details = {
            "individual_cost": empty_pair,
            "relative_path_error": empty_pair,
            "relative_endpoint_error": empty_pair,
            "valid_edge_mask": empty_edge,
        }
        return (empty_pair, details) if return_details else empty_pair

    source, target = edge_index
    individual = 0.5 * (
        trajectory_cost[source, :, None] + trajectory_cost[target, None, :])
    predicted_relative = (
        raw_future[source, :, None, :, :] -
        raw_future[target, None, :, :, :])                    # [E,K,K,T,D]
    target_relative = (
        ground_truth[source] - ground_truth[target])[:, None, None, :, :]
    relative_distance = torch.linalg.vector_norm(
        predicted_relative - target_relative, dim=-1)         # [E,K,K,T]
    common_mask = temporal_mask[source] & temporal_mask[target]
    common_weight = common_mask[:, None, None, :].to(raw_future.dtype)
    common_count = common_weight.sum(dim=-1).clamp_min(1.0)
    relative_path = (
        relative_distance * common_weight).sum(dim=-1) / common_count

    timestep = torch.arange(num_steps, device=raw_future.device)
    last_common = torch.where(
        common_mask, timestep[None, :], -torch.ones_like(timestep)[None, :]
    ).amax(dim=-1).clamp_min(0)
    end_index = last_common[:, None, None, None].expand(
        edge_count, num_samples, num_samples, 1)
    relative_endpoint = relative_distance.gather(
        dim=3, index=end_index).squeeze(-1)
    valid_edge = valid_agent[source] & valid_agent[target] & common_mask.any(-1)
    edge_valid_float = valid_edge[:, None, None].to(raw_future.dtype)
    relative_path = relative_path * edge_valid_float
    relative_endpoint = relative_endpoint * edge_valid_float
    pair_cost = (
        individual + lambda_rel_geom * relative_path +
        lambda_rel_end * relative_endpoint)
    # Invalid edges are omitted by downstream masks; zeroing them also makes
    # standalone inspection unsurprising and finite.
    pair_cost = pair_cost * edge_valid_float
    if return_details:
        return pair_cost, {
            "individual_cost": individual * edge_valid_float,
            "relative_path_error": relative_path,
            "relative_endpoint_error": relative_endpoint,
            "valid_edge_mask": valid_edge,
        }
    return pair_cost


def pair_target_distribution(pair_oracle_cost: Tensor,
                             temperature: float = 0.5,
                             detach: bool = True) -> Tensor:
    """Turn ``[E,K,K]`` oracle costs into per-edge distributions."""
    _check_floating_finite(pair_oracle_cost, "pair_oracle_cost")
    if pair_oracle_cost.ndim != 3 or \
            pair_oracle_cost.shape[1] != pair_oracle_cost.shape[2] or \
            pair_oracle_cost.shape[1] <= 0:
        raise ValueError("pair_oracle_cost must have shape [E,K,K], K > 0")
    temperature = _positive(temperature, "temperature")
    source = pair_oracle_cost.detach() if detach else pair_oracle_cost
    if source.shape[0] == 0:
        return source.clone()
    flat = torch.softmax(-source.flatten(1) / temperature, dim=-1)
    return flat.reshape_as(source)


def _distribution_loss_per_item(predicted_logits: Tensor, target_prob: Tensor,
                                temperature: float, loss_type: str,
                                eps: float) -> Tuple[Tensor, Tensor]:
    if predicted_logits.shape != target_prob.shape:
        raise ValueError("predicted logits and target probabilities must match")
    if predicted_logits.ndim < 2:
        raise ValueError("distribution tensors require an item dimension")
    if (target_prob < 0).any():
        raise ValueError("target probabilities must be non-negative")
    flat_target = target_prob.flatten(1)
    normalizer = flat_target.sum(dim=-1, keepdim=True)
    if (normalizer <= eps).any():
        raise ValueError("each target distribution must have positive mass")
    flat_target = flat_target / normalizer
    predicted_log_prob = F.log_softmax(
        predicted_logits.flatten(1) / temperature, dim=-1)
    cross_entropy = -(flat_target * predicted_log_prob).sum(dim=-1)
    loss_type = str(loss_type).lower()
    if loss_type in {"ce", "cross_entropy", "soft_ce"}:
        return cross_entropy, predicted_log_prob.reshape_as(predicted_logits)
    if loss_type == "kl":
        target_log = flat_target.clamp_min(eps).log()
        kl = torch.where(
            flat_target > 0,
            flat_target * (target_log - predicted_log_prob),
            torch.zeros_like(flat_target)).sum(dim=-1)
        return kl, predicted_log_prob.reshape_as(predicted_logits)
    raise ValueError("loss_type must be 'kl' or 'cross_entropy'")


def _weighted_edge_mean(per_edge: Tensor, edge_mask: Optional[Tensor],
                        edge_weight: Optional[Tensor],
                        *gradient_sources: Tensor) -> Tensor:
    edge_count = per_edge.shape[0]
    valid = _bool_mask(edge_mask, (edge_count,), per_edge.device, "edge_mask")
    if edge_weight is None:
        weight = torch.ones_like(per_edge)
    else:
        _check_floating_finite(edge_weight, "edge_weight")
        if edge_weight.shape != (edge_count,):
            raise ValueError("edge_weight must have shape [E]")
        if (edge_weight < 0).any():
            raise ValueError("edge_weight must be non-negative")
        weight = edge_weight.to(device=per_edge.device, dtype=per_edge.dtype)
    weight = weight * valid.to(per_edge.dtype)
    denominator = weight.sum()
    if not valid.any() or not bool((denominator > 0).item()):
        return _zero(per_edge, *gradient_sources)
    return (per_edge * weight).sum() / denominator


def pair_score_loss(
        pair_scores: Tensor,
        pair_oracle_cost: Optional[Tensor] = None,
        target_prob: Optional[Tensor] = None,
        target_temperature: float = 0.5,
        prediction_temperature: float = 1.0,
        loss_type: str = "kl",
        edge_mask: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None,
        detach_target: bool = True,
        eps: float = 1e-8,
        return_details: bool = False,
) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
    """Soft CE/KL supervision for candidate-pair compatibility scores."""
    _check_floating_finite(pair_scores, "pair_scores")
    if pair_scores.ndim != 3 or pair_scores.shape[1] != pair_scores.shape[2] or \
            pair_scores.shape[1] <= 0:
        raise ValueError("pair_scores must have shape [E,K,K], K > 0")
    prediction_temperature = _positive(
        prediction_temperature, "prediction_temperature")
    eps = _positive(eps, "eps")
    if target_prob is None:
        if pair_oracle_cost is None:
            raise ValueError("pair_oracle_cost or target_prob is required")
        if pair_oracle_cost.shape != pair_scores.shape:
            raise ValueError("pair_oracle_cost must match pair_scores")
        target_prob = pair_target_distribution(
            pair_oracle_cost, target_temperature, detach_target)
    else:
        _check_floating_finite(target_prob, "target_prob")
        if detach_target:
            target_prob = target_prob.detach()
    if pair_scores.shape[0] == 0:
        loss = _zero(pair_scores)
        details = {
            "per_edge_loss": pair_scores.new_empty((0,)),
            "target_prob": target_prob,
            "predicted_log_prob": pair_scores.clone(),
        }
        return (loss, details) if return_details else loss
    per_edge, predicted_log_prob = _distribution_loss_per_item(
        pair_scores, target_prob, prediction_temperature, loss_type, eps)
    loss = _weighted_edge_mean(
        per_edge, edge_mask, edge_weight, pair_scores)
    details = {
        "per_edge_loss": per_edge,
        "target_prob": target_prob,
        "predicted_log_prob": predicted_log_prob,
    }
    return (loss, details) if return_details else loss


# Descriptive alias used by some integration paths.
pair_score_distribution_loss = pair_score_loss


def relative_assignment_matrices(permutations: Tensor,
                                 edge_index: Tensor,
                                 scene_index: Optional[Tensor] = None) -> Tensor:
    """Return ``Q_ij=P_i@P_j.T`` with shape ``[E,K,K]``."""
    _check_floating_finite(permutations, "permutations")
    if permutations.ndim != 3 or permutations.shape[1] != permutations.shape[2]:
        raise ValueError("permutations must have shape [N,K,K]")
    if (permutations < 0).any():
        raise ValueError("permutations must be non-negative")
    edge_index = _validate_edges(
        edge_index, permutations.shape[0], permutations.device, scene_index)
    if edge_index.shape[1] == 0:
        return permutations.new_empty(
            (0, permutations.shape[1], permutations.shape[2]))
    source, target = edge_index
    return permutations[source] @ permutations[target].transpose(-2, -1)


def pair_assignment_loss(
        permutations: Tensor,
        edge_index: Tensor,
        pair_oracle_cost: Optional[Tensor] = None,
        target_prob: Optional[Tensor] = None,
        target_temperature: float = 0.5,
        loss_type: str = "kl",
        edge_mask: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None,
        scene_index: Optional[Tensor] = None,
        detach_target: bool = True,
        eps: float = 1e-8,
        return_details: bool = False,
) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
    """Supervise the synchronized relative assignment ``P_src @ P_dst.T``."""
    eps = _positive(eps, "eps")
    q_matrix = relative_assignment_matrices(
        permutations, edge_index, scene_index)
    if target_prob is None:
        if pair_oracle_cost is None:
            raise ValueError("pair_oracle_cost or target_prob is required")
        if pair_oracle_cost.shape != q_matrix.shape:
            raise ValueError("pair_oracle_cost must have shape [E,K,K]")
        target_prob = pair_target_distribution(
            pair_oracle_cost, target_temperature, detach_target)
    else:
        _check_floating_finite(target_prob, "target_prob")
        if target_prob.shape != q_matrix.shape:
            raise ValueError("target_prob must have shape [E,K,K]")
        if detach_target:
            target_prob = target_prob.detach()
    if q_matrix.shape[0] == 0:
        loss = _zero(permutations, q_matrix)
        details = {
            "per_edge_loss": q_matrix.new_empty((0,)),
            "relative_assignment": q_matrix,
            "relative_assignment_prob": q_matrix.clone(),
            "target_prob": target_prob,
        }
        return (loss, details) if return_details else loss

    q_prob = q_matrix / q_matrix.flatten(1).sum(dim=-1).clamp_min(
        eps)[:, None, None]
    flat_q_log = q_prob.flatten(1).clamp_min(eps).log()
    flat_target = target_prob.flatten(1)
    flat_target = flat_target / flat_target.sum(dim=-1, keepdim=True).clamp_min(eps)
    cross_entropy = -(flat_target * flat_q_log).sum(dim=-1)
    loss_type = str(loss_type).lower()
    if loss_type in {"ce", "cross_entropy", "soft_ce"}:
        per_edge = cross_entropy
    elif loss_type == "kl":
        per_edge = torch.where(
            flat_target > 0,
            flat_target * (flat_target.clamp_min(eps).log() - flat_q_log),
            torch.zeros_like(flat_target)).sum(dim=-1)
    else:
        raise ValueError("loss_type must be 'kl' or 'cross_entropy'")
    loss = _weighted_edge_mean(
        per_edge, edge_mask, edge_weight, permutations)
    details = {
        "per_edge_loss": per_edge,
        "relative_assignment": q_matrix,
        "relative_assignment_prob": q_prob,
        "target_prob": target_prob,
    }
    return (loss, details) if return_details else loss


def permutation_entropy_loss(permutations: Tensor,
                             agent_mask: Optional[Tensor] = None,
                             normalize_by_log_k: bool = False,
                             eps: float = 1e-8) -> Tensor:
    """Mean ``-(1/K) sum_as P[a,s]log(P[a,s])`` over valid agents."""
    _check_floating_finite(permutations, "permutations")
    if permutations.ndim != 3 or permutations.shape[1] != permutations.shape[2] or \
            permutations.shape[1] <= 0:
        raise ValueError("permutations must have shape [N,K,K], K > 0")
    if (permutations < 0).any():
        raise ValueError("permutations must be non-negative")
    eps = _positive(eps, "eps")
    num_agents, num_samples, _ = permutations.shape
    valid = _bool_mask(
        agent_mask, (num_agents,), permutations.device, "agent_mask")
    entropy = -torch.where(
        permutations > 0,
        permutations * permutations.clamp_min(eps).log(),
        torch.zeros_like(permutations)).sum(dim=(1, 2)) / num_samples
    if normalize_by_log_k and num_samples > 1:
        entropy = entropy / math.log(num_samples)
    return _mean_valid(entropy, valid, permutations)


def relation_prior_kl_loss(
        relation_posterior: Tensor,
        relation_prior: Tensor,
        edge_mask: Optional[Tensor] = None,
        pair_mask: Optional[Tensor] = None,
        inputs_are_logits: bool = False,
        eps: float = 1e-8,
) -> Tensor:
    """Mean candidate posterior-to-observation-prior KL, ``KL(q||q0)``.

    ``relation_posterior`` is ``[E,K,K,M]``.  ``relation_prior`` may be
    ``[E,M]``, ``[E,1,1,M]`` or the full posterior shape.
    """
    _check_floating_finite(relation_posterior, "relation_posterior")
    _check_floating_finite(relation_prior, "relation_prior")
    if relation_posterior.ndim != 4 or relation_posterior.shape[-1] <= 0:
        raise ValueError("relation_posterior must have shape [E,K,K,M]")
    edge_count, num_rows, num_cols, num_modes = relation_posterior.shape
    if relation_prior.ndim == 2:
        if relation_prior.shape != (edge_count, num_modes):
            raise ValueError("relation_prior must have shape [E,M]")
        relation_prior = relation_prior[:, None, None, :]
    try:
        relation_prior = torch.broadcast_to(
            relation_prior, relation_posterior.shape)
    except RuntimeError as exc:
        raise ValueError("relation_prior is not broadcastable to [E,K,K,M]") from exc
    if relation_prior.device != relation_posterior.device or \
            relation_prior.dtype != relation_posterior.dtype:
        raise ValueError("relation posterior/prior must match device and dtype")
    eps = _positive(eps, "eps")
    if inputs_are_logits:
        posterior = torch.softmax(relation_posterior, dim=-1)
        prior = torch.softmax(relation_prior, dim=-1)
    else:
        if (relation_posterior < 0).any() or (relation_prior < 0).any():
            raise ValueError("relation probabilities must be non-negative")
        posterior = relation_posterior / relation_posterior.sum(
            dim=-1, keepdim=True).clamp_min(eps)
        prior = relation_prior / relation_prior.sum(
            dim=-1, keepdim=True).clamp_min(eps)
    pair_kl = torch.where(
        posterior > 0,
        posterior * (posterior.clamp_min(eps).log() -
                     prior.clamp_min(eps).log()),
        torch.zeros_like(posterior)).sum(dim=-1)
    valid_edge = _bool_mask(
        edge_mask, (edge_count,), posterior.device, "edge_mask")
    valid_pair = _bool_mask(
        pair_mask, (edge_count, num_rows, num_cols), posterior.device,
        "pair_mask")
    valid = valid_pair & valid_edge[:, None, None]
    if valid.any():
        return pair_kl[valid].mean()
    return _zero(relation_posterior)


def gate_regularization_loss(
        pair_gate: Tensor,
        regularization: str = "none",
        target_sparsity: float = 0.5,
        edge_mask: Optional[Tensor] = None,
) -> Tensor:
    """Optional mild gate regularizer; ``none`` is the V4 default."""
    _check_floating_finite(pair_gate, "pair_gate")
    if pair_gate.ndim not in (1, 3):
        raise ValueError("pair_gate must have shape [E] or [E,K,K]")
    if (pair_gate < 0).any() or (pair_gate > 1).any():
        raise ValueError("pair_gate values must lie in [0,1]")
    edge_count = pair_gate.shape[0]
    valid_edge = _bool_mask(
        edge_mask, (edge_count,), pair_gate.device, "edge_mask")
    mode = str(regularization).lower()
    if mode in {"none", "off", "disabled"}:
        return _zero(pair_gate)
    if not valid_edge.any():
        return _zero(pair_gate)
    selected = pair_gate[valid_edge]
    if mode in {"l1", "mean", "sparse"}:
        return selected.mean()
    if mode in {"target", "target_sparsity"}:
        target_sparsity = float(target_sparsity)
        if not math.isfinite(target_sparsity) or not 0 <= target_sparsity <= 1:
            raise ValueError("target_sparsity must lie in [0,1]")
        return (selected.mean() - target_sparsity).square()
    raise ValueError("regularization must be 'none', 'l1', or 'target_sparsity'")


def compute_multiway_coupling_loss(
        permutations: Tensor,
        pair_scores: Optional[Tensor],
        raw_future: Tensor,
        ground_truth: Tensor,
        edge_index: Tensor,
        future_mask: Optional[Tensor] = None,
        agent_mask: Optional[Tensor] = None,
        scene_index: Optional[Tensor] = None,
        relation_posterior: Optional[Tensor] = None,
        relation_prior: Optional[Tensor] = None,
        pair_gate: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None,
        lambda_fde: float = 1.0,
        lambda_rel_geom: float = 1.0,
        lambda_rel_end: float = 1.0,
        tau_joint: float = 0.5,
        tau_pair_target: float = 0.5,
        tau_pair_pred: float = 1.0,
        no_harm_margin: float = 0.0,
        weight_alignment: float = 1.0,
        weight_pair_score: float = 0.5,
        weight_pair_assignment: float = 0.5,
        weight_no_harm: float = 2.0,
        weight_perm_entropy: float = 0.02,
        weight_relation_prior: float = 0.02,
        weight_gate_reg: float = 0.0,
        pair_score_loss_type: str = "kl",
        pair_assignment_loss_type: str = "kl",
        gate_regularization: str = "none",
        gate_target_sparsity: float = 0.5,
        normalized_softmin: bool = True,
        return_details: bool = False,
) -> Dict[str, Tensor]:
    """Compute the complete configurable V4 coupling objective."""
    weights = {
        "alignment": _finite_nonnegative(weight_alignment, "weight_alignment"),
        "pair_score": _finite_nonnegative(weight_pair_score, "weight_pair_score"),
        "pair_assignment": _finite_nonnegative(
            weight_pair_assignment, "weight_pair_assignment"),
        "no_harm": _finite_nonnegative(weight_no_harm, "weight_no_harm"),
        "perm_entropy": _finite_nonnegative(
            weight_perm_entropy, "weight_perm_entropy"),
        "relation_prior": _finite_nonnegative(
            weight_relation_prior, "weight_relation_prior"),
        "gate_reg": _finite_nonnegative(weight_gate_reg, "weight_gate_reg"),
    }
    trajectory_cost, trajectory_details = trajectory_gt_costs(
        raw_future, ground_truth, future_mask, lambda_fde, agent_mask,
        return_details=True)
    effective_agent_mask = trajectory_details["valid_agent_mask"]
    alignment = alignment_losses(
        permutations, trajectory_cost, effective_agent_mask, scene_index,
        tau_joint, no_harm_margin, normalized_softmin)
    pair_cost, oracle_details = pair_oracle_costs(
        raw_future, ground_truth, edge_index, trajectory_cost, future_mask,
        agent_mask, lambda_fde, lambda_rel_geom, lambda_rel_end, scene_index,
        return_details=True)
    edge_mask = oracle_details["valid_edge_mask"]
    target_prob = pair_target_distribution(
        pair_cost, tau_pair_target, detach=True)

    if pair_scores is None:
        if weights["pair_score"] > 0 and pair_cost.shape[0]:
            raise ValueError("pair_scores is required when pair-score loss is enabled")
        loss_pair_score = _zero(permutations)
        pair_score_details: Dict[str, Tensor] = {}
    else:
        loss_pair_score, pair_score_details = pair_score_loss(
            pair_scores, target_prob=target_prob,
            prediction_temperature=tau_pair_pred,
            loss_type=pair_score_loss_type, edge_mask=edge_mask,
            edge_weight=edge_weight, return_details=True)
    loss_pair_assignment, assignment_details = pair_assignment_loss(
        permutations, edge_index, target_prob=target_prob,
        loss_type=pair_assignment_loss_type, edge_mask=edge_mask,
        edge_weight=edge_weight, scene_index=scene_index,
        return_details=True)
    loss_entropy = permutation_entropy_loss(
        permutations, effective_agent_mask)
    if relation_posterior is None and relation_prior is None:
        loss_relation = _zero(permutations)
    elif relation_posterior is None or relation_prior is None:
        raise ValueError("relation posterior and prior must be supplied together")
    else:
        loss_relation = relation_prior_kl_loss(
            relation_posterior, relation_prior, edge_mask=edge_mask)
    if pair_gate is None:
        loss_gate = _zero(permutations)
    else:
        loss_gate = gate_regularization_loss(
            pair_gate, gate_regularization, gate_target_sparsity, edge_mask)

    result = {
        "loss_alignment": alignment["loss_alignment"],
        "loss_pair_score": loss_pair_score,
        "loss_pair_assignment": loss_pair_assignment,
        "loss_no_harm": alignment["loss_no_harm"],
        "loss_perm_entropy": loss_entropy,
        "loss_relation_prior": loss_relation,
        "loss_gate_reg": loss_gate,
        "loss_identity": alignment["loss_identity"],
    }
    result["loss_total"] = (
        weights["alignment"] * result["loss_alignment"] +
        weights["pair_score"] * result["loss_pair_score"] +
        weights["pair_assignment"] * result["loss_pair_assignment"] +
        weights["no_harm"] * result["loss_no_harm"] +
        weights["perm_entropy"] * result["loss_perm_entropy"] +
        weights["relation_prior"] * result["loss_relation_prior"] +
        weights["gate_reg"] * result["loss_gate_reg"])
    if return_details:
        result.update({
            "trajectory_cost": trajectory_cost,
            "trajectory_ade": trajectory_details["ade"],
            "trajectory_fde": trajectory_details["fde"],
            "pair_oracle_cost": pair_cost,
            "pair_target_prob": target_prob,
            "valid_agent_mask": effective_agent_mask,
            "valid_edge_mask": edge_mask,
            "relative_assignment": assignment_details["relative_assignment"],
            "expected_slot_cost": alignment["expected_slot_cost"],
        })
        if pair_score_details:
            result["predicted_pair_log_prob"] = pair_score_details[
                "predicted_log_prob"]
    return result


def compute_scene_balanced_multiway_coupling_loss(
        permutations: Tensor,
        pair_scores: Optional[Tensor],
        raw_future: Tensor,
        ground_truth: Tensor,
        edge_index: Tensor,
        future_mask: Optional[Tensor] = None,
        agent_mask: Optional[Tensor] = None,
        scene_index: Optional[Tensor] = None,
        relation_posterior: Optional[Tensor] = None,
        relation_prior: Optional[Tensor] = None,
        pair_gate: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None,
        return_details: bool = False,
        **loss_kwargs,
) -> Dict[str, Tensor]:
    """Exactly average the existing single-scene V4 objective over scenes.

    This wrapper changes only reduction order.  Every scene is sliced from the
    packed sparse tensors, its edge indices are mapped back to local agent
    indices, and :func:`compute_multiway_coupling_loss` is called unchanged.
    Consequently each component is mathematically identical to evaluating the
    same windows one by one, including variable-N, variable-E, singleton and
    empty-graph cases.
    """
    num_agents = int(raw_future.shape[0])
    if scene_index is None:
        return compute_multiway_coupling_loss(
            permutations=permutations,
            pair_scores=pair_scores,
            raw_future=raw_future,
            ground_truth=ground_truth,
            edge_index=edge_index,
            future_mask=future_mask,
            agent_mask=agent_mask,
            scene_index=None,
            relation_posterior=relation_posterior,
            relation_prior=relation_prior,
            pair_gate=pair_gate,
            edge_weight=edge_weight,
            return_details=return_details,
            **loss_kwargs,
        )
    scene_index = torch.as_tensor(
        scene_index, device=raw_future.device, dtype=torch.long)
    if scene_index.shape != (num_agents,):
        raise ValueError(f"scene_index must have shape ({num_agents},)")
    edge_index = _validate_edges(
        edge_index, num_agents, raw_future.device, scene_index)
    scene_ids = torch.unique(scene_index, sorted=True)
    if scene_ids.numel() == 0:
        return compute_multiway_coupling_loss(
            permutations=permutations,
            pair_scores=pair_scores,
            raw_future=raw_future,
            ground_truth=ground_truth,
            edge_index=edge_index,
            future_mask=future_mask,
            agent_mask=agent_mask,
            scene_index=None,
            relation_posterior=relation_posterior,
            relation_prior=relation_prior,
            pair_gate=pair_gate,
            edge_weight=edge_weight,
            return_details=return_details,
            **loss_kwargs,
        )

    per_scene = []
    for scene_id in scene_ids:
        node_mask = scene_index.eq(scene_id)
        node_indices = node_mask.nonzero(as_tuple=False).flatten()
        global_to_local = torch.full(
            (num_agents,), -1, dtype=torch.long, device=raw_future.device)
        global_to_local[node_indices] = torch.arange(
            node_indices.numel(), device=raw_future.device)
        if edge_index.shape[1]:
            edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
            local_edge_index = global_to_local[edge_index[:, edge_mask]]
        else:
            edge_mask = torch.zeros(
                0, dtype=torch.bool, device=raw_future.device)
            local_edge_index = edge_index.clone()

        def node_slice(value):
            return None if value is None else value[node_indices]

        def edge_slice(value):
            return None if value is None else value[edge_mask]

        details = compute_multiway_coupling_loss(
            permutations=permutations[node_indices],
            pair_scores=edge_slice(pair_scores),
            raw_future=raw_future[node_indices],
            ground_truth=ground_truth[node_indices],
            edge_index=local_edge_index,
            future_mask=node_slice(future_mask),
            agent_mask=node_slice(agent_mask),
            scene_index=None,
            relation_posterior=edge_slice(relation_posterior),
            relation_prior=edge_slice(relation_prior),
            pair_gate=edge_slice(pair_gate),
            edge_weight=edge_slice(edge_weight),
            return_details=return_details,
            **loss_kwargs,
        )
        per_scene.append(details)

    scalar_names = (
        "loss_alignment", "loss_pair_score", "loss_pair_assignment",
        "loss_no_harm", "loss_perm_entropy", "loss_relation_prior",
        "loss_gate_reg", "loss_identity", "loss_total",
    )
    result = {
        name: torch.stack([details[name] for details in per_scene]).mean()
        for name in scalar_names
    }
    if return_details:
        # Per-scene details remain separate so no accidental cross-scene
        # interpretation of local edge/node axes is possible.
        result["per_scene"] = per_scene
        result["scene_ids"] = scene_ids
    return result


# Plural spelling retained as a convenient integration alias.
compute_multiway_coupling_losses = compute_multiway_coupling_loss
multiway_coupling_losses = compute_multiway_coupling_loss


def validate_hard_permutation(permutation: Tensor, num_samples: Optional[int] = None,
                              name: str = "permutation") -> Tensor:
    """Validate ``perm[n,s]=candidate`` and return it as ``long``."""
    if not isinstance(permutation, torch.Tensor) or permutation.ndim != 2:
        raise ValueError(f"{name} must have shape [N,K]")
    if num_samples is None:
        num_samples = permutation.shape[1]
    if permutation.shape[1] != num_samples or num_samples <= 0:
        raise ValueError(f"{name} must have K={num_samples} columns")
    if (permutation.is_floating_point() or permutation.is_complex()):
        if not torch.isfinite(permutation).all() or \
                not torch.equal(permutation, permutation.round()):
            raise ValueError(f"{name} must contain finite integer indices")
    permutation = permutation.to(dtype=torch.long)
    expected = torch.arange(
        num_samples, device=permutation.device).expand(permutation.shape[0], -1)
    if permutation.numel() and not torch.equal(
            permutation.sort(dim=-1).values, expected):
        raise ValueError(f"{name} has duplicate or missing candidate indices")
    return permutation


def gather_by_permutation(raw_future: Tensor, permutation: Tensor) -> Tensor:
    """Gather trajectories only; no coordinate interpolation is performed."""
    if raw_future.ndim < 2:
        raise ValueError("raw_future must have at least [N,K] dimensions")
    permutation = validate_hard_permutation(
        permutation, raw_future.shape[1])
    if permutation.shape[0] != raw_future.shape[0]:
        raise ValueError("permutation agent dimension must match raw_future")
    if permutation.device != raw_future.device:
        raise ValueError("permutation and raw_future must share a device")
    index = permutation.reshape(
        permutation.shape + (1,) * (raw_future.ndim - 2)).expand_as(raw_future)
    return raw_future.gather(dim=1, index=index)


def _hard_group_cost_diagnostics(
        trajectory_cost: Tensor,
        permutation: Tensor,
        group_index: Optional[Tensor] = None,
        agent_mask: Optional[Tensor] = None,
        eps: float = 1e-8,
) -> Dict[str, Tensor]:
    _check_floating_finite(trajectory_cost, "trajectory_cost")
    if trajectory_cost.ndim != 2 or trajectory_cost.shape[1] <= 0:
        raise ValueError("trajectory_cost must have shape [N,K]")
    num_agents, num_samples = trajectory_cost.shape
    permutation = validate_hard_permutation(permutation, num_samples)
    if permutation.shape[0] != num_agents:
        raise ValueError("permutation and trajectory_cost must share N")
    if permutation.device != trajectory_cost.device:
        raise ValueError("permutation and costs must share a device")
    valid_agent = _bool_mask(
        agent_mask, (num_agents,), trajectory_cost.device, "agent_mask")
    group_ids, compact = _canonical_groups(
        group_index, num_agents, trajectory_cost.device)
    num_groups = group_ids.numel()
    counts = torch.bincount(
        compact[valid_agent], minlength=num_groups).to(
            device=trajectory_cost.device, dtype=trajectory_cost.dtype)
    oracle_sum = trajectory_cost.new_zeros((num_groups,))
    raw_sum = trajectory_cost.new_zeros((num_groups, num_samples))
    aligned_sum = torch.zeros_like(raw_sum)
    if num_agents:
        valid_float = valid_agent.to(trajectory_cost.dtype)
        oracle_sum.index_add_(
            0, compact, trajectory_cost.min(dim=-1).values * valid_float)
        raw_sum.index_add_(
            0, compact, trajectory_cost * valid_float[:, None])
        aligned_cost = trajectory_cost.gather(1, permutation)
        aligned_sum.index_add_(
            0, compact, aligned_cost * valid_float[:, None])
    denominator = counts.clamp_min(1.0)
    oracle = oracle_sum / denominator
    raw = (raw_sum / denominator[:, None]).min(dim=-1).values \
        if num_groups else trajectory_cost.new_empty((0,))
    aligned = (aligned_sum / denominator[:, None]).min(dim=-1).values \
        if num_groups else trajectory_cost.new_empty((0,))
    valid_group = counts > 0
    oracle = torch.where(valid_group, oracle, torch.zeros_like(oracle))
    raw = torch.where(valid_group, raw, torch.zeros_like(raw))
    aligned = torch.where(valid_group, aligned, torch.zeros_like(aligned))
    headroom = (raw - oracle).clamp_min(0.0)
    delta = raw - aligned
    recovery = delta / headroom.clamp_min(_positive(eps, "eps"))
    recovery = torch.where(
        valid_group, recovery, torch.zeros_like(recovery))
    return {
        "group_ids": group_ids,
        "group_size": counts,
        "valid_group_mask": valid_group,
        "per_group_oracle": oracle,
        "per_group_raw": raw,
        "per_group_aligned": aligned,
        "per_group_headroom": headroom,
        "per_group_delta": delta,
        "per_group_recovery_ratio": recovery,
        "alignment_oracle": _mean_valid(oracle, valid_group, trajectory_cost),
        "raw_joint_cost": _mean_valid(raw, valid_group, trajectory_cost),
        "aligned_joint_cost": _mean_valid(aligned, valid_group, trajectory_cost),
        "alignment_headroom": _mean_valid(headroom, valid_group, trajectory_cost),
        "delta_joint": _mean_valid(delta, valid_group, trajectory_cost),
        "alignment_recovery_ratio": _mean_valid(
            recovery, valid_group, trajectory_cost),
    }


def alignment_headroom_diagnostics(
        trajectory_cost: Tensor,
        permutation: Tensor,
        scene_index: Optional[Tensor] = None,
        agent_mask: Optional[Tensor] = None,
        eps: float = 1e-8,
) -> Dict[str, Tensor]:
    """Oracle/raw/aligned/headroom/recovery diagnostics from hard assignments."""
    return _hard_group_cost_diagnostics(
        trajectory_cost, permutation, scene_index, agent_mask, eps)


def marginal_preservation_diagnostics(
        raw_future: Tensor,
        aligned_future: Tensor,
        ground_truth: Tensor,
        future_mask: Optional[Tensor] = None,
        agent_mask: Optional[Tensor] = None,
        tolerance: float = 1e-6,
) -> Dict[str, Tensor]:
    """Check hard alignment preserves marginal minADE and minFDE."""
    _validate_trajectories(raw_future, ground_truth)
    if aligned_future.shape != raw_future.shape:
        raise ValueError("aligned_future must have the same shape as raw_future")
    _check_floating_finite(aligned_future, "aligned_future")
    if aligned_future.device != raw_future.device or \
            aligned_future.dtype != raw_future.dtype:
        raise ValueError("raw and aligned futures must match device/dtype")
    tolerance = _finite_nonnegative(tolerance, "tolerance")
    _, raw_details = trajectory_gt_costs(
        raw_future, ground_truth, future_mask, 0.0, agent_mask, True)
    _, aligned_details = trajectory_gt_costs(
        aligned_future, ground_truth, future_mask, 0.0, agent_mask, True)
    valid = raw_details["valid_agent_mask"]
    raw_min_ade = raw_details["ade"].min(dim=-1).values
    aligned_min_ade = aligned_details["ade"].min(dim=-1).values
    raw_min_fde = raw_details["fde"].min(dim=-1).values
    aligned_min_fde = aligned_details["fde"].min(dim=-1).values
    ade_difference = (raw_min_ade - aligned_min_ade).abs()
    fde_difference = (raw_min_fde - aligned_min_fde).abs()
    max_ade = ade_difference[valid].max() if valid.any() else _zero(raw_future)
    max_fde = fde_difference[valid].max() if valid.any() else _zero(raw_future)
    preserved = (max_ade <= tolerance) & (max_fde <= tolerance)
    return {
        "raw_minADE_per_agent": raw_min_ade,
        "aligned_minADE_per_agent": aligned_min_ade,
        "raw_minFDE_per_agent": raw_min_fde,
        "aligned_minFDE_per_agent": aligned_min_fde,
        "Raw_minADE": _mean_valid(raw_min_ade, valid, raw_future),
        "Aligned_minADE": _mean_valid(aligned_min_ade, valid, aligned_future),
        "Raw_minFDE": _mean_valid(raw_min_fde, valid, raw_future),
        "Aligned_minFDE": _mean_valid(aligned_min_fde, valid, aligned_future),
        "marginal_ADE_max_abs_error": max_ade,
        "marginal_FDE_max_abs_error": max_fde,
        "marginal_preserved": preserved,
        "valid_agent_mask": valid,
    }


def assert_marginal_preservation(
        raw_future: Tensor,
        aligned_future: Tensor,
        ground_truth: Tensor,
        future_mask: Optional[Tensor] = None,
        agent_mask: Optional[Tensor] = None,
        tolerance: float = 1e-6,
) -> Dict[str, Tensor]:
    """Raise a precise assertion if marginal best-of-K metrics changed."""
    result = marginal_preservation_diagnostics(
        raw_future, aligned_future, ground_truth, future_mask, agent_mask,
        tolerance)
    if not bool(result["marginal_preserved"].item()):
        raise AssertionError(
            "hard alignment changed marginal metrics: "
            f"max |delta minADE|={result['marginal_ADE_max_abs_error'].item():.3e}, "
            f"max |delta minFDE|={result['marginal_FDE_max_abs_error'].item():.3e}")
    return result


def connected_component_labels(
        edge_index: Tensor,
        num_agents: int,
        scene_index: Optional[Tensor] = None,
        device=None,
) -> Tuple[Tensor, Tensor]:
    """Return deterministic component labels and per-agent component sizes."""
    if not isinstance(num_agents, int) or num_agents < 0:
        raise ValueError("num_agents must be a non-negative integer")
    if device is None:
        device = edge_index.device if isinstance(edge_index, torch.Tensor) \
            else torch.device("cpu")
    edges = _validate_edges(edge_index, num_agents, device, scene_index)
    if num_agents == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    parent = list(range(num_agents))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for left, right in edges.detach().cpu().t().tolist():
        union(int(left), int(right))
    root_to_label: Dict[int, int] = {}
    labels = []
    for node in range(num_agents):
        root = find(node)
        if root not in root_to_label:
            root_to_label[root] = len(root_to_label)
        labels.append(root_to_label[root])
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
    counts = torch.bincount(labels_tensor, minlength=len(root_to_label))
    return labels_tensor, counts[labels_tensor]


def size_bucket_label(size: int, bucket_type: str = "N") -> str:
    """Return the fixed V4 scene/component-size bucket label."""
    size = int(size)
    if size < 1:
        raise ValueError("size must be at least one")
    bucket_type = str(bucket_type).lower()
    prefix = "N" if bucket_type in {"n", "agent", "agents", "scene"} \
        else "comp_size" if bucket_type in {"component", "comp", "comp_size"} \
        else None
    if prefix is None:
        raise ValueError("bucket_type must be 'N' or 'component'")
    if size == 1:
        suffix = "=1"
    elif size == 2:
        suffix = "=2"
    elif size <= 5:
        suffix = "=3~5"
    elif size <= 10:
        suffix = "=6~10"
    else:
        suffix = ">10"
    return prefix + suffix


def aggregate_size_buckets(
        metrics: Mapping[str, Union[Tensor, Sequence[float]]],
        sizes: Union[Tensor, Sequence[int]],
        bucket_type: str = "N",
) -> Dict[str, Dict[str, float]]:
    """Aggregate aligned diagnostic vectors into fixed V4 size buckets."""
    size_tensor = torch.as_tensor(sizes, dtype=torch.long).detach().cpu()
    if size_tensor.ndim != 1 or (size_tensor < 1).any():
        raise ValueError("sizes must be a vector of positive integers")
    converted: Dict[str, Tensor] = {}
    for name, values in metrics.items():
        value = torch.as_tensor(values, dtype=torch.float64).detach().cpu()
        if value.ndim != 1 or value.shape[0] != size_tensor.shape[0]:
            raise ValueError(f"metric {name!r} must have shape [{size_tensor.shape[0]}]")
        if not torch.isfinite(value).all():
            raise ValueError(f"metric {name!r} contains NaN or Inf")
        converted[str(name)] = value
    canonical_sizes = (1, 2, 3, 6, 11)
    result: Dict[str, Dict[str, float]] = OrderedDict()
    for representative in canonical_sizes:
        label = size_bucket_label(representative, bucket_type)
        labels = [size_bucket_label(int(value), bucket_type)
                  for value in size_tensor.tolist()]
        selected = torch.tensor(
            [item == label for item in labels], dtype=torch.bool)
        bucket: Dict[str, float] = {"count": int(selected.sum().item())}
        for metric_name, values in converted.items():
            bucket[metric_name] = (
                float(values[selected].mean().item()) if selected.any() else 0.0)
        result[label] = bucket
    return result


def hard_alignment_diagnostics(
        raw_future: Tensor,
        ground_truth: Tensor,
        permutation: Tensor,
        future_mask: Optional[Tensor] = None,
        agent_mask: Optional[Tensor] = None,
        scene_index: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        lambda_fde: float = 1.0,
        tolerance: float = 1e-6,
        eps: float = 1e-8,
) -> Dict[str, object]:
    """Complete hard V4 oracle/headroom/joint/marginal/bucket diagnostics."""
    cost, details = trajectory_gt_costs(
        raw_future, ground_truth, future_mask, lambda_fde, agent_mask, True)
    permutation = validate_hard_permutation(permutation, raw_future.shape[1])
    aligned = gather_by_permutation(raw_future, permutation)
    result: Dict[str, object] = dict(_hard_group_cost_diagnostics(
        cost, permutation, scene_index, details["valid_agent_mask"], eps))
    ade_group = _hard_group_cost_diagnostics(
        details["ade"], permutation, scene_index,
        details["valid_agent_mask"], eps)
    fde_group = _hard_group_cost_diagnostics(
        details["fde"], permutation, scene_index,
        details["valid_agent_mask"], eps)
    result.update({
        "Raw_JADE": ade_group["raw_joint_cost"],
        "Aligned_JADE": ade_group["aligned_joint_cost"],
        "delta_JADE": ade_group["delta_joint"],
        "Raw_JFDE": fde_group["raw_joint_cost"],
        "Aligned_JFDE": fde_group["aligned_joint_cost"],
        "delta_JFDE": fde_group["delta_joint"],
        "per_scene_Raw_JADE": ade_group["per_group_raw"],
        "per_scene_Aligned_JADE": ade_group["per_group_aligned"],
        "per_scene_Raw_JFDE": fde_group["per_group_raw"],
        "per_scene_Aligned_JFDE": fde_group["per_group_aligned"],
    })
    marginal = marginal_preservation_diagnostics(
        raw_future, aligned, ground_truth, future_mask, agent_mask, tolerance)
    result.update(marginal)

    valid_scene = result["valid_group_mask"]
    sizes = result["group_size"][valid_scene].long()
    scene_metrics = {
        "Raw_JADE": ade_group["per_group_raw"][valid_scene],
        "JADE": ade_group["per_group_aligned"][valid_scene],
        "Raw_JFDE": fde_group["per_group_raw"][valid_scene],
        "JFDE": fde_group["per_group_aligned"][valid_scene],
        "recovery_ratio": result["per_group_recovery_ratio"][valid_scene],
    }
    result["N_size_buckets"] = aggregate_size_buckets(
        scene_metrics, sizes, "N")

    if edge_index is not None:
        component_label, _ = connected_component_labels(
            edge_index, raw_future.shape[0], scene_index, raw_future.device)
        component = _hard_group_cost_diagnostics(
            cost, permutation, component_label,
            details["valid_agent_mask"], eps)
        component_ade = _hard_group_cost_diagnostics(
            details["ade"], permutation, component_label,
            details["valid_agent_mask"], eps)
        component_fde = _hard_group_cost_diagnostics(
            details["fde"], permutation, component_label,
            details["valid_agent_mask"], eps)
        comp_valid = component["valid_group_mask"]
        result["component_size_buckets"] = aggregate_size_buckets({
            "Raw_JADE": component_ade["per_group_raw"][comp_valid],
            "JADE": component_ade["per_group_aligned"][comp_valid],
            "Raw_JFDE": component_fde["per_group_raw"][comp_valid],
            "JFDE": component_fde["per_group_aligned"][comp_valid],
            "recovery_ratio": component["per_group_recovery_ratio"][comp_valid],
        }, component["group_size"][comp_valid].long(), "component")
        result["component_labels"] = component_label
    return result


__all__ = [
    "trajectory_gt_costs",
    "softmin_cost",
    "expected_joint_slot_costs",
    "alignment_losses",
    "pair_oracle_costs",
    "pair_target_distribution",
    "pair_score_loss",
    "pair_score_distribution_loss",
    "relative_assignment_matrices",
    "pair_assignment_loss",
    "permutation_entropy_loss",
    "relation_prior_kl_loss",
    "gate_regularization_loss",
    "compute_multiway_coupling_loss",
    "compute_multiway_coupling_losses",
    "multiway_coupling_losses",
    "validate_hard_permutation",
    "gather_by_permutation",
    "alignment_headroom_diagnostics",
    "marginal_preservation_diagnostics",
    "assert_marginal_preservation",
    "connected_component_labels",
    "size_bucket_label",
    "aggregate_size_buckets",
    "hard_alignment_diagnostics",
]

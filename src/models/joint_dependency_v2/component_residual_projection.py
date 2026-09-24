"""Parameter-free component-relative projection for Stage-B V2-A."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class ComponentMetadata:
    """Immutable connected-component metadata for one sparse graph batch."""

    degree: torch.Tensor
    active_mask: torch.Tensor
    active_index: torch.Tensor
    component_id: torch.Tensor
    component_count: torch.Tensor
    num_components: int


def build_component_metadata(
    edge_index: torch.Tensor,
    num_agents: int,
    scene_index: Optional[torch.Tensor] = None,
) -> ComponentMetadata:
    """Build deterministic undirected active-component metadata once.

    Component IDs follow ascending minimum agent index. Isolated agents have
    ``component_id=-1`` and are excluded from all projection reductions.
    """
    if not isinstance(num_agents, int) or num_agents < 0:
        raise ValueError("num_agents must be a non-negative integer")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2,E]")
    if edge_index.dtype != torch.long:
        raise ValueError("edge_index must have dtype torch.long")
    device = edge_index.device
    if scene_index is not None:
        if scene_index.shape != (num_agents,):
            raise ValueError("scene_index must have shape [N]")
        if scene_index.device != device:
            raise ValueError("scene_index and edge_index must share a device")

    degree = torch.zeros(num_agents, dtype=torch.long, device=device)
    edge_count = edge_index.shape[1]
    if edge_count == 0:
        inactive = torch.zeros(num_agents, dtype=torch.bool, device=device)
        return ComponentMetadata(
            degree=degree,
            active_mask=inactive,
            active_index=torch.empty(0, dtype=torch.long, device=device),
            component_id=torch.full(
                (num_agents,), -1, dtype=torch.long, device=device),
            component_count=torch.empty(0, dtype=torch.long, device=device),
            num_components=0,
        )

    if bool(((edge_index < 0) | (edge_index >= num_agents)).any().item()):
        raise ValueError("edge_index contains an out-of-range agent index")
    src, dst = edge_index
    if bool((src == dst).any().item()):
        raise ValueError("edge_index must not contain self edges")
    if scene_index is not None and bool(
            (scene_index[src] != scene_index[dst]).any().item()):
        raise ValueError("edge_index contains a cross-scene edge")

    ones = torch.ones(edge_count, dtype=torch.long, device=device)
    degree.index_add_(0, src, ones)
    degree.index_add_(0, dst, ones)
    active_mask = degree > 0
    active_index = active_mask.nonzero(as_tuple=False).flatten()

    # Connectivity is static window metadata. A single CPU union-find avoids
    # device synchronization inside any diffusion step and gives canonical,
    # edge-order-independent component IDs.
    parent = list(range(num_agents))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    for left, right in edge_index.detach().cpu().t().tolist():
        union(int(left), int(right))

    active_cpu = active_index.detach().cpu().tolist()
    roots = sorted({find(int(agent)) for agent in active_cpu})
    compact = {root: component for component, root in enumerate(roots)}
    component_cpu = [-1] * num_agents
    counts = [0] * len(roots)
    for agent in active_cpu:
        component = compact[find(int(agent))]
        component_cpu[int(agent)] = component
        counts[component] += 1
    component_id = torch.tensor(
        component_cpu, dtype=torch.long, device=device)
    component_count = torch.tensor(counts, dtype=torch.long, device=device)
    return ComponentMetadata(
        degree=degree,
        active_mask=active_mask,
        active_index=active_index,
        component_id=component_id,
        component_count=component_count,
        num_components=len(roots),
    )


def component_zero_mean_projection(
    raw_delta: torch.Tensor,
    metadata: ComponentMetadata,
) -> torch.Tensor:
    """Project ``[N,T,2]`` residuals into component-relative subspaces."""
    if raw_delta.ndim != 3 or raw_delta.shape[-1] != 2:
        raise ValueError("raw_delta must have shape [N,T,2]")
    num_agents = raw_delta.shape[0]
    for name, tensor, shape in (
            ("degree", metadata.degree, (num_agents,)),
            ("active_mask", metadata.active_mask, (num_agents,)),
            ("component_id", metadata.component_id, (num_agents,))):
        if tensor.shape != shape:
            raise ValueError(f"metadata {name} has an invalid shape")
        if tensor.device != raw_delta.device:
            raise ValueError(f"metadata {name} is on the wrong device")
    if metadata.active_index.device != raw_delta.device or \
            metadata.component_count.device != raw_delta.device:
        raise ValueError("component metadata is on the wrong device")
    if metadata.component_count.shape != (metadata.num_components,):
        raise ValueError("component_count has an invalid shape")

    raw = raw_delta.float()
    raw_is_finite = torch.isfinite(raw).all()
    if raw.device.type == "cuda":
        torch._assert_async(raw_is_finite, "raw_delta contains NaN or Inf")
    elif not bool(raw_is_finite.item()):
        raise ValueError("raw_delta contains NaN or Inf")
    if metadata.num_components == 0:
        return torch.zeros_like(raw)

    active = metadata.active_index
    component = metadata.component_id.index_select(0, active)
    sums = raw.new_zeros((metadata.num_components,) + tuple(raw.shape[1:]))
    sums.index_add_(0, component, raw.index_select(0, active))
    counts = metadata.component_count.to(raw.dtype).view(-1, 1, 1)
    means = sums / counts
    active_relative = raw.index_select(0, active) - \
        means.index_select(0, component)
    projected = torch.zeros_like(raw)
    projected.index_copy_(0, active, active_relative)
    projected_is_finite = torch.isfinite(projected).all()
    if projected.device.type == "cuda":
        torch._assert_async(
            projected_is_finite, "projected residual contains NaN or Inf")
    elif not bool(projected_is_finite.item()):
        raise ValueError("projected residual contains NaN or Inf")
    return projected


__all__ = [
    "ComponentMetadata",
    "build_component_metadata",
    "component_zero_mean_projection",
]

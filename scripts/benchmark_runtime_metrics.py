"""Rejected logging candidate retained only for benchmark reproduction."""
from __future__ import annotations

import torch


class UnifiedMetricBuffer:
    """Keep the historical Python summation order, with one optional GPU copy.

    Only detached scalar tensors are retained. Losses used for backward are
    untouched. Flush at an optimizer-update boundary, including a partial one.
    """

    def __init__(self, deferred: bool = False) -> None:
        self.deferred = bool(deferred)
        self.groups = []

    def begin_microbatch(self) -> None:
        self.groups.append([])

    def add(self, total, pair, agent_part, bridge_part, agent_weight, scene_weight):
        if not self.groups:
            raise RuntimeError("begin_microbatch must precede add")
        values = (total.detach(), pair.detach(), agent_part.detach(),
                  (bridge_part - agent_part).detach())
        if not self.deferred:
            values = tuple(float(value) for value in values)
        self.groups[-1].append((values, float(agent_weight), float(scene_weight)))

    def flush(self) -> list[tuple[float, float, float]]:
        if not self.groups:
            return []
        if any(not group for group in self.groups):
            raise RuntimeError("cannot flush an empty logical microbatch")
        copied = None
        if self.deferred:
            scalars = [value for group in self.groups for row, _, _ in group for value in row]
            # float() matches each scalar's original value. CPU/Python addition
            # below is unchanged; a GPU reduction would change its rounding.
            copied = iter(torch.stack(scalars).cpu().tolist())
        result = []
        for group in self.groups:
            total_value = pair_value = bridge_value = 0.0
            for values, agent_weight, scene_weight in group:
                total, pair, agent_part, geometry_part = (
                    tuple(next(copied) for _ in range(4)) if copied is not None else values)
                total_value += total
                pair_value += scene_weight * pair
                bridge_value += agent_weight * agent_part + scene_weight * geometry_part
            result.append((total_value, pair_value, bridge_value))
        self.groups.clear()
        return result

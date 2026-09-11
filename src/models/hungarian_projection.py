"""Hard projection of soft candidate-to-slot assignments.

The convention in this module is deliberately explicit and shared by the V4
trajectory coupler:

``assignment[..., candidate, slot]``
    gives the (soft or hard) weight of placing ``candidate`` in joint sample
    ``slot``.

Consequently ``permutation[..., slot] = candidate`` and aligned trajectories
are formed only with ``raw.gather(sample_dim, permutation)``.  The default
projection is the globally optimal linear-sum assignment from SciPy.  A
deterministic greedy projection is retained as an explicitly reported fallback
and as an ablation; it is not silently presented as Hungarian.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Tuple

import torch
from torch import nn

try:  # SciPy is preferred, but inference remains usable without it.
    from scipy.optimize import linear_sum_assignment as _linear_sum_assignment
except ImportError as _scipy_import_error:  # pragma: no cover - CI has SciPy.
    _linear_sum_assignment = None
else:
    _scipy_import_error = None


def validate_bijection(permutation: torch.Tensor, num_samples: int | None = None) -> None:
    """Raise ``ValueError`` unless every row is a permutation of ``0..K-1``.

    ``permutation`` may be ``[K]`` or batched ``[..., K]``.  Validation is
    strict: duplicate, missing, negative and out-of-range indices all fail.
    """
    if permutation.ndim < 1:
        raise ValueError("permutation must have at least one dimension")
    if permutation.dtype == torch.bool or torch.is_floating_point(permutation):
        raise TypeError("permutation must use an integer dtype")
    size = permutation.shape[-1] if num_samples is None else int(num_samples)
    if size <= 0 or permutation.shape[-1] != size:
        raise ValueError("permutation's final dimension must equal positive K")
    expected = torch.arange(size, device=permutation.device, dtype=permutation.dtype)
    expected = expected.expand(permutation.reshape(-1, size).shape[0], -1)
    observed = permutation.reshape(-1, size).sort(dim=-1).values
    if not torch.equal(observed, expected):
        raise ValueError(
            "Projection is not a bijection: every candidate index must appear "
            "exactly once in each permutation"
        )


def _greedy_max_assignment(score: torch.Tensor) -> torch.Tensor:
    """Return a deterministic greedy ``perm[slot] = candidate`` assignment."""
    size = score.shape[0]
    remaining_candidates = list(range(size))
    remaining_slots = list(range(size))
    permutation = torch.empty(size, dtype=torch.long, device=score.device)
    detached = score.detach()
    while remaining_candidates:
        candidate_index = torch.as_tensor(
            remaining_candidates, dtype=torch.long, device=score.device)
        slot_index = torch.as_tensor(
            remaining_slots, dtype=torch.long, device=score.device)
        submatrix = detached.index_select(0, candidate_index).index_select(1, slot_index)
        flat = int(submatrix.reshape(-1).argmax().item())
        local_candidate = flat // len(remaining_slots)
        local_slot = flat % len(remaining_slots)
        candidate = remaining_candidates.pop(local_candidate)
        slot = remaining_slots.pop(local_slot)
        permutation[slot] = candidate
    return permutation


def _project_one(score: torch.Tensor, method: str) -> Tuple[torch.Tensor, str]:
    if method == "hungarian" and _linear_sum_assignment is not None:
        candidate, slot = _linear_sum_assignment(-score.detach().to("cpu").double().numpy())
        permutation = torch.empty(score.shape[0], dtype=torch.long, device=score.device)
        permutation[torch.as_tensor(slot, dtype=torch.long, device=score.device)] = \
            torch.as_tensor(candidate, dtype=torch.long, device=score.device)
        return permutation, "hungarian"
    return _greedy_max_assignment(score), "greedy"


def project_permutations(
    probability: torch.Tensor,
    method: str = "hungarian",
    *,
    allow_greedy_fallback: bool = True,
) -> Dict[str, Any]:
    """Project one or many soft assignments to strict hard permutations.

    Parameters
    ----------
    probability:
        ``[K,K]`` or ``[N,K,K]`` finite scores.  Rows are original candidate
        indices and columns are joint-sample slots.  Values need not already be
        doubly stochastic because assignment projection only uses their order.
    method:
        ``"hungarian"`` (default) or the explicit ``"greedy"`` ablation.
    allow_greedy_fallback:
        If SciPy is unavailable, permit a deterministic greedy fallback.  A
        warning is emitted and the returned metadata records the fallback.

    Returns
    -------
    dict
        ``permutation`` has ``perm[..., slot] = candidate``;
        ``hard_assignment`` follows ``[..., candidate, slot]``; and
        ``metadata`` reports the requested/actual method, fallback state,
        objective, batch size and convention.
    """
    if method not in {"hungarian", "greedy"}:
        raise ValueError("method must be 'hungarian' or 'greedy'")
    if probability.ndim not in {2, 3}:
        raise ValueError("probability must have shape [K,K] or [N,K,K]")
    if probability.shape[-2] != probability.shape[-1]:
        raise ValueError("probability matrices must be square")
    if probability.shape[-1] <= 0:
        raise ValueError("probability matrices must be non-empty")
    if not torch.is_floating_point(probability):
        raise TypeError("probability must be floating point")
    if not bool(torch.isfinite(probability).all()):
        raise ValueError("probability contains NaN or Inf")

    fallback = method == "hungarian" and _linear_sum_assignment is None
    if fallback and not allow_greedy_fallback:
        raise RuntimeError("SciPy is required for Hungarian projection") from _scipy_import_error
    if fallback:
        warnings.warn(
            "SciPy is unavailable; falling back from Hungarian to greedy "
            "permutation projection. This is an explicit lower-quality fallback.",
            RuntimeWarning,
            stacklevel=2,
        )

    was_unbatched = probability.ndim == 2
    batched = probability.unsqueeze(0) if was_unbatched else probability
    permutations = []
    actual_methods = []
    for score in batched:
        permutation, actual_method = _project_one(score, method)
        validate_bijection(permutation, score.shape[0])
        permutations.append(permutation)
        actual_methods.append(actual_method)
    permutation = torch.stack(permutations, dim=0)

    # hard[candidate, slot] = 1, whereas one_hot(perm) is [slot, candidate].
    hard_assignment = torch.nn.functional.one_hot(
        permutation, num_classes=batched.shape[-1]
    ).transpose(-1, -2).to(dtype=probability.dtype)
    selected = batched.gather(1, permutation.unsqueeze(1)).squeeze(1)
    objective = selected.sum(dim=-1)
    actual_method = actual_methods[0] if len(set(actual_methods)) == 1 else "mixed"
    metadata: Dict[str, Any] = {
        "requested_method": method,
        "actual_method": actual_method,
        "used_scipy": actual_method == "hungarian",
        "fallback_used": bool(fallback),
        "fallback_reason": "scipy_unavailable" if fallback else None,
        "objective": objective[0] if was_unbatched else objective,
        "batch_size": int(batched.shape[0]),
        "num_samples": int(batched.shape[-1]),
        "convention": "matrix[candidate,slot]; permutation[slot]=candidate",
    }

    if was_unbatched:
        permutation = permutation[0]
        hard_assignment = hard_assignment[0]
    validate_bijection(permutation, probability.shape[-1])
    return {
        "permutation": permutation,
        "hard_assignment": hard_assignment,
        "assignment": hard_assignment,
        "metadata": metadata,
    }


def project_to_permutation(
    probability: torch.Tensor,
    method: str = "hungarian",
    *,
    allow_greedy_fallback: bool = True,
    return_metadata: bool = False,
):
    """Convenience wrapper returning ``perm`` or ``(perm, metadata)``.

    The richer :func:`project_permutations` result is useful when the hard
    assignment matrix is also needed.  This wrapper matches the concise V4
    pseudocode while still allowing projection metadata to be requested.
    """
    result = project_permutations(
        probability, method=method, allow_greedy_fallback=allow_greedy_fallback
    )
    if return_metadata:
        return result["permutation"], result["metadata"]
    return result["permutation"]


class HungarianProjection(nn.Module):
    """Stateless module wrapper returning projection and metadata dictionaries."""

    def __init__(self, method: str = "hungarian", allow_greedy_fallback: bool = True):
        super().__init__()
        if method not in {"hungarian", "greedy"}:
            raise ValueError("method must be 'hungarian' or 'greedy'")
        self.method = method
        self.allow_greedy_fallback = bool(allow_greedy_fallback)

    def forward(self, probability: torch.Tensor) -> Dict[str, Any]:
        return project_permutations(
            probability,
            method=self.method,
            allow_greedy_fallback=self.allow_greedy_fallback,
        )


__all__ = [
    "HungarianProjection",
    "project_permutations",
    "project_to_permutation",
    "validate_bijection",
]

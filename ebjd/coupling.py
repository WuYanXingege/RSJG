"""Marginal-preserving joint coupling for frozen endpoint proposal banks."""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .data import SceneBatch
from .proposals import file_sha256


def couple_scene(
    proposals: np.ndarray, reference: np.ndarray, valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each agent's proposals to shared reference-world slots exactly.

    ``permutation[k, i]`` is the native proposal index used by output world
    ``k`` for agent ``i``. Invalid agents and single-agent scenes retain the
    identity mapping. Ground truth is deliberately absent from this API.
    """
    if proposals.shape != reference.shape or proposals.ndim != 3:
        raise ValueError("proposals and reference must have equal [K,N,2] shape")
    if proposals.shape[-1] != 2 or valid.shape != proposals.shape[1:2]:
        raise ValueError("invalid endpoint or valid-mask shape")
    worlds, agents, _ = proposals.shape
    identity = np.broadcast_to(
        np.arange(worlds, dtype=np.int64)[:, None], (worlds, agents)).copy()
    coupled = proposals.copy()
    valid_ids = np.flatnonzero(valid)
    if len(valid_ids) <= 1:
        return coupled, identity
    for agent in valid_ids:
        difference = (
            reference[:, None, agent].astype(np.float64)
            - proposals[None, :, agent].astype(np.float64))
        rows, columns = linear_sum_assignment(np.square(difference).sum(-1))
        identity[rows, agent] = columns
        coupled[:, agent] = proposals[identity[:, agent], agent]
        if not np.array_equal(np.sort(identity[:, agent]), np.arange(worlds)):
            raise RuntimeError("linear assignment did not produce a permutation")
    return coupled, identity


def couple_bank(
    proposals: np.ndarray, reference: np.ndarray, valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply :func:`couple_scene` to an identity-aligned complete bank."""
    if proposals.shape != reference.shape or proposals.ndim != 4:
        raise ValueError("bank endpoints must have equal [S,K,N,2] shape")
    if valid.shape != (proposals.shape[0], proposals.shape[2]):
        raise ValueError("bank valid mask must be [S,N]")
    coupled = np.empty_like(proposals)
    permutations = np.empty(proposals.shape[:-1], dtype=np.int64)
    for scene in range(proposals.shape[0]):
        coupled[scene], permutations[scene] = couple_scene(
            proposals[scene], reference[scene], valid[scene])
    return coupled, permutations


def gather_worlds(values: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    """Gather proposal-aligned values with output-world to native indices.

    Values must be ``[B,K,N,...]`` and permutation ``[B,K,N]``. This is used
    for both endpoints and initial bridge noise, preventing a coupling change
    from also changing which random noise belongs to a native proposal.
    """
    if values.ndim < 4 or permutation.ndim != 3:
        raise ValueError("values must be [B,K,N,...] and permutation [B,K,N]")
    if values.shape[:3] != permutation.shape:
        raise ValueError("world/agent dimensions differ")
    index = permutation.to(device=values.device, dtype=torch.long)
    index = index.view(*index.shape, *([1] * (values.ndim - 3)))
    index = index.expand_as(values)
    return torch.gather(values, dim=1, index=index)


def relative_pairing_change(permutation: np.ndarray, valid: np.ndarray) -> dict:
    """Measure genuine pair changes, invariant to common global relabeling."""
    if permutation.ndim != 2 or valid.shape != permutation.shape[1:2]:
        raise ValueError("permutation must be [K,N] and valid [N]")
    worlds = permutation.shape[0]
    valid_ids = np.flatnonzero(valid)
    changes: list[float] = []
    for left_offset, left in enumerate(valid_ids):
        for right in valid_ids[left_offset + 1:]:
            mapping = np.empty(worlds, dtype=np.int64)
            mapping[permutation[:, left]] = permutation[:, right]
            changes.append(float(np.mean(mapping != np.arange(worlds))))
    return {
        "pair_count": len(changes),
        "mean_relative_change": float(np.mean(changes)) if changes else 0.0,
        "changed_pair_count": int(sum(value > 0.0 for value in changes)),
    }


class MPJCEndpointBank:
    """Identity-bound G/R/H endpoint bank with frozen MPJC permutations."""

    def __init__(
        self, path: str, *, expected_candidate_sha256: str,
        expected_manifest_sha256: str, expected_split: str,
    ) -> None:
        self.path = str(path)
        self.sha256 = file_sha256(path)
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "native_endpoints", "reference_endpoints", "coupled_endpoints",
                "permutation", "truth", "valid", "scene_ids", "agent_ids",
                "metadata_json",
            }
            if set(archive.files) != required:
                raise ValueError("MPJC bank members differ from frozen schema")
            self.native = torch.from_numpy(archive["native_endpoints"].copy()).float()
            self.reference = torch.from_numpy(archive["reference_endpoints"].copy()).float()
            self.coupled = torch.from_numpy(archive["coupled_endpoints"].copy()).float()
            self.permutation = torch.from_numpy(archive["permutation"].copy()).long()
            self.truth = torch.from_numpy(archive["truth"].copy()).float()
            self.valid = torch.from_numpy(archive["valid"].copy()).bool()
            self.scene_ids = [str(value) for value in archive["scene_ids"]]
            self.agent_ids = [[str(value) for value in row] for row in archive["agent_ids"]]
            import json
            self.metadata = json.loads(str(archive["metadata_json"].item()))
        checks = {
            "candidate_checkpoint_sha256": expected_candidate_sha256,
            "manifest_sha256": expected_manifest_sha256,
            "split": expected_split, "worlds": 20, "ddim_steps": 20,
            "coordinate_unit": "m",
        }
        mismatch = {
            key: {"expected": value, "actual": self.metadata.get(key)}
            for key, value in checks.items() if self.metadata.get(key) != value}
        if mismatch:
            raise ValueError(f"MPJC bank identity mismatch: {mismatch}")
        if self.native.shape != self.coupled.shape or self.native.ndim != 4:
            raise ValueError("MPJC endpoints must have equal [S,K,N,2] shape")
        if self.permutation.shape != self.native.shape[:-1]:
            raise ValueError("MPJC permutation dimensions differ")
        self._index = {scene_id: index for index, scene_id in enumerate(self.scene_ids)}

    def batch(
        self, batch: SceneBatch, mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mode not in {"native", "coupled"}:
            raise ValueError("MPJC mode must be native or coupled")
        source_points = self.native if mode == "native" else self.coupled
        points = torch.zeros(
            len(batch.scene_ids), 20, batch.observed.shape[1], 2,
            dtype=torch.float32)
        permutations = torch.arange(20).view(1, 20, 1).expand(
            len(batch.scene_ids), 20, batch.observed.shape[1]).clone()
        for local, (scene_id, identities) in enumerate(zip(
            batch.scene_ids, batch.agent_ids, strict=True,
        )):
            if scene_id not in self._index:
                raise KeyError(f"scene is absent from MPJC bank: {scene_id}")
            source = self._index[scene_id]
            count = int(self.valid[source].sum())
            if self.agent_ids[source][:count] != [str(value) for value in identities]:
                raise ValueError(f"MPJC agent identity/order mismatch: {scene_id}")
            if count != int(batch.valid[local].sum()):
                raise ValueError(f"MPJC valid count mismatch: {scene_id}")
            points[local, :, :count] = source_points[source, :, :count]
            if mode == "coupled":
                permutations[local, :, :count] = self.permutation[source, :, :count]
        return points, permutations

"""Identity-bound frozen native-goal proposal banks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .data import SceneBatch


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class NativeProposalBank:
    """A complete split bank with explicit scene and agent identity checks."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_checkpoint_sha256: str,
        expected_manifest_sha256: str,
        expected_split: str,
    ) -> None:
        self.path = Path(path).resolve()
        self.sha256 = file_sha256(self.path)
        with np.load(self.path, allow_pickle=False) as archive:
            required = {
                "endpoints", "truth", "valid", "scene_ids", "agent_ids",
                "metadata_json",
            }
            if set(archive.files) != required:
                raise ValueError("proposal bank members differ from the frozen schema")
            self.endpoints = torch.from_numpy(archive["endpoints"].copy()).float()
            self.truth = torch.from_numpy(archive["truth"].copy()).float()
            self.valid = torch.from_numpy(archive["valid"].copy()).bool()
            self.scene_ids = [str(value) for value in archive["scene_ids"]]
            self.agent_ids = [[str(value) for value in row]
                              for row in archive["agent_ids"]]
            self.metadata = json.loads(str(archive["metadata_json"].item()))
        if self.endpoints.ndim != 4 or self.endpoints.shape[1] != 20:
            raise ValueError("proposal bank endpoints must be [S,20,N,2]")
        if self.endpoints.shape[-1] != 2 or self.truth.shape != self.endpoints.shape[:1] + self.endpoints.shape[2:]:
            raise ValueError("proposal bank endpoint/truth dimensions differ")
        if self.valid.shape != self.truth.shape[:2]:
            raise ValueError("proposal bank valid mask dimensions differ")
        checks = {
            "checkpoint_sha256": expected_checkpoint_sha256,
            "manifest_sha256": expected_manifest_sha256,
            "split": expected_split,
            "worlds": 20,
            "coordinate_unit": "m",
        }
        mismatches = {
            name: {"expected": expected, "actual": self.metadata.get(name)}
            for name, expected in checks.items()
            if self.metadata.get(name) != expected
        }
        if mismatches:
            raise ValueError(f"proposal bank identity mismatch: {mismatches}")
        if len(set(self.scene_ids)) != len(self.scene_ids):
            raise ValueError("proposal bank contains duplicate scene IDs")
        self._index = {scene_id: index for index, scene_id in enumerate(self.scene_ids)}

    def batch(self, batch: SceneBatch) -> torch.Tensor:
        """Return padded proposals in exactly the batch's agent order."""
        result = torch.zeros(
            len(batch.scene_ids), 20, batch.observed.shape[1], 2,
            dtype=torch.float32)
        for local, (scene_id, identities) in enumerate(zip(
            batch.scene_ids, batch.agent_ids, strict=True,
        )):
            if scene_id not in self._index:
                raise KeyError(f"scene is absent from proposal bank: {scene_id}")
            source = self._index[scene_id]
            count = int(self.valid[source].sum())
            source_ids = self.agent_ids[source][:count]
            if source_ids != [str(value) for value in identities]:
                raise ValueError(f"proposal agent identity/order mismatch: {scene_id}")
            if count != int(batch.valid[local].sum()):
                raise ValueError(f"proposal valid count mismatch: {scene_id}")
            result[local, :, :count] = self.endpoints[source, :, :count]
        return result

    @staticmethod
    def select_training_world(
        proposals: torch.Tensor, future: torch.Tensor, valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stable scene-level GT matching without constructing hybrid worlds."""
        error = torch.linalg.vector_norm(
            proposals.float() - future[:, None, :, -1, :].float(), dim=-1)
        cost = (error * valid[:, None].float()).sum(-1) / valid.sum(-1, keepdim=True)
        world = cost.argmin(dim=1)
        selected = proposals[torch.arange(proposals.shape[0]), world]
        return selected, world


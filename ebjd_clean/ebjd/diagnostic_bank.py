"""Portable, identity-checked trajectory banks for frozen-model diagnostics.

The format deliberately uses only JSON plus non-object NPZ arrays.  A bank is
therefore readable with ``allow_pickle=False`` and does not depend on either
model implementation once export has completed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


BANK_SCHEMA = "ebjd-trajectory-bank-v1"
SEED_MANIFEST_SCHEMA = "ebjd-trajectory-bank-seed-manifest-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def canonical_source(value: str) -> str:
    """Map an absolute ETH/UCY source filename to its stable sequence stem."""
    return Path(str(value)).stem


def canonical_window_key(source: str, frame_ids: Iterable[int]) -> str:
    frames = [int(value) for value in frame_ids]
    if len(frames) != 20 or len(set(frames)) != 20:
        raise ValueError("a canonical 8->12 window requires 20 unique frames")
    return f"{canonical_source(source)}:{frames[0]}-{frames[-1]}"


def atomic_json(payload: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_npz(path: str | Path, **arrays: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class BankScene:
    predictions: np.ndarray
    future: np.ndarray
    history: np.ndarray
    frame_ids: np.ndarray
    agent_ids: np.ndarray
    source_sequence: str
    scene_id: str
    window_key: str
    dt_seconds: float

    @property
    def agents(self) -> int:
        return int(self.future.shape[0])


def validate_scene(scene: BankScene, *, worlds: int = 20) -> None:
    prediction = np.asarray(scene.predictions)
    future = np.asarray(scene.future)
    history = np.asarray(scene.history)
    frame_ids = np.asarray(scene.frame_ids)
    agent_ids = np.asarray(scene.agent_ids)
    if prediction.shape != (worlds, future.shape[0], 12, 2):
        raise ValueError(
            f"predictions must be [{worlds},N,12,2], got {prediction.shape}")
    if future.ndim != 3 or future.shape[1:] != (12, 2):
        raise ValueError("future must be [N,12,2]")
    if history.shape != (future.shape[0], 8, 2):
        raise ValueError("history must be [N,8,2]")
    if frame_ids.shape != (20,) or len(set(frame_ids.tolist())) != 20:
        raise ValueError("frame_ids must contain 20 unique entries")
    if agent_ids.shape != (future.shape[0],):
        raise ValueError("agent_ids must have one entry per valid agent")
    identifiers = [str(value) for value in agent_ids.tolist()]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("agent_ids are duplicated")
    if future.shape[0] < 1:
        raise ValueError("a bank scene cannot be empty")
    for name, value in (
        ("predictions", prediction), ("future", future), ("history", history),
    ):
        if value.dtype != np.float32:
            raise TypeError(f"{name} must be metre-space float32")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or Inf")
    if not math.isclose(float(scene.dt_seconds), 0.4, abs_tol=1e-7):
        raise ValueError("dt_seconds must be 0.4")
    expected = canonical_window_key(scene.source_sequence, frame_ids)
    if scene.window_key != expected:
        raise ValueError(
            f"window identity mismatch: {scene.window_key!r} != {expected!r}")


def save_scene(scene: BankScene, path: str | Path) -> str:
    validate_scene(scene)
    atomic_npz(
        path,
        predictions=np.asarray(scene.predictions, dtype=np.float32),
        future=np.asarray(scene.future, dtype=np.float32),
        history=np.asarray(scene.history, dtype=np.float32),
        frame_ids=np.asarray(scene.frame_ids, dtype=np.int64),
        agent_ids=np.asarray(scene.agent_ids, dtype=np.str_),
        source_sequence=np.asarray(scene.source_sequence, dtype=np.str_),
        scene_id=np.asarray(scene.scene_id, dtype=np.str_),
        window_key=np.asarray(scene.window_key, dtype=np.str_),
        dt_seconds=np.asarray(scene.dt_seconds, dtype=np.float32),
    )
    return sha256_file(path)


def load_scene(path: str | Path, expected_sha256: str | None = None) -> BankScene:
    source = Path(path)
    if expected_sha256 is not None and sha256_file(source) != expected_sha256:
        raise ValueError(f"bank scene SHA256 mismatch: {source}")
    with np.load(source, allow_pickle=False) as payload:
        required = {
            "predictions", "future", "history", "frame_ids", "agent_ids",
            "source_sequence", "scene_id", "window_key", "dt_seconds",
        }
        if set(payload.files) != required:
            raise ValueError(
                f"bank scene fields differ: {sorted(set(payload.files) ^ required)}")
        scene = BankScene(
            predictions=payload["predictions"], future=payload["future"],
            history=payload["history"], frame_ids=payload["frame_ids"],
            agent_ids=payload["agent_ids"],
            source_sequence=str(payload["source_sequence"].item()),
            scene_id=str(payload["scene_id"].item()),
            window_key=str(payload["window_key"].item()),
            dt_seconds=float(payload["dt_seconds"].item()),
        )
    validate_scene(scene)
    return scene


def hard_scene_statistics(
    predictions: np.ndarray, future: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Independent NumPy reference for P20 metre-space hard metrics."""
    prediction = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(future, dtype=np.float64)
    if prediction.ndim != 4 or prediction.shape[1:] != target.shape:
        raise ValueError("expected predictions [P,N,12,2] and future [N,12,2]")
    distance = np.linalg.norm(prediction - target[None], axis=-1)
    per_world_ade = distance.mean(axis=-1)  # [P,N]
    per_world_fde = distance[..., -1]
    agent_minade = per_world_ade.min(axis=0)
    agent_minfde = per_world_fde.min(axis=0)
    return {
        "per_world_ade": per_world_ade,
        "per_world_fde": per_world_fde,
        "agent_minade": agent_minade,
        "agent_minfde": agent_minfde,
        "scene_marginal_ade": float(agent_minade.mean()),
        "scene_marginal_fde": float(agent_minfde.mean()),
        "JADE": float(per_world_ade.mean(axis=1).min()),
        "JFDE": float(per_world_fde.mean(axis=1).min()),
    }


def normalized_softmin(values: np.ndarray, temperature: float, axis: int) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    value = np.asarray(values, dtype=np.float64)
    shifted = -value / float(temperature)
    maximum = shifted.max(axis=axis, keepdims=True)
    logmeanexp = (
        np.log(np.exp(shifted - maximum).mean(axis=axis))
        + np.squeeze(maximum, axis=axis)
    )
    return -float(temperature) * logmeanexp


def summarize_scenes(scenes: Iterable[BankScene]) -> dict[str, float | int]:
    agent_ade: list[np.ndarray] = []
    agent_fde: list[np.ndarray] = []
    marginal_ade: list[float] = []
    marginal_fde: list[float] = []
    joint_ade: list[float] = []
    joint_fde: list[float] = []
    for scene in scenes:
        statistics = hard_scene_statistics(scene.predictions, scene.future)
        agent_ade.append(statistics["agent_minade"])
        agent_fde.append(statistics["agent_minfde"])
        marginal_ade.append(float(statistics["scene_marginal_ade"]))
        marginal_fde.append(float(statistics["scene_marginal_fde"]))
        joint_ade.append(float(statistics["JADE"]))
        joint_fde.append(float(statistics["JFDE"]))
    if not joint_ade:
        raise ValueError("cannot summarize an empty bank")
    all_ade = np.concatenate(agent_ade)
    all_fde = np.concatenate(agent_fde)
    result: dict[str, float | int] = {
        "minADE": float(all_ade.mean()), "minFDE": float(all_fde.mean()),
        "JADE": float(np.mean(joint_ade)), "JFDE": float(np.mean(joint_fde)),
        "scene_weighted_minADE": float(np.mean(marginal_ade)),
        "scene_weighted_minFDE": float(np.mean(marginal_fde)),
        "agent_count": int(all_ade.size), "scene_count": len(joint_ade),
    }
    result["coordination_gap_ADE"] = float(
        result["JADE"] - result["scene_weighted_minADE"])
    result["coordination_gap_FDE"] = float(
        result["JFDE"] - result["scene_weighted_minFDE"])
    return result


class SeedBank:
    """Lazy hash-validating reader for one model/inference-seed manifest."""

    def __init__(self, manifest_path: str | Path, verify_hashes: bool = True):
        self.path = Path(manifest_path).resolve()
        self.payload = json.loads(self.path.read_text(encoding="utf-8"))
        if self.payload.get("schema") != SEED_MANIFEST_SCHEMA:
            raise ValueError("unsupported trajectory bank seed manifest")
        files = self.payload.get("files", [])
        if len(files) != int(self.payload.get("scene_count", -1)):
            raise ValueError("seed manifest scene count mismatch")
        identities = [item["identity"] for item in files]
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate window identity in seed bank")
        self._items = {item["identity"]: item for item in files}
        self.verify_hashes = verify_hashes

    def identities(self) -> list[str]:
        return list(self._items)

    def load(self, identity: str) -> BankScene:
        item = self._items[identity]
        path = (self.path.parent / item["path"]).resolve()
        return load_scene(
            path, item["sha256"] if self.verify_hashes else None)


def reorder_like(reference: BankScene, candidate: BankScene) -> BankScene:
    """Reorder candidate agents by exact IDs after strict source/frame checks."""
    if reference.window_key != candidate.window_key:
        raise ValueError("cannot align different canonical windows")
    if reference.source_sequence != candidate.source_sequence:
        raise ValueError("source sequence mismatch")
    if not np.array_equal(reference.frame_ids, candidate.frame_ids):
        raise ValueError("frame identity mismatch")
    reference_ids = [str(value) for value in reference.agent_ids]
    candidate_ids = [str(value) for value in candidate.agent_ids]
    if set(reference_ids) != set(candidate_ids):
        raise ValueError("agent identity set mismatch")
    positions = np.asarray([candidate_ids.index(value) for value in reference_ids])
    aligned = BankScene(
        candidate.predictions[:, positions], candidate.future[positions],
        candidate.history[positions], candidate.frame_ids,
        candidate.agent_ids[positions], candidate.source_sequence,
        candidate.scene_id, candidate.window_key, candidate.dt_seconds)
    validate_scene(aligned)
    return aligned


def assert_observation_parity(
    reference: BankScene, candidate: BankScene, tolerance_m: float = 1e-5,
) -> dict[str, float]:
    aligned = reorder_like(reference, candidate)
    history = float(np.max(np.abs(
        reference.history.astype(np.float64) - aligned.history.astype(np.float64))))
    future = float(np.max(np.abs(
        reference.future.astype(np.float64) - aligned.future.astype(np.float64))))
    if history > tolerance_m or future > tolerance_m:
        raise ValueError(
            f"coordinate identity mismatch: history={history}, future={future}")
    return {"history_max_abs_m": history, "future_max_abs_m": future}

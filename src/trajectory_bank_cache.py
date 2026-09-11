"""Frozen upstream trajectory banks and variable-N scene packing for V4.

The cache stores only immutable inputs to the trainable V4 coupler.  Each
record keeps independent ``K``-trajectory banks along a separate seed axis;
the dataset selects exactly one seed before collation, so packing can never
turn ``S_seed * K`` into a larger candidate set.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Sampler


CACHE_FORMAT_VERSION = 1
MANIFEST_FILENAME = "trajectory_bank_manifest.json"
INDEX_FILENAME = "index.json"


def _atomic_json_dump(payload: Mapping, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch_save(payload: Mapping, path: Path) -> None:
    """Atomically publish one complete cache record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def file_fingerprint(path: str) -> Dict[str, object]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Upstream checkpoint does not exist: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while True:
            chunk = handle.read(4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def deterministic_trajectory_seed(
        global_seed: int, scene_name: str, frame_start: int,
        seed_index: int) -> int:
    """Stable seed independent of Python's randomized ``hash()``."""
    key = (
        f"v{CACHE_FORMAT_VERSION}|{int(global_seed)}|{scene_name}|"
        f"{int(frame_start)}|{int(seed_index)}"
    ).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    return value % (2**31 - 1)


def default_cache_root(args) -> Path:
    if getattr(args, "trajectory_bank_cache_root", None):
        return Path(args.trajectory_bank_cache_root).expanduser().resolve()
    checkpoint = file_fingerprint(args.pretrain_path)
    label = f"{args.upstream_generator}_{checkpoint['sha256'][:12]}"
    return (Path(args.base_dir) / args.save_base_dir / str(args.test_set) /
            str(args.goal_model_type) / "trajectory_bank_cache_v1" /
            label).resolve()


def expected_manifest(args) -> Dict[str, object]:
    """Return every field that determines cached numerical contents."""
    return {
        "cache_version": CACHE_FORMAT_VERSION,
        "dataset": str(args.dataset),
        "test_set": str(args.test_set),
        "obs_length": int(args.obs_length),
        "pred_length": int(args.pred_length),
        "dt": float(args.trajectory_dt),
        "K": int(args.num_samples),
        "num_cached_seeds": int(args.num_cached_seeds_per_window),
        "trajectory_bank_seed_base": int(args.trajectory_bank_seed_base),
        "upstream_checkpoint": file_fingerprint(args.pretrain_path),
        "upstream_generator_type": str(args.upstream_generator),
        "ddpm_steps": int(args.ddpm_step),
        "ddim_steps": int(args.ddim_step),
        "trunk_stage_step": int(args.trunk_stage_step),
        "branch_stage_step": int(args.branch_stage_step),
        "use_ttst": bool(args.use_ttst),
        "down_factor": int(args.down_factor),
        "coordinate_system": "world_metres",
        "candidate_axis_semantics": "independent_K_per_seed",
        "internal_validation_strategy": str(
            args.internal_validation_strategy),
        "internal_validation_fraction": float(
            args.internal_validation_fraction),
        "internal_validation_seed": int(args.internal_validation_seed),
    }


def _manifest_mismatches(actual: Mapping, expected: Mapping) -> List[str]:
    mismatches = []
    for key, expected_value in expected.items():
        actual_value = actual.get(key, "<missing>")
        if actual_value != expected_value:
            mismatches.append(
                f"{key}: cached={actual_value!r}, requested={expected_value!r}")
    return mismatches


def prepare_cache_root(args, *, for_build: bool) -> tuple[Path, Dict[str, object]]:
    """Validate the manifest or create/recover a build directory.

    Incompatible data are never overwritten silently.  With the explicit
    force flag the whole old root is renamed to a timestamped sibling, which
    keeps it recoverable.
    """
    root = default_cache_root(args)
    expected = expected_manifest(args)
    manifest_path = root / MANIFEST_FILENAME
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            actual = json.load(handle)
        mismatches = _manifest_mismatches(actual, expected)
        if mismatches:
            if not (for_build and args.force_rebuild_trajectory_bank):
                raise RuntimeError(
                    "Trajectory-bank manifest mismatch:\n  - " +
                    "\n  - ".join(mismatches) +
                    "\nUse --force_rebuild_trajectory_bank True to move "
                    "the incompatible cache aside and rebuild it.")
            backup = root.with_name(
                root.name + time.strftime(".stale_%Y%m%d_%H%M%S"))
            shutil.move(str(root), str(backup))
            root.mkdir(parents=True, exist_ok=False)
        elif not for_build and actual.get("status") != "complete":
            raise RuntimeError(
                f"Trajectory-bank cache is not complete: {manifest_path}")
        return root, actual if not mismatches else expected
    if not for_build:
        raise FileNotFoundError(
            f"Trajectory-bank manifest not found: {manifest_path}. "
            "Run --phase trajectory_cache first.")
    root.mkdir(parents=True, exist_ok=True)
    initial = dict(expected)
    initial.update({"status": "building", "created_at": time.time()})
    _atomic_json_dump(initial, manifest_path)
    return root, initial


def finalize_manifest(root: Path, manifest: Mapping,
                      split_counts: Mapping[str, int]) -> None:
    payload = dict(manifest)
    payload.update({
        "status": "complete",
        "completed_at": time.time(),
        "split_window_counts": {
            str(key): int(value) for key, value in split_counts.items()
        },
    })
    _atomic_json_dump(payload, root / MANIFEST_FILENAME)


def validate_cache_record(record: Mapping, manifest: Mapping) -> None:
    required = {
        "window_id", "scene_name", "frame_ids", "agent_ids",
        "obs_world", "gt_future_world", "future_mask", "agent_mask",
        "raw_trajectory_banks", "metadata",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"Trajectory cache record missing {sorted(missing)}")
    obs = record["obs_world"]
    gt = record["gt_future_world"]
    banks = record["raw_trajectory_banks"]
    if obs.ndim != 3 or obs.shape[-1] != 2:
        raise ValueError("obs_world must have shape [N,T_obs,2]")
    num_agents = obs.shape[0]
    expected_bank = (
        int(manifest["num_cached_seeds"]), num_agents, int(manifest["K"]),
        int(manifest["pred_length"]), 2)
    if tuple(banks.shape) != expected_bank:
        raise ValueError(
            f"raw_trajectory_banks must be {expected_bank}, got "
            f"{tuple(banks.shape)}")
    if tuple(gt.shape) != (
            num_agents, int(manifest["pred_length"]), 2):
        raise ValueError("gt_future_world has an invalid shape")
    if not all(torch.isfinite(value).all() for value in (obs, gt, banks)):
        raise ValueError("Trajectory cache record contains NaN or Inf")
    metadata = record["metadata"]
    for key in ("cache_format_version", "K", "T_obs", "T_pred", "dt",
                "coordinate_system", "upstream_checkpoint_id",
                "upstream_generator_type"):
        if key not in metadata:
            raise ValueError(f"Trajectory cache metadata missing {key}")
    if int(metadata["K"]) != int(manifest["K"]):
        raise ValueError("Record K does not match the cache manifest")


class TrajectoryBankCacheDataset(Dataset):
    """Map-style dataset selecting one cached seed per window."""

    def __init__(self, args, split: str):
        if split not in {"train", "valid", "test"}:
            raise ValueError("split must be train, valid, or test")
        self.args = args
        self.split = split
        self.root, self.manifest = prepare_cache_root(args, for_build=False)
        index_path = self.root / split / INDEX_FILENAME
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing trajectory cache index: {index_path}")
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
        self.records = index["records"]
        self.source_ids = [item["window_id"] for item in self.records]
        self.internal_validation_source = index.get(
            "internal_validation_source")
        self.epoch = 0
        self.fixed_seed_index: Optional[int] = 0 if split != "train" else None

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_seed_index(self, seed_index: Optional[int]) -> None:
        if seed_index is not None and not (
                0 <= int(seed_index) < int(self.manifest["num_cached_seeds"])):
            raise ValueError("cached seed index is out of range")
        self.fixed_seed_index = (
            None if seed_index is None else int(seed_index))

    def selected_seed_index(self, index: int) -> int:
        if self.fixed_seed_index is not None:
            return self.fixed_seed_index
        key = (
            f"train-select|{int(self.args.seed)}|{self.epoch}|{int(index)}"
        ).encode("utf-8")
        value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        return value % int(self.manifest["num_cached_seeds"])

    def __getitem__(self, index: int) -> Dict[str, object]:
        info = self.records[index]
        path = self.root / self.split / info["file"]
        record = torch.load(path, map_location="cpu", weights_only=False)
        validate_cache_record(record, self.manifest)
        seed_index = self.selected_seed_index(index)
        # Keep the seed axis out of the packed tensor.  This assertion is the
        # executable guard against accidentally evaluating K*S_seed.
        raw = record["raw_trajectory_banks"][seed_index]
        if raw.shape[1] != int(self.manifest["K"]):
            raise AssertionError("Selected bank changed the K=20 semantics")
        return {
            **{key: value for key, value in record.items()
               if key != "raw_trajectory_banks"},
            "raw_future_world": raw,
            "cached_seed_index": seed_index,
            "cached_seed": int(record["seed_values"][seed_index]),
        }


class DynamicSceneBatchSampler(Sampler[List[int]]):
    """Greedy packing without ever splitting a synchronized window.

    The edge budget uses the complete-graph upper bound from the cache index,
    therefore the online radius/TTC graph is guaranteed not to exceed it.
    An individually oversized scene remains an indivisible singleton pack and
    is explicitly reported through ``oversized_singletons``.
    """

    def __init__(self, dataset: TrajectoryBankCacheDataset, *, shuffle: bool,
                 max_agents: int, max_edges: int, max_scenes: int,
                 seed: int):
        self.dataset = dataset
        self.shuffle = bool(shuffle)
        self.max_agents = int(max_agents)
        self.max_edges = int(max_edges)
        self.max_scenes = int(max_scenes)
        self.seed = int(seed)
        self.epoch = 0
        self.oversized_singletons = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _packs(self) -> List[List[int]]:
        order = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
        packs: List[List[int]] = []
        current: List[int] = []
        agents = edges = 0
        oversized = 0
        for index in order:
            item = self.dataset.records[index]
            n = int(item["num_agents"])
            e = int(item["edge_upper_bound"])
            individually_oversized = n > self.max_agents or e > self.max_edges
            if individually_oversized:
                if current:
                    packs.append(current)
                    current, agents, edges = [], 0, 0
                packs.append([index])
                oversized += 1
                continue
            would_overflow = current and (
                agents + n > self.max_agents or
                edges + e > self.max_edges or
                len(current) + 1 > self.max_scenes)
            if would_overflow:
                packs.append(current)
                current, agents, edges = [], 0, 0
            current.append(index)
            agents += n
            edges += e
        if current:
            packs.append(current)
        self.oversized_singletons = oversized
        return packs

    def __iter__(self) -> Iterator[List[int]]:
        yield from self._packs()

    def __len__(self) -> int:
        return len(self._packs())


def pack_trajectory_bank_records(
        records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    """Concatenate variable-N cached windows along the agent dimension."""
    if not records:
        raise ValueError("Cannot collate an empty scene pack")
    keys = ("obs_world", "gt_future_world", "raw_future_world",
            "future_mask", "agent_mask")
    for record in records:
        if any(key not in record for key in keys):
            raise ValueError("A cached scene record is incomplete")
    candidate_counts = {int(record["raw_future_world"].shape[1])
                        for record in records}
    if len(candidate_counts) != 1:
        raise ValueError("All packed scenes must use the same K")
    sizes = [int(record["obs_world"].shape[0]) for record in records]
    scene_ptr = [0]
    for size in sizes:
        scene_ptr.append(scene_ptr[-1] + size)
    scene_index = torch.repeat_interleave(
        torch.arange(len(records), dtype=torch.long),
        torch.tensor(sizes, dtype=torch.long))
    packed = {
        key: torch.cat([record[key] for record in records], dim=0)
        for key in keys
    }
    packed.update({
        "scene_index": scene_index,
        "scene_ptr": torch.tensor(scene_ptr, dtype=torch.long),
        "scene_names": [str(record["scene_name"]) for record in records],
        "window_ids": [str(record["window_id"]) for record in records],
        "frame_ids": [record["frame_ids"] for record in records],
        "agent_ids": [record["agent_ids"] for record in records],
        "cached_seed_indices": torch.tensor(
            [int(record["cached_seed_index"]) for record in records],
            dtype=torch.long),
        "cached_seeds": torch.tensor(
            [int(record["cached_seed"]) for record in records],
            dtype=torch.long),
        "num_scenes": len(records),
        "num_agents": scene_ptr[-1],
        "K": next(iter(candidate_counts)),
    })
    if packed["raw_future_world"].shape[1] != packed["K"]:
        raise AssertionError("Packing merged cached seeds into the K axis")
    return packed


def get_trajectory_bank_dataloader(args, split: str) -> DataLoader:
    dataset = TrajectoryBankCacheDataset(args, split)
    sampler = DynamicSceneBatchSampler(
        dataset,
        shuffle=(args.shuffle_train_batches if split == "train" else
                 args.shuffle_test_batches),
        max_agents=(args.max_agents_per_pack
                    if args.use_multi_scene_packing else 1_000_000),
        max_edges=(args.max_edges_per_pack
                   if args.use_multi_scene_packing else 1_000_000_000),
        max_scenes=(args.max_scenes_per_pack
                    if args.use_multi_scene_packing else 1),
        seed=int(args.seed),
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=pack_trajectory_bank_records,
        num_workers=int(args.num_workers),
        pin_memory=bool(getattr(args, "use_cuda", False)),
        persistent_workers=False,
    )


def write_split_index(root: Path, split: str,
                      records: Sequence[Mapping[str, object]],
                      internal_validation_source: Optional[str] = None) -> None:
    payload = {
        "split": split,
        "num_windows": len(records),
        "internal_validation_source": internal_validation_source,
        "records": list(records),
    }
    _atomic_json_dump(payload, root / split / INDEX_FILENAME)


def pack_statistics(loader: DataLoader) -> Dict[str, float]:
    sampler = loader.batch_sampler
    packs = sampler._packs()
    scene_counts = [len(pack) for pack in packs]
    agent_counts = [sum(int(loader.dataset.records[i]["num_agents"])
                        for i in pack) for pack in packs]
    edge_bounds = [sum(int(loader.dataset.records[i]["edge_upper_bound"])
                       for i in pack) for pack in packs]
    count = max(len(packs), 1)
    return {
        "num_packs": len(packs),
        "num_windows": len(loader.dataset),
        "avg_scenes_per_pack": sum(scene_counts) / count,
        "avg_agents_per_pack": sum(agent_counts) / count,
        "avg_edge_upper_bound_per_pack": sum(edge_bounds) / count,
        "max_agents_in_pack": max(agent_counts, default=0),
        "max_edge_upper_bound_in_pack": max(edge_bounds, default=0),
        "oversized_singletons": sampler.oversized_singletons,
    }


__all__ = [
    "CACHE_FORMAT_VERSION", "DynamicSceneBatchSampler",
    "TrajectoryBankCacheDataset", "atomic_torch_save",
    "default_cache_root", "deterministic_trajectory_seed",
    "expected_manifest", "file_fingerprint",
    "finalize_manifest", "get_trajectory_bank_dataloader",
    "pack_statistics", "pack_trajectory_bank_records",
    "prepare_cache_root", "validate_cache_record", "write_split_index",
]

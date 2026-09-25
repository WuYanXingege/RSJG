#!/usr/bin/env python3
"""Content-level audit of the canonical JDV2 train/valid/test loaders.

This tool is read-only.  It instantiates the same loaders as the saved
experiment configuration, then fingerprints the immutable identity and
trajectory fields of every selected batch-cache record.  Cache filenames are
reported, but are deliberately not used to decide whether two windows are
the same.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from torch.utils.data import Subset

from src.batch_cache_io import load_batch_cache
from src.data_loader import get_dataloader
from src.parser import check_and_add_additional_args


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY_ROOT / (
    "outputs/joint_dependency_v2/eth/joint_dependency_v2/runs/"
    "jdv2_stage_b_v2a_eth_seed2035/config.yaml")
DEFAULT_OUTPUT = REPOSITORY_ROOT / (
    "outputs/joint_dependency_v2/eth/joint_dependency_v2/"
    "stage_b_v2a/adoption_protocol/split_audit.json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(
        _plain(payload), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_source(path_value: str) -> str:
    """Make provenance stable across workspace mount points."""
    normalized = str(path_value).replace("\\", "/")
    marker = "/data/"
    if marker in normalized:
        return "data/" + normalized.split(marker, 1)[1]
    path = Path(path_value)
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _array_bytes_hash(identity: dict[str, Any], batch_data: dict[str, Any]) -> str:
    """Fingerprint exact synchronized identities and trajectory contents."""
    digest = hashlib.sha256()
    digest.update(json.dumps(
        identity, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8"))
    # These are the canonical immutable cache fields that identify the same
    # synchronized agent trajectories.  Image/map tensors are intentionally
    # excluded: they are scene-level auxiliaries rather than window identity.
    for name in ("frame_ids", "seq_list", "abs_pixel_coord", "scene_index",
                 "scene_ptr"):
        if name not in batch_data:
            continue
        value = batch_data[name]
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(name.encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _loader_members(loader: torch.utils.data.DataLoader) -> Iterable[tuple[str, Path]]:
    dataset = loader.dataset
    if isinstance(dataset, Subset):
        base = dataset.dataset
        indices = list(dataset.indices)
    else:
        base = dataset
        indices = list(range(len(dataset)))
    for index in indices:
        cache_id = base.ids[int(index)]
        yield cache_id, Path(base.path_to_folder) / cache_id


def _record(cache_id: str, cache_path: Path) -> dict[str, Any]:
    batch_data, batch_id = load_batch_cache(str(cache_path))
    frame_ids = _plain(batch_id.get("frame_ids", batch_data.get("frame_ids")))
    agent_ids = _plain(batch_id.get("agent_ids", []))
    source_path = str(batch_id.get("data_file_path", ""))
    canonical_source = _canonical_source(source_path)
    identity = {
        "scene_name": str(batch_id.get("scene_name", "")),
        "data_file_path": canonical_source,
        "starting_frames": _plain(batch_id.get("starting_frames", [])),
        "frame_ids": frame_ids,
        "agent_ids": agent_ids,
        "synchronized_window": bool(batch_id.get(
            "synchronized_window", False)),
        "batch_format_version": int(batch_id.get(
            "batch_format_version", 0)),
    }
    source = Path(source_path)
    source_sha = _sha256_file(source) if source.is_file() else None
    return {
        "cache_id": cache_id,
        "cache_path": str(cache_path.relative_to(REPOSITORY_ROOT)),
        "cache_sha256": _sha256_file(cache_path),
        "scene_name": identity["scene_name"],
        "frame_ids": frame_ids,
        "agent_ids": agent_ids,
        "source_path": canonical_source,
        "source_sha256": source_sha,
        "exact_id_sha256": _json_hash(identity),
        "content_sha256": _array_bytes_hash(identity, batch_data),
    }


def _summarize_split(loader: torch.utils.data.DataLoader) -> dict[str, Any]:
    records = [_record(cache_id, path)
               for cache_id, path in _loader_members(loader)]
    scenes = Counter(record["scene_name"] for record in records)
    sources: dict[str, dict[str, Any]] = {}
    for record in records:
        entry = sources.setdefault(record["source_path"], {
            "sha256": record["source_sha256"], "window_count": 0})
        entry["window_count"] += 1
        if entry["sha256"] != record["source_sha256"]:
            raise AssertionError("one source path produced inconsistent hashes")
    return {
        "window_count": len(records),
        "scene_window_counts": dict(sorted(scenes.items())),
        "sources": dict(sorted(sources.items())),
        "records": records,
    }


def _overlap(left: dict[str, Any], right: dict[str, Any], key: str) -> dict[str, Any]:
    left_values = [record[key] for record in left["records"]]
    right_values = [record[key] for record in right["records"]]
    left_set, right_set = set(left_values), set(right_values)
    common = left_set & right_set
    count = len(common)
    return {
        "unique_overlap_count": count,
        "left_window_count": len(left_values),
        "right_window_count": len(right_values),
        "left_overlap_percent": 100.0 * count / max(len(left_values), 1),
        "right_overlap_percent": 100.0 * count / max(len(right_values), 1),
        "ordered_sequences_identical": left_values == right_values,
        "left_internal_duplicate_count": len(left_values) - len(left_set),
        "right_internal_duplicate_count": len(right_values) - len(right_set),
    }


def _pairwise(splits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for left_name, right_name in (("train", "valid"), ("train", "test"),
                                  ("valid", "test")):
        left, right = splits[left_name], splits[right_name]
        output[f"{left_name}_vs_{right_name}"] = {
            "exact_id": _overlap(left, right, "exact_id_sha256"),
            "exact_content": _overlap(left, right, "content_sha256"),
        }
    return output


def _load_args(config_path: Path):
    with config_path.open() as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError("canonical config must be a mapping")
    args = argparse.Namespace(**payload)
    # Apply the same derived switches/path validation used by main_parser,
    # without writing or mutating the saved configuration.
    return check_and_add_additional_args(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    cli = parser.parse_args()

    config_path = cli.config.resolve()
    args = _load_args(config_path)
    if args.model_selection_split != "dataset_valid":
        raise RuntimeError(
            "this adoption audit expects the canonical dataset_valid config")
    loaders = {name: get_dataloader(args, name)
               for name in ("train", "valid", "test")}
    splits = {name: _summarize_split(loader)
              for name, loader in loaders.items()}
    pairwise = _pairwise(splits)
    valid_test_id = pairwise["valid_vs_test"]["exact_id"]
    valid_test_content = pairwise["valid_vs_test"]["exact_content"]
    if (valid_test_id["left_overlap_percent"] == 100.0 and
            valid_test_id["right_overlap_percent"] == 100.0 and
            valid_test_content["left_overlap_percent"] == 100.0 and
            valid_test_content["right_overlap_percent"] == 100.0):
        classification = "fully_identical"
    elif (valid_test_id["unique_overlap_count"] or
          valid_test_content["unique_overlap_count"]):
        classification = "partially_overlapping"
    else:
        classification = "disjoint"

    result = {
        "schema": "jdv2-evaluation-split-audit-v1",
        "config_path": str(config_path.relative_to(REPOSITORY_ROOT)),
        "model_selection_split": args.model_selection_split,
        "final_test_split": args.final_test_split,
        "split_classification": classification,
        "splits": splits,
        "pairwise_overlap": pairwise,
    }
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    with cli.output.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "output": str(cli.output),
        "classification": classification,
        "window_counts": {
            name: split["window_count"] for name, split in splits.items()},
        "pairwise_overlap": pairwise,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

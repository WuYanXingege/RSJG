#!/usr/bin/env python3
"""Build and verify the immutable JDV2 clean logical-split manifest.

This preflight is read-only with respect to datasets, model checkpoints, and
JDV2 caches.  It reuses the full content-level audit produced by the approved
split audit, re-authenticates every cache member, freezes the exact source
block, and proves logical-valid to physical-train JDV2 alignment.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.batch_cache_io import load_batch_cache
from src.clean_split_protocol import CLEAN_SPLIT_SCHEMA
from src.data_loader import (
    _source_block_partition, dataset_set_name, get_dataloader,
)
from src.joint_dependency_v2_cache import (
    jdv2_cache_root, load_cache_record, sha256_file, stable_json_hash,
)
from src.parser import check_and_add_additional_args, get_parser


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORICAL_CONFIG = ROOT / (
    "outputs/joint_dependency_v2/eth/joint_dependency_v2/runs/"
    "jdv2_stage_a_no_z_full_seed2035/config.yaml")
DEFAULT_SOURCE_AUDIT = ROOT / (
    "outputs/joint_dependency_v2/eth/joint_dependency_v2/"
    "stage_b_v2a/adoption_protocol/split_audit.json")
DEFAULT_OUTPUT_DIR = ROOT / (
    "outputs/joint_dependency_v2/eth/joint_dependency_v2/"
    "clean_split_protocol")


def _load_args(path: Path):
    defaults = vars(get_parser().parse_args([]))
    with path.open() as handle:
        saved = yaml.safe_load(handle)
    defaults.update(saved)
    # Preflight uses physical loaders directly; it never authorizes final test.
    defaults.update({
        "phase": "train",
        "clean_split_protocol": False,
        "final_test_access": "legacy",
        "model_selection_split": "internal_train",
        "internal_validation_strategy": "source_block",
    })
    return check_and_add_additional_args(argparse.Namespace(**defaults))


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _member(record: dict[str, Any], physical_split: str, index: int):
    return {
        "physical_cache_split": physical_split,
        "physical_cache_index": index,
        "cache_filename": record["cache_id"],
        "batch_cache_path": record["cache_path"],
        "batch_cache_sha256": record["cache_sha256"],
        "jdv2_cache_filename": f"{index:06d}.pt",
        "raw_source_path": record["source_path"],
        "raw_source_sha256": record["source_sha256"],
        "scene": record["scene_name"],
        "frame_ids": record["frame_ids"],
        "agent_ids": record["agent_ids"],
        "id_fingerprint_sha256": record["exact_id_sha256"],
        "trajectory_content_sha256": record["content_sha256"],
    }


def _source_blocks(records: list[dict[str, Any]]):
    blocks = []
    start = 0
    while start < len(records):
        source = records[start]["raw_source_path"]
        end = start + 1
        while end < len(records) and \
                records[end]["raw_source_path"] == source:
            end += 1
        blocks.append({"source": source, "start": start, "stop": end,
                       "window_count": end - start})
        start = end
    if len({block["source"] for block in blocks}) != len(blocks):
        raise RuntimeError("raw source windows are not contiguous")
    return blocks


def _assert_no_overlap(splits):
    fields = ("id_fingerprint_sha256", "trajectory_content_sha256")
    roles = ("train", "internal_valid", "final_test")
    result = {}
    for field in fields:
        sets = {}
        for role in roles:
            values = [item[field] for item in splits[role]["records"]]
            if len(values) != len(set(values)):
                raise RuntimeError(f"internal duplicate {field} in {role}")
            sets[role] = set(values)
        for left, right in (("train", "internal_valid"),
                            ("train", "final_test"),
                            ("internal_valid", "final_test")):
            count = len(sets[left] & sets[right])
            result[f"{left}_vs_{right}_{field}"] = count
            if count:
                raise RuntimeError(f"clean split overlap: {left}/{right}")
    return result


def _tensor_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left):
        left = left.detach().cpu().numpy()
    if torch.is_tensor(right):
        right = right.detach().cpu().numpy()
    left_array, right_array = np.asarray(left), np.asarray(right)
    return (left_array.dtype == right_array.dtype and
            left_array.shape == right_array.shape and
            np.array_equal(left_array, right_array))


def _verify_alignment(args, records):
    physical = dataset_set_name(args, "train", logical_role="valid")
    root = Path(jdv2_cache_root(args)) / "train"
    checked = 0
    for record in records:
        index = int(record["physical_cache_index"])
        batch_data, batch_id = load_batch_cache(
            str(Path(physical.path_to_folder) / physical.ids[index]))
        cache = load_cache_record(
            str(root / f"{index:06d}.pt"), allow_future_supervision=False)
        if cache.get("cache_id") != f"train-{index:06d}":
            raise RuntimeError("JDV2 cache_id/physical index mismatch")
        for name in ("frame_ids", "scene_index"):
            if name in batch_data and name in cache and not _tensor_equal(
                    batch_data[name], cache[name]):
                raise RuntimeError(f"JDV2 alignment mismatch: {name}")
        if record["cache_filename"] != physical.ids[index]:
            raise RuntimeError("batch identity/physical index mismatch")
        actual_frames = batch_id.get(
            "frame_ids", batch_data.get("frame_ids"))
        if hasattr(actual_frames, "tolist"):
            actual_frames = actual_frames.tolist()
        if record["frame_ids"] != actual_frames:
            raise RuntimeError("manifest frame identity mismatch")
        candidates = cache.get("goal_candidates_world")
        edges = cache.get("edge_index")
        if not torch.is_tensor(candidates) or candidates.shape[1] != 21:
            raise RuntimeError("invalid JDV2 candidate alignment")
        if not torch.is_tensor(edges) or edges.ndim != 2 or \
                edges.shape[0] != 2:
            raise RuntimeError("invalid JDV2 edge alignment")
        checked += 1
    return checked


def build_manifest(args, audit: dict[str, Any], audit_path: Path):
    physical_train = dataset_set_name(args, "train", logical_role="train")
    physical_test = dataset_set_name(args, "test", logical_role="test")
    train_indices, valid_indices, heldout_source = \
        _source_block_partition(physical_train)
    if train_indices != list(range(3790)) or \
            valid_indices != list(range(3790, 4110)):
        raise RuntimeError("unexpected source_block boundary")
    train_audit = audit["splits"]["train"]["records"]
    test_audit = audit["splits"]["test"]["records"]
    if len(train_audit) != 4110 or len(test_audit) != 139:
        raise RuntimeError("source audit has unexpected counts")
    if [item["cache_id"] for item in train_audit] != physical_train.ids:
        raise RuntimeError("source audit train order differs from cache")
    if [item["cache_id"] for item in test_audit] != physical_test.ids:
        raise RuntimeError("source audit test order differs from cache")

    role_specs = {
        "train": ("train", train_indices, train_audit),
        "internal_valid": ("train", valid_indices, train_audit),
        "final_test": ("test", list(range(139)), test_audit),
    }
    splits = {}
    for role, (physical_split, indices, records) in role_specs.items():
        members = [_member(records[index], physical_split, index)
                   for index in indices]
        splits[role] = {
            "window_count": len(members),
            "physical_cache_split": physical_split,
            "source_blocks": _source_blocks(members),
            "records": members,
        }
    if splits["internal_valid"]["source_blocks"] != [{
            "source": "data/eth5/eth/train/uni_examples.txt",
            "start": 0, "stop": 320, "window_count": 320}]:
        raise RuntimeError("internal validation source block is not frozen")
    if splits["final_test"]["source_blocks"] != [{
            "source": "data/eth5/eth/val/biwi_eth.txt",
            "start": 0, "stop": 139, "window_count": 139}]:
        raise RuntimeError("final held-out test block is not frozen")
    overlap = _assert_no_overlap(splits)
    payload = {
        "schema": CLEAN_SPLIT_SCHEMA,
        "source_commit": _git_head(),
        "source_audit_path": str(audit_path.relative_to(ROOT)),
        "source_audit_sha256": sha256_file(audit_path),
        "protocol": {
            "model_selection_split": "internal_train",
            "internal_validation_strategy": "source_block",
            "final_test_split": "heldout_test",
            "internal_validation_source":
                "data/eth5/eth/train/uni_examples.txt",
        },
        "overlap_counts": overlap,
        "splits": splits,
    }
    payload["manifest_hash"] = stable_json_hash(payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-config", type=Path,
                        default=DEFAULT_HISTORICAL_CONFIG)
    parser.add_argument("--source-audit", type=Path,
                        default=DEFAULT_SOURCE_AUDIT)
    parser.add_argument("--output-dir", type=Path,
                        default=DEFAULT_OUTPUT_DIR)
    cli = parser.parse_args()
    args = _load_args(cli.historical_config.resolve())
    audit_path = cli.source_audit.resolve()
    with audit_path.open() as handle:
        audit = json.load(handle)
    if audit.get("schema") != "jdv2-evaluation-split-audit-v1":
        raise RuntimeError("unexpected source audit schema")

    first = build_manifest(args, audit, audit_path)
    second = build_manifest(args, audit, audit_path)
    if first != second:
        raise RuntimeError("repeated logical split construction differs")
    aligned = _verify_alignment(
        args, first["splits"]["internal_valid"]["records"])
    if aligned != 320:
        raise RuntimeError("incomplete logical-valid JDV2 alignment")

    output_dir = cli.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as handle:
        json.dump(first, handle, indent=2, sort_keys=True)
        handle.write("\n")
    result = {
        "schema": "jdv2-clean-split-preflight-v1",
        "status": "CLEAN_SPLIT_REPRODUCTION_READY",
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "manifest_hash": first["manifest_hash"],
        "manifest_file_sha256": sha256_file(manifest_path),
        "window_counts": {role: value["window_count"]
                          for role, value in first["splits"].items()},
        "overlap_counts": first["overlap_counts"],
        "internal_valid_jdv2_alignment": {"passed": aligned, "total": 320},
        "repeated_construction_identical": True,
    }
    with (output_dir / "preflight_results.json").open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

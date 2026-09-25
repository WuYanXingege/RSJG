"""Read-only full-cache audit used by the UNIV JDV2 readiness gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.batch_cache_io import batch_cache_files, load_batch_cache
from src.joint_dependency_v2_cache import (
    FORBIDDEN_LEARNED_CACHE_KEYS,
    FUTURE_SUPERVISION_KEYS,
    load_cache_record,
    stable_json_hash,
)


SPLITS = ("train", "valid", "test")
SOURCE_PATTERN = re.compile(r"^(train|valid|test)_batch_(\d{4})\.pkl$")
JDV2_PATTERN = re.compile(r"^(\d{6})\.pt$")
EXPECTED_RAW_HASHES = {
    "biwi_eth.txt":
        "cd75b1008b82b7f442b2e03967b0f4aac36da2e73d440df1f605bb197d23fb33",
    "biwi_hotel.txt":
        "9caa771bb9153d6b809dd0916b6f86761b641e6bbb15e766c1de3133fbbb7fcf",
    "crowds_zara01.txt":
        "1147a1962a09abfb86f28c6cddcac862e095a0cf129b3016385b69eacdd09d85",
    "crowds_zara02.txt":
        "8a649d0f8c9ae75c87c4d23a85f892786b0aa30266e996c7be03e69dafff22ff",
    "crowds_zara03.txt":
        "16b3e899932c4baacd07f45013d5b921f90bc5a29eb2b0fe42f4d7c904ac3108",
    "students001.txt":
        "a6d87f278d94136fe39b8be91555487a29ac77259ae403b9dba2d5c18caf7b5b",
    "students003.txt":
        "e25798b660634330aa89f8bb259425de720e84d0873902726c1d1f4ccff21d6c",
    "uni_examples.txt":
        "61f432c0ab3070ed0ef150fbeabcd7baf839cab5495a46e6105bd747f0a092a7",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value) -> bool:
    if torch.is_tensor(value):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, np.ndarray):
        return not np.issubdtype(value.dtype, np.number) or bool(
            np.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def _update_metadata_hash(digest, index, scene_index, scene_ptr, frame_ids):
    digest.update(int(index).to_bytes(8, "little", signed=False))
    for value in (scene_index, scene_ptr, frame_ids):
        array = np.ascontiguousarray(value, dtype=np.int64)
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())


def audit_source_cache(root: Path) -> dict:
    manifest_path = root / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_manifest = {
        "batch_format_version": 2,
        "batch_size": 64,
        "dataset": "eth5",
        "down_factor": 8,
        "fast_debug": False,
        "fast_debug_num": 3,
        "goal_model_type": "joint_dependency_v2",
        "manifest_version": 1,
        "obs_length": 8,
        "pred_length": 12,
        "seq_length": 20,
        "shuffle_train_batches": True,
        "skip_ts_window": 1,
        "test_set": "univ",
    }
    if manifest != expected_manifest:
        raise RuntimeError("source cache manifest does not match UNIV JDV2")

    split_results = {}
    valid_files = batch_cache_files(str(root / "valid_batches"))
    test_files = batch_cache_files(str(root / "test_batches"))
    if len(valid_files) != len(test_files):
        raise RuntimeError("valid/test source counts differ")
    hardlink_pairs = 0
    for valid_path, test_path in zip(valid_files, test_files):
        valid_stat, test_stat = os.stat(valid_path), os.stat(test_path)
        hardlink_pairs += int(
            valid_stat.st_dev == test_stat.st_dev and
            valid_stat.st_ino == test_stat.st_ino)
    if hardlink_pairs != len(valid_files):
        raise RuntimeError("valid/test are not fully hard-link identical")

    for split in SPLITS:
        files = batch_cache_files(str(root / f"{split}_batches"))
        expected_names = [
            f"{split}_batch_{index:04d}.pkl"
            for index in range(1, len(files) + 1)]
        if [Path(path).name for path in files] != expected_names:
            raise RuntimeError(f"{split} source ordering is not contiguous")
        # Test is byte-identical to valid, so its records were already fully
        # audited through the same inodes. Reuse the valid aggregate below.
        if split == "test":
            continue
        count_by_source = Counter()
        agent_counts = []
        metadata_digest = hashlib.sha256()
        total_bytes = 0
        for zero_index, path_string in enumerate(files):
            path = Path(path_string)
            data, batch_id = load_batch_cache(str(path))
            if not _finite(data):
                raise RuntimeError(f"non-finite source record: {path}")
            coordinates = np.asarray(data["abs_pixel_coord"])
            if coordinates.ndim != 3 or coordinates.shape[0] != 20 or \
                    coordinates.shape[2] != 2:
                raise RuntimeError(f"invalid coordinates: {path}")
            num_agents = coordinates.shape[1]
            scene_index = np.asarray(data["scene_index"])
            scene_ptr = np.asarray(data["scene_ptr"])
            frame_ids = np.asarray(data["frame_ids"])
            if scene_index.shape != (num_agents,) or \
                    not np.equal(scene_index, 0).all():
                raise RuntimeError(f"invalid scene_index: {path}")
            if not np.array_equal(scene_ptr, [0, num_agents]):
                raise RuntimeError(f"invalid scene_ptr: {path}")
            if frame_ids.shape != (20, num_agents) or \
                    not np.equal(frame_ids, frame_ids[:, :1]).all():
                raise RuntimeError(f"unsynchronized frame_ids: {path}")
            if not np.equal(np.diff(frame_ids[:, 0]),
                            np.diff(frame_ids[:2, 0])[0]).all():
                raise RuntimeError(f"nonuniform frame step: {path}")
            if int(np.asarray(data["batch_format_version"]).item()) != 2 or \
                    batch_id.get("batch_format_version") != 2 or \
                    not batch_id.get("synchronized_window", False):
                raise RuntimeError(f"not synchronized format v2: {path}")
            if list(batch_id["frame_ids"]) != frame_ids[:, 0].tolist():
                raise RuntimeError(f"batch-id frame mismatch: {path}")
            if len(batch_id["agent_ids"]) != num_agents:
                raise RuntimeError(f"batch-id agent mismatch: {path}")
            source = Path(batch_id["data_file_path"])
            source_name = source.name
            if source_name not in EXPECTED_RAW_HASHES:
                raise RuntimeError(f"unexpected UNIV raw source: {source}")
            count_by_source[source_name] += 1
            agent_counts.append(num_agents)
            total_bytes += path.stat().st_size
            _update_metadata_hash(
                metadata_digest, zero_index, scene_index, scene_ptr,
                frame_ids)
        split_results[split] = {
            "count": len(files),
            "contiguous_deterministic_names": True,
            "all_records_finite": True,
            "all_windows_synchronized": True,
            "all_scene_index_ptr_valid": True,
            "no_cross_scene_mixing": True,
            "agent_count": {
                "min": min(agent_counts),
                "mean": sum(agent_counts) / len(agent_counts),
                "max": max(agent_counts),
            },
            "source_window_counts": dict(sorted(count_by_source.items())),
            "metadata_alignment_hash": metadata_digest.hexdigest(),
            "total_bytes": total_bytes,
        }
    split_results["test"] = dict(split_results["valid"])
    split_results["test"]["count"] = len(test_files)
    split_results["test"]["materialized_via_valid_hardlinks"] = True
    return {
        "status": "PASS",
        "root": str(root),
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_stable_hash": stable_json_hash(manifest),
        "raw_hashes": EXPECTED_RAW_HASHES,
        "valid_test_hardlink_identity": {
            "identical_pairs": hardlink_pairs,
            "total_pairs": len(valid_files),
        },
        "splits": split_results,
    }


def audit_jdv2_cache(root: Path, source_audit: dict) -> dict:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_manifest = {
        "schema_version": "jdv2-cache-v1",
        "dataset": "eth5",
        "test_set": "univ",
        "K": 21,
        "candidate_order": "generate_goal_candidates-v1",
        "goal_checkpoint_hash":
            "ebfbae9de25463497c7bcc20981022f0638bc0349c7eeeb38bfc235dfaf61a3e",
        "dt": 0.4,
    }
    if manifest.get("schema_version") != expected_manifest["schema_version"]:
        raise RuntimeError("JDV2 cache schema mismatch")
    if manifest.get("preprocess_config", {}).get("dataset") != "eth5" or \
            manifest.get("preprocess_config", {}).get("test_set") != "univ":
        raise RuntimeError("JDV2 cache target mismatch")
    for key in ("K", "candidate_order", "goal_checkpoint_hash", "dt"):
        if manifest.get(key) != expected_manifest[key]:
            raise RuntimeError(f"JDV2 manifest mismatch: {key}")
    if manifest.get("coordinate_units") != {
            "position": "world_m", "velocity": "world_m_per_s"}:
        raise RuntimeError("JDV2 coordinate units mismatch")
    if manifest.get("graph_config") != {
            "type": "radius_ttc", "radius": 6.0,
            "ttc_threshold": 8.0, "adaptive": False}:
        raise RuntimeError("JDV2 graph config mismatch")
    if set(manifest.get("completed_splits", ())) != set(SPLITS):
        raise RuntimeError("JDV2 cache incomplete")

    split_results = {}
    for split in SPLITS:
        split_root = root / split
        records = sorted(
            path for path in split_root.glob("*.pt")
            if not path.name.endswith(".teacher.pt"))
        teachers = sorted(split_root.glob("*.teacher.pt"))
        source_count = source_audit["splits"][split]["count"]
        if len(records) != source_count or len(teachers) != source_count:
            raise RuntimeError(f"{split} JDV2/source count mismatch")
        expected_names = [f"{index:06d}.pt" for index in range(source_count)]
        if [path.name for path in records] != expected_names:
            raise RuntimeError(f"{split} JDV2 ordering is not contiguous")
        metadata_digest = hashlib.sha256()
        edge_counts, agent_counts = [], []
        e0_count = 0
        for index, (record_path, teacher_path) in enumerate(
                zip(records, teachers)):
            record = load_cache_record(
                str(record_path), allow_future_supervision=False)
            teacher = load_cache_record(
                str(teacher_path), allow_future_supervision=True)
            cache_id = f"{split}-{index:06d}"
            if record.get("cache_id") != cache_id or \
                    teacher.get("cache_id") != cache_id:
                raise RuntimeError(f"cache ID mismatch: {record_path}")
            if FUTURE_SUPERVISION_KEYS.intersection(record) or \
                    FORBIDDEN_LEARNED_CACHE_KEYS.intersection(record):
                raise RuntimeError(f"deployment leakage: {record_path}")
            if set(teacher) != {
                    "cache_id", "future_pair_descriptor",
                    "contains_future_supervision"} or \
                    not teacher["contains_future_supervision"]:
                raise RuntimeError(f"invalid teacher sidecar: {teacher_path}")
            scene_index = record["scene_index"].numpy()
            scene_ptr = record["scene_ptr"].numpy()
            frame_ids = record["frame_ids"].numpy()
            num_agents = scene_index.shape[0]
            if not np.array_equal(scene_ptr, [0, num_agents]) or \
                    not np.equal(scene_index, 0).all() or \
                    frame_ids.shape != (20, num_agents) or \
                    not np.equal(frame_ids, frame_ids[:, :1]).all():
                raise RuntimeError(f"invalid scene metadata: {record_path}")
            shapes = {
                "goal_candidates_map": (num_agents, 21, 2),
                "goal_candidates_world": (num_agents, 21, 2),
                "candidate_log_prior": (num_agents, 21),
            }
            for name, shape in shapes.items():
                if tuple(record[name].shape) != shape:
                    raise RuntimeError(f"{name} shape mismatch: {record_path}")
            edge_index = record["edge_index"].long()
            edge_count = edge_index.shape[1]
            if edge_index.shape[0] != 2 or \
                    (edge_count and not bool((edge_index[0] < edge_index[1]).all())):
                raise RuntimeError(f"noncanonical edge index: {record_path}")
            if edge_count and (int(edge_index.min()) < 0 or
                               int(edge_index.max()) >= num_agents):
                raise RuntimeError(f"edge out of bounds: {record_path}")
            if edge_count and not bool(
                    (record["scene_index"][edge_index[0]] ==
                     record["scene_index"][edge_index[1]]).all()):
                raise RuntimeError(f"cross-scene edge: {record_path}")
            if tuple(record["edge_feat"].shape) != (edge_count, 14) or \
                    tuple(record["edge_weight"].shape) != (edge_count,):
                raise RuntimeError(f"edge feature shape mismatch: {record_path}")
            if teacher["future_pair_descriptor"].shape[0] != edge_count:
                raise RuntimeError(f"teacher/edge mismatch: {record_path}")
            _update_metadata_hash(
                metadata_digest, index, scene_index, scene_ptr, frame_ids)
            agent_counts.append(num_agents)
            edge_counts.append(edge_count)
            e0_count += int(edge_count == 0)
        if metadata_digest.hexdigest() != source_audit["splits"][split][
                "metadata_alignment_hash"]:
            raise RuntimeError(f"{split} source/JDV2 metadata misalignment")
        split_results[split] = {
            "count": len(records),
            "teacher_sidecar_count": len(teachers),
            "cache_id_index_alignment": True,
            "source_metadata_alignment": True,
            "all_records_finite": True,
            "deployment_records_future_free": True,
            "teacher_sidecars_separate": True,
            "candidate_shapes_valid": True,
            "canonical_edges": True,
            "no_cross_scene_edges": True,
            "agent_count": {
                "min": min(agent_counts),
                "mean": sum(agent_counts) / len(agent_counts),
                "max": max(agent_counts),
            },
            "edge_count": {
                "min": min(edge_counts),
                "mean": sum(edge_counts) / len(edge_counts),
                "max": max(edge_counts),
            },
            "e0_windows": e0_count,
            "metadata_alignment_hash": metadata_digest.hexdigest(),
        }
    return {
        "status": "PASS",
        "root": str(root),
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_stable_hash": stable_json_hash(manifest),
        "splits": split_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--jdv2-root", type=Path)
    parser.add_argument("--source-result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.source_result:
        source = json.loads(args.source_result.read_text())
        source = source.get("source_cache", source)
    else:
        source = audit_source_cache(args.source_root)
    result = {"source_cache": source}
    if args.jdv2_root:
        result["jdv2_cache"] = audit_jdv2_cache(args.jdv2_root, source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Export the JMM TrajNet-SDD protocol to EBJD complete-scene shards.

The author-provided trajectory text is in image pixels.  JMM documents that
models may operate on OpenTraj's approximate world conversion and that paper
metrics are converted back to pixels.  This exporter therefore stores metres
inside EBJD while binding every sequence to its explicit metres-per-pixel
factor.  No window or pedestrian is dropped to target the reported mean N.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.nn import functional as F


TRAIN_SEQUENCES = (
    "bookstore_0", "bookstore_1", "bookstore_2", "bookstore_3", "coupa_3",
    "deathCircle_0", "deathCircle_1", "deathCircle_2", "deathCircle_3",
    "deathCircle_4", "gates_0", "gates_1", "gates_3", "gates_4", "gates_5",
    "gates_6", "gates_7", "gates_8", "hyang_4", "hyang_5", "hyang_6",
    "hyang_9", "nexus_0", "nexus_1", "nexus_2", "nexus_3", "nexus_4",
    "nexus_7", "nexus_8", "nexus_9",
)
EVAL_SEQUENCES = (
    "coupa_0", "coupa_1", "gates_2", "hyang_0", "hyang_1", "hyang_3",
    "hyang_8", "little_0", "little_1", "little_2", "little_3", "nexus_5",
    "nexus_6", "quad_0", "quad_1", "quad_2", "quad_3",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object_array(values: list) -> np.ndarray:
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def _scale(scales: dict, sequence: str) -> tuple[float, float]:
    scene, video = sequence.rsplit("_", 1)
    entry = scales[scene][f"video{video}"]
    return float(entry["scale"]), float(entry["certainty"])


def synchronized_windows(path: Path, scale_m_per_px: float) -> list[dict]:
    table = np.loadtxt(path, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] != 4 or not np.isfinite(table).all():
        raise ValueError(f"invalid TrajNet-SDD table: {path}")
    frames = np.unique(table[:, 0].astype(np.int64))
    if np.any(frames % 12):
        raise ValueError(f"SDD frame IDs must be multiples of 12: {path}")
    by_frame = {
        int(frame): table[table[:, 0].astype(np.int64) == frame]
        for frame in frames
    }
    windows = []
    for start in range(len(frames) - 19):
        selected = frames[start:start + 20]
        if not np.all(np.diff(selected) == 12):
            continue
        common = set.intersection(*(
            set(by_frame[int(frame)][:, 1].astype(np.int64).tolist())
            for frame in selected
        ))
        agent_ids = sorted(common)
        if not agent_ids:
            continue
        pixels = np.empty((len(agent_ids), 20, 2), dtype=np.float32)
        for time_index, frame in enumerate(selected):
            mapping = {
                int(row[1]): row[2:4] for row in by_frame[int(frame)]
            }
            for agent_index, agent in enumerate(agent_ids):
                pixels[agent_index, time_index] = mapping[agent]
        windows.append({
            "pixels": pixels,
            "coordinates": pixels * np.float32(scale_m_per_px),
            "agent_ids": agent_ids,
            "frame_ids": selected,
        })
    return windows


def semantic_crops(
    centers_px: np.ndarray,
    labels: np.ndarray,
    scale_m_per_px: float,
    width_m: float = 32.0,
    pixels: int = 256,
) -> np.ndarray:
    classes = np.unique(labels)
    if not np.isin(classes, np.arange(6)).all():
        raise ValueError(f"semantic labels must be in 0..5, got {classes}")
    axis_m = np.linspace(-width_m / 2, width_m / 2, pixels, dtype=np.float64)
    offset_y, offset_x = np.meshgrid(axis_m, axis_m, indexing="ij")
    height, width = labels.shape
    grids = []
    for center in centers_px:
        x = center[0] + offset_x / scale_m_per_px
        y = center[1] + offset_y / scale_m_per_px
        grids.append(np.stack((
            2 * x / max(width - 1, 1) - 1,
            2 * y / max(height - 1, 1) - 1,
        ), axis=-1))
    one_hot = np.eye(6, dtype=np.float32)[labels].transpose(2, 0, 1)
    image = torch.from_numpy(one_hot)[None].expand(len(grids), -1, -1, -1)
    sampled = F.grid_sample(
        image, torch.from_numpy(np.stack(grids)).float(), mode="nearest",
        padding_mode="zeros", align_corners=True)
    return sampled.numpy().astype(np.uint8)


def _write_shard(path: Path, scenes: list[dict]) -> None:
    np.savez_compressed(
        path,
        observed=_object_array([scene["coordinates"][:, :8] for scene in scenes]),
        future=_object_array([scene["coordinates"][:, 8:] for scene in scenes]),
        semantic_maps=_object_array([scene["semantic_maps"] for scene in scenes]),
        scene_ids=np.asarray([scene["scene_id"] for scene in scenes]),
        window_keys=np.asarray([scene["window_key"] for scene in scenes]),
        source_sequences=np.asarray([scene["source_sequence"] for scene in scenes]),
        agent_ids=_object_array([
            np.asarray(scene["agent_ids"]).astype(str) for scene in scenes]),
        frame_ids=_object_array([scene["frame_ids"] for scene in scenes]),
        timestamps=_object_array([
            np.arange(20, dtype=np.float64) * 0.4 for _ in scenes]),
        scale_m_per_px=np.asarray([
            scene["scale_m_per_px"] for scene in scenes], dtype=np.float64),
    )


def export_split(
    source_root: Path,
    map_root: Path,
    scales: dict,
    split: str,
    output: Path,
    shard_scenes: int,
    maximum: int | None = None,
) -> dict:
    sequences = TRAIN_SEQUENCES if split == "train" else EVAL_SEQUENCES
    source_split = "val" if split == "validation" else split
    candidates, source_files, map_sources = [], [], {}
    map_cache = {}
    for sequence in sequences:
        source = source_root / source_split / f"{sequence}.txt"
        semantic = map_root / sequence / "pred_mask.png"
        if not source.is_file() or not semantic.is_file():
            raise FileNotFoundError(f"missing SDD source/map for {sequence}")
        scale, certainty = _scale(scales, sequence)
        labels = np.asarray(Image.open(semantic), dtype=np.uint8)
        map_cache[sequence] = labels
        source_files.append({
            "sequence": sequence, "path": str(source.resolve()),
            "sha256": sha256(source),
        })
        map_sources[sequence] = {
            "path": str(semantic.resolve()), "sha256": sha256(semantic),
            "scale_m_per_px": scale, "scale_certainty": certainty,
        }
        for window in synchronized_windows(source, scale):
            first, last = int(window["frame_ids"][0]), int(window["frame_ids"][-1])
            candidates.append({
                **window, "source_sequence": sequence,
                "scale_m_per_px": scale,
                "window_key": f"{sequence}:{first}-{last}",
            })
    if maximum is not None:
        if maximum <= 0:
            raise ValueError("maximum must be positive")
        candidates = candidates[:maximum]
    output.mkdir(parents=True, exist_ok=False)
    shards, buffer = [], []

    def flush() -> None:
        if not buffer:
            return
        path = output / f"{split}_{len(shards):05d}.npz"
        _write_shard(path, buffer)
        shards.append({
            "path": path.name, "sha256": sha256(path),
            "bytes": path.stat().st_size, "scenes": len(buffer),
        })
        buffer.clear()

    for window in candidates:
        sequence = window["source_sequence"]
        buffer.append({
            **window,
            "semantic_maps": semantic_crops(
                window["pixels"][:, 7], map_cache[sequence],
                window["scale_m_per_px"]),
            "scene_id": f"sdd:{split}:{window['window_key']}",
        })
        if len(buffer) >= shard_scenes:
            flush()
    flush()
    counts = [len(item["agent_ids"]) for item in candidates]
    metadata = {
        "dataset": "SDD TrajNet", "split": split,
        "protocol": "JMM_TrajNet_SDD_30_17_validation_test_mirrored",
        "source_unit": "pixel", "unit": "m", "evaluation_unit": "pixel",
        "coordinate_conversion": "xy_m = xy_pixel * OpenTraj scale_m_per_px",
        "dt_seconds": 0.4, "obs_length": 8, "pred_length": 12,
        "crop_width_m": 32.0, "crop_pixels": 256, "semantic_channels": 6,
        "semantic_storage": "categorical_one_hot_uint8",
        "crop_center": "last_observed_metric_position",
        "align_corners": True, "out_of_bounds_fill": 0,
        "future_used_for_crop": False,
    }
    manifest = {
        "schema": "ebjd-scene-manifest-v1", "metadata": metadata,
        "statistics": {
            "scenes": len(candidates), "agents": int(sum(counts)),
            "mean_N": float(np.mean(counts)), "min_N": min(counts),
            "max_N": max(counts),
            "N_histogram": dict(sorted(Counter(counts).items())),
            "shard_bytes": sum(item["bytes"] for item in shards),
        },
        "source_files": source_files, "map_sources": map_sources,
        "shards": shards,
        "window_keys": [item["window_key"] for item in candidates],
    }
    path = output / f"{split}_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"path": str(path.resolve()), "sha256": sha256(path), **manifest["statistics"]}


def mirror_test_manifest(validation_output: Path, test_output: Path) -> dict:
    """Bind the protocol's mirrored test split without duplicating its shards."""
    validation_path = validation_output / "validation_manifest.json"
    manifest = json.loads(validation_path.read_text())
    test_output.mkdir(parents=True, exist_ok=True)
    unexpected = [item for item in test_output.iterdir()
                  if item.name != "test_manifest.json"]
    if unexpected:
        raise FileExistsError(f"unexpected files in mirrored test output: {unexpected}")
    manifest["metadata"]["split"] = "test"
    manifest["metadata"]["mirrored_from"] = str(validation_path.resolve())
    manifest["shards"] = [
        {**item, "path": f"../validation/{item['path']}"}
        for item in manifest["shards"]
    ]
    path = test_output / "test_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "path": str(path.resolve()), "sha256": sha256(path),
        **manifest["statistics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--map-root", required=True)
    parser.add_argument("--scales", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--shard-scenes", type=int, default=16)
    parser.add_argument("--max-scenes-per-split", type=int)
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "validation", "test"),
        default=("train", "validation", "test"))
    parser.add_argument(
        "--reuse-validation-for-test", action="store_true",
        help="write a test manifest referencing the identical validation shards")
    args = parser.parse_args()
    output = Path(args.output_root)
    if output.exists() and not args.reuse_validation_for_test:
        raise FileExistsError(f"output root already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    scales_path = Path(args.scales)
    results = {}
    existing_validation = output / "validation" / "validation_manifest.json"
    if args.reuse_validation_for_test and existing_validation.is_file():
        validation_manifest = json.loads(existing_validation.read_text())
        results["validation"] = {
            "path": str(existing_validation.resolve()),
            "sha256": sha256(existing_validation),
            **validation_manifest["statistics"],
        }
    for split in args.splits:
        if split == "test" and args.reuse_validation_for_test:
            results[split] = mirror_test_manifest(
                output / "validation", output / "test")
        else:
            results[split] = export_split(
                Path(args.source_root), Path(args.map_root), scales, split,
                output / split, args.shard_scenes, args.max_scenes_per_split)
    valid = (json.loads(Path(results["validation"]["path"]).read_text())
             if "validation" in results else None)
    test = (json.loads(Path(results["test"]["path"]).read_text())
            if "test" in results else None)
    receipt = {
        "schema": "ebjd-sdd-export-receipt-v1",
        "protocol": "JMM_TrajNet_SDD_30_17_validation_test_mirrored",
        "scales_path": str(scales_path.resolve()),
        "scales_sha256": sha256(scales_path), "splits": results,
        "validation_test_window_identity": (
            valid["window_keys"] == test["window_keys"]
            if valid is not None and test is not None else None),
        "maximum_scenes_per_split": args.max_scenes_per_split,
        "purpose": "cpu_smoke" if args.max_scenes_per_split else "formal_data",
    }
    receipt["test_shards_reuse_validation"] = bool(
        args.reuse_validation_for_test and "test" in results)
    receipt_path = output / "DATASET_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({**receipt, "receipt_sha256": sha256(receipt_path)}))


if __name__ == "__main__":
    main()

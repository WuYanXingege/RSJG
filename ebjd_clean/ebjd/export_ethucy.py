"""Export synchronized ETH/UCY 8->12 worlds without invoking a legacy model."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


SEQUENCE_TO_SCENE = {
    "biwi_eth": "eth", "biwi_hotel": "hotel",
    "students001": "univ", "students003": "univ", "uni_examples": "univ",
    "crowds_zara01": "zara1", "crowds_zara02": "zara2",
    "crowds_zara03": "zara2",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def world_to_pixel(world: np.ndarray, homography: np.ndarray, scene: str) -> np.ndarray:
    homogeneous = np.concatenate((world.astype(np.float64), np.ones((len(world), 1))), axis=1)
    pixel = (np.linalg.inv(homography) @ homogeneous.T).T
    pixel = pixel[:, :2] / pixel[:, 2:3]
    if scene in {"eth", "hotel"}:
        pixel = pixel[:, [1, 0]]
    return pixel


def pixel_to_world(pixel: np.ndarray, homography: np.ndarray, scene: str) -> np.ndarray:
    pixel = pixel.astype(np.float64)
    if scene in {"eth", "hotel"}:
        pixel = pixel[:, [1, 0]]
    homogeneous = np.concatenate((pixel, np.ones((len(pixel), 1))), axis=1)
    world = (homography @ homogeneous.T).T
    return world[:, :2] / world[:, 2:3]


def semantic_crops(
    centers_world: np.ndarray,
    labels: np.ndarray,
    homography: np.ndarray,
    scene: str,
    width_m: float = 32.0,
    pixels: int = 256,
) -> np.ndarray:
    """Nearest categorical sampling on an x/y world grid, align_corners=True."""
    classes = np.unique(labels)
    if not np.isin(classes, np.arange(6)).all():
        raise ValueError(f"semantic labels must be categorical values 0..5, got {classes}")
    axis = np.linspace(-width_m / 2, width_m / 2, pixels, dtype=np.float64)
    offset_y, offset_x = np.meshgrid(axis, axis, indexing="ij")
    grids = []
    height, width = labels.shape
    for center in centers_world:
        world = np.stack((center[0] + offset_x, center[1] + offset_y), axis=-1)
        pixel = world_to_pixel(world.reshape(-1, 2), homography, scene).reshape(pixels, pixels, 2)
        normalized_x = 2 * pixel[..., 0] / max(width - 1, 1) - 1
        normalized_y = 2 * pixel[..., 1] / max(height - 1, 1) - 1
        grids.append(np.stack((normalized_x, normalized_y), axis=-1))
    one_hot = np.eye(6, dtype=np.float32)[labels].transpose(2, 0, 1)
    image = torch.from_numpy(one_hot)[None].expand(len(grids), -1, -1, -1)
    grid = torch.from_numpy(np.stack(grids)).float()
    sampled = F.grid_sample(
        image, grid, mode="nearest", padding_mode="zeros", align_corners=True)
    return sampled.numpy().astype(np.uint8)


def synchronized_windows(path: Path) -> list[dict]:
    table = np.loadtxt(path, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] != 4 or not np.isfinite(table).all():
        raise ValueError(f"invalid ETH/UCY table: {path}")
    identities = table[:, :2].astype(np.int64)
    if len(np.unique(identities, axis=0)) != len(identities):
        raise ValueError(f"duplicate frame/agent row in ETH/UCY table: {path}")
    frames = np.unique(table[:, 0].astype(np.int64))
    differences = np.diff(frames)
    positive = differences[differences > 0]
    if not len(positive):
        return []
    frame_step = Counter(positive.tolist()).most_common(1)[0][0]
    by_frame = {
        int(frame): table[table[:, 0].astype(np.int64) == frame]
        for frame in frames}
    windows = []
    for start in range(len(frames) - 19):
        selected = frames[start:start + 20]
        if not np.all(np.diff(selected) == frame_step):
            continue
        common = None
        for frame in selected:
            agents = set(by_frame[int(frame)][:, 1].astype(np.int64).tolist())
            common = agents if common is None else common & agents
        agent_ids = sorted(common or [])
        if not agent_ids:
            continue
        coordinates = np.empty((len(agent_ids), 20, 2), dtype=np.float32)
        for time_index, frame in enumerate(selected):
            rows = by_frame[int(frame)]
            mapping = {int(row[1]): row[2:4] for row in rows}
            for agent_index, agent in enumerate(agent_ids):
                coordinates[agent_index, time_index] = mapping[agent]
        windows.append({
            "coordinates": coordinates, "agent_ids": agent_ids,
            "frame_ids": selected.astype(np.int64), "frame_step": int(frame_step),
        })
    return windows


def object_array(values: list) -> np.ndarray:
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def write_shard(path: Path, scenes: list[dict]) -> None:
    np.savez_compressed(
        path,
        observed=object_array([scene["coordinates"][:, :8] for scene in scenes]),
        future=object_array([scene["coordinates"][:, 8:] for scene in scenes]),
        semantic_maps=object_array([scene["semantic_maps"] for scene in scenes]),
        scene_ids=np.asarray([scene["scene_id"] for scene in scenes]),
        window_keys=np.asarray([scene["window_key"] for scene in scenes]),
        source_sequences=np.asarray([scene["source_sequence"] for scene in scenes]),
        agent_ids=object_array([np.asarray(scene["agent_ids"]).astype(str) for scene in scenes]),
        frame_ids=object_array([scene["frame_ids"] for scene in scenes]),
        timestamps=object_array([
            scene["frame_ids"].astype(np.float64) * (0.4 / scene["frame_step"])
            for scene in scenes]),
    )


def export_split(
    source_root: Path, fold: str, split: str, output: Path,
    shard_scenes: int, maximum: int | None,
) -> dict:
    source_split = "val" if split == "validation" else split
    source_dir = source_root / fold / source_split
    candidates, source_files, map_files = [], [], {}
    map_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for path in sorted(source_dir.glob("*.txt")):
        sequence = path.stem.replace("_train", "").replace("_val", "")
        source_scene = SEQUENCE_TO_SCENE[sequence]
        homography_path = source_root / source_scene / "H.txt"
        semantic_path = source_root / source_scene / "pred_mask.png"
        homography = np.loadtxt(homography_path, dtype=np.float64)
        labels = np.asarray(Image.open(semantic_path), dtype=np.uint8)
        source_files.append({"path": str(path.resolve()), "sha256": sha256(path)})
        map_files[source_scene] = {
            "semantic_path": str(semantic_path.resolve()),
            "semantic_sha256": sha256(semantic_path),
            "homography_path": str(homography_path.resolve()),
            "homography_sha256": sha256(homography_path),
        }
        map_cache[source_scene] = (homography, labels)
        for window in synchronized_windows(path):
            first, last = int(window["frame_ids"][0]), int(window["frame_ids"][-1])
            key = f"{sequence}:{first}-{last}"
            candidates.append({
                **window, "window_key": key, "source_sequence": sequence,
                "source_scene": source_scene,
            })
    if not candidates:
        raise ValueError(f"no synchronized windows exported for {split}")

    population_counts = [len(scene["agent_ids"]) for scene in candidates]
    if maximum is None or len(candidates) <= maximum:
        selected = candidates
    else:
        if maximum <= 0:
            raise ValueError("max-scenes-per-split must be positive")
        selected = candidates[:maximum]
        maximum_index = max(range(len(candidates)), key=lambda index: population_counts[index])
        maximum_scene = candidates[maximum_index]
        if maximum_scene["window_key"] not in {scene["window_key"] for scene in selected}:
            selected[-1] = maximum_scene

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"export split directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    shards, buffer, roundtrip_max = [], [], 0.0

    def flush() -> None:
        if not buffer:
            return
        shard_path = output / f"{split}_{len(shards):05d}.npz"
        write_shard(shard_path, buffer)
        shards.append({
            "path": shard_path.name, "sha256": sha256(shard_path),
            "bytes": shard_path.stat().st_size, "scenes": len(buffer),
        })
        buffer.clear()

    for window in selected:
        coordinates = window["coordinates"]
        source_scene = window["source_scene"]
        homography, labels = map_cache[source_scene]
        pixel = world_to_pixel(coordinates.reshape(-1, 2), homography, source_scene)
        recovered = pixel_to_world(pixel, homography, source_scene)
        roundtrip_max = max(
            roundtrip_max,
            float(np.abs(recovered - coordinates.reshape(-1, 2)).max()))
        buffer.append({
            **window,
            "semantic_maps": semantic_crops(
                coordinates[:, 7], labels, homography, source_scene),
            "scene_id": f"{fold}:{split}:{window['window_key']}",
        })
        if len(buffer) == shard_scenes:
            flush()
    flush()
    counts = [len(scene["agent_ids"]) for scene in selected]
    metadata = {
        "dataset": "ETH/UCY", "fold": fold, "split": split,
        "protocol": "official_leave_one_scene_out_sources",
        "unit": "m", "dt_seconds": 0.4, "obs_length": 8, "pred_length": 12,
        "crop_width_m": 32.0, "crop_pixels": 256, "semantic_channels": 6,
        "semantic_storage": "categorical_one_hot_uint8",
        "crop_center": "last_observed_world_position_only",
        "world_grid": "x=columns,y=rows", "pixel_grid": "x=columns,y=rows",
        "align_corners": True, "out_of_bounds_fill": 0,
        "future_used_for_crop": False,
    }
    manifest = {
        "schema": "ebjd-scene-manifest-v1", "metadata": metadata,
        "statistics": {
            "scenes": len(selected), "agents": sum(counts), "min_N": min(counts),
            "max_N": max(counts), "N_histogram": dict(sorted(Counter(counts).items())),
            "shard_bytes": sum(item["bytes"] for item in shards),
            "world_pixel_roundtrip_max_abs_m": roundtrip_max,
        },
        "source_population_statistics": {
            "scenes": len(candidates), "agents": sum(population_counts),
            "min_N": min(population_counts), "max_N": max(population_counts),
            "N_histogram": dict(sorted(Counter(population_counts).items())),
            "bounded_export_contains_population_max_N": max(counts) == max(population_counts),
        },
        "source_files": source_files, "map_sources": map_files,
        "shards": shards, "window_keys": [scene["window_key"] for scene in selected],
    }
    manifest_path = output / f"{split}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"manifest": str(manifest_path.resolve()), "sha256": sha256(manifest_path), **manifest["statistics"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export complete synchronized ETH/UCY scenes")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--fold", required=True, choices=tuple(SEQUENCE_TO_SCENE.values()))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--shard-scenes", type=int, default=16)
    parser.add_argument("--max-scenes-per-split", type=int)
    args = parser.parse_args()
    source_root, output_root = Path(args.source_root), Path(args.output_root)
    results = {}
    manifests = {}
    for split in ("train", "validation", "test"):
        results[split] = export_split(
            source_root, args.fold, split, output_root / split,
            args.shard_scenes, args.max_scenes_per_split)
        manifest = json.loads(Path(results[split]["manifest"]).read_text(encoding="utf-8"))
        manifests[split] = set(manifest["window_keys"])
    overlaps = {
        "train_validation": sorted(manifests["train"] & manifests["validation"]),
        "train_test": sorted(manifests["train"] & manifests["test"]),
        "validation_test": sorted(manifests["validation"] & manifests["test"]),
    }
    receipt = {
        "schema": "ebjd-export-receipt-v1", "fold": args.fold,
        "source_root": str(source_root.resolve()), "splits": results,
        "overlap_counts": {key: len(value) for key, value in overlaps.items()},
        "mirrored_validation_test": bool(overlaps["validation_test"]),
        "overlap_examples": {key: value[:20] for key, value in overlaps.items()},
        "maximum_scenes_per_split": args.max_scenes_per_split,
        "purpose": "bounded_real_smoke" if args.max_scenes_per_split else "formal_export",
    }
    receipt_path = output_root / "DATASET_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**receipt, "receipt_sha256": sha256(receipt_path)}, sort_keys=True))


if __name__ == "__main__":
    main()

"""Audit a scene manifest against trusted local GDTS preprocessing artefacts.

The legacy pickle files are treated as trusted local inputs.  This utility does
not import or execute any legacy project module.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from ebjd.export_ethucy import pixel_to_world, sha256, world_to_pixel


def load_legacy_fragments(directory: Path) -> dict[tuple[int, int], dict]:
    fragments: dict[tuple[int, int], dict] = {}
    for path in sorted(directory.glob("*.pkl")):
        with path.open("rb") as handle:
            data, identity = pickle.load(handle)  # noqa: S301 - trusted local artefact
        coordinates = np.asarray(data["abs_pixel_coord"], dtype=np.float64)
        semantic = np.asarray(data["tensor_image"])
        for index, (agent, start) in enumerate(zip(
            identity["agent_ids"], identity["starting_frames"], strict=True
        )):
            key = (int(agent), int(start))
            if key in fragments:
                raise ValueError(f"duplicate legacy trajectory identity: {key}")
            fragments[key] = {
                "pixel": coordinates[:, index], "semantic": semantic,
                "source": str(path.resolve()),
            }
    if not fragments:
        raise ValueError(f"no legacy pickle batches found below {directory}")
    return fragments


def iter_manifest_scenes(manifest_path: Path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for descriptor in manifest["shards"]:
        path = manifest_path.parent / descriptor["path"]
        if sha256(path) != descriptor["sha256"]:
            raise ValueError(f"shard SHA256 mismatch: {path}")
        with np.load(path, allow_pickle=True) as payload:
            for index in range(len(payload["scene_ids"])):
                yield {
                    "observed": np.asarray(payload["observed"][index]),
                    "future": np.asarray(payload["future"][index]),
                    "semantic_maps": np.asarray(payload["semantic_maps"][index]),
                    "agent_ids": [int(value) for value in payload["agent_ids"][index]],
                    "frame_ids": np.asarray(payload["frame_ids"][index], dtype=np.int64),
                    "window_key": str(payload["window_keys"][index]),
                    "scene_id": str(payload["scene_ids"][index]),
                }


def load_grouped_scenes(directory: Path, homography: np.ndarray, source_scene: str) -> dict:
    grouped = {}
    for path in sorted(directory.glob("*.pkl")):
        with path.open("rb") as handle:
            data, identity = pickle.load(handle)  # noqa: S301 - trusted local artefact
        frames = tuple(int(value) for value in identity["frame_ids"])
        original_agents = tuple(int(value) for value in identity["agent_ids"])
        order = np.argsort(np.asarray(original_agents), kind="stable")
        agents = tuple(original_agents[index] for index in order)
        if not identity.get("synchronized_window") or len(frames) != 20:
            raise ValueError(f"not a synchronized 20-frame grouped batch: {path}")
        key = (frames, agents)
        if key in grouped:
            raise ValueError(f"duplicate grouped scene identity: {key}")
        pixel = np.asarray(data["abs_pixel_coord"], dtype=np.float64).transpose(1, 0, 2)
        pixel = pixel[order]
        grouped[key] = {
            "world": pixel_to_world(
                pixel.reshape(-1, 2), homography, source_scene).reshape(pixel.shape),
            "source": str(path.resolve()),
        }
    if not grouped:
        raise ValueError(f"no grouped pickle batches found below {directory}")
    return grouped


def semantic_control_points(
    scene: dict, labels: np.ndarray, homography: np.ndarray, source_scene: str,
) -> tuple[int, int, int]:
    axis = np.linspace(-16.0, 16.0, 256, dtype=np.float64)
    controls = ((0, 0), (0, 255), (255, 0), (255, 255), (64, 64),
                (64, 192), (128, 128), (192, 64), (192, 192))
    agreements = comparisons = out_of_bounds = 0
    centers = scene["observed"][:, -1]
    for agent_index, center in enumerate(centers):
        crop = scene["semantic_maps"][agent_index]
        for row, column in controls:
            world = np.asarray([[center[0] + axis[column], center[1] + axis[row]]])
            pixel = world_to_pixel(world, homography, source_scene)[0]
            px, py = int(np.rint(pixel[0])), int(np.rint(pixel[1]))
            actual_vector = crop[:, row, column]
            if 0 <= px < labels.shape[1] and 0 <= py < labels.shape[0]:
                expected = int(labels[py, px])
                actual = int(actual_vector.argmax())
                comparisons += 1
                agreements += int(actual_vector.sum() == 1 and actual == expected)
            else:
                out_of_bounds += 1
                agreements += int(actual_vector.sum() == 0)
                comparisons += 1
    return agreements, comparisons, out_of_bounds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--legacy-batches", required=True)
    parser.add_argument("--homography", required=True)
    parser.add_argument("--semantic-map", required=True)
    parser.add_argument("--source-scene", required=True, choices=("eth", "hotel", "univ", "zara1", "zara2"))
    parser.add_argument("--grouped-batches")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    legacy_dir = Path(args.legacy_batches).resolve()
    homography_path = Path(args.homography).resolve()
    semantic_path = Path(args.semantic_map).resolve()
    homography = np.loadtxt(homography_path, dtype=np.float64)
    labels = np.asarray(Image.open(semantic_path), dtype=np.uint8)
    fragments = load_legacy_fragments(legacy_dir)
    legacy_semantic = next(iter(fragments.values()))["semantic"]
    legacy_classes = legacy_semantic.argmax(0)
    source_at_legacy_grid = labels[
        :legacy_classes.shape[0] * 8:8, :legacy_classes.shape[1] * 8:8]
    legacy_map_agreement = float((legacy_classes == source_at_legacy_grid).mean())

    compared_agents = compared_points = missing = semantic_ok = semantic_total = oob = 0
    maximum_pixel_error = maximum_world_error = 0.0
    examples = []
    manifest_scenes = list(iter_manifest_scenes(manifest_path))
    for scene in manifest_scenes:
        world = np.concatenate((scene["observed"], scene["future"]), axis=1)
        start = int(scene["frame_ids"][0])
        for agent_index, agent in enumerate(scene["agent_ids"]):
            legacy = fragments.get((agent, start))
            if legacy is None:
                missing += 1
                continue
            expected_pixel = world_to_pixel(world[agent_index], homography, args.source_scene)
            recovered_world = pixel_to_world(legacy["pixel"], homography, args.source_scene)
            pixel_error = float(np.abs(expected_pixel - legacy["pixel"]).max())
            world_error = float(np.abs(recovered_world - world[agent_index]).max())
            maximum_pixel_error = max(maximum_pixel_error, pixel_error)
            maximum_world_error = max(maximum_world_error, world_error)
            compared_agents += 1
            compared_points += world.shape[1]
            if len(examples) < 8:
                examples.append({
                    "window_key": scene["window_key"], "agent_id": agent,
                    "start_frame": start, "pixel_max_abs": pixel_error,
                    "world_max_abs_m": world_error,
                })
        ok, total, outside = semantic_control_points(
            scene, labels, homography, args.source_scene)
        semantic_ok += ok
        semantic_total += total
        oob += outside

    result = {
        "schema": "ebjd-export-parity-v1",
        "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "legacy_batches": str(legacy_dir),
        "homography": {"path": str(homography_path), "sha256": sha256(homography_path)},
        "semantic_map": {"path": str(semantic_path), "sha256": sha256(semantic_path)},
        "trajectory": {
            "compared_agents": compared_agents, "compared_points": compared_points,
            "missing_legacy_fragments": missing,
            "maximum_pixel_absolute_error": maximum_pixel_error,
            "maximum_world_absolute_error_m": maximum_world_error,
            "examples": examples,
        },
        "semantic_crop_control_points": {
            "agreements": semantic_ok, "comparisons": semantic_total,
            "out_of_bounds_controls": oob,
            "agreement_fraction": semantic_ok / max(semantic_total, 1),
        },
        "legacy_semantic_map": {
            "downsample_factor": 8,
            "shape": list(legacy_semantic.shape),
            "one_hot_values": sorted(np.unique(legacy_semantic).astype(float).tolist()),
            "source_top_left_nearest_agreement_fraction": legacy_map_agreement,
        },
    }
    grouped_passed = True
    if args.grouped_batches:
        grouped_dir = Path(args.grouped_batches).resolve()
        grouped = load_grouped_scenes(grouped_dir, homography, args.source_scene)
        exported = {}
        for scene in manifest_scenes:
            order = np.argsort(np.asarray(scene["agent_ids"]), kind="stable")
            key = (
                tuple(scene["frame_ids"]),
                tuple(scene["agent_ids"][index] for index in order),
            )
            exported[key] = np.concatenate(
                (scene["observed"], scene["future"]), axis=1)[order]
        missing_from_export = sorted(set(grouped) - set(exported))
        missing_from_grouped = sorted(set(exported) - set(grouped))
        grouped_max_error = max((
            float(np.abs(exported[key] - grouped[key]["world"]).max())
            for key in set(exported) & set(grouped)), default=float("inf"))
        result["grouped_scene_identity"] = {
            "directory": str(grouped_dir), "grouped_scenes": len(grouped),
            "exported_scenes": len(exported),
            "missing_from_export_count": len(missing_from_export),
            "missing_from_grouped_count": len(missing_from_grouped),
            "missing_from_export_examples": [str(value) for value in missing_from_export[:3]],
            "missing_from_grouped_examples": [str(value) for value in missing_from_grouped[:3]],
            "maximum_world_absolute_error_m": grouped_max_error,
        }
        grouped_passed = bool(
            not missing_from_export and not missing_from_grouped
            and grouped_max_error < 1e-6)
    result["passed"] = bool(
        compared_agents > 0 and missing == 0
        and maximum_pixel_error < 5e-5 and maximum_world_error < 1e-6
        and semantic_ok == semantic_total and legacy_map_agreement == 1.0
        and grouped_passed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

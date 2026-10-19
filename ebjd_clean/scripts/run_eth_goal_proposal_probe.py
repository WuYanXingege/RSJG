#!/usr/bin/env python3
"""Frozen A/B/C ETH endpoint-proposal probe; no optimizer is constructed."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from ebjd.data import SceneManifestDataset
from ebjd.goal_proposal import (
    AgentRecord, paired_delta, pixel_to_world_m, split_native_ttst,
    summarize_rows,
)
from ebjd.vendor import TrainableMapUNet


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def window_key(item: dict) -> str:
    return f"{item['source_sequence']}:{item['frame_ids'][0]}-{item['frame_ids'][-1]}"


def load_c_bank(path: Path) -> dict[str, dict]:
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "COMPLETED" or manifest.get("worlds") != 20:
        raise RuntimeError("route C bank is not a completed P20 bank")
    answer = {}
    for descriptor in manifest["files"]:
        scene_path = path.parent / descriptor["path"]
        if sha256(scene_path) != descriptor["sha256"]:
            raise RuntimeError(f"route C scene hash mismatch: {scene_path}")
        with np.load(scene_path, allow_pickle=True) as value:
            answer[descriptor["identity"]] = {
                name: np.asarray(value[name]) for name in (
                    "predictions", "future", "history", "agent_ids")
            }
    return answer


def map_input(item: dict) -> torch.Tensor:
    observed = item["observed"].float()
    maps = item["semantic_maps"].float()
    center = observed[:, -1:, :]
    relative = observed - center
    x, y = relative[..., 0] / 16.0, relative[..., 1] / 16.0
    axis = torch.linspace(-1, 1, 256)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    sigma = 0.5 / 16.0
    squared = (
        (xx - x[..., None, None]) ** 2 +
        (yy - y[..., None, None]) ** 2)
    history = torch.exp(-0.5 * squared / (sigma * sigma))
    return torch.cat((maps, history), dim=1)


def translated_parent_state(state: dict) -> dict:
    translated = {}
    for index in range(5):
        for convolution in (0, 2):
            old = f"goal_module.encoder.enc_blocks.{index}.block.{convolution}"
            new = f"encoders.{index}.net.{convolution}"
            translated[new + ".weight"] = state[old + ".weight"]
            translated[new + ".bias"] = state[old + ".bias"]
    for index in range(4):
        old_up = f"goal_module.decoder.upconvs.{index}.up.1"
        new_up = f"up_blocks.{index}.up"
        translated[new_up + ".weight"] = state[old_up + ".weight"]
        translated[new_up + ".bias"] = state[old_up + ".bias"]
        for convolution in (0, 2):
            old = f"goal_module.decoder.dec_blocks.{index}.block.{convolution}"
            new = f"up_blocks.{index}.conv.net.{convolution}"
            translated[new + ".weight"] = state[old + ".weight"]
            translated[new + ".bias"] = state[old + ".bias"]
    translated["future_logits.weight"] = state["goal_module.head.out_layer.weight"]
    translated["future_logits.bias"] = state["goal_module.head.out_layer.bias"]
    return translated


def load_ebjd_unet(checkpoint: Path, device: torch.device) -> TrainableMapUNet:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model"]
    prefix = "encoder.map_unet."
    selected = {key[len(prefix):]: value for key, value in state.items()
                if key.startswith(prefix)}
    module = TrainableMapUNet(14)
    module.load_state_dict(selected, strict=True)
    return module.to(device).float().eval()


def load_parent_unet(checkpoint: Path, device: torch.device) -> TrainableMapUNet:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    module = TrainableMapUNet(14)
    module.load_state_dict(translated_parent_state(state), strict=True)
    return module.to(device).float().eval()


def run_heatmap_route(
    label: str, module: TrainableMapUNet, dataset: SceneManifestDataset,
    ttst, device: torch.device, seed: int, limit: int | None,
) -> tuple[list[AgentRecord], dict, list[dict]]:
    set_seed(seed)
    records, raw = [], []
    calls = 0
    started = time.monotonic()
    with torch.inference_mode():
        for scene_index in range(min(len(dataset), limit or len(dataset))):
            item = dataset[scene_index]
            tensor = map_input(item).to(device)
            _, logits = module(tensor)
            calls += 1
            probability = torch.sigmoid(logits[:, -1:]).float()
            native = ttst(probability, num_goals=20, device=device)
            clusters, argmax = split_native_ttst(
                native.detach().cpu().numpy(), 20)
            clusters = clusters[:, :, 0, :]
            argmax = argmax[:, 0, :]
            centers = item["observed"][:, -1].numpy()
            world = pixel_to_world_m(clusters, centers)
            argmax_world = pixel_to_world_m(argmax, centers)
            key = window_key(item)
            for agent, agent_id in enumerate(item["agent_ids"]):
                records.append(AgentRecord(
                    label, key, item["scene_id"], item["source_sequence"],
                    str(agent_id), len(item["agent_ids"]),
                    item["future"][agent, -1].numpy(), centers[agent],
                    world[:, agent], argmax_world[agent]))
            raw.append({
                "window_key": key, "scene_id": item["scene_id"],
                "agent_ids": item["agent_ids"],
                "candidates_m": world.astype(np.float32),
                "argmax_m": argmax_world.astype(np.float32),
            })
            if (scene_index + 1) % 50 == 0:
                print(f"{label}: {scene_index + 1}/{min(len(dataset), limit or len(dataset))}", flush=True)
    return records, {
        "forward_calls": calls, "elapsed_seconds": time.monotonic() - started,
        "precision": "FP32", "optimizer_updates": 0,
    }, raw


def run_gdts_route(
    args, dataset: SceneManifestDataset, device: torch.device, ttst,
    limit: int | None,
) -> tuple[list[AgentRecord], dict, list[dict]]:
    gdts_root = Path(args.gdts_root).resolve()
    sys.path.insert(0, str(gdts_root))
    from src.data_src.dataset_src.dataset_create import create_dataset
    from src.models.model_utils.U_net_CNN import UNet
    from src.models.model_utils.cnn_big_images_utils import (
        create_CNN_inputs_loop, create_tensor_image,
    )
    config = yaml.safe_load(Path(args.gdts_config).read_text())
    native_dataset = create_dataset(config["dataset"], load_visual_data=True)
    geometry = native_dataset.scenes[config["test_set"]]
    semantic = create_tensor_image(
        geometry.semantic_map_pred, down_factor=config["down_factor"])
    module = UNet(
        enc_chs=(14, 32, 32, 64, 64, 64),
        dec_chs=(64, 64, 64, 32, 32), out_chs=12)
    payload = torch.load(args.gdts_checkpoint, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]
    goal_state = {key[len("goal_module."):]: value for key, value in state.items()
                  if key.startswith("goal_module.")}
    module.load_state_dict(goal_state, strict=True)
    module = module.to(device).float().eval()
    set_seed(args.seed)
    records, raw, calls = [], [], 0
    started = time.monotonic()
    with torch.inference_mode():
        for scene_index in range(min(len(dataset), limit or len(dataset))):
            item = dataset[scene_index]
            world_full = torch.cat((item["observed"], item["future"]), dim=1).transpose(0, 1)
            pixels = geometry.make_pixel_coord_torch(world_full)
            downsampled = pixels / float(config["down_factor"])
            trajectory_maps = create_CNN_inputs_loop(downsampled, semantic)
            image = semantic.unsqueeze(0).repeat(len(item["agent_ids"]), 1, 1, 1)
            tensor = torch.cat((image, trajectory_maps[:, :8]), dim=1).to(device)
            probability = torch.sigmoid(module(tensor)[:, -1:]).float()
            calls += 1
            native = ttst(probability, num_goals=20, device=device)
            clusters, argmax = split_native_ttst(
                native.detach().cpu().numpy(), 20)
            clusters = torch.from_numpy(clusters[:, :, 0, :]).to(device).float()
            argmax = torch.from_numpy(argmax[:, 0, :]).to(device).float()
            scale = float(config["down_factor"])
            world = torch.stack([
                geometry.make_world_coord_torch((value * scale).unsqueeze(0))[0]
                for value in clusters]).cpu().numpy()
            argmax_world = geometry.make_world_coord_torch(
                (argmax * scale).unsqueeze(0))[0].cpu().numpy()
            centers = item["observed"][:, -1].numpy()
            key = window_key(item)
            for agent, agent_id in enumerate(item["agent_ids"]):
                records.append(AgentRecord(
                    "A", key, item["scene_id"], item["source_sequence"],
                    str(agent_id), len(item["agent_ids"]),
                    item["future"][agent, -1].numpy(), centers[agent],
                    world[:, agent], argmax_world[agent]))
            raw.append({
                "window_key": key, "scene_id": item["scene_id"],
                "agent_ids": item["agent_ids"],
                "candidates_m": world.astype(np.float32),
                "argmax_m": argmax_world.astype(np.float32),
            })
            if (scene_index + 1) % 50 == 0:
                print(f"A: {scene_index + 1}/{min(len(dataset), limit or len(dataset))}", flush=True)
    return records, {
        "forward_calls": calls, "elapsed_seconds": time.monotonic() - started,
        "precision": "FP32", "optimizer_updates": 0,
    }, raw


def route_c_records(dataset, bank, limit=None):
    records, raw = [], []
    for scene_index in range(min(len(dataset), limit or len(dataset))):
        item = dataset[scene_index]
        key = window_key(item)
        value = bank[key]
        if list(map(str, value["agent_ids"])) != list(map(str, item["agent_ids"])):
            raise RuntimeError(f"route C agent identity mismatch: {key}")
        if not np.allclose(value["future"], item["future"].numpy(), atol=1e-5):
            raise RuntimeError(f"route C truth mismatch: {key}")
        candidates = value["predictions"][:, :, -1, :]
        center = item["observed"][:, -1].numpy()
        for agent, agent_id in enumerate(item["agent_ids"]):
            records.append(AgentRecord(
                "C", key, item["scene_id"], item["source_sequence"], str(agent_id),
                len(item["agent_ids"]), item["future"][agent, -1].numpy(),
                center[agent], candidates[:, agent]))
        raw.append({
            "window_key": key, "scene_id": item["scene_id"],
            "agent_ids": item["agent_ids"],
            "candidates_m": candidates.astype(np.float32),
        })
    return records, {"forward_calls": 0, "elapsed_seconds": 0.0,
                     "precision": "stored FP32 bank", "optimizer_updates": 0}, raw


def save_raw(raw: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        window_keys=np.asarray([x["window_key"] for x in raw]),
        scene_ids=np.asarray([x["scene_id"] for x in raw]),
        agent_ids=np.asarray([np.asarray(x["agent_ids"]) for x in raw], dtype=object),
        candidates=np.asarray([x["candidates_m"] for x in raw], dtype=object),
        argmax=np.asarray([x.get("argmax_m") for x in raw], dtype=object),
    )


def stratified(rows: list[dict]) -> dict:
    answer = {}
    for field, values in (
        ("agents", ("N=1", "N>=2")),
        ("crop", ("in_crop", "out_of_crop")),
    ):
        key = "stratum_agents" if field == "agents" else "crop_stratum"
        answer[field] = {}
        weighted = 0.0
        for value in values:
            subset = [row for row in rows if row[key] == value]
            if subset:
                answer[field][value] = summarize_rows(subset)
                weighted += len(subset) * answer[field][value]["minFDE20_mean_m"]
        answer[field]["weighted_reconstruction_m"] = weighted / len(rows)
    return answer


def case_table(route_rows: dict[str, list[dict]], seed: int) -> list[dict]:
    keyed = {
        route: {(x["window_key"], x["agent_id"]): x for x in rows}
        for route, rows in route_rows.items()
    }
    keys = list(keyed["A"])
    delta = {key: keyed["B"][key]["minfde20_m"] - keyed["C"][key]["minfde20_m"]
             for key in keys}
    chosen = {
        "B_vs_C_max_improvement": min(keys, key=delta.get),
        "B_vs_C_max_regression": max(keys, key=delta.get),
        "B_C_bad_A_good": max(keys, key=lambda key: (
            min(keyed["B"][key]["minfde20_m"], keyed["C"][key]["minfde20_m"])
            - keyed["A"][key]["minfde20_m"])),
    }
    outside = [key for key in keys if keyed["A"][key]["crop_stratum"] == "out_of_crop"]
    if outside:
        chosen["out_of_crop"] = max(outside, key=lambda key: keyed["B"][key]["minfde20_m"])
    rng = np.random.default_rng(seed)
    chosen["fixed_seed_random"] = keys[int(rng.integers(len(keys)))]
    result = []
    for category, key in chosen.items():
        row = {"category": category, "window_key": key[0], "agent_id": key[1]}
        for route in route_rows:
            row[f"{route}_minFDE20_m"] = keyed[route][key]["minfde20_m"]
        result.append(row)
    return result


def plot_cases(cases, route_rows, raw_by_route, dataset, output):
    raw_lookup = {
        route: {x["window_key"]: x for x in raw}
        for route, raw in raw_by_route.items()
    }
    item_lookup = {window_key(dataset[i]): dataset[i] for i in range(len(dataset))}
    row_lookup = {
        route: {(x["window_key"], x["agent_id"]): x for x in rows}
        for route, rows in route_rows.items()
    }
    fig, axes = plt.subplots(len(cases), 3, figsize=(12, 3.5 * len(cases)), squeeze=False)
    colors = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}
    for row_index, case in enumerate(cases):
        key, agent_id = case["window_key"], case["agent_id"]
        item = item_lookup[key]
        position = list(map(str, item["agent_ids"])).index(str(agent_id))
        for column, route in enumerate(("A", "B", "C")):
            ax = axes[row_index, column]
            candidates = raw_lookup[route][key]["candidates_m"][:, position]
            history = item["observed"][position].numpy()
            future = item["future"][position].numpy()
            ax.plot(history[:, 0], history[:, 1], "k.-", label="history")
            ax.plot(future[:, 0], future[:, 1], "r.-", label="future")
            ax.scatter(candidates[:, 0], candidates[:, 1], s=18, alpha=.65,
                       color=colors[route], label=f"{route} candidates")
            ax.scatter(future[-1, 0], future[-1, 1], marker="*", s=80, color="red")
            ax.set_aspect("equal", adjustable="datalim")
            metric = row_lookup[route][(key, str(agent_id))]["minfde20_m"]
            ax.set_title(f"{case['category']} | {route} | {metric:.3f} m")
            ax.grid(alpha=.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ebjd-checkpoint", required=True)
    parser.add_argument("--gdts-root", required=True)
    parser.add_argument("--gdts-config", required=True)
    parser.add_argument("--gdts-checkpoint", required=True)
    parser.add_argument("--c-bank-manifest", required=True)
    parser.add_argument("--parent-goal-checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-output", required=True)
    parser.add_argument("--seed", type=int, default=2035)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--run-b0", action="store_true")
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-ebjd-sha256", required=True)
    parser.add_argument("--expected-gdts-sha256", required=True)
    args = parser.parse_args()
    paths = {name: Path(getattr(args, name)).resolve() for name in (
        "manifest", "ebjd_checkpoint", "gdts_config", "gdts_checkpoint",
        "c_bank_manifest")}
    for name, expected in (
        ("manifest", args.expected_manifest_sha256),
        ("ebjd_checkpoint", args.expected_ebjd_sha256),
        ("gdts_checkpoint", args.expected_gdts_sha256),
    ):
        actual = sha256(paths[name])
        if actual != expected:
            raise RuntimeError(f"{name} SHA256 mismatch: {actual}")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("this audited probe requires an available CUDA device")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device.index or 0)
    dataset = SceneManifestDataset(paths["manifest"])
    sys.path.insert(0, str(Path(args.gdts_root).resolve()))
    from src.models.model_utils.sampling_2D_map import TTST_test_time_sampling_trick

    routes, execution, raw = {}, {}, {}
    c_bank = load_c_bank(paths["c_bank_manifest"])
    c_records, execution["C"], raw["C"] = route_c_records(
        dataset, c_bank, args.max_scenes)
    routes["C"] = [record.row() for record in c_records]
    a_records, execution["A"], raw["A"] = run_gdts_route(
        args, dataset, device, TTST_test_time_sampling_trick, args.max_scenes)
    routes["A"] = [record.row() for record in a_records]
    ebjd = load_ebjd_unet(paths["ebjd_checkpoint"], device)
    b_records, execution["B"], raw["B"] = run_heatmap_route(
        "B", ebjd, dataset, TTST_test_time_sampling_trick, device,
        args.seed, args.max_scenes)
    routes["B"] = [record.row() for record in b_records]
    if args.run_b0:
        if not args.parent_goal_checkpoint:
            raise ValueError("--run-b0 requires --parent-goal-checkpoint")
        parent_path = Path(args.parent_goal_checkpoint).resolve()
        parent = load_parent_unet(parent_path, device)
        b0_records, execution["B0"], raw["B0"] = run_heatmap_route(
            "B0", parent, dataset, TTST_test_time_sampling_trick, device,
            args.seed, args.max_scenes)
        routes["B0"] = [record.row() for record in b0_records]

    expected_scenes = args.max_scenes or 253
    expected_agents = None if args.max_scenes else 364
    if len(raw["A"]) != expected_scenes or any(len(x) != len(routes["A"]) for x in routes.values()):
        raise RuntimeError("route coverage or pairing mismatch")
    if expected_agents is not None and len(routes["A"]) != expected_agents:
        raise RuntimeError("full probe did not cover 364 agent occurrences")

    summaries = {route: summarize_rows(rows) for route, rows in routes.items()}
    strata = {route: stratified(rows) for route, rows in routes.items()}
    deltas = {
        f"{left}_to_{right}": paired_delta(routes[left], routes[right])
        for left, right in (("A", "B"), ("A", "C"), ("B", "C"))
    }
    if "B0" in routes:
        deltas["B0_to_B"] = paired_delta(routes["B0"], routes["B"])
    cases = case_table(routes, args.seed)
    output = Path(args.output).resolve()
    raw_output = Path(args.raw_output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw_output.mkdir(parents=True, exist_ok=True)
    for route, value in raw.items():
        save_raw(value, raw_output / f"route_{route}.npz")
    fieldnames = sorted({key for rows in routes.values() for row in rows for key in row})
    with (output / "agent_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for route in routes:
            writer.writerows(routes[route])
    plot_cases(cases, routes, raw, dataset, output / "case_panels.png")
    result = {
        "schema": "ebjd-eth-goal-proposal-three-arm-v1",
        "status": "COMPLETE" if args.max_scenes is None else "PILOT",
        "optimizer_updates": 0, "seed": args.seed, "unit": "m",
        "scenes": len(raw["A"]), "agent_occurrences": len(routes["A"]),
        "protocol": {
            "observation_frames": 8, "prediction_frames": 12,
            "dt_seconds": 0.4, "candidates": 20,
            "route_A": "native GDTS goal module/input/TTST; 20 cluster centers",
            "route_B": "EBJD epoch-5 final-channel sigmoid; same native TTST",
            "route_C": "certified EBJD epoch-5 P20 gamma=1 rollout endpoints",
            "argmax": "auxiliary only; excluded from main K=20",
            "independent_test": False, "validation_test_relation": "mirror",
        },
        "identity": {
            "manifest": str(paths["manifest"]),
            "manifest_sha256": sha256(paths["manifest"]),
            "ebjd_checkpoint": str(paths["ebjd_checkpoint"]),
            "ebjd_checkpoint_sha256": sha256(paths["ebjd_checkpoint"]),
            "gdts_checkpoint": str(paths["gdts_checkpoint"]),
            "gdts_checkpoint_sha256": sha256(paths["gdts_checkpoint"]),
            "c_bank_manifest": str(paths["c_bank_manifest"]),
            "c_bank_manifest_sha256": sha256(paths["c_bank_manifest"]),
            "probe_commit": git_head(Path(__file__).resolve().parents[2]),
            "gdts_code_commit": git_head(Path(args.gdts_root).resolve()),
        },
        "execution": execution,
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device.index or 0)),
        "summaries": summaries, "strata": strata,
        "paired_deltas": deltas, "cases": cases,
        "raw_artifacts_local_only": {
            route: str(raw_output / f"route_{route}.npz") for route in raw},
    }
    atomic_json(result, output / "summary.json")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

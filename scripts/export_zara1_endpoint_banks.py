#!/usr/bin/env python3
"""Export frozen ZARA1 endpoint banks without changing either sampler.

The GDTS exporter reads the actual goal conditions returned by the same
forward call that produces the final trajectories.  It does not resample,
reorder, truncate, or match candidates to ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def manifest_identity(manifest_path: Path) -> tuple[list[dict], dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for descriptor in manifest["shards"]:
        shard = manifest_path.parent / descriptor["path"]
        if sha256(shard) != descriptor["sha256"]:
            raise ValueError(f"manifest shard SHA256 mismatch: {shard}")
        with np.load(shard, allow_pickle=True) as payload:
            for index in range(len(payload["scene_ids"])):
                records.append({
                    "scene_id": str(payload["scene_ids"][index]),
                    "agent_ids": [str(value) for value in payload["agent_ids"][index]],
                    "frame_ids": [int(value) for value in payload["frame_ids"][index]],
                    "future": np.asarray(payload["future"][index], dtype=np.float32),
                })
    if len(records) != int(manifest["statistics"]["scenes"]):
        raise ValueError("manifest record count mismatch")
    return records, manifest


def pack_bank(records: list[dict], endpoint_records: dict[str, np.ndarray]) -> dict:
    worlds = {array.shape[0] for array in endpoint_records.values()}
    if len(worlds) != 1:
        raise ValueError("endpoint records disagree on K")
    k = worlds.pop()
    maximum = max(len(record["agent_ids"]) for record in records)
    endpoints = np.full((len(records), k, maximum, 2), np.nan, dtype=np.float32)
    truth = np.full((len(records), maximum, 2), np.nan, dtype=np.float32)
    valid = np.zeros((len(records), maximum), dtype=np.bool_)
    agent_ids = np.full((len(records), maximum), "", dtype="<U32")
    scene_ids = np.asarray([record["scene_id"] for record in records], dtype="<U128")
    for scene_index, record in enumerate(records):
        count = len(record["agent_ids"])
        value = endpoint_records.get(record["scene_id"])
        if value is None or value.shape != (k, count, 2):
            raise ValueError(f"missing or malformed endpoints for {record['scene_id']}")
        endpoints[scene_index, :, :count] = value
        truth[scene_index, :count] = record["future"][:, -1]
        valid[scene_index, :count] = True
        agent_ids[scene_index, :count] = record["agent_ids"]
    if not np.isfinite(endpoints[valid[:, None].repeat(k, axis=1)]).all():
        raise ValueError("non-finite endpoint bank")
    return {
        "endpoints": endpoints, "truth": truth, "valid": valid,
        "scene_ids": scene_ids, "agent_ids": agent_ids,
    }


def save_bank(path: Path, arrays: dict, metadata: dict) -> dict:
    atomic_npz(
        path, **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
    receipt = dict(metadata)
    receipt.update({
        "path": str(path.resolve()), "sha256": sha256(path),
        "bytes": path.stat().st_size,
    })
    atomic_json(path.with_suffix(".json"), receipt)
    return receipt


def legacy_cli(recipe: dict, scene_cache_root: Path, output_root: Path) -> list[str]:
    return [
        "--dataset", "eth5", "--test_set", str(recipe["fold"]), "--phase", "test",
        "--device", "cuda:0", "--batch_size", "64", "--num_workers", "0",
        "--down_factor", "8", "--skip_ts_window", "1",
        "--goal_model_type", "social", "--training_stage", "finetune",
        "--data_augmentation", "False", "--shuffle_train_batches", "False",
        "--shuffle_test_batches", "False", "--num_samples", "20",
        "--ddpm_step", "100", "--ddim_step", "20", "--trunk_stage_step", "30",
        "--reproducibility", "True", "--seed", str(recipe["seed"]),
        "--batch_cache_root", str(scene_cache_root.resolve()),
        "--use_wandb", "False",
    ]


def configure_legacy_evaluator(args):
    recipe = yaml.safe_load(Path(args.recipe).read_text(encoding="utf-8"))
    legacy = Path(recipe["legacy_gdts"]["root"]).resolve()
    runtime = Path(args.runtime_root).resolve()
    sys.path.insert(0, str(legacy))
    os.chdir(runtime)
    from src.models.model import GDTS
    from src.parser import check_and_add_additional_args, get_parser
    from src.trainer import trainer

    parsed = get_parser().parse_args(legacy_cli(
        recipe, Path(args.scene_cache_root), Path(args.output_dir)))
    parsed = check_and_add_additional_args(parsed)
    parsed.model_dir = str(Path(args.output_dir).resolve() / "runtime")
    parsed.save_dir = parsed.model_dir
    evaluator = trainer(parsed)
    parsed.goal_model_type = "independent"
    parsed.training_stage = "baseline"
    parsed.use_scene_latent = False
    parsed.use_dynamic_relation = False
    parsed.use_joint_energy = False
    parsed.use_dependency_corrector = False
    parsed.jdv2_active = False
    parsed.joint_goal_enabled = False
    parsed.best_metric = "auto"
    evaluator.args = parsed
    evaluator.net = GDTS(parsed, evaluator.device).to(evaluator.device)
    evaluator._pending_training_state = None
    evaluator._load_state_file(str(Path(args.checkpoint).resolve()), weights_only=True)
    evaluator.net.eval()
    return evaluator


def gdts_scene_identity(batch_id: dict) -> tuple[list[int], list[str]]:
    frame_ids = [int(value.reshape(-1)[0]) for value in batch_id["frame_ids"]]
    agent_ids = [str(int(value.reshape(-1)[0])) for value in batch_id["agent_ids"]]
    return frame_ids, agent_ids


@torch.no_grad()
def export_gdts(args) -> None:
    checkpoint = Path(args.checkpoint).resolve()
    manifest_path = Path(args.manifest).resolve()
    if sha256(checkpoint) != args.checkpoint_sha256:
        raise ValueError("GDTS checkpoint SHA256 mismatch")
    if sha256(manifest_path) != args.manifest_sha256:
        raise ValueError("validation manifest SHA256 mismatch")
    records, manifest = manifest_identity(manifest_path)
    lookup = {
        (tuple(record["frame_ids"]), tuple(sorted(record["agent_ids"], key=int))): record
        for record in records
    }
    evaluator = configure_legacy_evaluator(args)
    from src.utils import isolated_random_seed

    loader = evaluator.data_loaders["valid"]
    first_data, first_id = next(iter(loader))
    with isolated_random_seed(args.seed, use_cuda=True):
        inputs, _ = evaluator.net.prepare_inputs(first_data, first_id)
        with evaluator._autocast_context():
            plain_output, _ = evaluator.net.forward(inputs, if_test=True)
    with isolated_random_seed(args.seed, use_cuda=True):
        inputs, _ = evaluator.net.prepare_inputs(first_data, first_id)
        with evaluator._autocast_context():
            captured_output, captured_aux = evaluator.net.forward(inputs, if_test=True)
        _ = captured_aux["goal_point"][:20].detach().cpu()
    parity_max_abs = float((plain_output - captured_output).abs().max().cpu())
    if not torch.equal(plain_output, captured_output):
        raise RuntimeError(f"goal capture changed GDTS output: max_abs={parity_max_abs}")

    final_records: dict[str, np.ndarray] = {}
    proposal_records: dict[str, np.ndarray] = {}
    if evaluator.device.type == "cuda":
        torch.cuda.synchronize(evaluator.device)
    started = time.perf_counter()
    with isolated_random_seed(args.seed, use_cuda=True):
        for batch_data, batch_id in loader:
            inputs, _ = evaluator.net.prepare_inputs(batch_data, batch_id)
            with evaluator._autocast_context():
                trajectories, auxiliary = evaluator.net.forward(inputs, if_test=True)
            frame_ids, agent_ids = gdts_scene_identity(batch_id)
            record = lookup.get((
                tuple(frame_ids), tuple(sorted(agent_ids, key=int))))
            if record is None:
                raise RuntimeError(
                    f"GDTS window not present in manifest: frames={frame_ids}, agents={agent_ids}")
            scene = inputs["scene"]
            final_pixel = trajectories[:, -1].detach() * float(evaluator.args.down_factor)
            proposal_pixel = auxiliary["goal_point"][:20].detach() * float(
                evaluator.args.down_factor)
            final_world = scene.make_world_coord_torch(final_pixel)
            proposal_world = scene.make_world_coord_torch(proposal_pixel)
            # The trusted grouped cache retains its source-column order while
            # the paper manifest uses stable numeric agent-ID order.  Reorder
            # only the agent axis by explicit identity; world/candidate axes
            # and the A--B correspondence remain untouched.
            cache_index = {agent_id: index for index, agent_id in enumerate(agent_ids)}
            agent_order = [cache_index[agent_id] for agent_id in record["agent_ids"]]
            final_world = final_world[:, agent_order]
            proposal_world = proposal_world[:, agent_order]
            final_records[record["scene_id"]] = final_world.float().cpu().numpy()
            proposal_records[record["scene_id"]] = proposal_world.float().cpu().numpy()
    if evaluator.device.type == "cuda":
        torch.cuda.synchronize(evaluator.device)
    elapsed = time.perf_counter() - started
    if len(final_records) != len(records):
        raise RuntimeError("GDTS export did not cover the complete manifest")
    final_arrays = pack_bank(records, final_records)
    proposal_arrays = pack_bank(records, proposal_records)
    capture_hash = hashlib.sha256(
        final_arrays["endpoints"].tobytes() + proposal_arrays["endpoints"].tobytes()
    ).hexdigest()
    common = {
        "schema": "ebjd-frozen-endpoint-bank-v1",
        "fold": str(yaml.safe_load(Path(args.recipe).read_text(encoding="utf-8"))["fold"]),
        "split": "validation", "seed": args.seed, "worlds": 20,
        "ddim_steps": 20, "coordinate_unit": "m",
        "checkpoint": str(checkpoint), "checkpoint_sha256": args.checkpoint_sha256,
        "manifest": str(manifest_path), "manifest_sha256": args.manifest_sha256,
        "scene_count": len(records),
        "agent_occurrence_count": int(manifest["statistics"]["agents"]),
        "shared_forward_capture_id": capture_hash,
        "capture_no_effect_exact": True, "capture_parity_max_abs": parity_max_abs,
        "gpu_elapsed_seconds": elapsed,
    }
    output = Path(args.output_dir).resolve()
    final_receipt = save_bank(
        output / f"gdts_final_A_seed{args.seed}.npz", final_arrays,
        {**common, "path_label": "A", "endpoint_source": "final_native_trajectory"})
    proposal_receipt = save_bank(
        output / f"gdts_proposal_B_seed{args.seed}.npz", proposal_arrays,
        {**common, "path_label": "B",
         "endpoint_source": "actual_goal_condition_for_corresponding_final_output"})
    atomic_json(output / f"gdts_export_receipt_seed{args.seed}.json", {
        "status": "COMPLETE", "A": final_receipt, "B": proposal_receipt,
        "gpu_elapsed_seconds": elapsed,
    })
    print(json.dumps({"A": final_receipt, "B": proposal_receipt}, sort_keys=True))


@torch.no_grad()
def export_ebjd(args) -> None:
    from ebjd.config import build_model
    from ebjd.data import SceneManifestDataset, collate_scenes
    from ebjd.sampling import predict

    checkpoint = Path(args.checkpoint).resolve()
    manifest_path = Path(args.manifest).resolve()
    if sha256(checkpoint) != args.checkpoint_sha256:
        raise ValueError("EBJD checkpoint SHA256 mismatch")
    if sha256(manifest_path) != args.manifest_sha256:
        raise ValueError("validation manifest SHA256 mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload["resolved_config"]
    configured_manifest = Path(config["paths"]["validation_manifest"])
    if not configured_manifest.is_absolute():
        configured_manifest = (Path.cwd() / configured_manifest).resolve()
    if sha256(configured_manifest) != args.manifest_sha256:
        raise ValueError("checkpoint resolved config is bound to another manifest")
    records, manifest = manifest_identity(manifest_path)
    loader = DataLoader(
        SceneManifestDataset(manifest_path), batch_size=1, shuffle=False,
        collate_fn=collate_scenes, num_workers=0)
    device = torch.device(args.device)
    model = build_model(config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    endpoint_records: dict[str, np.ndarray] = {}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for batch in loader:
        scene_id = batch.scene_ids[0]
        agent_ids = list(batch.agent_ids[0])
        batch = batch.to(device)
        trajectories, _ = predict(
            model, batch.observed, batch.semantic_maps, batch.valid,
            worlds=20, steps=20, generator=generator)
        count = int(batch.valid[0].sum().item())
        if agent_ids != next(
            record["agent_ids"] for record in records if record["scene_id"] == scene_id
        ):
            raise RuntimeError(f"EBJD agent identity mismatch: {scene_id}")
        endpoint_records[scene_id] = trajectories[0, :, :count, -1].float().cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    arrays = pack_bank(records, endpoint_records)
    metadata = {
        "schema": "ebjd-frozen-endpoint-bank-v1",
        "fold": str(config["experiment"]["fold"]),
        "split": "validation", "seed": args.seed, "worlds": 20,
        "ddim_steps": 20, "coordinate_unit": "m", "path_label": "C",
        "endpoint_source": "ebjd_parent_final_raw_trajectory",
        "checkpoint": str(checkpoint), "checkpoint_sha256": args.checkpoint_sha256,
        "manifest": str(manifest_path), "manifest_sha256": args.manifest_sha256,
        "scene_count": len(records),
        "agent_occurrence_count": int(manifest["statistics"]["agents"]),
        "gpu_elapsed_seconds": elapsed, "strict_checkpoint_load": True,
        "config_source": "checkpoint_embedded_resolved_config",
    }
    receipt = save_bank(
        Path(args.output_dir).resolve() / "ebjd_parent_C_seed2035.npz",
        arrays, metadata)
    atomic_json(Path(args.output_dir).resolve() / "ebjd_export_receipt.json", {
        "status": "COMPLETE", "C": receipt, "gpu_elapsed_seconds": elapsed,
    })
    print(json.dumps(receipt, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoint", required=True)
    common.add_argument("--checkpoint-sha256", required=True)
    common.add_argument("--manifest", required=True)
    common.add_argument("--manifest-sha256", required=True)
    common.add_argument("--output-dir", required=True)
    common.add_argument("--seed", type=int, default=2035)
    gdts = subparsers.add_parser("gdts", parents=[common])
    gdts.add_argument("--recipe", required=True)
    gdts.add_argument("--runtime-root", required=True)
    gdts.add_argument("--scene-cache-root", required=True)
    ebjd = subparsers.add_parser("ebjd", parents=[common])
    ebjd.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.mode == "gdts":
        export_gdts(args)
    else:
        export_ebjd(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CPU protocol, migration, real-loader and V2 wiring preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from ebjd.config import build_model, load_config, trainer_options
from ebjd.data import SceneManifestDataset, collate_scenes
from ebjd.train import validate_v2_fork_payload
from ebjd.trainer import EBJDTrainer


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_dict_sha(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    protocol = Path(args.protocol_dir).resolve()
    parent = Path(args.parent).resolve()
    if sha256(parent) != args.expected_sha256:
        raise ValueError("V2 CPU preflight parent SHA256 mismatch")
    payload = torch.load(parent, map_location="cpu", weights_only=False)
    variants = {
        "full": "v2_full", "no_token": "v2_no_token",
        "no_anchor": "v2_no_anchor",
    }
    results = {}
    assembler_states = {}
    full_model = None
    full_config = None
    for variant, ablation in variants.items():
        config_path = protocol / f"V2_{variant.upper()}_CONFIG.json"
        config = load_config(config_path)
        identity = {
            "source_commit": config["experiment"]["source_commit"],
            "fold": "eth", "seed": 3101, "ablation": ablation,
            "run_id": f"preflight_{variant}", "smoke": False,
        }
        contract = validate_v2_fork_payload(
            payload, parent, args.expected_sha256, config, identity, ablation)
        torch.manual_seed(int(config["seed"]))
        model = build_model(config, ablation)
        model.encoder.activation_checkpointing = False
        model.denoiser.activation_checkpointing = False
        generator = torch.Generator().manual_seed(int(config["seed"]) + 17)
        trainer = EBJDTrainer(
            model, trainer_options(config, ablation), config, identity)
        restoration = trainer.restore_named_extension_payload(payload, generator)
        if model.joint_goal_assembler is not None:
            assembler_states[variant] = {
                name: value.detach().clone()
                for name, value in model.joint_goal_assembler.state_dict().items()}
        results[variant] = {
            "config": str(config_path), "config_sha256": sha256(config_path),
            "parameters": int(sum(value.numel() for value in model.parameters())),
            "trainable_parameters": int(sum(
                value.numel() for value in model.parameters() if value.requires_grad)),
            "memory_tokens": int(config["denoiser"]["memory_tokens_per_agent"]),
            "migration_contract": contract,
            "restoration": restoration,
            "model_state_sha256": tensor_dict_sha(model.state_dict()),
        }
        if variant == "full":
            full_model, full_config = model, config
        else:
            del trainer, model

    assembler_exact = all(torch.equal(
        assembler_states["full"][name], assembler_states["no_token"][name])
        for name in assembler_states["full"])
    if not assembler_exact:
        raise RuntimeError("FULL and NO_TOKEN assembler initialization differs")

    dataset = SceneManifestDataset(
        full_config["paths"]["train_manifest"],
        native_goal_bank=full_config["native_goal_conditioning"]["train_cache_manifest"])
    item = dataset[0]
    batch = collate_scenes([item])
    full_model.eval()
    with torch.no_grad():
        context = full_model.encode_context(
            batch.observed, batch.semantic_maps, batch.valid, batch.native_goals)
        training_context, labels = full_model.assign_training_anchors(
            context, batch.future)
        latent = full_model.representation.encode_target(
            batch.future, training_context.baseline,
            training_context.world_anchors[:, 0])[:, None]
        output = full_model.denoiser(latent, torch.tensor([0.5]), training_context)[0]
    real_loader = {
        "scene_id": batch.scene_ids[0],
        "agents": int(batch.valid.sum()),
        "native_goal_shape": list(batch.native_goals.shape),
        "memory_shape": list(context.memory.shape),
        "training_label_shape": list(labels.shape),
        "denoiser_output_shape": list(output.shape),
        "finite": bool(torch.isfinite(output).all()),
    }
    checks = {
        "three_named_migrations_exact": all(
            row["restoration"]["shared_model_state_exact"]
            and row["restoration"]["shared_optimizer_state_by_name_exact"]
            and row["restoration"]["new_optimizer_state_empty"]
            for row in results.values()),
        "isolated_assembler_initialization_exact": assembler_exact,
        "real_production_loader_has_P20": real_loader["native_goal_shape"][-2:] == [20, 2],
        "full_memory_is_28": real_loader["memory_shape"][-2:] == [28, 128],
        "real_forward_finite": real_loader["finite"],
        "parent_epoch_update_exact": (
            int(payload["epoch"]), int(payload["update_index"])) == (20, 5140),
    }
    receipt = {
        "schema": "ebjd-v2-integrated-cpu-preflight-v1",
        "status": "CPU_PASS_GPU_PENDING" if all(checks.values()) else "ENGINEERING_FAILED",
        "persisted_update": False,
        "parent": {"path": str(parent), "sha256": args.expected_sha256,
                   "epoch": int(payload["epoch"]),
                   "update_index": int(payload["update_index"])},
        "checks": checks, "variants": results,
        "real_loader_forward": real_loader,
    }
    atomic_json(Path(args.output), receipt)
    print(json.dumps(receipt, sort_keys=True))
    if receipt["status"] == "ENGINEERING_FAILED":
        raise RuntimeError("V2 CPU preflight failed")


if __name__ == "__main__":
    main()

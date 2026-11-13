#!/usr/bin/env python3
"""Prepare a fresh unified run from an existing fold-bound resolved config."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import yaml

from ebjd.config import config_hash
from network import profiles
from release_guard import ROOT, check_contract, file_digest, manifest_digest, verify_release
from train import atomic_json, make_early_stop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--profile", choices=("legacy6", "gated6", "gated8"), default="gated8")
    parser.add_argument("--config-dir", required=True, type=Path)
    parser.add_argument("--run-output", required=True, type=Path)
    parser.add_argument("--denoiser-uncheckpointed-max-agent-slots", type=int,
                        choices=(0, 16), default=0,
                        help="Opt-in SDD gated8 training runtime; keeps map checkpoints")
    args = parser.parse_args()
    source = verify_release()
    config_dir, output = args.config_dir.resolve(), args.run_output.resolve()
    if (config_dir == output or config_dir.is_relative_to(output) or
            output.is_relative_to(config_dir) or config_dir.is_relative_to(ROOT) or
            output.is_relative_to(ROOT)):
        raise ValueError("config directory and run output must be separate, outside source")
    for path in (config_dir, output):
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"refusing existing nonempty destination: {path}")
    base = (json.loads(args.base_config.read_text()) if args.base_config.suffix.lower() == ".json"
            else yaml.safe_load(args.base_config.read_text()))
    config = deepcopy(base)
    early = config["training"].get("early_stopping")
    if not early or early.get("max_epochs") != 30 or early.get("monitor") != "H_JADE":
        raise ValueError("use a complete, accepted 30-epoch unified fold contract")
    make_early_stop(early)
    config["denoiser"].update(profiles()["profiles"][args.profile])
    config["denoiser"]["ctf_feedback_basis"] = "legacy"
    config["representation"]["type"] = "invertible_endpoint_brownian_bridge"
    config["training"].update(
        epochs=30, new_module_lr=1e-4, unet_lr=1e-5, lr_warmup_epochs=5,
        lr_final_fraction=0.1, optimizer="adamw",
        rollout_training_enabled=False, paired_marginal_actual_step_projection_enabled=False,
        coupling_singleton_backward="legacy")
    if args.denoiser_uncheckpointed_max_agent_slots and (
            args.profile != "gated8" or config["experiment"]["fold"] != "sdd_trajnet_jmm"):
        raise ValueError("runtime measured only for SDD TrajNet-JMM gated8; keep 0 elsewhere")
    config["training"]["denoiser_uncheckpointed_max_agent_slots"] = (
        args.denoiser_uncheckpointed_max_agent_slots)
    config["experiment"].update(source_commit=source["upstream_commit"],
                                implementation_base=source["upstream_commit"][:7])
    config["paths"]["output_root"] = str(output)
    check_contract(config)
    paths = config["paths"]
    for split in ("train", "validation", "test"):
        if not Path(paths[f"{split}_manifest"]).is_file():
            raise FileNotFoundError(paths[f"{split}_manifest"])
    for key in ("accepted_fold_matched_gdts_unet", "matched_gdts_validation_result",
                "matched_gdts_validation_checkpoint"):
        if file_digest(Path(paths[key])) != paths[key + "_sha256"]:
            raise ValueError(f"bound asset hash differs: {key}")
    baseline = config["evaluation"]["matched_gdts_validation"]
    if (baseline["status"] != "bound_to_manifest" or
            baseline["manifest_sha256"] != file_digest(Path(paths["validation_manifest"]))):
        raise ValueError("baseline must be bound to the actual validation manifest")
    resolved = config_dir / "RESOLVED.json"
    atomic_json(resolved, config)
    receipt = {
        "profile": args.profile, "upstream_commit": source["upstream_commit"],
        "release_manifest_sha256": manifest_digest(),
        "base_config": str(args.base_config.resolve()),
        "base_config_sha256": file_digest(args.base_config),
        "config_hash": config_hash(config), "config_file_sha256": file_digest(resolved),
        "initialization": "fresh body; same-fold verified map U-Net; train-only fitted scales",
        "training_started": False,
        "metadata_normalization": {k: {"old": base["training"].get(k), "new": v}
            for k, v in config["training"].items() if base["training"].get(k) != v},
    }
    atomic_json(config_dir / "PREPARATION.json", receipt)
    print(json.dumps({"resolved": str(resolved), "profile": args.profile,
                      "training_started": False}))


if __name__ == "__main__":
    main()

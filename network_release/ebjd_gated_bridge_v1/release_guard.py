"""Fail-closed release identity and unified training-contract checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def manifest_digest() -> str:
    return file_digest(ROOT / "SOURCE_MANIFEST.json")


def verify_release() -> dict:
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
    for relative, item in manifest["files"].items():
        path = ROOT / relative
        if (path.is_symlink() or not path.is_file() or
                not path.resolve().is_relative_to(ROOT) or file_digest(path) != item["sha256"]):
            raise RuntimeError(f"release identity mismatch: {relative}")
    return manifest


def check_contract(config: dict) -> None:
    from ebjd.config import validate_supported_config
    validate_supported_config(config)
    checks = {
        ("training", "epochs"): 30, ("training", "new_module_lr"): 1e-4,
        ("training", "unet_lr"): 1e-5, ("training", "lr_warmup_epochs"): 5,
        ("training", "lr_final_fraction"): 0.1,
        ("training", "optimizer"): "adamw",
        ("training", "rollout_training_enabled"): False,
        ("training", "paired_marginal_actual_step_projection_enabled"): False,
        ("training", "coupling_singleton_backward"): "legacy",
        ("denoiser", "hidden_dim"): 192, ("denoiser", "heads"): 4,
        ("denoiser", "ff_dim"): 768, ("denoiser", "ctf_feedback_basis"): "legacy",
        ("representation", "type"): "invertible_endpoint_brownian_bridge",
        ("objectives", "coarse_v_weight"): 0.25,
        ("objectives", "relative_motion_weight"): 0.05,
        ("objectives", "map_bce_weight"): 0.1,
        ("evaluation", "worlds"): 20, ("evaluation", "steps"): 20,
        ("evaluation", "selection_seed"): 2035,
        ("evaluation", "inference_seeds"): [2035, 2036, 2037, 2038, 2039],
    }
    mismatches = {".".join(key): (config.get(key[0], {}).get(key[1]), expected)
                  for key, expected in checks.items()
                  if config.get(key[0], {}).get(key[1]) != expected}
    early = config["training"].get("early_stopping", {})
    if early.get("max_epochs") != 30 or early.get("monitor") != "H_JADE":
        mismatches["early_stopping"] = (early, "bound H_JADE contract with max_epochs=30")
    if mismatches:
        raise ValueError(f"release contract differs from effective unified runner: {mismatches}")


def before_training() -> None:
    if "--help" in sys.argv or "-h" in sys.argv:
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--resume", action="store_true")
    args, _ = parser.parse_known_args()
    manifest = verify_release()
    from ebjd.config import load_config
    config = load_config(args.config)
    check_contract(config)
    if (args.source_commit != manifest["upstream_commit"] or
            config["experiment"]["source_commit"] != args.source_commit or
            config["experiment"]["fold"] != args.fold):
        raise ValueError("release, configuration and CLI identities differ")
    output = Path(args.output).resolve()
    if output.is_relative_to(ROOT) or output != Path(config["paths"]["output_root"]).resolve():
        raise ValueError("output must match resolved config and be outside the source release")
    binding = Path(args.config).resolve().with_name("PREPARATION.json")
    receipt = json.loads(binding.read_text())
    if (receipt["release_manifest_sha256"] != manifest_digest() or
            receipt["config_file_sha256"] != file_digest(Path(args.config))):
        raise ValueError("prepare again after a release/configuration change")
    last = output / "last.pt"
    if args.resume:
        if not last.is_file():
            raise ValueError("resume requires an existing complete last.pt")
        import torch
        saved = torch.load(last, map_location="cpu", weights_only=False)["run_identity"]
        if (saved.get("release_manifest_sha256") != manifest_digest() or
                saved.get("config_sha256") != file_digest(Path(args.config))):
            raise ValueError("resume belongs to another release or configuration")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("fresh training requires a new or empty output directory")

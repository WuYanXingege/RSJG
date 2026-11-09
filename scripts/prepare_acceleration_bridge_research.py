#!/usr/bin/env python3
"""Resolve one fresh, isolated research arm from a verified fold configuration."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import subprocess

import yaml

from ebjd.config import load_config, validate_supported_config, config_hash
from run_zara1_unified_coupling_bridge import atomic_json


def prepare(base: dict, arm: dict, output: Path, commit: str) -> dict:
    if (output / "last.pt").exists():
        raise ValueError("research output already contains a training checkpoint")
    config = deepcopy(base)
    early = config["training"].get("early_stopping")
    if (not isinstance(early, dict) or
            early.get("monitor") != "H_JADE" or
            early.get("max_epochs") != config["training"]["epochs"] or
            not {"earliest_stop_epoch", "patience_validations",
                 "significant_relative_improvement"}.issubset(early)):
        raise ValueError("base requires a complete unified early-stop/epoch contract")
    if (config["denoiser"]["hidden_dim"], config["denoiser"]["ff_dim"]) != (192, 768):
        raise ValueError("research arms require the current 192/4/768 model")
    config["representation"]["type"] = arm["representation_type"]
    config["denoiser"]["ctf_feedback_basis"] = "legacy"
    config["training"].update({
        "coupling_singleton_backward": arm["coupling_singleton_backward"],
        "rollout_training_enabled": False,
        "paired_marginal_actual_step_projection_enabled": False,
        "ebjd_input_pipeline": "legacy_cpu_rigid",
    })
    config["training"].pop("ebjd_input_pipeline_receipt", None)
    config["experiment"].update(source_commit=commit, implementation_base=commit[:7])
    config["paths"]["output_root"] = str(output.resolve())
    validate_supported_config(config)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--arm", choices=("control", "acceleration", "singleton_fix"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("use a new empty research output directory")
    plan = yaml.safe_load((Path(__file__).resolve().parents[1] /
        "configs/research/acceleration_bridge_v1.yaml").read_text())
    dirty = subprocess.check_output([
        "git", "status", "--porcelain", "--untracked-files=all", "--",
        "ebjd", "scripts", "configs"], text=True)
    if dirty.strip():
        raise ValueError("commit research code/config before binding a fresh run")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    base = load_config(args.base_config)
    config = prepare(base, plan["arms"][args.arm], output, commit)
    atomic_json(output / "RESOLVED.json", config)
    atomic_json(output / "PREPARATION.json", {
        "arm": args.arm, "base_config": str(Path(args.base_config).resolve()),
        "source_commit": commit, "config_content_sha256": config_hash(config),
        "initialization": "fresh; reuse only the verified same-fold map U-Net",
        "scale_fit": "runner refits scales on the complete train manifest",
        "training_started": False,
    })
    print(str(output / "RESOLVED.json"))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare a fresh unified research run with an explicit backbone/depth."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

import yaml

from ebjd.config import config_hash, load_config, validate_supported_config
from prepare_acceleration_bridge_research import prepare
from run_zara1_unified_coupling_bridge import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--arm", choices=("legacy6", "gated6", "gated8"), default="gated8")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("research output must be new or empty")
    dirty = subprocess.check_output([
        "git", "status", "--porcelain", "--untracked-files=all", "--",
        "ebjd", "scripts", "configs"], text=True)
    if dirty.strip():
        raise ValueError("commit source/config before binding a research run")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    plan = yaml.safe_load((Path(__file__).resolve().parents[1] /
        "configs/research/endpoint_gated_dit_v1.yaml").read_text())
    config = prepare(load_config(args.base_config), plan["shared"], output, commit)
    config["denoiser"].update(plan["arms"][args.arm])
    validate_supported_config(config)
    atomic_json(output / "RESOLVED.json", config)
    atomic_json(output / "PREPARATION.json", {
        "arm": args.arm, "source_commit": commit,
        "base_config": str(Path(args.base_config).resolve()),
        "config_content_sha256": config_hash(config),
        "denoiser": config["denoiser"],
        "initialization": "fresh body; verified same-fold map U-Net only",
        "prior": "brownian_position", "training_started": False,
    })
    print(str(output / "RESOLVED.json"))


if __name__ == "__main__":
    main()

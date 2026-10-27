#!/usr/bin/env python3
"""Replay one frozen MPJC bridge mode to recover first-11-frame metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ebjd.config import build_model, load_config
from ebjd.coupling import MPJCEndpointBank
from ebjd.proposals import file_sha256
from run_zara1_gt_conditioned_bridge import evaluate_mode
from run_zara1_native_prior_pilot import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--mode", choices=("native", "coupled"), required=True)
    parser.add_argument("--seed", type=int, default=2035)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    if file_sha256(checkpoint) != args.checkpoint_sha256:
        raise ValueError("bridge checkpoint SHA256 mismatch")
    config = load_config(args.model_config)
    device = torch.device(args.device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(config, proposal_conditioned=True).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    bank = MPJCEndpointBank(
        args.bank,
        expected_candidate_sha256=args.candidate_sha256,
        expected_manifest_sha256=args.manifest_sha256,
        expected_split="validation",
    )
    result = {
        "schema": "ebjd-zara1-bridge-prefix-replay-v1",
        "status": "COMPLETE",
        "checkpoint_sha256": args.checkpoint_sha256,
        "seed": args.seed,
        "mode": args.mode,
        "mpjc_bank": bank.path,
        "mpjc_bank_sha256": bank.sha256,
        "result": evaluate_mode(model, config, bank, args.mode, device, args.seed),
    }
    atomic_json(Path(args.output).resolve(), result)
    print(result, flush=True)


if __name__ == "__main__":
    main()

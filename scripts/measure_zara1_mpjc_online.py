#!/usr/bin/env python3
"""Measure one complete cached-G reference→MPJC→bridge validation pass."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ebjd.config import build_model, load_config
from ebjd.coupling import couple_scene, gather_worlds
from ebjd.metrics import MetricAccumulator
from ebjd.proposals import NativeProposalBank, file_sha256
from ebjd.sampling import predict
from run_zara1_native_prior_pilot import atomic_json, loader


def load_model(config, checkpoint, expected_sha, device):
    if file_sha256(checkpoint) != expected_sha:
        raise ValueError(f"checkpoint SHA mismatch: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(config, proposal_conditioned=True).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--reference-checkpoint", required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--bridge-checkpoint", required=True)
    parser.add_argument("--bridge-sha256", required=True)
    parser.add_argument("--proposal-bank", required=True)
    parser.add_argument("--gdts-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--seed", type=int, default=2035)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_config(args.model_config)
    device = torch.device(args.device)
    reference = load_model(
        config, Path(args.reference_checkpoint), args.reference_sha256, device)
    bridge = load_model(
        config, Path(args.bridge_checkpoint), args.bridge_sha256, device)
    bank = NativeProposalBank(
        args.proposal_bank, expected_checkpoint_sha256=args.gdts_sha256,
        expected_manifest_sha256=args.manifest_sha256,
        expected_split="validation")
    reference_generator = torch.Generator(device=device).manual_seed(args.seed)
    bridge_generator = torch.Generator(device=device).manual_seed(args.seed)
    accumulator = MetricAccumulator()
    matching_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for cpu_batch in loader(config, "validation", torch.Generator().manual_seed(99173)):
            native = bank.batch(cpu_batch)
            batch = cpu_batch.to(device)
            native_device = native.to(device)
            _, reference_goals = predict(
                reference, batch.observed, batch.semantic_maps, batch.valid,
                worlds=20, steps=20, generator=reference_generator,
                proposal_goals=native_device, proposal_eta=1.0)
            match_started = time.perf_counter()
            coupled, permutation = couple_scene(
                native[0].numpy(), reference_goals[0].float().cpu().numpy(),
                cpu_batch.valid[0].numpy())
            matching_seconds += time.perf_counter() - match_started
            coupled = torch.from_numpy(coupled).unsqueeze(0).to(device)
            permutation = torch.from_numpy(permutation).unsqueeze(0).to(device)
            base_noise = torch.randn(
                1, 20, batch.observed.shape[1], 12, 2,
                device=device, dtype=torch.float32, generator=bridge_generator)
            trajectory, goals = predict(
                bridge, batch.observed, batch.semantic_maps, batch.valid,
                worlds=20, steps=20, proposal_goals=coupled, proposal_eta=0.0,
                initial_noise=gather_worlds(base_noise, permutation))
            if float((goals - coupled).abs().nan_to_num().max().cpu()) != 0.0:
                raise RuntimeError("online MPJC bridge changed fixed endpoints")
            accumulator.update(trajectory, batch.future, batch.valid)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result = {
        "schema": "ebjd-zara1-mpjc-online-runtime-v1", "status": "COMPLETE",
        "seed": args.seed, "worlds": 20, "ddim_steps_per_network": 20,
        "reference_checkpoint_sha256": args.reference_sha256,
        "bridge_checkpoint_sha256": args.bridge_sha256,
        "proposal_bank_sha256": bank.sha256,
        "cached_native_proposal_provider_not_included": True,
        "reference_plus_matching_plus_bridge_seconds": elapsed,
        "matching_cpu_seconds_inclusive_of_reference_tensor_transfer": matching_seconds,
        "metrics": accumulator.as_dict(),
        "note": "Actual sequential online prototype cost; cached-G provider is separate.",
    }
    atomic_json(Path(args.output), result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

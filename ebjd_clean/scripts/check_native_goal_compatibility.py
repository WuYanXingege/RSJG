"""Emit a zero-gate and candidate-permutation compatibility receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ebjd.model import EBJDModel


def compare(device: torch.device, bf16: bool) -> dict:
    torch.manual_seed(1)
    legacy = EBJDModel(native_goal_conditioning=False).to(device).eval()
    torch.manual_seed(2)
    candidate = EBJDModel(native_goal_conditioning=True).to(device).eval()
    missing, unexpected = candidate.load_state_dict(legacy.state_dict(), strict=False)
    if unexpected or not missing or not all(
            name.startswith("denoiser.native_goal_conditioner.") for name in missing):
        raise RuntimeError("model-extension key contract failed")
    torch.manual_seed(17)
    observed = torch.randn(1, 2, 8, 2, device=device)
    maps = torch.zeros(1, 2, 6, 32, 32, device=device)
    valid = torch.ones(1, 2, dtype=torch.bool, device=device)
    latent = torch.randn(1, 3, 2, 12, 2, device=device)
    time = torch.tensor([0.4], device=device)
    goals = torch.randn(1, 2, 20, 2, device=device)
    enabled = bf16 and device.type == "cuda"
    with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=enabled):
        reference = legacy(observed, maps, valid, latent, time)[0]
        zero_gate = candidate(observed, maps, valid, latent, time, goals)[0]
        candidate.denoiser.native_goal_conditioner.gate.fill_(0.2)
        ordered = candidate(observed, maps, valid, latent, time, goals)[0]
        permuted = candidate(observed, maps, valid, latent, time, goals.flip(2))[0]
    return {
        "device": str(device),
        "precision": "CUDA BF16 autocast" if enabled else "FP32",
        "zero_gate_max_abs_error": float((reference - zero_gate).abs().max().cpu()),
        "zero_gate_bitwise_equal": bool(torch.equal(reference, zero_gate)),
        "candidate_permutation_max_abs_error": float(
            (ordered - permuted).abs().max().cpu()),
        "candidate_permutation_atol": 2e-6 if not enabled else 2e-3,
        "candidate_permutation_rtol": 2e-6 if not enabled else 2e-3,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    receipt = {
        "schema": "ebjd-native-goal-compatibility-v1",
        "status": "PASS",
        "checks": [compare(device, False)],
    }
    if device.type == "cuda":
        receipt["checks"].append(compare(device, True))
    for check in receipt["checks"]:
        if not check["zero_gate_bitwise_equal"]:
            receipt["status"] = "FAIL"
        if check["candidate_permutation_max_abs_error"] > (
                check["candidate_permutation_atol"]
                + check["candidate_permutation_rtol"]):
            receipt["status"] = "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

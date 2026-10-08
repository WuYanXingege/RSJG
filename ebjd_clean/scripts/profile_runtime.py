"""Reproducible EBJD inference/rollout profiler; uses synthetic geometry only."""

from __future__ import annotations

import argparse
import json
import time

import torch

from ebjd.data import SyntheticSceneDataset, collate_scenes
from ebjd.model import EBJDModel
from ebjd.sampling import differentiable_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pixels", type=int, default=256)
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = EBJDModel().to(device).train(args.backward)
    batch = collate_scenes([
        SyntheticSceneDataset(1, args.agents, args.pixels, seed=11)[0]
    ]).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.set_grad_enabled(args.backward):
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=device.type == "cuda" and torch.cuda.is_bf16_supported()):
            context = model.encode_context(batch.observed, batch.semantic_maps, batch.valid)
            noise = torch.randn(1, args.worlds, batch.observed.shape[1], 12, 2, device=device)
            trajectory, goal, _ = differentiable_sample(
                model, context, noise, args.steps, checkpoint_steps=args.backward)
            loss = trajectory.float().square().mean() + goal.float().square().mean()
        if args.backward:
            loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result = {
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch": torch.__version__, "pixels": args.pixels, "agents": args.agents,
        "worlds": args.worlds, "steps": args.steps, "backward": args.backward,
        "elapsed_seconds": elapsed, "loss_finite": bool(torch.isfinite(loss)),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "parameters_with_grad": sum(parameter.grad is not None for parameter in model.parameters()),
    }
    if device.type == "cuda":
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

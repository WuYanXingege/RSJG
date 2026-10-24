#!/usr/bin/env python3
"""Compare last-call coarse and final endpoints on one formal DDIM chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ebjd.config import build_model, load_config
from ebjd.data import SceneManifestDataset, collate_scenes
from ebjd.representation import cosine_vp


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
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


@torch.no_grad()
def sample_goal_stages(model, observed, semantic_maps, valid, worlds, steps, generator):
    """Return the last denoiser call's coarse goal and the final decoded goal."""
    context = model.encode_context(observed, semantic_maps, valid)
    latent = torch.randn(
        observed.shape[0], worlds, observed.shape[1], 12, 2,
        device=observed.device, dtype=torch.float32, generator=generator)
    last_coarse_goal = None
    batch = latent.shape[0]
    for index in range(steps, 0, -1):
        time_value = index / steps
        time = torch.full(
            (batch,), time_value, device=latent.device, dtype=torch.float32)
        next_time = torch.full_like(time, (index - 1) / steps)
        velocity, _, _, coarse_goal = model.denoiser(latent, time, context)
        if index == 1:
            last_coarse_goal = coarse_goal.clone()
        with torch.autocast(device_type=latent.device.type, enabled=False):
            alpha, sigma = cosine_vp(time)
            next_alpha, next_sigma = cosine_vp(next_time)
            shape = (batch, 1, 1, 1, 1)
            alpha = alpha.float().view(shape)
            sigma = sigma.float().view(shape)
            next_alpha = next_alpha.float().view(shape)
            next_sigma = next_sigma.float().view(shape)
            clean = alpha * latent.float() - sigma * velocity.float()
            noise = sigma * latent.float() + alpha * velocity.float()
            latent = next_alpha * clean + next_sigma * noise
    if last_coarse_goal is None:
        raise RuntimeError("DDIM chain did not execute its last denoiser call")
    _, final_goal = model.representation.decode(latent, context.baseline)
    return last_coarse_goal, final_goal


class TerminalAccumulator:
    def __init__(self) -> None:
        self.groups = {
            "all": {"agent_error": 0.0, "agents": 0, "scene_error": 0.0, "scenes": 0},
            "N=1": {"agent_error": 0.0, "agents": 0, "scene_error": 0.0, "scenes": 0},
            "N>=2": {"agent_error": 0.0, "agents": 0, "scene_error": 0.0, "scenes": 0},
        }

    def update(self, goals: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor) -> None:
        # goals [B,P,N,2], truth [B,N,2], valid [B,N]
        distances = torch.linalg.vector_norm(
            goals.float() - truth[:, None].float(), dim=-1)
        for scene in range(goals.shape[0]):
            mask = valid[scene].bool()
            count = int(mask.sum())
            if count < 1:
                raise ValueError("terminal metrics require a valid agent")
            per_world = distances[scene, :, mask]
            marginal = per_world.min(0).values
            joint = per_world.mean(1).min()
            label = "N=1" if count == 1 else "N>=2"
            for name in ("all", label):
                group = self.groups[name]
                group["agent_error"] += float(marginal.double().sum())
                group["agents"] += count
                group["scene_error"] += float(joint.double())
                group["scenes"] += 1

    def summary(self) -> dict:
        result = {}
        for name, group in self.groups.items():
            if not group["scenes"]:
                result[name] = {"status": "NO_SCENES", **group}
                continue
            result[name] = {
                "status": "COMPLETE",
                "minFDE": group["agent_error"] / group["agents"],
                "JFDE": group["scene_error"] / group["scenes"],
                "agent_count": group["agents"],
                "scene_count": group["scenes"],
            }
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--seed", type=int, default=2035)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    actual = sha256(args.checkpoint)
    if actual != args.checkpoint_sha256:
        raise ValueError(f"checkpoint SHA256 mismatch: {actual}")
    config = load_config(args.config)
    evaluation = config["evaluation"]
    worlds, steps = int(evaluation["worlds"]), int(evaluation["steps"])
    if (worlds, steps) != (20, 20):
        raise ValueError("formal CTF goal-stage comparison requires 20 worlds/20 steps")
    manifest = config["paths"][f"{args.split}_manifest"]
    loader = DataLoader(
        SceneManifestDataset(manifest), batch_size=1, shuffle=False,
        collate_fn=collate_scenes, num_workers=0)
    device = torch.device(args.device)
    model = build_model(config).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    coarse, final = TerminalAccumulator(), TerminalAccumulator()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    for batch in loader:
        batch = batch.to(device)
        coarse_goal, final_goal = sample_goal_stages(
            model, batch.observed, batch.semantic_maps, batch.valid,
            worlds, steps, generator)
        truth = batch.future[..., -1, :]
        coarse.update(coarse_goal, truth, batch.valid)
        final.update(final_goal, truth, batch.valid)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    payload = {
        "schema": "ebjd-ctf-goal-stages-v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": actual,
        "config_sha256": sha256(args.config),
        "manifest": str(Path(manifest).resolve()),
        "manifest_sha256": sha256(manifest),
        "split": args.split,
        "seed": args.seed,
        "worlds": worlds,
        "steps": steps,
        "coarse_goal_source": "last_real_ddim_denoiser_call_at_t=1/steps",
        "coarse": coarse.summary(),
        "final": final.summary(),
        "elapsed_seconds": time.perf_counter() - started,
        "matched_gdts_subgroups": "NOT_VERIFIED_NO_EXISTING_EXPORT",
    }
    atomic_json(Path(args.output), payload)
    print(json.dumps({
        "coarse": payload["coarse"]["all"],
        "final": payload["final"]["all"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

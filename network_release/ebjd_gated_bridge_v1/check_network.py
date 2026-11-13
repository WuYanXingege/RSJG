#!/usr/bin/env python3
"""CPU-only standalone wiring check; synthetic data, no accuracy measurement."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

from ebjd.data import SyntheticSceneDataset, collate_scenes
from ebjd.unified import gather_candidates
from network import build_network, predict_g_h
from release_guard import ROOT, manifest_digest, verify_release
from train import atomic_json, bridge_and_pair_loss, optimizer_for


def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.is_relative_to(ROOT):
        raise ValueError("use a new receipt path outside the source snapshot")
    started = time.perf_counter()
    manifest = verify_release()
    torch.set_num_threads(2)
    profiles = {}
    for profile in ("legacy6", "gated6", "gated8"):
        torch.manual_seed(3101)
        model = build_network(profile)
        profiles[profile] = {
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "architecture": model.bridge.denoiser.architecture_signature(),
        }
    # Exercise the real 192/4/768 eight-block network on a small synthetic map.
    # This checks wiring, not a 256px performance/real-data acceptance gate.
    torch.manual_seed(3101)
    model = build_network("gated8").train()
    model.bridge.encoder.activation_checkpointing = False
    model.bridge.denoiser.activation_checkpointing = False
    dataset = SyntheticSceneDataset(2, 3, 32, seed=3101)
    items = []
    for index, count in enumerate((1, 2)):
        item = dict(dataset[index])
        for key in ("observed", "future", "semantic_maps"):
            item[key] = item[key][:count]
        items.append(item)
    batch = collate_scenes(items)
    proposals = batch.future[:, None, :, -1].expand(-1, 20, -1, -1).clone()
    proposals += torch.randn_like(proposals) * 0.2
    time_draw = torch.tensor([0.3, 0.7])
    noise = torch.randn(2, 1, 2, 12, 2)
    settings = {"training": {"weight_decay": 1e-4, "optimizer_betas": [0.9, 0.999],
                             "optimizer_eps": 1e-8}}
    optimizer = optimizer_for(model, settings)
    initial = {name: tensor_digest(p) for name, p in model.named_parameters()}
    grad_seen, losses = set(), []
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _, _, _ = bridge_and_pair_loss(
            model, batch, proposals, time_draw=time_draw, noise=noise)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite complete unified loss")
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f"non-finite gradient: {name}")
                if parameter.grad.abs().sum() > 0:
                    grad_seen.add(name)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    prefixes = ["coupling", "bridge.encoder.map_unet", "bridge.encoder.history_gru",
                "bridge.denoiser.endpoint_condition", "bridge.denoiser.clean_injection",
                "bridge.denoiser.map_injection", "bridge.denoiser.future_geometry_mlp"]
    prefixes += [f"bridge.denoiser.blocks.{i}.{branch}"
                 for i in (0, 7) for branch in ("temporal", "social", "cross", "ff")]
    changed = {name for name, p in model.named_parameters() if tensor_digest(p) != initial[name]}
    checks = {prefix: {"nonzero_gradient": any(n.startswith(prefix) for n in grad_seen),
                       "parameter_updated": any(n.startswith(prefix) for n in changed)}
              for prefix in prefixes}
    if not all(all(value.values()) for value in checks.values()):
        raise RuntimeError(f"missing trainable branch: {checks}")

    restored = build_network("gated8")
    restored.load_state_dict(deepcopy(model.state_dict()), strict=True)
    restored_optimizer = optimizer_for(restored, settings)
    restored_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    if not all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items()):
        raise RuntimeError("strict state restore changed parameters/buffers")
    for key, state in optimizer.state_dict()["state"].items():
        for name, value in state.items():
            other = restored_optimizer.state_dict()["state"][key][name]
            if isinstance(value, torch.Tensor) and not torch.equal(value, other):
                raise RuntimeError("optimizer moment restore differs")
    for wrong_profile in ("legacy6", "gated6"):
        wrong = build_network(wrong_profile)
        try:
            wrong.load_state_dict(model.state_dict(), strict=True)
        except RuntimeError:
            pass
        else:
            raise RuntimeError("wrong architecture accepted checkpoint")
        del wrong

    model.eval()
    initial_noise = torch.randn(2, 20, 2, 12, 2)
    result = predict_g_h(model, batch, proposals, initial_noise)
    if not torch.isfinite(result["G"]).all() or not torch.isfinite(result["H"]).all():
        raise RuntimeError("non-finite DDIM output")
    indices = result["permutation"]
    expected = torch.arange(20).view(1, 20, 1).expand_as(indices)
    assert torch.equal(indices.sort(dim=1).values, expected)
    assert torch.equal(result["H_goals"], gather_candidates(proposals, indices))
    # Model package must come entirely from the copied standalone directory.
    imports = {}
    for name, module in sys.modules.items():
        if name == "ebjd" or name.startswith("ebjd."):
            path = Path(module.__file__).resolve()
            if not path.is_relative_to(ROOT):
                raise RuntimeError(f"escaped standalone package: {name} -> {path}")
            imports[name] = str(path.relative_to(ROOT))
    atomic_json(output, {
        "status": "PASS_INTERFACE_ONLY", "upstream_commit": manifest["upstream_commit"],
        "release_manifest_sha256": manifest_digest(), "device": "cpu", "precision": "fp32",
        "synthetic": True, "map_pixels": 32, "real_data_accuracy_test": False,
        "temporary_optimizer_updates": 4, "formal_optimizer_updates": 0,
        "gpu_wall_seconds": 0, "wall_seconds": time.perf_counter() - started,
        "profiles": profiles, "losses": losses, "branch_checks": checks,
        "strict_restore": True, "optimizer_restore": True,
        "wrong_architecture_rejected": True, "paired_ddim_worlds": 20,
        "paired_ddim_steps": 20, "literal_endpoints": True,
        "candidate_permutations": True, "package_imports": imports,
        "torch": torch.__version__,
    })
    print(json.dumps({"status": "PASS_INTERFACE_ONLY", "receipt": str(output)}))


if __name__ == "__main__":
    main()

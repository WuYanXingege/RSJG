#!/usr/bin/env python3
"""Read-only 24-window split-attention observer for frozen V2_FULL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ebjd.config import build_model, load_config
from ebjd.data import SceneManifestDataset, collate_scenes


def identity_hash(item: dict) -> str:
    text = item["scene_id"] + "|" + "|".join(item["agent_ids"])
    return hashlib.sha256(text.encode()).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(-1) / (
        torch.linalg.vector_norm(a, dim=-1)
        * torch.linalg.vector_norm(b, dim=-1)).clamp_min(1e-12)


def redundancy(memory: torch.Tensor) -> float:
    value = torch.nn.functional.normalize(memory.float(), dim=-1)
    matrix = value @ value.transpose(-1, -2)
    count = matrix.shape[-1]
    if count <= 1:
        return 0.0
    mask = ~torch.eye(count, dtype=torch.bool, device=matrix.device)
    return float(matrix[..., mask].mean().cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    model = build_model(config, "v2_full").to(device).float().eval()
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    bank = config["native_goal_conditioning"]["validation_cache_by_seed"]["2035"]
    dataset = SceneManifestDataset(config["paths"]["validation_manifest"],
                                   native_goal_bank=bank)
    strata = {"N=1": [], "N>=2": []}
    for index in range(len(dataset)):
        item = dataset[index]
        name = "N=1" if len(item["agent_ids"]) == 1 else "N>=2"
        strata[name].append((identity_hash(item), index))
    selected = {name: [index for _, index in sorted(rows)[:12]]
                for name, rows in strata.items()}
    if any(len(rows) != 12 for rows in selected.values()):
        raise ValueError("held-out manifest lacks 12 windows in an observer stratum")

    sums: dict[str, list[float]] = defaultdict(list)
    neutral = []
    generator = torch.Generator(device=device).manual_seed(2035)
    for stratum, indices in selected.items():
        for index in indices:
            batch = collate_scenes([dataset[index]]).to(device)
            with torch.no_grad():
                context = model.encode_context(
                    batch.observed, batch.semantic_maps, batch.valid,
                    batch.native_goals)
                context, _, _ = model.assemble_world_anchors(
                    context, batch.scene_ids, batch.agent_ids, 2035)
                context = model.denoiser.prepare_static_context(context)
                noise = torch.randn(
                    1, 20, batch.observed.shape[1], 12, 2,
                    generator=generator, device=device)
                sums[f"{stratum}/motion_token_redundancy"].append(
                    redundancy(context.motion_memory[batch.valid]))
                sums[f"{stratum}/map_token_redundancy"].append(
                    redundancy(context.map_memory[batch.valid]))
                for tau in (1.0, 0.5, 0.05):
                    captured = {block: {"cross": [], "gate": None}
                                for block in range(6)}
                    handles = []
                    for block_index, block in enumerate(model.denoiser.blocks):
                        def block_pre(module, positional, keyword, i=block_index):
                            captured[i]["gate"] = keyword["memory_gate"].detach()
                        def cross_hook(module, positional, output, i=block_index):
                            captured[i]["cross"].append(output[0].detach())
                        handles.append(block.register_forward_pre_hook(
                            block_pre, with_kwargs=True))
                        handles.append(block.cross.register_forward_hook(cross_hook))
                    reference = model.denoiser(
                        noise, torch.tensor([tau], device=device), context)[0]
                    for handle in handles:
                        handle.remove()
                    observed = model.denoiser(
                        noise, torch.tensor([tau], device=device), context)[0]
                    neutral.append(bool(torch.equal(reference, observed)))
                    for block_index, block in enumerate(model.denoiser.blocks):
                        values = captured[block_index]["cross"]
                        if len(values) != 2:
                            raise RuntimeError("split observer expected two context calls")
                        gate = captured[block_index]["gate"].reshape(-1, 2)
                        bias = block.cross.out_proj.bias.detach().view(1, 1, -1)
                        motion = values[0] - bias
                        map_value = values[1] - bias
                        for query, token_slice in (
                            ("bridge", slice(0, 11)), ("goal", slice(11, 12))):
                            g_h = gate[:, 0, None, None]
                            g_m = gate[:, 1, None, None]
                            h = motion[:, token_slice]
                            m = map_value[:, token_slice]
                            prefix = f"{stratum}/tau={tau}/block={block_index}/{query}"
                            sums[prefix + "/motion_effective_mass"].append(float(g_h.mean().cpu()))
                            sums[prefix + "/map_effective_mass"].append(float(g_m.mean().cpu()))
                            sums[prefix + "/motion_mass_per_token"].append(
                                float((g_h / context.motion_memory.shape[-2]).mean().cpu()))
                            sums[prefix + "/map_mass_per_token"].append(
                                float((g_m / context.map_memory.shape[-2]).mean().cpu()))
                            sums[prefix + "/motion_contribution_norm"].append(
                                float(torch.linalg.vector_norm(g_h * h, dim=-1).mean().cpu()))
                            sums[prefix + "/map_contribution_norm"].append(
                                float(torch.linalg.vector_norm(g_m * m, dim=-1).mean().cpu()))
                            sums[prefix + "/group_contribution_cosine"].append(
                                float(cosine(h, m).mean().cpu()))
                            sums[prefix + "/gate_motion"].extend(
                                gate[:, 0].detach().cpu().tolist())
    summary = {name: {
        "mean": float(np.mean(values)), "p05": float(np.quantile(values, .05)),
        "p50": float(np.quantile(values, .50)), "p95": float(np.quantile(values, .95)),
        "count": len(values),
    } for name, values in sums.items()}
    receipt = {
        "schema": "ebjd-v2-split-attention-observer-v1",
        "status": "PASS" if all(neutral) else "FAIL",
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "selection": {
            "method": "identity SHA256 ascending within N=1/N>=2, no error ranking",
            "seed": 2035, "counts": {name: len(rows) for name, rows in selected.items()},
            "indices": selected,
        },
        "tau": [1.0, 0.5, 0.05], "blocks": 6,
        "observer_output_bitwise_neutral": all(neutral),
        "group_semantics_boundary": (
            "map tokens include observed-history heatmaps; labels denote memory branches, "
            "not pure map versus pure motion information."),
        "legacy_equal_logit_count_reference": {"motion_8": 1 / 9, "map_64": 8 / 9},
        "summary": summary,
    }
    atomic_json(Path(args.output), receipt)
    print(json.dumps({"status": receipt["status"],
                      "statistics": len(summary)}, sort_keys=True))
    if receipt["status"] != "PASS":
        raise RuntimeError("attention observer changed normal-path output")


if __name__ == "__main__":
    main()

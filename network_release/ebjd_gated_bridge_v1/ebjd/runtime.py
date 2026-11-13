"""Unchanged loader and train-only RMS fitting from the paper runner."""
from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from .data import SceneManifestDataset, collate_scenes, collate_scenes_compact_maps
from .representation import EndpointBridgeRepresentation, cv_baseline

def loader(config: dict, split: str, generator: torch.Generator) -> DataLoader:
    dataset = SceneManifestDataset(config["paths"][f"{split}_manifest"])
    workers = int(config["training"].get("data_loader_workers", 0))
    input_pipeline = str(config["training"].get(
        "ebjd_input_pipeline", "legacy_cpu_rigid"))
    if input_pipeline not in {"legacy_cpu_rigid", "compact_uint8_gpu_rigid_v1"}:
        raise ValueError(f"unsupported EBJD input pipeline: {input_pipeline}")
    compact_maps = input_pipeline == "compact_uint8_gpu_rigid_v1"
    arguments = dict(
        dataset=dataset,
        # The frozen endpoint attribution used one complete scene per
        # validation batch.  Preserve that RNG-to-scene mapping exactly;
        # training retains the formal four-scene logical microbatch.
        batch_size=(int(config["training"]["batch_scenes"])
                    if split == "train" else 1),
        shuffle=split == "train",
        collate_fn=(collate_scenes_compact_maps if compact_maps else collate_scenes),
        num_workers=workers, generator=generator,
        pin_memory=compact_maps and torch.cuda.is_available())
    if workers:
        arguments.update(
            prefetch_factor=int(config["training"].get(
                "data_loader_prefetch_factor", 2)), persistent_workers=True)
    return DataLoader(**arguments)


@torch.no_grad()
def fit_training_scales(model, loader) -> tuple[float, float] | None:
    if not isinstance(model.representation, EndpointBridgeRepresentation):
        return None
    goal_sum = bridge_sum = 0.0
    goal_count = bridge_count = 0
    for batch in loader:
        baseline, _ = cv_baseline(batch.observed)
        unscaled = model.representation._encode_unscaled(
            batch.future.double(), baseline.double())
        goal_mask = batch.valid[..., None, None].expand_as(unscaled[..., 11:, :])
        bridge_mask = batch.valid[..., None, None].expand_as(unscaled[..., :11, :])
        goal_sum += float(unscaled[..., 11:, :][goal_mask].square().sum())
        bridge_sum += float(unscaled[..., :11, :][bridge_mask].square().sum())
        goal_count += int(goal_mask.sum())
        bridge_count += int(bridge_mask.sum())
    goal_scale = max((goal_sum / max(goal_count, 1)) ** 0.5, 0.5)
    bridge_scale = max((bridge_sum / max(bridge_count, 1)) ** 0.5, 0.1)
    model.representation.set_scales(goal_scale, bridge_scale)
    model.encoder.goal_scale.fill_(goal_scale)
    return goal_scale, bridge_scale


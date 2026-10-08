"""Command line training entry point: ``python -m ebjd.train``."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import build_model, load_config, trainer_options
from .data import NPZSceneDataset, SyntheticSceneDataset, augment_batch, collate_scenes
from .metrics import hard_metrics
from .sampling import predict
from .trainer import EBJDTrainer
from .representation import EndpointBridgeRepresentation, cv_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train independent EBJD")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ablation", default="none")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume")
    parser.add_argument("--smoke", action="store_true", help="use synthetic data and one epoch")
    return parser.parse_args()


def _loader(config: dict, split: str, smoke: bool) -> DataLoader:
    if smoke:
        dataset = SyntheticSceneDataset(scenes=2, agents=2, pixels=32)
    else:
        path = config.get("paths", {}).get(f"{split}_npz")
        if not path:
            raise ValueError(f"paths.{split}_npz is required outside --smoke")
        dataset = NPZSceneDataset(path)
    batch_size = 1 if smoke else int(config["training"].get("batch_scenes", 4))
    return DataLoader(dataset, batch_size=batch_size, shuffle=split == "train",
                      collate_fn=collate_scenes, num_workers=0)


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
    return goal_scale, bridge_scale


@torch.no_grad()
def validate(model, loader, device: str, worlds: int, steps: int, seed: int = 2035) -> dict:
    model.eval()
    values = []
    generator = torch.Generator(device=device).manual_seed(seed)
    for batch in loader:
        batch = batch.to(device)
        trajectory, _ = predict(
            model, batch.observed, batch.semantic_maps, batch.valid,
            worlds=worlds, steps=steps, generator=generator)
        values.append(hard_metrics(trajectory, batch.future, batch.valid))
    names = values[0].__dataclass_fields__
    return {name: float(np.mean([getattr(item, name) for item in values])) for name in names}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 3101))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    model = build_model(config, args.ablation).to(device)
    options = trainer_options(config, args.ablation)
    if args.smoke:
        options.epochs = 1
        options.accumulation = 1
        options.rollout_start_epoch = 999
    output = Path(config.get("paths", {}).get("output_dir", "outputs/ebjd"))
    output.mkdir(parents=True, exist_ok=True)
    train_loader = _loader(config, "train", args.smoke)
    validation_loader = _loader(config, "validation", args.smoke)
    fitted_scales = fit_training_scales(model, train_loader) if not args.resume else None
    trainer = EBJDTrainer(model, options)
    start_epoch = 1
    if args.resume:
        start_epoch = int(trainer.load_checkpoint(args.resume)["epoch"]) + 1
    if fitted_scales:
        print(json.dumps({"fitted_goal_scale": fitted_scales[0],
                          "fitted_bridge_scale": fitted_scales[1]}), flush=True)
    log_path = output / "train.jsonl"
    best: tuple[float, float, int] | None = None
    for epoch in range(start_epoch, options.epochs + 1):
        model.train()
        pending = []
        for batch in train_loader:
            pending.append(augment_batch(batch).to(device))
            if len(pending) == options.accumulation:
                record = trainer.train_step(pending, epoch)
                record.update({"epoch": epoch, "update": trainer.update_index})
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                pending = []
        if pending:
            record = trainer.train_step(pending, epoch)
            record.update({"epoch": epoch, "update": trainer.update_index})
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        evaluation = config.get("evaluation", {})
        worlds = 2 if args.smoke else int(evaluation.get("worlds", 20))
        steps = 2 if args.smoke else int(evaluation.get("steps", 20))
        metrics = validate(model, validation_loader, str(device), worlds, steps)
        selection = (metrics["JADE"], metrics["JFDE"], epoch)
        trainer.save_checkpoint(output / "last.pt", epoch, {"validation": metrics})
        if best is None or selection < best:
            best = selection
            trainer.save_checkpoint(output / "best.pt", epoch, {"validation": metrics})
        print(json.dumps({"epoch": epoch, "validation": metrics}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

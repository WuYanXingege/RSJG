#!/usr/bin/env python3
"""One-update acceptance check for the immutable upstream GDTS HOTEL fold."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


OFFICIAL_COMMIT = "297d508558c10831983ea4b19c2b3e657459a449"
CODE_FILES = (
    "main.py",
    "train.sh",
    "src/parser.py",
    "src/trainer.py",
    "src/data_pre_process.py",
    "src/data_loader.py",
    "src/models/model.py",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    root = args.official_root.resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))

    import subprocess
    import torch
    from src.data_loader import get_dataloader
    from src.metrics import compute_metric_mask
    from src.models.model import GDTS
    from src.parser import get_parser, check_and_add_additional_args
    from src.utils import set_seed

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != OFFICIAL_COMMIT:
        raise RuntimeError("official source commit mismatch")
    train = root / "output/hotel/data_batches/train_batches"
    valid = root / "output/hotel/data_batches/valid_batches"
    if len(list(train.glob("*.pkl"))) != 591:
        raise RuntimeError("official train cache count")
    if len(list(valid.glob("*.pkl"))) != 19:
        raise RuntimeError("official valid cache count")
    query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True, capture_output=True)
    if query.returncode or query.stdout.strip():
        raise RuntimeError("GPU busy or query failed")

    argv = [
        "--dataset", "eth5", "--test_set", "hotel", "--phase", "train",
        "--num_epochs", "250", "--batch_size", "64",
        "--start_validation", "1", "--validate_every", "10",
        "--learning_rate", "0.001", "--skip_ts_window", "1",
        "--down_factor", "8", "--num_workers", "0",
        "--use_wandb", "False", "--seed", "2025",
    ]
    model_args = check_and_add_additional_args(
        get_parser().parse_args(argv))
    if not model_args.use_cuda or model_args.device != "cuda:0":
        raise RuntimeError("CUDA unavailable")
    set_seed(seed_value=model_args.seed, use_cuda=True)
    torch.cuda.reset_peak_memory_stats()
    loader = get_dataloader(model_args, "train")
    model = GDTS(model_args, torch.device(model_args.device)).to(
        model_args.device)
    prefixes = ("goal_module.", "registrar.", "diffnet.")
    before = {
        prefix: torch.cat([
            value.detach().reshape(-1).cpu()
            for name, value in model.named_parameters()
            if name.startswith(prefix)
        ]).clone()
        for prefix in prefixes
    }
    optimizer = torch.optim.Adam(
        model.parameters(), lr=model_args.learning_rate)
    batch, batch_ids = next(iter(loader))
    inputs, sequence = model.prepare_inputs(batch, batch_ids)
    optimizer.zero_grad(set_to_none=True)
    losses = model.get_loss(inputs, sequence)
    coefficients = model.set_losses_coeffs()
    loss = sum(losses[name] * coefficients[name] for name in losses)
    loss.backward()
    gradients = {
        prefix: sum(
            float(value.grad.detach().abs().sum())
            for name, value in model.named_parameters()
            if name.startswith(prefix) and value.grad is not None)
        for prefix in prefixes
    }
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), model_args.clip, error_if_nonfinite=True)
    optimizer.step()
    changed = {
        prefix: not torch.equal(
            before[prefix],
            torch.cat([
                value.detach().reshape(-1).cpu()
                for name, value in model.named_parameters()
                if name.startswith(prefix)
            ]))
        for prefix in prefixes
    }
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite official loss")
    if not all(value > 0 for value in gradients.values()):
        raise RuntimeError("missing official gradient family")
    if not all(changed.values()):
        raise RuntimeError("official parameter family unchanged")

    valid_loader = get_dataloader(model_args, "valid")
    valid_batch, valid_ids = next(iter(valid_loader))
    valid_inputs, valid_sequence = model.prepare_inputs(
        valid_batch, valid_ids)
    metric_mask = compute_metric_mask(valid_sequence)
    model.eval()
    with torch.no_grad():
        predictions, auxiliary = model(valid_inputs, if_test=True)
        validation = {}
        for metric_name in model.init_test_metrics():
            values = model.compute_model_metrics(
                metric_name=metric_name,
                predictions=predictions,
                metric_mask=metric_mask,
                all_aux_outputs=auxiliary,
                inputs=valid_inputs,
                obs_length=model_args.obs_length,
            )
            validation[metric_name] = {
                "count": len(values),
                "mean": float(sum(values) / len(values)),
            }
    if not validation or not all(
            torch.isfinite(torch.tensor(item["mean"]))
            for item in validation.values()):
        raise RuntimeError("invalid official validation metrics")

    result = {
        "status": "PASS",
        "official_commit": head,
        "code_sha256": {
            name: sha256(root / name) for name in CODE_FILES
        },
        "cache": {"train_batches": 591, "valid_batches": 19},
        "configuration": {
            "seed": 2025,
            "batch_size": 64,
            "learning_rate": 0.001,
            "precision": "fp32",
            "data_augmentation": True,
            "updates": 1,
        },
        "batch_agents": int(inputs["abs_pixel_coord"].shape[1]),
        "loss": float(loss.detach()),
        "components": {
            name: float(value.detach()) for name, value in losses.items()
        },
        "coefficients": coefficients,
        "gradient_l1": gradients,
        "gradient_norm": float(gradient_norm),
        "changed": changed,
        "validation_smoke": {
            "batches": 1,
            "samples": 20,
            "metrics": validation,
        },
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    destination = args.receipt.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, destination)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

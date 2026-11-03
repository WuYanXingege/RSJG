#!/usr/bin/env python3
"""CPU-only SDD data/interface preflight; never creates a formal proposal bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ebjd.data import SceneManifestDataset, augment_batch, collate_scenes
from ebjd.model import EBJDModel
from ebjd.objectives import diffusion_loss, map_loss
from ebjd.representation import EndpointBridgeDecomposition, cv_baseline


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    train_path = Path(args.train_manifest)
    validation_path, test_path = Path(args.validation_manifest), Path(args.test_manifest)
    train = SceneManifestDataset(train_path)
    validation = SceneManifestDataset(validation_path)
    test = SceneManifestDataset(test_path)
    n1 = next(index for index in range(len(validation))
              if validation[index]["observed"].shape[0] == 1)
    nm = next(index for index in range(len(validation))
              if validation[index]["observed"].shape[0] >= 2)
    batch = collate_scenes([validation[n1], validation[nm]])
    padded_finite = all(torch.isfinite(value).all() for value in (
        batch.observed, batch.future, batch.semantic_maps))
    generator = torch.Generator().manual_seed(3101)
    augmented = augment_batch(batch, generator)
    augmentation_finite = all(torch.isfinite(value).all() for value in (
        augmented.observed, augmented.future, augmented.semantic_maps))

    baseline, _ = cv_baseline(batch.observed)
    representation = EndpointBridgeDecomposition()
    latent = representation.encode_target(batch.future, baseline)
    reconstructed, _ = representation.decode(latent, baseline)
    reconstruction_max_abs_m = float((reconstructed - batch.future).abs().max())

    # A single real SDD scene is sufficient to exercise the full 192/4/768
    # forward/backward interface without claiming a benchmark result.
    one = collate_scenes([validation[n1]])
    model = EBJDModel(hidden_dim=192, attention_heads=4, ff_dim=768)
    model.encoder.activation_checkpointing = False
    model.denoiser.activation_checkpointing = False
    context = model.encode_context(one.observed, one.semantic_maps, one.valid)
    noisy = torch.randn(1, 1, 1, 12, 2, generator=torch.Generator().manual_seed(7))
    final, coarse, _, _ = model.denoiser(noisy, torch.tensor([0.5]), context)
    target = torch.randn(final.shape, generator=torch.Generator().manual_seed(8))
    loss = diffusion_loss(final, coarse, target, one.valid) + 0.1 * map_loss(
        one.future, context)
    loss.backward()
    gradients_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters())

    validation_manifest = json.loads(validation_path.read_text())
    test_manifest = json.loads(test_path.read_text())
    receipt = {
        "schema": "ebjd-sdd-cpu-preflight-v1",
        "purpose": "interface_only_not_a_formal_proposal_or_performance_run",
        "validation_manifest_sha256": sha256(validation_path),
        "test_manifest_sha256": sha256(test_path),
        "validation_scenes": len(validation), "test_scenes": len(test),
        "validation_test_window_identity": (
            validation_manifest["window_keys"] == test_manifest["window_keys"]),
        "single_agent_index": n1, "multi_agent_index": nm,
        "multi_agent_count": int(validation[nm]["observed"].shape[0]),
        "padding_shape": list(batch.observed.shape),
        "padding_and_values_finite": bool(padded_finite),
        "augmentation_finite_and_identity_preserved": bool(
            augmentation_finite and augmented.agent_ids == batch.agent_ids),
        "coordinate_roundtrip": "covered_by_exporter_scale_and_source metadata",
        "representation_reconstruction_max_abs_m": reconstruction_max_abs_m,
        "full_192_4_768_loss": float(loss.detach()),
        "full_192_4_768_backward_finite": bool(gradients_finite),
        "train_manifest_sha256": sha256(train_path),
        "formal_train_split_loaded": len(train) == 5720,
        "formal_train_scenes": len(train),
        "formal_train_contains_nexus_2": any(
            item["sequence"] == "nexus_2"
            for item in json.loads(train_path.read_text())["source_files"]),
        "status": "PASS_DATA_AND_CPU_INTERFACE" if (
            len(train) == 5720 and padded_finite and augmentation_finite and gradients_finite
            and reconstruction_max_abs_m <= 1e-4) else "FAIL",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Recover and verify official six-class Y-Net SDD prediction masks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def generate(reference: Path, model, preprocess) -> np.ndarray:
    image = cv2.imread(str(reference))
    if image is None:
        raise FileNotFoundError(reference)
    height, width = image.shape[:2]
    target_h, target_w = math.ceil(height / 32) * 32, math.ceil(width / 32) * 32
    padded = cv2.copyMakeBorder(
        image, 0, target_h - height, 0, target_w - width,
        cv2.BORDER_CONSTANT, value=0)
    value = preprocess(padded)
    tensor = torch.from_numpy(value.transpose(2, 0, 1)).float().unsqueeze(0)
    with torch.no_grad():
        scores = model(tensor)
        labels = scores.argmax(dim=1)[0, :height, :width]
    result = labels.cpu().numpy().astype(np.uint8)
    if result.shape != (height, width):
        raise RuntimeError("semantic output shape differs from reference image")
    if not set(np.unique(result)).issubset(set(range(6))):
        raise RuntimeError("semantic output is not a six-class label image")
    return result


def atomic_png(path: Path, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    if not cv2.imwrite(str(temporary), labels):
        raise RuntimeError(f"failed to write {temporary}")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--target-reference", required=True)
    parser.add_argument("--target-output", required=True)
    parser.add_argument("--control", action="append", default=[],
                        help="NAME:REFERENCE:EXISTING_PRED_MASK")
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--require-control-exact", action="store_true")
    args = parser.parse_args()
    import segmentation_models_pytorch as smp

    model_path = Path(args.model).resolve()
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()
    preprocess = smp.encoders.get_preprocessing_fn("resnet101", "imagenet")
    controls = []
    for specification in args.control:
        name, reference, expected = specification.split(":", 2)
        expected_array = cv2.imread(expected, cv2.IMREAD_UNCHANGED)
        if expected_array is None or expected_array.ndim != 2:
            raise RuntimeError(f"invalid control mask: {expected}")
        actual = generate(Path(reference), model, preprocess)
        exact = bool(np.array_equal(actual, expected_array))
        controls.append({
            "name": name, "reference": str(Path(reference).resolve()),
            "expected": str(Path(expected).resolve()),
            "expected_sha256": sha256(Path(expected)),
            "shape": list(actual.shape), "exact_label_array_equal": exact,
            "differing_pixels": int(np.count_nonzero(actual != expected_array)),
        })
    if len(controls) < 3:
        raise SystemExit("at least three control scenes are required")
    controls_exact = all(item["exact_label_array_equal"] for item in controls)
    if args.require_control_exact and not controls_exact:
        atomic_json(Path(args.receipt), {
            "schema": "ebjd-sdd-map-recovery-v1", "status": "FAIL_CONTROL",
            "model": str(model_path), "model_sha256": sha256(model_path),
            "controls": controls})
        raise SystemExit("official semantic-map control replay was not exact")
    labels = generate(Path(args.target_reference), model, preprocess)
    output = Path(args.target_output)
    atomic_png(output, labels)
    receipt = {
        "schema": "ebjd-sdd-map-recovery-v1",
        "status": "PASS" if controls_exact else "PASS_OFFICIAL_MODEL_WITH_LEGACY_CONTROL_MISMATCH",
        "model": str(model_path), "model_sha256": sha256(model_path),
        "target_reference": str(Path(args.target_reference).resolve()),
        "target_reference_sha256": sha256(Path(args.target_reference)),
        "target_output": str(output.resolve()), "target_sha256": sha256(output),
        "target_shape": list(labels.shape),
        "target_classes": [int(value) for value in np.unique(labels)],
        "controls": controls,
        "controls_exact": controls_exact,
        "control_interpretation": (
            "legacy pred_mask arrays are not outputs of the bound official model"
            if not controls_exact else "exact replay"),
        "manual_mask_used_as_prediction": False,
    }
    atomic_json(Path(args.receipt), receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()

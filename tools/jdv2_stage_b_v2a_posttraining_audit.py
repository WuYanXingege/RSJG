#!/usr/bin/env python3
"""Read-only post-training audit for the selected JDV2 Stage-B V2-A model."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch

from tools.audit_jdv2_stage_a import _active_evaluator
from tools.jdv2_stage_b_v1_failure_mechanism_audit import run_gradient_audit
from tools.jdv2_stage_b_v1_posttraining_audit import (
    _checkpoint_audit,
    _evaluate,
)
from tools.jdv2_stage_b_v2a_preflight import run_v1_and_counterfactual


STAGE_A_SHA256 = (
    "699336d49aaecbccfaac8b44f00c0df5a73b521548fc29d3da37d071148fd5bb")
STAGE_B_V1_SHA256 = (
    "e24a1cbfc5760db57bcb5cacd64c66a4afd1aed0f11bb1efefa8549933d21c1a")
DEPENDENCY_SOURCE_SHA256 = (
    "0ff28eb7107e49a0ad6f2bf339a65d8847f83c4a7cf315faa5a80cac4dc00522")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage-a-checkpoint", required=True)
    parser.add_argument("--stage-b-v1-checkpoint", required=True)
    parser.add_argument("--dependency-source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=[2035, 2036, 2037, 2038, 2039])
    parser.add_argument("--gradient-windows", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    stage_a = Path(args.stage_a_checkpoint).resolve()
    stage_b_v1 = Path(args.stage_b_v1_checkpoint).resolve()
    dependency_source = Path(args.dependency_source).resolve()
    output = Path(args.output).resolve()

    protected_before = {
        "stage_a": sha256(stage_a),
        "stage_b_v1": sha256(stage_b_v1),
        "dependency_corrector_source": sha256(dependency_source),
    }
    expected = {
        "stage_a": STAGE_A_SHA256,
        "stage_b_v1": STAGE_B_V1_SHA256,
        "dependency_corrector_source": DEPENDENCY_SOURCE_SHA256,
    }
    if protected_before != expected:
        raise RuntimeError(
            f"immutable artifact mismatch: {protected_before!r} != {expected!r}")

    evaluator, epoch = _active_evaluator(
        args.config, str(checkpoint), output.parent / "runtime", args.device)
    net = evaluator.net
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = {
        name: payload.get(name) for name in (
            "epoch", "training_stage", "stage_b_architecture_version",
            "stage_b_residual_projection", "stage_a_parent_checkpoint_sha256",
            "stage_a_freeze_manifest_sha256", "stage_a_freeze_source_commit",
            "best_selection", "best_metrics", "best_metrics_epochs",
            "stage_optimizer_steps_completed", "stage_progress")
    }
    if metadata["stage_b_architecture_version"] != "jdv2-stage-b-v2a":
        raise RuntimeError("best checkpoint is not jdv2-stage-b-v2a")
    if metadata["stage_b_residual_projection"] != "component_zero_mean":
        raise RuntimeError("best checkpoint lacks component_zero_mean provenance")
    if metadata["stage_a_parent_checkpoint_sha256"] != STAGE_A_SHA256:
        raise RuntimeError("best checkpoint has the wrong Stage-A parent")

    projection_parameter_names = [
        name for name, _ in net.named_parameters()
        if name.startswith("jdv2_residual_projection.")]
    result = {
        "status": "RUNNING",
        "protocol": {
            "split": "ETH validation",
            "windows": 139,
            "seeds": list(args.seeds),
            "checkpoint_epoch": int(epoch),
            "read_only": True,
            "paired_seed_protocol": True,
        },
        "provenance": {
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True).strip(),
            "v2a_checkpoint": str(checkpoint),
            "v2a_checkpoint_sha256": sha256(checkpoint),
            "protected_hashes_before": protected_before,
            "checkpoint_metadata": metadata,
        },
        "checkpoint_audit": _checkpoint_audit(stage_a, checkpoint),
        "parameter_contract": {
            "trainable_corrector_parameters": sum(
                parameter.numel() for name, parameter in net.named_parameters()
                if name.startswith("jdv2_corrector.")),
            "projection_parameter_count": 0,
            "projection_parameter_names": projection_parameter_names,
        },
    }
    if projection_parameter_names:
        raise RuntimeError(
            "component_zero_mean projection unexpectedly owns parameters")

    if output.exists():
        previous = json.loads(output.read_text())
        previous_hash = previous.get("provenance", {}).get(
            "v2a_checkpoint_sha256")
        if previous_hash != result["provenance"]["v2a_checkpoint_sha256"]:
            raise RuntimeError("existing audit belongs to another checkpoint")
        if "inference_audit" in previous:
            result["inference_audit"] = previous["inference_audit"]
    output.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    persist()
    if "inference_audit" not in result:
        result["inference_audit"] = _evaluate(evaluator, args.seeds)
    persist()
    result["identity_audit"] = run_v1_and_counterfactual(
        evaluator, seed=int(args.seeds[0]))
    result["identity_audit"]["semantic_note"] = (
        "The reused low-level audit retains the historical key "
        "v1_production_vs_reference. In this V2-A post-training run the "
        "loaded corrector is the V2-A checkpoint, so that key means the "
        "current V2-A corrector with projection=none versus the matching "
        "same-weight reference arithmetic; it is not a V1 epoch-10 metric.")
    persist()
    result["gradient_audit"] = run_gradient_audit(
        evaluator, "V2A_BEST", window_limit=args.gradient_windows)

    protected_after = {
        "stage_a": sha256(stage_a),
        "stage_b_v1": sha256(stage_b_v1),
        "dependency_corrector_source": sha256(dependency_source),
    }
    if protected_after != protected_before:
        raise RuntimeError("read-only audit changed an immutable artifact")
    result["provenance"]["protected_hashes_after"] = protected_after
    result["status"] = "COMPLETE"
    persist()
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

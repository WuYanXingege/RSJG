#!/usr/bin/env python3
"""Compact read-only audit for the selected JDV2 Stage-B V1 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from src.jdv2_audit import AUDIT_METRICS
from src.metrics import compute_metric_mask
from src.utils import isolated_random_seed
from tools.audit_jdv2_stage_a import _active_evaluator


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(values):
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _metric_summary(runs):
    return {
        stratum: {
            metric: _summary([
                row[stratum][metric] for row in runs
                if stratum in row and metric in row[stratum]])
            for metric in AUDIT_METRICS
        }
        for stratum in ("overall", "E=0", "E>0")
    }


def _checkpoint_audit(stage_a_path: Path, stage_b_path: Path):
    stage_a = torch.load(stage_a_path, map_location="cpu", weights_only=False)
    stage_b = torch.load(stage_b_path, map_location="cpu", weights_only=False)
    state_a = stage_a["model_state_dict"]
    state_b = stage_b["model_state_dict"]
    if set(state_a) != set(state_b):
        raise RuntimeError("Stage-A/Stage-B state-dict key sets differ")
    non_corrector_changed = []
    corrector_changed = []
    squared_delta = 0.0
    max_absolute_delta = 0.0
    corrector_parameters = 0
    for name, before in state_a.items():
        after = state_b[name]
        is_corrector = name.startswith("jdv2_corrector.")
        if is_corrector:
            corrector_parameters += before.numel()
        if torch.equal(before, after):
            continue
        delta = after.float() - before.float()
        record = {
            "name": name,
            "l2": float(torch.linalg.vector_norm(delta)),
            "max_abs": float(delta.abs().max()),
        }
        if is_corrector:
            corrector_changed.append(record)
            squared_delta += float(delta.square().sum())
            max_absolute_delta = max(
                max_absolute_delta, float(delta.abs().max()))
        else:
            non_corrector_changed.append(record)
    metadata = {
        name: stage_b.get(name) for name in (
            "epoch", "training_stage", "stage_b_architecture_version",
            "stage_a_parent_checkpoint_sha256",
            "stage_a_freeze_manifest_sha256",
            "stage_a_freeze_source_commit", "best_selection",
            "stage_optimizer_steps_completed", "stage_progress")
    }
    return {
        "stage_a_checkpoint_sha256": _sha256(stage_a_path),
        "stage_b_checkpoint_sha256": _sha256(stage_b_path),
        "state_key_sets_equal": True,
        "corrector_parameter_count": int(corrector_parameters),
        "corrector_changed_tensor_count": len(corrector_changed),
        "corrector_delta_l2": math.sqrt(squared_delta),
        "corrector_delta_max_abs": max_absolute_delta,
        "non_corrector_changed_tensor_count": len(non_corrector_changed),
        "non_corrector_changed_tensors": non_corrector_changed,
        "only_corrector_changed": (
            not non_corrector_changed and bool(corrector_changed)),
        "checkpoint_metadata": metadata,
    }


class _ResidualTrace:
    def __init__(self, net):
        self.net = net
        self.edge_class = None
        self.base_rms = None
        self.rows = []
        self._diffnet_forward = None
        self._corrector_forward = None

    def set_window(self, edge_class):
        self.edge_class = edge_class

    def __enter__(self):
        self._diffnet_forward = self.net.diffnet.forward
        self._corrector_forward = self.net.jdv2_corrector.forward

        def diffnet_forward(*args, **kwargs):
            output = self._diffnet_forward(*args, **kwargs)
            self.base_rms = float(
                output.detach().float().square().mean().sqrt().cpu())
            return output

        def corrector_forward(*args, **kwargs):
            output = self._corrector_forward(*args, **kwargs)
            delta_rms = float(
                output.detach().float().square().mean().sqrt().cpu())
            base_rms = float(self.base_rms)
            edge_index = args[2]
            is_e0 = int(edge_index.shape[1]) == 0
            self.rows.append({
                "edge_class": self.edge_class,
                "delta_epsilon_rms": delta_rms,
                "epsilon_base_rms": base_rms,
                "delta_base_rms_ratio": (
                    delta_rms / max(base_rms, 1e-12)),
                "e0_exact_zero": bool(
                    not is_e0 or torch.equal(
                        output, torch.zeros_like(output))),
            })
            return output

        self.net.diffnet.forward = diffnet_forward
        self.net.jdv2_corrector.forward = corrector_forward
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.net.diffnet.forward = self._diffnet_forward
        self.net.jdv2_corrector.forward = self._corrector_forward


@torch.no_grad()
def _evaluate(evaluator, seeds):
    net = evaluator.net
    net.eval()
    metric_runs = []
    coverage = []
    runtimes = []
    window_counts = defaultdict(int)
    trace = _ResidualTrace(net)
    if evaluator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(evaluator.device)
    with trace:
        for seed in seeds:
            print(f"STAGE_B_POSTTRAIN seed={seed}", flush=True)
            metric_values = defaultdict(lambda: defaultdict(list))
            started_seed = time.perf_counter()
            with isolated_random_seed(
                    int(seed), use_cuda=evaluator.device.type == "cuda"):
                for window_index, (batch_data, batch_id) in enumerate(
                        evaluator.data_loaders["valid"]):
                    inputs, sequence = net.prepare_inputs(batch_data, batch_id)
                    mask = compute_metric_mask(sequence)
                    edge_count = int(
                        inputs["jdv2_cache"]["edge_index"].shape[1])
                    edge_class = "E>0" if edge_count else "E=0"
                    window_counts[(int(seed), edge_class)] += 1
                    trace.set_window(edge_class)
                    net.jdv2_sampler.set_sampling_context(
                        int(seed), int(window_index))
                    with evaluator._autocast_context():
                        prediction, auxiliary = net.forward(
                            inputs, if_test=True)
                    ids = auxiliary["joint_candidate_index"][
                        :, :net.args.num_samples].long()
                    unique = torch.tensor([
                        torch.unique(row).numel() for row in ids],
                        dtype=torch.float32)
                    coverage.extend(unique.tolist())
                    for metric in AUDIT_METRICS:
                        values = net.compute_model_metrics(
                            metric_name=metric, predictions=prediction,
                            metric_mask=mask,
                            all_aux_outputs=auxiliary, inputs=inputs,
                            obs_length=net.args.obs_length)
                        metric_values["overall"][metric].extend(values)
                        metric_values[edge_class][metric].extend(values)
                    del inputs, sequence, prediction, auxiliary, mask
            if evaluator.device.type == "cuda":
                torch.cuda.synchronize(evaluator.device)
            runtimes.append({
                "seed": int(seed),
                "seconds": float(time.perf_counter() - started_seed),
            })
            metric_runs.append({
                "seed": int(seed),
                **{
                    stratum: {
                        metric: float(np.mean(values))
                        for metric, values in metrics.items()}
                    for stratum, metrics in metric_values.items()
                },
            })
    residual = {}
    for stratum in ("overall", "E=0", "E>0"):
        rows = (trace.rows if stratum == "overall" else
                [row for row in trace.rows
                 if row["edge_class"] == stratum])
        residual[stratum] = {
            key: _summary([row[key] for row in rows])
            for key in (
                "delta_epsilon_rms", "epsilon_base_rms",
                "delta_base_rms_ratio")
        }
        residual[stratum]["e0_exact_zero_rate"] = (
            float(np.mean([row["e0_exact_zero"] for row in rows]))
            if rows else None)
    peak_allocated = 0
    peak_reserved = 0
    if evaluator.device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(evaluator.device))
        peak_reserved = int(torch.cuda.max_memory_reserved(evaluator.device))
    return {
        "metric_runs": metric_runs,
        "metric_summary": _metric_summary(metric_runs),
        "coverage": {
            "agent_count": len(coverage),
            "mean_unique": float(np.mean(coverage)),
            "minimum_unique": int(min(coverage)),
            "maximum_unique": int(max(coverage)),
            "coverage_20_of_20_rate": float(np.mean(
                np.asarray(coverage) == 20)),
        },
        "residual": residual,
        "runtime": {
            "per_seed": runtimes,
            "total_seconds": float(sum(row["seconds"] for row in runtimes)),
            "mean_seconds_per_seed": float(np.mean([
                row["seconds"] for row in runtimes])),
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
        },
        "window_counts": {
            str(seed): {
                edge_class: window_counts[(int(seed), edge_class)]
                for edge_class in ("E=0", "E>0")}
            for seed in seeds
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stage-a-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[2035, 2036, 2037, 2038, 2039])
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    stage_a = Path(args.stage_a_checkpoint).resolve()
    output = Path(args.output).resolve()
    evaluator, epoch = _active_evaluator(
        args.config, str(checkpoint), output.parent / "evaluation", args.device)
    result = {
        "protocol": {
            "split": "ETH validation",
            "windows": 139,
            "seeds": args.seeds,
            "checkpoint_epoch": int(epoch),
            "read_only": True,
        },
        "checkpoint_audit": _checkpoint_audit(stage_a, checkpoint),
        "inference_audit": _evaluate(evaluator, args.seeds),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

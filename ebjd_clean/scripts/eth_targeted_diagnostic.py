#!/usr/bin/env python3
"""Reproducible CPU-first diagnostics for the formal EBJD ETH run.

The ``cpu`` command never loads a model checkpoint.  It computes constant-
velocity and representation statistics on the bound scene manifests, performs
same-tensor evaluator parity on the real ETH windows, snapshots complete
training records, and renders the observed training-stage figure.  The
``banks`` command consumes portable frozen prediction-bank manifests when GPU
exports become available; it does not perform inference or optimization.  The
``export`` command invokes a SHA-bound frozen exporter and refuses to run
unless GPU idleness has been explicitly checked.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from ebjd.diagnostic_bank import (
    SeedBank, assert_observation_parity, atomic_json, hard_scene_statistics,
    reorder_like, sha256_file, stable_seed,
)
from ebjd.metrics import hard_metric_statistics, per_world_errors


KS = (1, 4, 8, 12, 16, 20)
METRICS = ("minADE", "minFDE", "JADE", "JFDE")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_manifest_scenes(manifest_path: str | Path):
    """Yield only trajectory/identity members; semantic arrays are not read."""
    source = Path(manifest_path).resolve()
    manifest = read_json(source)
    if manifest.get("schema") != "ebjd-scene-manifest-v1":
        raise ValueError("unsupported scene manifest")
    for descriptor in manifest["shards"]:
        shard = source.parent / descriptor["path"]
        if sha256_file(shard) != descriptor["sha256"]:
            raise ValueError(f"scene shard hash mismatch: {shard}")
        with np.load(shard, allow_pickle=True) as payload:
            names = set(payload.files)
            required = {
                "observed", "future", "scene_ids", "agent_ids",
                "frame_ids", "timestamps", "source_sequences", "window_keys",
            }
            if not required.issubset(names):
                raise ValueError(f"scene shard fields missing: {required - names}")
            for index in range(len(payload["observed"])):
                yield {
                    "observed": np.asarray(payload["observed"][index], dtype=np.float64),
                    "future": np.asarray(payload["future"][index], dtype=np.float64),
                    "scene_id": str(payload["scene_ids"][index]),
                    "agent_ids": np.asarray(payload["agent_ids"][index]).astype(str),
                    "frame_ids": np.asarray(payload["frame_ids"][index], dtype=np.int64),
                    "timestamps": np.asarray(payload["timestamps"][index], dtype=np.float64),
                    "source_sequence": str(payload["source_sequences"][index]),
                    "window_key": str(payload["window_keys"][index]),
                }


def cv_prediction(observed: np.ndarray, dt: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
    velocity = (
        (observed[:, -1] - observed[:, -2]) / dt
        + (observed[:, -2] - observed[:, -3]) / dt
    ) / 2.0
    horizon = np.arange(1, 13, dtype=np.float64)[None, :, None] * dt
    return observed[:, -1:, :] + horizon * velocity[:, None, :], velocity


def unscaled_latent(future: np.ndarray, baseline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fraction = np.arange(1, 12, dtype=np.float64) / 12.0
    covariance = np.minimum(fraction[:, None], fraction[None, :]) - np.outer(
        fraction, fraction)
    cholesky = np.linalg.cholesky(covariance)
    goal = future[:, -1] - baseline[:, -1]
    residual = future[:, :11] - baseline[:, :11] - fraction[None, :, None] * goal[:, None]
    flattened = residual.transpose(1, 0, 2).reshape(11, -1)
    bridge = np.linalg.solve(cholesky, flattened).reshape(
        11, future.shape[0], 2).transpose(1, 0, 2)
    return goal, bridge


def quantiles(values: Iterable[float]) -> dict[str, float | int]:
    value = np.asarray(list(values), dtype=np.float64)
    if value.size == 0:
        return {"count": 0, "mean": math.nan, "median": math.nan,
                "p90": math.nan, "p95": math.nan, "max": math.nan}
    return {
        "count": int(value.size), "mean": float(value.mean()),
        "median": float(np.quantile(value, 0.50)),
        "p90": float(np.quantile(value, 0.90)),
        "p95": float(np.quantile(value, 0.95)), "max": float(value.max()),
    }


class MetricGroup:
    def __init__(self) -> None:
        self.agent_ade: list[float] = []
        self.agent_fde: list[float] = []
        self.scene_marginal_ade: list[float] = []
        self.scene_marginal_fde: list[float] = []
        self.jade: list[float] = []
        self.jfde: list[float] = []

    def add(self, predictions: np.ndarray, future: np.ndarray) -> None:
        item = hard_scene_statistics(predictions, future)
        self.agent_ade.extend(np.asarray(item["agent_minade"]).tolist())
        self.agent_fde.extend(np.asarray(item["agent_minfde"]).tolist())
        self.scene_marginal_ade.append(float(item["scene_marginal_ade"]))
        self.scene_marginal_fde.append(float(item["scene_marginal_fde"]))
        self.jade.append(float(item["JADE"]))
        self.jfde.append(float(item["JFDE"]))

    def result(self) -> dict[str, float | int]:
        if not self.jade:
            raise ValueError("empty metric group")
        result = {
            "scene_count": len(self.jade), "agent_occurrence_count": len(self.agent_ade),
            "minADE": float(np.mean(self.agent_ade)),
            "minFDE": float(np.mean(self.agent_fde)),
            "JADE": float(np.mean(self.jade)), "JFDE": float(np.mean(self.jfde)),
            "scene_weighted_minADE": float(np.mean(self.scene_marginal_ade)),
            "scene_weighted_minFDE": float(np.mean(self.scene_marginal_fde)),
        }
        result["coordination_gap_ADE"] = float(
            result["JADE"] - result["scene_weighted_minADE"])
        result["coordination_gap_FDE"] = float(
            result["JFDE"] - result["scene_weighted_minFDE"])
        return result


def group_labels(count: int) -> tuple[str, ...]:
    labels = ["overall", "N=1" if count == 1 else "N>=2"]
    if count == 2:
        labels.append("N=2")
    if count >= 3:
        labels.append("N>=3")
    return tuple(labels)


def load_gdts_metrics(path: str | Path):
    spec = importlib.util.spec_from_file_location("targeted_gdts_metrics", Path(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load GDTS metric implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluator_parity(scenes: list[dict], gdts_metric_path: str | Path) -> dict:
    gdts = load_gdts_metrics(gdts_metric_path)
    maximum = defaultdict(float)
    maximum_relative = defaultdict(float)
    first_failure = None
    real_n1_gap = 0.0
    for scene_index, scene in enumerate(scenes):
        observed, future = scene["observed"], scene["future"]
        cv, _ = cv_prediction(observed)
        predictions = np.repeat(cv[None], 20, axis=0).astype(np.float32)
        target = future.astype(np.float32)
        ebjd_pred = torch.from_numpy(predictions)[None]
        ebjd_target = torch.from_numpy(target)[None]
        valid = torch.ones(1, target.shape[0], dtype=torch.bool)
        stats = hard_metric_statistics(ebjd_pred, ebjd_target, valid)
        ebjd_ade, ebjd_fde = per_world_errors(ebjd_pred, ebjd_target, valid)

        full_prediction = torch.cat((
            torch.from_numpy(observed.astype(np.float32))[None].expand(20, -1, -1, -1),
            torch.from_numpy(predictions),
        ), dim=2).permute(0, 2, 1, 3)
        full_target = torch.from_numpy(np.concatenate((observed, future), axis=1).astype(
            np.float32)).permute(1, 0, 2)
        mask = torch.ones(target.shape[0], dtype=torch.bool)
        gdts_agent_ade = torch.tensor(gdts.minADE_at_K(
            full_prediction, full_target, mask, obs_length=8))
        gdts_agent_fde = torch.tensor(gdts.minFDE_at_K(
            full_prediction, full_target, mask, obs_length=8))
        gdts_jade = torch.tensor(gdts.JADE(
            full_prediction, full_target, mask, obs_length=8))
        gdts_jfde = torch.tensor(gdts.JFDE(
            full_prediction, full_target, mask, obs_length=8))
        gdts_world_distance = torch.linalg.vector_norm(
            full_prediction[:, 8:] - full_target[None, 8:], dim=-1)
        gdts_world_ade = gdts_world_distance.mean(1).T[None]
        gdts_world_fde = gdts_world_distance[:, -1].T[None]

        comparisons = {
            "per_world_ADE": (ebjd_ade, gdts_world_ade),
            "per_world_FDE": (ebjd_fde, gdts_world_fde),
            "agent_minADE": (stats.agent_minade[0], gdts_agent_ade),
            "agent_minFDE": (stats.agent_minfde[0], gdts_agent_fde),
            "scene_marginal_ADE": (
                stats.scene_marginal_ade, gdts_agent_ade.mean()[None]),
            "scene_marginal_FDE": (
                stats.scene_marginal_fde, gdts_agent_fde.mean()[None]),
            "JADE": (stats.scene_jade, gdts_jade),
            "JFDE": (stats.scene_jfde, gdts_jfde),
        }
        for name, (left, right) in comparisons.items():
            error = float(torch.max(torch.abs(left.float() - right.float())))
            relative = float(torch.max(
                torch.abs(left.float() - right.float())
                / torch.clamp(torch.abs(right.float()), min=1e-12)))
            maximum[name] = max(maximum[name], error)
            maximum_relative[name] = max(maximum_relative[name], relative)
            if first_failure is None and not torch.allclose(
                    left.float(), right.float(), atol=1e-6, rtol=1e-5):
                first_failure = {
                    "scene_index": scene_index, "window_key": scene["window_key"],
                    "quantity": name, "max_abs_m": error,
                }
        if target.shape[0] == 1:
            real_n1_gap = max(
                real_n1_gap,
                abs(float(stats.scene_jade[0] - stats.scene_marginal_ade[0])),
                abs(float(stats.scene_jfde[0] - stats.scene_marginal_fde[0])))

    # Non-degenerate minimal example for permutation and padding invariants.
    target = torch.zeros(1, 2, 12, 2)
    prediction = torch.zeros(1, 2, 2, 12, 2)
    prediction[0, 0, 1, :, 0] = 10.0
    prediction[0, 1, 0, :, 0] = 10.0
    valid = torch.ones(1, 2, dtype=torch.bool)
    base = hard_metric_statistics(prediction, target, valid)
    common = hard_metric_statistics(prediction[:, [1, 0]], target, valid)
    synchronized_agents = hard_metric_statistics(
        prediction[:, :, [1, 0]], target[:, [1, 0]], valid[:, [1, 0]])
    agent = prediction.clone()
    agent[0, :, 1] = agent[0, [1, 0], 1]
    independent = hard_metric_statistics(agent, target, valid)
    padded_prediction = torch.cat((prediction, torch.full_like(prediction[:, :, :1], 999.0)), 2)
    padded_target = torch.cat((target, torch.full_like(target[:, :1], -999.0)), 1)
    padded_valid = torch.tensor([[True, True, False]])
    padded = hard_metric_statistics(padded_prediction, padded_target, padded_valid)
    synthetic = {
        "common_world_permutation_max_abs_m": max(
            abs(float(base.scene_jade[0] - common.scene_jade[0])),
            abs(float(base.scene_jfde[0] - common.scene_jfde[0]))),
        "agent_axis_synchronous_permutation_max_abs_m": max(
            float(torch.max(torch.abs(
                torch.sort(base.agent_minade[0]).values
                - torch.sort(synchronized_agents.agent_minade[0]).values))),
            float(torch.max(torch.abs(
                torch.sort(base.agent_minfde[0]).values
                - torch.sort(synchronized_agents.agent_minfde[0]).values))),
            abs(float(base.scene_jade[0] - synchronized_agents.scene_jade[0])),
            abs(float(base.scene_jfde[0] - synchronized_agents.scene_jfde[0]))),
        "independent_world_marginal_max_abs_m": max(
            float(torch.max(torch.abs(base.agent_minade - independent.agent_minade))),
            float(torch.max(torch.abs(base.agent_minfde - independent.agent_minfde)))),
        "independent_world_joint_change_ADE_m": float(
            independent.scene_jade[0] - base.scene_jade[0]),
        "independent_world_joint_change_FDE_m": float(
            independent.scene_jfde[0] - base.scene_jfde[0]),
        "padding_valid_agent_max_abs_m": max(
            float(torch.max(torch.abs(base.agent_minade - padded.agent_minade[:, :2]))),
            float(torch.max(torch.abs(base.agent_minfde - padded.agent_minfde[:, :2]))),
            abs(float(base.scene_jade[0] - padded.scene_jade[0])),
            abs(float(base.scene_jfde[0] - padded.scene_jfde[0]))),
    }
    return {
        "status": "PASSED" if first_failure is None else "FAILED",
        "atol_m": 1e-6, "rtol": 1e-5,
        "real_manifest_scene_count": len(scenes),
        "maximum_absolute_error_m": dict(maximum),
        "maximum_relative_error": dict(maximum_relative),
        "first_failure": first_failure,
        "shape_contract": {
            "ebjd_prediction": "[B=1,P=20,N,T=12,xy=2]",
            "gdts_prediction": "[P=20,T=20,N,xy=2] with observed prefix",
            "valid_mask": "[N] all true for unpadded real-scene parity",
        },
        "aggregation_weights": {
            "marginal": "agent occurrences",
            "joint": "scene windows",
        },
        "real_N1_joint_marginal_max_abs_m": real_n1_gap,
        "synthetic_invariants": synthetic,
    }


def manifest_statistics(manifest_path: str | Path) -> tuple[dict, list[dict], list[dict]]:
    started = time.monotonic()
    rows: list[dict] = []
    frame_rows: list[dict] = []
    goal_sq = bridge_sq = 0.0
    goal_count = bridge_count = 0
    scenes = list(iter_manifest_scenes(manifest_path))
    groups: dict[str, MetricGroup] = defaultdict(MetricGroup)
    for scene in scenes:
        observed, future = scene["observed"], scene["future"]
        baseline, velocity = cv_prediction(observed)
        goal, bridge = unscaled_latent(future, baseline)
        goal_sq += float(np.square(goal).sum()); goal_count += goal.size
        bridge_sq += float(np.square(bridge).sum()); bridge_count += bridge.size
        cv = baseline[None].astype(np.float32)
        for label in group_labels(observed.shape[0]):
            groups[label].add(cv, future.astype(np.float32))
        per_agent_bridge = np.sqrt(np.mean(np.square(bridge), axis=(1, 2)))
        for agent in range(observed.shape[0]):
            rows.append({
                "window_key": scene["window_key"], "source": scene["source_sequence"],
                "N": observed.shape[0], "group": "N=1" if observed.shape[0] == 1 else "N>=2",
                "agent_id": str(scene["agent_ids"][agent]),
                "observed_speed_mps": float(np.linalg.norm(velocity[agent])),
                "future_displacement_m": float(np.linalg.norm(future[agent, -1] - observed[agent, -1])),
                "cv_endpoint_residual_m": float(np.linalg.norm(goal[agent])),
                "bridge_unscaled_agent_rms_m": float(per_agent_bridge[agent]),
            })
        frame_diff = np.diff(scene["frame_ids"])
        time_diff = np.diff(scene["timestamps"])
        frame_rows.append({
            "window_key": scene["window_key"], "source": scene["source_sequence"],
            "frame_step_min": int(frame_diff.min()), "frame_step_max": int(frame_diff.max()),
            "timestamp_step_min_s": float(time_diff.min()),
            "timestamp_step_max_s": float(time_diff.max()),
        })
    scalar_fields = (
        "observed_speed_mps", "future_displacement_m",
        "cv_endpoint_residual_m", "bridge_unscaled_agent_rms_m",
    )
    distribution = {}
    for label in ("overall", "N=1", "N>=2"):
        selected = rows if label == "overall" else [x for x in rows if x["group"] == label]
        distribution[label] = {
            field: quantiles(float(item[field]) for item in selected)
            for field in scalar_fields}
    result = {
        "manifest": str(Path(manifest_path).resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "scene_count": len(scenes), "agent_occurrence_count": len(rows),
        "goal_component_rms_m": math.sqrt(goal_sq / goal_count),
        "bridge_component_rms_m": math.sqrt(bridge_sq / bridge_count),
        "distributions": distribution,
        "cv_metrics": {label: group.result() for label, group in groups.items()},
        "elapsed_cpu_seconds": time.monotonic() - started,
    }
    return result, rows, frame_rows


def read_jsonl_boundary(path: str | Path) -> tuple[list[dict], dict]:
    source = Path(path).resolve()
    boundary = source.stat().st_size
    records = []
    with source.open("rb") as handle:
        raw = handle.read(boundary)
    complete = raw[:raw.rfind(b"\n") + 1] if b"\n" in raw else b""
    for number, line in enumerate(complete.splitlines(), 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at complete line {number}") from error
    return records, {
        "path": str(source), "byte_boundary": boundary,
        "complete_bytes": len(complete), "complete_records": len(records),
        "sha256_complete_prefix": hashlib.sha256(complete).hexdigest(),
        "mtime_ns_at_boundary": source.stat().st_mtime_ns,
    }


def training_trends(train_log: str | Path, epoch_summary: str | Path) -> tuple[list[dict], dict]:
    updates, update_boundary = read_jsonl_boundary(train_log)
    epochs, epoch_boundary = read_jsonl_boundary(epoch_summary)
    per_epoch: dict[int, list[dict]] = defaultdict(list)
    for record in updates:
        if isinstance(record.get("epoch"), int):
            per_epoch[int(record["epoch"])].append(record)
    epoch_by_id = {int(item["epoch"]): item for item in epochs if "validation" in item}
    fields = (
        "loss", "diffusion", "rollout", "geometry", "map",
        "learning_rate", "unet_learning_rate", "active_rollout_weight",
        "effective_rollout_coefficient", "total_gradient_norm_before_clip",
        "parameter_norm_after", "step_wall_seconds", "projection_active",
        "rollout_update",
    )
    rows = []
    for epoch in sorted(epoch_by_id):
        records = per_epoch.get(epoch, [])
        summary = epoch_by_id[epoch]
        row: dict[str, object] = {
            "epoch": epoch, "updates": summary.get("updates", "missing"),
            "end_update_inclusive": summary.get("end_update_inclusive", "missing"),
            "wall_seconds": summary.get("wall_seconds", "missing"),
            "normal_step_seconds": summary.get("normal_step_seconds", "missing"),
            "rollout_step_seconds": summary.get("rollout_step_seconds", "missing"),
            "rollout_updates": summary.get("rollout_updates", "missing"),
            "projection_active_updates": summary.get("projection_active_updates", "missing"),
        }
        for field in fields:
            values = [float(x[field]) for x in records
                      if isinstance(x.get(field), (int, float)) and math.isfinite(float(x[field]))]
            row[f"mean_{field}"] = float(np.mean(values)) if values else "missing"
        for name, value in summary["validation"].items():
            if isinstance(value, (int, float)):
                row[f"validation_{name}"] = value
        rows.append(row)
    result = {
        "completed_epoch_count": len(rows),
        "latest_complete_epoch": max(epoch_by_id) if epoch_by_id else None,
        "active_epoch_at_train_log_boundary": max(per_epoch) if per_epoch else None,
        "update_log_boundary": update_boundary, "epoch_log_boundary": epoch_boundary,
        "rollout_start_epoch": 21, "rollout_ramp_end_epoch": 40,
    }
    return rows, result


def plot_training(rows: list[dict], output_stem: Path, qa_scripts: Path | None) -> dict:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7, "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": 0.8, "legend.frameon": False,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    epoch = np.asarray([int(row["epoch"]) for row in rows])
    baseline = {"minADE": .41705285937904, "minFDE": .5800323037873278,
                "JADE": .4899067671402641, "JFDE": .7434117917354282}
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), constrained_layout=True)
    colors = {"ADE": "#3B6FB6", "FDE": "#D97757"}
    panels = (
        (axes[0, 0], ("minADE", "minFDE"), "Marginal error (m)"),
        (axes[0, 1], ("JADE", "JFDE"), "Joint error (m)"),
    )
    for axis, names, ylabel in panels:
        for name, family in zip(names, ("ADE", "FDE"), strict=True):
            y = np.asarray([float(row[f"validation_{name}"]) for row in rows])
            axis.plot(epoch, y, marker="o", ms=2.4, lw=1.1, color=colors[family], label=name)
            axis.axhline(baseline[name], color=colors[family], ls=":", lw=0.8,
                         label=f"GDTS {name}")
        axis.set(xlabel="Epoch", ylabel=ylabel)
        axis.grid(axis="y", lw=0.4, alpha=0.3)
        axis.legend(fontsize=5.8, ncol=2)
    axes[1, 0].plot(epoch, [float(row["mean_diffusion"]) for row in rows],
                    color="#3B6FB6", lw=1.1, label="Diffusion")
    rollout = [float(row["mean_rollout"]) if row["mean_rollout"] != "missing" else math.nan
               for row in rows]
    axes[1, 0].plot(epoch, rollout, color="#D97757", lw=1.1, label="Rollout (logged mean)")
    axes[1, 0].set(xlabel="Epoch", ylabel="Training loss component")
    axes[1, 0].legend(fontsize=6)
    wall = np.asarray([float(row["wall_seconds"]) / 60 for row in rows])
    weight = np.asarray([
        float(row["mean_active_rollout_weight"])
        if row["mean_active_rollout_weight"] != "missing" else 0.0 for row in rows])
    axes[1, 1].plot(epoch, wall, color="#666666", lw=1.1, label="Wall time")
    twin = axes[1, 1].twinx()
    twin.plot(epoch, weight, color="#7B61A8", lw=1.1, label="Rollout weight")
    axes[1, 1].set(xlabel="Epoch", ylabel="Wall time (min)")
    twin.set_ylabel("Rollout weight", color="#7B61A8")
    axes[1, 1].text(0.03, 0.94, "Wall time", transform=axes[1, 1].transAxes,
                    color="#555555", fontsize=6, va="top")
    axes[1, 1].text(0.03, 0.86, "Rollout weight", transform=axes[1, 1].transAxes,
                    color="#7B61A8", fontsize=6, va="top")
    for label, axis in zip("abcd", axes.flat, strict=True):
        axis.axvline(21, color="#222222", ls="--", lw=0.7)
        if epoch.max() >= 40:
            axis.axvline(40, color="#222222", ls="--", lw=0.7)
        axis.text(-0.12, 1.06, label, transform=axis.transAxes,
                  fontsize=8, fontweight="bold", va="top")
    axes[1, 0].grid(axis="y", lw=0.4, alpha=0.3)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    alignment = "not_run"
    if qa_scripts is not None:
        import sys
        sys.path.insert(0, str(qa_scripts))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(
            fig, json_out=str(output_stem) + ".alignment-layout.json",
            overlay_svg=str(output_stem) + ".alignment-overlay.svg",
            tolerance_pt=1.5, gutter_tolerance_pt=1.5,
            require_panel_labels=True, exclude_axes=[twin], strict=True)
        alignment = "passed"
    fig.savefig(str(output_stem) + ".svg", bbox_inches="tight")
    fig.savefig(str(output_stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(output_stem) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(output_stem) + ".tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)
    return {
        "core_conclusion": "The ETH gap predates rollout and remains through the latest completed epoch.",
        "archetype": "quantitative grid", "backend": "Python/matplotlib",
        "replicate_unit": "one training seed; no uncertainty interval",
        "alignment_gate": alignment,
    }


def bank_metrics(bank: SeedBank) -> tuple[dict, dict[str, dict]]:
    groups: dict[str, MetricGroup] = defaultdict(MetricGroup)
    scenes = {}
    for identity in bank.identities():
        scene = bank.load(identity)
        scenes[identity] = scene
        for label in group_labels(scene.agents):
            groups[label].add(scene.predictions, scene.future)
    return {label: item.result() for label, item in groups.items()}, scenes


def endpoint_summary(scenes: dict[str, object], scale: float) -> tuple[list[dict], dict]:
    rows = []
    for identity, scene in scenes.items():
        endpoint = scene.predictions[:, :, -1].astype(np.float64)
        truth = scene.future[:, -1].astype(np.float64)
        center = endpoint.mean(0)
        radius = np.linalg.norm(endpoint - center[None], axis=-1)
        pairwise = np.linalg.norm(endpoint[:, None] - endpoint[None, :], axis=-1)
        upper = np.triu_indices(endpoint.shape[0], 1)
        cv, _ = cv_prediction(scene.history.astype(np.float64))
        cv_residual = np.linalg.norm(cv[:, -1] - truth, axis=-1)
        for agent in range(scene.agents):
            nearest = float(np.linalg.norm(endpoint[:, agent] - truth[agent], axis=-1).min())
            rows.append({
                "window_key": identity, "agent_id": str(scene.agent_ids[agent]),
                "N": scene.agents, "group": "N=1" if scene.agents == 1 else "N>=2",
                "center_offset_m": float(np.linalg.norm(center[agent] - truth[agent])),
                "mean_radius_m": float(radius[:, agent].mean()),
                "rms_radius_m": float(np.sqrt(np.square(radius[:, agent]).mean())),
                "mean_pairwise_endpoint_m": float(pairwise[:, :, agent][upper].mean()),
                "nearest_of_20_endpoint_m": nearest,
                "cv_endpoint_residual_m": float(cv_residual[agent]),
                "center_offset_train_scale": float(np.linalg.norm(center[agent] - truth[agent]) / scale),
                "rms_radius_train_scale": float(np.sqrt(np.square(radius[:, agent]).mean()) / scale),
            })
    fields = (
        "center_offset_m", "mean_radius_m", "rms_radius_m",
        "mean_pairwise_endpoint_m", "nearest_of_20_endpoint_m",
        "center_offset_train_scale", "rms_radius_train_scale",
        "cv_endpoint_residual_m",
    )
    return rows, {label: {field: quantiles(
        row[field] for row in rows if label == "overall" or row["group"] == label)
        for field in fields} for label in ("overall", "N=1", "N>=2")}


def k_curve(scenes: dict[str, object], model: str) -> tuple[list[dict], float]:
    rows, previous = [], None
    violation = 0.0
    permutations = {
        identity: np.random.default_rng(stable_seed("eth-targeted-k", identity)).permutation(20)
        for identity in scenes}
    for k in KS:
        group = MetricGroup()
        for identity, scene in scenes.items():
            subset = permutations[identity][:k]
            group.add(scene.predictions[subset], scene.future)
        result = group.result()
        rows.append({"model": model, "K": k, **result,
                     "subset_rule": "fixed scene-common seeded permutation nested prefix"})
        if previous is not None:
            for metric in METRICS:
                violation = max(violation, float(result[metric] - previous[metric]))
        previous = result
    return rows, violation


def pearson(values_a: list[float], values_b: list[float]) -> float | None:
    a, b = np.asarray(values_a, dtype=np.float64), np.asarray(values_b, dtype=np.float64)
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def select_cases(rows: list[dict], model: str) -> list[dict]:
    candidates = [row for row in rows if row["model"] == model]
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda row: row["delta_minFDE_m"])
    values = np.asarray([row["delta_minFDE_m"] for row in ordered])
    selected = [
        ("maximum_negative_delta_minFDE", ordered[0]),
        ("maximum_positive_delta_minFDE", ordered[-1]),
    ]
    for quantile, label in ((.50, "nearest_median_delta_minFDE"),
                            (.90, "nearest_p90_delta_minFDE")):
        target = float(np.quantile(values, quantile))
        selected.append((label, min(
            ordered, key=lambda row: abs(row["delta_minFDE_m"] - target))))
    rng = np.random.default_rng(stable_seed("eth-targeted-case", model))
    selected.append(("fixed_seed_random", ordered[int(rng.integers(len(ordered)))]))
    unique, result = set(), []
    for rule, row in selected:
        key = (row["window_key"], row["agent_id"])
        if key in unique:
            continue
        unique.add(key)
        result.append({"selection_rule": rule, **row})
    return result


def render_case_plots(
    cases: list[dict], scenes_by_model: dict[str, dict[str, object]], output: Path,
) -> list[str]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7, "axes.spines.right": False, "axes.spines.top": False,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for index, case in enumerate(cases):
        model = case["model"]; identity = case["window_key"]
        reference = scenes_by_model["GDTS"][identity]
        candidate = reorder_like(reference, scenes_by_model[model][identity])
        agent = [str(value) for value in reference.agent_ids].index(case["agent_id"])
        cv, _ = cv_prediction(reference.history.astype(np.float64))
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), constrained_layout=True)
        for panel, (label, scene) in zip(
                axes, (("GDTS", reference), (model, candidate)), strict=True):
            for world in scene.predictions[:, agent]:
                panel.plot(world[:, 0], world[:, 1], color="#7B61A8", alpha=.22, lw=.65)
                panel.scatter(world[-1, 0], world[-1, 1], color="#7B61A8",
                              alpha=.35, s=6, linewidths=0)
            panel.plot(reference.history[agent, :, 0], reference.history[agent, :, 1],
                       color="#222222", lw=1.4, marker="o", ms=2, label="Observed")
            panel.plot(reference.future[agent, :, 0], reference.future[agent, :, 1],
                       color="#D97757", lw=1.5, marker="o", ms=2, label="Ground truth")
            panel.plot(cv[agent, :, 0], cv[agent, :, 1], color="#3B6FB6",
                       lw=1.0, ls="--", label="Constant velocity")
            panel.set_title(label); panel.set_aspect("equal", adjustable="datalim")
            panel.set_xlabel("World x (m)"); panel.set_ylabel("World y (m)")
        axes[0].legend(fontsize=6)
        fig.suptitle(
            f"{case['selection_rule']} | {identity} | agent {case['agent_id']} | "
            f"delta minFDE={case['delta_minFDE_m']:.3f} m", fontsize=7)
        stem = output / f"case_{index:02d}_{model}_{case['selection_rule']}"
        fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
        fig.savefig(str(stem) + ".png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        files.extend((str(stem) + ".pdf", str(stem) + ".png"))
    return files


def run_banks(args) -> None:
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    supplied = {}
    for value in args.bank:
        if "=" not in value:
            raise ValueError("--bank must be MODEL=/absolute/manifest.json")
        name, path = value.split("=", 1)
        supplied[name] = SeedBank(path)
    if "GDTS" not in supplied or len(supplied) < 2:
        raise ValueError("bank mode needs GDTS and at least one EBJD bank")
    identities = set(supplied["GDTS"].identities())
    if any(set(bank.identities()) != identities for bank in supplied.values()):
        raise ValueError("bank window identity sets differ")
    metrics, scenes_by_model, endpoint_rows, endpoint_result = {}, {}, [], {}
    k_rows, monotonic = [], {}
    replay_parity = {}
    for model, bank in supplied.items():
        metrics[model], scenes = bank_metrics(bank); scenes_by_model[model] = scenes
        archived = bank.payload.get("replay_metrics", {})
        differences = {
            name: abs(float(metrics[model]["overall"][name]) - float(archived[name]))
            for name in METRICS if name in archived
        }
        replay_parity[model] = {
            "maximum_absolute_error_m": max(differences.values(), default=None),
            "differences_m": differences,
            "status": (
                "PASSED" if differences and max(differences.values()) <= 1e-6
                else "MISSING_OR_FAILED"),
        }
        rows, summary = endpoint_summary(scenes, args.goal_scale)
        endpoint_rows.extend({"model": model, **row} for row in rows)
        endpoint_result[model] = summary
        curve, violation = k_curve(scenes, model); k_rows.extend(curve)
        monotonic[model] = violation
    paired_rows, contributions, observation_parity = [], {}, {}
    gdts = scenes_by_model["GDTS"]
    total_scenes = len(identities)
    total_agents = sum(scene.agents for scene in gdts.values())
    for model, scenes in scenes_by_model.items():
        if model == "GDTS":
            continue
        group_values = defaultdict(lambda: {"ADE": [], "FDE": [], "scenes": set()})
        observation_parity[model] = {
            "history_max_abs_m": 0.0, "future_max_abs_m": 0.0}
        for identity in sorted(identities):
            ref = gdts[identity]
            candidate = reorder_like(ref, scenes[identity])
            parity = assert_observation_parity(ref, candidate)
            for name, value in parity.items():
                observation_parity[model][name] = max(
                    observation_parity[model][name], value)
            ref_stats = hard_scene_statistics(ref.predictions, ref.future)
            candidate_stats = hard_scene_statistics(candidate.predictions, candidate.future)
            delta_ade = np.asarray(candidate_stats["agent_minade"]) - np.asarray(ref_stats["agent_minade"])
            delta_fde = np.asarray(candidate_stats["agent_minfde"]) - np.asarray(ref_stats["agent_minfde"])
            label = "N=1" if ref.agents == 1 else "N>=2"
            for index in range(ref.agents):
                paired_rows.append({
                    "model": model, "window_key": identity,
                    "agent_id": str(ref.agent_ids[index]), "group": label,
                    "delta_minADE_m": float(delta_ade[index]),
                    "delta_minFDE_m": float(delta_fde[index]),
                    "gdts_minADE_m": float(ref_stats["agent_minade"][index]),
                    "gdts_minFDE_m": float(ref_stats["agent_minfde"][index]),
                    "candidate_minADE_m": float(candidate_stats["agent_minade"][index]),
                    "candidate_minFDE_m": float(candidate_stats["agent_minfde"][index]),
                })
            group_values[label]["ADE"].extend(delta_ade.tolist())
            group_values[label]["FDE"].extend(delta_fde.tolist())
            group_values[label]["scenes"].add(identity)
        contributions[model] = {}
        for label, values in group_values.items():
            item = {"agent_occurrences": len(values["ADE"]),
                    "scene_count": len(values["scenes"])}
            for metric in ("ADE", "FDE"):
                array = np.asarray(values[metric])
                q = quantiles(array)
                tail = np.sort(array)[-max(1, math.ceil(.05 * array.size)):]
                item[f"delta_min{metric}"] = q
                item[f"agent_weighted_min{metric}_contribution_m"] = float(
                    array.sum() / total_agents)
                item[f"worst_5pct_min{metric}_contribution_m"] = float(
                    tail.sum() / total_agents)
            contributions[model][label] = item
        for label in ("N=1", "N>=2"):
            candidate = metrics[model][label]; reference = metrics["GDTS"][label]
            contributions[model][label]["scene_weighted_JADE_contribution_m"] = (
                candidate["JADE"] - reference["JADE"]) * candidate["scene_count"] / total_scenes
            contributions[model][label]["scene_weighted_JFDE_contribution_m"] = (
                candidate["JFDE"] - reference["JFDE"]) * candidate["scene_count"] / total_scenes
        contributions[model]["reconstruction"] = {}
        for metric in ("minADE", "minFDE"):
            key = f"agent_weighted_{metric}_contribution_m"
            pieces = sum(
                contributions[model][label][key]
                for label in ("N=1", "N>=2"))
            total = metrics[model]["overall"][metric] - metrics["GDTS"]["overall"][metric]
            contributions[model]["reconstruction"][metric] = {
                "total_delta_m": total, "sum_group_contributions_m": pieces,
                "residual_m": pieces - total,
            }
        for metric in ("JADE", "JFDE"):
            key = f"scene_weighted_{metric}_contribution_m"
            pieces = sum(
                contributions[model][label][key]
                for label in ("N=1", "N>=2"))
            total = metrics[model]["overall"][metric] - metrics["GDTS"]["overall"][metric]
            contributions[model]["reconstruction"][metric] = {
                "total_delta_m": total, "sum_group_contributions_m": pieces,
                "residual_m": pieces - total,
            }
    endpoint_lookup = {
        (row["model"], row["window_key"], row["agent_id"]): row
        for row in endpoint_rows}
    correlations = {}
    for model in supplied:
        if model == "GDTS":
            continue
        rows = [row for row in paired_rows if row["model"] == model]
        correlations[model] = {}
        for delta in ("delta_minADE_m", "delta_minFDE_m"):
            correlations[model][delta] = {}
            for field in (
                    "center_offset_m", "rms_radius_m",
                    "mean_pairwise_endpoint_m", "cv_endpoint_residual_m"):
                correlations[model][delta][field] = pearson(
                    [row[delta] for row in rows],
                    [endpoint_lookup[(model, row["window_key"], row["agent_id"])][field]
                     for row in rows])
    cases = []
    for model in supplied:
        if model != "GDTS":
            cases.extend(select_cases(paired_rows, model))
    case_files = render_case_plots(cases, scenes_by_model, output / "cases")
    write_csv(output / "GROUP_METRICS.csv", [
        {"model": model, "group": group, **values}
        for model, items in metrics.items() for group, values in items.items()])
    write_csv(output / "PAIRED_AGENT_DELTAS.csv", paired_rows)
    write_csv(output / "ENDPOINT_DISTRIBUTION.csv", endpoint_rows)
    write_csv(output / "K_CURVE.csv", k_rows)
    atomic_json({"schema": "ebjd-eth-fixed-case-selection-v1", "cases": cases},
                output / "CASE_SELECTION.json")
    result = {
        "schema": "ebjd-eth-targeted-bank-diagnostic-v1",
        "status": "FULL_BANK_ANALYSIS_COMPLETE",
        "models": list(supplied), "scene_count": len(identities),
        "metrics": metrics, "gap_contributions": contributions,
        "endpoint_summary": endpoint_result,
        "endpoint_delta_pearson_correlations": correlations,
        "k_monotonic_max_positive_m": monotonic,
        "bank_replay_parity": replay_parity,
        "cross_bank_observation_parity": observation_parity,
        "case_selection": cases, "case_plot_files": case_files,
        "optimizer_updates": 0, "frozen_forward_calls_this_command": 0,
    }
    atomic_json(result, output / "BANK_RESULTS.json")
    print(json.dumps(result, indent=2, sort_keys=True))


def run_export(args) -> None:
    """Delegate to one frozen-bank exporter; never performs optimization."""
    if not args.confirm_gpu_idle:
        raise RuntimeError(
            "bank export is intentionally blocked until --confirm-gpu-idle is set")
    script_dir = Path(__file__).resolve().parent
    common = [
        "--config", str(Path(args.config).resolve()),
        "--checkpoint", str(Path(args.checkpoint).resolve()),
        "--output-root", str(Path(args.output_root).resolve()),
        "--expected-checkpoint-sha256", args.expected_checkpoint_sha256,
        "--expected-config-sha256", args.expected_config_sha256,
        "--expected-manifest-sha256", args.expected_manifest_sha256,
        "--seeds", str(args.seed), "--device", args.device,
    ]
    if args.max_scenes is not None:
        common.extend(("--max-scenes", str(args.max_scenes)))
    if args.resume:
        common.append("--resume")
    if args.model == "ebjd":
        command = [
            sys.executable, str(script_dir / "export_epoch45_diagnostic_bank.py"),
            *common, "--bank-name", args.bank_name,
            "--model-label", args.model_label,
            "--manifest-role", args.manifest_role,
        ]
    else:
        if not args.gdts_root or not args.manifest:
            raise ValueError("GDTS export requires --gdts-root and --manifest")
        command = [
            sys.executable,
            str(script_dir / "export_gdts_manifest_diagnostic_bank.py"),
            "--gdts-root", str(Path(args.gdts_root).resolve()),
            "--manifest", str(Path(args.manifest).resolve()), *common,
        ]
    environment = os.environ.copy()
    package_root = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = package_root + (
        os.pathsep + environment["PYTHONPATH"]
        if environment.get("PYTHONPATH") else "")
    subprocess.run(command, check=True, env=environment)


def run_report(args) -> None:
    """Rebuild a compact report from immutable CPU/bank result JSON files."""
    cpu_path = Path(args.cpu_results).resolve()
    cpu = read_json(cpu_path)
    bank_path = Path(args.bank_results).resolve() if args.bank_results else None
    bank = read_json(bank_path) if bank_path else None
    status = (
        "FULL_DIAGNOSTIC_COMPLETE"
        if bank is not None and bank.get("status") == "FULL_BANK_ANALYSIS_COMPLETE"
        else "CPU_COMPLETE_GPU_PENDING")
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "ebjd-eth-targeted-rebuilt-report-v1", "status": status,
        "cpu_results": str(cpu_path), "cpu_results_sha256": sha256_file(cpu_path),
        "bank_results": str(bank_path) if bank_path else None,
        "bank_results_sha256": sha256_file(bank_path) if bank_path else None,
        "optimizer_updates": 0,
        "frozen_model_forward_calls": (
            0 if bank is None else "recorded in supplied bank manifests"),
        "metric_parity": cpu["evaluator_parity"],
        "scale_shift": cpu["scale_shift"],
        "cv_metrics": cpu["validation_manifest"]["cv_metrics"],
        "training_trends": cpu["training_trends"],
        "bank_analysis": bank,
        "pending": cpu["pending"] if bank is None else [],
    }
    atomic_json(summary, output / "REBUILT_REPORT.json")
    parity = cpu["evaluator_parity"]
    validation = cpu["validation_manifest"]
    overall = validation["cv_metrics"]["overall"]
    text = (
        "---\n"
        "experiment_id: EBJD-ETH-TARGETED-DIAGNOSTIC-20261015\n"
        "date: 2026-10-15\n"
        f"status: {status.lower()}\n"
        "optimizer_updates: 0\n"
        "---\n\n"
        "# ETH targeted diagnostic (rebuilt)\n\n"
        f"- Status: `{status}`\n"
        f"- Evaluator parity: `{parity['status']}`; maximum absolute difference "
        f"`{max(parity['maximum_absolute_error_m'].values()):.9g} m`.\n"
        "- ETH train-to-validation goal/bridge RMS ratio: "
        f"`{cpu['scale_shift']['goal_ratio_validation_over_train']:.6f}x` / "
        f"`{cpu['scale_shift']['bridge_ratio_validation_over_train']:.6f}x`.\n"
        f"- CV on validation: minADE `{overall['minADE']:.6f}`, minFDE "
        f"`{overall['minFDE']:.6f}`, JADE `{overall['JADE']:.6f}`, "
        f"JFDE `{overall['JFDE']:.6f}` m.\n"
        "- Latest complete training epoch at CPU boundary: "
        f"`{cpu['training_trends']['latest_complete_epoch']}`.\n"
        "- Frozen bank analysis: "
        f"`{'included' if bank is not None else 'pending'}`.\n"
    )
    atomic_text(text, output / "REBUILT_REPORT.md")
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_cpu(args) -> None:
    started = time.monotonic()
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    train, train_rows, train_frames = manifest_statistics(args.train_manifest)
    validation, validation_rows, validation_frames = manifest_statistics(args.validation_manifest)
    parity = evaluator_parity(
        list(iter_manifest_scenes(args.validation_manifest)), args.gdts_metrics)
    trend_rows, trends = training_trends(args.train_log, args.epoch_summary)
    write_csv(output / "LATENT_AND_MOTION_AGENT_STATS.csv", [
        {"split": "train", **row} for row in train_rows] + [
        {"split": "validation", **row} for row in validation_rows])
    write_csv(output / "FRAME_TIME_AUDIT.csv", [
        {"split": "train", **row} for row in train_frames] + [
        {"split": "validation", **row} for row in validation_frames])
    write_csv(output / "TRAINING_TREND.csv", trend_rows)
    figure = plot_training(
        trend_rows, output / "figures" / "eth_training_stage",
        Path(args.figure_qa_scripts).resolve() if args.figure_qa_scripts else None)
    status = "CPU_COMPLETE_GPU_PENDING"
    result = {
        "schema": "ebjd-eth-targeted-cpu-diagnostic-v1", "status": status,
        "optimizer_updates": 0, "frozen_model_forward_calls": 0,
        "train_manifest": train, "validation_manifest": validation,
        "evaluator_parity": parity, "training_trends": trends,
        "figure_contract": figure,
        "scale_shift": {
            "goal_ratio_validation_over_train": (
                validation["goal_component_rms_m"] / train["goal_component_rms_m"]),
            "bridge_ratio_validation_over_train": (
                validation["bridge_component_rms_m"] / train["bridge_component_rms_m"]),
        },
        "pending": [
            "GDTS and EBJD epoch-5/latest frozen P20 banks",
            "model N=1/N>=2 gap contribution decomposition",
            "model endpoint center/spread/nearest diagnostics",
            "model K=1/4/8/12/16/20 curves and fixed-rule case plots",
        ],
        "cpu_elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(result, output / "CPU_RESULTS.json")
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cpu = commands.add_parser("cpu", help="run all no-model CPU diagnostics")
    cpu.add_argument("--train-manifest", required=True)
    cpu.add_argument("--validation-manifest", required=True)
    cpu.add_argument("--train-log", required=True)
    cpu.add_argument("--epoch-summary", required=True)
    cpu.add_argument("--gdts-metrics", required=True)
    cpu.add_argument("--output-dir", required=True)
    cpu.add_argument("--figure-qa-scripts")
    cpu.set_defaults(function=run_cpu)
    banks = commands.add_parser("banks", help="analyze existing frozen banks; no forward")
    banks.add_argument("--bank", action="append", required=True,
                       help="repeat MODEL=/absolute/seed_manifest.json")
    banks.add_argument("--goal-scale", type=float, required=True)
    banks.add_argument("--output-dir", required=True)
    banks.set_defaults(function=run_banks)
    export = commands.add_parser(
        "export", help="SHA-bound frozen P20 export; zero optimizer updates")
    export.add_argument("--model", choices=("ebjd", "gdts"), required=True)
    export.add_argument("--config", required=True)
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output-root", required=True)
    export.add_argument("--expected-checkpoint-sha256", required=True)
    export.add_argument("--expected-config-sha256", required=True)
    export.add_argument("--expected-manifest-sha256", required=True)
    export.add_argument("--seed", type=int, default=2035)
    export.add_argument("--device", default="cuda:0")
    export.add_argument("--max-scenes", type=int)
    export.add_argument("--resume", action="store_true")
    export.add_argument("--confirm-gpu-idle", action="store_true")
    export.add_argument("--bank-name", default="ebjd")
    export.add_argument("--model-label", default="EBJD")
    export.add_argument("--manifest-role", choices=("test", "validation"),
                        default="validation")
    export.add_argument("--gdts-root")
    export.add_argument("--manifest")
    export.set_defaults(function=run_export)
    report = commands.add_parser(
        "report", help="rebuild compact Markdown/JSON from diagnostic results")
    report.add_argument("--cpu-results", required=True)
    report.add_argument("--bank-results")
    report.add_argument("--output-dir", required=True)
    report.set_defaults(function=run_report)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()

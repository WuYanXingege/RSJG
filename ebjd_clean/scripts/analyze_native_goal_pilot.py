"""Aggregate the frozen ETH native-goal pilot and render audit figures.

The script consumes only read-only evaluation JSON files and local prediction
banks.  It writes compact source-data tables, a machine-readable diagnostic
summary, and two publication-style figures.  Full prediction banks remain
local and are never copied into the documentation bundle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker


METRICS = ("minADE", "minFDE", "JADE", "JFDE")
K_VALUES = (1, 4, 8, 12, 16, 20)
METHODS = ("epoch5", "control", "candidate", "disabled", "gdts")
COLORS = {
    "epoch5": "#8A8F98",
    "control": "#B7BDC8",
    "candidate": "#2878B5",
    "disabled": "#D7A85B",
    "gdts": "#4A9B72",
}
LABELS = {
    "epoch5": "EBJD epoch 5",
    "control": "Control epoch 10",
    "candidate": "Native-goal candidate",
    "disabled": "Candidate, condition off",
    "gdts": "Matched GDTS",
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def sample_stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "sample_standard_deviation": float(statistics.stdev(values)),
    }


def metric_vector(payload: dict, name: str) -> list[float]:
    return [float(row["metrics"][name]) for row in payload["per_seed"]]


def gdts_metric_vector(payload: dict, name: str) -> list[float]:
    mapping = {"minADE": "minADE@K", "minFDE": "minFDE@K"}
    key = mapping.get(name, name)
    return [float(row[key]) for row in payload["runs"]]


def compare(candidate: list[float], reference: list[float]) -> dict:
    deltas = [a - b for a, b in zip(candidate, reference, strict=True)]
    mean_candidate = statistics.fmean(candidate)
    mean_reference = statistics.fmean(reference)
    return {
        "candidate_minus_reference": float(mean_candidate - mean_reference),
        "relative_change_percent": float(
            100.0 * (mean_candidate - mean_reference) / mean_reference),
        "per_seed_delta": [float(x) for x in deltas],
        "improved_seed_count": int(sum(x < 0 for x in deltas)),
    }


def scene_statistics(prediction: np.ndarray, future: np.ndarray) -> dict:
    distance = np.linalg.norm(prediction - future[None], axis=-1)
    ade = distance.mean(-1)  # [K,N]
    fde = distance[..., -1]  # [K,N]
    return {
        "agent_minade": ade.min(0),
        "agent_minfde": fde.min(0),
        "scene_jade": float(ade.mean(1).min()),
        "scene_jfde": float(fde.mean(1).min()),
        "best_jfde_world": int(np.argmin(fde.mean(1))),
        "single_candidate_fde": fde.reshape(-1),
        "endpoints": prediction[..., -1, :],
    }


def bank_diagnostics(path: Path) -> tuple[dict, list[dict]]:
    bank = np.load(path, allow_pickle=True)
    minfdes: list[float] = []
    single: list[float] = []
    centroid_errors: list[float] = []
    centered_rms: list[float] = []
    pairwise: list[float] = []
    native_oracle: list[float] = []
    native_recalls = {"0.5": 0, "1.0": 0}
    agent_count = 0
    scenes: list[dict] = []
    for index, scene_id in enumerate(bank["scene_ids"]):
        future = np.asarray(bank["future"][index], dtype=np.float64)
        prediction = np.asarray(bank["predictions"][index], dtype=np.float64)
        stats = scene_statistics(prediction, future)
        endpoints = stats["endpoints"]  # [K,N,2]
        target = future[:, -1]
        centroid = endpoints.mean(0)
        minfdes.extend(stats["agent_minfde"].tolist())
        single.extend(stats["single_candidate_fde"].tolist())
        centroid_errors.extend(np.linalg.norm(centroid - target, axis=-1).tolist())
        centered_rms.extend(np.sqrt(np.mean(
            np.sum((endpoints - centroid[None]) ** 2, axis=-1), axis=0)).tolist())
        for agent in range(endpoints.shape[1]):
            delta = endpoints[:, agent, None] - endpoints[None, :, agent]
            distances = np.linalg.norm(delta, axis=-1)
            upper = distances[np.triu_indices(endpoints.shape[0], 1)]
            pairwise.append(float(upper.mean()))
        native = bank["native_goals"][index]
        if native is not None:
            native = np.asarray(native, dtype=np.float64)
            errors = np.linalg.norm(native - target[:, None], axis=-1).min(1)
            native_oracle.extend(errors.tolist())
            for threshold in (0.5, 1.0):
                native_recalls[str(threshold)] += int((errors <= threshold).sum())
        agent_count += future.shape[0]
        scenes.append({
            "scene_id": str(scene_id),
            "index": int(index),
            "agents": int(future.shape[0]),
            "scene_jfde": stats["scene_jfde"],
            "best_jfde_world": stats["best_jfde_world"],
            "native_oracle_fde": None if native is None else float(errors.mean()),
        })
    result = {
        "scene_count": int(len(bank["scene_ids"])),
        "agent_occurrence_count": int(agent_count),
        "minFDE_P90": float(np.quantile(minfdes, 0.90)),
        "minFDE_P95": float(np.quantile(minfdes, 0.95)),
        "average_single_candidate_FDE": float(np.mean(single)),
        "candidate_endpoint_centroid_error_to_GT": float(np.mean(centroid_errors)),
        "candidate_endpoint_centered_RMS_relative_to_candidate_mean": float(
            np.mean(centered_rms)),
        "candidate_endpoint_mean_pairwise_distance": float(np.mean(pairwise)),
    }
    if native_oracle:
        result.update({
            "native_goal_oracle_FDE": float(np.mean(native_oracle)),
            "native_goal_recall_0p5": float(native_recalls["0.5"] / agent_count),
            "native_goal_recall_1p0": float(native_recalls["1.0"] / agent_count),
        })
    return result, scenes


def summarize_bank_group(root: Path) -> tuple[dict, dict[int, list[dict]]]:
    per_seed = {}
    scenes = {}
    for seed in range(2035, 2040):
        diagnostic, scene_rows = bank_diagnostics(root / f"seed_{seed}.npz")
        per_seed[str(seed)] = diagnostic
        scenes[seed] = scene_rows
    scalar_names = [
        name for name, value in per_seed["2035"].items()
        if isinstance(value, float)
    ]
    aggregate = {
        name: sample_stats([per_seed[str(seed)][name] for seed in range(2035, 2040)])
        for name in scalar_names
    }
    return {"per_seed": per_seed, "aggregate": aggregate}, scenes


def aggregate_k_curves(payloads: dict[str, dict]) -> dict:
    output = {}
    for method in ("epoch5", "control", "candidate", "disabled"):
        payload = payloads[method]
        output[method] = {}
        for k in K_VALUES:
            output[method][str(k)] = {
                metric: sample_stats([
                    float(row["k_curves"][str(k)][metric])
                    for row in payload["per_seed"]
                ]) for metric in METRICS
            }
    return output


def aggregate_strata(payloads: dict[str, dict]) -> dict:
    output = {}
    for method in ("epoch5", "control", "candidate", "disabled"):
        output[method] = {}
        for stratum in ("N=1", "N>=2"):
            rows = [row["strata"][stratum] for row in payloads[method]["per_seed"]]
            output[method][stratum] = {
                "scene_count": int(rows[0]["scene_count"]),
                "agent_occurrence_count": int(rows[0]["agent_count"]),
                "metrics": {
                    metric: sample_stats([float(row[metric]) for row in rows])
                    for metric in METRICS
                },
            }
    return output


def engineering_receipt(asset_root: Path, payloads: dict[str, dict]) -> dict:
    runs = asset_root / "runs_v4/eth/3101/none"
    names = {
        "control": "eth_native_goal_control_6f7a05f_20261019",
        "candidate": "eth_native_goal_candidate_6f7a05f_20261019",
    }
    logs = {}
    epochs = {}
    for arm, name in names.items():
        run = runs / name
        logs[arm] = [json.loads(line) for line in
                     (run / "train.jsonl").read_text().splitlines()]
        epochs[arm] = [json.loads(line) for line in
                       (run / "EPOCH_SUMMARY.jsonl").read_text().splitlines()]
    paired_fields = (
        "epoch", "update", "logical_scene_agent_identity_sha256",
        "logical_scene_id_sha256", "augmentation_sha256", "random_draw_sha256",
        "learning_rate", "unet_learning_rate", "active_rollout_weight",
        "effective_rollout_coefficient", "rollout_update", "projection_active",
    )
    mismatch = []
    if len(logs["control"]) != len(logs["candidate"]):
        mismatch.append({"field": "record_count"})
    for index, (control, candidate) in enumerate(zip(
            logs["control"], logs["candidate"], strict=True)):
        for field in paired_fields:
            if control[field] != candidate[field]:
                mismatch.append({
                    "record_index": index, "field": field,
                    "control": control[field], "candidate": candidate[field],
                })
                if len(mismatch) >= 20:
                    break
        if len(mismatch) >= 20:
            break
    candidate_gradients = np.asarray([
        row["native_goal_gate_gradient"] for row in logs["candidate"]],
        dtype=np.float64)
    training_cost = {}
    for arm in names:
        updates = sum(int(row["updates"]) for row in epochs[arm])
        normal_seconds = sum(float(row["normal_step_seconds"]) for row in epochs[arm])
        training_cost[arm] = {
            "updates": updates,
            "normal_update_seconds": normal_seconds,
            "seconds_per_update": normal_seconds / updates,
            "total_epoch_wall_seconds_including_validation_and_io": sum(
                float(row["wall_seconds"]) for row in epochs[arm]),
            "validation_seconds_total": sum(
                float(row["validation_seconds"]) for row in epochs[arm]),
            "peak_cuda_allocated_bytes": max(
                int(row["peak_cuda_allocated_bytes"]) for row in epochs[arm]),
        }
    eval_bank_roots = {
        "epoch5": asset_root / "evaluation/banks_epoch005",
        "control": asset_root / "evaluation_v4/banks_control",
        "candidate": asset_root / "evaluation_v4/banks_candidate",
        "disabled": asset_root / "evaluation_v4/banks_candidate_disabled",
    }
    inference_timing = {}
    for method, root in eval_bank_roots.items():
        times = [(root / f"seed_{seed}.npz").stat().st_mtime
                 for seed in range(2035, 2040)]
        intervals = np.diff(times)
        inference_timing[method] = {
            "mean_inter_seed_completion_interval_seconds": float(intervals.mean()),
            "sample_standard_deviation_seconds": float(intervals.std(ddof=1)),
            "interval_count": int(len(intervals)),
            "scope": "full 253-scene seed evaluation plus compressed bank write",
            "caveat": "mtime-derived completion interval; not a synchronized kernel benchmark",
            "NFE_per_prediction": 20,
        }
    inference_timing["gdts"] = {
        "mean_elapsed_seconds_per_seed": float(statistics.fmean(
            row["elapsed_seconds"] for row in payloads["gdts"]["runs"])),
        "sample_standard_deviation_seconds": float(statistics.stdev(
            row["elapsed_seconds"] for row in payloads["gdts"]["runs"])),
        "scope": "full 253-scene seed evaluation",
        "NFE_per_prediction": 20,
    }
    provider_manifests = {
        "train_seed3101": asset_root / "train_seed3101_provider092_rngfix/manifest.json",
        **{f"validation_seed{seed}": asset_root /
           f"validation_seed{seed}_provider092_rngfix/manifest.json"
           for seed in range(2035, 2040)},
    }
    provider = {name: {
        "elapsed_seconds": float(read_json(path)["elapsed_seconds"]),
        "manifest_sha256": sha256_file(path),
    } for name, path in provider_manifests.items()}
    checkpoint_paths = {
        arm: runs / name / "epoch_010.pt" for arm, name in names.items()}
    return {
        "implementation_commit": "6f7a05f718accf322dc0dedb6d024111a115e6f8",
        "formal_checkpoint_sha256": {
            arm: sha256_file(path) for arm, path in checkpoint_paths.items()},
        "pairing": {
            "status": "PASS" if not mismatch else "FAIL",
            "updates_per_arm": len(logs["control"]),
            "fields_checked": list(paired_fields),
            "mismatches": mismatch,
        },
        "conditioner": {
            "trainable_parameter_increment": 74945,
            "initial_gate_before_first_update": 0.0,
            "gate_after_first_update": float(logs["candidate"][0]["native_goal_gate"]),
            "final_gate": float(logs["candidate"][-1]["native_goal_gate"]),
            "final_condition_to_hidden_norm_ratio": float(
                logs["candidate"][-1]["native_goal_condition_norm_ratio"]),
            "finite_nonzero_gate_gradient_updates": int(np.sum(
                np.isfinite(candidate_gradients) & (candidate_gradients != 0))),
            "formal_candidate_updates": int(len(candidate_gradients)),
        },
        "training_cost": training_cost,
        "inference_timing": inference_timing,
        "provider_cache_generation": provider,
        "provider_validation_total_seconds": float(sum(
            value["elapsed_seconds"] for name, value in provider.items()
            if name.startswith("validation_seed"))),
        "rollout_updates": {
            arm: int(sum(row["rollout_updates"] for row in epochs[arm]))
            for arm in names},
    }


def select_cases(parent_path: Path, candidate_path: Path) -> list[dict]:
    parent = np.load(parent_path, allow_pickle=True)
    candidate = np.load(candidate_path, allow_pickle=True)
    if not np.array_equal(parent["scene_ids"], candidate["scene_ids"]):
        raise RuntimeError("parent and candidate scene order differs")
    rows = []
    for index, scene_id in enumerate(candidate["scene_ids"]):
        future = np.asarray(candidate["future"][index], dtype=np.float64)
        parent_stats = scene_statistics(
            np.asarray(parent["predictions"][index], dtype=np.float64), future)
        candidate_stats = scene_statistics(
            np.asarray(candidate["predictions"][index], dtype=np.float64), future)
        native = np.asarray(candidate["native_goals"][index], dtype=np.float64)
        native_oracle = np.linalg.norm(
            native - future[:, -1, None], axis=-1).min(1).mean()
        rows.append({
            "scene_id": str(scene_id), "index": int(index),
            "agents": int(future.shape[0]),
            "parent_JFDE": parent_stats["scene_jfde"],
            "candidate_JFDE": candidate_stats["scene_jfde"],
            "parent_minus_candidate_JFDE": (
                parent_stats["scene_jfde"] - candidate_stats["scene_jfde"]),
            "native_A_oracle_FDE": float(native_oracle),
            "parent_best_world": parent_stats["best_jfde_world"],
            "candidate_best_world": candidate_stats["best_jfde_world"],
        })
    selected = []
    improve = max(rows, key=lambda row: row["parent_minus_candidate_JFDE"])
    selected.append({"rule": "maximum_parent_to_candidate_JFDE_improvement", **improve})
    degrade = min(rows, key=lambda row: row["parent_minus_candidate_JFDE"])
    selected.append({"rule": "maximum_parent_to_candidate_JFDE_degradation", **degrade})
    a_good = [row for row in rows if row["native_A_oracle_FDE"] <= 0.5]
    if not a_good:
        raise RuntimeError("no scene satisfies native A oracle FDE <= 0.5 m")
    bad = max(a_good, key=lambda row: row["candidate_JFDE"])
    selected.append({
        "rule": "native_A_oracle_FDE_le_0p5_max_candidate_JFDE", **bad})
    used = {row["scene_id"] for row in selected}
    remaining = [row for row in rows if row["scene_id"] not in used]
    random_row = min(remaining, key=lambda row: hashlib.sha256(
        ("native-goal-case-v1|" + row["scene_id"]).encode("utf-8")).hexdigest())
    selected.append({
        "rule": "minimum_SHA256_native-goal-case-v1_scene_id_excluding_prior_cases",
        **random_row,
    })
    return selected


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def configure_matplotlib() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def alignment_gate(fig, stem: Path) -> None:
    skill_scripts = Path("/home/ee615/.codex/skills/nature-figure/scripts")
    if str(skill_scripts) not in sys.path:
        sys.path.insert(0, str(skill_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(stem.with_suffix(".alignment.json")),
        overlay_svg=str(stem.with_suffix(".alignment.svg")),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )


def export_figure(fig, stem: Path) -> None:
    alignment_gate(fig, stem)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax, label: str) -> None:
    ax.annotate(label, xy=(0, 1), xycoords="axes fraction",
                xytext=(0, 8), textcoords="offset points",
                fontsize=8, fontweight="bold", ha="left", va="top")


def colored_text_legend(ax, labels: list[str], colors: list[str], **kwargs) -> None:
    """Add a text-only color key above an axis, without graphical handles."""
    ncol = int(kwargs.get("ncol", len(labels)))
    rows = []
    for start in range(0, len(labels), ncol):
        cells = [TextArea(label, textprops={"size": 6, "color": color})
                 for label, color in zip(
                     labels[start:start + ncol], colors[start:start + ncol],
                     strict=True)]
        rows.append(HPacker(children=cells, align="center", pad=0, sep=9))
    packed = VPacker(children=rows, align="center", pad=0, sep=2)
    artist = AnchoredOffsetbox(
        loc="lower center", child=packed, pad=0, frameon=False,
        bbox_to_anchor=(0.5, 1.01), bbox_transform=ax.transAxes,
        borderpad=0)
    ax.add_artist(artist)


def plot_summary(summary: dict, output: Path) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
    ax = axes[0, 0]
    x = np.arange(len(METRICS))
    width = 0.15
    for offset, method in enumerate(METHODS):
        values = [summary["methods"][method][metric]["mean"] for metric in METRICS]
        errors = [summary["methods"][method][metric]["sample_standard_deviation"]
                  for metric in METRICS]
        ax.bar(x + (offset - 2) * width, values, width, yerr=errors,
               color=COLORS[method], label=LABELS[method], capsize=1.5,
               linewidth=0.4, edgecolor="white")
    ax.set_xticks(x, METRICS)
    ax.set_ylabel("Error (m), mean ± seed SD")
    ax.set_title("Five-seed frozen-checkpoint comparison", pad=40)
    colored_text_legend(
        ax, [LABELS[m] for m in METHODS], [COLORS[m] for m in METHODS],
        ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    panel_label(ax, "a")

    ax = axes[0, 1]
    deltas = summary["comparisons"]["candidate_vs_control"]
    seed_x = np.arange(2035, 2040)
    for metric, color, marker in zip(
            METRICS, ("#2878B5", "#E07A5F", "#6B8E75", "#8064A2"),
            ("o", "s", "^", "D"), strict=True):
        values = np.asarray(deltas[metric]["per_seed_delta"])
        ax.plot(seed_x, values, marker=marker, markersize=3.2,
                linewidth=1.0, color=color, label=metric)
    ax.axhline(0, color="#555555", linewidth=0.7, linestyle="--")
    ax.set_xticks(seed_x)
    ax.set_xlabel("Inference seed")
    ax.set_ylabel("Candidate − Control (m)")
    ax.set_title("Paired change; lower is better", pad=40)
    colored_text_legend(
        ax, list(METRICS), ["#2878B5", "#E07A5F", "#6B8E75", "#8064A2"],
        ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    panel_label(ax, "b")

    ax = axes[1, 0]
    for method in ("epoch5", "control", "candidate", "disabled"):
        means = [summary["k_curves"][method][str(k)]["minFDE"]["mean"]
                 for k in K_VALUES]
        sds = [summary["k_curves"][method][str(k)]["minFDE"]
               ["sample_standard_deviation"] for k in K_VALUES]
        ax.errorbar(K_VALUES, means, yerr=sds, marker="o", markersize=3,
                    linewidth=1.0, capsize=1.5, color=COLORS[method],
                    label=LABELS[method])
    ax.set_xticks(K_VALUES)
    ax.set_xlabel("Fixed prediction prefix K")
    ax.set_ylabel("minFDE (m), mean ± seed SD")
    ax.set_title("Candidate-budget response", pad=40)
    colored_text_legend(
        ax, [LABELS[m] for m in ("epoch5", "control", "candidate", "disabled")],
        [COLORS[m] for m in ("epoch5", "control", "candidate", "disabled")],
        ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    panel_label(ax, "c")

    ax = axes[1, 1]
    strata = ("N=1", "N>=2")
    width = 0.18
    for offset, method in enumerate(("epoch5", "control", "candidate", "disabled")):
        values = [summary["strata"][method][s]["metrics"]["JFDE"]["mean"]
                  for s in strata]
        errors = [summary["strata"][method][s]["metrics"]["JFDE"]
                  ["sample_standard_deviation"] for s in strata]
        ax.bar(np.arange(2) + (offset - 1.5) * width, values, width,
               yerr=errors, capsize=1.5, color=COLORS[method],
               label=LABELS[method], linewidth=0.4, edgecolor="white")
    counts = [summary["strata"]["candidate"][s]["scene_count"] for s in strata]
    ax.set_xticks(np.arange(2), [f"{s}\n({n} scenes)" for s, n in zip(strata, counts)])
    ax.set_ylabel("JFDE (m), mean ± seed SD")
    ax.set_title("Scene-size stratification", pad=40)
    colored_text_legend(
        ax, [LABELS[m] for m in ("epoch5", "control", "candidate", "disabled")],
        [COLORS[m] for m in ("epoch5", "control", "candidate", "disabled")],
        ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    panel_label(ax, "d")
    export_figure(fig, output / "native_goal_pilot_summary")


def plot_cases(cases: list[dict], parent_path: Path, candidate_path: Path,
               output: Path) -> None:
    configure_matplotlib()
    parent = np.load(parent_path, allow_pickle=True)
    candidate = np.load(candidate_path, allow_pickle=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), constrained_layout=True)
    titles = ("Maximum improvement", "Maximum degradation",
              "Native A good, Candidate poor", "Hash-fixed random scene")
    agent_colors = ("#1F77B4", "#D95F02", "#2CA02C", "#9467BD", "#8C564B")
    for panel, (ax, case, title) in enumerate(zip(
            axes.flat, cases, titles, strict=True)):
        index = case["index"]
        observed = np.asarray(candidate["observed"][index], dtype=np.float64)
        future = np.asarray(candidate["future"][index], dtype=np.float64)
        native = np.asarray(candidate["native_goals"][index], dtype=np.float64)
        parent_pred = np.asarray(parent["predictions"][index], dtype=np.float64)
        candidate_pred = np.asarray(candidate["predictions"][index], dtype=np.float64)
        for agent in range(future.shape[0]):
            color = agent_colors[agent % len(agent_colors)]
            for world in range(parent_pred.shape[0]):
                ax.plot(parent_pred[world, agent, :, 0],
                        parent_pred[world, agent, :, 1], color="#AEB4BD",
                        linewidth=0.35, alpha=0.20, zorder=1)
                ax.plot(candidate_pred[world, agent, :, 0],
                        candidate_pred[world, agent, :, 1], color="#77AADD",
                        linewidth=0.35, alpha=0.20, zorder=1)
            ax.scatter(native[agent, :, 0], native[agent, :, 1], marker="x",
                       color="#4A9B72", s=8, linewidth=0.55, alpha=0.65, zorder=2)
            ax.plot(observed[agent, :, 0], observed[agent, :, 1], color=color,
                    linewidth=1.0, linestyle=":", zorder=3)
            ax.plot(future[agent, :, 0], future[agent, :, 1], color=color,
                    linewidth=1.6, zorder=4)
            pw = case["parent_best_world"]
            cw = case["candidate_best_world"]
            ax.plot(parent_pred[pw, agent, :, 0], parent_pred[pw, agent, :, 1],
                    color="#666A73", linewidth=1.0, linestyle="--", zorder=3)
            ax.plot(candidate_pred[cw, agent, :, 0], candidate_pred[cw, agent, :, 1],
                    color="#2878B5", linewidth=1.2, linestyle="-.", zorder=3)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(
            f"{title}\nParent/Candidate JFDE "
            f"{case['parent_JFDE']:.2f}/{case['candidate_JFDE']:.2f} m; "
            f"A oracle {case['native_A_oracle_FDE']:.2f} m")
        panel_label(ax, chr(ord("a") + panel))
    handles = [
        mpl.lines.Line2D([], [], color="#333333", linestyle=":", label="Observed"),
        mpl.lines.Line2D([], [], color="#333333", linewidth=1.6, label="Ground truth"),
        mpl.lines.Line2D([], [], color="#4A9B72", marker="x", linestyle="None",
                         label="Native A endpoints"),
        mpl.lines.Line2D([], [], color="#666A73", linestyle="--", label="Parent best world"),
        mpl.lines.Line2D([], [], color="#2878B5", linestyle="-.", label="Candidate best world"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, frameon=False)
    export_figure(fig, output / "native_goal_pilot_fixed_cases")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--skip-figures", action="store_true",
                        help="refresh tables/JSON without rerendering figures")
    args = parser.parse_args()
    asset_root = args.asset_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "epoch5": asset_root / "evaluation/epoch005_seeds2035_2039.json",
        "control": asset_root / "evaluation_v4/control_epoch010_seeds2035_2039.json",
        "candidate": asset_root / "evaluation_v4/candidate_epoch010_seeds2035_2039.json",
        "disabled": asset_root / "evaluation_v4/candidate_disabled_epoch010_seeds2035_2039.json",
        "gdts": asset_root / "evaluation_v4/matched_gdts_seeds2035_2039.json",
    }
    payloads = {name: read_json(path) for name, path in paths.items()}
    methods = {}
    vectors = {}
    for method in METHODS:
        methods[method] = {}
        vectors[method] = {}
        for metric in METRICS:
            vector = (gdts_metric_vector(payloads[method], metric)
                      if method == "gdts" else metric_vector(payloads[method], metric))
            vectors[method][metric] = vector
            methods[method][metric] = sample_stats(vector)
            methods[method][metric]["per_seed"] = vector
    comparisons = {}
    for reference in ("control", "epoch5", "gdts", "disabled"):
        comparisons[f"candidate_vs_{reference}"] = {
            metric: compare(vectors["candidate"][metric], vectors[reference][metric])
            for metric in METRICS
        }
    k_curves = aggregate_k_curves(payloads)
    strata = aggregate_strata(payloads)
    bank_roots = {
        "epoch5": asset_root / "evaluation/banks_epoch005",
        "control": asset_root / "evaluation_v4/banks_control",
        "candidate": asset_root / "evaluation_v4/banks_candidate",
        "disabled": asset_root / "evaluation_v4/banks_candidate_disabled",
    }
    bank_diagnostics_payload = {}
    for method, root in bank_roots.items():
        bank_diagnostics_payload[method], _ = summarize_bank_group(root)
    candidate_endpoint = {
        str(row["seed"]): row["endpoint_diagnostics"]
        for row in payloads["candidate"]["per_seed"]
    }
    cases = select_cases(
        bank_roots["epoch5"] / "seed_2035.npz",
        bank_roots["candidate"] / "seed_2035.npz")
    all_four_better = all(
        comparisons["candidate_vs_control"][metric]["improved_seed_count"] == 5
        for metric in METRICS)
    screening = (
        comparisons["candidate_vs_control"]["minFDE"]["relative_change_percent"] <= -5
        and all(comparisons["candidate_vs_control"][metric]
                ["candidate_minus_reference"] <= 0 for metric in METRICS)
        and comparisons["candidate_vs_control"]["minFDE"]["improved_seed_count"] >= 4
    )
    exceeds_parent = all(
        comparisons["candidate_vs_epoch5"][metric]["candidate_minus_reference"] <= 0
        for metric in METRICS)
    contract = all(
        comparisons["candidate_vs_gdts"][metric]["candidate_minus_reference"] <= 0
        for metric in METRICS)
    summary = {
        "schema": "ebjd-eth-native-goal-pilot-analysis-v1",
        "status": "COMPLETE",
        "unit": "m",
        "seeds": list(range(2035, 2040)),
        "metrics_lower_is_better": list(METRICS),
        "methods": methods,
        "comparisons": comparisons,
        "screening": {
            "positive_short_horizon_signal": bool(screening),
            "candidate_better_than_control_all_four_metrics_all_five_seeds": bool(
                all_four_better),
            "candidate_exceeds_epoch5_all_four_mean_metrics": bool(exceeds_parent),
            "original_contract_candidate_no_worse_than_matched_gdts_all_four": bool(
                contract),
        },
        "k_curves": k_curves,
        "strata": strata,
        "bank_diagnostics": bank_diagnostics_payload,
        "candidate_endpoint_diagnostics_from_evaluator": candidate_endpoint,
        "fixed_case_selection_seed": 2035,
        "fixed_cases": cases,
        "engineering": engineering_receipt(asset_root, payloads),
        "definitions": {
            "candidate_endpoint_centroid_error_to_GT": (
                "Euclidean distance between the mean of 20 predicted endpoints and GT; "
                "averaged over agent occurrences"),
            "candidate_endpoint_centered_RMS_relative_to_candidate_mean": (
                "RMS radius of 20 predicted endpoints about their own candidate mean; "
                "averaged over agent occurrences"),
            "candidate_endpoint_mean_pairwise_distance": (
                "mean of all 190 unordered endpoint-pair distances per agent; "
                "averaged over agent occurrences"),
            "average_single_candidate_FDE": (
                "mean endpoint error before best-of-20 selection over candidates and agents"),
            "fixed_K_curve": "prefix of the same stored P20 bank; no resampling",
        },
    }
    atomic_json(summary, output / "RESULTS_AND_DIAGNOSTICS.json")

    metric_rows = []
    for method in METHODS:
        for metric in METRICS:
            for seed, value in zip(range(2035, 2040), vectors[method][metric], strict=True):
                metric_rows.append({
                    "method": LABELS[method], "metric": metric,
                    "seed": seed, "value_m": value,
                })
    write_csv(output / "source_data_metrics.csv",
              ["method", "metric", "seed", "value_m"], metric_rows)
    k_rows = []
    for method in ("epoch5", "control", "candidate", "disabled"):
        for row in payloads[method]["per_seed"]:
            for k in K_VALUES:
                for metric in METRICS:
                    k_rows.append({
                        "method": LABELS[method], "seed": row["seed"], "K": k,
                        "metric": metric,
                        "value_m": row["k_curves"][str(k)][metric],
                    })
    write_csv(output / "source_data_k_curves.csv",
              ["method", "seed", "K", "metric", "value_m"], k_rows)
    strata_rows = []
    for method in ("epoch5", "control", "candidate", "disabled"):
        for row in payloads[method]["per_seed"]:
            for stratum in ("N=1", "N>=2"):
                source = row["strata"][stratum]
                strata_rows.append({
                    "method": LABELS[method], "seed": row["seed"],
                    "stratum": stratum, "scene_count": source["scene_count"],
                    "agent_occurrence_count": source["agent_count"],
                    "JFDE_m": source["JFDE"], "JADE_m": source["JADE"],
                })
    write_csv(output / "source_data_strata.csv",
              ["method", "seed", "stratum", "scene_count",
               "agent_occurrence_count", "JFDE_m", "JADE_m"], strata_rows)
    write_csv(output / "source_data_fixed_cases.csv", list(cases[0].keys()), cases)
    if not args.skip_figures:
        plot_summary(summary, output)
        plot_cases(cases, bank_roots["epoch5"] / "seed_2035.npz",
                   bank_roots["candidate"] / "seed_2035.npz", output)
    print(json.dumps(summary["screening"], sort_keys=True))


if __name__ == "__main__":
    main()

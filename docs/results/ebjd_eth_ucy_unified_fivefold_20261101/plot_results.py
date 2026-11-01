#!/usr/bin/env python3
"""Create submission-ready figures for the EBJD unified five-fold report."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG")
ASSETS = ROOT / "RSJG_EBJD_assets"
FOLDS = ["ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2"]
COLORS = {
    "ETH": "#4477AA", "HOTEL": "#EE6677", "UNIV": "#228833",
    "ZARA1": "#CCBB44", "ZARA2": "#66CCEE", "MACRO": "#AA3377",
}
EXPERIMENTS = {
    "ETH": ASSETS / "experiments/eth_unified_coupling_bridge_v1_20261030_104912",
    "HOTEL": ASSETS / "experiments/hotel_unified_coupling_bridge_v1_20261030_211601",
    "UNIV": ASSETS / "experiments/univ_unified_coupling_bridge_v1_20261031_111541",
    "ZARA1": ASSETS / "experiments/zara1_unified_coupling_bridge_v1_20261028_214026",
    "ZARA2": ASSETS / "experiments/zara2_unified_coupling_bridge_v1_20261029_094715",
}


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def alignment_audit(fig, output_dir: Path, stem: str) -> None:
    scripts_dir = os.environ.get("NATURE_FIGURE_SCRIPTS")
    if not scripts_dir:
        raise RuntimeError("Set NATURE_FIGURE_SCRIPTS to the nature-figure scripts directory")
    sys.path.insert(0, scripts_dir)
    from audit_panel_alignment import require_matplotlib_panel_alignment

    require_matplotlib_panel_alignment(
        fig,
        json_out=output_dir / f"{stem}.alignment.json",
        overlay_svg=output_dir / f"{stem}.alignment_overlay.svg",
        tolerance_pt=1.5,
        strict=True,
    )


def save(fig, output_dir: Path, stem: str) -> None:
    alignment_audit(fig, output_dir, stem)
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def style() -> None:
    plt.rcParams.update({
        "font.family": "Liberation Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def metric_lookup(summary_rows: list[dict]) -> dict:
    return {(r["scope"], r["method"], r["metric"]): float(r["mean"]) for r in summary_rows}


def figure_vs_gdts(summary_rows: list[dict], output_dir: Path) -> None:
    lookup = metric_lookup(summary_rows)
    metrics = ["minADE", "minFDE", "JADE", "JFDE"]
    labels = FOLDS + ["MACRO"]
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.7), constrained_layout=True)
    for letter, metric, ax in zip("abcd", metrics, axes.flat):
        values = []
        for label in labels:
            scope = "MACRO_5FOLD" if label == "MACRO" else label
            values.append(lookup[(scope, "H", metric)] - lookup[(scope, "GDTS", metric)])
        x = np.arange(len(labels))
        ax.bar(x, values, color=[COLORS[l] for l in labels], width=0.72)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_xticks(x, labels, rotation=30, ha="right", rotation_mode="anchor")
        ax.set_ylabel("H − matched GDTS (m)")
        ax.set_title(f"{letter}  {metric}", loc="left", fontweight="bold")
        ax.text(0.99, 0.04, "lower is better", transform=ax.transAxes, ha="right", color="#555555")
    fig.suptitle("Unified EBJD: fold-wise change relative to matched GDTS", fontweight="bold")
    save(fig, output_dir, "fivefold_h_vs_gdts")


def figure_pairing(summary_rows: list[dict], output_dir: Path) -> None:
    lookup = metric_lookup(summary_rows)
    metrics = ["JADE", "JFDE"]
    labels = FOLDS + ["MACRO"]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), constrained_layout=True)
    for letter, metric, ax in zip("ab", metrics, axes):
        values = []
        for label in labels:
            scope = "MACRO_5FOLD" if label == "MACRO" else label
            values.append(lookup[(scope, "H", metric)] - lookup[(scope, "G", metric)])
        x = np.arange(len(labels))
        ax.bar(x, values, color=[COLORS[l] for l in labels], width=0.72)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_xticks(x, labels, rotation=30, ha="right", rotation_mode="anchor")
        ax.set_ylabel("H − G (m)")
        ax.set_title(f"{letter}  {metric}", loc="left", fontweight="bold")
        ax.text(0.99, 0.05, "negative = pairing gain", transform=ax.transAxes, ha="right", color="#555555")
    fig.suptitle("Learned reassignment improves joint metrics on every fold", fontweight="bold")
    save(fig, output_dir, "fivefold_pairing_joint_gain")


def figure_curves(output_dir: Path) -> None:
    curves = {fold: rows(root / "training/epoch_metrics.csv") for fold, root in EXPERIMENTS.items()}
    for row in curves["ZARA1"]:
        g = ast.literal_eval(row["G"])
        h = ast.literal_eval(row["H"])
        for metric in ["minADE", "minFDE", "JADE", "JFDE"]:
            row[f"G_{metric}"] = g[metric]
            row[f"H_{metric}"] = h[metric]
    selected = {"ETH": 19, "HOTEL": 21, "UNIV": 30, "ZARA1": 30, "ZARA2": 23}
    fig = plt.figure(figsize=(7.8, 5.1), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.18])
    axes = np.array([[fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
                     [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]])
    legend_ax = fig.add_subplot(grid[:, 2])
    legend_ax.set_axis_off()
    specs = [
        ("minADE", "G_minADE", "H_minADE"),
        ("JADE", "G_JADE", "H_JADE"),
        ("H JFDE", None, "H_JFDE"),
        ("training loss", None, "loss"),
    ]
    for letter, (title, g_key, h_key), ax in zip("abcd", specs, axes.flat):
        for fold in FOLDS:
            rs = curves[fold]
            epoch = [int(r["epoch"]) for r in rs]
            if g_key:
                ax.plot(epoch, [float(r[g_key]) for r in rs], color=COLORS[fold], alpha=0.45,
                        linewidth=1.0, linestyle="--")
            ax.plot(epoch, [float(r[h_key]) for r in rs], color=COLORS[fold], linewidth=1.25,
                    label=fold)
            ax.scatter([selected[fold]], [float(rs[selected[fold] - 1][h_key])], color=COLORS[fold],
                       edgecolor="white", linewidth=0.35, s=22, zorder=4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("m" if title != "training loss" else "loss")
        ax.set_title(f"{letter}  {title}", loc="left", fontweight="bold")
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    legend_ax.text(0.0, 0.78, "fold", color="#333333", fontweight="bold")
    for y, fold in zip(np.linspace(0.68, 0.32, len(FOLDS)), FOLDS):
        legend_ax.text(0.0, y, fold, color=COLORS[fold], fontweight="bold")
    fig.suptitle("Unified training trajectories; circles mark selected epochs", fontweight="bold")
    save(fig, output_dir, "fivefold_training_curves")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style()
    summary_rows = rows(args.analysis_dir / "FIVEFOLD_METRICS.csv")
    figure_vs_gdts(summary_rows, args.output_dir)
    figure_pairing(summary_rows, args.output_dir)
    figure_curves(args.output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create the two compact five-fold JMM collision figures."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

SKILL_SCRIPTS = Path("/home/ee615/.codex/skills/nature-figure/scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment


COLORS = {"GDTS": "#4C78A8", "G": "#F2CF5B", "H": "#E45756"}
FOLDS = ("ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2")

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.linewidth": 0.8, "legend.frameon": False,
})


def read_summary(path: Path) -> dict:
    output = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["subset"] == "all" and row["scope"] in FOLDS:
                output[(row["scope"], row["method"], row["metric"])] = (
                    float(row["mean_percent"]), float(row["sd_ddof1_percent"]))
    return output


def export(fig, stem: Path, labels: list[str]) -> None:
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, json_out=str(stem) + ".alignment.json",
        overlay_svg=str(stem) + ".alignment.svg", tolerance_pt=1.5,
        gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def methods_figure(data: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), constrained_layout=True)
    x = np.arange(len(FOLDS)); width = 0.25
    for axis, metric, label in zip(axes, ("CRJADE", "CRmean"), ("a", "b")):
        for offset, method in enumerate(("GDTS", "G", "H")):
            values = [data[(fold, method, metric)][0] for fold in FOLDS]
            errors = [data[(fold, method, metric)][1] for fold in FOLDS]
            axis.bar(x + (offset - 1) * width, values, width, yerr=errors,
                     color=COLORS[method], label=method, capsize=2, linewidth=0)
        axis.set_xticks(x, FOLDS)
        axis.set_ylabel(f"{metric} (%)")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, zorder=0)
        axis.text(-0.11, 1.03, label, transform=axis.transAxes, fontweight="bold",
                  fontsize=8, va="bottom")
    axes[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(1.04, 1.18))
    export(fig, output / "fivefold_cr_methods", ["a", "b"])


def delta_figure(data: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), constrained_layout=True)
    x = np.arange(len(FOLDS)); width = 0.34
    for axis, metric, label in zip(axes, ("CRJADE", "CRmean"), ("a", "b")):
        for offset, (reference, color) in enumerate((("G", "#72B7B2"), ("GDTS", "#E45756"))):
            values = [data[(fold, "H", metric)][0] - data[(fold, reference, metric)][0]
                      for fold in FOLDS]
            axis.bar(x + (offset - .5) * width, values, width, color=color,
                     label=f"H − {reference}", linewidth=0)
        axis.axhline(0, color="#333333", linewidth=0.8)
        axis.set_xticks(x, FOLDS)
        axis.set_ylabel(f"Δ{metric} (percentage points)")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, zorder=0)
        axis.text(-0.11, 1.03, label, transform=axis.transAxes, fontweight="bold",
                  fontsize=8, va="bottom")
        axis.text(.99, .02, "lower is better", transform=axis.transAxes,
                  ha="right", va="bottom", color="#555555", fontsize=6)
    axes[0].legend(ncol=2, loc="upper center", bbox_to_anchor=(1.04, 1.18))
    export(fig, output / "fivefold_cr_deltas", ["a", "b"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    data = read_summary(args.summary)
    methods_figure(data, args.output); delta_figure(data, args.output)


if __name__ == "__main__":
    main()

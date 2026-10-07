#!/usr/bin/env python3
"""Render the audit training-dynamics figure from frozen CSV source data."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 8.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.5,
    "legend.fontsize": 6.8,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.15,
})
import matplotlib.pyplot as plt

SKILL_SCRIPTS = Path("/home/ee615/.codex/skills/nature-figure/scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402


COLORS = {"A0": "#0072B2", "A1": "#D55E00"}
LINESTYLES = {3101: "-", 3102: "--", 3103: ":"}
LOG_MODE_COUNT = 1.3862943611198906
width_mm = 180.0
height_mm = 122.0


def number(value: str):
    if value == "" or value is None:
        return math.nan
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    with args.input.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            grouped[(raw["arm"], int(raw["training_seed"]))].append(raw)
    for rows in grouped.values():
        rows.sort(key=lambda item: int(float(item["epoch"])))

    figure_size_inches = (width_mm / 25.4, height_mm / 25.4)
    fig, axes = plt.subplots(2, 2, figsize=figure_size_inches, sharex=True)
    panels = axes.ravel()
    specifications = [
        ("loss_total", "Training objective", "Loss"),
        ("valid_JADE", "Selection validation", "JADE (m)"),
        ("dynamic_relation_entropy", "Candidate-conditioned relation", r"Entropy / $\log 4$"),
        ("dynamic_dominant_mode_fraction", "Aggregate dynamic usage", "Dominant-mode fraction"),
    ]
    for (arm, seed), rows in sorted(grouped.items()):
        epochs = [int(float(row["epoch"])) for row in rows]
        label = f"{arm}, seed {seed}" + (" (partial)" if len(rows) < 40 else "")
        for index, (field, _, _) in enumerate(specifications):
            values = [number(row.get(field, "")) for row in rows]
            if field == "dynamic_relation_entropy":
                values = [value / LOG_MODE_COUNT for value in values]
            panels[index].plot(epochs, values, color=COLORS[arm], linestyle=LINESTYLES[seed],
                               alpha=0.92, label=label)
        jade = [number(row.get("valid_JADE", "")) for row in rows]
        if jade:
            selected = min(range(len(jade)), key=lambda idx: (jade[idx],
                           number(rows[idx].get("valid_JFDE", "")), epochs[idx]))
            panels[1].scatter([epochs[selected]], [jade[selected]], s=18,
                              color=COLORS[arm], edgecolor="white", linewidth=0.45, zorder=4)

    for index, (axis, (_, title, ylabel)) in enumerate(zip(panels, specifications)):
        axis.set_title(title, loc="left", pad=4)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=2.5, width=0.65)
        axis.text(-0.13, 1.05, chr(ord("a") + index), transform=axis.transAxes,
                  fontsize=9, fontweight="bold", va="bottom", ha="left")
    for axis in axes[-1, :]:
        axis.set_xlabel("Epoch")
    panels[2].set_ylim(bottom=0)
    panels[3].set_ylim(0.2, 1.01)

    handles, labels = panels[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.005), handlelength=2.4, columnspacing=1.2)
    fig.subplots_adjust(left=0.105, right=0.985, top=0.94, bottom=0.155,
                        hspace=0.34, wspace=0.30)
    stem = args.output_dir / "training_dynamics"
    require_matplotlib_panel_alignment(
        fig,
        axes=list(panels),
        panel_ids=["a", "b", "c", "d"],
        row_groups=[["a", "b"], ["c", "d"]],
        column_groups=[["a", "c"], ["b", "d"]],
        json_out=str(stem.with_suffix(".alignment.json")),
        overlay_svg=str(stem.with_suffix(".alignment-overlay.svg")),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


if __name__ == "__main__":
    main()

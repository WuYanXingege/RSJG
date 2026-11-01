#!/usr/bin/env python3
"""Render compact, publication-oriented plots for training-seed repeats."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = {"minADE": "#0072B2", "minFDE": "#D55E00", "JADE": "#009E73", "JFDE": "#CC79A7"}


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, stem: Path) -> None:
    # Literal extensions keep the submission-format contract statically auditable.
    raster_dpi = 300
    for suffix, kwargs in ((".svg", {}), (".pdf", {}), (".png", {"dpi": raster_dpi}),
                           (".tiff", {"dpi": raster_dpi})):
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def require_alignment(fig: plt.Figure, out: Path, stem: str) -> None:
    scripts = os.environ.get("NATURE_FIGURE_SCRIPTS")
    if not scripts:
        return
    sys.path.insert(0, scripts)
    from audit_panel_alignment import require_matplotlib_panel_alignment
    require_matplotlib_panel_alignment(
        fig, json_out=out / f"{stem}.alignment.json", strict=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", required=True)
    args = parser.parse_args()
    out = Path(args.summary_dir).resolve() / "figures"
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "font.family": "sans-serif",
                         "font.sans-serif": ["Liberation Sans", "DejaVu Sans"]})
    seeds = (3101, 3102, 3103)

    edge = read_rows(out / "hotel_edge_deltas.csv")
    fig, ax = plt.subplots(figsize=(3.5, 2.45), constrained_layout=True)
    x, width = np.arange(3), 0.34
    for offset, metric in zip((-width / 2, width / 2), ("minADE", "minFDE")):
        values = [float(next(r["H_minus_GDTS"] for r in edge
                             if int(r["training_seed"]) == seed and r["metric"] == metric))
                  for seed in seeds]
        ax.bar(x + offset, values, width, label=metric, color=COLORS[metric])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, [str(seed) for seed in seeds])
    ax.set(xlabel="Training seed", ylabel="H − matched GDTS (m)",
           title="HOTEL marginal metric differences")
    ax.legend(frameon=False, ncol=2)
    save(fig, out / "hotel_edge_deltas")

    joint = read_rows(out / "pairing_joint_deltas.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55), constrained_layout=True)
    for ax, fold in zip(axes, ("HOTEL", "ZARA2")):
        for metric in ("JADE", "JFDE"):
            values = [float(next(r["H_minus_G"] for r in joint
                                 if r["fold"] == fold and int(r["training_seed"]) == seed
                                 and r["metric"] == metric)) for seed in seeds]
            ax.plot(seeds, values, marker="o", color=COLORS[metric], label=metric)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(seeds); ax.set(xlabel="Training seed", title=fold)
    axes[0].set_ylabel("H − G (m; lower is better)")
    axes[1].legend(frameon=False)
    require_alignment(fig, out, "pairing_joint_deltas")
    save(fig, out / "pairing_joint_deltas")

    distribution = read_rows(out / "training_seed_distributions.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.7), constrained_layout=True)
    for ax, metric in zip(axes.flat, ("minADE", "minFDE", "JADE", "JFDE")):
        for xpos, fold in enumerate(("HOTEL", "ZARA2")):
            values = [float(next(r["H"] for r in distribution
                                 if r["fold"] == fold and int(r["training_seed"]) == seed
                                 and r["metric"] == metric)) for seed in seeds]
            baseline = float(next(r["GDTS"] for r in distribution
                                  if r["fold"] == fold and r["metric"] == metric))
            ax.scatter(np.full(3, xpos) + np.array([-0.07, 0, 0.07]), values,
                       color=COLORS[metric], s=26)
            ax.hlines(baseline, xpos - 0.18, xpos + 0.18,
                      color="black", linestyles="--", lw=1)
        ax.set_xticks((0, 1), ("HOTEL", "ZARA2")); ax.set_ylabel("m"); ax.set_title(metric)
    fig.suptitle("H across training seeds (dashed: fixed matched GDTS)", fontsize=10)
    require_alignment(fig, out, "training_seed_distributions")
    save(fig, out / "training_seed_distributions")

    qa = {"figures": ["hotel_edge_deltas", "pairing_joint_deltas",
                       "training_seed_distributions"],
          "formats": ["svg", "pdf", "png", "tiff"], "status": "RENDERED"}
    scripts = os.environ.get("NATURE_FIGURE_SCRIPTS")
    if scripts:
        failures = []
        source_check = subprocess.run(
            [sys.executable, str(Path(scripts) / "validate_figure.py"), __file__,
             "--backend", "python", "--strict"], capture_output=True, text=True)
        if source_check.returncode:
            failures.append({"command": source_check.args,
                             "stderr": source_check.stderr[-2000:]})
        for stem in qa["figures"]:
            commands = (
                [sys.executable, str(Path(scripts) / "audit_pdf_text.py"), str(out / f"{stem}.pdf")],
                [sys.executable, str(Path(scripts) / "audit_figure_collisions.py"), str(out / f"{stem}.pdf"),
                 "--json-out", str(out / f"{stem}.collisions.json")],
            )
            for command in commands:
                result = subprocess.run(command, capture_output=True, text=True)
                if result.returncode:
                    failures.append({"command": command, "stderr": result.stderr[-2000:]})
        qa["failures"] = failures
        qa["status"] = "PASS" if not failures else "QA_FAILED"
        if failures:
            (out / "FIGURE_QA.json").write_text(json.dumps(qa, indent=2) + "\n")
            raise RuntimeError("figure QA failed; see FIGURE_QA.json")
    else:
        qa["status"] = "QA_NOT_RUN_NO_SKILL_PATH"
    (out / "FIGURE_QA.json").write_text(json.dumps(qa, indent=2) + "\n")


if __name__ == "__main__":
    main()

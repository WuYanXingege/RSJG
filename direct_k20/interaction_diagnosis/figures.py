"""Render anonymous paired S0 trajectory cases from frozen test predictions."""

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import ScaledTranslation
import numpy as np


CASES = {
    "highest_L": "Highest mean individual endpoint error",
    "highest_G": "Largest shared-slot gap",
    "median_J": "Nearest the median joint endpoint error",
}
OBSERVED = "#82919D"
TRUTH = "#202B34"
JOINT = "#BD511C"
INDIVIDUAL = "#106D9C"
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8,
    "axes.linewidth": 0.65,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.labelcolor": TRUTH,
    "xtick.color": TRUTH,
    "ytick.color": TRUTH,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
}


def select_cases(rows):
    """Pick exact maxima and the nearest median, with anonymous ordinal ties.

    Each rule ranks the complete fold independently. A window can therefore
    satisfy two rules. The caller supplies the S0 endpoint row for each window.
    """
    candidates = list(rows)
    if not candidates:
        raise ValueError("case selection requires a complete nonempty fold")
    numbers = [row["window"] for row in candidates]
    if len(set(numbers)) != len(candidates):
        raise ValueError("case selection expects one row per anonymous window")
    for metric in ("J", "L", "G"):
        if not np.isfinite([float(row[f"g0_{metric}_m"]) for row in candidates]).all():
            raise ValueError("case selection metric must be finite")
    median = float(np.median([float(row["g0_J_m"]) for row in candidates]))
    return [
        ("highest_L", min(candidates, key=lambda row: (-float(row["g0_L_m"]), row["window"]))),
        ("highest_G", min(candidates, key=lambda row: (-float(row["g0_G_m"]), row["window"]))),
        ("median_J", min(candidates, key=lambda row: (abs(float(row["g0_J_m"]) - median), row["window"]))),
    ]


def _coordinates(observed, *paths):
    # Center on the observed scene and align its average motion to local x.
    origin = observed[:, -1].mean(axis=0)
    heading = (observed[:, -1] - observed[:, -4]).mean(axis=0)
    magnitude = np.linalg.norm(heading)
    if magnitude < 1e-8:
        heading = np.array([1.0, 0.0])
    else:
        heading = heading / magnitude
    right = np.array([-heading[1], heading[0]])
    rotation = np.stack((heading, right), axis=1)
    return [(points - origin) @ rotation for points in (observed, *paths)]


def _line(axis, series, color, *, width, style="solid", alpha=1.0, zorder=2):
    artist = LineCollection(series, colors=color, linewidths=width, linestyles=style,
                            capstyle="round", alpha=alpha, zorder=zorder)
    axis.add_collection(artist)


def _panel(axis, history, future, path, x_limits, y_limits, color, style, marker):
    n = len(history)
    width = 1.2 if n <= 8 else 0.82 if n <= 24 else 0.65
    alpha = 0.94 if n <= 24 else 0.82
    axis.set(xlim=x_limits, ylim=y_limits, xlabel="Local x (m)")
    axis.set_aspect("equal", adjustable="box")
    axis.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    axis.tick_params(axis="both", width=0.65, length=2.5, pad=3, labelsize=8)
    axis.grid(color="#E8EDF0", linewidth=0.5)
    axis.set_axisbelow(True)
    _line(axis, history, OBSERVED, width=width, alpha=alpha)
    observed_end = history[:, -1:]
    _line(axis, np.concatenate((observed_end, future), axis=1), TRUTH, width=width,
          alpha=alpha, zorder=3)
    _line(axis, np.concatenate((observed_end, path), axis=1), color, width=width,
          style=style, alpha=alpha, zorder=4)
    axis.scatter(*observed_end[:, 0].T, s=10 if n <= 8 else 6, marker="o",
                 facecolors="white", edgecolors=OBSERVED, linewidths=0.65, zorder=5)
    axis.scatter(*future[:, -1].T, s=15 if n <= 8 else 10, marker="+",
                 color=TRUTH, linewidths=0.8, zorder=6)
    axis.scatter(*path[:, -1].T, s=9 if n <= 8 else 6, marker=marker,
                 color=color, linewidths=0, zorder=7)


def render_case(path, *, fold, case, window, observed, truth, path0, selected_metrics=None,
                qa_dir=None, require_matplotlib_panel_alignment=None):
    """Make one PNG from seed-2035 S0 trajectories without exporting coordinates.

    Both panels share a local metric coordinate frame and the same limits.
    The left panel uses one JFDE-winning slot. The right draws each person's
    nearest endpoint trajectory. The exact coordinates and PDF QA views stay
    on the host.
    """
    if fold.lower() not in ("hotel", "univ") or case not in CASES:
        raise ValueError("unknown fold or case selection")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,48}", str(window)):
        raise ValueError("expected an anonymous window ordinal")
    target = Path(path)
    if target.suffix.lower() != ".png":
        raise ValueError("public case plots must be raster PNG")
    history, future, trajectories = [np.asarray(value, dtype=np.float64)
                                     for value in (observed, truth, path0)]
    n = len(history)
    if (n < 1 or history.shape != (n, 8, 2) or future.shape != (n, 12, 2)
            or trajectories.shape != (20, n, 12, 2)
            or not all(np.isfinite(value).all() for value in (history, future, trajectories))):
        raise ValueError("case requires finite packed K20 trajectories for every effective agent")
    errors = np.linalg.norm(trajectories[:, :, -1] - future[None, :, -1], axis=-1)
    winner = int(errors.mean(axis=1).argmin())
    personal = errors.argmin(axis=0)
    joint = float(errors[winner].mean())
    marginal = float(errors[personal, np.arange(n)].mean())
    gap = joint - marginal
    if selected_metrics is not None:
        for metric, value in (("J", joint), ("L", marginal), ("G", gap)):
            if not np.isclose(float(selected_metrics[f"g0_{metric}_m"]), value, atol=1e-5, rtol=1e-6):
                raise ValueError(f"figure metric {metric} differs from the selected S0 case")
    observation, target_future, joint_path, individual_path = _coordinates(
        history, future, trajectories[winner], trajectories[personal, np.arange(n)],
    )
    bounds = np.concatenate([points.reshape(-1, 2)
                             for points in (observation, target_future, joint_path, individual_path)])
    lower, upper = bounds.min(axis=0), bounds.max(axis=0)
    span = upper - lower
    padded_span = span + 2 * np.maximum(0.2, 0.07 * span)
    # Give nearly collinear paths room for metre ticks at equal scale.
    half_span = 0.5 * np.maximum(padded_span, 0.55 * padded_span.max())
    center = 0.5 * (lower + upper)
    x_limits, y_limits = zip(center - half_span, center + half_span)
    target.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(7.1, 4.5), facecolor="white")
        try:
            left, right = fig.subplots(1, 2, sharex=True, sharey=True, gridspec_kw={
                "left": 0.092, "right": 0.944, "bottom": 0.182, "top": 0.768, "wspace": 0.20155,
            })
            _panel(left, observation, target_future, joint_path, x_limits, y_limits, JOINT, "solid", "s")
            _panel(right, observation, target_future, individual_path, x_limits, y_limits, INDIVIDUAL,
                   (0, (3.4, 1.6)), "D")
            left.set_ylabel("Local y (m)")
            for axis, letter, label, color in ((left, "a", "One joint slot", JOINT),
                                                (right, "b", "Closest endpoint per person", INDIVIDUAL)):
                base = axis.transAxes + ScaledTranslation(0, 5 / 72, fig.dpi_scale_trans)
                axis.text(0, 1, letter, transform=base, fontsize=9, fontweight="bold",
                          color=TRUTH, va="baseline")
                axis.text(0, 1, label,
                          transform=base + ScaledTranslation(12 / 72, 0, fig.dpi_scale_trans),
                          fontsize=9, fontweight="bold", color=color, va="baseline")
            fig.text(0.09, 0.956, f"{fold.upper()}  |  {CASES[case]}",
                     fontsize=13, fontweight="bold", va="top", color=TRUTH)
            fig.text(0.09, 0.897,
                     f"{window}   ·   N = {n}   ·   S0, seed 2035   ·   "
                     f"J {joint:.3f} m   L {marginal:.3f} m   G {gap:.3f} m",
                     fontsize=8.5, va="top", color=TRUTH)
            handles = [
                Line2D([], [], color=OBSERVED, marker="o", markerfacecolor="white", markersize=3,
                       label="Observed"),
                Line2D([], [], color=TRUTH, marker="+", markersize=5, label="Ground truth"),
                Line2D([], [], color=JOINT, marker="s", markersize=3, label="Joint JFDE winner"),
                Line2D([], [], color=INDIVIDUAL, linestyle=(0, (3.4, 1.6)), marker="D",
                       markersize=3, label="Person minFDE winner"),
            ]
            fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.031),
                       ncol=4, handlelength=2.1, columnspacing=1.55, fontsize=8)
            fig.canvas.draw()
            boxes = np.array([axis.get_position().bounds for axis in (left, right)])
            # Equal boxes and y alignment after the aspect constraint is applied.
            if not np.allclose(boxes[0, 1:], boxes[1, 1:], rtol=0, atol=1e-10):
                raise ValueError("rendered trajectory panels do not share a common metric scale")
            if qa_dir is not None:
                private = Path(qa_dir)
                private.mkdir(parents=True, exist_ok=True)
                if require_matplotlib_panel_alignment is not None:
                    require_matplotlib_panel_alignment(
                        fig, json_out=private / f"{target.stem}.alignment.json",
                        strict=True, tolerance_pt=1.5, gutter_tolerance_pt=1.5,
                        require_panel_labels=True,
                    )
                fig.savefig(private / f"{target.stem}.pdf", metadata={
                    "CreationDate": None, "ModDate": None,
                })
                fig.savefig(private / f"{target.stem}.svg")
            fig.savefig(target, dpi=300, metadata={"Software": "Matplotlib"})
        finally:
            plt.close(fig)
    return dict(fold=fold.lower(), case=case, window=str(window), n=n,
                J_m=joint, L_m=marginal, G_m=gap, plot=target.name,
                seed=2035, coordinates="translated and rotated local metric frame",
                figure_model="S0")

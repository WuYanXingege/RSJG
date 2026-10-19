"""Pure helpers for the frozen ETH endpoint-proposal comparison."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def pixel_to_local_m(
    pixels: np.ndarray, *, width: int = 256, height: int = 256,
    crop_width_m: float = 32.0,
) -> np.ndarray:
    """Map align-corners heatmap coordinates ``(x,y)`` to local metres."""
    value = np.asarray(pixels, dtype=np.float64)
    if value.shape[-1] != 2 or width < 2 or height < 2:
        raise ValueError("pixels must end in (x,y), and map sides must exceed one")
    answer = value.copy()
    answer[..., 0] = (2.0 * value[..., 0] / (width - 1) - 1.0) * crop_width_m / 2.0
    answer[..., 1] = (2.0 * value[..., 1] / (height - 1) - 1.0) * crop_width_m / 2.0
    return answer


def pixel_to_world_m(
    pixels: np.ndarray, centers_m: np.ndarray, **kwargs,
) -> np.ndarray:
    local = pixel_to_local_m(pixels, **kwargs)
    center = np.asarray(centers_m, dtype=np.float64)
    while center.ndim < local.ndim:
        center = np.expand_dims(center, axis=0)
    return local + center


def split_native_ttst(samples: np.ndarray, num_clusters: int = 20):
    """Return native cluster branches and the appended argmax trunk."""
    value = np.asarray(samples)
    if value.shape[0] != num_clusters + 1:
        raise ValueError(
            f"native TTST must return {num_clusters} clusters plus one argmax")
    return value[:num_clusters], value[num_clusters]


def effective_unique(points: np.ndarray, tolerance_m: float = 1e-4) -> int:
    value = np.asarray(points, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError("points must be [K,2]")
    quantized = np.rint(value / tolerance_m).astype(np.int64)
    return int(np.unique(quantized, axis=0).shape[0])


@dataclass(frozen=True)
class AgentRecord:
    route: str
    window_key: str
    scene_id: str
    source_sequence: str
    agent_id: str
    agents_in_scene: int
    gt_endpoint: np.ndarray
    last_observed: np.ndarray
    candidates: np.ndarray
    argmax: np.ndarray | None = None

    def row(self) -> dict:
        candidates = np.asarray(self.candidates, dtype=np.float64)
        target = np.asarray(self.gt_endpoint, dtype=np.float64)
        center = np.asarray(self.last_observed, dtype=np.float64)
        error = np.linalg.norm(candidates - target[None], axis=-1)
        radius = np.linalg.norm(candidates - center[None], axis=-1)
        pairwise = np.linalg.norm(
            candidates[:, None] - candidates[None, :], axis=-1)
        triangle = pairwise[np.triu_indices(len(candidates), 1)]
        relative = target - center
        out_endpoint = bool(np.any(np.abs(relative) > 16.0 + 1e-9))
        row = {
            "route": self.route,
            "window_key": self.window_key,
            "scene_id": self.scene_id,
            "source_sequence": self.source_sequence,
            "agent_id": self.agent_id,
            "agents_in_scene": int(self.agents_in_scene),
            "stratum_agents": "N=1" if self.agents_in_scene == 1 else "N>=2",
            "crop_stratum": "out_of_crop" if out_endpoint else "in_crop",
            "gt_x_m": float(target[0]), "gt_y_m": float(target[1]),
            "center_x_m": float(center[0]), "center_y_m": float(center[1]),
            "minfde20_m": float(error.min()),
            "recall_0p5": float(error.min() <= 0.5),
            "recall_1p0": float(error.min() <= 1.0),
            "single_candidate_error_mean_m": float(error.mean()),
            "proposal_rms_radius_m": float(np.sqrt(np.mean(radius ** 2))),
            "pairwise_distance_mean_m": float(triangle.mean()),
            "effective_unique_candidates": effective_unique(candidates),
        }
        if self.argmax is not None:
            row["argmax_error_m"] = float(np.linalg.norm(
                np.asarray(self.argmax, dtype=np.float64) - target))
        return row


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot summarize no agent rows")
    values = np.asarray([row["minfde20_m"] for row in rows], dtype=np.float64)
    scalar_means = (
        "recall_0p5", "recall_1p0", "single_candidate_error_mean_m",
        "proposal_rms_radius_m", "pairwise_distance_mean_m",
        "effective_unique_candidates", "argmax_error_m",
    )
    result = {
        "agents": len(rows),
        "minFDE20_mean_m": float(values.mean()),
        "minFDE20_median_m": float(np.median(values)),
        "minFDE20_p90_m": float(np.quantile(values, 0.90)),
        "minFDE20_p95_m": float(np.quantile(values, 0.95)),
    }
    for key in scalar_means:
        present = [float(row[key]) for row in rows if key in row]
        if present:
            result[key + "_mean"] = float(np.mean(present))
    return result


def paired_delta(rows_left: list[dict], rows_right: list[dict]) -> dict:
    identity = lambda row: (row["window_key"], row["agent_id"])
    left = {identity(row): row for row in rows_left}
    right = {identity(row): row for row in rows_right}
    if left.keys() != right.keys():
        raise ValueError("paired routes have different agent identities")
    delta = np.asarray([
        right[key]["minfde20_m"] - left[key]["minfde20_m"]
        for key in left
    ], dtype=np.float64)
    return {
        "definition": "right_minus_left; negative favours right",
        "pairs": int(delta.size), "mean_m": float(delta.mean()),
        "median_m": float(np.median(delta)),
        "p10_m": float(np.quantile(delta, 0.10)),
        "p90_m": float(np.quantile(delta, 0.90)),
        "right_better_fraction": float(np.mean(delta < 0)),
        "equal_fraction": float(np.mean(delta == 0)),
    }


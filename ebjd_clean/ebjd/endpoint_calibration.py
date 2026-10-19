"""Read-only endpoint-spread calibration for fixed trajectory banks.

The transform changes only the endpoint deviation around the per-agent world
mean.  It preserves world/agent pairing, the candidate mean trajectory, and
the bridge residual relative to each candidate's linear endpoint bridge.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np

from .diagnostic_bank import BankScene, hard_scene_statistics


DEFAULT_K = (1, 2, 4, 8, 12, 16, 20)


def calibrate_endpoint_spread(
    predictions: np.ndarray, factor: float,
) -> np.ndarray:
    """Apply the fixed-bank endpoint-spread transform in float64.

    ``predictions`` is ``[P,N,12,2]``.  A factor of one is the literal
    identity.  No ground truth is accepted by this function.
    """
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim != 4 or values.shape[-2:] != (12, 2):
        raise ValueError("predictions must be [P,N,12,2]")
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("endpoint spread factor must be finite and positive")
    endpoints = values[..., -1, :]
    endpoint_mean = endpoints.mean(axis=0, keepdims=True)
    deviation = endpoints - endpoint_mean
    ramp = (np.arange(1, 13, dtype=np.float64) / 12.0)[None, None, :, None]
    return values + ramp * (float(factor) - 1.0) * deviation[:, :, None, :]


def bridge_residual(
    predictions: np.ndarray, history: np.ndarray,
) -> np.ndarray:
    """Return residual around the straight bridge to each world endpoint."""
    values = np.asarray(predictions, dtype=np.float64)
    observed = np.asarray(history, dtype=np.float64)
    start = observed[:, -1, :]
    endpoints = values[..., -1, :]
    ramp = (np.arange(1, 13, dtype=np.float64) / 12.0)[None, None, :, None]
    linear = start[None, :, None, :] + ramp * (
        endpoints[:, :, None, :] - start[None, :, None, :])
    return values - linear


def _metric_summary(
    scene_values: Iterable[tuple[np.ndarray, np.ndarray]],
) -> dict[str, float | int]:
    agent_ade: list[np.ndarray] = []
    agent_fde: list[np.ndarray] = []
    scene_marginal_ade: list[float] = []
    scene_marginal_fde: list[float] = []
    jade: list[float] = []
    jfde: list[float] = []
    for predictions, future in scene_values:
        stats = hard_scene_statistics(predictions, future)
        agent_ade.append(np.asarray(stats["agent_minade"]))
        agent_fde.append(np.asarray(stats["agent_minfde"]))
        scene_marginal_ade.append(float(stats["scene_marginal_ade"]))
        scene_marginal_fde.append(float(stats["scene_marginal_fde"]))
        jade.append(float(stats["JADE"]))
        jfde.append(float(stats["JFDE"]))
    if not jade:
        return {
            "minADE": float("nan"), "minFDE": float("nan"),
            "JADE": float("nan"), "JFDE": float("nan"),
            "scene_weighted_minADE": float("nan"),
            "scene_weighted_minFDE": float("nan"),
            "coordination_gap_ADE": float("nan"),
            "coordination_gap_FDE": float("nan"),
            "agent_count": 0, "scene_count": 0,
        }
    ade = np.concatenate(agent_ade)
    fde = np.concatenate(agent_fde)
    result: dict[str, float | int] = {
        "minADE": float(ade.mean()), "minFDE": float(fde.mean()),
        "JADE": float(np.mean(jade)), "JFDE": float(np.mean(jfde)),
        "scene_weighted_minADE": float(np.mean(scene_marginal_ade)),
        "scene_weighted_minFDE": float(np.mean(scene_marginal_fde)),
        "agent_count": int(ade.size), "scene_count": len(jade),
    }
    result["coordination_gap_ADE"] = float(
        result["JADE"] - result["scene_weighted_minADE"])
    result["coordination_gap_FDE"] = float(
        result["JFDE"] - result["scene_weighted_minFDE"])
    return result


def summarize_factor(
    scenes: list[BankScene], factor: float, k_values: Iterable[int] = DEFAULT_K,
) -> dict:
    """Return formal metrics, strata, diversity, invariants, and K curves."""
    transformed: list[tuple[BankScene, np.ndarray]] = [
        (scene, calibrate_endpoint_spread(scene.predictions, factor))
        for scene in scenes
    ]
    full = _metric_summary((prediction, scene.future)
                           for scene, prediction in transformed)
    strata = {
        "N=1": _metric_summary(
            (prediction, scene.future) for scene, prediction in transformed
            if scene.agents == 1),
        "N>=2": _metric_summary(
            (prediction, scene.future) for scene, prediction in transformed
            if scene.agents >= 2),
    }

    center_bias: list[np.ndarray] = []
    squared_radius_sum = 0.0
    radius_count = 0
    pairwise: list[np.ndarray] = []
    candidate_fde: list[np.ndarray] = []
    candidate_mean_max_abs = 0.0
    bridge_max_abs = 0.0
    for scene, prediction in transformed:
        original = scene.predictions.astype(np.float64)
        original_mean = original.mean(axis=0)
        candidate_mean_max_abs = max(
            candidate_mean_max_abs,
            float(np.max(np.abs(prediction.mean(axis=0) - original_mean))))
        bridge_max_abs = max(
            bridge_max_abs,
            float(np.max(np.abs(
                bridge_residual(prediction, scene.history)
                - bridge_residual(original, scene.history)))))
        endpoints = prediction[..., -1, :]
        centers = endpoints.mean(axis=0)
        center_bias.append(np.linalg.norm(
            centers - scene.future[:, -1, :].astype(np.float64), axis=-1))
        deviations = endpoints - centers[None]
        squared_radius_sum += float(np.square(deviations).sum())
        radius_count += int(endpoints.shape[0] * endpoints.shape[1])
        candidate_fde.append(np.linalg.norm(
            endpoints - scene.future[None, :, -1, :], axis=-1).reshape(-1))
        pair_indices = list(combinations(range(endpoints.shape[0]), 2))
        pairwise.append(np.concatenate([
            np.linalg.norm(endpoints[left] - endpoints[right], axis=-1)
            for left, right in pair_indices
        ]))
    all_bias = np.concatenate(center_bias)
    all_pairwise = np.concatenate(pairwise)
    all_candidate_fde = np.concatenate(candidate_fde)
    diversity = {
        "endpoint_center_bias_mean_m": float(all_bias.mean()),
        "endpoint_center_bias_rms_m": float(np.sqrt(np.square(all_bias).mean())),
        "endpoint_rms_radius_m": float(
            np.sqrt(squared_radius_sum / radius_count)),
        "endpoint_pairwise_distance_mean_m": float(all_pairwise.mean()),
        "single_candidate_FDE_mean_m": float(all_candidate_fde.mean()),
    }

    curve = []
    for k in k_values:
        k = int(k)
        if k < 1 or k > int(scenes[0].predictions.shape[0]):
            raise ValueError(f"K={k} is outside the bank world count")
        metrics = _metric_summary(
            (prediction[:k], scene.future) for scene, prediction in transformed)
        curve.append({"K": k, **metrics})
    return {
        "factor": float(factor), "metrics": full, "strata": strata,
        "coverage": diversity, "k_curve": curve,
        "invariants": {
            "candidate_mean_max_abs_m": candidate_mean_max_abs,
            "bridge_residual_max_abs_m": bridge_max_abs,
            "world_order_preserved": True,
            "agent_order_preserved": True,
        },
    }


def occurrence_changes(
    scenes: list[BankScene], factor: float,
) -> list[dict]:
    """Per-agent paired changes for deterministic case selection."""
    rows: list[dict] = []
    for scene_index, scene in enumerate(scenes):
        before = hard_scene_statistics(scene.predictions, scene.future)
        after_prediction = calibrate_endpoint_spread(scene.predictions, factor)
        after = hard_scene_statistics(after_prediction, scene.future)
        endpoints = after_prediction[..., -1, :]
        center = endpoints.mean(axis=0)
        for agent_index, agent_id in enumerate(scene.agent_ids.tolist()):
            before_fde = float(before["agent_minfde"][agent_index])
            after_fde = float(after["agent_minfde"][agent_index])
            rows.append({
                "scene_index": scene_index,
                "agent_index": agent_index,
                "agent_id": str(agent_id),
                "window_key": scene.window_key,
                "group": "N=1" if scene.agents == 1 else "N>=2",
                "before_minFDE_m": before_fde,
                "after_minFDE_m": after_fde,
                "delta_minFDE_m": after_fde - before_fde,
                "endpoint_center_bias_m": float(np.linalg.norm(
                    center[agent_index] - scene.future[agent_index, -1])),
                "max_endpoint_radius_m": float(np.linalg.norm(
                    endpoints[:, agent_index] - center[agent_index], axis=-1).max()),
            })
    return rows

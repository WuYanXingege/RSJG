#!/usr/bin/env python3
"""CPU-only HOTEL epoch-45 bank parity and frozen-output diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from ebjd.diagnostic_bank import (
    SeedBank, assert_observation_parity, atomic_json, hard_scene_statistics,
    normalized_softmin, reorder_like, sha256_file, stable_seed,
    summarize_scenes,
)


SEEDS = (2035, 2036, 2037, 2038, 2039)
KS = (4, 8, 12, 16, 20)
METHODS = ("EBJD", "GDTS")
METRICS = ("minADE", "minFDE", "JADE", "JFDE")
COLORS = {"EBJD": "#3B6FB6", "GDTS": "#777777"}


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def quantiles(values: list[float] | np.ndarray) -> dict[str, float]:
    value = np.asarray(values, dtype=np.float64)
    if value.size == 0:
        return {name: float("nan") for name in ("mean", "q05", "q25", "q50", "q75", "q95")}
    return {
        "mean": float(value.mean()),
        "q05": float(np.quantile(value, 0.05)),
        "q25": float(np.quantile(value, 0.25)),
        "q50": float(np.quantile(value, 0.50)),
        "q75": float(np.quantile(value, 0.75)),
        "q95": float(np.quantile(value, 0.95)),
    }


def load_banks(root: Path):
    banks = {method: {} for method in METHODS}
    manifest_receipts = []
    maximum_history = maximum_future = 0.0
    for seed in SEEDS:
        ebjd_bank = SeedBank(root / "ebjd" / f"seed_{seed}" / "manifest.json")
        gdts_bank = SeedBank(root / "gdts" / f"seed_{seed}" / "manifest.json")
        if set(ebjd_bank.identities()) != set(gdts_bank.identities()):
            raise RuntimeError(f"model bank window sets differ for seed {seed}")
        for bank in (ebjd_bank, gdts_bank):
            manifest_receipts.append({
                "model": bank.payload["model"], "seed": seed,
                "path": str(bank.path), "sha256": sha256_file(bank.path),
                "scene_count": bank.payload["scene_count"],
                "agent_occurrences": bank.payload["agent_occurrences"],
                "forward_calls": bank.payload["forward_calls"],
                "elapsed_seconds": bank.payload["elapsed_seconds"],
                "checkpoint_sha256": bank.payload["checkpoint_sha256"],
                "config_sha256": bank.payload["config_sha256"],
                "source_manifest_sha256": bank.payload["source_manifest_sha256"],
                "execution_commit": bank.payload["execution_commit"],
                "inference_precision": bank.payload["inference_precision"],
                "autocast": bank.payload["autocast"],
                "sampler": bank.payload["sampler"],
            })
        order = ebjd_bank.identities()
        ebjd_scenes, gdts_scenes = {}, {}
        for identity in order:
            ebjd = ebjd_bank.load(identity)
            gdts = reorder_like(ebjd, gdts_bank.load(identity))
            differences = assert_observation_parity(ebjd, gdts, 1e-5)
            maximum_history = max(maximum_history, differences["history_max_abs_m"])
            maximum_future = max(maximum_future, differences["future_max_abs_m"])
            ebjd_scenes[identity] = ebjd
            gdts_scenes[identity] = gdts
        banks["EBJD"][seed] = ebjd_scenes
        banks["GDTS"][seed] = gdts_scenes
    return banks, manifest_receipts, {
        "history_max_abs_m": maximum_history,
        "future_max_abs_m": maximum_future,
        "tolerance_m": 1e-5,
    }


def parity(banks, ebjd_reference: Path, gdts_reference: Path):
    ebjd_payload = json.loads(ebjd_reference.read_text(encoding="utf-8"))
    gdts_payload = json.loads(gdts_reference.read_text(encoding="utf-8"))
    rows, maximum = [], 0.0
    summaries = {method: {} for method in METHODS}
    for seed in SEEDS:
        for method in METHODS:
            summary = summarize_scenes(banks[method][seed].values())
            summaries[method][seed] = summary
            if method == "EBJD":
                reference = next(
                    item["metrics"] for item in ebjd_payload["per_seed"]
                    if int(item["seed"]) == seed)
                names = {name: name for name in METRICS}
            else:
                reference = next(
                    item["metrics"] for item in gdts_payload["runs"]
                    if int(item["evaluation_seed"]) == seed)
                names = {
                    "minADE": "minADE@K", "minFDE": "minFDE@K",
                    "JADE": "JADE", "JFDE": "JFDE",
                }
            for name in METRICS:
                difference = abs(float(summary[name]) - float(reference[names[name]]))
                maximum = max(maximum, difference)
                rows.append({
                    "model": method, "seed": seed, "metric": name,
                    "bank_replay": summary[name],
                    "frozen_reference": reference[names[name]],
                    "absolute_difference_m": difference,
                    "passed_1e-6": difference <= 1e-6,
                })
    return summaries, rows, {
        "status": "PASSED" if maximum <= 1e-6 else "FAILED",
        "maximum_absolute_difference_m": maximum,
        "threshold_m": 1e-6,
        "ebjd_reference_sha256": sha256_file(ebjd_reference),
        "gdts_reference_sha256": sha256_file(gdts_reference),
    }


def motion_metadata(scene):
    dt = float(scene.dt_seconds)
    history = scene.history.astype(np.float64)
    future = scene.future.astype(np.float64)
    history_velocity = (
        (history[:, -1] - history[:, -2]) / dt
        + (history[:, -2] - history[:, -3]) / dt) / 2.0
    history_speed = np.linalg.norm(history_velocity, axis=-1)
    full_future = np.concatenate((history[:, -1:, :], future), axis=1)
    segment_speed = np.linalg.norm(np.diff(full_future, axis=1), axis=-1) / dt
    future_speed = segment_speed.mean(axis=1)
    net_velocity = (future[:, -1] - history[:, -1]) / (12 * dt)
    net_speed = np.linalg.norm(net_velocity, axis=-1)
    dot = (history_velocity * net_velocity).sum(axis=-1)
    denominator = history_speed * net_speed
    cosine = np.divide(dot, denominator, out=np.ones_like(dot), where=denominator > 0)
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    valid_direction = (history_speed >= 0.1) & (net_speed >= 0.1)
    stop = future_speed <= 0.1
    deceleration = (history_speed >= 0.3) & (future_speed <= 0.5 * history_speed)
    turn = valid_direction & (angle >= 30)
    straight = valid_direction & (angle <= 15) & ~stop & ~deceleration
    ratio = np.divide(
        future_speed, history_speed, out=np.full_like(future_speed, np.nan),
        where=history_speed > 1e-12)

    positions = history[:, -1]
    has_edge = False
    for first in range(scene.agents):
        for second in range(first + 1, scene.agents):
            relative_position = positions[first] - positions[second]
            if np.linalg.norm(relative_position) > 6.0:
                continue
            relative_velocity = history_velocity[first] - history_velocity[second]
            speed_squared = float(np.dot(relative_velocity, relative_velocity))
            time_to_cpa = 0.0 if speed_squared <= 1e-12 else float(np.clip(
                -np.dot(relative_position, relative_velocity) / speed_squared,
                0.0, 4.8))
            cpa = np.linalg.norm(
                relative_position + time_to_cpa * relative_velocity)
            if cpa <= 2.0:
                has_edge = True
    if scene.agents == 1:
        interaction = "N=1"
    elif has_edge:
        interaction = "multi_with_edge"
    else:
        interaction = "multi_without_edge"
    return {
        "history_speed": history_speed, "future_speed": future_speed,
        "speed_ratio": ratio, "turn_angle_deg": angle,
        "stop_or_slow": stop, "marked_deceleration": deceleration,
        "turn": turn, "near_straight": straight,
        "interaction": interaction, "has_edge": has_edge,
        "history_velocity": history_velocity,
    }


def scene_categories(scene, metadata):
    count = scene.agents
    if count == 1:
        group = "N=1"
    elif count <= 3:
        group = "N=2-3"
    else:
        group = "N>=4"
    masks = [
        ("overall", "all", np.ones(count, dtype=bool)),
        ("count_group", group, np.ones(count, dtype=bool)),
        ("exact_count", f"N={count}", np.ones(count, dtype=bool)),
        ("interaction", metadata["interaction"], np.ones(count, dtype=bool)),
    ]
    labels = ("stop_or_slow", "marked_deceleration", "turn", "near_straight")
    union = np.zeros(count, dtype=bool)
    for label in labels:
        mask = metadata[label]
        union |= mask
        masks.append(("motion", label, mask))
    masks.append(("motion", "unlabelled", ~union))
    return masks


def error_arrays(scene):
    distance = np.linalg.norm(
        scene.predictions.astype(np.float64) - scene.future[None].astype(np.float64),
        axis=-1)
    return distance.mean(axis=-1), distance[..., -1]


def endpoint_arrays(scene, ade, fde):
    endpoints = scene.predictions[:, :, -1].astype(np.float64)
    targets = scene.future[:, -1].astype(np.float64)
    center = endpoints.mean(axis=0)
    spread = np.sqrt(np.mean(np.sum((endpoints - center[None]) ** 2, axis=-1), axis=0))
    return {
        "best_ADE": ade.min(axis=0), "best_FDE": fde.min(axis=0),
        "median_candidate_FDE": np.median(fde, axis=0),
        "p90_candidate_FDE": np.quantile(fde, 0.9, axis=0),
        "endpoint_center_offset": np.linalg.norm(center - targets, axis=-1),
        "endpoint_spread_rms": spread,
        "endpoint_covariance_trace": spread ** 2,
        **{f"hit_{threshold:g}m": (fde.min(axis=0) <= threshold).astype(float)
           for threshold in (0.1, 0.2, 0.5, 1.0)},
    }


def stratified_diagnostics(banks):
    aggregate = defaultdict(lambda: defaultdict(list))
    identifiers = defaultdict(lambda: {"scenes": set(), "agents": set()})
    paired = defaultdict(lambda: defaultdict(list))
    paired_ids = defaultdict(lambda: {"scenes": set(), "agents": set()})
    continuous = defaultdict(list)
    scene_deltas = defaultdict(list)
    error_cache = {method: {seed: {} for seed in SEEDS} for method in METHODS}

    reference_scenes = banks["EBJD"][SEEDS[0]]
    metadata_by_identity = {
        identity: motion_metadata(scene) for identity, scene in reference_scenes.items()}
    for identity, scene in reference_scenes.items():
        metadata = metadata_by_identity[identity]
        for name in ("history_speed", "future_speed", "speed_ratio", "turn_angle_deg"):
            continuous[name].extend(metadata[name][np.isfinite(metadata[name])].tolist())

    for seed in SEEDS:
        for identity in banks["EBJD"][seed]:
            ebjd_scene = banks["EBJD"][seed][identity]
            metadata = metadata_by_identity[identity]
            method_values = {}
            for method in METHODS:
                scene = banks[method][seed][identity]
                ade, fde = error_arrays(scene)
                error_cache[method][seed][identity] = (ade, fde)
                endpoints = endpoint_arrays(scene, ade, fde)
                method_values[method] = endpoints
                for group_type, category, mask in scene_categories(scene, metadata):
                    if not mask.any():
                        continue
                    key = (method, group_type, category)
                    for name, values in endpoints.items():
                        aggregate[key][name].extend(values[mask].tolist())
                    aggregate[key]["scene_marginal_ADE"].append(float(
                        endpoints["best_ADE"][mask].mean()))
                    aggregate[key]["scene_marginal_FDE"].append(float(
                        endpoints["best_FDE"][mask].mean()))
                    aggregate[key]["subset_JADE"].append(float(
                        ade[:, mask].mean(axis=1).min()))
                    aggregate[key]["subset_JFDE"].append(float(
                        fde[:, mask].mean(axis=1).min()))
                    identifiers[key]["scenes"].add(identity)
                    for index in np.flatnonzero(mask):
                        identifiers[key]["agents"].add(
                            (identity, str(scene.agent_ids[index])))
            delta_ade = method_values["EBJD"]["best_ADE"] - method_values["GDTS"]["best_ADE"]
            delta_fde = method_values["EBJD"]["best_FDE"] - method_values["GDTS"]["best_FDE"]
            scene_deltas[identity].append(float(delta_fde.mean()))
            for group_type, category, mask in scene_categories(ebjd_scene, metadata):
                if not mask.any():
                    continue
                key = (group_type, category)
                paired[key]["delta_minADE"].extend(delta_ade[mask].tolist())
                paired[key]["delta_minFDE"].extend(delta_fde[mask].tolist())
                paired_ids[key]["scenes"].add(identity)
                for index in np.flatnonzero(mask):
                    paired_ids[key]["agents"].add(
                        (identity, str(ebjd_scene.agent_ids[index])))

    strata_rows = []
    for key in sorted(aggregate):
        method, group_type, category = key
        row = {
            "model": method, "group_type": group_type, "category": category,
            "unique_scenes": len(identifiers[key]["scenes"]),
            "unique_agent_occurrences": len(identifiers[key]["agents"]),
            "inference_seed_agent_records": len(aggregate[key]["best_ADE"]),
        }
        for name, values in aggregate[key].items():
            row[name] = float(np.mean(values)) if values else "NA"
        strata_rows.append(row)
    paired_rows = []
    for key in sorted(paired):
        group_type, category = key
        row = {
            "contrast": "EBJD-GDTS", "group_type": group_type,
            "category": category,
            "unique_scenes": len(paired_ids[key]["scenes"]),
            "unique_agent_occurrences": len(paired_ids[key]["agents"]),
        }
        for name, values in paired[key].items():
            for label, value in quantiles(values).items():
                row[f"{name}_{label}"] = value
        paired_rows.append(row)
    continuous_rows = []
    for name, values in continuous.items():
        continuous_rows.append({"variable": name, "count": len(values), **quantiles(values)})
    ranked = sorted(
        ((identity, float(np.mean(values))) for identity, values in scene_deltas.items()),
        key=lambda item: item[1])
    cases = {
        "most_improved": [{"identity": i, "mean_delta_minFDE_m": v} for i, v in ranked[:5]],
        "most_worsened": [{"identity": i, "mean_delta_minFDE_m": v} for i, v in ranked[-5:][::-1]],
    }
    return strata_rows, paired_rows, continuous_rows, cases, error_cache, metadata_by_identity


class Accumulator:
    def __init__(self):
        self.agent_ade = 0.0
        self.agent_fde = 0.0
        self.agents = 0
        self.scene_marginal_ade = 0.0
        self.scene_marginal_fde = 0.0
        self.jade = 0.0
        self.jfde = 0.0
        self.scenes = 0

    def update(self, ade: np.ndarray, fde: np.ndarray, indices: np.ndarray):
        selected_ade, selected_fde = ade[indices], fde[indices]
        agent_ade, agent_fde = selected_ade.min(0), selected_fde.min(0)
        self.agent_ade += float(agent_ade.sum())
        self.agent_fde += float(agent_fde.sum())
        self.agents += int(ade.shape[1])
        self.scene_marginal_ade += float(agent_ade.mean())
        self.scene_marginal_fde += float(agent_fde.mean())
        self.jade += float(selected_ade.mean(1).min())
        self.jfde += float(selected_fde.mean(1).min())
        self.scenes += 1

    def result(self):
        return {
            "minADE": self.agent_ade / self.agents,
            "minFDE": self.agent_fde / self.agents,
            "JADE": self.jade / self.scenes,
            "JFDE": self.jfde / self.scenes,
            "scene_weighted_minADE": self.scene_marginal_ade / self.scenes,
            "scene_weighted_minFDE": self.scene_marginal_fde / self.scenes,
        }


def k_diagnostics(error_cache):
    prefix_rows, random_rows, soft_rows = [], [], []
    monotonic_max = 0.0
    for seed in SEEDS:
        prefix_by_method = {method: {} for method in METHODS}
        for k in KS:
            accumulators = {method: Accumulator() for method in METHODS}
            for identity in error_cache["EBJD"][seed]:
                for method in METHODS:
                    ade, fde = error_cache[method][seed][identity]
                    accumulators[method].update(ade, fde, np.arange(k))
            for method in METHODS:
                result = accumulators[method].result()
                prefix_by_method[method][k] = result
                prefix_rows.append({
                    "selection": "nested_prefix", "model": method,
                    "seed": seed, "K": k, **result})
        for method in METHODS:
            for left, right in zip(KS, KS[1:]):
                for metric in METRICS:
                    monotonic_max = max(monotonic_max,
                        prefix_by_method[method][right][metric]
                        - prefix_by_method[method][left][metric])

        # Normalized soft K4 uses the fixed first-four proxy, scene weighting.
        for method in METHODS:
            values = defaultdict(list)
            for identity, (ade, fde) in error_cache[method][seed].items():
                for metric, error in (("ADE", ade[:4]), ("FDE", fde[:4])):
                    values[f"soft_marginal_{metric}"].append(float(
                        normalized_softmin(error, 0.05, 0).mean()))
                    values[f"soft_joint_{metric}"].append(float(
                        normalized_softmin(error.mean(axis=1), 0.05, 0)))
            soft_rows.append({
                "model": method, "seed": seed, "K": 4,
                "eta_m": 0.05,
                **{name: float(np.mean(item)) for name, item in values.items()},
                **{f"hard_K4_{name}": prefix_by_method[method][4][name]
                   for name in METRICS},
                **{f"hard_K20_{name}": prefix_by_method[method][20][name]
                   for name in METRICS},
            })

        for k in KS:
            repeats = 1 if k == 20 else 100
            accumulators = {
                method: [Accumulator() for _ in range(repeats)]
                for method in METHODS}
            for identity in error_cache["EBJD"][seed]:
                indices = [
                    np.random.default_rng(stable_seed(
                        "world-subset", seed, identity, k, repeat)).choice(
                            20, size=k, replace=False)
                    for repeat in range(repeats)]
                for method in METHODS:
                    ade, fde = error_cache[method][seed][identity]
                    for repeat, subset in enumerate(indices):
                        accumulators[method][repeat].update(ade, fde, subset)
            for method in METHODS:
                for repeat, accumulator in enumerate(accumulators[method]):
                    random_rows.append({
                        "selection": "shared_random_subset", "model": method,
                        "seed": seed, "K": k, "repeat": repeat,
                        **accumulator.result()})
    if monotonic_max > 1e-12:
        raise RuntimeError(f"K-prefix monotonicity failed by {monotonic_max}")

    summary_rows = []
    for selection, source in (("nested_prefix", prefix_rows),
                              ("shared_random_subset", random_rows)):
        for method in METHODS:
            for k in KS:
                selected = [row for row in source
                            if row["model"] == method and row["K"] == k]
                row = {"selection": selection, "model": method, "K": k,
                       "records": len(selected)}
                for metric in METRICS:
                    for label, value in quantiles([x[metric] for x in selected]).items():
                        row[f"{metric}_{label}"] = value
                summary_rows.append(row)
    return prefix_rows, random_rows, summary_rows, soft_rows, {
        "prefix_monotonic_max_positive_m": monotonic_max,
        "status": "PASSED",
    }


def shuffle_diagnostics(error_cache):
    rows, invariants = [], {
        "candidate_set_max_abs": 0.0, "marginal_max_abs_m": 0.0,
        "common_permutation_max_abs_m": 0.0,
        "single_agent_joint_max_abs_m": 0.0,
    }
    original = {method: {} for method in METHODS}
    for seed in SEEDS:
        for method in METHODS:
            base = Accumulator()
            for identity, (ade, fde) in error_cache[method][seed].items():
                base.update(ade, fde, np.arange(20))
            original[method][seed] = base.result()
        for repeat in range(100):
            by_method = {}
            for method in METHODS:
                jade, jfde = [], []
                marginal_ade, marginal_fde = [], []
                agent_ade, agent_fde = [], []
                for identity, (ade, fde) in error_cache[method][seed].items():
                    worlds, agents = ade.shape
                    shuffled_ade = np.empty_like(ade)
                    shuffled_fde = np.empty_like(fde)
                    for agent in range(agents):
                        permutation = np.random.default_rng(stable_seed(
                            "independent-world-permutation", seed, identity,
                            repeat, agent)).permutation(worlds)
                        shuffled_ade[:, agent] = ade[permutation, agent]
                        shuffled_fde[:, agent] = fde[permutation, agent]
                        invariants["candidate_set_max_abs"] = max(
                            invariants["candidate_set_max_abs"],
                            float(np.max(np.abs(
                                np.sort(shuffled_ade[:, agent]) -
                                np.sort(ade[:, agent])))))
                    before_ade, after_ade = ade.min(0), shuffled_ade.min(0)
                    before_fde, after_fde = fde.min(0), shuffled_fde.min(0)
                    invariants["marginal_max_abs_m"] = max(
                        invariants["marginal_max_abs_m"],
                        float(np.max(np.abs(before_ade - after_ade))),
                        float(np.max(np.abs(before_fde - after_fde))))
                    agent_ade.extend(after_ade.tolist())
                    agent_fde.extend(after_fde.tolist())
                    marginal_ade.append(float(after_ade.mean()))
                    marginal_fde.append(float(after_fde.mean()))
                    jade.append(float(shuffled_ade.mean(1).min()))
                    jfde.append(float(shuffled_fde.mean(1).min()))

                    common = np.random.default_rng(stable_seed(
                        "common-world-permutation", seed, identity, repeat)).permutation(worlds)
                    common_jade = float(ade[common].mean(1).min())
                    common_jfde = float(fde[common].mean(1).min())
                    invariants["common_permutation_max_abs_m"] = max(
                        invariants["common_permutation_max_abs_m"],
                        abs(common_jade - float(ade.mean(1).min())),
                        abs(common_jfde - float(fde.mean(1).min())))
                    if agents == 1:
                        invariants["single_agent_joint_max_abs_m"] = max(
                            invariants["single_agent_joint_max_abs_m"],
                            abs(jade[-1] - float(ade.mean(1).min())),
                            abs(jfde[-1] - float(fde.mean(1).min())))
                by_method[method] = {
                    "minADE": float(np.mean(agent_ade)),
                    "minFDE": float(np.mean(agent_fde)),
                    "scene_weighted_minADE": float(np.mean(marginal_ade)),
                    "scene_weighted_minFDE": float(np.mean(marginal_fde)),
                    "JADE": float(np.mean(jade)), "JFDE": float(np.mean(jfde)),
                }
                rows.append({
                    "model": method, "seed": seed, "repeat": repeat,
                    "original_JADE": original[method][seed]["JADE"],
                    "original_JFDE": original[method][seed]["JFDE"],
                    "shuffled_JADE": by_method[method]["JADE"],
                    "shuffled_JFDE": by_method[method]["JFDE"],
                    "delta_JADE": by_method[method]["JADE"] - original[method][seed]["JADE"],
                    "delta_JFDE": by_method[method]["JFDE"] - original[method][seed]["JFDE"],
                    "shuffled_minADE": by_method[method]["minADE"],
                    "shuffled_minFDE": by_method[method]["minFDE"],
                })
            rows.append({
                "model": "EBJD-GDTS", "seed": seed, "repeat": repeat,
                "original_JADE": original["EBJD"][seed]["JADE"] - original["GDTS"][seed]["JADE"],
                "original_JFDE": original["EBJD"][seed]["JFDE"] - original["GDTS"][seed]["JFDE"],
                "shuffled_JADE": by_method["EBJD"]["JADE"] - by_method["GDTS"]["JADE"],
                "shuffled_JFDE": by_method["EBJD"]["JFDE"] - by_method["GDTS"]["JFDE"],
                "delta_JADE": (
                    by_method["EBJD"]["JADE"] - by_method["GDTS"]["JADE"]
                    - original["EBJD"][seed]["JADE"] + original["GDTS"][seed]["JADE"]),
                "delta_JFDE": (
                    by_method["EBJD"]["JFDE"] - by_method["GDTS"]["JFDE"]
                    - original["EBJD"][seed]["JFDE"] + original["GDTS"][seed]["JFDE"]),
            })
    invariant_status = all(value <= 1e-12 for value in invariants.values())
    invariants["status"] = "PASSED" if invariant_status else "FAILED"
    summary = []
    for model in (*METHODS, "EBJD-GDTS"):
        selected = [row for row in rows if row["model"] == model]
        item = {"model": model, "records": len(selected)}
        for name in ("original_JADE", "original_JFDE", "shuffled_JADE",
                     "shuffled_JFDE", "delta_JADE", "delta_JFDE"):
            for label, value in quantiles([row[name] for row in selected]).items():
                item[f"{name}_{label}"] = value
        summary.append(item)
    return rows, summary, invariants


def _alignment(fig, stem: Path, labels: list[str]):
    skill_scripts = Path("/home/ee615/.codex/skills/nature-figure/scripts")
    sys.path.insert(0, str(skill_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    require_matplotlib_panel_alignment(
        fig, json_out=str(stem) + ".alignment-layout.json",
        overlay_svg=str(stem) + ".alignment-overlay.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        require_panel_labels=True, strict=True)


def save_figure(fig, stem: Path, labels: list[str]):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    _alignment(fig, stem, labels)
    fig.savefig(str(stem) + ".pdf", bbox_inches="tight")
    fig.savefig(str(stem) + ".svg", bbox_inches="tight")
    fig.savefig(str(stem) + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_k(summary_rows, figure_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
    for axis, pair, ylabel in zip(
        axes, (("minADE", "JADE"), ("minFDE", "JFDE")),
        ("Average displacement error (m)", "Final displacement error (m)"),
        strict=True):
        for method in METHODS:
            selected = [row for row in summary_rows
                        if row["selection"] == "shared_random_subset"
                        and row["model"] == method]
            x = np.asarray([int(row["K"]) for row in selected])
            for metric, linestyle in zip(pair, ("-", "--"), strict=True):
                y = np.asarray([float(row[f"{metric}_mean"]) for row in selected])
                lower = np.asarray([float(row[f"{metric}_q05"]) for row in selected])
                upper = np.asarray([float(row[f"{metric}_q95"]) for row in selected])
                axis.plot(x, y, marker="o", ms=3, lw=1.2, ls=linestyle,
                          color=COLORS[method], label=f"{method} {metric}")
                axis.fill_between(x, lower, upper, color=COLORS[method], alpha=0.10)
        axis.set(xlabel="World budget K", ylabel=ylabel, xticks=KS)
        axis.grid(axis="y", lw=0.4, alpha=0.3)
        axis.legend(fontsize=6, ncol=2, loc="upper center",
                    bbox_to_anchor=(0.5, -0.24), frameon=False)
    for label, axis in zip(("a", "b"), axes, strict=True):
        axis.text(-0.10, 1.08, label, transform=axis.transAxes,
                  fontsize=8, fontweight="bold", va="top", ha="left")
    save_figure(fig, figure_dir / "epoch045_k_curve", ["a", "b"])


def plot_shuffle(rows, figure_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
    for axis, metric in zip(axes, ("JADE", "JFDE"), strict=True):
        values = [
            [float(row[f"delta_{metric}"]) for row in rows
             if row["model"] == method]
            for method in METHODS]
        boxes = axis.boxplot(values, positions=[0, 1], widths=0.55,
                             patch_artist=True, showfliers=False)
        for patch, method in zip(boxes["boxes"], METHODS, strict=True):
            patch.set_facecolor(COLORS[method]); patch.set_alpha(0.55)
        axis.axhline(0, color="#222222", lw=0.7)
        axis.set_xticks([0, 1], METHODS)
        axis.set_ylabel(f"Independent permutation Δ{metric} (m)")
        axis.grid(axis="y", lw=0.4, alpha=0.3)
    for label, axis in zip(("a", "b"), axes, strict=True):
        axis.text(-0.10, 1.08, label, transform=axis.transAxes,
                  fontsize=8, fontweight="bold", va="top", ha="left")
    save_figure(fig, figure_dir / "epoch045_world_permutation", ["a", "b"])


def plot_cases(banks, cases, metadata, figure_dir: Path):
    selected = cases["most_worsened"] + cases["most_improved"]
    fig, axes = plt.subplots(2, 5, figsize=(11.0, 4.6), constrained_layout=True)
    for position, (axis, item) in enumerate(zip(axes.flat, selected, strict=True)):
        identity = item["identity"]
        ebjd = banks["EBJD"][2035][identity]
        gdts = banks["GDTS"][2035][identity]
        history_velocity = metadata[identity]["history_velocity"]
        horizon = np.arange(1, 13)[:, None] * ebjd.dt_seconds
        for agent in range(ebjd.agents):
            axis.plot(ebjd.history[agent, :, 0], ebjd.history[agent, :, 1],
                      color="#222222", lw=1.0)
            axis.plot(ebjd.future[agent, :, 0], ebjd.future[agent, :, 1],
                      color="#009E73", lw=1.5)
            cv = ebjd.history[agent, -1] + horizon * history_velocity[agent]
            axis.plot(cv[:, 0], cv[:, 1], color="#CC79A7", ls=":", lw=1.0)
            for world in range(20):
                axis.plot(ebjd.predictions[world, agent, :, 0],
                          ebjd.predictions[world, agent, :, 1],
                          color=COLORS["EBJD"], alpha=0.10, lw=0.45)
                axis.plot(gdts.predictions[world, agent, :, 0],
                          gdts.predictions[world, agent, :, 1],
                          color=COLORS["GDTS"], alpha=0.08, lw=0.45)
        direction = "Worsened" if position < 5 else "Improved"
        axis.set_title(
            f"{direction} {position % 5 + 1}: ΔFDE={item['mean_delta_minFDE_m']:+.3f} m",
            fontsize=6.2)
        axis.set_aspect("equal", adjustable="datalim")
        axis.tick_params(labelsize=5)
        label = chr(ord("a") + position)
        axis.text(-0.13, 1.06, label, transform=axis.transAxes,
                  fontsize=8, fontweight="bold", va="top")
    fig.legend(handles=[
        Line2D([0], [0], color="#222222", lw=1.2, label="History"),
        Line2D([0], [0], color="#009E73", lw=1.5, label="Ground truth"),
        Line2D([0], [0], color="#CC79A7", lw=1.0, ls=":",
               label="Constant velocity"),
        Line2D([0], [0], color=COLORS["EBJD"], lw=1.0, alpha=0.6,
               label="EBJD candidates"),
        Line2D([0], [0], color=COLORS["GDTS"], lw=1.0, alpha=0.6,
               label="GDTS candidates"),
    ], loc="outside lower center", ncol=5, frameon=False, fontsize=6.5)
    save_figure(fig, figure_dir / "epoch045_fixed_rule_cases",
                [chr(ord("a") + index) for index in range(10)])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CPU-only parity, coverage, K and world-permutation diagnostics")
    parser.add_argument("--bank-root", required=True)
    parser.add_argument("--ebjd-reference", required=True)
    parser.add_argument("--gdts-reference", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--figure-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    figures = Path(args.figure_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    banks, manifests, identity = load_banks(Path(args.bank_root).resolve())
    bank_root = Path(args.bank_root).resolve()
    export_indices = {
        method: json.loads((bank_root / method / "index.json").read_text(
            encoding="utf-8")) for method in ("ebjd", "gdts")}
    summaries, parity_rows, parity_result = parity(
        banks, Path(args.ebjd_reference).resolve(), Path(args.gdts_reference).resolve())
    if parity_result["status"] != "PASSED":
        raise RuntimeError("bank parity failed; downstream diagnostics are provisional")
    strata, paired, continuous, cases, errors, metadata = stratified_diagnostics(banks)
    prefix, random_rows, k_summary, soft, k_invariants = k_diagnostics(errors)
    shuffle_rows, shuffle_summary, shuffle_invariants = shuffle_diagnostics(errors)

    write_csv(output / "parity.csv", parity_rows)
    write_csv(output / "marginal_strata.csv", strata)
    write_csv(output / "paired_deltas.csv", paired)
    write_csv(output / "motion_continuous.csv", continuous)
    write_csv(output / "k_prefix_per_seed.csv", prefix)
    write_csv(output / "k_random_subset_replicates.csv", random_rows)
    write_csv(output / "k_curve_summary.csv", k_summary)
    write_csv(output / "soft_k4_proxy.csv", soft)
    write_csv(output / "world_permutation_replicates.csv", shuffle_rows)
    write_csv(output / "world_permutation_summary.csv", shuffle_summary)
    atomic_json({"schema": "ebjd-diagnostic-bank-receipts-v1", "banks": manifests},
                output / "BANK_RECEIPTS.json")
    atomic_json(cases, output / "FIXED_RULE_CASES.json")

    plot_k(k_summary, figures)
    plot_shuffle(shuffle_rows, figures)
    plot_cases(banks, cases, metadata, figures)

    mean_metrics = {
        method: {metric: float(np.mean([
            summaries[method][seed][metric] for seed in SEEDS]))
            for metric in (*METRICS, "scene_weighted_minADE",
                           "scene_weighted_minFDE", "coordination_gap_ADE",
                           "coordination_gap_FDE")}
        for method in METHODS}
    result = {
        "schema": "ebjd-hotel-epoch045-diagnostic-v1",
        "status": "COMPLETED_ZERO_TRAINING_DIAGNOSTIC",
        "parity": parity_result,
        "identity_matching": identity,
        "mean_metrics": mean_metrics,
        "per_seed_replay": summaries,
        "bank_receipts_sha256": sha256_file(output / "BANK_RECEIPTS.json"),
        "k_invariants": k_invariants,
        "world_permutation_invariants": shuffle_invariants,
        "world_permutation_summary": shuffle_summary,
        "fixed_rule_cases": cases,
        "optimizer_updates": 0,
        "forward_calls": int(sum(item["forward_calls"] for item in manifests)),
        "gpu_export_seconds": float(sum(
            item["elapsed_seconds"] for item in export_indices.values())),
        "cuda_peak_memory_bytes": int(max(
            item["cuda_peak_memory_bytes"] for item in export_indices.values())),
        "training_seed": 3101, "inference_seeds": list(SEEDS),
        "independent_test": False,
        "validation_test_identity": "mirrored_official_package_internal_only",
        "counts": {"scenes_per_seed": 445, "agent_occurrences_per_seed": 1197,
                   "worlds": 20, "prediction_steps": 12},
        "files": {
            path.name: sha256_file(path) for path in output.iterdir()
            if path.is_file() and path.suffix in {".csv", ".json"}
            and path.name != "EBJD_HOTEL_EPOCH045_DIAGNOSTIC_2026-10-12.json"
        },
        "figure_files": {
            path.name: sha256_file(path) for path in figures.iterdir()
            if path.is_file() and path.suffix in {".pdf", ".svg", ".png"}
            and "overlay" not in path.name
        },
    }
    result_path = output / "EBJD_HOTEL_EPOCH045_DIAGNOSTIC_2026-10-12.json"
    atomic_json(result, result_path)
    print(json.dumps({
        "result": str(result_path), "parity": parity_result,
        "identity": identity, "mean_metrics": mean_metrics,
        "shuffle": shuffle_summary,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

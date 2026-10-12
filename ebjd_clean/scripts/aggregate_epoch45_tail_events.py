#!/usr/bin/env python3
"""Aggregate epoch-45 paired marginal tails by occurrence, event and track."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from ebjd.diagnostic_bank import SeedBank, reorder_like


SEEDS = (2035, 2036, 2037, 2038, 2039)
HIT_THRESHOLDS = (0.1, 0.2, 0.5, 1.0)


def atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refuse to write empty diagnostic table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def agent_statistics(predictions, future):
    distance = np.linalg.norm(
        predictions.astype(np.float64) - future[None].astype(np.float64), axis=-1)
    ade = distance.mean(axis=-1)
    fde = distance[..., -1]
    endpoints = predictions[:, -1, :].astype(np.float64)
    target = future[-1].astype(np.float64)
    centre = endpoints.mean(axis=0)
    result = {
        "minADE": float(ade.min()),
        "minFDE": float(fde.min()),
        "median_candidate_FDE": float(np.median(fde)),
        "p90_candidate_FDE": float(np.quantile(fde, 0.9)),
        "endpoint_center_offset": float(np.linalg.norm(centre - target)),
        "endpoint_spread_rms": float(np.sqrt(
            np.mean(np.sum((endpoints - centre) ** 2, axis=-1)))),
    }
    for threshold in HIT_THRESHOLDS:
        result[f"hit_{threshold:.1f}m"] = float(fde.min() <= threshold)
    return result


def motion(history, future, dt):
    history_velocity = (
        (history[-1] - history[-2]) / dt
        + (history[-2] - history[-3]) / dt) / 2.0
    history_speed = float(np.linalg.norm(history_velocity))
    full = np.concatenate((history[-1:], future), axis=0)
    future_speed = float(np.linalg.norm(np.diff(full, axis=0), axis=-1).mean() / dt)
    return {
        "history_speed_mps": history_speed,
        "future_speed_mps": future_speed,
        "speed_ratio": future_speed / history_speed if history_speed > 1e-12 else np.nan,
        "marked_deceleration": bool(
            history_speed >= 0.3 and future_speed <= 0.5 * history_speed),
    }


def build_events(occurrences: list[dict]) -> list[dict]:
    by_track = defaultdict(list)
    all_starts = defaultdict(list)
    for row in occurrences:
        track = (row["source_sequence"], row["agent_id"])
        all_starts[track].append(int(row["frame_start"]))
        if row["marked_deceleration"]:
            by_track[track].append(row)
    events = []
    for track, rows in by_track.items():
        rows.sort(key=lambda item: int(item["frame_start"]))
        unique_starts = sorted(set(all_starts[track]))
        differences = np.diff(unique_starts)
        nominal_stride = int(np.median(differences)) if differences.size else 10
        maximum_gap = max(int(round(1.5 * nominal_stride)), nominal_stride)
        groups = []
        current = []
        previous_start = None
        for row in rows:
            start = int(row["frame_start"])
            if previous_start is not None and start - previous_start > maximum_gap:
                groups.append(current)
                current = []
            current.append(row)
            previous_start = start
        if current:
            groups.append(current)
        for index, group in enumerate(groups, start=1):
            delta_sum = float(sum(item["delta_minFDE_m"] for item in group))
            events.append({
                "event_id": f"{track[0]}:{track[1]}:decel_{index:03d}",
                "source_sequence": track[0], "agent_id": track[1],
                "window_count": len(group),
                "first_window_start": min(item["frame_start"] for item in group),
                "last_window_start": max(item["frame_start"] for item in group),
                "union_frame_start": min(item["frame_start"] for item in group),
                "union_frame_end": max(item["frame_end"] for item in group),
                "nominal_window_start_stride": nominal_stride,
                "maximum_consecutive_start_gap": maximum_gap,
                "mean_delta_minADE_m": float(np.mean([
                    item["delta_minADE_m"] for item in group])),
                "mean_delta_minFDE_m": float(np.mean([
                    item["delta_minFDE_m"] for item in group])),
                "sum_delta_minFDE_m": delta_sum,
                "mean_history_speed_mps": float(np.mean([
                    item["history_speed_mps"] for item in group])),
                "mean_future_speed_mps": float(np.mean([
                    item["future_speed_mps"] for item in group])),
                "occurrence_keys_json": json.dumps(
                    [item["occurrence_key"] for item in group],
                    separators=(",", ":")),
            })
    return sorted(events, key=lambda item: item["sum_delta_minFDE_m"], reverse=True)


def track_rows(occurrences: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in occurrences:
        grouped[(row["source_sequence"], row["agent_id"])].append(row)
    result = []
    for (source, agent), rows in grouped.items():
        result.append({
            "source_sequence": source, "agent_id": agent,
            "window_count": len(rows),
            "deceleration_window_count": sum(
                int(row["marked_deceleration"]) for row in rows),
            "mean_delta_minADE_m": float(np.mean([
                row["delta_minADE_m"] for row in rows])),
            "mean_delta_minFDE_m": float(np.mean([
                row["delta_minFDE_m"] for row in rows])),
            "sum_delta_minFDE_m": float(sum(
                row["delta_minFDE_m"] for row in rows)),
            "frame_start": min(row["frame_start"] for row in rows),
            "frame_end": max(row["frame_end"] for row in rows),
        })
    return sorted(result, key=lambda item: item["sum_delta_minFDE_m"], reverse=True)


def continuity_audit(reference_scenes: dict, tracks: set[tuple[str, str]]):
    results = []
    for source, agent in sorted(tracks):
        positions = defaultdict(list)
        for scene in reference_scenes.values():
            if scene.source_sequence != source or agent not in scene.agent_ids.tolist():
                continue
            index = scene.agent_ids.tolist().index(agent)
            trajectory = np.concatenate((scene.history[index], scene.future[index]), axis=0)
            for frame, position in zip(scene.frame_ids.tolist(), trajectory, strict=True):
                positions[int(frame)].append(position.astype(np.float64))
        overlap_discrepancy = 0.0
        canonical = []
        for frame, values in sorted(positions.items()):
            stack = np.stack(values)
            overlap_discrepancy = max(
                overlap_discrepancy,
                float(np.linalg.norm(stack - stack[0], axis=-1).max()))
            canonical.append((frame, stack.mean(axis=0)))
        step_jumps = [
            float(np.linalg.norm(right[1] - left[1]))
            for left, right in zip(canonical, canonical[1:])]
        maximum_jump = max(step_jumps, default=0.0)
        results.append({
            "source_sequence": source, "agent_id": agent,
            "unique_frames": len(canonical),
            "maximum_overlap_position_discrepancy_m": overlap_discrepancy,
            "maximum_consecutive_position_jump_m": maximum_jump,
            "obvious_jump_over_5m": maximum_jump > 5.0,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.bank_root).resolve()
    output = Path(args.output_dir).resolve()
    banks = {
        model: {
            seed: SeedBank(root / model.lower() / f"seed_{seed}" / "manifest.json")
            for seed in SEEDS}
        for model in ("EBJD", "GDTS")
    }
    accumulators = defaultdict(list)
    reference_scenes = {}
    for seed in SEEDS:
        identities = banks["EBJD"][seed].identities()
        if set(identities) != set(banks["GDTS"][seed].identities()):
            raise ValueError("EBJD/GDTS window identities differ")
        for identity in identities:
            ebjd = banks["EBJD"][seed].load(identity)
            gdts = reorder_like(ebjd, banks["GDTS"][seed].load(identity))
            if seed == SEEDS[0]:
                reference_scenes[identity] = ebjd
            for index, agent in enumerate(ebjd.agent_ids.tolist()):
                key = (ebjd.source_sequence, identity, str(agent))
                ebjd_stats = agent_statistics(
                    ebjd.predictions[:, index], ebjd.future[index])
                gdts_stats = agent_statistics(
                    gdts.predictions[:, index], gdts.future[index])
                accumulators[key].append((ebjd_stats, gdts_stats))
    occurrences = []
    for (source, identity, agent), values in accumulators.items():
        if len(values) != len(SEEDS):
            raise ValueError("occurrence does not contain all five inference seeds")
        scene = reference_scenes[identity]
        index = scene.agent_ids.tolist().index(agent)
        motion_values = motion(
            scene.history[index].astype(np.float64),
            scene.future[index].astype(np.float64), float(scene.dt_seconds))
        ebjd_mean = {
            name: float(np.mean([item[0][name] for item in values]))
            for name in values[0][0]}
        gdts_mean = {
            name: float(np.mean([item[1][name] for item in values]))
            for name in values[0][1]}
        row = {
            "occurrence_key": f"{identity}|agent={agent}",
            "source_sequence": source, "window_key": identity,
            "frame_start": int(scene.frame_ids[0]),
            "frame_end": int(scene.frame_ids[-1]), "agent_id": agent,
            "inference_seed_count": len(values),
            **motion_values,
            "ebjd_minADE_m": ebjd_mean["minADE"],
            "gdts_minADE_m": gdts_mean["minADE"],
            "delta_minADE_m": ebjd_mean["minADE"] - gdts_mean["minADE"],
            "ebjd_minFDE_m": ebjd_mean["minFDE"],
            "gdts_minFDE_m": gdts_mean["minFDE"],
            "delta_minFDE_m": ebjd_mean["minFDE"] - gdts_mean["minFDE"],
        }
        for model, summary in (("ebjd", ebjd_mean), ("gdts", gdts_mean)):
            for name in (
                "median_candidate_FDE", "p90_candidate_FDE",
                "endpoint_center_offset", "endpoint_spread_rms",
                *(f"hit_{threshold:.1f}m" for threshold in HIT_THRESHOLDS),
            ):
                row[f"{model}_{name}"] = summary[name]
        occurrences.append(row)
    occurrences.sort(key=lambda item: item["delta_minFDE_m"], reverse=True)
    events = build_events(occurrences)
    tracks = track_rows(occurrences)
    total = len(occurrences)
    overall_delta = float(np.mean([row["delta_minFDE_m"] for row in occurrences]))
    for rows in (events, tracks):
        for row in rows:
            contribution = float(row["sum_delta_minFDE_m"]) / total
            row["signed_contribution_to_overall_minFDE_m"] = contribution
            row["signed_contribution_percent_of_net_delta"] = (
                100 * contribution / overall_delta if abs(overall_delta) > 1e-15
                else np.nan)
    sensitivity = []
    all_keys = {row["occurrence_key"] for row in occurrences}
    for count in (1, 3, 5):
        removed = {
            key for event in events[:count]
            for key in json.loads(event["occurrence_keys_json"])}
        kept = [
            row["delta_minFDE_m"] for row in occurrences
            if row["occurrence_key"] in all_keys - removed]
        sensitivity.append({
            "excluded_top_deceleration_events": count,
            "excluded_unique_occurrences": len(removed),
            "remaining_occurrences": len(kept),
            "sensitivity_mean_delta_minFDE_m": float(np.mean(kept)),
        })
    top_tracks = {
        (row["source_sequence"], row["agent_id"]) for row in tracks[:10]}
    continuity = continuity_audit(reference_scenes, top_tracks)
    write_csv(output / "epoch45_agent_occurrences_five_seed_mean.csv", occurrences)
    write_csv(output / "epoch45_deceleration_events.csv", events)
    write_csv(output / "epoch45_track_aggregation.csv", tracks)
    write_csv(output / "epoch45_top_event_exclusion_sensitivity.csv", sensitivity)
    write_csv(output / "epoch45_top_track_continuity.csv", continuity)
    result = {
        "schema": "ebjd-epoch45-heavy-tail-event-aggregation-v1",
        "status": "COMPLETED",
        "aggregation_order": (
            "average five inference seeds within exact source/window/agent "
            "occurrence, then aggregate consecutive deceleration-labelled "
            "window starts within 1.5 nominal strides on the same track"),
        "event_independence_limitation": (
            "heuristic temporal groups are not asserted to be statistically "
            "or physically independent events"),
        "occurrence_count": total,
        "overall_mean_delta_minFDE_m": overall_delta,
        "contribution_note": (
            "signed percentages can exceed 100% because improvements outside "
            "the reported group offset positive tail contributions"),
        "top_worsened_occurrences": occurrences[:10],
        "top_deceleration_events": events[:10],
        "top_tracks": tracks[:10],
        "sensitivity": sensitivity,
        "continuity": continuity,
        "official_metric_exclusions": 0,
    }
    atomic_json(result, output / "EPOCH45_HEAVY_TAIL_EVENT_SUMMARY.json")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

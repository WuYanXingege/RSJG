"""Command line P20 evaluation entry point: ``python -m ebjd.evaluate``."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import build_model, load_config
from .data import SceneManifestDataset, collate_scenes
from .metrics import MetricAccumulator
from .sampling import predict


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate independent EBJD")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--ablation", default="none")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--prediction-bank-root",
        help="optional local-only directory for compressed per-seed predictions")
    parser.add_argument(
        "--disable-native-goal-conditioner", action="store_true",
        help="registered intervention: retain weights/cache but bypass conditioning")
    parser.add_argument(
        "--pairing-rounds", type=int, choices=(0, 2), default=2,
        help="2 for learned coordinate updates; 0 for frozen initial random pairing")
    parser.add_argument(
        "--geometry-mapping", choices=("configured", "direct_index"),
        default="configured", help="frozen-model geometry sensitivity intervention")
    parser.add_argument(
        "--endpoint-projection-radius", type=float,
        help="read-only L2 endpoint correction radius around assigned anchors")
    args = parser.parse_args()
    config = load_config(args.config)
    path = config.get("paths", {}).get(f"{args.split}_manifest")
    if not path:
        raise ValueError(f"paths.{args.split}_manifest is required")
    device = torch.device(args.device)
    model = build_model(config, args.ablation).to(device).float()
    checkpoint_sha256_before = sha256(args.checkpoint)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    if args.disable_native_goal_conditioner:
        conditioner = model.denoiser.native_goal_conditioner
        if conditioner is None:
            raise ValueError("checkpoint has no native-goal conditioner to disable")
        conditioner.enabled = False
    if args.geometry_mapping == "direct_index":
        model.denoiser.geometry_mapping = "direct_index"
    model.eval()
    evaluation = config.get("evaluation", {})
    worlds, steps = int(evaluation.get("worlds", 20)), int(evaluation.get("steps", 20))
    results = []
    bank_index = []
    with torch.no_grad(), torch.autocast(
            device_type=device.type, enabled=False):
        for seed in evaluation.get("inference_seeds", [2035, 2036, 2037, 2038, 2039]):
            cache = None
            if "native_goal_conditioning" in config:
                cache = config["native_goal_conditioning"][
                    "validation_cache_by_seed"].get(str(seed))
                if cache is None:
                    raise ValueError(f"native-goal cache missing inference seed {seed}")
            dataset = SceneManifestDataset(path, native_goal_bank=cache)
            loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                collate_fn=collate_scenes, num_workers=0)
            generator = torch.Generator(device=device).manual_seed(int(seed))
            accumulator = MetricAccumulator()
            k_values = (1, 4, 8, 12, 16, 20)
            k_accumulators = {value: MetricAccumulator() for value in k_values}
            stratum_accumulators = {
                "N=1": MetricAccumulator(), "N>=2": MetricAccumulator()}
            endpoint_sums = {
                "agents": 0, "native_oracle_fde": 0.0,
                "diffusion_oracle_fde": 0.0, "native_to_diffusion_nn": 0.0,
                "native_recall_0p5": 0.0, "diffusion_recall_0p5": 0.0,
                "native_recall_1p0": 0.0, "diffusion_recall_1p0": 0.0,
                "native_cv_relative_radius": 0.0,
                "diffusion_cv_relative_radius": 0.0,
            }
            endpoint_errors: list[float] = []
            endpoint_single_errors: list[float] = []
            endpoint_center_errors: list[float] = []
            endpoint_centered_rms: list[float] = []
            prediction_rows = []
            for batch in loader:
                batch = batch.to(device)
                prediction, predicted_goal = predict(
                    model, batch.observed, batch.semantic_maps, batch.valid,
                    worlds, steps, generator, batch.native_goals,
                    batch.scene_ids, batch.agent_ids, int(seed),
                    pairing_rounds=args.pairing_rounds,
                    endpoint_projection_radius=args.endpoint_projection_radius)
                accumulator.update(prediction, batch.future, batch.valid)
                if args.prediction_bank_root:
                    count = int(batch.valid.sum())
                    prediction_rows.append({
                        "scene_id": batch.scene_ids[0],
                        "agent_ids": np.asarray(batch.agent_ids[0]),
                        "observed": batch.observed[0, :count].cpu().numpy(),
                        "future": batch.future[0, :count].cpu().numpy(),
                        "predictions": prediction[0, :, :count].cpu().numpy(),
                        "output_goals": predicted_goal[0, :, :count].cpu().numpy(),
                        "native_goals": None if batch.native_goals is None else
                            batch.native_goals[0, :count].cpu().numpy(),
                    })
                for value in k_values:
                    k_accumulators[value].update(
                        prediction[:, :value], batch.future, batch.valid)
                group = "N=1" if int(batch.valid.sum()) == 1 else "N>=2"
                stratum_accumulators[group].update(
                    prediction, batch.future, batch.valid)
                if batch.native_goals is not None:
                    mask = batch.valid
                    target = batch.future[..., -1, :][mask]
                    native = batch.native_goals[mask]
                    diffuse = predicted_goal.permute(0, 2, 1, 3)[mask]
                    native_error = torch.linalg.vector_norm(
                        native - target[:, None], dim=-1).min(-1).values
                    diffuse_error = torch.linalg.vector_norm(
                        diffuse - target[:, None], dim=-1).min(-1).values
                    single_error = torch.linalg.vector_norm(
                        diffuse - target[:, None], dim=-1)
                    centre = diffuse.mean(1)
                    centre_error = torch.linalg.vector_norm(centre - target, dim=-1)
                    centred_rms = torch.sqrt(
                        (diffuse - centre[:, None]).square().sum(-1).mean(-1))
                    cross = torch.cdist(native, diffuse).min(-1).values.mean(-1)
                    cv = batch.observed[..., -1, :][mask] + 12 * 0.4 * (
                        batch.observed[..., -1, :][mask]
                        - batch.observed[..., -2, :][mask]) / 0.4
                    native_radius = torch.linalg.vector_norm(
                        native - cv[:, None], dim=-1).mean(-1)
                    diffuse_radius = torch.linalg.vector_norm(
                        diffuse - cv[:, None], dim=-1).mean(-1)
                    count = int(mask.sum())
                    endpoint_sums["agents"] += count
                    endpoint_sums["native_oracle_fde"] += float(native_error.sum())
                    endpoint_sums["diffusion_oracle_fde"] += float(diffuse_error.sum())
                    endpoint_sums["native_to_diffusion_nn"] += float(cross.sum())
                    endpoint_sums["native_recall_0p5"] += float((native_error <= .5).sum())
                    endpoint_sums["diffusion_recall_0p5"] += float((diffuse_error <= .5).sum())
                    endpoint_sums["native_recall_1p0"] += float((native_error <= 1).sum())
                    endpoint_sums["diffusion_recall_1p0"] += float((diffuse_error <= 1).sum())
                    endpoint_sums["native_cv_relative_radius"] += float(native_radius.sum())
                    endpoint_sums["diffusion_cv_relative_radius"] += float(diffuse_radius.sum())
                    endpoint_errors.extend(diffuse_error.detach().cpu().tolist())
                    endpoint_single_errors.extend(single_error.detach().cpu().flatten().tolist())
                    endpoint_center_errors.extend(centre_error.detach().cpu().tolist())
                    endpoint_centered_rms.extend(centred_rms.detach().cpu().tolist())
            means = accumulator.as_dict()
            nonfinite = {
                name: value for name, value in means.items()
                if isinstance(value, (int, float)) and not math.isfinite(float(value))}
            if nonfinite:
                raise FloatingPointError(
                    f"non-finite formal metrics for seed {seed}: {nonfinite}")
            if means["scene_count"] != len(dataset):
                raise RuntimeError(
                    f"formal evaluation skipped scenes for seed {seed}: "
                    f"{means['scene_count']} != {len(dataset)}")
            expected_agents = int(dataset.manifest["statistics"]["agents"])
            if means["agent_count"] != expected_agents:
                raise RuntimeError(
                    f"formal evaluation skipped agent occurrences for seed {seed}: "
                    f"{means['agent_count']} != {expected_agents}")
            k_curves = {str(value): k_accumulators[value].as_dict()
                        for value in k_values}
            strata = {
                name: value.as_dict() for name, value in stratum_accumulators.items()
                if value.scene_count > 0}
            count = max(1, int(endpoint_sums["agents"]))
            endpoint = {
                name: (value if name == "agents" else float(value) / count)
                for name, value in endpoint_sums.items()}
            endpoint.update({
                "diffusion_minFDE_p90": float(np.quantile(endpoint_errors, 0.90)),
                "diffusion_minFDE_p95": float(np.quantile(endpoint_errors, 0.95)),
                "average_single_candidate_FDE": float(np.mean(endpoint_single_errors)),
                "endpoint_center_error": float(np.mean(endpoint_center_errors)),
                "endpoint_centered_RMS": float(np.mean(endpoint_centered_rms)),
            })
            conditioner = model.denoiser.native_goal_conditioner
            results.append({
                "seed": int(seed), "metrics": means, "k_curves": k_curves,
                "strata": strata, "endpoint_diagnostics": endpoint,
                "conditioner": None if conditioner is None else {
                    "gate_raw": float(conditioner.gate.detach().float().cpu()),
                    "gate_tanh": float(torch.tanh(
                        conditioner.gate.detach().float()).cpu()),
                    "last_condition_norm_ratio": float(
                        conditioner.last_condition_norm_ratio or 0.0),
                },
            })
            if args.prediction_bank_root:
                root = Path(args.prediction_bank_root).resolve()
                root.mkdir(parents=True, exist_ok=True)
                bank_path = root / f"seed_{seed}.npz"
                def objects(name):
                    value = np.empty(len(prediction_rows), dtype=object)
                    for row_index, row in enumerate(prediction_rows):
                        value[row_index] = row[name]
                    return value
                np.savez_compressed(
                    bank_path,
                    scene_ids=np.asarray([x["scene_id"] for x in prediction_rows]),
                    agent_ids=objects("agent_ids"), observed=objects("observed"),
                    future=objects("future"), predictions=objects("predictions"),
                    output_goals=objects("output_goals"),
                    native_goals=objects("native_goals"),
                )
                bank_index.append({
                    "seed": int(seed), "path": str(bank_path),
                    "sha256": sha256(bank_path), "scenes": len(prediction_rows),
                })
    metric_names = (
        "minADE", "minFDE", "JADE", "JFDE",
        "scene_weighted_minADE", "scene_weighted_minFDE",
        "coordination_gap_ADE", "coordination_gap_FDE")
    means = {
        name: float(np.mean([x["metrics"][name] for x in results]))
        for name in metric_names}
    sample_standard_deviation = {
        name: float(statistics.stdev([x["metrics"][name] for x in results]))
        for name in metric_names}
    manifest_payload = dataset.manifest
    checkpoint_sha256_after = sha256(args.checkpoint)
    if checkpoint_sha256_after != checkpoint_sha256_before:
        raise RuntimeError("checkpoint bytes changed during formal evaluation")
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()), "split": args.split,
        "checkpoint_sha256": checkpoint_sha256_before,
        "checkpoint_unchanged_during_evaluation": True,
        "checkpoint_epoch": int(payload["epoch"]),
        "manifest": str(Path(path).resolve()),
        "manifest_sha256": sha256(path),
        "scene_count": int(manifest_payload["statistics"]["scenes"]),
        "agent_occurrence_count": int(manifest_payload["statistics"]["agents"]),
        "validation_test_identity": evaluation["validation_test_identity"],
        "independent_test": False,
        "inference_batch_size": 1, "model_dtype": "fp32",
        "autocast_enabled": False, "ddim_eta": float(config["diffusion"]["ddim_eta"]),
        "per_agent_reordering": bool(evaluation["per_agent_reordering"]),
        "native_goal_conditioner_disabled": bool(
            args.disable_native_goal_conditioner),
        "pairing_rounds": int(args.pairing_rounds),
        "geometry_mapping_intervention": args.geometry_mapping,
        "endpoint_projection_radius": args.endpoint_projection_radius,
        "worlds": worlds, "steps": steps, "per_seed": results,
        "prediction_bank_index_local_only": bank_index,
        "mean": means, "sample_standard_deviation": sample_standard_deviation,
    }
    output = Path(args.output)
    atomic_json(summary, output)
    print(json.dumps(summary["mean"], sort_keys=True))


if __name__ == "__main__":
    main()

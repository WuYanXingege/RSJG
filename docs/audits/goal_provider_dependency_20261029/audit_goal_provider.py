#!/usr/bin/env python3
"""CPU-only dependency/replay audit for the native GDTS goal provider."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.optimize import linear_sum_assignment


ROOT = Path("/media/ee615/cede1227-a7b1-4ab4-b6cb-dc50ddd8a336/RSJG")
PAPER = ROOT / "RSJG_EBJD_Paper"
ASSETS = ROOT / "RSJG_EBJD_assets"
LEGACY = ROOT / "RSJG_JDV2_clean"
HERE = Path(__file__).resolve().parent
FOLDS = {
    "zara1": {
        "recipe": PAPER / "configs/asset_preparation/zara1_gdts.yaml",
        "goal": ASSETS / "checkpoints/zara1/goal_unet_best.pt",
        "matched": ASSETS / "checkpoints/zara1/matched_gdts_best.pt",
        "cache": ASSETS / "zara1_full_pipeline_v1/gdts_scene_cache",
    },
    "zara2": {
        "recipe": PAPER / "configs/asset_preparation/zara2_gdts.yaml",
        "goal": ASSETS / (
            "experiments/formal_zara2_gt_bridge_h_20261028_100627/"
            "gdts/goal_unet_best.pt"),
        "matched": ASSETS / (
            "experiments/formal_zara2_gt_bridge_h_20261028_100627/"
            "gdts/matched_gdts_best.pt"),
        "cache": ASSETS / (
            "experiments/formal_zara2_gt_bridge_h_20261028_100627/"
            "gdts/scene_cache"),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state(path: Path) -> tuple[dict, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload, payload["model_state_dict"]


def goal_state(source: dict) -> dict:
    prefix = "goal_module."
    result = {key[len(prefix):]: value for key, value in source.items()
              if key.startswith(prefix)}
    if len(result) != 46:
        raise RuntimeError(f"expected 46 Goal U-Net tensors, got {len(result)}")
    return result


def parameter_comparison() -> dict:
    rows, summary = [], {}
    for fold, item in FOLDS.items():
        goal_payload, goal_full = state(item["goal"])
        matched_payload, matched_full = state(item["matched"])
        goal = goal_state(goal_full)
        matched = goal_state(matched_full)
        if goal.keys() != matched.keys():
            raise RuntimeError(f"{fold}: explicit Goal U-Net key spaces differ")
        changed = 0
        for key in goal:
            left, right = goal[key], matched[key]
            if left.shape != right.shape:
                raise RuntimeError(f"{fold}: shape mismatch for {key}")
            equal = torch.equal(left, right)
            changed += int(not equal)
            rows.append({
                "fold": fold, "module": "goal_module",
                "parameter_or_buffer": key,
                "tensor_kind": "parameter",
                "pretrain_key": "goal_module." + key,
                "matched_key": "goal_module." + key,
                "shape": "x".join(map(str, left.shape)),
                "covered_by_goal_pretrain": True,
                "exact_equal": equal,
                "max_abs_diff": float((left.float() - right.float()).abs().max()),
            })
        summary[fold] = {
            "goal_pretrain": str(item["goal"]),
            "goal_pretrain_sha256": sha256(item["goal"]),
            "goal_pretrain_epoch": int(goal_payload["epoch"]),
            "matched": str(item["matched"]),
            "matched_sha256": sha256(item["matched"]),
            "matched_epoch": int(matched_payload["epoch"]),
            "matched_training_stage": matched_payload.get("training_stage"),
            "goal_pretrain_initialization": matched_payload.get(
                "goal_pretrain_initialization"),
            "required_tensors": len(goal), "changed_tensors": changed,
            "all_pretrain_tensors_covered": True,
            "has_running_buffers": any(
                "running_" in key or "num_batches_tracked" in key for key in goal),
        }
    with (HERE / "PARAMETER_COMPARISON.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (HERE / "PARAMETER_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def capture_rng() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state()}


def restore_rng(value: dict) -> None:
    random.setstate(value["python"]); np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])


def jsonable(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return value


def compare_candidates(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a[:20].float(), b[:20].float()
    ordered = torch.linalg.vector_norm(a - b, dim=-1)
    set_max = 0.0
    for agent in range(a.shape[1]):
        distances = torch.cdist(a[:, agent], b[:, agent]).cpu().numpy()
        rows, columns = linear_sum_assignment(distances)
        set_max = max(set_max, float(distances[rows, columns].max()))
    return {
        "candidate_count": int(a.shape[0]),
        "agent_count": int(a.shape[1]),
        "ordered_max_euclidean": float(ordered.max()),
        "set_assignment_max_euclidean": set_max,
        "ordered_exact": bool(torch.equal(a, b)),
        "set_exact": set_max == 0.0,
    }


def to_world(inputs: dict, candidates: torch.Tensor, down_factor: float) -> torch.Tensor:
    """Apply the exact native-export pixel-to-world conversion."""
    pixel = candidates * float(down_factor)
    return inputs["scene"].make_world_coord_torch(pixel).float()


def configure_reference(fold: str):
    item = FOLDS[fold]
    recipe = yaml.safe_load(item["recipe"].read_text(encoding="utf-8"))
    sys.path.insert(0, str(LEGACY))
    runtime = HERE / f"runtime_{fold}"
    runtime.mkdir(exist_ok=True)
    data_link = runtime / "data"
    source_data = Path(recipe["source_data_root"]).resolve()
    if not data_link.exists() and not data_link.is_symlink():
        data_link.symlink_to(source_data, target_is_directory=True)
    os.chdir(runtime)
    from src.models.model import GDTS
    from src.parser import check_and_add_additional_args, get_parser
    from src.trainer import trainer
    cli = [
        "--dataset", "eth5", "--test_set", fold, "--phase", "test",
        "--device", "cpu", "--batch_size", "64", "--num_workers", "0",
        "--down_factor", "8", "--skip_ts_window", "1",
        "--goal_model_type", "social", "--training_stage", "finetune",
        "--data_augmentation", "False", "--shuffle_train_batches", "False",
        "--shuffle_test_batches", "False", "--num_samples", "20",
        "--ddpm_step", "100", "--ddim_step", "20", "--trunk_stage_step", "30",
        "--reproducibility", "True", "--seed", "3101",
        "--batch_cache_root", str(item["cache"]), "--use_wandb", "False"]
    parsed = check_and_add_additional_args(get_parser().parse_args(cli))
    parsed.model_dir = str(runtime / "model"); parsed.save_dir = parsed.model_dir
    evaluator = trainer(parsed)
    parsed.goal_model_type = "independent"; parsed.training_stage = "baseline"
    parsed.use_scene_latent = False; parsed.use_dynamic_relation = False
    parsed.use_joint_energy = False; parsed.use_dependency_corrector = False
    parsed.jdv2_active = False; parsed.joint_goal_enabled = False
    evaluator.args = parsed
    evaluator.net = GDTS(parsed, torch.device("cpu")).cpu()
    evaluator._pending_training_state = None
    evaluator._load_state_file(str(item["matched"]), weights_only=True)
    evaluator.net.eval()
    return evaluator


def select_windows(loader) -> tuple[list[int], dict[int, int]]:
    counts = {}
    for index, (_, identity) in enumerate(loader):
        counts[index] = len(identity["agent_ids"])
    ordered = sorted(counts, key=lambda index: (counts[index], index))
    chosen = ordered[:4] + [ordered[int(i * (len(ordered) - 1) / 9)]
                            for i in range(1, 9)] + ordered[-4:]
    # Preserve selection order while eliminating rare quantile duplicates.
    unique = list(dict.fromkeys(chosen))
    for index in ordered:
        if len(unique) == 16:
            break
        if index not in unique:
            unique.append(index)
    return unique[:16], counts


def replay_fold(fold: str) -> dict:
    evaluator = configure_reference(fold)
    from minimal_goal_provider import NativeGoalProvider
    _, pretrain_full = state(FOLDS[fold]["goal"])
    _, matched_full = state(FOLDS[fold]["matched"])
    module_b = NativeGoalProvider(goal_state(pretrain_full)).cpu().eval()
    module_c = NativeGoalProvider(goal_state(matched_full)).cpu().eval()
    selected, counts = select_windows(evaluator.data_loaders["valid"])
    selected_set = set(selected)
    records = []
    with torch.no_grad():
        for index, (batch_data, batch_id) in enumerate(evaluator.data_loaders["valid"]):
            if index not in selected_set:
                continue
            inputs, _ = evaluator.net.prepare_inputs(batch_data, batch_id)
            scene_id = str(batch_id.get("manifest_scene_id", [f"index:{index}"]))
            scene_name = jsonable(batch_id.get("scene_name"))
            agent_ids = jsonable(batch_id.get("agent_ids"))
            for seed in (3101, 2035):
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                rng = capture_rng()
                captured = []
                hook = evaluator.net.goal_module.register_forward_hook(
                    lambda _m, _i, output: captured.append(output.detach().clone()))
                restore_rng(rng)
                _, aux_a1 = evaluator.net._legacy_encode(inputs, if_test=True)
                logits_a1 = captured.pop(); hook.remove()
                captured = []
                hook = evaluator.net.goal_module.register_forward_hook(
                    lambda _m, _i, output: captured.append(output.detach().clone()))
                restore_rng(rng)
                _, aux_a2 = evaluator.net._legacy_encode(inputs, if_test=True)
                logits_a2 = captured.pop(); hook.remove()
                if not torch.equal(logits_a1, logits_a2) or not torch.equal(
                        aux_a1["goal_point"], aux_a2["goal_point"]):
                    raise RuntimeError(f"{fold} reference A/A is not repeatable")
                restore_rng(rng)
                logits_b, goals_b = module_b(
                    inputs["tensor_image"], inputs["input_traj_maps"])
                restore_rng(rng)
                logits_c, goals_c = module_c(
                    inputs["tensor_image"], inputs["input_traj_maps"])
                # The provider emits map-grid coordinates.  Formal banks store
                # metres after down-factor restoration and the scene transform.
                a = to_world(inputs, aux_a1["goal_point"], evaluator.args.down_factor)
                goals_b = to_world(inputs, goals_b, evaluator.args.down_factor)
                goals_c = to_world(inputs, goals_c, evaluator.args.down_factor)
                record = {
                    "fold": fold, "loader_index": index, "scene_id": scene_id,
                    "scene_name": scene_name, "agent_ids": agent_ids,
                    "agents": counts[index], "seed": seed,
                    "A_A_logits_exact": True, "A_A_candidates_exact": True,
                    "A_B_logits_max_abs": float((logits_a1 - logits_b).abs().max()),
                    "A_C_logits_max_abs": float((logits_a1 - logits_c).abs().max()),
                    "A_B": compare_candidates(a, goals_b),
                    "A_C": compare_candidates(a, goals_c),
                }
                records.append(record)
    expected = 16 * 2
    if len(records) != expected:
        raise RuntimeError(f"{fold}: expected {expected} replay records, got {len(records)}")
    return {
        "fold": fold, "device": "cpu", "dtype": "float32", "K": 20,
        "seeds": [3101, 2035], "selected_loader_indices": selected,
        "agent_counts": [counts[index] for index in selected],
        "records": records,
        "A_A_repeatable_all": all(row["A_A_candidates_exact"] for row in records),
        "A_C_exact_all": all(row["A_C"]["ordered_exact"] and
                              row["A_C_logits_max_abs"] == 0 for row in records),
        "A_B_exact_all": all(row["A_B"]["ordered_exact"] and
                              row["A_B_logits_max_abs"] == 0 for row in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args()
    summary = parameter_comparison()
    if args.static_only:
        print(json.dumps(summary, sort_keys=True)); return
    replay = {fold: replay_fold(fold) for fold in FOLDS}
    (HERE / "REPLAY_RESULTS.json").write_text(
        json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({fold: {"A_C_exact_all": value["A_C_exact_all"],
                            "A_B_exact_all": value["A_B_exact_all"]}
                      for fold, value in replay.items()}, sort_keys=True))


if __name__ == "__main__":
    main()

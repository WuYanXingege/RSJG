"""Real CUDA/BF16 no-optimizer smoke for UNIV strict-no-z Stage A."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data._utils.collate import default_collate

from src.joint_dependency_v2_cache import load_cache_record, sha256_file
from src.parser import check_and_add_additional_args, get_parser
from src.trainer import trainer


def _module_hash(modules) -> str:
    digest = hashlib.sha256()
    for module in modules:
        for name, tensor in module.state_dict().items():
            digest.update(name.encode("utf-8"))
            value = tensor.detach().cpu().contiguous()
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(bytes(value.untyped_storage()))
    return digest.hexdigest()


def _all_finite(value) -> bool:
    if torch.is_tensor(value):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return True


def _load_args(config_path: Path):
    config = yaml.safe_load(config_path.read_text())
    parser = get_parser()
    unknown = set(config) - set(vars(parser.parse_args([])))
    if unknown:
        raise RuntimeError(f"unknown config keys: {sorted(unknown)}")
    parser.set_defaults(**config)
    args = parser.parse_args([])
    args.device = "cuda:0"
    args.num_workers = 0
    args.data_augmentation = False
    return check_and_add_additional_args(args)


def _find_edge_examples(cache_root: Path) -> dict[str, int]:
    examples = {}
    for path in sorted((cache_root / "train").glob("*.pt")):
        if path.name.endswith(".teacher.pt"):
            continue
        record = load_cache_record(
            str(path), allow_future_supervision=False)
        edge_count = int(record["edge_index"].shape[1])
        key = "e0" if edge_count == 0 else "e_gt0"
        examples.setdefault(key, int(path.stem))
        if len(examples) == 2:
            return examples
    raise RuntimeError("UNIV train cache lacks E=0 or E>0 smoke window")


def run_smoke(config_path: Path) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("UNIV readiness smoke requires real CUDA")
    args = _load_args(config_path)
    if args.training_stage != "joint_goal" or \
            args.jdv2_latent_objective != "strict_no_z" or \
            args.jdv2_refinement_policy != \
            "exact_lexicographic_persistent_tie":
        raise RuntimeError("UNIV smoke did not resolve frozen Stage-A method")
    expected_checkpoint_hash = (
        "ebfbae9de25463497c7bcc20981022f0638bc0349c7eeeb38bfc235dfaf61a3e")
    if sha256_file(args.jdv2_source_checkpoint) != expected_checkpoint_hash:
        raise RuntimeError("UNIV GDTS checkpoint authentication failed")

    torch.cuda.reset_peak_memory_stats()
    harness = trainer(args)
    net = harness.net.eval()
    dataset = harness.data_loaders["train"].dataset
    cache_root = Path(args.jdv2_cache_root)
    examples = _find_edge_examples(cache_root)

    baseline_modules = net._baseline_modules()
    baseline_before = _module_hash(baseline_modules)
    corrector_before = _module_hash((net.jdv2_corrector,))
    if any(parameter.requires_grad for module in baseline_modules
           for parameter in module.parameters()):
        raise RuntimeError("baseline GDTS is not frozen")
    if any(parameter.requires_grad for parameter in
           net.jdv2_corrector.parameters()):
        raise RuntimeError("DependencyCorrector is active in Stage A")

    smoke_windows = {}
    for ordinal, (kind, index) in enumerate(sorted(examples.items())):
        batch_data, batch_id = default_collate([dataset[index]])
        net.jdv2_sampler.set_sampling_context(2035, index)
        with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.bfloat16):
            inputs, _ = net.prepare_inputs(batch_data, batch_id)
            predictions, auxiliary = net(inputs, if_test=True)
        candidate_index = auxiliary["joint_candidate_index"][:, :20]
        num_agents = inputs["scene_index"].numel()
        edge_index = auxiliary["dependency_state"]["edge_index"]
        edge_count = edge_index.shape[1]
        unique = [torch.unique(row).numel() == 20 for row in candidate_index]
        if not all(unique):
            raise RuntimeError("deployed candidate coverage is not 20/20")
        if predictions.shape != (20, 20, num_agents, 2):
            raise RuntimeError("prediction interface shape mismatch")
        if auxiliary["joint_candidate_index"].shape != (num_agents, 21):
            raise RuntimeError("candidate-ID interface shape mismatch")
        if auxiliary["joint_goal_points_world"].shape != (
                num_agents, 21, 2):
            raise RuntimeError("world-goal interface shape mismatch")
        relation = auxiliary["dependency_state"]["relation_embedding"]
        if relation.shape != (edge_count, 20, 16):
            raise RuntimeError("relation interface shape mismatch")
        if edge_count and not bool(
                (inputs["scene_index"][edge_index[0]] ==
                 inputs["scene_index"][edge_index[1]]).all()):
            raise RuntimeError("cross-scene edge in smoke")
        if not _all_finite((predictions, auxiliary)):
            raise RuntimeError("non-finite Stage-A smoke output")
        if (kind == "e0") != (edge_count == 0):
            raise RuntimeError("smoke edge stratum mismatch")
        smoke_windows[kind] = {
            "train_index": index,
            "num_agents": num_agents,
            "edge_count": edge_count,
            "prediction_shape": list(predictions.shape),
            "joint_candidate_index_shape": list(
                auxiliary["joint_candidate_index"].shape),
            "joint_goal_points_world_shape": list(
                auxiliary["joint_goal_points_world"].shape),
            "relation_embedding_shape": list(relation.shape),
            "coverage_unique_per_agent": [
                int(torch.unique(row).numel()) for row in candidate_index],
            "finite": True,
            "no_cross_scene_edges": True,
        }

    baseline_after = _module_hash(baseline_modules)
    corrector_after = _module_hash((net.jdv2_corrector,))
    if baseline_before != baseline_after:
        raise RuntimeError("baseline GDTS changed during no-optimizer smoke")
    if corrector_before != corrector_after:
        raise RuntimeError("DependencyCorrector changed during Stage-A smoke")
    return {
        "status": "PASS",
        "device": torch.cuda.get_device_name(0),
        "precision": "bf16_autocast_with_frozen_fp32_islands",
        "optimizer_constructed": False,
        "source_checkpoint_hash": expected_checkpoint_hash,
        "cache_manifest_hash": args.jdv2_cache_manifest_hash,
        "cache_source_commit": args.jdv2_cache_source_commit,
        "strict_no_z": True,
        "refinement_policy": args.jdv2_refinement_policy,
        "baseline_gdts_frozen": True,
        "baseline_state_unchanged": True,
        "dependency_corrector_trainable": False,
        "dependency_corrector_state_unchanged": True,
        "source_cache_authenticated": True,
        "jdv2_cache_authenticated": True,
        "windows": smoke_windows,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_smoke(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

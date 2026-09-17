"""Explicit deterministic cache lifecycle for Joint Dependency V2."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch


JDV2_CACHE_SCHEMA_VERSION = "jdv2-cache-v1"
JDV2_MANIFEST = "manifest.json"
FUTURE_SUPERVISION_KEYS = frozenset({
    "future_position_world",
    "future_velocity_world",
    "future_pair_descriptor",
})
FORBIDDEN_LEARNED_CACHE_KEYS = frozenset({
    "agent_feat", "p_z", "q_z", "p_relation", "q_relation",
    "energy_factor", "corrector_feature", "learned_graph_gate",
    "future_gru_feature",
})


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash checkpoint bytes, never a path string."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_hash(paths: Iterable[str | os.PathLike[str]]) -> str:
    """Stable identity from ordered cache names, sizes and file bytes."""
    digest = hashlib.sha256()
    for raw_path in sorted(str(Path(path).resolve()) for path in paths):
        path = Path(raw_path)
        digest.update(path.name.encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _source_commit(worktree: str = ".") -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=worktree,
            stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def jdv2_cache_root(args) -> str:
    explicit = getattr(args, "jdv2_cache_root", None)
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    return os.path.join(
        os.path.abspath(getattr(args, "save_dir", ".")), "jdv2_cache")


def build_manifest(
    args,
    source_split_files: Mapping[str, Iterable[str]],
    goal_checkpoint_path: str,
    completed_splits: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    if not os.path.isfile(goal_checkpoint_path):
        raise FileNotFoundError(
            f"Frozen goal checkpoint not found: {goal_checkpoint_path}")
    split_hashes = {
        name: split_hash(paths) for name, paths in source_split_files.items()}
    preprocessing = {
        "dataset": args.dataset,
        "test_set": args.test_set,
        "obs_length": int(args.obs_length),
        "pred_length": int(args.pred_length),
        "down_factor": int(args.down_factor),
        "skip_ts_window": int(args.skip_ts_window),
    }
    manifest = {
        "schema_version": JDV2_CACHE_SCHEMA_VERSION,
        "source_commit": _source_commit(),
        "dataset_hash": stable_json_hash(split_hashes),
        "split_hash": split_hashes,
        "preprocess_config": preprocessing,
        "goal_checkpoint_hash": sha256_file(goal_checkpoint_path),
        "graph_config": {
            "type": args.graph_type,
            "radius": float(args.graph_radius),
            "ttc_threshold": float(args.ttc_threshold),
            "adaptive": False,
        },
        "coordinate_units": {
            "position": "world_m", "velocity": "world_m_per_s"},
        "dt": float(args.trajectory_dt),
        "K": int(args.num_goal_candidates),
        "dtype": "float32",
        "candidate_order": "generate_goal_candidates-v1",
        "deterministic_seed": int(args.seed),
        "contains_future_supervision": True,
        "completed_splits": sorted(completed_splits or ()),
        "creation_time": datetime.now(timezone.utc).isoformat(),
    }
    return manifest


def _assert_cache_record(record: Mapping[str, Any]) -> None:
    forbidden = FORBIDDEN_LEARNED_CACHE_KEYS.intersection(record)
    if forbidden:
        raise ValueError(
            "JDV2 cache contains trainable-network outputs: " +
            ", ".join(sorted(forbidden)))
    for name, value in record.items():
        if torch.is_tensor(value) and value.is_floating_point() and \
                not torch.isfinite(value).all():
            raise FloatingPointError(
                f"cache tensor {name} is non-finite: shape={tuple(value.shape)} "
                f"dtype={value.dtype}")


def atomic_torch_save(record: Mapping[str, Any], path: str) -> None:
    """Validate then atomically replace a single cache record."""
    _assert_cache_record(record)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    os.close(descriptor)
    try:
        torch.save(dict(record), temporary)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json_save(value: Mapping[str, Any], path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=target.name + ".", suffix=".tmp", dir=str(target.parent))
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_manifest(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    ignore_creation_time: bool = True,
) -> str:
    """Reject any architecture/data/checkpoint mismatch and return its hash."""
    ignored = {"creation_time", "completed_splits"} if ignore_creation_time \
        else set()
    comparable_actual = {k: v for k, v in actual.items() if k not in ignored}
    comparable_expected = {k: v for k, v in expected.items() if k not in ignored}
    if comparable_actual != comparable_expected:
        mismatched = sorted(
            key for key in set(comparable_actual) | set(comparable_expected)
            if comparable_actual.get(key) != comparable_expected.get(key))
        raise RuntimeError(
            "JDV2 cache manifest mismatch for " + ", ".join(mismatched) +
            "; run --phase build-jdv2-cache")
    return stable_json_hash(dict(actual))


def load_cache_record(path: str, *, allow_future_supervision: bool) -> Dict[str, Any]:
    record = torch.load(path, map_location="cpu")
    if not isinstance(record, dict):
        raise TypeError("JDV2 cache record must be a mapping")
    _assert_cache_record(record)
    if not allow_future_supervision:
        leaking = FUTURE_SUPERVISION_KEYS.intersection(record)
        if leaking:
            raise RuntimeError(
                "Future-supervision cache fields are unavailable during "
                "test/deployment: " + ", ".join(sorted(leaking)))
    return record


class JointDependencyV2CacheBuilder:
    """Build frozen candidates, sparse graph and teacher descriptors.

    Pseudocode::

        load synchronized source windows and frozen goal checkpoint
        for split/window:
            run frozen Goal U-Net -> ordered K candidates + log priors
            build parameter-free sparse graph
            compute deterministic future pair descriptors
            atomically write record
        atomically publish complete manifest
    """

    def __init__(self, args) -> None:
        self.args = args
        if args.goal_model_type != "joint_dependency_v2" or \
                not args.jdv2_active:
            raise ValueError("JDV2 cache builder requires active JDV2")
        checkpoint = args.jdv2_source_checkpoint or args.pretrain_path
        if checkpoint is None:
            raise ValueError(
                "build-jdv2-cache requires --jdv2_source_checkpoint")
        self.checkpoint = os.path.abspath(os.path.expanduser(checkpoint))
        self.root = jdv2_cache_root(args)

    def build(self) -> None:
        # Imports stay lazy so manifest/cache unit tests do not require the
        # image stack or construct a model.
        from src.data_grouping import batch_cache_path
        from src.data_loader import get_dataloader
        from src.models.interaction_graph import build_interaction_graph
        from src.models.model import GDTS
        from src.models.model_utils.sampling_2D_map import generate_goal_candidates
        from src.models.joint_dependency_v2.future_teacher import (
            future_pair_descriptor,
        )

        device = torch.device(self.args.device)
        model = GDTS(self.args, device).to(device).eval()
        checkpoint = torch.load(self.checkpoint, map_location=device)
        state = checkpoint.get("model_state_dict", checkpoint)
        goal_state = {
            key[len("goal_module."):]: value
            for key, value in state.items() if key.startswith("goal_module.")}
        if not goal_state:
            goal_state = state
        model.goal_module.load_state_dict(goal_state, strict=True)
        for parameter in model.goal_module.parameters():
            parameter.requires_grad_(False)

        source_root = Path(batch_cache_path(self.args))
        source_files = {
            split: sorted(str(path) for path in
                          (source_root / f"{split}_batches").glob("*.pkl*"))
            for split in ("train", "valid", "test")}
        completed = []
        with torch.no_grad():
            for split in ("train", "valid", "test"):
                # Cache record index must match the sorted immutable source id,
                # independent of the training shuffle policy.
                original_train_shuffle = self.args.shuffle_train_batches
                original_test_shuffle = self.args.shuffle_test_batches
                self.args.shuffle_train_batches = False
                self.args.shuffle_test_batches = False
                loader = get_dataloader(self.args, set_name=split)
                self.args.shuffle_train_batches = original_train_shuffle
                self.args.shuffle_test_batches = original_test_shuffle
                target_dir = Path(self.root) / split
                target_dir.mkdir(parents=True, exist_ok=True)
                for index, (batch_data, batch_id) in enumerate(loader):
                    inputs, _ = model.prepare_inputs(batch_data, batch_id)
                    num_agents = inputs["x_augmented"].shape[1]
                    image = inputs["tensor_image"].unsqueeze(0).repeat(
                        num_agents, 1, 1, 1)
                    maps = inputs["input_traj_maps"][:, :self.args.obs_length]
                    logits = model.goal_module(torch.cat((image, maps), dim=1))
                    probability_map = torch.sigmoid(logits[:, -1:])
                    goals_map, probability = generate_goal_candidates(
                        probability_map,
                        num_candidates=self.args.num_goal_candidates,
                        device=device, use_ttst=self.args.use_ttst)
                    goals_world = model._map_goals_to_world(
                        goals_map, inputs["scene"])
                    edge_index, edge_feat, edge_weight = build_interaction_graph(
                        inputs["obs_traj_world"], inputs["scene_index"],
                        graph_type=self.args.graph_type,
                        radius=self.args.graph_radius,
                        ttc_threshold=self.args.ttc_threshold,
                        dt=self.args.trajectory_dt)
                    future = inputs["world_coord"][
                        self.args.obs_length:].permute(1, 0, 2)
                    descriptor = future_pair_descriptor(
                        future, inputs["obs_traj_world"][:, -1], edge_index)
                    record = {
                        "cache_id": f"{split}-{index:06d}",
                        "scene_index": inputs["scene_index"].cpu(),
                        "scene_ptr": inputs["scene_ptr"].cpu(),
                        "frame_ids": inputs["frame_ids"].cpu(),
                        "edge_index": edge_index.cpu(),
                        "edge_feat": edge_feat.cpu(),
                        "edge_weight": edge_weight.cpu(),
                        "goal_candidates_map": goals_map.cpu(),
                        "goal_candidates_world": goals_world.cpu(),
                        "candidate_log_prior": probability.float().clamp_min(
                            1e-8).log().cpu(),
                        "contains_future_supervision": False,
                    }
                    atomic_torch_save(
                        record, str(target_dir / f"{index:06d}.pt"))
                    atomic_torch_save({
                        "cache_id": f"{split}-{index:06d}",
                        "future_pair_descriptor": descriptor.cpu(),
                        "contains_future_supervision": True,
                    }, str(target_dir / f"{index:06d}.teacher.pt"))
                completed.append(split)
        manifest = build_manifest(
            self.args, source_files, self.checkpoint,
            completed_splits=completed)
        atomic_json_save(manifest, os.path.join(self.root, JDV2_MANIFEST))
        print(f"JDV2 cache built at {self.root}")


__all__ = [
    "FORBIDDEN_LEARNED_CACHE_KEYS",
    "FUTURE_SUPERVISION_KEYS",
    "JDV2_CACHE_SCHEMA_VERSION",
    "JDV2_MANIFEST",
    "JointDependencyV2CacheBuilder",
    "atomic_json_save",
    "atomic_torch_save",
    "build_manifest",
    "jdv2_cache_root",
    "load_cache_record",
    "sha256_file",
    "split_hash",
    "stable_json_hash",
    "validate_manifest",
]

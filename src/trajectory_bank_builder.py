"""Offline frozen-upstream trajectory-bank construction."""

from __future__ import annotations

import hashlib
import os
import random
import time
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch
from tqdm import tqdm

from src.data_loader import get_dataloader
from src.models.model import GDTS
from src.trajectory_bank_cache import (
    CACHE_FORMAT_VERSION,
    atomic_torch_save,
    deterministic_trajectory_seed,
    finalize_manifest,
    prepare_cache_root,
    validate_cache_record,
    write_split_index,
)


def _set_all_rng(seed: int, use_cuda: bool) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _collated_scalar(value):
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _collated_scalar(value[0])
    if torch.is_tensor(value) and value.numel() == 1:
        return value.item()
    return value


def _collated_integer_list(value):
    if torch.is_tensor(value):
        return [int(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            if torch.is_tensor(item):
                result.extend(int(v) for v in item.reshape(-1).tolist())
            elif isinstance(item, (list, tuple)):
                result.extend(_collated_integer_list(item))
            else:
                result.append(int(item))
        return result
    return [int(value)]


class TrajectoryBankCacheBuilder:
    """Generate immutable multi-seed K=20 proposal banks for each window."""

    def __init__(self, args):
        if args.training_stage != "multiway_coupling" or \
                args.trajectory_coupling != "multiway_v4":
            raise ValueError("Trajectory banks are defined for V4 coupling only")
        if not args.freeze_upstream_generator:
            raise ValueError("Trajectory-bank caching requires frozen upstream")
        self.args = args
        self.root, self.manifest = prepare_cache_root(args, for_build=True)
        self.device = torch.device(args.device)
        self.model = GDTS(args, self.device).to(self.device)
        checkpoint = torch.load(
            args.pretrain_path, map_location=self.device, weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint)
        incompatible = self.model.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "Unexpected keys in frozen upstream checkpoint: " +
                repr(incompatible.unexpected_keys))
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        print(f"Trajectory-bank cache root: {self.root}")
        print(
            "Cache contract: "
            f"S_seed={args.num_cached_seeds_per_window}, "
            f"K={args.num_samples}; seeds are stored separately and never "
            "merged along K.")

    def _record_path(self, split: str, position: int,
                     window_id: str) -> tuple[Path, str]:
        digest = hashlib.sha256(window_id.encode("utf-8")).hexdigest()[:12]
        filename = f"window_{position:06d}_{digest}.pt"
        return self.root / split / filename, filename

    @torch.inference_mode()
    def _generate_record(self, inputs: Mapping[str, torch.Tensor],
                         batch_id: Mapping, split: str,
                         position: int) -> Dict[str, object]:
        scene_name = str(_collated_scalar(batch_id["scene_name"]))
        frame_ids = inputs["frame_ids"][:, 0].detach().cpu().long()
        frame_start = int(frame_ids[0])
        agent_ids = _collated_integer_list(batch_id["agent_ids"])
        identity = (
            f"{split}|{scene_name}|{frame_start}|"
            f"{','.join(str(value) for value in agent_ids)}"
        )
        window_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        seed_values = []
        banks = []
        for q in range(int(self.args.num_cached_seeds_per_window)):
            seed = deterministic_trajectory_seed(
                self.args.trajectory_bank_seed_base, scene_name,
                frame_start, q)
            _set_all_rng(seed, self.args.use_cuda)
            predictions, _ = self.model._v4_upstream_predictions(inputs)
            raw_world = self.model._future_predictions_world(
                predictions, inputs)
            if raw_world.shape[1] != int(self.args.num_samples):
                raise AssertionError("Upstream cache generation changed K")
            seed_values.append(seed)
            banks.append(raw_world.detach().cpu().float())

        seq_list = inputs["seq_list"].detach()
        gt = inputs["world_coord"][self.args.obs_length:].permute(
            1, 0, 2).contiguous()
        future_mask = seq_list[self.args.obs_length:].permute(
            1, 0).bool().contiguous()
        agent_mask = seq_list.cumprod(dim=0)[-1].bool()
        checkpoint = self.manifest["upstream_checkpoint"]
        return {
            "window_id": window_id,
            "logical_split": split,
            "scene_name": scene_name,
            "frame_ids": frame_ids,
            "agent_ids": torch.tensor(agent_ids, dtype=torch.long),
            "data_file_path": str(_collated_scalar(
                batch_id.get("data_file_path", ""))),
            "obs_world": inputs["obs_traj_world"].detach().cpu().float(),
            "gt_future_world": gt.detach().cpu().float(),
            "future_mask": future_mask.detach().cpu(),
            "agent_mask": agent_mask.detach().cpu(),
            "raw_trajectory_banks": torch.stack(banks, dim=0),
            "seed_values": seed_values,
            "metadata": {
                "cache_format_version": CACHE_FORMAT_VERSION,
                "upstream_checkpoint_id": checkpoint["sha256"],
                "upstream_checkpoint_path": checkpoint["path"],
                "upstream_generator_type": self.args.upstream_generator,
                "K": int(self.args.num_samples),
                "T_obs": int(self.args.obs_length),
                "T_pred": int(self.args.pred_length),
                "dt": float(self.args.trajectory_dt),
                "coordinate_system": "world_metres",
                "seed_axis_semantics": "independent_K_per_seed",
                "position": int(position),
            },
        }

    def _build_split(self, split: str) -> int:
        # Raw synchronized-window loaders are deliberately retained as the
        # online ablation.  Force deterministic order for resumable filenames.
        original_train_shuffle = self.args.shuffle_train_batches
        original_test_shuffle = self.args.shuffle_test_batches
        self.args.shuffle_train_batches = False
        self.args.shuffle_test_batches = False
        try:
            loader = get_dataloader(self.args, set_name=split)
        finally:
            self.args.shuffle_train_batches = original_train_shuffle
            self.args.shuffle_test_batches = original_test_shuffle

        index_records = []
        split_dir = self.root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        bar = tqdm(loader, desc=f"Cache {split}", ascii=True, ncols=100)
        for position, (batch_data, batch_id) in enumerate(bar):
            if (self.args.fast_debug and
                    position >= int(self.args.fast_debug_num)):
                break
            inputs, _ = self.model.prepare_inputs(batch_data, batch_id)
            scene_name = str(_collated_scalar(batch_id["scene_name"]))
            frame_start = int(inputs["frame_ids"][0, 0].item())
            agent_ids = _collated_integer_list(batch_id["agent_ids"])
            identity = (
                f"{split}|{scene_name}|{frame_start}|"
                f"{','.join(str(value) for value in agent_ids)}"
            )
            window_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            path, filename = self._record_path(split, position, window_id)
            if path.exists():
                record = torch.load(
                    path, map_location="cpu", weights_only=False)
                validate_cache_record(record, self.manifest)
                if record["window_id"] != window_id:
                    raise RuntimeError(f"Cache identity mismatch at {path}")
            else:
                record = self._generate_record(
                    inputs, batch_id, split, position)
                validate_cache_record(record, self.manifest)
                atomic_torch_save(record, path)
            num_agents = int(record["obs_world"].shape[0])
            index_records.append({
                "file": filename,
                "window_id": record["window_id"],
                "scene_name": record["scene_name"],
                "frame_start": int(record["frame_ids"][0]),
                "num_agents": num_agents,
                "edge_upper_bound": num_agents * (num_agents - 1) // 2,
            })
            bar.set_postfix(
                N=num_agents, seeds=len(record["seed_values"]),
                K=record["raw_trajectory_banks"].shape[2])
            del inputs, batch_data, record
        heldout_source = getattr(
            loader.dataset, "internal_validation_source", None)
        write_split_index(
            self.root, split, index_records, heldout_source)
        return len(index_records)

    def build(self) -> Dict[str, int]:
        started = time.perf_counter()
        counts = {split: self._build_split(split)
                  for split in ("train", "valid", "test")}
        finalize_manifest(self.root, self.manifest, counts)
        elapsed = time.perf_counter() - started
        print(
            f"Trajectory-bank cache complete: {counts}, "
            f"elapsed={elapsed:.1f}s, root={self.root}")
        return counts


__all__ = ["TrajectoryBankCacheBuilder"]

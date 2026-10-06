#!/usr/bin/env python3
"""Separate-process real-CUDA resume probe for HOTEL A1."""

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from src.jdv2_objective_state import FIELD, MC
from src.parser import check_and_add_additional_args
from src.p2_checkpoint import state_hash
from src.trainer import trainer
from src.utils import set_seed


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, path)


def same(left, right):
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            same(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            same(a, b) for a, b in zip(left, right))
    return left == right


def resolved(config_path, runtime, load_checkpoint, step_cap):
    with open(config_path) as handle:
        fields = yaml.safe_load(handle)
    fields.update({
        "phase": "train", "run_name": "hotel_a1_resume_probe",
        "load_checkpoint": load_checkpoint, "jdv2_step_cap": step_cap,
        "jdv2_arm_time_limit_seconds": 1800.0,
        "jdv2_queue_deadline_utc": None, "jdv2_numbered_resume": False,
        "jdv2_full_resume_state": True, "use_wandb": False,
    })
    args = check_and_add_additional_args(SimpleNamespace(**fields))
    args.save_dir = str(runtime)
    args.model_dir = str(runtime)
    args.config = str(runtime / "config.yaml")
    return args


def initialize_training(owner, epoch):
    owner.net.configure_training_epoch(epoch)
    owner.optimizer = owner._set_optimizer(owner._optimizer_parameter_groups())
    owner.scheduler = owner._set_scheduler(owner.optimizer)
    owner._stage_optimizer_steps_completed = 0
    owner._optimizer_attempts = 0
    owner._skipped_optimizer_updates = 0
    owner._failed_optimizer_updates = 0


def produce(config_path, runtime, receipt):
    args = resolved(config_path, runtime, None, 2)
    if args.jdv2_goal_objective != MC:
        raise RuntimeError("Resume probe requires the A1 MC config")
    set_seed(args.seed, use_cuda=True)
    owner = trainer(args)
    initialize_training(owner, 1)
    owner._arm_started_monotonic = time.monotonic()
    owner._train_epoch(1)
    owner.scheduler.step()
    owner._mc_complete_epoch(1)
    owner._save_checkpoint(1, last_epoch=True)
    checkpoint_path = runtime / "saved_models" / "last_model.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not payload.get("resume_safe") or payload.get("runtime_rng_state") is None:
        raise RuntimeError("Produced checkpoint is not a full safe resume")
    atomic_json(receipt, {
        "status": "PASS", "phase": "produce", "checkpoint": str(checkpoint_path),
        "epoch": payload["epoch"], "model_state_hash": state_hash(
            payload["model_state_dict"]),
        "optimizer_attempts": payload["optimizer_attempts"],
        "successful_updates": payload["stage_optimizer_steps_completed"],
        "draw_calls": payload[FIELD]["draw_calls"],
        "sampled_agent_draws": payload[FIELD]["sampled_agent_draws"],
        "resume_safe": payload["resume_safe"],
        "checkpoint_role": payload["checkpoint_role"],
    })


def consume(config_path, runtime, receipt):
    args = resolved(config_path, runtime, "last", 3)
    set_seed(args.seed, use_cuda=True)
    owner = trainer(args)
    start_epoch = owner._load_or_restart()
    initialize_training(owner, start_epoch)
    checkpoint = owner._pending_training_state
    owner._restore_mc_training_state(checkpoint)
    owner._restore_best_state(checkpoint)
    owner._restore_stopping_state(checkpoint, same_stage=True)
    owner._restore_jdv2_runtime_state(checkpoint)
    if not same(owner._runtime_rng_state(), checkpoint["runtime_rng_state"]):
        raise RuntimeError("Runtime RNG/loader generator state did not restore exactly")
    if state_hash(owner.net.state_dict()) != state_hash(
            checkpoint["model_state_dict"]):
        raise RuntimeError("Model state changed during resume")
    before = {
        "optimizer_attempts": owner._optimizer_attempts,
        "successful_updates": owner._stage_optimizer_steps_completed,
        "draw_calls": owner.jdv2_objective_rng.draw_calls,
        "sampled_agent_draws": owner.jdv2_objective_rng.sampled_agent_draws,
    }
    owner._arm_started_monotonic = time.monotonic()
    owner._train_epoch(start_epoch)
    after = {
        "optimizer_attempts": owner._optimizer_attempts,
        "successful_updates": owner._stage_optimizer_steps_completed,
        "draw_calls": owner.jdv2_objective_rng.draw_calls,
        "sampled_agent_draws": owner.jdv2_objective_rng.sampled_agent_draws,
    }
    if start_epoch != 2 or before["optimizer_attempts"] != 2 or \
            after["optimizer_attempts"] != 3 or \
            after["successful_updates"] != 3 or after["draw_calls"] != 3:
        raise RuntimeError("Resume continuation counters are inconsistent")
    atomic_json(receipt, {
        "status": "PASS", "phase": "consume", "start_epoch": start_epoch,
        "immediate_runtime_state_exact": True, "before": before,
        "after_one_additional_update": after,
        "scope": "metadata/RNG/loader/optimizer restore exact; CUDA replay bitwise equality not claimed",
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("produce", "consume"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    runtime = Path(args.runtime).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    action = produce if args.mode == "produce" else consume
    action(Path(args.config), runtime, Path(args.receipt))


if __name__ == "__main__":
    main()

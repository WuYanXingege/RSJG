#!/usr/bin/env python3
"""Two-update real CUDA acceptance for paired HOTEL Stage-A arms."""

import argparse
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from src.jdv2_objective_state import MC
from src.parser import check_and_add_additional_args
from src.p2_checkpoint import state_hash
from src.trainer import trainer
from src.utils import set_seed


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, path)


def run_arm(config_path: Path, runtime_root: Path, steps: int = 2) -> dict:
    with config_path.open() as handle:
        fields = yaml.safe_load(handle)
    label = f'{fields["jdv2_goal_objective"]}_seed{fields["seed"]}'
    fields.update({
        "phase": "train", "jdv2_step_cap": steps,
        "jdv2_arm_time_limit_seconds": 1800.0,
        "jdv2_queue_deadline_utc": None,
        "jdv2_full_resume_state": False,
        "jdv2_numbered_resume": False,
        "run_name": f"hotel_smoke_{label}",
        "use_wandb": False,
    })
    # Saved YAML deliberately records derived/transient fields as null.  The
    # production entrypoint recomputes them before constructing the loader;
    # the direct smoke harness must exercise that same derivation.
    args = check_and_add_additional_args(SimpleNamespace(**fields))
    args.save_dir = str(runtime_root / label)
    args.model_dir = str(runtime_root / label)
    args.config = str(runtime_root / label / "config.yaml")
    set_seed(args.seed, use_cuda=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    owner = trainer(args)
    initial_hash = state_hash(owner.net.state_dict())
    owner.net.configure_training_epoch(1)
    owner.optimizer = owner._set_optimizer(owner._optimizer_parameter_groups())
    owner.scheduler = owner._set_scheduler(owner.optimizer)
    owner._stage_optimizer_steps_completed = 0
    owner._optimizer_attempts = 0
    owner._skipped_optimizer_updates = 0
    owner._failed_optimizer_updates = 0
    owner._arm_started_monotonic = time.monotonic()
    started = time.monotonic()
    losses = owner._train_epoch(1)
    elapsed = time.monotonic() - started
    owner._assert_jdv2_frozen_contract()
    if owner._optimizer_attempts != steps or \
            owner._stage_optimizer_steps_completed != steps:
        raise RuntimeError("Update cap did not stop on the exact boundary")
    if not getattr(owner, "_step_cap_reached", False):
        raise RuntimeError("Step cap was not observed before the next fetch")
    if not all(math.isfinite(float(value)) for value in losses.values()):
        raise RuntimeError("Smoke losses are non-finite")
    diagnostics = owner.last_train_joint_diagnostics
    if diagnostics.get("num_edges", 0.0) <= 0:
        raise RuntimeError("Two-update smoke did not cover a real E>0 window")
    mc = None
    if fields["jdv2_goal_objective"] == MC:
        objective = owner.jdv2_objective_rng
        mc = {
            "draw_calls": objective.draw_calls,
            "sampled_agent_draws": objective.sampled_agent_draws,
            "successful_optimizer_updates":
                objective.successful_optimizer_updates,
            "pending_backward": objective.pending_backward,
        }
        if mc["draw_calls"] != steps or mc["pending_backward"]:
            raise RuntimeError("MC owner lifecycle mismatch in smoke")
    return {
        "status": "PASS", "config": str(config_path), "label": label,
        "objective": fields["jdv2_goal_objective"], "seed": fields["seed"],
        "initial_state_hash": initial_hash,
        "frozen_state_hash": owner._jdv2_frozen_state_sha256,
        "optimizer_attempts": owner._optimizer_attempts,
        "successful_updates": owner._stage_optimizer_steps_completed,
        "gradient_parameter_count": len(owner._gradient_parameter_names),
        "gradient_parameters": sorted(owner._gradient_parameter_names),
        "registered_trainable_parameters": int(sum(
            value.numel() for value in owner.net.parameters()
            if value.requires_grad)),
        "losses": {key: float(value) for key, value in losses.items()},
        "joint_diagnostics": diagnostics, "mc": mc,
        "elapsed_seconds": elapsed,
        "seconds_per_update": elapsed / float(steps),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a0-config", required=True)
    parser.add_argument("--a1-config", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--steps", default=2, type=int)
    args = parser.parse_args()
    if args.steps < 2:
        raise ValueError("At least two updates are required")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 is required for HOTEL A0/A1")
    runtime = Path(args.runtime_root).resolve()
    a0 = run_arm(Path(args.a0_config), runtime / "A0", args.steps)
    a1 = run_arm(Path(args.a1_config), runtime / "A1", args.steps)
    if a0["initial_state_hash"] != a1["initial_state_hash"]:
        raise RuntimeError("Paired A0/A1 model initial states differ")
    if a0["frozen_state_hash"] != a1["frozen_state_hash"]:
        raise RuntimeError("Paired A0/A1 frozen families differ")
    if max(a0["peak_cuda_reserved_bytes"],
           a1["peak_cuda_reserved_bytes"]) > 12 * 1024 ** 3:
        raise RuntimeError("Smoke exceeded the 12 GiB reserved-memory gate")
    atomic_json(Path(args.receipt), {
        "status": "PASS", "device": torch.cuda.get_device_name(0),
        "bf16_supported": True, "A0": a0, "A1": a1,
        "scope": ("independent initialized copies; "
                  f"{args.steps} real updates per arm"),
    })


if __name__ == "__main__":
    main()

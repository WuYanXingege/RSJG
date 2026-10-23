#!/usr/bin/env python3
"""Resumable serial ZARA1 asset preparation and EBJD training pipeline."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
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


def run_checked(command: list[str], *, cwd: Path, capture: bool = False) -> str:
    result = subprocess.run(
        command, cwd=cwd, check=True, text=True,
        capture_output=capture, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    return result.stdout.strip() if capture else ""


def git_output(root: Path, *arguments: str) -> str:
    return run_checked(["git", "-C", str(root), *arguments], cwd=root, capture=True)


def is_pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def tail_text(path: Path, limit: int = 512_000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - limit))
        return handle.read().decode("utf-8", errors="replace")


def legacy_progress(log_path: Path, cache_root: Path, fold: str = "zara1") -> dict:
    text = tail_text(log_path)
    matches = re.findall(
        r"Epoch\s+(\d+)\.\s+Train batches:.*?(\d+)/(\d+)",
        text.replace("\r", "\n"), flags=re.DOTALL)
    if matches:
        epoch, within, total = (int(value) for value in matches[-1])
        return {"epoch": epoch, "update": (epoch - 1) * total + within,
                "updates_in_epoch": within, "updates_per_epoch": total}
    confirmation = cache_root / "eth5" / fold / "data_batches" / "finished_train_batches.txt"
    total_match = re.search(r"(\d+)", confirmation.read_text()) if confirmation.is_file() else None
    return {"epoch": 0, "update": 0,
            "updates_per_epoch": int(total_match.group(1)) if total_match else None}


def ebjd_progress(log_path: Path) -> dict:
    train_jsonl_match = re.search(r"output=([^\s]+)", tail_text(log_path, 64_000))
    if not train_jsonl_match:
        return {"epoch": 0, "update": 0}
    log = Path(train_jsonl_match.group(1)) / "train.jsonl"
    if not log.is_file() or not log.stat().st_size:
        return {"epoch": 0, "update": 0, "train_jsonl": str(log)}
    last = tail_text(log, 64_000).strip().splitlines()[-1]
    record = json.loads(last)
    return {"epoch": int(record["epoch"]), "update": int(record["update"]),
            "train_jsonl": str(log)}


def gpu_processes() -> list[dict]:
    command = [
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError("nvidia-smi failed: " + result.stderr.strip())
    processes = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [value.strip() for value in line.split(",", 2)]
        processes.append({"pid": int(fields[0]), "name": fields[1],
                          "used_memory_mib": int(fields[2])})
    return processes


def merge(left: dict, right: dict) -> dict:
    result = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


class Pipeline:
    def __init__(self, recipe_path: Path, dry_run: bool = False) -> None:
        self.recipe_path = recipe_path.resolve()
        self.recipe = yaml.safe_load(self.recipe_path.read_text(encoding="utf-8"))
        self.paper = Path(self.recipe["paper_root"]).resolve()
        self.assets = Path(self.recipe["asset_root"]).resolve()
        self.root = Path(self.recipe["pipeline_root"]).resolve()
        self.runtime = self.root / "gdts_runtime"
        self.independent_cache = self.root / "gdts_independent_cache"
        self.scene_cache = self.root / "gdts_scene_cache"
        self.logs = self.root / "logs"
        self.status_path = self.root / "STATUS.json"
        self.python = str(Path(self.recipe["python"]).resolve())
        self.dry_run = dry_run
        self.initial_commit = git_output(self.paper, "rev-parse", "HEAD")
        fresh_status = {
            "schema": "ebjd-zara1-serial-pipeline-v1",
            "fold": "zara1", "training_seed": int(self.recipe["seed"]),
            "pipeline_source_commit": self.initial_commit,
            "stage": "initializing", "status": "RUNNING",
            "epoch": 0, "update": 0, "child_pid": None,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "failed_reason": None,
        }
        if self.status_path.is_file() and not dry_run:
            previous = json.loads(self.status_path.read_text(encoding="utf-8"))
            if previous.get("schema") != fresh_status["schema"]:
                raise RuntimeError("existing pipeline status has an unknown schema")
            self.status = previous
            prior_source = self.status.get("pipeline_source_commit")
            if prior_source != self.initial_commit:
                self.status["previous_pipeline_source_commit"] = prior_source
            self.status["pipeline_source_commit"] = self.initial_commit
            self.status["supervisor_restarted_at"] = datetime.now(timezone.utc).isoformat()
        else:
            self.status = fresh_status
        self.status["supervisor_pid"] = os.getpid()

    def persist(self, **updates) -> None:
        self.status.update(updates)
        self.status["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(self.status_path, self.status)

    def ensure_layout(self) -> None:
        self.logs.mkdir(parents=True, exist_ok=True)
        self.runtime.mkdir(parents=True, exist_ok=True)
        data_link = self.runtime / "data"
        source = Path(self.recipe["source_data_root"]).resolve()
        if not data_link.exists() and not data_link.is_symlink():
            data_link.symlink_to(source, target_is_directory=True)
        if data_link.resolve() != source:
            raise RuntimeError("GDTS runtime data symlink target mismatch")
        legacy = Path(self.recipe["legacy_gdts"]["root"]).resolve()
        if git_output(legacy, "rev-parse", "HEAD") != self.recipe["legacy_gdts"]["commit"]:
            raise RuntimeError("fixed legacy GDTS source commit mismatch")
        for manifest in self.recipe["manifests"].values():
            if not Path(manifest).is_file():
                raise FileNotFoundError(manifest)

    def wait_for_idle_gpu(self, stage: str) -> None:
        while True:
            processes = gpu_processes()
            if not processes:
                return
            self.persist(stage=stage, status="WAITING_FOR_GPU",
                         gpu_processes=processes, child_pid=None)
            time.sleep(60)

    def run_stage(
        self, stage: str, command: list[str], log_path: Path,
        progress=None, gpu: bool = True,
    ) -> None:
        if self.dry_run:
            print(json.dumps({"stage": stage, "command": command}))
            return
        if gpu:
            self.wait_for_idle_gpu(stage)
        existing_pid = self.status.get("child_pid") if self.status.get("stage") == stage else None
        if existing_pid and is_pid_running(int(existing_pid)):
            self.persist(stage=stage, status="RUNNING_EXISTING_CHILD")
            while is_pid_running(int(existing_pid)):
                if progress:
                    self.persist(**progress())
                time.sleep(15)
        else:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("ab", buffering=0) as log_handle:
                process = subprocess.Popen(
                    command, cwd=self.paper, stdout=log_handle,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    env={**os.environ, "PYTHONUNBUFFERED": "1",
                         "PYTHONDONTWRITEBYTECODE": "1"})
                self.persist(stage=stage, status="RUNNING", child_pid=process.pid,
                             log_path=str(log_path), command=command,
                             gpu_processes=(gpu_processes() if gpu else []))
                while process.poll() is None:
                    if progress:
                        self.persist(**progress())
                    time.sleep(15)
                if progress:
                    self.persist(**progress())
                self.persist(child_pid=None, exit_code=process.returncode)
                if process.returncode:
                    raise RuntimeError(
                        f"stage {stage} failed with exit {process.returncode}; "
                        f"see {log_path}")

    def legacy_command(self, stage: str, cache: Path, goal: Path | None = None) -> list[str]:
        command = [
            self.python, "-u", "-B", str(self.paper / "scripts" / "run_legacy_gdts_stage.py"),
            "--recipe", str(self.recipe_path), "--stage", stage,
            "--runtime-root", str(self.runtime), "--batch-cache-root", str(cache),
        ]
        if goal is not None:
            command += ["--goal-checkpoint", str(goal)]
        return command

    def copy_atomic(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)

    @staticmethod
    def checkpoint_epoch(path: Path) -> int | None:
        if not path.is_file():
            return None
        try:
            return int(torch.load(path, map_location="cpu", weights_only=False)["epoch"])
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            return None

    def validate_goal(self, source: Path, destination: Path) -> str:
        last = source.parent / "goal_pretrain_last_model.pt"
        payload = torch.load(source, map_location="cpu", weights_only=False)
        last_payload = torch.load(last, map_location="cpu", weights_only=False)
        if payload.get("checkpoint_type") != "gdts_goal_pretrain":
            raise RuntimeError("selected goal checkpoint has the wrong type")
        if payload.get("test_set") != "zara1" or int(last_payload.get("epoch", 0)) != 150:
            raise RuntimeError("goal checkpoint fold or completion epoch mismatch")
        if not last_payload.get("full_runtime_resume_contract"):
            raise RuntimeError("goal last checkpoint is not fully resumable")
        state = payload.get("model_state_dict", {})
        if not state or any(not torch.isfinite(tensor).all() for tensor in state.values()):
            raise RuntimeError("goal checkpoint tensors are absent or non-finite")
        self.copy_atomic(source, destination)
        digest = sha256(destination)
        if digest != sha256(source):
            raise RuntimeError("goal checkpoint copy changed bytes")
        architecture = yaml.safe_load((self.paper / "configs/architecture.yaml").read_text())
        training = yaml.safe_load((self.paper / "configs/training.yaml").read_text())
        fold = yaml.safe_load((self.paper / "configs/folds/zara1.yaml").read_text())
        config = merge(merge(architecture, training), fold)
        config["paths"]["accepted_fold_matched_gdts_unet"] = str(destination)
        config["paths"]["accepted_fold_matched_gdts_unet_sha256"] = digest
        sys.path.insert(0, str(self.paper))
        from ebjd.config import build_model
        model = build_model(config, "none")
        if any(not torch.isfinite(tensor).all() for tensor in model.encoder.map_unet.state_dict().values()):
            raise RuntimeError("strict-loaded EBJD Goal U-Net is non-finite")
        del model, payload, last_payload
        return digest

    def validate_baseline(self, source: Path, destination: Path, goal_hash: str) -> str:
        last = source.parent / "last_model.pt"
        selected = torch.load(source, map_location="cpu", weights_only=False)
        resume = torch.load(last, map_location="cpu", weights_only=False)
        if int(resume.get("epoch", 0)) != 250 or not resume.get("full_runtime_resume_contract"):
            raise RuntimeError("GDTS baseline did not finish with a complete resume checkpoint")
        initialization = selected.get("goal_pretrain_initialization") or {}
        if initialization.get("sha256") != goal_hash:
            raise RuntimeError("GDTS baseline is not bound to the selected goal checkpoint")
        state = selected.get("model_state_dict", {})
        if not state or any(not torch.isfinite(tensor).all() for tensor in state.values()):
            raise RuntimeError("GDTS baseline tensors are absent or non-finite")
        self.copy_atomic(source, destination)
        digest = sha256(destination)
        if digest != sha256(source):
            raise RuntimeError("GDTS checkpoint copy changed bytes")
        del selected, resume
        return digest

    def parity(self, split: str) -> Path:
        target = self.root / "receipts" / f"{split}_manifest_parity.json"
        command = [
            self.python, "-B", str(self.paper / "scripts/audit_export_parity.py"),
            "--manifest", self.recipe["manifests"][split],
            "--legacy-batches", str(self.independent_cache / "eth5/zara1/data_batches" / f"{'valid' if split == 'validation' else 'test'}_batches"),
            "--grouped-batches", str(self.scene_cache / "eth5/zara1/data_batches_joint_v2" / f"{'valid' if split == 'validation' else 'test'}_batches"),
            "--homography", str(Path(self.recipe["source_data_root"]) / "eth5/zara1/H.txt"),
            "--semantic-map", str(Path(self.recipe["source_data_root"]) / "eth5/zara1/pred_mask.png"),
            "--source-scene", "zara1", "--output", str(target),
        ]
        self.run_stage(f"{split}_parity", command, self.logs / f"{split}_parity.log", gpu=False)
        receipt = json.loads(target.read_text(encoding="utf-8"))
        if not receipt.get("passed"):
            raise RuntimeError(f"{split} parity failed")
        return target

    def matched_eval(
        self, name: str, checkpoint: Path, checkpoint_hash: str,
        manifest: Path, parity: Path, seeds: list[int], split: str,
    ) -> Path:
        output = self.root / name
        result_path = output / "results.json"
        if result_path.is_file():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                existing.get("status") == "complete"
                and existing.get("evaluation_seeds") == seeds
                and existing.get("checkpoint_sha256") == checkpoint_hash
                and existing.get("manifest_sha256") == sha256(manifest)
            ):
                self.persist(stage=name, status="COMPLETE_REUSED",
                             child_pid=None, result_path=str(result_path))
                return result_path
        command = [
            self.python, "-u", "-B", str(self.paper / "scripts/evaluate_matched_gdts.py"),
            "--recipe", str(self.recipe_path), "--runtime-root", str(self.runtime),
            "--scene-cache-root", str(self.scene_cache),
            "--checkpoint", str(checkpoint), "--checkpoint-sha256", checkpoint_hash,
            "--manifest", str(manifest), "--parity-receipt", str(parity),
            "--output-dir", str(output), "--split", split,
            "--seeds", *[str(seed) for seed in seeds],
        ]
        self.run_stage(name, command, self.logs / f"{name}.log", gpu=True)
        return result_path

    def bind_fold(
        self, goal: Path, goal_hash: str, baseline: Path, baseline_hash: str,
        evaluation: Path,
    ) -> str:
        result = json.loads(evaluation.read_text(encoding="utf-8"))
        if result.get("status") != "complete" or result.get("evaluation_seeds") != [2035]:
            raise RuntimeError("selection baseline result is incomplete")
        binding_result = self.assets / "references/zara1_matched_gdts_seed2035.json"
        self.copy_atomic(evaluation, binding_result)
        result_hash = sha256(binding_result)
        record = result["runs"][0]
        manifest = Path(self.recipe["manifests"]["validation"])
        fold_path = self.paper / "configs/folds/zara1.yaml"
        payload = yaml.safe_load(fold_path.read_text(encoding="utf-8"))
        payload["paths"].update({
            "output_root": str(self.assets / "outputs/zara1"),
            "accepted_fold_matched_gdts_unet": str(goal),
            "accepted_fold_matched_gdts_unet_sha256": goal_hash,
            "matched_gdts_validation_checkpoint": str(baseline),
            "matched_gdts_validation_checkpoint_sha256": baseline_hash,
            "matched_gdts_validation_result": str(binding_result),
            "matched_gdts_validation_result_sha256": result_hash,
        })
        payload["evaluation"]["matched_gdts_validation"] = {
            "status": "bound_to_manifest",
            "minADE": float(record["metrics"]["minADE"]),
            "minFDE": float(record["metrics"]["minFDE"]),
            "source": "locally_trained_GDTS_ZARA1_seed3101_seed2035_synchronized_scene_manifest",
            "reference_limitation": "one_training_seed_and_validation_test_are_mirrored_official_package_members",
            "manifest_sha256": sha256(manifest), "checkpoint_sha256": baseline_hash,
            "ebjd_initialization_sha256": goal_hash, "fold": "zara1", "unit": "m",
            "worlds": 20, "steps": 20, "evaluation_seed": 2035,
            "scene_count": int(record["scene_count"]),
            "agent_occurrence_count": int(record["agent_occurrence_count"]),
            "result_sha256": result_hash,
        }
        if git_output(self.paper, "rev-parse", "HEAD") != self.initial_commit:
            raise RuntimeError("paper HEAD changed while preparing assets")
        if git_output(self.paper, "diff", "--name-only").strip():
            raise RuntimeError("tracked paper files changed while preparing assets")
        fold_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        run_checked(["git", "add", "configs/folds/zara1.yaml"], cwd=self.paper)
        run_checked(["git", "commit", "-m", "bind ZARA1 matched GDTS assets"], cwd=self.paper)
        commit = git_output(self.paper, "rev-parse", "HEAD")
        run_checked(["git", "push", "origin", "train/ebjd-zara1-full-v1"], cwd=self.paper)
        remote = git_output(
            self.paper, "ls-remote", "origin", "refs/heads/train/ebjd-zara1-full-v1")
        if not remote.startswith(commit + "\t"):
            raise RuntimeError("remote ZARA1 branch does not match binding commit")
        resolved = self.paper / "configs/resolved/zara1.json"
        run_checked([
            self.python, "-B", str(self.paper / "scripts/resolve_config.py"),
            "--fold", "zara1", "--output", str(resolved),
        ], cwd=self.paper)
        return commit

    def write_ebjd_startup_contract(
        self, resolved: Path, run_id: str, device: str,
    ) -> Path:
        """Create the fail-closed contract consumed before EBJD update one."""
        # Use the exact loader used by ``python -m ebjd.train``.  PyYAML and
        # json.loads assign different scalar types to JSON scientific notation
        # such as 1e-05, which is numerically harmless but changes a typed
        # semantic hash and must not produce a false startup-gate failure.
        sys.path.insert(0, str(self.paper))
        from ebjd.config import load_config, semantic_config_hash
        config = load_config(resolved)
        training = config["training"]
        recipe = self.recipe["ebjd"]
        expected_values = {
            "epochs": 250,
            "rollout_training_enabled": False,
            "paired_marginal_actual_step_projection_enabled": False,
            "hidden_dim": 192,
            "attention_heads": 4,
            "ff_dim": 768,
            "execution_strategy": "prefetch_runtime_v1",
            "training_runtime_optimizations": True,
            "data_loader_workers": 4,
            "data_loader_prefetch_factor": 2,
        }
        actual_values = {
            "epochs": int(training["epochs"]),
            "rollout_training_enabled": bool(
                training.get("rollout_training_enabled", True)),
            "paired_marginal_actual_step_projection_enabled": bool(
                training.get(
                    "paired_marginal_actual_step_projection_enabled", True)),
            "hidden_dim": int(config["denoiser"]["hidden_dim"]),
            "attention_heads": int(config["denoiser"]["heads"]),
            "ff_dim": int(config["denoiser"]["ff_dim"]),
            "execution_strategy": str(training.get("execution_strategy")),
            "training_runtime_optimizations": bool(
                training.get("training_runtime_optimizations", False)),
            "data_loader_workers": int(training.get("data_loader_workers", 0)),
            "data_loader_prefetch_factor": int(
                training.get("data_loader_prefetch_factor", 0)),
        }
        recipe_values = {
            "epochs": int(recipe["epochs"]),
            "rollout_training_enabled": bool(
                recipe["rollout_training_enabled"]),
            "paired_marginal_actual_step_projection_enabled": bool(
                recipe["paired_marginal_actual_step_projection_enabled"]),
            "hidden_dim": int(config["denoiser"]["hidden_dim"]),
            "attention_heads": int(config["denoiser"]["heads"]),
            "ff_dim": int(config["denoiser"]["ff_dim"]),
            "execution_strategy": str(recipe["execution_strategy"]),
            "training_runtime_optimizations": bool(
                recipe["training_runtime_optimizations"]),
            "data_loader_workers": int(recipe["data_loader_workers"]),
            "data_loader_prefetch_factor": int(
                recipe["data_loader_prefetch_factor"]),
        }
        if actual_values != expected_values or recipe_values != expected_values:
            raise RuntimeError(
                "ZARA1 EBJD e250 rollout-OFF contract mismatch: "
                f"resolved={actual_values}, recipe={recipe_values}")
        if int(config["evaluation"]["worlds"]) != 20 or int(
            config["evaluation"]["steps"]) != 20:
            raise RuntimeError("ZARA1 formal inference must remain 20 worlds/20 steps")
        commit = git_output(self.paper, "rev-parse", "HEAD")
        if config["experiment"]["source_commit"] != commit:
            raise RuntimeError("resolved config is not bound to the current paper commit")
        if git_output(self.paper, "diff", "--name-only").strip():
            raise RuntimeError("tracked paper files changed before formal EBJD launch")
        path = self.root / "receipts/ebjd_e250_rollout_off_startup_contract.json"
        expected = {
            "resolved_config_path": str(resolved.resolve()),
            "resolved_config_sha256": sha256(resolved),
            "semantic_config_hash": semantic_config_hash(config),
            "source_commit": commit,
            **expected_values,
            "evaluation_worlds": 20,
            "evaluation_steps": 20,
            "run_id": run_id,
            "ablation": "none",
            "device": device,
        }
        atomic_json(path, {
            "schema": "ebjd-zara1-e250-rollout-off-launch-v1",
            "expected": expected,
            "scheduler": {
                "type": "epoch_warmup_cosine",
                "total_epochs": 250,
                "warmup_epochs": int(training["lr_warmup_epochs"]),
                "cosine_decay_epoch_start": int(training["lr_warmup_epochs"]) + 1,
                "cosine_decay_epoch_end": 250,
                "cosine_denominator_epochs": 250 - int(
                    training["lr_warmup_epochs"]),
                "initial_new_module_lr": float(training["new_module_lr"]),
                "initial_unet_lr": float(training["unet_lr"]),
                "final_lr_fraction": float(training["lr_final_fraction"]),
                "independent_total_steps_or_t_max": None,
            },
            "launch_parameters": {
                "config": str(resolved.resolve()), "ablation": "none",
                "device": device, "run_id": run_id,
                "resume_policy": "same_run_last_pt_epoch_boundary_only",
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        return path

    def freeze_selection(self, run_dir: Path) -> tuple[Path, str]:
        selection_path = run_dir / "SELECTION_STATUS.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("best_eligible") is not None and (run_dir / "best.pt").is_file():
            checkpoint = run_dir / "best.pt"
            role = "best_eligible"
            candidate = selection["best_eligible"]
        else:
            checkpoint = run_dir / "least_violation.pt"
            role = "least_violation"
            candidate = selection.get("least_violation")
        if not checkpoint.is_file() or candidate is None:
            raise RuntimeError("formal EBJD selector produced no checkpoint")
        digest = sha256(checkpoint)
        atomic_json(run_dir / "FORMAL_SELECTION_RECEIPT.json", {
            "schema": "ebjd-formal-selection-receipt-v1", "fold": "zara1",
            "training_seed": 3101, "selected_role": role,
            "performance_contract_passed": role == "best_eligible",
            "candidate": candidate, "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest,
        })
        return checkpoint, digest

    def execute(self) -> None:
        self.ensure_layout()
        goal_run = self.runtime / "output/zara1/runs" / self.recipe["goal_pretrain"]["run_name"]
        baseline_run = self.runtime / "output/zara1/runs" / self.recipe["gdts_baseline"]["run_name"]
        goal_log = self.logs / "goal_pretrain.log"
        baseline_log = self.logs / "gdts_baseline.log"
        if self.dry_run:
            print(json.dumps({"pipeline_root": str(self.root), "initial_commit": self.initial_commit}))
        goal_source = goal_run / "saved_models/goal_pretrain_best_model.pt"
        goal_asset = self.assets / "checkpoints/zara1/goal_unet_best.pt"
        goal_last = goal_run / "saved_models/goal_pretrain_last_model.pt"
        if self.checkpoint_epoch(goal_last) != 150 or not goal_source.is_file():
            self.run_stage(
                "goal_pretrain", self.legacy_command("goal", self.independent_cache),
                goal_log, progress=lambda: legacy_progress(goal_log, self.independent_cache))
        if self.dry_run:
            return
        goal_hash = self.validate_goal(goal_source, goal_asset)
        goal_payload = torch.load(goal_asset, map_location="cpu", weights_only=False)
        self.persist(stage="goal_pretrain", status="COMPLETE",
                     epoch=150, update=150 * legacy_progress(goal_log, self.independent_cache).get("updates_per_epoch", 0),
                     goal_checkpoint=str(goal_asset), goal_checkpoint_sha256=goal_hash,
                     goal_selected_epoch=int(goal_payload["epoch"]))
        del goal_payload

        baseline_source = baseline_run / "saved_models/best_model.pt"
        baseline_asset = self.assets / "checkpoints/zara1/matched_gdts_best.pt"
        baseline_last = baseline_run / "saved_models/last_model.pt"
        if self.checkpoint_epoch(baseline_last) != 250 or not baseline_source.is_file():
            self.run_stage(
                "gdts_baseline", self.legacy_command("baseline", self.independent_cache, goal_asset),
                baseline_log, progress=lambda: legacy_progress(baseline_log, self.independent_cache))
        baseline_hash = self.validate_baseline(baseline_source, baseline_asset, goal_hash)
        selected_epoch = int(torch.load(
            baseline_asset, map_location="cpu", weights_only=False)["epoch"])
        self.persist(stage="gdts_baseline", status="COMPLETE", epoch=250,
                     gdts_checkpoint=str(baseline_asset),
                     gdts_checkpoint_sha256=baseline_hash,
                     gdts_selected_epoch=selected_epoch)

        scene_log = self.logs / "synchronized_scene_cache.log"
        scene_root = self.scene_cache / "eth5/zara1/data_batches_joint_v2"
        scene_complete = all(
            (scene_root / f"finished_{split}_batches.txt").is_file()
            for split in ("train", "valid", "test"))
        if not scene_complete:
            self.run_stage(
                "synchronized_scene_cache",
                self.legacy_command("scene_cache", self.scene_cache), scene_log,
                gpu=False)
        validation_parity = self.parity("validation")
        selection_result = self.matched_eval(
            "matched_gdts_selection_seed2035", baseline_asset, baseline_hash,
            Path(self.recipe["manifests"]["validation"]), validation_parity,
            [2035], "valid")
        bound_fold = yaml.safe_load(
            (self.paper / "configs/folds/zara1.yaml").read_text(encoding="utf-8"))
        already_bound = (
            bound_fold["evaluation"]["matched_gdts_validation"].get("status")
            == "bound_to_manifest"
            and bound_fold["paths"].get("accepted_fold_matched_gdts_unet_sha256") == goal_hash
            and bound_fold["paths"].get("matched_gdts_validation_checkpoint_sha256") == baseline_hash
        )
        if already_bound:
            binding_commit = git_output(self.paper, "rev-parse", "HEAD")
            run_checked([
                self.python, "-B", str(self.paper / "scripts/resolve_config.py"),
                "--fold", "zara1", "--output",
                str(self.paper / "configs/resolved/zara1.json"),
            ], cwd=self.paper)
        else:
            binding_commit = self.bind_fold(
                goal_asset, goal_hash, baseline_asset, baseline_hash, selection_result)
        self.persist(stage="assets_bound", status="COMPLETE",
                     binding_commit=binding_commit,
                     resolved_config=str(self.paper / "configs/resolved/zara1.json"))

        resolved = self.paper / "configs/resolved/zara1.json"
        run_id = "zara1_full_wide192_rollout_off_seed3101_e250_v1"
        run_dir = self.assets / "outputs/zara1/zara1/3101/none" / run_id
        train_log = self.logs / "ebjd_training.log"
        startup_contract = self.write_ebjd_startup_contract(
            resolved, run_id, "cuda:0")
        train_command = [
            self.python, "-u", "-B", "-m", "ebjd.train", "--config", str(resolved),
            "--ablation", "none", "--device", "cuda:0", "--run-id", run_id,
            "--startup-contract", str(startup_contract),
        ]
        if (run_dir / "last.pt").is_file():
            train_command += ["--resume", str(run_dir / "last.pt")]
        self.persist(
            stage="ebjd_startup_gate", status="PASSED", epoch=0, update=0,
            startup_contract=str(startup_contract),
            startup_contract_sha256=sha256(startup_contract),
            ebjd_epochs=250, rollout_training_enabled=False,
            paired_marginal_actual_step_projection_enabled=False,
            ebjd_run_id=run_id)
        if self.checkpoint_epoch(run_dir / "last.pt") != 250:
            self.run_stage(
                "ebjd_training", train_command, train_log,
                progress=lambda: ebjd_progress(train_log))
        selected, selected_hash = self.freeze_selection(run_dir)
        if self.checkpoint_epoch(run_dir / "final.pt") != 250:
            raise RuntimeError("formal ZARA1 final.pt is not the epoch-250 boundary")
        self.persist(stage="ebjd_training", status="COMPLETE", epoch=250,
                     selected_checkpoint=str(selected),
                     selected_checkpoint_sha256=selected_hash)

        ebjd_result = self.root / "final_evaluation/ebjd_seeds2035_2039.json"
        ebjd_command = [
            self.python, "-u", "-B", "-m", "ebjd.evaluate",
            "--config", str(resolved), "--checkpoint", str(selected),
            "--checkpoint-sha256", selected_hash, "--split", "test",
            "--ablation", "none", "--device", "cuda:0", "--output", str(ebjd_result),
        ]
        ebjd_complete = False
        if ebjd_result.is_file():
            existing = json.loads(ebjd_result.read_text(encoding="utf-8"))
            ebjd_complete = (
                existing.get("checkpoint_sha256") == selected_hash
                and len(existing.get("per_seed", [])) == 5)
        if not ebjd_complete:
            self.run_stage("ebjd_five_seed_evaluation", ebjd_command,
                           self.logs / "ebjd_five_seed_evaluation.log", gpu=True)
        test_parity = self.parity("test")
        gdts_result = self.matched_eval(
            "matched_gdts_test_seeds2035_2039", baseline_asset, baseline_hash,
            Path(self.recipe["manifests"]["test"]), test_parity,
            [2035, 2036, 2037, 2038, 2039], "test")
        self.persist(stage="complete", status="COMPLETE", epoch=250,
                     ebjd_five_seed_result=str(ebjd_result),
                     gdts_five_seed_result=str(gdts_result), child_pid=None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", default="configs/asset_preparation/zara1_gdts.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    recipe = Path(args.recipe)
    if not recipe.is_absolute():
        recipe = Path(__file__).resolve().parents[1] / recipe
    pipeline = Pipeline(recipe, args.dry_run)
    pipeline.root.mkdir(parents=True, exist_ok=True)
    lock_path = pipeline.root / "pipeline.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another ZARA1 pipeline supervisor holds the lock") from exc
        try:
            pipeline.execute()
        except Exception as exc:
            pipeline.persist(status="FAILED", child_pid=None,
                             failed_reason=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()

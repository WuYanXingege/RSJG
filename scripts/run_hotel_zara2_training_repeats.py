#!/usr/bin/env python3
"""Serial, resumable HOTEL/ZARA2 training-seed repeat pipeline."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
ASSETS = ROOT / "RSJG_EBJD_assets"
PYTHON = ROOT / ".conda/rsjg/bin/python"
SEQUENCE = [("hotel", 3102), ("zara2", 3102), ("hotel", 3103), ("zara2", 3103)]
SOURCE_RUNS = {
    "hotel": ASSETS / "experiments/hotel_unified_coupling_bridge_v1_20261030_211601",
    "zara2": ASSETS / "experiments/zara2_unified_coupling_bridge_v1_20261029_094715",
}
CONFIG_NAMES = {
    "hotel": "RESOLVED_HOTEL_UNIFIED.json",
    "zara2": "RESOLVED_ZARA2_UNIFIED.json",
}
TRAIN_BANKS = {
    "hotel": SOURCE_RUNS["hotel"] / "proposal_banks/native_train/gdts_proposal_train_manifest_seed3101.npz",
    "zara2": SOURCE_RUNS["zara2"] / "proposal_banks/native_train/gdts_proposal_train_seed3101.npz",
}
VALIDATION_PATTERNS = {
    fold: str(root / "proposal_banks/native_validation/seed_{seed}/gdts_proposal_B_seed{seed}.npz")
    for fold, root in SOURCE_RUNS.items()
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class RepeatPipeline:
    def __init__(self, root: Path, source_commit: str,
                 carried_forward_gpu_seconds: float = 0.0) -> None:
        self.root = root.resolve()
        self.source_commit = source_commit
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "protocol").mkdir(exist_ok=True)
        (self.root / "summary").mkdir(exist_ok=True)
        self.manifest_path = self.root / "RUN_MANIFEST.json"
        self.budget_path = self.root / "GPU_BUDGET.json"
        self.manifest = read_json(self.manifest_path) if self.manifest_path.exists() else {
            "schema": "ebjd-hotel-zara2-training-repeats-v1",
            "source_commit": source_commit,
            "sequence": [{"fold": f, "training_seed": s} for f, s in SEQUENCE],
            "provider_seed": 3101,
            "status": "INITIALIZING",
            "completed": [],
        }
        self.budget = read_json(self.budget_path) if self.budget_path.exists() else {
            "limit_seconds": 48 * 3600,
            "consumed_seconds": carried_forward_gpu_seconds,
            "remaining_seconds": 48 * 3600 - carried_forward_gpu_seconds,
            "stages": ([{"stage": "PRELAUNCH_FAILED_ATTEMPT",
                         "seconds": carried_forward_gpu_seconds,
                         "status": "carried_forward"}]
                       if carried_forward_gpu_seconds else []),
        }

    def update(self, status: str, **fields) -> None:
        self.manifest.update({
            "status": status,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            **fields,
        })
        atomic_json(self.manifest_path, self.manifest)

    def charge(self, stage: str, seconds: float, status: str) -> None:
        self.budget["consumed_seconds"] += seconds
        self.budget["remaining_seconds"] = max(
            0.0, self.budget["limit_seconds"] - self.budget["consumed_seconds"])
        self.budget["stages"].append({
            "stage": stage,
            "seconds": seconds,
            "status": status,
            "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        atomic_json(self.budget_path, self.budget)

    def verify_source(self) -> None:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PAPER, text=True).strip()
        if head != self.source_commit:
            raise RuntimeError(f"source HEAD changed: expected {self.source_commit}, got {head}")
        if self.manifest["source_commit"] != self.source_commit:
            raise RuntimeError("resume source commit mismatch")

    def prepare_protocol(self) -> None:
        assets = {}
        for fold in ("hotel", "zara2"):
            source = SOURCE_RUNS[fold]
            config_path = source / "configs" / CONFIG_NAMES[fold]
            config = read_json(config_path)
            expected = {
                "epochs": 30,
                "rollout_training_enabled": False,
                "paired_marginal_actual_step_projection_enabled": False,
                "lr_final_fraction": 0.1,
                "lr_warmup_epochs": 5,
            }
            for key, value in expected.items():
                if config["training"][key] != value:
                    raise RuntimeError(f"{fold}: effective contract mismatch for {key}")
            if (config["denoiser"]["hidden_dim"], config["denoiser"]["heads"],
                    config["denoiser"]["ff_dim"], config["denoiser"]["coarse_blocks"],
                    config["denoiser"]["refinement_blocks"]) != (192, 4, 768, 3, 3):
                raise RuntimeError(f"{fold}: architecture mismatch")
            bank = TRAIN_BANKS[fold]
            if not bank.is_file():
                raise FileNotFoundError(bank)
            validation = {
                str(seed): sha256(Path(VALIDATION_PATTERNS[fold].format(seed=seed)))
                for seed in range(2035, 2040)
            }
            assets[fold] = {
                "seed3101_effective_config": str(config_path),
                "seed3101_effective_config_sha256": sha256(config_path),
                "train_bank": str(bank),
                "train_bank_sha256": sha256(bank),
                "provider_seed": 3101,
                "validation_bank_sha256": validation,
                "matched_gdts_results": str(source / "reports/PER_SEED_METRICS.csv"),
                "matched_gdts_results_sha256": sha256(source / "reports/PER_SEED_METRICS.csv"),
                "train_manifest": config["paths"]["train_manifest"],
                "train_manifest_sha256": sha256(Path(config["paths"]["train_manifest"])),
                "validation_manifest": config["paths"]["validation_manifest"],
                "validation_manifest_sha256": sha256(Path(config["paths"]["validation_manifest"])),
                "map_initialization": config["paths"]["accepted_fold_matched_gdts_unet"],
                "map_initialization_sha256": config["paths"]["accepted_fold_matched_gdts_unet_sha256"],
                "matched_gdts_checkpoint_sha256": config["paths"]["matched_gdts_validation_checkpoint_sha256"],
            }
        atomic_json(self.root / "protocol/SOURCE_ASSETS.json", assets)
        atomic_json(self.root / "protocol/TRAINING_CONTRACT.json", {
            "architecture": "192/4/768, 3+3 FJAB, FULL/Raw",
            "epochs": 30,
            "pair_weight": 1.0,
            "sinkhorn_temperature": 1.0,
            "sinkhorn_iterations": 20,
            "pair_loss_temperature": 0.1,
            "proposal_eta": 0.0,
            "rollout": False,
            "paired_actual_step_projection": False,
            "worlds": 20,
            "ddim_steps": 20,
            "training_seeds": [3102, 3103],
            "provider_seed": 3101,
            "inference_seeds": list(range(2035, 2040)),
            "source_commit": self.source_commit,
        })

    def effective_config(self, fold: str, training_seed: int) -> Path:
        run_root = self.root / fold / str(training_seed)
        for name in ("configs", "logs", "training", "evaluations", "reports"):
            (run_root / name).mkdir(parents=True, exist_ok=True)
        destination = run_root / "configs/effective_config.json"
        if destination.exists():
            existing = read_json(destination)
            if int(existing["seed"]) != training_seed:
                raise RuntimeError(f"{fold}/{training_seed}: existing config seed mismatch")
            return destination
        source = SOURCE_RUNS[fold] / "configs" / CONFIG_NAMES[fold]
        config = read_json(source)
        config["seed"] = training_seed
        config["experiment"].update({
            "source_commit": self.source_commit,
            "implementation_base": self.source_commit[:7],
        })
        config["paths"]["output_root"] = str(run_root / "training")
        atomic_json(destination, config)
        return destination

    def run_one(self, fold: str, training_seed: int) -> None:
        run_root = self.root / fold / str(training_seed)
        final_path = run_root / "training/FINAL_EVALUATION.json"
        if final_path.exists() and read_json(final_path).get("status") == "COMPLETE":
            self.postprocess(fold, training_seed)
            return
        remaining = self.budget["limit_seconds"] - self.budget["consumed_seconds"]
        reserve = 1800.0 if fold == "hotel" else 3600.0
        if remaining <= reserve:
            self.update("BUDGET_PARTIAL", current_fold=fold, current_training_seed=training_seed,
                        failure="insufficient remaining budget to preserve final evaluation")
            raise RuntimeError("global GPU budget reserve exhausted")
        config = self.effective_config(fold, training_seed)
        command = [
            str(PYTHON), "-u", "scripts/run_zara1_unified_coupling_bridge.py",
            "--config", str(config),
            "--fold", fold,
            "--compact-scenes",
            "--training-seed", str(training_seed),
            "--provider-seed", "3101",
            "--train-bank", str(TRAIN_BANKS[fold]),
            "--validation-bank-pattern", VALIDATION_PATTERNS[fold],
            "--output", str(run_root / "training"),
            "--source-commit", self.source_commit,
            "--max-gpu-seconds", str(remaining),
            "--final-reserve-seconds", str(reserve),
            "--resume",
        ]
        log = run_root / "logs/console.log"
        self.update("RUNNING", current_fold=fold, current_training_seed=training_seed,
                    command=command, log=str(log), effective_config_sha256=sha256(config))
        started = time.perf_counter()
        outcome = "failed"
        try:
            with log.open("a", encoding="utf-8") as handle:
                process = subprocess.Popen(
                    command, cwd=PAPER, stdout=handle, stderr=subprocess.STDOUT,
                    start_new_session=False)
                self.update("RUNNING", current_fold=fold, current_training_seed=training_seed,
                            command=command, log=str(log), child_pid=process.pid,
                            effective_config_sha256=sha256(config))
                return_code = process.wait()
            if return_code:
                raise RuntimeError(f"{fold}/{training_seed} exited with code {return_code}")
            if not final_path.exists() or read_json(final_path).get("status") != "COMPLETE":
                outcome = "budget_partial"
                self.update("BUDGET_PARTIAL", current_fold=fold,
                            current_training_seed=training_seed,
                            failure="run ended without a complete five-seed FINAL_EVALUATION")
                raise RuntimeError(f"{fold}/{training_seed} is incomplete")
            outcome = "complete"
        finally:
            self.charge(f"{fold.upper()}_{training_seed}", time.perf_counter() - started, outcome)
        self.postprocess(fold, training_seed)
        completed = self.manifest.setdefault("completed", [])
        key = {"fold": fold, "training_seed": training_seed}
        if key not in completed:
            completed.append(key)
        self.update("RUN_COMPLETE", current_fold=fold, current_training_seed=training_seed)

    def postprocess(self, fold: str, training_seed: int) -> None:
        run_root = self.root / fold / str(training_seed)
        final = read_json(run_root / "training/FINAL_EVALUATION.json")
        reference_rows = read_csv(SOURCE_RUNS[fold] / "reports/PER_SEED_METRICS.csv")
        reference = {int(row["seed"]): row for row in reference_rows}
        output_rows = []
        for run in final["runs"]:
            seed = int(run["seed"])
            row = {"training_seed": training_seed, "inference_seed": seed}
            for method in ("G", "H"):
                for metric, value in run[method].items():
                    row[f"{method}_{metric}"] = value
            for metric in ("minADE", "minFDE", "JADE", "JFDE"):
                gdts = float(reference[seed][f"GDTS_{metric}"])
                row[f"GDTS_{metric}"] = gdts
                row[f"H_minus_G_{metric}"] = run["H"][metric] - run["G"][metric]
                row[f"H_minus_GDTS_{metric}"] = run["H"][metric] - gdts
            output_rows.append(row)
        metrics_path = run_root / "reports/PER_SEED_METRICS.csv"
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
            writer.writeheader(); writer.writerows(output_rows)
        for seed in range(2035, 2040):
            source = run_root / f"training/evaluations/seed_{seed}.json"
            shutil.copy2(source, run_root / f"evaluations/seed_{seed}.json")
        shutil.copy2(run_root / "training/FINAL_EVALUATION.json",
                     run_root / "evaluations/FINAL_EVALUATION.json")
        selected = Path(final["selected_checkpoint"])
        selected_sha = sha256(selected)
        if selected_sha != final["selected_checkpoint_sha256"]:
            raise RuntimeError(f"{fold}/{training_seed}: selected checkpoint hash mismatch")
        selected_epoch = None
        for row in read_csv(run_root / "training/checkpoint_index.csv"):
            if row["sha256"] == selected_sha:
                selected_epoch = int(row["epoch"])
                break
        atomic_json(run_root / "reports/CHECKPOINT_MANIFEST.json", {
            "selected": str(selected),
            "selected_sha256": selected_sha,
            "selected_epoch": selected_epoch,
            "selection_identity": final["selection_identity"],
            "source_commit": self.source_commit,
            "effective_config_sha256": sha256(run_root / "configs/effective_config.json"),
        })
        contract_pass = all(
            final["summary"]["H"][m]["mean"] <=
            sum(float(reference[s][f"GDTS_{m}"]) for s in reference) / len(reference)
            for m in ("minADE", "minFDE", "JADE", "JFDE"))
        pairing_pass = all(
            final["summary"]["H"][m]["mean"] < final["summary"]["G"][m]["mean"]
            for m in ("JADE", "JFDE"))
        report = [
            f"# {fold.upper()} training seed {training_seed} 正式复验", "",
            "- 状态：COMPLETE",
            f"- provider seed：3101；training seed：{training_seed}",
            f"- 选中：`{final['selection_identity']}`，epoch {selected_epoch}",
            f"- checkpoint SHA256：`{selected_sha}`",
            f"- matched GDTS 四指标合同：`{'PASS' if contract_pass else 'FAIL'}`",
            f"- H 相对 G 的 JADE/JFDE 均值改善：`{'PASS' if pairing_pass else 'FAIL'}`",
            "- 正式评测执行了逐 batch 的 G/H 终点多重集合精确相等检查；minFDE 保持另见逐 seed CSV。",
            "- 证据边界：一个训练 seed，五个推理 seed。", "",
        ]
        (run_root / "reports/REPORT_ZH.md").write_text("\n".join(report), encoding="utf-8")

    def summarize(self) -> None:
        command = [
            str(PYTHON), "scripts/summarize_hotel_zara2_training_repeats.py",
            "--experiment-root", str(self.root),
        ]
        subprocess.run(command, cwd=PAPER, check=True)

    def execute(self) -> None:
        lock_path = self.root / "PIPELINE.lock"
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.verify_source()
            self.prepare_protocol()
            atomic_json(self.budget_path, self.budget)
            self.update("READY")
            try:
                for fold, training_seed in SEQUENCE:
                    self.run_one(fold, training_seed)
                self.update("CPU_SUMMARY_RUNNING")
                self.summarize()
                self.update("COMPLETE")
            except Exception as error:
                if self.manifest.get("status") != "BUDGET_PARTIAL":
                    self.update("FAILED", failure=repr(error))
                raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--carried-forward-gpu-seconds", type=float, default=0.0)
    args = parser.parse_args()
    RepeatPipeline(Path(args.experiment_root), args.source_commit,
                   args.carried_forward_gpu_seconds).execute()


if __name__ == "__main__":
    main()

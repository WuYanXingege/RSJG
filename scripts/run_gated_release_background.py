#!/usr/bin/env python3
"""One detached, identity-bound gated release training/evaluation process.

Run under nohup + setsid. No retry loop: failures retain complete checkpoints
and stop. Reinvocation resumes only a complete last.pt of the same release.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone

PAPER = Path(__file__).resolve().parents[1]
ASSETS = PAPER.parent / "RSJG_EBJD_assets"


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def progress(training):
    path = training / "train.jsonl"
    if not path.exists(): return {"epoch": 0, "update": 0}
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 65536))
        rows = handle.read().splitlines()
    for row in reversed(rows):
        try:
            value = json.loads(row)
            return {key: value[key] for key in
                    ("epoch", "update", "loss", "rollout_update", "projection_active") if key in value}
        except (ValueError, KeyError):
            continue
    return {"epoch": 0, "update": 0}


def resume_command(command, training):
    if (training / "last.pt").is_file():
        return [*command, "--resume"]
    if training.exists() and any(training.iterdir()):
        raise RuntimeError("nonempty output without complete last.pt; refusing silent restart")
    return list(command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(ASSETS / "experiments"):
        raise ValueError("run root must be under persistent experiments")
    manifest = json.loads((root / "RUN_MANIFEST.json").read_text())
    training = Path(manifest["training_output"])
    if training.parent != root:
        raise ValueError("training output must belong to this run")
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    with (root / "SUPERVISOR.lock").open("a+") as run_lock:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_lock.seek(0); run_lock.truncate(); run_lock.write(str(os.getpid())); run_lock.flush()
        state_path = root / "PIPELINE_STATE.json"
        child = None
        started = time.monotonic()
        def state(status, **extra):
            atomic_json(state_path, {
                "status": status, "supervisor_pid": os.getpid(),
                "child_pid": child.pid if child else None,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.monotonic() - started,
                "training_output": str(training),
                "console_log": str(root / "logs/console.log"),
                **progress(training), **extra})
        try:
            head = subprocess.check_output(["git", "-C", str(PAPER), "rev-parse", "HEAD"], text=True).strip()
            if head != manifest["workspace_head"]:
                raise RuntimeError("workspace HEAD differs from frozen launch manifest")
            subprocess.run(["git", "-C", str(PAPER), "diff", "--exit-code", "HEAD", "--",
                "ebjd", "scripts", "configs", "network_release"], check=True, stdout=subprocess.DEVNULL)
            for path, expected in manifest["launch_file_hashes"].items():
                if digest(path) != expected:
                    raise RuntimeError(f"launch identity changed: {path}")
            final_path = training / "FINAL_EVALUATION.json"
            if final_path.exists() and json.loads(final_path.read_text()).get("status") == "COMPLETE":
                state("COMPLETE", reason="already completed; not restarted")
                return
            command = resume_command(manifest["command"], training)
            lock_path = ASSETS / "locks/ebjd_gpu0_training.lock"
            lock_path.parent.mkdir(exist_ok=True)
            with lock_path.open("a+") as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                while True:
                    gpu_processes = subprocess.check_output([
                        "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
                    if not gpu_processes: break
                    state("WAITING_FOR_GPU", existing_compute_pids=gpu_processes)
                    time.sleep(15)
                env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
                       "OPENBLAS_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1",
                       "PYTHONUNBUFFERED": "1"}
                env.pop("PYTHONPATH", None)
                with (root / "logs/console.log").open("a", buffering=1) as log:
                    child = subprocess.Popen(command, cwd=manifest["release_directory"],
                        env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    atomic_json(root / "LAUNCH_RECEIPT.json", {
                        "pid": child.pid, "supervisor_pid": os.getpid(),
                        "command": command, "workspace_head": head,
                        "release_manifest_sha256": manifest["release_manifest_sha256"],
                        "resumed": "--resume" in command,
                        "started_at_utc": datetime.now(timezone.utc).isoformat()})
                    while child.poll() is None:
                        state("TRAINING_OR_EVALUATING")
                        time.sleep(10)
                final = json.loads(final_path.read_text()) if final_path.exists() else {}
                complete = child.returncode == 0 and final.get("status") == "COMPLETE"
                status = "COMPLETE" if complete else "STOPPED_INCOMPLETE"
                state(status, returncode=child.returncode,
                      failure=None if complete else "see console.log; no automatic restart",
                      final_evaluation=str(final_path) if final else None)
                atomic_json(root / "EXECUTION_RECEIPT.json", {
                    "status": status, "returncode": child.returncode,
                    "training_and_evaluation_wall_seconds": time.monotonic()-started,
                    "final_evaluation": final})
                if complete:
                    lines = ["# ETH gated8 训练及冻结评测结果", "",
                             f"选中权重：`{final['selected_checkpoint']}`",
                             f"选择身份：`{final['selection_identity']}`", "",
                             "一个训练seed3101，五个推理seed2035–2039；validation/test镜像。",
                             "", "| 方法 | minADE | minFDE | JADE | JFDE |", "|---|---:|---:|---:|---:|"]
                    for method in ("G", "H"):
                        values = final["summary"][method]
                        lines.append("| " + method + " | " + " | ".join(
                            f"{values[k]['mean']:.6f} ± {values[k]['sample_standard_deviation']:.6f}"
                            for k in ("minADE", "minFDE", "JADE", "JFDE")) + " |")
                    (root / "reports/FINAL_REPORT_ZH.md").write_text("\n".join(lines) + "\n")
        except BaseException as exc:
            if child is not None and child.poll() is None:
                child.terminate(); child.wait()
            state("FAILED", failure=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()

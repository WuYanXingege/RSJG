"""One durable controller for the two-stage, five-fold BCEP–EBJD formal run."""
import argparse
from contextlib import contextmanager
import fcntl
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

import torch
import yaml
from bcep.contracts import canonical_hash, digest, require
from bcep_formal.control import pause_requested
from bcep_formal.data import FormalData
from bcep_joint.common import read, write
from .bank import ExportPaused
from .formal_manifest import (BASE_CONFIG_SHA, FOLDS, MATRIX, METRIC, PLAN_SHA, SUBSTEPS, budgets,
                              contract, freeze_all, immutable, macro, read_freeze, stage_proof)
from .formal_reports import PUBLIC_BRANCH, final_tables, fold_tables, publish
from .frozen_test import case_folder
from .workflow import CONFIG, ROOT, load_contract

VERSION = "2026-11-18 两大阶段整理版"
EXPERIMENT = "bcep_ebjd_v2_formal_top5_20261118"


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def gpu_info():
    def smi(query):
        return subprocess.check_output(["nvidia-smi", query, "--format=csv,noheader,nounits"],
                                       text=True, timeout=15).strip()
    require(not smi("--query-compute-apps=pid,process_name"), "GPU occupied; no work was launched")
    return smi("--query-gpu=name,uuid,driver_version,memory.total,memory.free")


@contextmanager
def locked(root):
    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield lock


def prepare(plan, checkout):
    plan, checkout = plan.resolve(), checkout.resolve()
    require(digest(plan) == PLAN_SHA and digest(CONFIG) == BASE_CONFIG_SHA, "unapproved base inputs")
    require(not git("status", "--porcelain"), "freeze and commit code first")
    require(git("branch", "--show-current", cwd=checkout) == PUBLIC_BRANCH and checkout != ROOT,
            "independent reporter checkout required")
    cfg = yaml.safe_load(CONFIG.read_text())
    require(cfg["allow_real_training"] is False and cfg["joint"]["allow_real_training"] is False,
            "base training authorization changed")
    root = Path(read(plan)["assets"]).parent / "bcep_ebjd_v2_integrated"
    require(root.resolve() == root and not root.exists(), "existing root: inspect the original run")
    require(torch.__version__ == "2.8.0+cu128" and torch.version.cuda == "12.8", "runtime mismatch")
    gpu = gpu_info()
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "CUDA environment must be set")
    cfg["allow_real_training"] = True
    # Assemble everything privately before exposing a prepared run.
    staging = root.with_name("." + root.name + ".prepare")
    staging.mkdir(exist_ok=True)
    require(not any(p.name not in {"AUTHORIZED.yaml", "AUTHORIZED.tmp", "FORMAL_RUN.json",
                                   "FORMAL_RUN.json.tmp", "STATUS.json", "STATUS.json.tmp"}
                    for p in staging.iterdir()), "unexpected file in preparation folder")
    auth = staging / "AUTHORIZED.yaml"
    tmp = auth.with_suffix(".tmp")
    tmp.write_text(CONFIG.read_text().replace("\nallow_real_training: false\n", "\nallow_real_training: true\n", 1))
    require(yaml.safe_load(tmp.read_text()) == cfg, "authorization diff")
    tmp.replace(auth)
    entries = {}
    projected = 0
    # FP32 weights + two Adam moments per complete epoch. Add space for serialization
    # overhead, candidate files/receipts, all test matrices and temporary atomic writes.
    for fold in FOLDS:
        _, split, folder, _, identity = load_contract(auth, plan, fold)
        require(folder == root / fold and not folder.exists(), "occupied training folder")
        FormalData(split, "training", verify=False)  # Validate disjoint member sets without opening future data.
        caps = budgets(cfg, split)
        entries[fold] = dict(identity=identity, caps=caps, windows={p: len(split[p + "_indices"])
                             for p in ("train", "selection_dev", "heldout_confirmation")})
        counts = {p: sum(len(split["members"][i]["agent_ids"]) for i in split[p+"_indices"])
                  for p in ("train", "selection_dev", "heldout_confirmation")}
        projected += 12*(626144*100 + 50050*20 + 7596022*100)
        projected += (3*counts["train"] + 6*counts["selection_dev"])*20*2*4
        projected += counts["heldout_confirmation"]*5*(6*12*20*2*4 + 6*20*2*4 + 5*20*8)
    require(sum(sum(v["caps"].values()) for v in entries.values()) == 533000, "five-fold budget changed")
    fs = os.statvfs(staging)
    free = shutil.disk_usage(staging).free
    estimate = math.ceil(projected*1.20) + 3*1024**3
    require(free > estimate, f"disk capacity: {free} available, estimated {estimate} necessary")
    manifest = dict(experiment=EXPERIMENT, version=VERSION, commit=git("rev-parse", "HEAD"), workspace=str(ROOT),
                    input_plan=str(plan), input_plan_sha256=PLAN_SHA, base_config_sha256=BASE_CONFIG_SHA,
                    config_sha256=digest(auth), config_identity=canonical_hash(cfg), output_root=str(root),
                    report_checkout=str(checkout), folds=entries, order=list(FOLDS), matrix=MATRIX,
                    max_total_updates=533000, training_seeds=[3102, 3102, 3201], gpu=gpu,
                    torch=torch.__version__, cuda=torch.version.cuda, python=sys.executable,
                    runtime=dict(precision="FP32", deterministic=True, sdp="math", TF32=False,
                                 cublas=":4096:8", omp=os.environ.get("OMP_NUM_THREADS")),
                    retention="all complete epochs plus distinct dev Top5, rank1/best and last",
                    disk_precheck=dict(free_bytes=free, estimated_bytes=estimate, free_inodes=fs.f_favail),
                    created_unix=time.time())
    write(staging / "FORMAL_RUN.json", manifest)
    write(staging / "STATUS.json", dict(status="PREPARED", fold=FOLDS[0], macro_stage=macro("density"),
                                     substep="density", optimizer_updates=0))
    staging.rename(root)
    print(root, flush=True)


class Controller:
    def __init__(self, root, lock, resume):
        self.root = root
        self.lock = lock
        self.run = read(root / "FORMAL_RUN.json")
        self.child = None
        self.stop = False
        self.fold = FOLDS[0]
        self.stage = "density"
        self.state = read(root / "STATUS.json")
        if not resume:
            require(self.state["status"] == "PREPARED", "existing run requires explicit resume")
        else:
            require(self.state["status"] in ("FAILED", "PAUSED_RESUMABLE", "RUNNING", "PREPARED"),
                    "run is not resumable")
        require(self.run["commit"] == git("rev-parse", "HEAD") and not git("status", "--porcelain"),
                "code identity changed")
        require(self.run["python"] == sys.executable and torch.__version__ == self.run["torch"]
                and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "runtime changed")
        for fold in FOLDS:
            contract(root, fold)
        gpu_info()
        self.started = time.time()

    def request_stop(self, *_):
        self.stop = True
        from bcep_formal.control import request_pause
        request_pause()
        if self.child is not None and self.child.poll() is None:
            self.child.send_signal(signal.SIGTERM)

    def status(self, name, **other):
        self.state = dict(status=name, experiment=EXPERIMENT, commit=self.run["commit"], fold=self.fold,
                          macro_stage=macro(self.stage), substep=self.stage, controller_pid=os.getpid(),
                          child_pid=self.child.pid if self.child and self.child.poll() is None else None,
                          started_unix=self.started, updated_unix=time.time(), **other)
        write(self.root / "STATUS.json", self.state)

    def check(self, rc=0):
        if self.stop or pause_requested() or rc == 75:
            raise ExportPaused(self.root / self.fold / self.stage)
        if rc:
            raise RuntimeError(f"{self.fold}/{self.stage} worker returned {rc}")

    def invoke(self, args):
        self.check()
        path = self.root / "logs" / self.fold / (self.stage + ".log")
        path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, BCEP_FORMAL_LOCK_FD=str(self.lock.fileno()), PYTHONUNBUFFERED="1")
        command = [sys.executable, "-u", "-m", *args]
        with path.open("a") as log:
            self.child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                          stderr=subprocess.STDOUT, pass_fds=(self.lock.fileno(),))
            if self.stop:
                self.child.send_signal(signal.SIGTERM)
            self.status("RUNNING", log=str(path), command=command)
            rc = self.child.wait()
        self.child = None
        self.check(rc)

    def training_fold(self):
        ctx = contract(self.root, self.fold)
        folder = ctx[2]
        prior = {}
        for self.stage in SUBSTEPS:
            self.check()
            path = folder / "receipts" / (self.stage + ".json")
            if not path.exists():
                args = ["bcep_v2.workflow", self.stage, "--config", str(self.root / "AUTHORIZED.yaml"),
                        "--plan", self.run["input_plan"], "--fold", self.fold, "--execute", "--device", "cuda"]
                if self.stage in METRIC:
                    args += ["--until-update", str(self.run["folds"][self.fold]["caps"][self.stage])]
                    if (folder / self.stage / "last.pt").exists():
                        args.append("--resume")
                self.invoke(args)
            proof = stage_proof(ctx, self.stage, prior)
            self.check()
            immutable(path, proof)
            prior[self.stage] = proof
            if self.stage in ("banks", "joint"):
                stages = SUBSTEPS[:4] if self.stage == "banks" else ("joint",)
                immutable(folder / "receipts" / ("STAGE_1.json" if self.stage == "banks" else "STAGE_2.json"),
                          dict(status="COMPLETE", macro_stage=macro(self.stage),
                               evidence={s: canonical_hash(prior[s]) for s in stages}))
        fold_tables(self.root, self.fold, prior)
        if not (self.root / "publishing" / (self.fold + ".json")).exists() or (
                read(self.root / "publishing" / (self.fold + ".json"))["status"] != "PUBLISHED"):
            publish(self.root, Path(self.run["report_checkout"]), self.fold)
        return prior

    def test_case(self, frozen, kind, rank=1):
        self.check()
        self.stage = f"test_{kind}_rank{rank}"
        folder = self.root / self.fold / "evaluation/head_matrix" if kind == "heads" else case_folder(
            self.root, frozen["folds"][self.fold]["JOINT_TOP5"][rank-1])
        receipt = folder / "COMPLETE.json"
        if not receipt.exists():
            self.invoke(["bcep_v2.frozen_test", kind, "--fold", self.fold, "--rank", str(rank),
                         "--root", str(self.root)])
        done = read(receipt)
        seal = done["seal"] if kind == "heads" else done["identity"]["frozen_run"]["seal"]
        header = read(folder / "IDENTITY.json")
        if kind == "heads":
            require(done["identity"] == canonical_hash(header)
                    and header["heads"] == frozen["folds"][self.fold]["PROVIDER_TOP5"], "wrong endpoint matrix")
        else:
            require(done["identity"] == header and header["checkpoint"] == frozen["folds"][self.fold]["JOINT_TOP5"][rank-1]
                    and header["mode"] == ("GH" if kind == "official" else "H"), "wrong trajectory matrix")
        require(seal == frozen["content_sha256"] and done["status"] == "COMPLETE" and
                done["metrics_sha256"] == digest(folder / "PER_SEED_METRICS.csv"), "test receipt mismatch")
        self.check()

    def execute(self):
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        try:
            self.status("RUNNING")
            proofs = {}
            for self.fold in FOLDS:
                proofs[self.fold] = self.training_fold()
            self.check()
            freeze_all(self.root, proofs)
            frozen = read_freeze(self.root)
            for self.fold in FOLDS:
                self.test_case(frozen, "official")
            for self.fold in FOLDS:
                for rank in range(2, 6):
                    self.test_case(frozen, "joint", rank)
                self.test_case(frozen, "heads")
            final_tables(self.root, frozen)
            publish(self.root, Path(self.run["report_checkout"]), "final")
            self.check()
            self.status("COMPLETE", freeze=frozen["content_sha256"])
        except ExportPaused:
            self.status("PAUSED_RESUMABLE", reason="stop request or worker exit 75")
            return 75
        except BaseException as exc:
            self.status("FAILED", reason=repr(exc))
            traceback.print_exc()
            return 1
        return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "start", "resume", "run", "status", "publish"))
    p.add_argument("--plan", type=Path)
    p.add_argument("--report-checkout", type=Path)
    p.add_argument("--root", type=Path)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    if a.command == "prepare":
        require(a.plan and a.report_checkout, "plan and report checkout are required")
        prepare(a.plan, a.report_checkout)
        return 0
    require(a.root is not None, "--root is required")
    root = a.root.resolve()
    if a.command == "status":
        import json
        state = read(root / "STATUS.json")
        progress = root / state["fold"] / state["substep"] / "PROGRESS.json"
        print(json.dumps(dict(state, progress=read(progress) if progress.exists() else None), indent=2))
        return 0
    if a.command == "publish":
        publish(root, Path(read(root / "FORMAL_RUN.json")["report_checkout"]), "manual")
        return 0 if read(root / "publishing/manual.json")["status"] == "PUBLISHED" else 1
    if a.command in ("start", "resume"):
        # The inherited locked file description closes only after the controller AND its child exit.
        with locked(root) as lock, (root / "controller.log").open("a") as log:
            cmd = [sys.executable, "-u", "-m", "bcep_v2.formal_run", "run", "--root", str(root)]
            if a.command == "resume":
                cmd.append("--resume")
            env = dict(os.environ, BCEP_CONTROLLER_FD=str(lock.fileno()))
            child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, cwd=ROOT, env=env, pass_fds=(lock.fileno(),))
            print(f"controller_pid={child.pid}", flush=True)
        return 0
    fd = os.environ.get("BCEP_CONTROLLER_FD")
    if fd is not None:
        from .formal_manifest import validate_worker_lease
        validate_worker_lease(root, fd)
        with os.fdopen(int(fd), "a") as lock:
            return Controller(root, lock, a.resume).execute()
    with locked(root) as lock:
        return Controller(root, lock, a.resume).execute()


if __name__ == "__main__":
    raise SystemExit(main())

"""Foreground TEST_ONLY engineering acceptance. Never loads real training examples or weights."""

import argparse
import fcntl
import gc
from pathlib import Path
import subprocess
import tempfile
import time
import traceback

import torch
import yaml
from torch.optim.optimizer import register_optimizer_step_pre_hook, register_optimizer_step_post_hook
from bcep.contracts import digest
from bcep_joint.checks import equal
from bcep_joint.common import read, write, seed_all
from ebjd.data import SyntheticSceneDataset, collate_scenes
from .joint import build_joint
from .provider import GEOMETRY, Provider
from .runtime import configure_cuda
from .stages import density_callbacks, head_callbacks, joint_callbacks
from .state import run_stage
from .workflow import CONFIG, recipe


class SyntheticWindows:
    def __init__(self, counts, dev=()):
        self.items = []
        self.validation_reads = 0
        for i, n in enumerate(counts):
            item = SyntheticSceneDataset(1, n, pixels=256, seed=701 + i)[0]
            item["scene_id"] = f"TEST_ONLY-{i}"
            item["semantic_maps"][:, 1, :, ::2] = 1
            item["semantic_maps"][:, 2, :, 1::2] = 1
            self.items.append(item)
        self.split = dict(
            members=[dict(agent_ids=x["agent_ids"]) for x in self.items], selection_dev_indices=list(dev)
        )

    def item(self, i, purpose):
        if purpose == "selection_dev":
            self.validation_reads += 1
        return self.items[i]

    def batch(self, ids, purpose):
        return collate_scenes([self.item(i, purpose) for i in ids])


class SyntheticGoals:
    """Analytic observed-only fixture; these tensors have no formal bank identity."""

    def __init__(self, seed):
        self.seed = seed

    def batch(self, ids, batch):
        rows = []
        for i, o in zip(ids, batch.observed):
            gen = torch.Generator().manual_seed(self.seed * 1000 + i)
            rows.append(o[None, :, -1] + torch.randn(20, len(o), 2, generator=gen))
        return torch.stack(rows)


def setup(stage, data, cfg, device):
    weights = [SyntheticGoals(s) for s in range(2035, 2040)]
    if stage == "joint":
        c = dict(cfg["joint"], scales=dict(goal=1.0, bridge=0.25))
        m = build_joint(c).to(device)
        with torch.no_grad():
            m.coupling.compatibility[-1].weight.normal_(std=0.015)
        callbacks = joint_callbacks(m, data, device, c, weights[0], weights[0])
    else:
        p = Provider().to(device)
        if stage == "density":
            m = p.base
            callbacks = density_callbacks(p, data, device)
        else:
            m = p.head
            with torch.no_grad():
                m.readout[-1].weight.normal_(std=0.025)
            callbacks = head_callbacks(p, data, device, dict(layout=GEOMETRY), weights[:2], weights)
    return m, callbacks


def check_split_max(plan_path, cfg):
    plan = read(plan_path)
    if digest(plan["split_lock"]) != cfg["split_lock_sha256"]:
        raise ValueError("split lock mismatch")
    maxima = {}
    for name, row in plan["folds"].items():
        path = Path(row["split"]["path"])
        if digest(path) != cfg["folds"][name]["split_sha256"]:
            raise ValueError("split metadata mismatch")
        maxima[name] = max(len(m["agent_ids"]) for m in read(path)["members"])
    return maxima


def checked_state(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def compare_files(a, b):
    la, lb = checked_state(a), checked_state(b)
    assert set(la) == set(lb)
    for k in la:
        try:
            equal(la[k], lb[k])
        except AssertionError as exc:
            raise AssertionError("resume differs in " + k) from exc
    return dict(
        epoch=la["epoch"], update=la["update"], cursor=la["cursor"], completed_epoch=la["completed_epoch"]
    )


def recovery(stage, cfg, temp):
    r = recipe(cfg, stage)
    logical = r["logical_scenes"]
    n = 2 * logical + logical // 2
    data = SyntheticWindows([1 + i % 3 for i in range(n + 1)], dev=[n])
    check_calls = dict(head_calls=0, ddim_denoiser_calls=0)

    def count_eval(key):
        def count(module, args, out):
            if not module.training:
                check_calls[key] += 1

        return count

    def arm(folder, cap, resume=False):
        seed_all(1001)
        m, callbacks = setup(stage, data, cfg, "cuda")
        if stage == "joint":
            m.bridge.denoiser.register_forward_hook(count_eval("ddim_denoiser_calls"))
        if stage == "head":
            m.register_forward_hook(count_eval("head_calls"))
        state = run_stage(
            folder,
            dict(stage=stage, source="TEST_ONLY"),
            r,
            m,
            list(range(n)),
            *callbacks,
            cap,
            resume=resume,
        )
        assert state["update"] == cap
        return state["cursor"]

    a, b = temp / "continuous", temp / "resumed"
    arm(a, 4)
    partial_cursor = arm(b, 2)
    assert partial_cursor == 2 * logical
    mid = checked_state(b / "last.pt")
    assert mid["epoch_steps"] == 2 and mid["epoch_counts"] and mid["epoch_raw_sums"]
    del mid
    gc.collect()
    arm(b, 4, True)
    final = compare_files(a / "last.pt", b / "last.pt")
    assert final == dict(epoch=2, update=4, cursor=logical, completed_epoch=1)
    compare_files(a / "best.pt", b / "best.pt")

    def selector(folder):
        rows = read(folder / "TOP5.json")
        for row in rows:
            assert digest(folder / row["file"]) == row["sha256"]
        return [{k: v for k, v in row.items() if k != "sha256"} for row in rows]

    equal(selector(a), selector(b))
    assert data.validation_reads == 2
    assert check_calls["head_calls"] == (10 if stage == "head" else 0)
    assert check_calls["ddim_denoiser_calls"] == (40 if stage == "joint" else 0)
    return dict(
        final=final,
        logical_scenes=logical,
        tail_scenes=logical // 2,
        partial_cursor=2 * logical,
        validation_windows_total=data.validation_reads,
        **check_calls,
        exact=True,
        state_fields="all; both selected state files and logical TOP5 verified",
    )


def shape_check(stage, b, n, cfg, track, budget):
    data = SyntheticWindows([n] * b)
    seed_all(2035)
    m, (backward, _) = setup(stage, data, cfg, "cuda")
    order = torch.Generator().manual_seed(7)
    views = torch.Generator().manual_seed(11)
    measurements = []
    for warm in (True, False):
        if budget["smoke_calls"] >= 16:
            raise RuntimeError("smoke ceiling reached")
        budget["smoke_calls"] += 1
        track()
        m.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        metrics = backward(list(range(b)), order, views)
        torch.cuda.synchronize()
        seconds = time.monotonic() - start
        alloc, res = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        grads = [p.grad for p in m.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        measurements.append(
            dict(
                warmup=warm,
                seconds=seconds,
                peak_allocated_bytes=alloc,
                peak_reserved_bytes=res,
                loss=metrics["loss"],
                edges=metrics.get("effective_edges"),
            )
        )
    return dict(B=b, N=n, measurements=measurements)


def main():
    if not __debug__:
        raise RuntimeError("engineering checks require enabled Python assertions")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--checks",
        nargs="+",
        choices=["density", "head", "joint", "shapes"],
        default=["density", "head", "joint", "shapes"],
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text())
    assert cfg["allow_real_training"] is False and cfg["joint"]["allow_real_training"] is False
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    with (out / "TEST_ONLY.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = out / "TEST_ONLY_GPU.json"
        ledger = (
            read(path)
            if path.exists()
            else dict(
                synthetic_started=0,
                synthetic_completed=0,
                real_updates=0,
                smoke_calls=0,
                gpu_seconds=0.0,
                checks={},
                attempts=[],
            )
        )
        if ledger.get("active"):
            raise RuntimeError("previous acceptance did not close; inspect its receipt first")
        maxima = check_split_max(args.plan, cfg)
        busy = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
        ).strip()
        if busy:
            raise RuntimeError("BLOCKED: GPU has a compute process")
        start = time.monotonic()
        prior = ledger["gpu_seconds"]
        attempt = dict(checks=args.checks, completed=[], status="RUNNING")
        ledger["attempts"].append(attempt)
        ledger.update(active=True, max_valid_agents=maxima)

        def track():
            ledger["gpu_seconds"] = prior + time.monotonic() - start
            write(path, ledger)
            if ledger["gpu_seconds"] >= 1200:
                raise TimeoutError("cumulative GPU budget exhausted")

        def before(*_):
            if ledger["synthetic_started"] >= 32:
                raise RuntimeError("GPU optimizer attempt ceiling reached")
            ledger["synthetic_started"] += 1
            track()

        def after(*_):
            ledger["synthetic_completed"] += 1
            track()

        pre = register_optimizer_step_pre_hook(before)
        post = register_optimizer_step_post_hook(after)
        try:
            track()
            configure_cuda()
            ledger["device"] = torch.cuda.get_device_name(0)
            ledger["torch"] = str(torch.__version__)
            ledger["cuda"] = torch.version.cuda
            for stage in args.checks:
                jobs = (
                    [(stage, None, None)]
                    if stage != "shapes"
                    else [
                        ("density", 1, max(maxima.values())),
                        ("head", 1, max(maxima.values())),
                        ("joint", 1, 1),
                        ("joint", 4, 8),
                        ("joint", 1, max(maxima.values())),
                    ]
                )
                for name, b, n in jobs:
                    key = name if b is None else f"{name}_B{b}_N{n}"
                    if ledger["checks"].get(key, {}).get("status") == "PASS":
                        continue
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    t = time.monotonic()
                    track()
                    try:
                        if b is None:
                            with tempfile.TemporaryDirectory(dir=out, prefix="TEST_ONLY_") as tmp:
                                result = recovery(name, cfg, Path(tmp))
                        else:
                            result = shape_check(name, b, n, cfg, track, ledger)
                        torch.cuda.synchronize()
                        ledger["checks"][key] = dict(
                            status="PASS",
                            seconds=time.monotonic() - t,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                            **result,
                        )
                        attempt["completed"].append(key)
                    except Exception as exc:
                        ledger["checks"][key] = dict(
                            status="FAIL",
                            reason=str(exc),
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                        )
                        attempt.setdefault("failures", {})[key] = traceback.format_exc()
                        print(key, "FAIL", exc, flush=True)
                        if isinstance(exc, (TimeoutError, torch.OutOfMemoryError)):
                            raise
                    track()
            attempt["status"] = "FAIL" if attempt.get("failures") else "PASS"
        except BaseException as exc:
            attempt.update(status="INCOMPLETE", error=str(exc))
            raise
        finally:
            ledger.update(active=False, gpu_seconds=prior + time.monotonic() - start)
            pre.remove()
            post.remove()
            write(path, ledger)
    return int(attempt["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())

"""Identity, stage evidence and the immutable dev-only five-fold test barrier."""
import csv
import fcntl
import json
import math
import os
import time
from pathlib import Path

import torch

from bcep.contracts import canonical_hash, digest, require
from bcep_joint.common import local_seed, read, write
from .bank import Bank, bank_identity
from .workflow import METRIC, SEEDS, load_contract, recipe

FOLDS = ("zara2", "zara1", "hotel", "univ", "eth")
SUBSTEPS = ("density", "base-cache", "head", "banks", "joint")
CAPS = dict(zip(FOLDS, ((67100, 13420, 16800), (72300, 14460, 18100),
                        (78000, 15600, 19500), (68000, 13600, 17000),
                        (82100, 16420, 20600)), strict=True))
BASE_CONFIG_SHA = "0d4034a54443a40f4b046a8e88a423c64f2b7d3ad4e7f14f2b7de0d0be535985"
PLAN_SHA = "eb341bdaf47e00b09cd79b17ff42207c707a7c443bd148c74363508453a4e178"
PURPOSE = "heldout_confirmation"
OFFICIAL = "OFFICIAL_DEV_SELECTED_RANK1"
DIAGNOSTIC = "EXPLORATORY_DEV_TOP5"
MATRIX = dict(folds=list(FOLDS), seeds=list(SEEDS), K=20, steps=20, M=256,
              joint_H_ranks=[1, 2, 3, 4, 5], joint_G_ranks=[1], head_ranks=[1, 2, 3, 4, 5],
              joint_H_groups=125, joint_G_groups=25, provider_endpoint_groups=125,
              official=OFFICIAL, dev_order=["dev_value", "epoch"],
              posthoc_joint=["mean_H_JADE", "dev_rank", "epoch"],
              posthoc_head=["mean_agent_micro_minFDE", "dev_rank", "epoch"])


def macro(stage):
    if stage in SUBSTEPS[:4]:
        return "Stage 1 / BCEP Provider learning"
    if stage == "joint":
        return "Stage 2 / EBJD joint trajectory learning"
    return "Frozen test"


def immutable(path, value):
    if path.exists():
        require(read(path) == value, f"immutable manifest differs: {path.name}")
    else:
        write(path, value)


def write_csv(path, rows):
    require(bool(rows), "empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        out = csv.DictWriter(stream, fieldnames=fields)
        out.writeheader()
        for row in rows:
            out.writerow({k: json.dumps(v, sort_keys=True, ensure_ascii=False) if isinstance(v, (list, dict))
                          else v for k, v in row.items()})
    tmp.replace(path)


def validate_worker_lease(root, inherited_fd):
    require(inherited_fd is not None, "formal GPU child must inherit the controller lock")
    fd = int(inherited_fd)
    require(os.path.samestat(os.fstat(fd), (root / "controller.lock").stat()), "unrelated lock")
    # A separate open-file description cannot acquire this while the controller holds it.
    with (root / "controller.lock").open("r") as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
    raise RuntimeError("controller lease is not exclusive")


def contract(root, fold):
    run = read(root / "FORMAL_RUN.json")
    require(digest(root / "AUTHORIZED.yaml") == run["config_sha256"]
            and digest(run["input_plan"]) == PLAN_SHA, "formal inputs changed")
    cfg, split, folder, geo, ident = load_contract(root / "AUTHORIZED.yaml", run["input_plan"], fold)
    require(folder == root / fold and ident == run["folds"][fold]["identity"], "wrong code or run")
    return cfg, split, folder, geo, ident


def budgets(cfg, split):
    expected = {}
    count = len(split["train_indices"])
    require(count == cfg["folds"][split["fold"]]["train_windows"], "train members differ")
    for stage in METRIC:
        r = recipe(cfg, stage)
        updates = math.ceil(count / r["logical_scenes"])
        require(updates == cfg["folds"][split["fold"]][stage + "_updates_per_epoch"], "updates/epoch")
        expected[stage] = updates * r["max_epochs"]
    require(tuple(expected.values()) == CAPS[split["fold"]], "uncertified budget")
    return expected


def dev_ranks(history):
    require(len({r["epoch"] for r in history}) == len(history)
            and all(math.isfinite(r["validation"]) for r in history), "invalid dev history")
    return sorted(history, key=lambda r: (r["validation"], r["epoch"]))[:5]


def bank_set(ctx, refined):
    cfg, split, folder, geo, ident = ctx
    base = read(folder / "density/FROZEN.json")
    head = read(folder / "head/FROZEN.json") if refined else None
    items = ([("train", 2035, None), ("selection_dev", 2035, None)] if refined else
             [("train", local_seed("bcep-v2-train-view", cfg["provider_seed"], v), v) for v in range(2)]
             + [("selection_dev", s, None) for s in SEEDS])
    result = {}
    for purpose, seed, view in items:
        path = folder / ("refined_banks" if refined else "base_cache") / f"{purpose}_{seed}"
        identity = bank_identity(split, purpose, seed, geo, ident["sources"], base, head, view=view)
        bank = Bank(path, identity, split, interruptible=True)
        result[path.name] = bank.hash
    return result


def stage_identity(ctx, stage, previous):
    cfg, _, folder, _, ident = ctx
    result = dict(ident, stage=stage)
    if stage in ("head", "joint"):
        result["base"] = previous["density"]["selected"]
    if stage == "head":
        banks = previous["base-cache"]["banks"]
        names = [f'train_{local_seed("bcep-v2-train-view", cfg["provider_seed"], v)}' for v in range(2)]
        result["caches"] = [banks[k] for k in names + [f"selection_dev_{s}" for s in SEEDS]]
    if stage == "joint":
        result.update(head=previous["head"]["selected"], train_bank=previous["banks"]["banks"]["train_2035"],
                      dev_bank=previous["banks"]["banks"]["selection_dev_2035"])
    return result


def load_rank(folder, row, identity, model=None):
    """Read only. Never calls the training selector or writes to an upstream folder."""
    path = folder / row["file"]
    require(row["file"] == f"epoch_{row['epoch']:03d}.pt" and digest(path) == row["checkpoint_sha256"],
            "rank file/hash mismatch")
    state = torch.load(path, map_location="cpu", weights_only=False)
    require(state["identity"] == identity and canonical_hash(identity) == row["identity"]
            and state["update"] == row["update"] > 0 and state["completed_epoch"] == row["epoch"]
            and state["cursor"] == 0 and state["order"] is None and state["history"][-1]["validation"] == row["dev_value"]
            and state["status"] in ("EPOCH_COMPLETE", "COMPLETE")
            and {"optimizer", "model", "scheduler", "rng", "epoch_sums", "stale"} <= state.keys(),
            "checkpoint is not the completed dev-selected state")
    if model is not None:
        model.load_state_dict(state["model"], strict=True)
        model.eval().requires_grad_(False)


def stage_proof(ctx, stage, previous):
    cfg, split, folder, geo, ident = ctx
    if stage not in METRIC:
        return dict(status="COMPLETE", stage=stage, banks=bank_set(ctx, stage == "banks"))
    dest = folder / stage
    identity = stage_identity(ctx, stage, previous)
    last = torch.load(dest / "last.pt", map_location="cpu", weights_only=False)
    history = read(dest / "EPOCH_METRICS.json")
    require(last["status"] == "COMPLETE" and last["identity"] == identity and last["history"] == history
            and last["recipe"] == recipe(cfg, stage) and last["stop_reason"] in ("early_stop", "max_epochs")
            and last["completed_epoch"] == len(history) >= last["recipe"]["minimum"]
            and last["update"] <= budgets(cfg, split)[stage], "stage has no validated terminal state")
    ordered = dev_ranks(history)
    index = read(dest / "TOP5.json")
    require(len(ordered) == len(index) == 5, "five complete epochs required")
    ranks = []
    for rank, (r, indexed) in enumerate(zip(ordered, index, strict=True), 1):
        sha = indexed["sha256"]
        require(indexed == dict(r, sha256=sha), "Top5 does not match dev history")
        upstream = {k: v for k, v in identity.items() if k in ("base", "head", "caches", "train_bank", "dev_bank")}
        row = dict(fold=ident["fold"], stage=stage, macro_stage=macro(stage), dev_rank=rank, epoch=r["epoch"],
                   update=r["update"], selection_metric=METRIC[stage], dev_value=r["validation"], file=r["file"],
                   checkpoint_sha256=sha, identity=canonical_hash(identity), sources=ident["sources"],
                   config=ident["config"], split=ident["split"], members=ident["members"], upstream=upstream)
        load_rank(dest, row, identity)
        ranks.append(row)
    require(len({r["checkpoint_sha256"] for r in ranks}) == 5, "aliased top-five files")
    first = ranks[0]
    selected = dict(stage=stage, trained=True, frozen=True, weights_sha256=first["checkpoint_sha256"],
                    epoch=first["epoch"], updates=first["update"], selection=METRIC[stage], value=first["dev_value"],
                    identity=canonical_hash(identity))
    if stage == "head":
        selected["base"] = identity["base"]
    require(read(dest / "FROZEN.json") == selected and digest(dest / "best.pt") == first["checkpoint_sha256"],
            "frozen rank1 disagrees with best")
    return dict(status="COMPLETE", stage=stage, epoch=len(history), update=last["update"],
                stop_reason=last["stop_reason"], history_sha256=digest(dest / "EPOCH_METRICS.json"),
                selected=selected, ranks=ranks)


def paired_indexes(proofs):
    density = proofs["density"]["ranks"][0]
    providers = [dict(row, density_checkpoint=density["checkpoint_sha256"], density_epoch=density["epoch"],
                      density_file="density/" + density["file"], head_file="head/" + row["file"])
                 for row in proofs["head"]["ranks"]]
    joints = [dict(row, density_checkpoint=density["checkpoint_sha256"],
                   head_checkpoint=providers[0]["checkpoint_sha256"], provider_dev_rank=1)
              for row in proofs["joint"]["ranks"]]
    return dict(PROVIDER_TOP5=providers, JOINT_TOP5=joints)


def freeze_all(root, proofs):
    run = read(root / "FORMAL_RUN.json")
    require(set(proofs) == set(FOLDS) and all(set(p) == set(SUBSTEPS) for p in proofs.values()),
            "all five folds must be complete")
    frozen = dict(schema="bcep-v2-formal-dev-top5-freeze-v1", run=canonical_hash(run), code=run["commit"],
                  matrix=MATRIX, folds={f: dict(substeps=proofs[f], **paired_indexes(proofs[f])) for f in FOLDS})
    path = root / "ALL_FOLDS_FROZEN.json"
    stamp = read(path)["frozen_unix"] if path.exists() else time.time()
    payload = dict(frozen, frozen_unix=stamp)
    immutable(path, dict(payload, content_sha256=canonical_hash(payload)))


def read_freeze(root):
    record = read(root / "ALL_FOLDS_FROZEN.json")
    content = {k: v for k, v in record.items() if k != "content_sha256"}
    require(record["content_sha256"] == canonical_hash(content)
            and record["run"] == canonical_hash(read(root / "FORMAL_RUN.json"))
            and record["matrix"] == MATRIX and set(record["folds"]) == set(FOLDS), "freeze identity mismatch")
    for fold in FOLDS:
        stages = record["folds"][fold]["substeps"]
        require(set(stages) == set(SUBSTEPS), "global barrier is incomplete")
        for stage, proof in stages.items():
            require(proof["status"] == "COMPLETE" and read(root / fold / "receipts" / (stage + ".json")) == proof,
                    "stale or incomplete stage receipt")
    return record

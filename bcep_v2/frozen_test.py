"""Read-only, post-freeze evaluation of the predeclared checkpoint matrix."""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

from bcep.contracts import canonical_hash, digest, require
from bcep_formal.control import install
from bcep_formal.data import FormalData
from bcep_joint.common import read, seed_all, write
from .bank import Bank, bank_identity, check_export_pause, export
from .evaluation import evaluate
from .formal_manifest import (DIAGNOSTIC, FOLDS, OFFICIAL, PURPOSE, contract, dev_ranks, immutable,
                              load_rank, read_freeze, stage_identity, validate_worker_lease, write_csv)
from .joint import build_joint
from .provider import CandidateHead, Provider, spatial_encoding
from .runtime import configure_cuda
from .workflow import SEEDS, run_command


def verify_ranks(folder, stages):
    for stage in ("density", "head", "joint"):
        proof = stages[stage]
        directory = folder / stage
        require(digest(directory / "EPOCH_METRICS.json") == proof["history_sha256"]
                and read(directory / "FROZEN.json") == proof["selected"], "dev provenance changed")
        history = dev_ranks(read(directory / "EPOCH_METRICS.json"))
        index = read(directory / "TOP5.json")
        for r, original, frozen in zip(history, index, proof["ranks"], strict=True):
            require(original == dict(r, sha256=frozen["checkpoint_sha256"]) and
                    (r["epoch"], r["update"], r["validation"]) ==
                    (frozen["epoch"], frozen["update"], frozen["dev_value"]), "frozen ranking mismatch")


def case_folder(root, row):
    stage = row["stage"]
    base = root / row["fold"] / "evaluation"
    if stage == "joint" and row["dev_rank"] == 1:
        return base / "official"
    return base / stage / f"rank{row['dev_rank']}_e{row['epoch']:03d}_{row['checkpoint_sha256']}"


@torch.no_grad()
def evaluate_case(root, fold, kind, rank):
    frozen = read_freeze(root)
    ctx = contract(root, fold)
    cfg, split, folder, geo, ident = ctx
    stages = frozen["folds"][fold]["substeps"]
    verify_ranks(folder, stages)
    seal = frozen["content_sha256"]
    if kind != "official":
        for f in FOLDS:
            done = read(root / f / "evaluation/official/COMPLETE.json")
            require(done["status"] == "COMPLETE" and done["identity"]["frozen_run"]["seal"] == seal
                    and digest(root / f / "evaluation/official/PER_SEED_METRICS.csv") == done["metrics_sha256"],
                    "official matrix must precede diagnostics")
    data = FormalData(split, "export")
    device = torch.device("cuda")
    seed_all(cfg["provider_seed"])
    p = Provider().to(device).eval().requires_grad_(False)
    for stage, module in (("density", p.base), ("head", p.head)):
        load_rank(folder / stage, stages[stage]["ranks"][0], stage_identity(ctx, stage, stages), module)
    density, head = (stages[k]["selected"] for k in ("density", "head"))

    def get_bank(seed, refined, create):
        area = "refined_banks" if refined else "test_base20"
        dest = folder / area / f"{PURPOSE}_{seed}"
        identity = bank_identity(split, PURPOSE, seed, geo, ident["sources"], density, head if refined else None)
        if create:
            def predict(idx):
                x = data.item(idx, PURPOSE, False)
                obs = x["observed"].to(device)
                return p.candidates(obs, x["semantic_maps"].to(device),
                                    torch.ones(len(obs), device=device, dtype=torch.bool),
                                    x["scene_id"], x["agent_ids"], seed, geo, refined=refined)
            export(dest, identity, split, predict)
        return Bank(dest, identity, split, interruptible=True)

    banks = {s: get_bank(s, True, kind == "official") for s in SEEDS}
    if kind == "heads":
        bases = {s: get_bank(s, False, True) for s in SEEDS}
        return head_matrix(ctx, frozen, p, data, bases, banks, device)
    ranks = frozen["folds"][fold]["JOINT_TOP5"]
    row = ranks[rank - 1]
    joint_cfg = dict(cfg["joint"], scales=split["bridge_scales"])
    seed_all(joint_cfg["downstream_seed"])
    model = build_joint(joint_cfg).to(device)
    load_rank(folder / "joint", row, stage_identity(ctx, "joint", stages), model)
    return evaluate(split, model, row, banks, case_folder(root, row), device,
                    mode="GH" if rank == 1 else "H", extra_identity=dict(
                        seal=seal, role=OFFICIAL if rank == 1 else DIAGNOSTIC, sources=ident["sources"],
                        provider=[density["weights_sha256"], head["weights_sha256"]]))


@torch.no_grad()
def head_matrix(ctx, frozen, provider, data, bases, banks, device):
    _, split, folder, geo, ident = ctx
    fold = split["fold"]
    stages = frozen["folds"][fold]["substeps"]
    ranks = frozen["folds"][fold]["PROVIDER_TOP5"]
    heads = []
    for row in ranks:
        head = CandidateHead().to(device)
        load_rank(folder / "head", row, stage_identity(ctx, "head", stages), head)
        heads.append(head)
    ids = split[PURPOSE + "_indices"]
    dest = folder / "evaluation/head_matrix"
    identity = dict(seal=frozen["content_sha256"], density=stages["density"]["selected"],
                    heads=ranks, sources=ident["sources"], members=canonical_hash([split["members"][i] for i in ids]),
                    base_banks={str(s): b.hash for s, b in bases.items()},
                    official_banks={str(s): b.hash for s, b in banks.items()}, M=256, K=20, steps=0)
    immutable(dest / "IDENTITY.json", identity)
    idhash = canonical_hash(identity)
    # One feature map per window; all five seeds and all five independent head instances reuse it.
    row_hashes = {}
    for idx in ids:
        check_export_pause(dest, idx)
        output = dest / f"{idx}.npz"
        side = output.with_suffix(".json")
        if side.exists():
            receipt = read(side)
            require(receipt == dict(identity=idhash, index=idx, sha256=digest(output)), "changed head output")
        else:
            x = data.item(idx, PURPOSE, False)
            obs, maps = x["observed"].to(device), x["semantic_maps"].to(device)
            valid = torch.ones(len(obs), device=device, dtype=torch.bool)
            context, features = spatial_encoding(provider.base, obs, maps, valid)
            values = {}
            for seed in SEEDS:
                base = bases[seed].get(idx).to(device)
                for row, head in zip(ranks, heads, strict=True):
                    result, _ = head(provider.base, obs, maps, valid, base, context, features, geo)
                    result = result.cpu()
                    require(result.shape == (20, len(obs), 2) and torch.isfinite(result).all(), "invalid head output")
                    if row["dev_rank"] == 1:
                        require(torch.equal(result, banks[seed].get(idx)), "rank1 differs from official refined bank")
                    else:
                        values[f"r{row['dev_rank']}_s{seed}"] = result.numpy()
            tmp = output.with_suffix(".tmp.npz")
            if output.exists():
                with np.load(output, allow_pickle=False) as orphan:
                    require(set(orphan.files) == set(values) and all(
                        np.array_equal(orphan[k], v) for k, v in values.items()), "uncommitted head output differs")
            else:
                np.savez_compressed(tmp, **values)
                tmp.replace(output)
            receipt = dict(identity=idhash, index=idx, sha256=digest(output))
            write(side, receipt)
        row_hashes[str(idx)] = receipt["sha256"]
    immutable(dest / "PREDICTIONS_FROZEN.json", dict(identity=idhash, rows=row_hashes))
    scoring = FormalData(split, "scoring")
    counts = {group: [0, 0] for group in ("all", "N1", "Nge2")}
    totals = {}
    for idx in ids:
        check_export_pause(dest, idx)
        truth = scoring.item(idx, PURPOSE)["future"][:, -1]
        groups = ("all", "N1" if len(truth) == 1 else "Nge2")
        for group in groups:
            counts[group][0] += len(truth)
            counts[group][1] += 1
        with np.load(dest / f"{idx}.npz", allow_pickle=False) as arrays:
            for seed in SEEDS:
                for row in ranks:
                    rank = row["dev_rank"]
                    goals = banks[seed].get(idx) if rank == 1 else torch.from_numpy(arrays[f"r{rank}_s{seed}"])
                    total = (goals - truth).norm(dim=-1).amin(0).double().sum().item()
                    for group in groups:
                        key = (rank, seed, group)
                        totals[key] = totals.get(key, 0.) + total
    table = []
    for (rank, seed, group), total in totals.items():
        agents, scenes = counts[group]
        table.append(dict(fold=fold, seed=seed, group=group, method="Provider", dev_rank=rank,
                          agent_count=agents, scene_count=scenes, sum_minFDE=total, minFDE=total/agents,
                          base20=bases[seed].hash, bank_sha256=banks[seed].hash if rank == 1 else idhash))
    target = dest / "PER_SEED_METRICS.csv"
    write_csv(target, table)
    receipt = dict(status="COMPLETE", identity=idhash, seal=frozen["content_sha256"], metrics_sha256=digest(target))
    immutable(dest / "COMPLETE.json", receipt)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("official", "joint", "heads"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--fold", choices=FOLDS, required=True)
    p.add_argument("--rank", type=int, default=1)
    a = p.parse_args()
    require((a.kind == "joint" and 2 <= a.rank <= 5) or (a.kind != "joint" and a.rank == 1), "matrix range")
    validate_worker_lease(a.root, os.environ.get("BCEP_FORMAL_LOCK_FD"))
    install()
    configure_cuda()
    return run_command(lambda: evaluate_case(a.root, a.fold, a.kind, a.rank), a.root / a.fold, a.kind)


if __name__ == "__main__":
    raise SystemExit(main())

"""Freeze observed-only five-seed paths before reading held-out labels."""

import csv
import numpy as np
import torch
from bcep.contracts import canonical_hash, digest, require
from bcep_formal.control import pause_requested
from bcep_formal.data import FormalData
from bcep_joint.common import read, write
from bcep_joint.inference import generate, metric_invariants, shuffle_paths
from ebjd.metrics import MetricAccumulator
from ebjd.unified import gather_candidates


@torch.no_grad()
def evaluate(split, model, frozen, banks, folder, device):
    identity = dict(
        checkpoint=frozen,
        split=canonical_hash(split),
        banks={str(seed): b.hash for seed, b in banks.items()},
        K=20,
        steps=20,
        unit="m",
    )
    require(list(banks) == list(range(2035, 2040)), "incomplete seed set")
    if (folder / "IDENTITY.json").exists():
        require(read(folder / "IDENTITY.json") == identity, "cannot overwrite a different evaluation")
    write(folder / "IDENTITY.json", identity)
    obs = FormalData(split, "export")
    indices = split["heldout_confirmation_indices"]
    for seed, bank in banks.items():
        rows = []
        dest = folder / str(seed)
        dest.mkdir(exist_ok=True)
        for idx in indices:
            if pause_requested():
                raise InterruptedError("evaluation paused between windows")
            path = dest / f"{idx}.npz"
            side = path.with_suffix(".json")
            if side.exists():
                row = read(side)
                require(row == dict(index=idx, sha256=digest(path)), "changed trajectory")
            else:
                x = obs.batch([idx], "heldout_confirmation", False)
                goals = bank.batch([idx], x).to(device)
                x = x.to(device)
                g, h, permutation, _ = generate(
                    model, x.observed, x.semantic_maps, x.valid, x.scene_ids, x.agent_ids, goals, seed
                )
                tmp = path.with_suffix(".tmp.npz")
                np.savez_compressed(
                    tmp, G=g.cpu().numpy(), H=h.cpu().numpy(), pi=permutation.cpu().numpy()
                )
                tmp.replace(path)
                row = dict(index=idx, sha256=digest(path))
                write(side, row)
            rows.append(row)
        write(dest / "COMPLETE.json", dict(identity=canonical_hash(identity), rows=rows))
    scoring = FormalData(split, "scoring")
    table = []
    for seed, bank in banks.items():
        acc = {}
        for idx in indices:
            x = scoring.batch([idx], "heldout_confirmation")
            with np.load(folder / str(seed) / f"{idx}.npz", allow_pickle=False) as z:
                g, h, pi = (torch.from_numpy(z[name]) for name in ("G", "H", "pi"))
            goals = bank.batch([idx], x)
            metric_invariants(g, h, goals, gather_candidates(goals, pi), x.future, x.valid)
            paths = dict(G=g, H=h)
            for j in range(5):
                paths[f"H_shuffle_{j}"] = shuffle_paths(h, x.valid, x.scene_ids, seed, j)
            for group in ("all", "N1" if int(x.valid.sum()) == 1 else "Nge2"):
                for method, path in paths.items():
                    acc.setdefault((group, method), MetricAccumulator()).update(path, x.future, x.valid)
        for (group, method), value in acc.items():
            table.append(
                dict(
                    fold=split["fold"],
                    seed=seed,
                    group=group,
                    method=method,
                    **{k: v for k, v in value.as_dict().items() if k != "sums"},
                )
            )
    target = folder / "PER_SEED_METRICS.csv"
    tmp = target.with_suffix(".tmp")
    with tmp.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    tmp.replace(target)
    receipt = dict(status="COMPLETE", identity=identity, metrics_sha256=digest(target))
    write(folder / "COMPLETE.json", receipt)
    return receipt

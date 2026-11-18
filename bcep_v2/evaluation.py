"""Freeze observed-only five-seed paths before reading held-out labels."""

import csv
import numpy as np
import torch
from bcep.contracts import canonical_hash, digest, require
from .bank import check_export_pause
from bcep_formal.data import FormalData
from bcep_joint.common import read, write
from bcep_joint.inference import check_permutation, generate, metric_invariants, shuffle_paths
from ebjd.metrics import MetricAccumulator, hard_metric_statistics
from ebjd.unified import gather_candidates


@torch.no_grad()
def evaluate(split, model, frozen, banks, folder, device, *, mode="GH", extra_identity=None):
    identity = dict(
        checkpoint=frozen,
        split=canonical_hash(split),
        banks={str(seed): b.hash for seed, b in banks.items()},
        K=20,
        steps=20,
        unit="m",
    )
    require(list(banks) == list(range(2035, 2040)), "incomplete seed set")
    require(mode in ("GH", "H"), "unknown trajectory evaluation mode")
    if extra_identity is not None:
        identity.update(mode=mode, frozen_run=extra_identity)
    if extra_identity is not None and folder.exists() and not (folder / "IDENTITY.json").exists():
        require(not any(folder.iterdir()), "refusing unidentified formal prediction files")
    if (folder / "IDENTITY.json").exists():
        require(read(folder / "IDENTITY.json") == identity, "cannot overwrite a different evaluation")
    else:
        write(folder / "IDENTITY.json", identity)
    case_hash = canonical_hash(identity)
    obs = FormalData(split, "export")
    indices = split["heldout_confirmation_indices"]
    for seed, bank in banks.items():
        rows = []
        dest = folder / str(seed)
        dest.mkdir(exist_ok=True)
        for idx in indices:
            check_export_pause(dest, idx)
            path = dest / f"{idx}.npz"
            side = path.with_suffix(".json")
            source = dict(index=idx)
            if extra_identity is not None:
                source.update(identity=case_hash, seed=seed, member=split["members"][idx])
            if side.exists():
                row = read(side)
                require(row == dict(source, sha256=digest(path)), "changed trajectory")
            else:
                x = obs.batch([idx], "heldout_confirmation", False)
                goals = bank.batch([idx], x).to(device)
                x = x.to(device)
                g, h, permutation, _ = generate(
                    model, x.observed, x.semantic_maps, x.valid, x.scene_ids, x.agent_ids, goals, seed, mode=mode
                )
                tmp = path.with_suffix(".tmp.npz")
                arrays = dict(H=h.cpu().numpy(), pi=permutation.cpu().numpy())
                if g is not None:
                    arrays["G"] = g.cpu().numpy()
                if path.exists():
                    with np.load(path, allow_pickle=False) as orphan:
                        require(set(orphan.files) == set(arrays) and all(
                            np.array_equal(orphan[key], value) for key, value in arrays.items()),
                            "uncommitted trajectory differs from frozen replay")
                else:
                    np.savez_compressed(tmp, **arrays)
                    tmp.replace(path)
                row = dict(source, sha256=digest(path))
                write(side, row)
            rows.append(row)
        write(dest / "COMPLETE.json", dict(identity=canonical_hash(identity), rows=rows))
    scoring = FormalData(split, "scoring")
    table = []
    for seed, bank in banks.items():
        acc = {}
        dest = folder / str(seed)
        previous = dest / "SCORE.json"
        score_identity = dict(identity=canonical_hash(identity), predictions=digest(dest / "COMPLETE.json"))
        if previous.exists():
            saved = read(previous)
            require(saved["source"] == score_identity and canonical_hash(saved["rows"]) == saved["rows_hash"],
                    "changed seed score")
            table.extend(saved["rows"])
            continue
        for idx in indices:
            check_export_pause(dest, idx)
            x = scoring.batch([idx], "heldout_confirmation")
            with np.load(folder / str(seed) / f"{idx}.npz", allow_pickle=False) as z:
                h, pi = (torch.from_numpy(z[name]) for name in ("H", "pi"))
                g = torch.from_numpy(z["G"]) if mode == "GH" else None
            goals = bank.batch([idx], x)
            check_permutation(pi, x.valid)
            gathered = gather_candidates(goals, pi)
            torch.testing.assert_close(h[..., -1, :], gathered, atol=1e-6, rtol=1e-6)
            paths = dict(H=h)
            if mode == "GH":
                metric_invariants(g, h, goals, gathered, x.future, x.valid)
                paths["G"] = g
                for j in range(5):
                    paths[f"H_shuffle_{j}"] = shuffle_paths(h, x.valid, x.scene_ids, seed, j)
            else:
                stats = hard_metric_statistics(h, x.future, x.valid)
                error = (goals - x.future[:, None, :, -1]).norm(dim=-1)
                assigned_error = (gathered - x.future[:, None, :, -1]).norm(dim=-1)
                joint = (assigned_error * x.valid[:, None]).sum(-1).min(1).values / x.valid.sum(1)
                torch.testing.assert_close(error.min(1).values[x.valid], stats.agent_minfde[x.valid],
                                           atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(joint, stats.scene_jfde, atol=1e-6, rtol=1e-6)
            for group in ("all", "N1" if int(x.valid.sum()) == 1 else "Nge2"):
                for method, path in paths.items():
                    acc.setdefault((group, method), MetricAccumulator()).update(path, x.future, x.valid)
        seed_rows = []
        for (group, method), value in acc.items():
            seed_rows.append(
                dict(
                    fold=split["fold"],
                    seed=seed,
                    group=group,
                    method=method,
                    **{k: v for k, v in value.as_dict().items() if k != "sums"},
                )
            )
        write(previous, dict(source=score_identity, rows=seed_rows, rows_hash=canonical_hash(seed_rows)))
        table.extend(seed_rows)
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

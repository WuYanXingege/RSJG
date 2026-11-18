"""Versioned, per-window candidate cache. Ground truth never enters this API."""

import numpy as np
import torch
from bcep.contracts import canonical_hash, digest
from bcep_joint.common import write, read


def bank_identity(
    split, purpose, seed, geometry, sources, base, head=None, *, view=None, test_only=False
):
    if (
        head is not None
        and not test_only
        and not (
            head.get("trained")
            and head.get("frozen")
            and head.get("stage") == "head"
            and head.get("base") == base
            and head.get("updates", 0) > 0
        )
    ):
        raise ValueError("formal v2 export requires trained frozen head from this base")
    if not test_only and not (
        base["trained"]
        and base["frozen"]
        and base["stage"] == "density"
        and base.get("updates", 0) > 0
        and base.get("weights_sha256")
    ):
        raise ValueError("untrained density")
    members = [split["members"][idx] for idx in split[purpose + "_indices"]]
    return dict(
        schema="bcep-v2-spatial-budget-head-v1" if head else "bcep-v2-base-cache-v1",
        fold=split["fold"],
        split=canonical_hash(split),
        member_hash=canonical_hash(members),
        geometry=canonical_hash(geometry),
        scales=split["scales"],
        sources=sources,
        purpose=purpose,
        seed=seed,
        view=view,
        base=base,
        head=head,
        augmentation="canonical",
        test_only=test_only,
        trained=not test_only,
        no_future_as_input=True,
        K=20,
        M=256,
        dtype="float32",
        unit="m",
        selector=dict(
            method="kmeans++/Lloyd",
            n_init=5,
            max_iter=100,
            tolerance_m=1e-6,
            ordering="original kmeans center order",
        ),
    )


def export(folder, identity, split, predict):
    folder.mkdir(parents=True, exist_ok=True)
    header = folder / "IDENTITY.json"
    if header.exists() and read(header) != identity:
        raise ValueError("cache identity mismatch")
    if (folder / "COMPLETE.json").exists():
        Bank(folder, identity, split)
        return
    if not header.exists() and any(folder.iterdir()):
        raise ValueError("refusing to adopt unidentified cache files")
    write(header, identity)
    ids = split[identity["purpose"] + "_indices"]
    rows = []
    for idx in ids:
        member = split["members"][idx]
        path = folder / f"{idx}.npy"
        receipt = path.with_suffix(".json")
        if receipt.exists():
            row = read(receipt)
            if row["member"] != member or row["sha256"] != digest(path):
                raise ValueError("incomplete or changed cache row")
        else:
            with torch.no_grad():
                goals = predict(idx).detach().cpu().numpy()
            if (
                goals.dtype != np.float32
                or goals.shape != (20, len(member["agent_ids"]), 2)
                or not np.isfinite(goals).all()
            ):
                raise ValueError("invalid candidate output")
            temporary = path.with_suffix(".tmp.npy")
            np.save(temporary, goals, allow_pickle=False)
            temporary.replace(path)
            row = dict(index=idx, member=member, sha256=digest(path))
            write(receipt, row)
        rows.append(row)
    write(folder / "COMPLETE.json", dict(identity_hash=canonical_hash(identity), rows=rows))


class Bank:
    def __init__(self, folder, identity, split):
        self.folder = folder
        header = read(folder / "IDENTITY.json")
        done = read(folder / "COMPLETE.json")
        if header != identity or done["identity_hash"] != canonical_hash(identity):
            raise ValueError("candidate provenance mismatch")
        ids = split[identity["purpose"] + "_indices"]
        if [row["index"] for row in done["rows"]] != ids:
            raise ValueError("window order mismatch")
        self.rows = {}
        for row in done["rows"]:
            idx = row["index"]
            if row["member"] != split["members"][idx] or digest(folder / f"{idx}.npy") != row["sha256"]:
                raise ValueError("member or candidate digest mismatch")
            self.rows[idx] = row
        self.hash = canonical_hash(done)

    def get(self, idx):
        if idx not in self.rows:
            raise ValueError("window outside bank")
        return torch.from_numpy(np.load(self.folder / f"{idx}.npy", allow_pickle=False))

    def batch(self, ids, batch):
        goals = torch.zeros(len(ids), 20, batch.valid.shape[1], 2)
        for b, idx in enumerate(ids):
            member = self.rows[idx]["member"]
            if (batch.scene_ids[b], batch.frame_ids[b], batch.agent_ids[b]) != (
                member["scene_id"],
                member["frame_ids"],
                member["agent_ids"],
            ):
                raise ValueError("batch identity mismatch")
            goals[b, :, batch.valid[b]] = self.get(idx)
        return goals

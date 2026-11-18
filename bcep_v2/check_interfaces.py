"""Read-only, bounded ZARA2 compatibility check using two training members."""

import argparse
import json
from pathlib import Path
import torch
from bcep.contracts import digest, require
from bcep_formal.data import FormalData
from bcep_joint.common import read, seed_all
from ebjd.unified import gather_candidates
from selector_control.core import local_seed
from selector_control.algorithms import kmeans_centers
from .workflow import CONFIG, load_contract
from .provider import Provider
from .joint import build_joint


@torch.no_grad()
def check(plan_path):
    seed_all(470)
    cfg, split, _, geo, identity = load_contract(CONFIG, plan_path, "zara2")
    fold = Path(read(plan_path)["folds"]["zara2"]["output"])
    loaded = {}
    for stage, receipt in [("provider", "PROVIDER_FROZEN.json"), ("joint", "JOINT_FROZEN.json")]:
        frozen = read(fold / receipt)
        sha = digest(fold / stage / "best.pt")
        require(sha == frozen["sha256"], "unexpected source checkpoint")
        loaded[stage] = torch.load(fold / stage / "best.pt", map_location="cpu", weights_only=False)[
            "model"
        ]
    p = Provider().eval()
    p.base.load_state_dict(loaded["provider"], strict=True)
    m = build_joint(dict(cfg["joint"], scales=split["bridge_scales"]), loaded["joint"]).eval()
    added = sorted(set(m.state_dict()) - set(loaded["joint"]))
    for name, value in loaded["joint"].items():
        if not torch.equal(value, m.state_dict()[name]):
            raise ValueError("migration changed " + name)
    data = FormalData(split, "export", verify=False)
    train = split["train_indices"]
    ids = [next(i for i in train if len(split["members"][i]["agent_ids"]) == n) for n in (1, 2)]
    rows = []
    for idx in ids:
        require(
            digest(split["records"][idx]["path"]) == split["records"][idx]["sha256"], "record mismatch"
        )
        x = data.batch([idx], "train", False)
        obs, maps, v = x.observed[0], x.semantic_maps[0], x.valid[0]
        counter = []
        hook = p.base.encoder.map[0].register_forward_hook(lambda *args: counter.append(1))
        goals = p.candidates(obs, maps, v, x.scene_ids[0], x.agent_ids[0], 2035, geo)
        hook.remove()
        atoms, _ = p.base.sample_support(obs, maps, v, x.scene_ids[0], x.agent_ids[0], 2035, 256)
        legacy = torch.stack(
            [
                torch.as_tensor(
                    kmeans_centers(
                        atoms[:, i].numpy(),
                        20,
                        local_seed("bcep_flow", x.scene_ids[0], agent, 2035),
                        n_init=5,
                        max_iter=100,
                        tolerance_m=1e-6,
                    )["centers"]
                ).float()
                for i, agent in enumerate(x.agent_ids[0])
            ],
            dim=1,
        )
        require(torch.equal(goals, legacy) and len(counter) == 1, "zero residual/encoder reuse mismatch")
        context = m.encode_context(x.observed, x.semantic_maps, x.valid)
        h, pi, _ = m.coupling.assign(
            context, goals[None], m.bridge.representation.goal_scale, straight_through=False
        )
        require(torch.equal(h, gather_candidates(goals[None], pi)), "assignment mismatch")
        rows.append(
            dict(
                agents=len(obs),
                K=20,
                crop_pixels=maps.shape[-1],
                map_cnn_calls=len(counter),
                zero_head_exact=True,
                hard_gather_exact=True,
            )
        )
    return dict(
        fold="zara2",
        purpose="train_observed_only",
        rows=rows,
        optimizer_updates=0,
        density_parameters=sum(q.numel() for q in p.base.parameters()),
        head_parameters=sum(q.numel() for q in p.head.parameters()),
        joint_parameters=sum(q.numel() for q in m.parameters()),
        joint_added_parameters=sum(q.numel() for n, q in m.named_parameters() if n in added),
        joint_added_keys=added,
        provider_added_keys=["head." + n for n in p.head.state_dict()],
        source_checkpoint_sha256={s: digest(fold / s / "best.pt") for s in loaded},
        split_sha256=identity["split"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    print(json.dumps(check(parser.parse_args().plan), indent=2))

"""Stage objectives. Production and CPU fixtures share these functions."""

import torch
from bcep_formal.optimizations import provider_loss_batched
from ebjd.data import augment_batch_with_points
from ebjd.metrics import MetricAccumulator
from bcep_joint.inference import generate
from .provider import spatial_encoding, head_loss_components
from .objective import bridge_and_pair_loss


def augment(batch, goals, generator):
    # Carry basis probes through the *same* transform as each scene's candidates.
    centers = (batch.observed[:, :, -1] * batch.valid[..., None]).sum(1) / batch.valid.sum(1)[:, None]
    basis = torch.cat([torch.zeros(1, 2), torch.eye(2)])
    probes = (centers[:, None, None, :] + basis[None, :, None, :]).expand(-1, -1, goals.shape[2], -1)
    batch, points = augment_batch_with_points(batch, torch.cat([goals, probes], 1), generator)
    basis_out = points[:, -3:, 0]
    return (
        batch,
        points[:, :-3],
        dict(
            center=centers,
            rotation=(basis_out[:, 1:] - basis_out[:, :1]).transpose(1, 2),
            scene_ids=batch.scene_ids,
            kind="rigid",
            align_corners=False,
        ),
    )


def count_agents(split, ids):
    return sum(len(split["members"][i]["agent_ids"]) for i in ids)


def density_callbacks(provider, data, device):
    def compute(idx, purpose):
        x = data.item(idx, purpose)
        obs, maps, future = (x[key].to(device) for key in ("observed", "semantic_maps", "future"))
        valid = torch.ones(len(obs), device=device, dtype=torch.bool)
        return provider_loss_batched(provider.base, obs, maps, valid, future)

    def backward(ids, order, views):
        count = count_agents(data.split, ids)
        loss = 0.0
        sums = dict(nll_final_sum=0.0, h4_nll_sum=0.0, h8_nll_sum=0.0)
        for idx in ids:
            row = compute(idx, "train")
            value = row["loss_total"] * (row["valid_agent_count"] / count)
            value.backward()
            loss += float(value.detach())
            sums["nll_final_sum"] += float(row["nll_final_sum"].detach())
            for h in (4, 8):
                sums[f"h{h}_nll_sum"] += float(row["auxiliary"][h]["nll_sum"].detach())
        return dict(loss=loss, valid_agent_count=count, **sums)

    @torch.no_grad()
    def validate():
        ids = data.split["selection_dev_indices"]
        sums = {h: 0.0 for h in (4, 8, 12)}
        micro_agents = count_agents(data.split, ids)
        for i in ids:
            c = compute(i, "selection_dev")
            sums[12] += float(c["nll_final_sum"])
            for h in (4, 8):
                sums[h] += float(c["auxiliary"][h]["nll_sum"])
        validate.metrics = {f"h{h}_nll_agent_micro": value / micro_agents for h, value in sums.items()}
        return sums[12] / micro_agents

    return backward, validate


def head_callbacks(provider, data, device, geometry, training, validation):
    provider.base.eval().requires_grad_(False)

    def encode(batch):
        x = batch.to(device)
        obs, maps, valid = x.observed[0], x.semantic_maps[0], x.valid[0]
        with torch.no_grad():
            h, features = spatial_encoding(provider.base, obs, maps, valid)
        return x, h, features

    def score(encoded, goals):
        x, h, features = encoded
        obs, maps, valid = x.observed[0], x.semantic_maps[0], x.valid[0]
        refined, shift = provider.head(
            provider.base, obs, maps, valid, goals[0].to(device), h, features, geometry
        )
        return head_loss_components(refined, shift, x.future[0, :, -1], valid)

    def backward(ids, order, views):
        count = count_agents(data.split, ids)
        loss = 0.0
        sums = dict(head_fde_sum=0.0, head_move_sum=0.0, head_loss_sum=0.0)
        for i in ids:
            bank = training[int(torch.randint(len(training), (), generator=views))]
            batch = data.batch([i], "train")
            batch, goals, _ = augment(batch, bank.batch([i], batch), order)
            total, fde, move = score(encode(batch), goals)
            value = total / count
            value.backward()
            loss += float(value.detach())
            sums["head_loss_sum"] += float(total.detach())
            sums["head_fde_sum"] += float(fde.detach())
            sums["head_move_sum"] += float(move)
        return dict(loss=loss, valid_agent_count=count, **sums)

    @torch.no_grad()
    def validate():
        ids = data.split["selection_dev_indices"]
        seeds = range(2035, 2040)
        banks = dict(zip(seeds, validation, strict=True))
        sums = dict.fromkeys(seeds, 0.0)
        counts = dict.fromkeys(seeds, 0)
        for idx in ids:
            batch = data.batch([idx], "selection_dev")
            encoded = encode(batch)
            n = int(batch.valid.sum())
            for seed, bank in banks.items():
                _, distance, _ = score(encoded, bank.batch([idx], batch))
                sums[seed] += float(distance)
                counts[seed] += n
            del encoded
        if set(counts.values()) != {count_agents(data.split, ids)} or not ids:
            raise ValueError("invalid validation-agent counts")
        validate.metrics = {f"seed{s}_minFDE_m": sums[s] / counts[s] for s in seeds}
        validate.totals = {s: (sums[s], counts[s]) for s in seeds}
        return sum(validate.metrics.values()) / len(seeds)

    return backward, validate


def joint_callbacks(model, data, device, config, training, validation):
    def backward(ids, order, views):
        metrics = {}
        for start in range(0, len(ids), config["batch_scenes"]):
            micro = ids[start : start + config["batch_scenes"]]
            batch = data.batch(micro, "train")
            batch, goals, _ = augment(batch, training.batch(micro, batch), order)
            counts = batch.valid.sum(1).tolist()
            times = torch.randint(1, 101, (len(micro),), device=device).float() / 100
            noise = torch.randn(len(micro), 1, batch.valid.shape[1], 12, 2, device=device)
            a = 0
            while a < len(micro):
                b = a + 1
                while b < len(micro) and max(counts[a : b + 1]) * (b + 1 - a) <= config["group_max_slots"]:
                    b += 1
                physical = batch.compact_scenes(a, b).to(device)
                # Formal collation packs valid rows first. Verify before slicing.
                n = physical.valid.shape[1]
                if not torch.equal(batch.valid[a:b, :n], physical.valid.cpu()):
                    raise ValueError("noncanonical collator padding")
                _, _, _, agent, scene = bridge_and_pair_loss(
                    model,
                    physical,
                    goals[a:b, :, :n].to(device),
                    time_draw=times[a:b],
                    noise=noise[a:b, :, :n],
                )
                fraction = (b - a) / len(micro)
                loss = (agent * sum(counts[a:b]) / sum(counts) + scene * fraction) / config[
                    "gradient_accumulation"
                ]
                loss.backward()
                metrics["loss"] = metrics.get("loss", 0.0) + float(loss.detach())
                for key, value in model.last_pair_metrics.items():
                    w = (b - a) / len(ids) if key != "effective_edges" else 1
                    metrics[key] = metrics.get(key, 0.0) + float(value) * w
                c = model.last_loss_components
                for key in ("clean_path_raw_sum", "clean_path_signal_sum"):
                    metrics[key] = metrics.get(key, 0.0) + float(c[key])
                metrics["valid_scene_count"] = metrics.get("valid_scene_count", 0) + (b - a)
                for key in ("clean_path_term", "diffusion_term", "geometry_term", "pair_term", "map_term"):
                    w = sum(counts[a:b]) / sum(counts) if key in ("diffusion_term", "map_term") else fraction
                    metrics[key] = metrics.get(key, 0.0) + float(c[key]) * w / config["gradient_accumulation"]
                a = b
        return metrics

    def validate():
        acc = MetricAccumulator()
        for i in data.split["selection_dev_indices"]:
            batch = data.batch([i], "selection_dev")
            goals = validation.batch([i], batch).to(device)
            x = batch.to(device)
            _, h, _, _ = generate(
                model,
                x.observed,
                x.semantic_maps,
                x.valid,
                x.scene_ids,
                x.agent_ids,
                goals,
                2035,
                mode="H",
            )
            acc.update(h, x.future, x.valid)
        validate.metrics = acc.as_dict()
        return validate.metrics["JADE"]

    return backward, validate

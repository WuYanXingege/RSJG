"""Stage objectives. Production and CPU fixtures share these functions."""

import torch
from bcep_formal.optimizations import provider_loss_batched
from ebjd.data import augment_batch_with_points
from ebjd.metrics import MetricAccumulator
from ebjd.execution import enabled
from bcep_joint.inference import generate
from .provider import spatial_encoding, head_loss_components
from .objective import bridge_and_pair_loss


def _add_log(values, key, value, weight=1.0):
    # Match the old Python-double accumulation, without synchronizing each GPU
    # scalar after every physical microbatch. These tensors never retain a graph.
    item = value.detach().double() if enabled('logging') and isinstance(value, torch.Tensor) else float(value)
    values[key] = values.get(key, 0.0) + item * weight


def _finish_log(values):
    keys = [key for key, value in values.items() if isinstance(value, torch.Tensor)]
    if keys:
        host = torch.stack([values[key] for key in keys]).cpu().tolist()
        values.update(zip(keys, host))
    return values


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
        sums = dict(loss=0.0, nll_final_sum=0.0, h4_nll_sum=0.0, h8_nll_sum=0.0)
        for idx in ids:
            row = compute(idx, "train")
            value = row["loss_total"] * (row["valid_agent_count"] / count)
            value.backward()
            _add_log(sums, "loss", value)
            _add_log(sums, "nll_final_sum", row["nll_final_sum"])
            for h in (4, 8):
                _add_log(sums, f"h{h}_nll_sum", row["auxiliary"][h]["nll_sum"])
        return dict(valid_agent_count=count, **_finish_log(sums))

    @torch.no_grad()
    def validate():
        ids = data.split["selection_dev_indices"]
        sums = {h: 0.0 for h in (4, 8, 12)}
        micro_agents = count_agents(data.split, ids)
        for i in ids:
            c = compute(i, "selection_dev")
            _add_log(sums, 12, c["nll_final_sum"])
            for h in (4, 8):
                _add_log(sums, h, c["auxiliary"][h]["nll_sum"])
        sums = _finish_log(sums)
        validate.metrics = {f"h{h}_nll_agent_micro": value / micro_agents for h, value in sums.items()}
        return sums[12] / micro_agents

    return backward, validate


class _FrozenValidationContexts:
    """Bounded CPU cache for unaugmented dev only, tied to this frozen base.

    Keep admitted entries instead of LRU-evicting the whole prefix on each
    sequential full-dev pass. Nonadmitted windows are always recomputed.
    """
    def __init__(self, base, max_bytes):
        self.base=base;self.max_bytes=int(max_bytes);self.entries={}
        self.used=0;self.hits=0;self.misses=0;self.version=None

    def refresh(self):
        values=tuple(self.base.parameters())+tuple(self.base.buffers())
        version=tuple((id(x),x._version,str(x.device),x.dtype) for x in values)
        if version!=self.version:
            self.entries.clear();self.used=0;self.version=version

    def get(self,key,device):
        if key in self.entries:
            self.hits+=1
            return tuple(x.to(device) if x is not None else None for x in self.entries[key])
        self.misses+=1
        return None

    def put(self,key,values):
        size=sum(x.numel()*x.element_size() for x in values if x is not None)
        if size+self.used<=self.max_bytes and key not in self.entries:
            self.entries[key]=tuple(x.detach().cpu() if x is not None else None for x in values)
            self.used+=size

    def info(self):
        return dict(bytes=self.used,max_bytes=self.max_bytes,entries=len(self.entries),hits=self.hits,misses=self.misses)


def head_callbacks(provider, data, device, geometry, training, validation, *, frozen_context_cache_bytes=1024**3):
    provider.base.eval().requires_grad_(False)
    contexts=_FrozenValidationContexts(provider.base,frozen_context_cache_bytes)

    def encode(batch,cache_key=None):
        x = batch.to(device)
        obs, maps, valid = x.observed[0], x.semantic_maps[0], x.valid[0]
        cached=contexts.get(cache_key,device) if cache_key is not None and enabled('head_cache') else None
        if cached is not None:
            return x,*cached
        with torch.no_grad():
            h, features = spatial_encoding(provider.base, obs, maps, valid)
        if cache_key is not None and enabled('head_cache'):
            contexts.put(cache_key,(h,features))
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
        sums = dict(loss=0.0, head_fde_sum=0.0, head_move_sum=0.0, head_loss_sum=0.0)
        for i in ids:
            bank = training[int(torch.randint(len(training), (), generator=views))]
            batch = data.batch([i], "train")
            batch, goals, _ = augment(batch, bank.batch([i], batch), order)
            total, fde, move = score(encode(batch), goals)
            value = total / count
            value.backward()
            _add_log(sums, "loss", value)
            _add_log(sums, "head_loss_sum", total)
            _add_log(sums, "head_fde_sum", fde)
            _add_log(sums, "head_move_sum", move)
        return dict(valid_agent_count=count, **_finish_log(sums))

    @torch.no_grad()
    def validate():
        contexts.refresh()
        ids = data.split["selection_dev_indices"]
        seeds = range(2035, 2040)
        banks = dict(zip(seeds, validation, strict=True))
        sums = dict.fromkeys(seeds, 0.0)
        counts = dict.fromkeys(seeds, 0)
        for idx in ids:
            batch = data.batch([idx], "selection_dev")
            encoded = encode(batch,cache_key=idx)
            n = int(batch.valid.sum())
            for seed, bank in banks.items():
                _, distance, _ = score(encoded, bank.batch([idx], batch))
                _add_log(sums, seed, distance)
                counts[seed] += n
            del encoded
        if set(counts.values()) != {count_agents(data.split, ids)} or not ids:
            raise ValueError("invalid validation-agent counts")
        sums = _finish_log(sums)
        validate.metrics = {f"seed{s}_minFDE_m": sums[s] / counts[s] for s in seeds}
        validate.totals = {s: (sums[s], counts[s]) for s in seeds}
        validate.context_cache_info=contexts.info()
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
                _add_log(metrics, "loss", loss)
                for key, value in model.last_pair_metrics.items():
                    w = (b - a) / len(ids) if key != "effective_edges" else 1
                    _add_log(metrics, key, value, w)
                c = model.last_loss_components
                for key in ("clean_path_raw_sum", "clean_path_signal_sum"):
                    _add_log(metrics, key, c[key])
                metrics["valid_scene_count"] = metrics.get("valid_scene_count", 0) + (b - a)
                for key in ("clean_path_term", "diffusion_term", "geometry_term", "pair_term", "map_term"):
                    w = sum(counts[a:b]) / sum(counts) if key in ("diffusion_term", "map_term") else fraction
                    # Keep the same multiply-then-divide order in FP64.
                    value = (c[key].detach().double() if enabled('logging') else float(c[key])) * w / config["gradient_accumulation"]
                    _add_log(metrics, key, value)
                a = b
        return _finish_log(metrics)

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

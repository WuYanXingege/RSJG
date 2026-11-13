#!/usr/bin/env python3
"""Bounded, paired complete-update benchmark of the current gated8 network.

No formal checkpoints are written. Input preparation is timed in every arm;
the same raw batches, rigid augmentation and diffusion RNG stream are replayed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import fields
import gc
import hashlib
import json
from pathlib import Path
import resource
import statistics
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from ebjd.config import build_model, load_config
from ebjd.data import SceneManifestDataset, collate_scenes, augment_batch_with_points
import ebjd.objectives as objectives
from benchmark_runtime_metrics import UnifiedMetricBuffer
from ebjd.unified import UnifiedCouplingBridge
from ebjd.encoders import EBJDContext
from run_zara1_unified_coupling_bridge import (
    atomic_json, bridge_and_pair_loss, make_bank, optimizer_for, seed_all,
)
from ebjd.proposals import file_sha256


def legacy_masked_mean(value, valid):
    mask = valid
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value).to(value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1)


def compact_masked_mean(value, valid):
    mask = valid
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    factor = 1
    for small, large in zip(mask.shape, value.shape):
        if small != large and small != 1:
            raise ValueError("mask cannot broadcast to loss")
        factor *= large // small
    mask = mask.to(value.dtype)
    return (value * mask).sum() / (mask.sum() * factor).clamp_min(1)


def hash_tensors(*values):
    h = hashlib.sha256()
    for value in values:
        array = value.detach().cpu().contiguous().numpy()
        h.update(str((array.shape, array.dtype)).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def scene_encoded_proxy(model, cpu, proposals, device):
    """Prototype: retain exact scene-local encoding/assignment, batch denoising.

    This wrapper is benchmark-only. Padding/cat operations retain autograd;
    no learned context or coupled endpoint is detached.
    """
    contexts, coupled_goals = [], []
    maximum = cpu.valid.shape[1]
    for scene in range(cpu.valid.shape[0]):
        active = cpu.valid[scene].nonzero(as_tuple=False).flatten()
        batch = cpu.compact_scenes(scene, scene + 1).to(device)
        goals = proposals[scene:scene + 1, :, active].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            context = model.encode_context(batch.observed, batch.semantic_maps, batch.valid)
            coupled, _, _ = model.coupling.assign(
                context, goals, model.bridge.representation.goal_scale, straight_through=True)
        contexts.append(context)
        pad = coupled.new_zeros(coupled.shape[0], coupled.shape[1], maximum-len(active), 2)
        coupled_goals.append(torch.cat((coupled, pad), dim=2))
    merged = {}
    for field in fields(EBJDContext):
        values = [getattr(context, field.name) for context in contexts]
        if not isinstance(values[0], torch.Tensor):
            merged[field.name] = values[0]
            continue
        padded = []
        for value in values:
            if field.name == "observed_bias":
                extra = maximum - value.shape[-1]
                padded.append(torch.nn.functional.pad(value, (0, extra, 0, extra)))
            else:
                shape = list(value.shape); shape[1] = maximum - value.shape[1]
                padded.append(torch.cat((value, value.new_zeros(shape)), dim=1))
        merged[field.name] = torch.cat(padded, dim=0)
    context = EBJDContext(**merged)
    coupled = torch.cat(coupled_goals, dim=0)
    return SimpleNamespace(
        encode_context=lambda *args: context,
        coupling=SimpleNamespace(assign=lambda *args, **kwargs: (coupled, None, None)),
        bridge=model.bridge, training_options=model.training_options)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-bank", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--updates", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-gpu-seconds", type=float, default=900)
    parser.add_argument("--candidate", choices=("deferred_and_mask", "full_microbatch", "scene_encoded_batch",
                                               "small_denoiser_graph"),
                        default="deferred_and_mask")
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--gate-updates", type=int, default=8)
    parser.add_argument("--gate-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("use a new benchmark output directory")
    torch.set_num_threads(2)
    config = load_config(args.config)
    config["denoiser"].update(block_type="endpoint_gated_dit_v1", coarse_blocks=4, refinement_blocks=4)
    dataset = SceneManifestDataset(config["paths"]["train_manifest"])
    bank = make_bank(args.train_bank, config["paths"]["matched_gdts_validation_checkpoint_sha256"],
                     file_sha256(config["paths"]["train_manifest"]), "train")
    batch_size = int(config["training"]["batch_scenes"])
    accumulation = int(config["training"]["gradient_accumulation"])
    count = (args.warmup + max(args.updates, args.gate_updates)) * accumulation * batch_size
    if count > len(dataset):
        raise ValueError("benchmark requires unique windows within an arm")
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(3101))[:count].tolist()
    # One immutable CPU initialization shared by all arms. No GPU yet.
    seed_all(3101)
    base = UnifiedCouplingBridge(build_model(config, proposal_conditioned=True))
    # Scales are taken from the prior verified full-train scale fit, not re-fit
    # to this measurement subset. CLI base config must supply those scales.
    base.training_options = {"coarse": 0.25, "geometry": 0.05, "map": 0.1}
    initial = deepcopy(base.state_dict())
    del base
    report = {"status": "RUNNING", "formal_optimizer_updates": 0,
              "temporary_optimizer_updates": 0, "runs": [],
              "config": str(Path(args.config).resolve()), "config_sha256": file_sha256(args.config),
              "train_bank_sha256": file_sha256(args.train_bank),
              "architecture": "gated8_192_4_768", "batch_scenes": batch_size,
              "accumulation": accumulation, "initialization": "same fresh body + same-fold U-Net",
              "data_loader_workers": int(config["training"].get("data_loader_workers", 0)),
              "input_mode": "legacy_cpu_rigid", "precision": "cuda_bf16_with_fp32_losses",
              "scales": {"goal": config["representation"]["goal_scale_m"],
                         "bridge": config["representation"]["bridge_scale_m"]},
              "budget_seconds": args.max_gpu_seconds}
    report["candidate"] = args.candidate
    report["benchmark_script_sha256"] = file_sha256(__file__)
    atomic_json(output / "RESULTS.json", report)
    gpu_start = time.perf_counter()
    original_mask = objectives._masked_mean
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; run with host GPU access")
        device = torch.device("cuda:0")
        report["gpu"] = torch.cuda.get_device_name(device)
        report["torch"] = torch.__version__

        def remaining():
            if time.perf_counter() - gpu_start >= args.max_gpu_seconds:
                raise TimeoutError("bounded benchmark GPU budget exhausted")

        def run(mode, label, updates, *, diagnostics=False):
            remaining()
            seed_all(3101)
            model = UnifiedCouplingBridge(build_model(config, proposal_conditioned=True)).to(device).train()
            model.training_options = {"coarse": 0.25, "geometry": 0.05, "map": 0.1}
            model.load_state_dict(initial, strict=True)
            model.bridge.denoiser.uncheckpointed_max_agent_slots = 0
            if mode == "small_denoiser_graph":
                model.bridge.denoiser.uncheckpointed_max_agent_slots = 16
            optimizer = optimizer_for(model, config)
            objectives._masked_mean = compact_masked_mean if mode == "deferred_and_mask" else legacy_masked_mean
            seed_all(7231)
            augmentation = torch.Generator().manual_seed(3101)
            workers = int(config["training"].get("data_loader_workers", 0))
            data_loader = DataLoader(
                dataset, batch_sampler=[order[i:i+batch_size] for i in range(0,len(order),batch_size)],
                collate_fn=collate_scenes, num_workers=workers,
                generator=torch.Generator().manual_seed(3101),
                **({"prefetch_factor": int(config["training"].get("data_loader_prefetch_factor", 2)),
                    "persistent_workers": True} if workers else {}))
            iterator = iter(data_loader)
            torch.cuda.reset_peak_memory_stats()
            rows, hashes, stats, timings, gradient_checks = [], [], [], [], {}
            cursor = 0
            phase = {}
            events = []
            handles = []
            if diagnostics:
                # One sampled update only. Event recording is excluded from
                # paired performance windows and never enters formal training.
                def pre(name):
                    def callback(module, inputs):
                        start = torch.cuda.Event(enable_timing=True)
                        start.record()
                        module._runtime_event = start
                    return callback
                def post(name):
                    def callback(module, inputs, result):
                        stop = torch.cuda.Event(enable_timing=True)
                        stop.record()
                        events.append((name, module._runtime_event, stop))
                    return callback
                for name, module in (("msce", model.bridge.encoder), ("coupling", model.coupling),
                                     ("denoiser", model.bridge.denoiser)):
                    handles += [module.register_forward_pre_hook(pre(name)),
                                module.register_forward_hook(post(name))]
            samples = []
            warmups = 1 if diagnostics else args.warmup
            for step in range(warmups + updates):
                remaining()
                torch.cuda.synchronize()
                started = time.perf_counter()
                read_seconds = prepare_seconds = backward_seconds = optimizer_seconds = 0.0
                optimizer.zero_grad(set_to_none=True)
                metrics = UnifiedMetricBuffer(deferred=mode == "deferred_and_mask")
                for micro in range(accumulation):
                    t = time.perf_counter()
                    ids = order[cursor:cursor + batch_size]
                    cursor += batch_size
                    cpu = next(iterator)
                    proposals = bank.batch(cpu)
                    read_seconds += time.perf_counter() - t
                    t = time.perf_counter()
                    cpu, proposals = augment_batch_with_points(cpu, proposals, augmentation)
                    prepare_seconds += time.perf_counter() - t
                    if diagnostics or (label.startswith("gate") and step >= warmups):
                        hashes.append(hash_tensors(cpu.observed, cpu.future, cpu.semantic_maps,
                                                   cpu.valid, proposals))
                    times = torch.randint(1, 101, (batch_size,), device=device).float() / 100
                    noise = torch.randn(batch_size, 1, cpu.observed.shape[1], 12, 2, device=device)
                    if label.startswith("gate") and step >= warmups:
                        hashes.append(hash_tensors(times, noise))
                    total_agents = int(cpu.valid.sum())
                    metrics.begin_microbatch()
                    samples.append({"agents": cpu.valid.sum(-1).tolist(),
                                    "padding_slots": int(cpu.valid.numel()), "scene_ids": cpu.scene_ids,
                                    "map_hw": list(cpu.semantic_maps.shape[-2:])})
                    active_model = (scene_encoded_proxy(model, cpu, proposals, device)
                                    if mode == "scene_encoded_batch" else model)
                    groups = ([None] if mode in {"full_microbatch", "scene_encoded_batch"}
                              else range(batch_size))
                    for scene in groups:
                        if scene is None:
                            # Existing unified runner's non-compacted path.
                            # All reductions already respect scene/agent masks.
                            physical, goals = cpu.to(device), proposals.to(device)
                            time_draw, noise_draw = times, noise
                            aw = sw = 1.0
                        else:
                            active = cpu.valid[scene].nonzero(as_tuple=False).flatten()
                            physical = cpu.compact_scenes(scene, scene + 1).to(device)
                            goals = proposals[scene:scene + 1, :, active].to(device)
                            time_draw = times[scene:scene + 1]
                            noise_draw = noise[scene:scene + 1, :, active.to(device)]
                            aw, sw = active.numel() / total_agents, 1.0 / batch_size
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            _, bridge, pair, agent_part, scene_part = bridge_and_pair_loss(
                                active_model, physical, goals, time_draw=time_draw, noise=noise_draw)
                            total = aw * agent_part + sw * scene_part
                        if diagnostics:
                            torch.cuda.synchronize(); t = time.perf_counter()
                        (total / accumulation).backward()
                        if diagnostics:
                            torch.cuda.synchronize(); backward_seconds += time.perf_counter() - t
                        metrics.add(total, pair, agent_part, bridge, aw, sw)
                coupling_grad = torch.sqrt(sum(p.grad.float().square().sum() for p in model.coupling.parameters() if p.grad is not None))
                bridge_grad = torch.sqrt(sum(p.grad.float().square().sum() for p in model.bridge.parameters() if p.grad is not None))
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if diagnostics:
                    torch.cuda.synchronize(); t = time.perf_counter()
                optimizer.step()
                report["temporary_optimizer_updates"] += 1
                if diagnostics:
                    torch.cuda.synchronize(); optimizer_seconds = time.perf_counter() - t
                values = metrics.flush()
                grad_values = torch.stack([coupling_grad, bridge_grad, norm]).float().cpu().tolist()
                row = {"loss": sum(v[0] for v in values) / accumulation,
                       "pair": sum(v[1] for v in values) / accumulation,
                       "bridge": sum(v[2] for v in values) / accumulation,
                       "gradient_norms": grad_values}
                if not all(np.isfinite(v) for v in [row["loss"], *grad_values]):
                    raise FloatingPointError("non-finite temporary update")
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if step >= warmups:
                    rows.append(row)
                    timings.append(elapsed)
                    stats.append({"read_collate": read_seconds, "cpu_augment": prepare_seconds,
                                  "backward_sampled": backward_seconds, "optimizer_sampled": optimizer_seconds})
                if step == warmups - 1:
                    events.clear()
            for handle in handles:
                handle.remove()
            for name, begin, end in events:
                phase[name] = phase.get(name, 0.0) + begin.elapsed_time(end) / 1000
            parameters = {name: p.detach().cpu().clone() for name, p in model.named_parameters()}
            opt_state = {name: {key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
                                for key, value in optimizer.state[p].items()}
                         for name, p in model.named_parameters() if p in optimizer.state}
            for prefix in ("bridge.encoder", "coupling", "bridge.denoiser"):
                gradient_checks[prefix] = any(n.startswith(prefix) and p.grad is not None and bool(p.grad.abs().sum())
                    for n,p in model.named_parameters())
            result = {"label":label,"mode":mode,"updates":updates,"warmup":warmups,
                      "s_per_update":statistics.fmean(timings),"median_s_per_update":statistics.median(timings),
                      "peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,
                      "host_max_rss_mib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                      "rows":rows,"phase_seconds_sampled":phase,"timing_components":stats,
                      "input_rng_hashes":hashes,"batch_statistics":samples,
                      "gradient_checks":gradient_checks,"timing_is_diagnostic":diagnostics}
            del model, optimizer, iterator, data_loader
            gc.collect(); torch.cuda.empty_cache()
            report["runs"].append(result)
            report["gpu_wall_seconds"]=time.perf_counter()-gpu_start
            atomic_json(output/"RESULTS.json",report)
            print(json.dumps({"label":label,"s/update":result["s_per_update"]}),flush=True)
            return result, parameters, opt_state

        if not args.skip_profile:
            diagnostic, _, _ = run("legacy","profile",1,diagnostics=True)
        gates = []
        for mode,label in (("legacy","gate_baseline"),("legacy","gate_repeat"),
                           (args.candidate,"gate_candidate")):
            result,params,state=run(mode,label,args.gate_updates)
            gates.append((result,params,state))
        reference,ref_params,ref_state=gates[0]
        gate_results=[]
        for result,params,state in gates[1:]:
            if result["input_rng_hashes"] != reference["input_rng_hashes"]:
                raise RuntimeError("paired input or RNG mismatch")
            relative=lambda a,b: abs(a-b)/max(abs(a),abs(b),1e-6)
            max_loss=max(relative(a["loss"],b["loss"]) for a,b in zip(reference["rows"],result["rows"]))
            max_grad=max(relative(a["gradient_norms"][-1],b["gradient_norms"][-1]) for a,b in zip(reference["rows"],result["rows"]))
            numerator=sum((params[k].double()-v.double()).square().sum() for k,v in ref_params.items())
            denominator=sum(v.double().square().sum() for v in ref_params.values())
            exact=all(torch.equal(params[k],v) for k,v in ref_params.items())
            state_exact=all(torch.equal(state[k][s],v) if isinstance(v,torch.Tensor) else state[k][s]==v
                            for k,entry in ref_state.items() for s,v in entry.items())
            gate_results.append({"label":result["label"],"max_loss_relative_difference":max_loss,
                "max_gradient_norm_relative_difference":max_grad,"parameter_relative_l2":float((numerator/denominator).sqrt()),
                "parameters_exact":exact,"optimizer_exact":state_exact})
            report["correctness"] = gate_results
            if max_loss>0.001 or max_grad>0.01 or not all(result["gradient_checks"].values()):
                raise RuntimeError("fixed numerical gate failed")
        report["correctness"]=gate_results
        atomic_json(output / "RESULTS.json", report)
        del gates,ref_params,ref_state,params,state
        for repeat in range(0 if args.gate_only else 2):
            # Counterbalance order to reduce warm-cache / temperature bias.
            modes=("legacy",args.candidate) if repeat==0 else (args.candidate,"legacy")
            for mode in modes:
                run(mode,f"timing_{repeat+1}_{mode}",args.updates)
        report["status"]="COMPLETE"
    except BaseException as exc:
        report.update(status="FAILED",reason=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
        raise
    finally:
        objectives._masked_mean=original_mask
        report["gpu_wall_seconds"]=time.perf_counter()-gpu_start
        atomic_json(output/"RESULTS.json",report)


if __name__ == "__main__":
    main()

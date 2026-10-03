#!/usr/bin/env python3
"""One canonical A pass per seed. CUDA launches only on an unoccupied GPU UUID.

preflight is metadata-only: no Torch/model import, no cache rebuilding.
export refuses existing output directories; interrupted records remain immutable.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.jdv2_stage_a_bank import (
    REPO, CONFIG, CONFIG_SHA, CHECKPOINT, CHECKPOINT_SHA, CACHE, CACHE_SHA,
    CACHE_PIN, SOURCE_BATCHES, SEEDS, SCRIPT_PATHS, PROTOCOL, METRICS,
    BankError, require, sha256, code_provenance, write_json, write_npz,
    array_contract, graph_summary, json_bytes, digest_bytes,
)
import numpy as np

def gpu_snapshot():
    def query(fields, kind):
        return subprocess.check_output(
            ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
            text=True).strip().splitlines()
    return {"time_utc": datetime.now(timezone.utc).isoformat(),
            "devices": [x.split(", ") for x in query("index,uuid,name,memory.total,memory.used,utilization.gpu", "gpu")],
            "compute_processes": [x.split(", ") for x in query("gpu_uuid,pid,process_name,used_memory", "compute-apps")]}

def require_idle(snapshot, uuid, allow_self=False):
    devices = [r for r in snapshot["devices"] if r[1] == uuid]
    require(len(devices) == 1, "GPU UUID unavailable", "BLOCKED_RESOURCE_UNAVAILABLE")
    owners = [p for p in snapshot["compute_processes"] if p[0] == uuid
              and not (allow_self and p[1] == str(os.getpid()))]
    require(not owners, f"GPU {uuid} has live compute owners: {owners}",
            "BLOCKED_RESOURCE_UNAVAILABLE")
    if not allow_self:
        require(int(devices[0][4]) <= 1024 and int(devices[0][5]) <= 5,
                "GPU not demonstrably idle", "BLOCKED_RESOURCE_UNAVAILABLE")

def preflight(require_committed=True):
    import yaml
    require(Path.cwd().resolve() == REPO, "Run from repository root", "BLOCKED_EXPORT_INPUT")
    code = code_provenance(require_committed)
    require(sha256(REPO / CONFIG) == CONFIG_SHA, "Config fingerprint changed", "BLOCKED_EXPORT_INPUT")
    require(sha256(REPO / CHECKPOINT) == CHECKPOINT_SHA, "Checkpoint changed", "BLOCKED_EXPORT_INPUT")
    require(sha256(REPO / CACHE / "manifest.json") == CACHE_SHA, "Cache manifest changed", "BLOCKED_EXPORT_INPUT")
    config = yaml.safe_load((REPO / CONFIG).read_text())
    manifest = json.loads((REPO / CACHE / "manifest.json").read_text())
    require(config["jdv2_cache_source_commit"] == manifest["source_commit"] == CACHE_PIN,
            "Cache pin mismatch", "BLOCKED_EXPORT_INPUT")
    require(digest_bytes(json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                                   ensure_ascii=True).encode()) == config["jdv2_cache_manifest_hash"],
            "Canonical cache content-hash mismatch", "BLOCKED_EXPORT_INPUT")
    ids = sorted(p.name for p in (REPO / SOURCE_BATCHES / "valid_batches").glob("*.pkl*"))
    require(len(ids) == 139, "Physical validation count changed", "BLOCKED_EXPORT_INPUT")
    require(all((REPO / CACHE / "valid" / f"{i:06d}.pt").is_file() for i in range(139)),
            "Missing frozen deployment record", "BLOCKED_EXPORT_INPUT")
    return {"checkpoint_sha256": CHECKPOINT_SHA, "config_sha256": CONFIG_SHA,
            "cache_manifest_sha256": CACHE_SHA, "cache_pin": CACHE_PIN,
            "cache_manifest_hash": config["jdv2_cache_manifest_hash"],
            "dataset_hash": manifest["dataset_hash"], "split_hash": manifest["split_hash"],
            "code": code, "physical_validation_filenames": ids,
            "full_source_cache_validation": "DEFERRED_TO_UNCHANGED_PRODUCTION_LOADER",
            "checkpoint_strict_load": "DEFERRED_NO_MODEL_CREATED",
            "source_batch_manifest_sha256": sha256(REPO / SOURCE_BATCHES / "cache_manifest.json")}

def to_numpy(tensor):
    import torch
    detached = tensor.detach()
    original = str(detached.dtype)
    if detached.dtype == torch.bfloat16:
        detached = detached.float()  # every finite BF16 value is exactly representable
    value = detached.cpu().contiguous().numpy().copy()
    return value, {"original_dtype": original, "storage_dtype": str(value.dtype),
                   "conversion": "exact BF16-to-FP32 widening" if original == "torch.bfloat16" else "none"}

def unbatch(value):
    import torch
    if torch.is_tensor(value):
        require(value.numel() == 1, "Expected a scalar stable ID", "FAILED_EXPORT")
        return value.item()
    if isinstance(value, (tuple, list)):
        require(len(value) == 1, "Ambiguous collated ID", "FAILED_EXPORT")
        return unbatch(value[0])
    if isinstance(value, np.generic):
        return value.item()
    require(isinstance(value, (str, int, float, bool)), "Unsupported ID", "FAILED_EXPORT")
    return value

def capture(evaluator, prediction, auxiliary, inputs, sequence, mask, batch_id, seed, index, provenance):
    """No sampling here. Prediction already integrated by the existing helper."""
    import torch
    net, args = evaluator.net, evaluator.net.args
    obs = args.obs_length
    scaled = prediction.detach() * args.down_factor
    gt_scaled = inputs["x_augmented"][:, :, 6:8].detach() * args.down_factor
    world = torch.stack([inputs["scene"].make_world_coord_torch(v) for v in scaled])
    gt_world = inputs["scene"].make_world_coord_torch(gt_scaled)
    y, y_dtype = to_numpy(world[:, obs:].permute(2, 0, 1, 3))
    gt, gt_dtype = to_numpy(gt_world[obs:].permute(1, 0, 2))
    n, p, t, _ = y.shape
    require((p, t) == (20, 12), "Unexpected prediction budget", "FAILED_EXPORT")
    require(inputs["scene_index"].unique().numel() == 1, "Packed scenes unsupported", "FAILED_EXPORT")
    arrays = {"Y": y, "GT": gt, "metric_mask": to_numpy(mask.bool())[0],
              "future_mask": to_numpy(sequence[obs:].T.bool())[0],
              "frame_ids": to_numpy(inputs["frame_ids"])[0],
              "sample_valid": np.ones((n, p), dtype=bool),
              "sample_weights": np.full((n, p), 1 / p, dtype=np.float64),
              "sample_indices": np.broadcast_to(np.arange(p, dtype=np.int64), (n, p)).copy(),
              "selected_candidate_ids": to_numpy(auxiliary["joint_candidate_index"][:, :p].long())[0],
              "joint_goals_world": to_numpy(auxiliary["joint_goal_points_world"][:, :p])[0],
              "joint_goals_map": to_numpy(auxiliary["joint_goal_points_map"][:, :p])[0],
              "trunk_goal_world": to_numpy(auxiliary["joint_goal_points_world"][:, p:])[0],
              "trunk_candidate_id": to_numpy(auxiliary["joint_candidate_index"][:, p:].long())[0],
              "edge_index": to_numpy(auxiliary["dependency_state"]["edge_index"].long())[0],
              "edge_weight": to_numpy(auxiliary["dependency_state"]["edge_weight"])[0],
              "goal_candidates_world_K": to_numpy(auxiliary["goal_candidates_world"])[0],
              "goal_candidates_map_K": to_numpy(auxiliary["goal_candidates_map"])[0],
              "candidate_log_prior_K": to_numpy(inputs["jdv2_cache"]["candidate_log_prior"])[0],
              "edge_relation_embedding_original_world": to_numpy(auxiliary["dependency_state"]["relation_embedding"])[0],
              "homography_H": np.asarray(inputs["scene"].H).copy()}
    arrays["degree"], arrays["component_index"] = graph_summary(n, arrays["edge_index"])
    baseline_lists = {}
    for name in (*METRICS.values(), "Joint_Goal_Endpoint_Error", "Joint_Goal_Compatibility"):
        baseline_lists[name] = [float(v) for v in net.compute_model_metrics(
            metric_name=name, predictions=prediction, metric_mask=mask,
            all_aux_outputs=auxiliary, inputs=inputs, obs_length=obs)]
    baseline = {"agent_minADE": baseline_lists["minADE@K"],
                "agent_minFDE": baseline_lists["minFDE@K"],
                "metrics": {name: float(np.mean(baseline_lists[production]))
                            for name, production in METRICS.items()},
                "all_production_lists": baseline_lists}
    source_path = Path(str(unbatch(batch_id["data_file_path"]))).resolve()
    require(source_path.is_file(), "Missing physical source file", "FAILED_EXPORT")
    source_id = source_path.relative_to((REPO / "data").resolve()).as_posix()
    scene_id = str(unbatch(batch_id["scene_name"]))
    filename = provenance["physical_validation_filenames"][index]
    metadata = {"source_id": source_id, "scene_id": scene_id,
                "window_id": f"valid/{filename}", "window_index": index,
                "inference_seed": seed,
                "agent_ids": [json.dumps(unbatch(v), ensure_ascii=False) for v in batch_id["agent_ids"]],
                "units": "world_metres", "dt": .4,
                "storage": {"Y": y_dtype, "GT": gt_dtype},
                "graph_provenance": "same_forward_history_only_frozen_cache",
                "baseline": baseline,
                "input_fingerprints": {
                    "raw_source_sha256": sha256(source_path),
                    "source_batch_sha256": sha256(REPO / SOURCE_BATCHES / "valid_batches" / filename),
                    "deployment_cache_sha256": sha256(REPO / CACHE / "valid" / f"{index:06d}.pt")},
                "auxiliary_axes": {"sample_aligned": list(("sample_indices", "selected_candidate_ids", "joint_goals_world", "joint_goals_map")),
                    "candidate_K": ["goal_candidates_world_K", "goal_candidates_map_K", "candidate_log_prior_K"],
                    "trunk_not_evaluation_sample": ["trunk_goal_world", "trunk_candidate_id"],
                    "edge_relation_embedding_original_world": "E,P,feature: original edge-context only; invalidated for independently recombined worlds; never scored as shuffled relations"}}
    return arrays, metadata

def export(args):
    # This gate precedes Torch import, output creation and all model setup.
    initial = gpu_snapshot()
    require_idle(initial, args.gpu_uuid)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    provenance = preflight()
    output = Path(args.output).resolve()
    require(output.is_relative_to(REPO / "outputs/joint_dependency_v2/eth/joint_dependency_v2/stage_a_banks"),
            "Output must be a new directory under stage_a_banks", "BLOCKED_EXPORT_INPUT")
    require(not output.exists(), "Output exists; no automatic retry/resampling or overwrite", "BLOCKED_EXPORT_INPUT")
    require_idle(gpu_snapshot(), args.gpu_uuid)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "INCOMPLETE.json", {"status": "INCOMPLETE", "provenance": provenance,
               "protocol": PROTOCOL, "command": sys.argv, "resource_gate": initial,
               "note": "Initial marker immutable. Only BANK_MANIFEST.json with COMPLETE establishes completion."})
    records, completed = [], 0
    try:
        import torch
        from src.metrics import compute_metric_mask
        from src.utils import isolated_random_seed
        from tools.audit_jdv2_stage_a import _active_evaluator
        from tools.jdv2_stage_b_v1_failure_mechanism_audit import velocities_to_predictions, rng_snapshot
        from tools.jdv2_stage_b_identity_contract_audit import _rng_equal
        evaluator, epoch = _active_evaluator(str(REPO / CONFIG), str(REPO / CHECKPOINT), output / "runtime", "cuda:0")
        net = evaluator.net
        require(epoch == 13 and net.args.training_stage == "joint_goal" and
                net.args.use_dependency_corrector is True and net.strict_no_z and
                net.args.amp_enabled and net.args.amp_dtype == "bf16",
                "Canonical route/precision mismatch", "BLOCKED_EXPORT_INPUT")
        head = net.jdv2_corrector.output[2]
        require(torch.count_nonzero(head.weight).item() == 0 and
                torch.count_nonzero(head.bias).item() == 0 and not net.training,
                "Stage-A output head not exactly zero/eval", "BLOCKED_EXPORT_INPUT")
        dataset = evaluator.data_loaders["valid"].dataset
        require(dataset.ids == provenance["physical_validation_filenames"] and len(dataset) == 139,
                "Loader order/protocol changed", "BLOCKED_EXPORT_INPUT")
        provenance.update(validated_cache_manifest=True, strict_checkpoint_load=True, exact_zero_eval_head=True,
                          dependencies={"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__},
                          resolved_args_sha256=digest_bytes(json_bytes(vars(net.args))))
        write_json(output / "RESOLVED_ARGS.json", vars(net.args))
        baseline_by_seed = {}
        with torch.no_grad():
            for seed in SEEDS:
                seed_records = []
                require_idle(gpu_snapshot(), args.gpu_uuid, allow_self=True)
                with isolated_random_seed(seed, use_cuda=True):
                    for index, (batch_data, batch_id) in enumerate(evaluator.data_loaders["valid"]):
                        # Stop our own export if another compute owner appears.
                        require_idle(gpu_snapshot(), args.gpu_uuid, allow_self=True)
                        inputs, sequence = net.prepare_inputs(batch_data, batch_id)
                        mask = compute_metric_mask(sequence)
                        net.jdv2_sampler.set_sampling_context(seed, index)
                        with evaluator._autocast_context():
                            contexts, auxiliary = net.encode(inputs, if_test=True)
                        with evaluator._autocast_context():
                            velocity = net.ts_sample(contexts, auxiliary["dependency_state"])
                        completed += 1
                        before_capture = rng_snapshot(True)
                        prediction = velocities_to_predictions(net, inputs, velocity)
                        arrays, row = capture(evaluator, prediction, auxiliary, inputs, sequence, mask, batch_id, seed, index, provenance)
                        relative = f"seed{seed}/window{index:06d}.npz"
                        write_npz(output / relative, arrays)
                        row.update(path=relative, sha256=sha256(output / relative), arrays=array_contract(arrays))
                        # Both numeric and JSON serialization must consume no model RNG.
                        json_bytes(row)
                        require(_rng_equal(before_capture, rng_snapshot(True)),
                                "Capture/serialization advanced RNG", "FAILED_EXPORT")
                        row["rng_capture_serialization_unchanged"] = True
                        write_json(output / f"seed{seed}/window{index:06d}.json", row)
                        require(_rng_equal(before_capture, rng_snapshot(True)), "Metadata writing advanced RNG", "FAILED_EXPORT")
                        records.append(row)
                        seed_records.append(row)
                        print(f"A_BANK seed={seed} window={index + 1}/139", flush=True)
                        del contexts, velocity, prediction, auxiliary, inputs, batch_data, arrays
                require(len(seed_records) == 139, "Incomplete seed", "FAILED_EXPORT")
                baseline_by_seed[str(seed)] = {}
                for name, production in METRICS.items():
                    selected = seed_records if name != "RME_legacy_full_mask" else [
                        r for r in seed_records if len(r["baseline"]["agent_minADE"]) == r["arrays"]["Y"]["shape"][0]]
                    if selected:
                        values = [v for r in selected for v in r["baseline"]["all_production_lists"][production]]
                        baseline_by_seed[str(seed)][name] = float(np.mean(values))
        require(completed == 695, "Wrong forward count", "FAILED_EXPORT")
        require(preflight()["code"] == provenance["code"], "Source changed during export", "FAILED_EXPORT")
        manifest = {"schema": "rsjg-canonical-a-bank-v1", "status": "COMPLETE",
                    "protocol": PROTOCOL, "provenance": provenance, "records": records,
                    "baseline_by_seed": baseline_by_seed, "forwards_completed": completed,
                    "command": sys.argv, "resource_gate": initial,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "unique_windows": 139, "scene_seed_records": len(records)}
        write_json(output / "BANK_MANIFEST.json", manifest)
        print(json.dumps({"status": "EXPORTED_NOT_YET_CPU_CERTIFIED",
                          "manifest": str(output / "BANK_MANIFEST.json"),
                          "sha256": sha256(output / "BANK_MANIFEST.json")}), flush=True)
    except BaseException as exc:
        write_json(output / "FAILED.json", {"status": getattr(exc, "status", "FAILED_EXPORT"),
                   "error": str(exc), "forwards_completed": completed,
                   "completed_scene_seed_records": len(records),
                   "completed_keys": [[r["inference_seed"], r["window_index"]] for r in records],
                   "resume_policy": "No automatic resampling; preserve this directory and resolve continuation explicitly."})
        raise

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preflight", "export"])
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        if args.command == "preflight":
            result = preflight(require_committed=True)
            result["gpu_snapshot"] = gpu_snapshot()
            print(json.dumps(result, indent=2))
        else:
            require(bool(args.gpu_uuid) and bool(args.output), "export requires --gpu-uuid and --output", "BLOCKED_EXPORT_INPUT")
            export(args)
    except BankError as exc:
        print(json.dumps({"status": exc.status, "error": str(exc)}), file=sys.stderr)
        return 2
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED_EXPORT_INPUT" if args.command == "preflight" else "FAILED_EXPORT",
                          "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

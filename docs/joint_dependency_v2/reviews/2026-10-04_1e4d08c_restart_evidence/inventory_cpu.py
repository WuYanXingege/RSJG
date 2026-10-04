#!/usr/bin/env python3
"""Read-only inventory of the two already-saved seed2036 prefixes. No Torch/model/replay."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

REPO = Path(__file__).resolve().parents[4]
BANKS = REPO / "outputs/joint_dependency_v2/eth/joint_dependency_v2/stage_a_banks"
OLD = BANKS / "canonical_a_desktop_authorized_20261003_seed2035_2039"
NEW = BANKS / "canonical_a_recovered_20261003_aa7022e_seed2035_2039"
PRIOR = REPO / "docs/joint_dependency_v2/reviews/2026-10-03_aa7022e_stage_a_bank_export_pairing"

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read(root, index):
    metadata = root / f"seed2036/window{index:06d}.json"
    row = json.loads(metadata.read_text())
    path = root / row["path"]
    assert row["inference_seed"] == 2036 and row["window_index"] == index
    assert sha(path) == row["sha256"]
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    assert set(arrays) == set(row["arrays"])
    for k, a in arrays.items():
        assert list(a.shape) == row["arrays"][k]["shape"]
        assert str(a.dtype) == row["arrays"][k]["dtype"]
        assert hashlib.sha256(a.tobytes()).hexdigest() == row["arrays"][k]["sha256"]
        assert a.dtype.kind in "biuf" and np.isfinite(a).all()
    return row, arrays

def main():
    declared = json.loads((PRIOR / "ARTIFACT_HASHES.json").read_text())["files"]
    for name, digest in declared.items():
        assert sha(PRIOR / name) == digest
    changes, target = [], None
    for index in range(36):
        left, a = read(OLD, index)
        right, b = read(NEW, index)
        assert set(a) == set(b)
        assert left["input_fingerprints"] == right["input_fingerprints"]
        assert all(left[k] == right[k] for k in ("agent_ids", "source_id", "window_id", "scene_id", "storage"))
        different = {}
        for name in a:
            assert a[name].shape == b[name].shape and a[name].dtype == b[name].dtype
            if a[name].tobytes() != b[name].tobytes():
                delta = np.abs(a[name].astype(np.float64) - b[name].astype(np.float64))
                different[name] = {"shape": list(a[name].shape), "dtype": str(a[name].dtype),
                    "different_elements": int(np.count_nonzero(a[name] != b[name])),
                    "max_abs_difference": float(delta.max()),
                    "first_array_coordinate": np.argwhere(a[name] != b[name])[0].tolist()}
        if different:
            changes.append({"window_index": index, "different_arrays": different})
        if index == 13:
            positions = np.argwhere(a["selected_candidate_ids"] != b["selected_candidate_ids"])
            filename = left["window_id"].split("/", 1)[1]
            inputs = {
                "raw_source_sha256": (REPO / "data").resolve() / left["source_id"],
                "source_batch_sha256": REPO / "outputs/joint_dependency_v2/cache/source_batches/eth5/eth/data_batches_jdv2_v2/valid_batches" / filename,
                "deployment_cache_sha256": REPO / "outputs/joint_dependency_v2/cache/eth_full_stage_a/valid/000013.pt"}
            verified = {}
            for key, path in inputs.items():
                actual = sha(path)
                assert actual == left["input_fingerprints"][key]
                verified[key] = actual
            target = {"window_index": 13, "source_id": left["source_id"], "window_id": left["window_id"],
                "N": a["Y"].shape[0], "E": a["edge_index"].shape[1],
                "available_array_fields": sorted(a), "equal_array_fields": sorted(set(a)-set(different)),
                "different_arrays": different, "input_bytes_rehashed": verified,
                "old_npz_sha256": left["sha256"], "new_npz_sha256": right["sha256"],
                "candidate_ID_changes": [{"agent_axis": int(i), "sample_axis": int(j),
                    "old": int(a["selected_candidate_ids"][i,j]), "new": int(b["selected_candidate_ids"][i,j])}
                    for i,j in positions],
                "same_forward_baseline_delta_new_minus_old": {
                    k: right["baseline"]["metrics"][k] - left["baseline"]["metrics"][k]
                    for k in left["baseline"]["metrics"]}}
    old_args = json.loads((OLD / "RESOLVED_ARGS.json").read_text())
    new_args = json.loads((NEW / "RESOLVED_ARGS.json").read_text())
    args_diff = sorted(k for k in set(old_args) | set(new_args) if old_args.get(k) != new_args.get(k))
    assert args_diff == ["model_dir", "save_dir"]
    old_marker = json.loads((OLD / "INCOMPLETE.json").read_text())
    new_manifest = json.loads((NEW / "BANK_MANIFEST.json").read_text())
    assert old_marker["provenance"]["code"]["source_files"] == new_manifest["provenance"]["code"]["source_files"]
    inventory = {}
    for name, root in (("old",OLD),("new",NEW)):
        inventory[name] = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    report = {
        "status": "COMPLETE_READ_ONLY_INVENTORY",
        "root_cause_status": "OPEN_INSUFFICIENT_INTERNAL_TRACES",
        "reviewed_source_commit": subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
        "scope": "Existing saved NPZ/metadata/file hashes and source inspection only; no metrics rerun, no model/torch import, no GPU, no sampling",
        "prior_archive_files_sha256_verified": len(declared),
        "prefix_records_per_attempt": 36, "NPZ_records_authenticated": 72,
        "first_different_saved_window_index": changes[0]["window_index"],
        "prefix_changes": changes, "target_window13": target,
        "resolved_args_different_keys": args_diff,
        "model_config_helper_source_fingerprints_equal": True,
        "saved_runtime_artifacts": {k:[p for p in v if p.startswith("runtime/")] for k,v in inventory.items()},
        "artifact_inventory_counts": {k:len(v) for k,v in inventory.items()},
        "artifact_inventory_sha256": {k:hashlib.sha256(json.dumps(v,separators=(",",":")).encode()).hexdigest() for k,v in inventory.items()},
        "non_window_artifacts": {k:[p for p in v if not p.startswith("seed")] for k,v in inventory.items()},
        "missing_in_both_export_artifacts": [
            "Round0 initial_candidate_index, sampled uniform/Gumbel bytes and generator states",
            "Round1/Round2 conditional scores, source/destination energies and accumulated costs",
            "exact-assignment realized tie priorities/objective tuples and intermediate selected IDs",
            "encoder agent/edge features, base relation logits, unary score and operator dtype/device traces",
            "trajectory generator noise, RNG snapshots/payloads before forward, velocity tensor",
            "operator execution trace and first-difference instrumentation"],
        "important_semantics": [
            "Saved relation embedding is selected_joint_relation AFTER final selected IDs, not the per-round relation tensor.",
            "Initial candidate IDs exist transiently in sampler return; the exporter does not serialize them.",
            "diagnostic_callback defaults None; reviewed export path does not attach a round-trace writer.",
            "Exact persistent refinement rounds do not use structured-Gumbel assignment noise; Round0 does use Gumbel initialization.",
            "Seed and tie-generation source are derivation recipes, not recorded realized payload evidence.",
            "rng_capture_serialization_unchanged is a boolean post-forward capture check, not persisted RNG state nor cross-run RNG equality.",
            "First changed saved record is not the first changed internal operator; NPZ key order is not execution order."],
        "replay_executed": False, "training_started": False, "SDD_resume_started": False}
    print(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CPU-only synthetic capture/NPZ/loader/parity/adversarial integration tests."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace, MethodType
from unittest.mock import patch
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import jdv2_stage_a_bank as bank
from tools import jdv2_export_stage_a_bank as exporter
from tools.jdv2_audit_stage_a_bank import audit_one, summarize

def main():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # Import functions only: no GDTS instance, no checkpoint, no CUDA initialization.
    from src.models.model import GDTS
    from src.data_src.scene_src.scene_eth5 import Scene_eth5
    from src.metrics import compute_metric_mask
    from tools.jdv2_stage_b_v1_failure_mechanism_audit import velocities_to_predictions, rng_snapshot
    from tools.jdv2_stage_b_identity_contract_audit import _rng_equal
    bank.require(not torch.cuda.is_initialized(), "CUDA was initialized before CPU tests")
    kernel = bank.load_kernel()
    _, jmm = kernel.modules()
    local = np.random.Generator(np.random.PCG64(202610035))
    results, rejected, max_parity, summary_rows = [], [], [], []
    def rejects(name, call, status=None):
        try:
            call()
        except (bank.BankError, ValueError, AssertionError) as exc:
            if status:
                assert isinstance(exc, bank.BankError) and exc.status == status, str(exc)
            rejected.append(name)
            return
        raise AssertionError("Invalid input accepted: " + name)

    # Temp artifacts are explicitly synthetic and automatically cleaned by this test.
    with tempfile.TemporaryDirectory(prefix="rsjg-stage-a-bank-cpu-") as raw_root:
        root = Path(raw_root)
        bank.write_json(root / "data/eth5/eth/synthetic.txt", {"scope": "SYNTHETIC_ONLY"})
        bank.write_json(root / bank.SOURCE_BATCHES / "valid_batches/synthetic.pkl", {})
        bank.write_json(root / bank.CACHE / "valid/000000.pt", {})
        scene_obj = Scene_eth5.__new__(Scene_eth5)
        scene_obj.name = "eth"
        scene_obj.H = np.asarray([[.1, 0., 0.], [0., .1, 0.], [0., 0., 1.]])
        args = SimpleNamespace(num_samples=20, seq_length=20, obs_length=8, down_factor=8)
        net = SimpleNamespace(args=args)
        net.compute_model_metrics = MethodType(GDTS.compute_model_metrics, net)
        evaluator = SimpleNamespace(net=net)
        cases = [("singleton", 1, [[], []]), ("multiagent_E0", 3, [[], []]),
                 ("mixed", 4, [[0, 1], [1, 2]]),
                 ("all_active", 3, [[0, 0, 1], [1, 2, 2]]),
                 ("partial_mask", 3, [[0, 1], [1, 2]]),
                 ("exact_ties", 2, [[0], [1]]), ("joint_change", 2, [[], []])]
        for name, n, edges in cases:
            x = torch.zeros((20, n, 8))
            x[:, :, 6:8] = torch.from_numpy(local.normal(size=(20, n, 2)).astype("float32"))
            sequence = torch.ones((20, n))
            if name == "partial_mask":
                sequence[-2:, -1] = 0
            mask = compute_metric_mask(sequence)
            edge = torch.tensor(edges, dtype=torch.int64).reshape(2, -1)
            frames = torch.arange(20, dtype=torch.int64)[:, None].expand(-1, n) * 6
            k_goals = torch.from_numpy(local.normal(size=(n, 21, 2)).astype("float32"))
            full_goal_world = scene_obj.make_world_coord_torch(k_goals * 8)
            auxiliary = {"joint_candidate_index": torch.arange(21).expand(n, -1),
                         "joint_goal_points_world": full_goal_world,
                         "joint_goal_points_map": k_goals,
                         "goal_candidates_world": full_goal_world,
                         "goal_candidates_map": k_goals,
                         "goal_point": k_goals.permute(1, 0, 2),
                         "edge_index": edge,
                         "dependency_state": {"edge_index": edge, "edge_weight": torch.ones(edge.shape[1]),
                            "relation_embedding": torch.zeros((edge.shape[1], 20, 16))}}
            inputs = {"x_augmented": x, "scene": scene_obj, "scene_index": torch.zeros(n, dtype=torch.int64),
                      "frame_ids": frames, "jdv2_cache": {"candidate_log_prior": torch.zeros((n, 21))}}
            ids = {"scene_name": ["eth"], "data_file_path": [str(root / "data/eth5/eth/synthetic.txt")],
                   "agent_ids": [torch.tensor([i]) for i in range(n)]}
            velocity = torch.from_numpy(local.normal(scale=.01, size=(20, n, 12, 2)).astype("float32"))
            before = rng_snapshot(False)
            prediction = velocities_to_predictions(net, inputs, velocity)
            for step in range(8, 20):
                torch.testing.assert_close(prediction[:, step], prediction[:, step - 1] + velocity[:, :, step - 8],
                                           atol=0, rtol=0)
            if name in ("exact_ties", "joint_change"):
                # Deliberately constructed CPU geometry, never real A predictions.
                inputs["x_augmented"].zero_()
                prediction.zero_()
                if name == "exact_ties":
                    inputs["x_augmented"][:, 0, 6] = -1.25
                    inputs["x_augmented"][:, 1, 6] = 1.25
                    prediction[1:, :, 0, 0] = -2.5
                    prediction[1:, :, 1, 0] = 2.5
                else:
                    prediction[10:, :, :, 0] = 12.5
            with patch.object(exporter, "REPO", root):
                arrays, record = exporter.capture(evaluator, prediction, auxiliary, inputs, sequence, mask,
                     ids, 2035, 0, {"physical_validation_filenames": ["synthetic.pkl"]})
            record.update(window_id=name, path=name + ".npz", rng_capture_serialization_unchanged=True)
            bank.write_npz(root / record["path"], arrays)
            record.update(sha256=bank.sha256(root / record["path"]), arrays=bank.array_contract(arrays))
            bank.write_json(root / (name + ".json"), record)
            assert _rng_equal(before, rng_snapshot(False))
            scene, restored = bank.load_record(root, record, kernel)
            for key in arrays:
                assert arrays[key].dtype == restored[key].dtype and arrays[key].tobytes() == restored[key].tobytes()
            original = kernel.score(scene, np, jmm)
            for metric in ("agent_minADE", "agent_minFDE"):
                bank.compare(original[metric], record["baseline"][metric], name + "/" + metric, max_parity)
            for metric in bank.METRICS:
                if metric in original["metrics"]:
                    bank.compare(original["metrics"][metric], record["baseline"]["metrics"][metric],
                                 name + "/" + metric, max_parity)
            names, detail, controls = audit_one(scene, kernel, jmm)
            # New detailed adapter must equal the frozen kernel, not a new metric/random protocol.
            previous = kernel.audit_scene(scene, bank.MASTERS, np, jmm)
            for metric in names:
                delta = detail["independent"][:, names.index(metric)] - original["metrics"][metric]
                np.testing.assert_allclose(delta.mean(),
                    previous["descriptive_deltas"][metric]["mean_shuffled_minus_original"], atol=1e-12, rtol=1e-12)
            summary_rows.append({"valid_agents": original["valid_agents"], "names": names, "arrays": detail})
            results.append({"fixture": name, "status": "PASS", **controls})
        # Round-trip guarantees exact BF16 value preservation via FP32 storage.
        bf16 = torch.from_numpy(local.normal(size=(2, 20, 12, 2)).astype("float32")).bfloat16()
        saved, dtype = exporter.to_numpy(bf16)
        assert torch.equal(torch.from_numpy(saved).bfloat16(), bf16)
        assert dtype["storage_dtype"] == "float32"
        # Validate reduction weights independently using unequal N fixtures.
        aggregate = summarize(summary_rows)
        expected = np.average([r["arrays"]["original"][r["names"].index("minADE")] for r in summary_rows],
                              weights=[r["valid_agents"] for r in summary_rows])
        assert aggregate["metrics"]["minADE"]["original"] == expected
        errors = []
        rejects("CPU_baseline_mismatch", lambda: bank.compare(1., 1.01, "deliberate", errors), "FAILED_CPU_PARITY")
        rejects("CPU_baseline_nonfinite", lambda: bank.compare(1., np.nan, "deliberate", errors), "FAILED_CPU_PARITY")
        rejects("CPU_baseline_shape", lambda: bank.compare([1.], [1., 2.], "deliberate", errors), "FAILED_CPU_PARITY")
        rejects("overwrite_json", lambda: bank.write_json(root / "joint_change.json", {}))
        rejects("overwrite_npz", lambda: bank.write_npz(root / record["path"], arrays))
        rejects("path_traversal", lambda: bank.safe_child(root, "../escape.npz"))
        rejects("wrong_file_hash", lambda: bank.load_record(root, dict(record, sha256="0" * 64), kernel))
        rejects("duplicate_agent_id", lambda: bank.load_record(root, dict(record, agent_ids=["x"] * n), kernel))
        rejects("wrong_units", lambda: bank.load_record(root, dict(record, units="pixel"), kernel))
        rejects("wrong_graph_origin", lambda: bank.load_record(root, dict(record, graph_provenance="GT_future"), kernel))
        rejects("RNG_capture_changed", lambda: bank.load_record(root, dict(record, rng_capture_serialization_unchanged=False), kernel))
        bad_dtype = copy.deepcopy(record)
        bad_dtype["storage"]["Y"]["original_dtype"] = "torch.float64"
        rejects("lossy_dtype_declaration", lambda: bank.load_record(root, bad_dtype, kernel))
        for label, mutate in [
            ("nonfinite", lambda a: a["Y"].__setitem__((0, 0, 0, 0), np.nan)),
            ("partial_future", lambda a: a["future_mask"].__setitem__((0, 0), False)),
            ("nonuniform_weight", lambda a: a["sample_weights"].__setitem__((0, 0), .1)),
            ("missing_sample", lambda a: a["sample_valid"].__setitem__((0, 0), False)),
            ("frame_mismatch", lambda a: a["frame_ids"].__setitem__((0, 0), -6)),
            ("candidate_out_of_range", lambda a: a["selected_candidate_ids"].__setitem__((0, 0), 21)),
            ("duplicate_sample_id", lambda a: a["sample_indices"].__setitem__((0, 1), 0)),
            ("aux_goal_mismatch", lambda a: a["joint_goals_world"].__setitem__((0, 0, 0), 100.)),
            ("component_mismatch", lambda a: a["degree"].__setitem__(0, 5)),
        ]:
            changed = {k: v.copy() for k, v in arrays.items()}
            mutate(changed)
            bad = dict(record, path="bad_" + label + ".npz", arrays=bank.array_contract(changed))
            bank.write_npz(root / bad["path"], changed)
            bad["sha256"] = bank.sha256(root / bad["path"])
            rejects(label, lambda: bank.load_record(root, bad, kernel))
        bank.write_npz(root / "object_reject_control.npz", {"numeric": np.zeros(1)})
        rejects("object_array_writer", lambda: bank.write_npz(root / "bad_object.npz", {"x": np.asarray([{}], dtype=object)}))
        for label, fake in [
            ("incomplete_bank", {"schema": "rsjg-canonical-a-bank-v1", "status": "INCOMPLETE"}),
            ("wrong_route_D", {"schema": "rsjg-canonical-a-bank-v1", "status": "COMPLETE",
                              "protocol": dict(bank.PROTOCOL, route="D")}),
            ("wrong_checkpoint", {"schema": "rsjg-canonical-a-bank-v1", "status": "COMPLETE",
                "protocol": bank.PROTOCOL, "provenance": {"checkpoint_sha256": "0" * 64}}),
        ]:
            path = root / (label + ".json")
            bank.write_json(path, fake)
            rejects(label, lambda: bank.certify_bank(path, bank.sha256(path)))
        path = root / "wrong_checkpoint.json"
        rejects("manifest_digest", lambda: bank.certify_bank(path, "0" * 64))
        busy = {"devices": [["0", "GPU-test", "fixture", "16000", "9000", "6"]],
                "compute_processes": [["GPU-test", "962176", "SDD", "8192"]]}
        rejects("busy_gpu_gate", lambda: exporter.require_idle(busy, "GPU-test"), "BLOCKED_RESOURCE_UNAVAILABLE")
        with patch.object(exporter, "gpu_snapshot", return_value=busy), patch.object(exporter, "preflight") as forbidden:
            rejects("busy_export_before_setup", lambda: exporter.export(
                SimpleNamespace(gpu_uuid="GPU-test", output=str(root / "must_not_exist"))), "BLOCKED_RESOURCE_UNAVAILABLE")
            forbidden.assert_not_called()
            assert not (root / "must_not_exist").exists()
        desktop = {"devices": [["0", "GPU-test", "fixture", "16000", "800", "28"]],
                   "compute_processes": [],
                   "process_types": [{"gpu_uuid": "GPU-test", "pid": "1351", "type": "G", "executable": "Xorg"}]}
        exporter.require_idle(desktop, "GPU-test", allow_desktop_graphics=True)
        rejects("desktop_not_implicitly_authorized", lambda: exporter.require_idle(desktop, "GPU-test"))
        rejects("desktop_optin_cannot_allow_compute", lambda: exporter.require_idle(
            dict(desktop, compute_processes=busy["compute_processes"]), "GPU-test", allow_desktop_graphics=True))
        for process_type, executable in [("C+G", "code"), ("C", "python"), ("G", "unknown-workload")]:
            bad = dict(desktop, process_types=[{"gpu_uuid": "GPU-test", "pid": "999",
                       "type": process_type, "executable": executable}])
            rejects("desktop_reject_" + process_type + "_" + executable,
                    lambda: exporter.require_idle(bad, "GPU-test", allow_desktop_graphics=True))
        rejects("desktop_no_type_evidence", lambda: exporter.require_idle(
            {k: v for k, v in desktop.items() if k != "process_types"},
            "GPU-test", allow_desktop_graphics=True))
        xml = "<nvidia_smi_log><gpu><uuid>GPU-test</uuid><processes><process_info><pid>1</pid><type>G</type><process_name>/usr/bin/code --secret=not-for-archive</process_name><used_memory>20 MiB</used_memory></process_info></processes></gpu></nvidia_smi_log>"
        assert exporter.process_types(xml)[0]["executable"] == "code"
        assert "not-for-archive" not in str(exporter.process_types(xml))
        assert not torch.cuda.is_initialized()
    print(json.dumps({"status": "PASS", "scope": "SYNTHETIC_CPU_ONLY", "fixtures": results,
                      "negative_tests": rejected, "negative_test_count": len(rejected),
                      "capture_loader_CPU_parity_max_abs_error": max(max_parity),
                      "capture_and_serialization_CPU_rng_unchanged": True,
                      "BF16_storage_exact_roundtrip": True, "weighted_aggregation": "PASS",
                      "detailed_adapter_matches_archived_kernel": True,
                      "master_seeds": bank.MASTERS, "atol": 1e-6, "rtol": 1e-6,
                      "CUDA_initialized": False, "model_constructed": False,
                      "checkpoint_loaded": False, "real_A_scene_count": 0,
                      "actual_full_bank_certification": "NOT_RUN_NO_BANK",
                      "CUDA_rng_capture_invariant": "IMPLEMENTED_NOT_RUN",
                      "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__}},
                     indent=2, allow_nan=False))

if __name__ == "__main__":
    main()

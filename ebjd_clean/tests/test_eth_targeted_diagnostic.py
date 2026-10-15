import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "eth_targeted_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("eth_targeted_diagnostic", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_cv_uses_mean_of_two_recent_velocities():
    observed = np.zeros((1, 8, 2), dtype=np.float64)
    observed[0, -3:, 0] = (0.0, 0.4, 1.2)
    baseline, velocity = MODULE.cv_prediction(observed)
    assert np.allclose(velocity[0], (1.5, 0.0))
    assert np.allclose(baseline[0, 0], (1.8, 0.0))
    assert np.allclose(baseline[0, -1], (8.4, 0.0))


def test_endpoint_bridge_unscaled_roundtrip_components():
    observed = np.zeros((2, 8, 2), dtype=np.float64)
    baseline, _ = MODULE.cv_prediction(observed)
    fraction = np.arange(1, 13, dtype=np.float64)[None, :, None] / 12
    goal = np.asarray([[[1.2, -0.6]], [[-0.3, 0.9]]])
    future = baseline + fraction * goal
    encoded_goal, bridge = MODULE.unscaled_latent(future, baseline)
    assert np.allclose(encoded_goal, goal[:, 0])
    assert np.max(np.abs(bridge)) < 1e-12


def test_metric_group_deterministic_joint_and_marginal():
    future = np.zeros((2, 12, 2), dtype=np.float32)
    prediction = np.zeros((1, 2, 12, 2), dtype=np.float32)
    prediction[0, 0, :, 0] = 1.0
    prediction[0, 1, :, 0] = 3.0
    group = MODULE.MetricGroup()
    group.add(prediction, future)
    result = group.result()
    assert result["minADE"] == 2.0
    assert result["JADE"] == 2.0
    assert result["coordination_gap_ADE"] == 0.0


def test_k_curve_is_monotone_for_nested_prefixes(tmp_path):
    from ebjd.diagnostic_bank import BankScene

    prediction = np.zeros((20, 1, 12, 2), dtype=np.float32)
    prediction[:, 0, :, 0] = np.arange(20, 0, -1)[:, None]
    future = np.zeros((1, 12, 2), dtype=np.float32)
    scene = BankScene(
        prediction, future, np.zeros((1, 8, 2), dtype=np.float32),
        np.arange(20), np.asarray(["1"]), "toy", "toy-scene",
        "toy:0-19", 0.4)
    rows, violation = MODULE.k_curve({"toy:0-19": scene}, "toy")
    assert violation <= 1e-12
    assert rows[-1]["minFDE"] <= rows[0]["minFDE"]


def test_bank_analysis_reconstructs_group_contributions(tmp_path):
    from ebjd.diagnostic_bank import (
        BANK_SCHEMA, SEED_MANIFEST_SCHEMA, BankScene, save_scene,
        summarize_scenes,
    )

    manifests = {}
    for model, offset in (("GDTS", 0.0), ("EBJD_epoch005", 1.0)):
        root = tmp_path / model
        files, scenes = [], []
        for scene_index, agents in enumerate((1, 2)):
            source = f"toy{scene_index}"
            ids = np.asarray([str(value) for value in range(agents)])
            prediction = np.zeros((20, agents, 12, 2), dtype=np.float32)
            prediction[..., 0] = offset
            scene = BankScene(
                prediction, np.zeros((agents, 12, 2), dtype=np.float32),
                np.zeros((agents, 8, 2), dtype=np.float32), np.arange(20),
                ids, source, f"scene-{scene_index}", f"{source}:0-19", 0.4)
            relative = Path("scenes") / f"scene_{scene_index:06d}.npz"
            digest = save_scene(scene, root / relative)
            files.append({
                "path": relative.as_posix(), "sha256": digest,
                "identity": scene.window_key, "valid_agents": agents,
            })
            scenes.append(scene)
        payload = {
            "schema": SEED_MANIFEST_SCHEMA, "bank_schema": BANK_SCHEMA,
            "status": "COMPLETED", "scene_count": 2,
            "replay_metrics": summarize_scenes(scenes), "files": files,
        }
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        manifests[model] = manifest
    output = tmp_path / "analysis"
    MODULE.run_banks(SimpleNamespace(
        bank=[f"{name}={path}" for name, path in manifests.items()],
        goal_scale=1.0, output_dir=str(output)))
    result = json.loads((output / "BANK_RESULTS.json").read_text())
    assert result["status"] == "FULL_BANK_ANALYSIS_COMPLETE"
    reconstruction = result["gap_contributions"]["EBJD_epoch005"]["reconstruction"]
    assert max(abs(value["residual_m"]) for value in reconstruction.values()) < 1e-12
    assert max(result["k_monotonic_max_positive_m"].values()) <= 1e-12
    assert result["case_selection"]

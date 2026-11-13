"""Release extraction must preserve training mathematics and be standalone."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "network_release/ebjd_gated_bridge_v1"


def functions(path):
    return {node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}


def isolated(directory, code, *args):
    # Python isolated mode ignores inherited PYTHONPATH and the paper cwd.
    return subprocess.run(
        [sys.executable, "-I", "-B", "-c",
         "import sys; sys.path.insert(0, sys.argv.pop(1)); " + code,
         str(directory), *map(str, args)],
        cwd=directory, env={**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"},
        text=True, capture_output=True, check=True)


def test_release_manifest_covers_clean_regular_sources():
    manifest = json.loads((RELEASE / "SOURCE_MANIFEST.json").read_text())
    for relative, entry in manifest["files"].items():
        path = RELEASE / relative
        assert path.is_file() and not path.is_symlink()
        assert path.suffix not in {".pt", ".pth", ".npz"}
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        if entry["transformation"] == "byte_copy":
            original = ROOT / entry["sources"][0]["path"]
            assert path.read_bytes() == original.read_bytes()


def test_loader_scale_fit_and_training_functions_are_unchanged():
    runtime = functions(RELEASE / "ebjd/runtime.py")
    assert runtime["loader"] == functions(ROOT / "scripts/run_zara1_native_prior_pilot.py")["loader"]
    assert runtime["fit_training_scales"] == functions(ROOT / "ebjd/train.py")["fit_training_scales"]
    original = functions(ROOT / "scripts/run_zara1_unified_coupling_bridge.py")
    exported = functions(RELEASE / "train.py")
    assert original.keys() == exported.keys()
    for name in original.keys() - {"main"}:
        assert original[name] == exported[name], name
    # The sole main() change is adding the release digest to run identity.
    tree = ast.parse((RELEASE / "train.py").read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    removed = 0
    for node in ast.walk(main):
        if isinstance(node, ast.Dict):
            for index in range(len(node.keys) - 1, -1, -1):
                if isinstance(node.keys[index], ast.Constant) and node.keys[index].value == "release_manifest_sha256":
                    node.keys.pop(index)
                    node.values.pop(index)
                    removed += 1
    assert removed == 1
    assert ast.dump(main, include_attributes=False) == original["main"]


def test_relocated_snapshot_imports_and_rejects_tampering(tmp_path):
    destination = tmp_path / "standalone"
    subprocess.run([sys.executable, "-B", str(ROOT / "scripts/export_clean_network.py"),
                    "--output", str(destination)], check=True, capture_output=True, text=True)
    result = isolated(destination,
        "from release_guard import verify_release; verify_release(); "
        "from train import bridge_and_pair_loss; from network import build_network; "
        "import ebjd, pathlib; "
        "assert pathlib.Path(ebjd.__file__).resolve().is_relative_to(pathlib.Path.cwd()); "
        "m=build_network('gated8'); print(sum(p.numel() for p in m.parameters()))")
    assert "11768797" in result.stdout
    (destination / "ebjd/model.py").write_text("# tampered\n")
    with pytest.raises(subprocess.CalledProcessError):
        isolated(destination, "from release_guard import verify_release; verify_release()")


def test_export_refuses_to_overwrite_independently_modified_model(tmp_path):
    destination = tmp_path / "release"
    command = [sys.executable, "-B", str(ROOT / "scripts/export_clean_network.py"),
               "--output", str(destination)]
    subprocess.run(command, check=True, capture_output=True)
    path = destination / "ebjd/model.py"
    path.write_text("# preserve this independent edit\n")
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(command, check=True, capture_output=True)
    assert path.read_text() == "# preserve this independent edit\n"


def test_prepared_json_roundtrip_and_rollout_guard(tmp_path):
    from conftest import load_paper_config
    config = load_paper_config()
    config["training"].update(
        epochs=30, new_module_lr=1e-4, unet_lr=1e-5, lr_warmup_epochs=5,
        lr_final_fraction=0.1, optimizer="adamw", rollout_training_enabled=False,
        paired_marginal_actual_step_projection_enabled=False,
        coupling_singleton_backward="legacy",
        early_stopping={"monitor": "H_JADE", "max_epochs": 30})
    config["denoiser"]["ctf_feedback_basis"] = "legacy"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    code = ("from ebjd.config import load_config; from release_guard import check_contract; "
            "c=load_config(sys.argv[1]); assert isinstance(c['training']['unet_lr'],float); "
            "check_contract(c)")
    isolated(RELEASE, code, path)
    config["training"]["rollout_training_enabled"] = True
    path.write_text(json.dumps(config))
    with pytest.raises(subprocess.CalledProcessError):
        isolated(RELEASE, code, path)

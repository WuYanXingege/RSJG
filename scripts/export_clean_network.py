#!/usr/bin/env python3
"""Export an auditable, standalone unified-network snapshot (no assets).

Model files are copied byte-for-byte. Only historical CLI dependencies and
the obsolete optimizer configuration label are removed from the release.
The canonical paper workspace remains the development source of truth.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "network_release/ebjd_gated_bridge_v1"
MODULES = (
    "__init__", "model", "encoders", "denoiser", "gated_dit", "layers",
    "representation", "unified", "objectives", "sampling", "metrics",
    "data", "proposals", "early_stopping", "goal_provider",
)
SUPPORT = (
    "README.md", "ARCHITECTURE_REVIEW_ZH.md", "METHOD.md", "DEPENDENCIES.md",
    "network.py", "release_guard.py", "prepare.py", "check_network.py",
    "requirements.txt", "pyproject.toml", ".gitignore", "configs/profiles.yaml",
)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def definition(source: str, name: str) -> str:
    node = next(n for n in ast.parse(source).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
    return "".join(source.splitlines(keepends=True)[start:node.end_lineno])


def remove_definition(source: str, name: str) -> str:
    body = definition(source, name)
    if source.count(body) != 1:
        raise ValueError(f"ambiguous definition: {name}")
    return source.replace(body, "")


def export(output: Path) -> dict:
    output = output.resolve()
    if output.is_symlink() or output == ROOT or ROOT.is_relative_to(output):
        raise ValueError("invalid export destination")
    output.mkdir(parents=True, exist_ok=True)
    previous_path = output / "SOURCE_MANIFEST.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    old_files = previous.get("files", {})
    files = {}
    source_commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()

    def write(relative: str, data: bytes, sources: list[str], transform="byte_copy"):
        target = output / relative
        if target.is_symlink():
            raise ValueError(f"refusing symlink destination: {target}")
        if target.exists() and target.read_bytes() != data:
            old = old_files.get(relative)
            if not old or digest(target.read_bytes()) != old["sha256"]:
                raise ValueError(f"refusing to overwrite independently modified file: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files[relative] = {
            "sha256": digest(data), "bytes": len(data), "transformation": transform,
            "sources": [{"path": s, "sha256": digest((ROOT / s).read_bytes())}
                        for s in sources],
        }

    for relative in SUPPORT:
        source = DEFAULT / relative
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"missing regular release support file: {source}")
        write(relative, source.read_bytes(), [str(source.relative_to(ROOT))],
              "release_interface_or_documentation")
    for name in MODULES:
        relative = f"ebjd/{name}.py"
        write(relative, (ROOT / relative).read_bytes(), [relative])
    for relative in ("ebjd/vendor/__init__.py", "ebjd/vendor/unet.py",
                     "ebjd/vendor/NOTICE.md", "LICENSE"):
        write(relative, (ROOT / relative).read_bytes(), [relative])

    source_config = (ROOT / "ebjd/config.py").read_text()
    config = remove_definition(source_config, "trainer_options")
    config = config.replace("from .trainer import TrainerOptions\n", "")
    old_label = '("training", "optimizer", "adamw_actual_step_projection")'
    if config.count(old_label) != 1:
        raise ValueError("optimizer config contract has changed; review export")
    config = config.replace(old_label, '("training", "optimizer", "adamw")')
    yaml_loader = "payload = yaml.safe_load(handle)"
    if config.count(yaml_loader) != 1:
        raise ValueError("configuration loader has changed; review export")
    config = config.replace(yaml_loader,
        'payload = json.load(handle) if Path(path).suffix.lower() == ".json" else yaml.safe_load(handle)')
    write("ebjd/config.py", config.encode(), ["ebjd/config.py"],
          "remove_unused_legacy_trainer_factory; label_effective_optimizer_adamw; parse_json_as_json")

    pilot_path = "scripts/run_zara1_native_prior_pilot.py"
    scales_path = "ebjd/train.py"
    runtime = '''"""Unchanged loader and train-only RMS fitting from the paper runner."""
from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from .data import SceneManifestDataset, collate_scenes, collate_scenes_compact_maps
from .representation import EndpointBridgeRepresentation, cv_baseline

'''
    runtime += definition((ROOT / pilot_path).read_text(), "loader") + "\n\n"
    runtime += definition((ROOT / scales_path).read_text(), "fit_training_scales") + "\n"
    write("ebjd/runtime.py", runtime.encode(), [pilot_path, scales_path],
          "exact_function_extraction:loader,fit_training_scales")

    runner_path = "scripts/run_zara1_unified_coupling_bridge.py"
    runner = (ROOT / runner_path).read_text()
    replacements = {
        "from ebjd.train import fit_training_scales": "from ebjd.runtime import fit_training_scales",
        "from run_zara1_native_prior_pilot import loader":
            "from ebjd.runtime import loader\nfrom release_guard import manifest_digest, before_training",
        '        "source_commit": args.source_commit,':
            '        "source_commit": args.source_commit,\n        "release_manifest_sha256": manifest_digest(),',
        'if __name__ == "__main__":\n    main()':
            'if __name__ == "__main__":\n    before_training()\n    main()',
    }
    for old, new in replacements.items():
        if runner.count(old) != 1:
            raise ValueError(f"runner export location changed: {old}")
        runner = runner.replace(old, new)
    write("train.py", runner.encode(), [runner_path],
          "generic_filename; local_helpers; release_identity_and_startup_guard")

    manifest = {
        "schema": "ebjd-clean-network-release-v1", "upstream_commit": source_commit,
        "status": "RESEARCH_IMPLEMENTATION_NOT_ACCURACY_VALIDATED",
        "default_profile": "gated8", "formal_result_profile": "legacy6",
        "files": files, "file_count": len(files),
        "total_bytes": sum(entry["bytes"] for entry in files.values()),
        "external_assets_included": False, "symlinks_included": False,
        "scope": "unified model, frozen provider API, existing training/evaluation path",
        "excluded": ["old workspaces", "legacy GDTS trainer", "old unconditional trainer",
                     "optimizer projection", "pipelines", "cleanup directories",
                     "datasets", "checkpoints", "banks", "historical large logs"],
    }
    temporary = previous_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(previous_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    result = export(args.output)
    print(json.dumps({"output": str(args.output.resolve()),
                      "files": result["file_count"], "bytes": result["total_bytes"]}))

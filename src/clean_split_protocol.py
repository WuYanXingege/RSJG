"""Fail-closed clean-split and final-test provenance contracts."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

from src.joint_dependency_v2_cache import sha256_file, stable_json_hash


CLEAN_SPLIT_SCHEMA = "jdv2-clean-split-v1"
FINAL_TEST_LOCK_SCHEMA = "jdv2-clean-final-test-lock-v1"
CLEAN_TEST_SEEDS = [2035, 2036, 2037, 2038, 2039]
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    return candidate.resolve()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open() as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid clean-protocol JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"clean-protocol JSON must be a mapping: {path}")
    return value


def _embedded_hash(value: Mapping[str, Any], field: str) -> str:
    payload = {key: item for key, item in value.items() if key != field}
    return stable_json_hash(payload)


def load_clean_split_manifest(
    path: str | os.PathLike[str], expected_hash: str,
) -> dict[str, Any]:
    """Load and authenticate the immutable logical split manifest."""
    manifest_path = _resolve(path)
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != CLEAN_SPLIT_SCHEMA:
        raise RuntimeError("unsupported clean split manifest schema")
    actual_hash = _embedded_hash(manifest, "manifest_hash")
    if manifest.get("manifest_hash") != actual_hash:
        raise RuntimeError("clean split manifest embedded hash mismatch")
    if expected_hash != actual_hash:
        raise RuntimeError("clean split manifest configured hash mismatch")
    expected_roles = {"train", "internal_valid", "final_test"}
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or set(splits) != expected_roles:
        raise RuntimeError("clean split manifest has invalid logical roles")
    for role, expected_count in (
            ("train", 3790), ("internal_valid", 320),
            ("final_test", 139)):
        split = splits[role]
        records = split.get("records")
        if (not isinstance(records, list) or
                split.get("window_count") != len(records) or
                len(records) != expected_count):
            raise RuntimeError(
                f"clean split manifest count mismatch for {role}")
        indices = [int(record["physical_cache_index"])
                   for record in records]
        if indices != sorted(indices) or len(indices) != len(set(indices)):
            raise RuntimeError(
                f"clean split manifest indices invalid for {role}")
    expected_indices = {
        'train': list(range(3790)),
        'internal_valid': list(range(3790, 4110)),
        'final_test': list(range(139)),
    }
    expected_physical = {
        'train': 'train', 'internal_valid': 'train',
        'final_test': 'test',
    }
    fingerprints = {}
    for role in expected_roles:
        records = splits[role]['records']
        indices = [int(record['physical_cache_index'])
                   for record in records]
        if indices != expected_indices[role]:
            raise RuntimeError(
                f'clean split manifest membership mismatch for {role}')
        if any(record.get('physical_cache_split') !=
               expected_physical[role] for record in records):
            raise RuntimeError(
                f'clean split manifest physical role mismatch for {role}')
        for field in ('id_fingerprint_sha256',
                      'trajectory_content_sha256'):
            values = [record.get(field) for record in records]
            if any(re.fullmatch(r'[0-9a-f]{64}', str(value or ''))
                   is None for value in values):
                raise RuntimeError('invalid clean split fingerprint')
            if len(values) != len(set(values)):
                raise RuntimeError(
                    f'duplicate clean split fingerprint in {role}')
            fingerprints[(role, field)] = set(values)
    for field in ('id_fingerprint_sha256',
                  'trajectory_content_sha256'):
        for left, right in (('train', 'internal_valid'),
                            ('train', 'final_test'),
                            ('internal_valid', 'final_test')):
            if fingerprints[(left, field)] & fingerprints[(right, field)]:
                raise RuntimeError('clean split fingerprint overlap')
    return manifest


def clean_split_records(args, logical_role: str) -> list[dict[str, Any]]:
    manifest = load_clean_split_manifest(
        args.clean_split_manifest_path, args.clean_split_manifest_hash)
    return list(manifest["splits"][logical_role]["records"])


def _current_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
            stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot establish source commit for final test") \
            from exc


def validate_final_test_lock(args) -> dict[str, Any]:
    """Verify the one-way lock required before any clean final-test access."""
    if not getattr(args, "clean_split_protocol", False):
        raise RuntimeError("final-test lock is only valid for clean protocol")
    if getattr(args, "final_test_access", "legacy") != "authorized":
        raise RuntimeError("clean final test is locked")
    lock_path = getattr(args, "final_test_lock_path", None)
    if not lock_path:
        raise RuntimeError("clean final test requires a lock artifact")
    lock = _load_json(_resolve(lock_path))
    if lock.get("schema") != FINAL_TEST_LOCK_SCHEMA:
        raise RuntimeError("unsupported clean final-test lock schema")
    if "lock_hash" not in lock:
        raise RuntimeError("clean final-test lock hash is missing")
    actual_lock_hash = _embedded_hash(lock, "lock_hash")
    if lock["lock_hash"] != actual_lock_hash:
        raise RuntimeError("clean final-test lock embedded hash mismatch")
    if lock.get("clean_split_manifest_hash") != \
            args.clean_split_manifest_hash:
        raise RuntimeError("final-test lock split manifest mismatch")
    # Re-authenticate the split manifest rather than trusting only the lock.
    load_clean_split_manifest(
        args.clean_split_manifest_path, args.clean_split_manifest_hash)
    if lock.get("source_commit") != _current_commit():
        raise RuntimeError("final-test lock source commit mismatch")
    if lock.get("test_seeds") != CLEAN_TEST_SEEDS:
        raise RuntimeError("final-test lock seed protocol mismatch")

    configs = lock.get("configs")
    checkpoints = lock.get("checkpoints")
    if not isinstance(configs, dict) or set(configs) != {"stage_a", "stage_b"}:
        raise RuntimeError("final-test lock requires both canonical configs")
    if (not isinstance(checkpoints, dict) or
            set(checkpoints) != {"stage_a", "stage_b"}):
        raise RuntimeError("final-test lock requires both selected checkpoints")
    for family, entries in (("config", configs),
                            ("checkpoint", checkpoints)):
        for stage, entry in entries.items():
            if not isinstance(entry, dict):
                raise RuntimeError(f"invalid {stage} {family} lock entry")
            path = _resolve(entry.get("path", ""))
            digest = entry.get("sha256")
            if not path.is_file() or not re.fullmatch(
                    r"[0-9a-f]{64}", str(digest or "")):
                raise RuntimeError(f"missing locked {stage} {family}")
            if sha256_file(path) != digest:
                raise RuntimeError(f"locked {stage} {family} hash mismatch")
            if family == "checkpoint":
                if int(entry.get("selection_epoch", 0)) < 1:
                    raise RuntimeError("invalid locked selection epoch")
                if not str(entry.get("selection_metric", "")):
                    raise RuntimeError("missing locked selection metric")

    target = getattr(args, "clean_evaluation_target", None)
    if target not in {"stage_a", "stage_b"}:
        raise RuntimeError("clean final test requires an evaluation target")
    expected_checkpoint = _resolve(
        Path(args.model_dir) / "saved_models" / "best_model.pt")
    locked_checkpoint = _resolve(checkpoints[target]["path"])
    if expected_checkpoint != locked_checkpoint:
        raise RuntimeError("evaluation target checkpoint differs from lock")
    return lock


__all__ = [
    "CLEAN_SPLIT_SCHEMA",
    "CLEAN_TEST_SEEDS",
    "FINAL_TEST_LOCK_SCHEMA",
    "clean_split_records",
    "load_clean_split_manifest",
    "validate_final_test_lock",
]

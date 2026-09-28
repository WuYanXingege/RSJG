#!/usr/bin/env python3
"""Prepare, export, and score GDTS on the official JMM ETH protocol.

The commands are deliberately separated.  ``infer`` only exports trajectories;
it never computes or reports metrics.  ``score`` is an explicit later action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.jmm_protocol import (  # noqa: E402
    JMM_ETH_EXPECTED_AGENT_INSTANCES,
    JMM_ETH_EXPECTED_MEAN_PEDS,
    JMM_ETH_EXPECTED_WINDOWS,
    JMM_ETH_SOURCE_SHA256,
    JMM_ETH_SOURCE_URL,
    JMM_NUM_SAMPLES,
    JMM_OBS_LEN,
    atomic_write_json,
    build_jmm_eth_windows,
    download_official_jmm_eth,
    score_standardized_trajectories,
    sha256_file,
    standardized_window_is_complete,
    summarize_windows,
    write_standardized_window,
)
from src.joint_dependency_v2_cache import (  # noqa: E402
    atomic_json_save,
    atomic_torch_save,
    load_cache_record,
)


DEFAULT_DATA_FILE = (
    PROJECT_ROOT / "data" / "jmm_official" / "eth" /
    "biwi_eth_agentformer.txt"
)
JMM_JDV2_CACHE_SCHEMA = "jmm-jdv2-deployment-cache-v1"
JMM_JDV2_CACHE_MANIFEST = "manifest.json"


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def _source_provenance() -> dict:
    """Record, but do not silently substitute for, artifact content hashes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--short", "--untracked-files=no"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
        dirty = None
    return {"source_commit": commit, "source_tracked_dirty": dirty}


def command_prepare_data(args: argparse.Namespace) -> None:
    destination = download_official_jmm_eth(args.data_file)
    windows = build_jmm_eth_windows(destination, strict_official=True)
    payload = summarize_windows(windows)
    payload.update({
        "data_file": str(destination.resolve()),
        "sha256": sha256_file(destination),
        "source_url": JMM_ETH_SOURCE_URL,
        "status": "ready",
    })
    _print_json(payload)


def command_inspect(args: argparse.Namespace) -> None:
    windows = build_jmm_eth_windows(
        args.data_file, strict_official=not args.allow_nonstandard_data
    )
    payload = summarize_windows(windows)
    payload.update({
        "data_file": str(Path(args.data_file).resolve()),
        "sha256": sha256_file(args.data_file),
        "strict_official": not args.allow_nonstandard_data,
    })
    _print_json(payload)


def _resolve_checkpoint(run_dir: Path, value: str) -> Path:
    direct = Path(value).expanduser()
    candidates = []
    if direct.is_absolute():
        candidates.append(direct)
    else:
        candidates.extend([Path.cwd() / direct, run_dir / direct])
        if value == "best":
            candidates.append(run_dir / "saved_models" / "best_model.pt")
        elif value.isdigit():
            candidates.append(
                run_dir / "saved_models" / f"epoch_{int(value):03d}.pt"
            )
        else:
            candidates.append(run_dir / "saved_models" / value)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Checkpoint not found. Searched:\n{searched}")


def _load_run_arguments(run_dir: Path, device_name: str):
    import torch
    import yaml

    from src.parser import get_parser, check_and_add_additional_args

    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Run config is not a mapping: {config_path}")

    parser = get_parser()
    parser.set_defaults(**config)
    model_args = parser.parse_args([])
    # These are evaluation invariants, not new training hyperparameters.
    model_args.phase = "test"
    model_args.device = device_name
    model_args.num_samples = JMM_NUM_SAMPLES
    model_args.num_joint_samples = None
    model_args.use_wandb = False
    model_args.data_augmentation = False
    model_args.load_checkpoint = None
    model_args.pretrain_path = None
    model_args.force_reprocess = False
    model_args = check_and_add_additional_args(model_args)
    model_args.model_dir = str(run_dir.resolve())
    model_args.save_dir = model_args.model_dir
    model_args.config = str(config_path.resolve())
    if model_args.dataset != "eth5" or model_args.test_set != "eth":
        raise ValueError(
            "Official ETH evaluation requires a run configured with "
            f"dataset=eth5/test_set=eth, got {model_args.dataset}/"
            f"{model_args.test_set}"
        )
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested {device_name}, but CUDA is unavailable in this process"
        )
    return model_args


def _resolve_source_checkpoint(model_args) -> Path:
    value = model_args.jdv2_source_checkpoint or model_args.pretrain_path
    if value is None:
        raise ValueError(
            "Active JDV2 JMM evaluation requires its frozen source "
            "Goal U-Net checkpoint"
        )
    checkpoint = Path(value).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = (PROJECT_ROOT / checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Frozen JDV2 source checkpoint not found: {checkpoint}"
        )
    return checkpoint


def _record_set_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def _jdv2_cache_identity(
    model_args,
    data_file: Path,
    source_checkpoint: Path,
    seed: int,
) -> dict:
    return {
        "schema_version": JMM_JDV2_CACHE_SCHEMA,
        "protocol": "Joint Metrics Matter ETH official",
        "coordinate_units": {
            "position": "world_m",
            "velocity": "world_m_per_s",
        },
        "contains_future_supervision": False,
        "teacher_sidecars": False,
        "data_file": str(data_file.resolve()),
        "data_sha256": sha256_file(data_file),
        "num_scenes": JMM_ETH_EXPECTED_WINDOWS,
        "num_agent_instances": JMM_ETH_EXPECTED_AGENT_INSTANCES,
        "goal_checkpoint": str(source_checkpoint.resolve()),
        "goal_checkpoint_sha256": sha256_file(source_checkpoint),
        "candidate_order": "generate_goal_candidates-v1",
        "candidate_seed_protocol": "seed_plus_scene_index",
        "candidate_seed": int(seed),
        "K": int(model_args.num_goal_candidates),
        "dt": float(model_args.trajectory_dt),
        "graph_config": {
            "type": model_args.graph_type,
            "radius": float(model_args.graph_radius),
            "ttc_threshold": float(model_args.ttc_threshold),
            "adaptive": False,
        },
    }


def _cache_record_path(cache_dir: Path, scene_index: int) -> Path:
    return cache_dir / "records" / f"{scene_index:06d}.pt"


def _collate_cache_record(record: dict) -> dict:
    """Reproduce the leading batch dimension added by a DataLoader."""
    import torch

    return {
        key: value.unsqueeze(0) if torch.is_tensor(value) else value
        for key, value in record.items()
    }


def _load_validated_jdv2_cache(
    cache_dir: Path,
    windows,
    expected_identity: dict,
) -> dict:
    manifest_path = cache_dir / JMM_JDV2_CACHE_MANIFEST
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Missing JMM JDV2 cache manifest: {manifest_path}; run "
            "build-jdv2-cache first"
        )
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected_identity.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"JMM JDV2 cache manifest mismatch: {mismatches}")
    record_paths = sorted((cache_dir / "records").glob("*.pt"))
    if len(record_paths) != len(windows):
        raise RuntimeError(
            f"JMM JDV2 cache expected {len(windows)} records, "
            f"found {len(record_paths)}"
        )
    if manifest.get("record_set_sha256") != _record_set_hash(record_paths):
        raise RuntimeError("JMM JDV2 cache record-set hash mismatch")
    if manifest.get("status") != "complete":
        raise RuntimeError("JMM JDV2 cache manifest is not complete")
    return manifest


def _validate_jdv2_cache_record(record: dict, window, scene_index: int) -> None:
    import torch

    expected_id = f"jmm-eth-{scene_index:06d}"
    if record.get("cache_id") != expected_id:
        raise RuntimeError(
            f"JMM JDV2 cache id mismatch: expected {expected_id}, "
            f"got {record.get('cache_id')}"
        )
    if record.get("contains_future_supervision") is not False:
        raise RuntimeError("JMM JDV2 deployment cache contains future supervision")
    if record.get("window_directory") != window.directory_name:
        raise RuntimeError("JMM JDV2 cache/window directory mismatch")
    expected_frames = torch.from_numpy(
        np.repeat(window.frame_ids[:, None], window.num_agents, axis=1)
    ).long()
    tensor_equalities = {
        "frame_ids": expected_frames,
        "scene_index": torch.zeros(window.num_agents, dtype=torch.long),
        "scene_ptr": torch.tensor([0, window.num_agents], dtype=torch.long),
    }
    for name, expected in tensor_equalities.items():
        value = record.get(name)
        if not torch.is_tensor(value) or not torch.equal(value.long(), expected):
            raise RuntimeError(f"JMM JDV2 cache {name} mismatch")
    candidate_count = int(record.get("num_goal_candidates", -1))
    candidate_shape = (window.num_agents, candidate_count)
    if candidate_count <= 0:
        raise RuntimeError("JMM JDV2 cache has invalid candidate count")
    expected_shapes = {
        "goal_candidates_world": (*candidate_shape, 2),
        "goal_candidates_map": (*candidate_shape, 2),
        "candidate_log_prior": candidate_shape,
    }
    for name, expected_shape in expected_shapes.items():
        value = record.get(name)
        if not torch.is_tensor(value) or tuple(value.shape) != expected_shape:
            raise RuntimeError(
                f"JMM JDV2 cache {name} shape mismatch: expected "
                f"{expected_shape}, got {getattr(value, 'shape', None)}"
            )
    edge_index = record.get("edge_index")
    edge_feat = record.get("edge_feat")
    edge_weight = record.get("edge_weight")
    if not torch.is_tensor(edge_index) or edge_index.ndim != 2 or \
            edge_index.shape[0] != 2:
        raise RuntimeError("JMM JDV2 edge_index must have shape [2,E]")
    edge_count = edge_index.shape[1]
    if not torch.is_tensor(edge_feat) or edge_feat.shape[0] != edge_count:
        raise RuntimeError("JMM JDV2 edge feature count mismatch")
    if not torch.is_tensor(edge_weight) or edge_weight.shape != (edge_count,):
        raise RuntimeError("JMM JDV2 edge weight count mismatch")
    if edge_index.numel() and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= window.num_agents
    ):
        raise RuntimeError("JMM JDV2 cache contains an invalid agent edge")


def _build_collated_model_batch(window, model_args, scene, tensor_image):
    import torch

    from src.data_grouping import SCENE_BATCH_FORMAT_VERSION
    from src.models.model_utils.cnn_big_images_utils import (
        create_CNN_inputs_loop,
    )

    world = torch.from_numpy(window.world_coordinates).float()
    full_pixel = scene.make_pixel_coord_torch(world)
    model_pixel = full_pixel / float(model_args.down_factor)
    trajectory_maps = create_CNN_inputs_loop(
        batch_abs_pixel_coords=model_pixel,
        tensor_image=tensor_image,
    )
    num_agents = window.num_agents
    frame_ids = torch.from_numpy(
        np.repeat(window.frame_ids[:, None], num_agents, axis=1)
    ).long()
    # Reproduce the leading batch_size=1 dimension added by GDTS's DataLoader.
    data = {
        "abs_pixel_coord": model_pixel.unsqueeze(0),
        "seq_list": torch.ones(1, model_args.seq_length, num_agents),
        "scene_index": torch.zeros(1, num_agents, dtype=torch.long),
        "scene_ptr": torch.tensor([[0, num_agents]], dtype=torch.long),
        "frame_ids": frame_ids.unsqueeze(0),
        "batch_format_version": torch.tensor(
            [SCENE_BATCH_FORMAT_VERSION], dtype=torch.long
        ),
        "tensor_image": tensor_image.unsqueeze(0),
        "input_traj_maps": trajectory_maps.unsqueeze(0),
    }
    batch_id = {
        "scene_name": ["eth"],
        "starting_frames": [int(window.frame_ids[0])],
        "agent_ids": [window.agent_ids.tolist()],
        "synchronized_window": [True],
    }
    return data, batch_id


def _predict_window(
    model, model_args, scene, tensor_image, window, cache_record=None
) -> np.ndarray:
    import torch

    batch_data, batch_id = _build_collated_model_batch(
        window, model_args, scene, tensor_image
    )
    if cache_record is not None:
        batch_data["jdv2_cache"] = _collate_cache_record(cache_record)
    inputs, _ = model.prepare_inputs(batch_data, batch_id)
    # Guard the world/pixel bridge: an accidental scale mismatch invalidates
    # every metric while still producing plausible-looking trajectories.
    expected_world = torch.from_numpy(window.world_coordinates).to(
        device=inputs["world_coord"].device,
        dtype=inputs["world_coord"].dtype,
    )
    roundtrip_error = (inputs["world_coord"] - expected_world).abs().max()
    if float(roundtrip_error.detach().cpu()) > 2e-3:
        raise RuntimeError(
            "World/pixel homography round-trip exceeded 2 mm: "
            f"{float(roundtrip_error.detach().cpu()):.6f} m"
        )

    all_output, _ = model.forward(inputs, if_test=True)
    if all_output.shape[:3] != (
        JMM_NUM_SAMPLES,
        model_args.seq_length,
        window.num_agents,
    ):
        raise RuntimeError(f"Unexpected model output shape: {all_output.shape}")
    future_model_pixel = all_output[:, JMM_OBS_LEN:].detach()
    predictions_world = []
    for sample in future_model_pixel:
        full_pixel = sample * float(model_args.down_factor)
        predictions_world.append(scene.make_world_coord_torch(full_pixel))
    return torch.stack(predictions_world).cpu().numpy()


def command_build_jdv2_cache(args: argparse.Namespace) -> None:
    """Build future-free frozen candidates/graph for official JMM windows."""
    import torch

    from src.models.interaction_graph import build_interaction_graph
    from src.models.model import GDTS
    from src.models.model_utils.cnn_big_images_utils import create_tensor_image
    from src.models.model_utils.sampling_2D_map import generate_goal_candidates
    from src.utils import set_seed

    run_dir = Path(args.run_dir).expanduser().resolve()
    data_file = Path(args.data_file).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    windows = build_jmm_eth_windows(data_file, strict_official=True)
    model_args = _load_run_arguments(run_dir, args.device)
    if not model_args.jdv2_active:
        raise ValueError("build-jdv2-cache requires an active JDV2 run config")
    source_checkpoint = _resolve_source_checkpoint(model_args)
    identity = _jdv2_cache_identity(
        model_args, data_file, source_checkpoint, args.seed
    )

    manifest_path = cache_dir / JMM_JDV2_CACHE_MANIFEST
    if manifest_path.is_file() and not args.overwrite:
        manifest = _load_validated_jdv2_cache(cache_dir, windows, identity)
        _print_json(manifest)
        print("JMM JDV2 deployment cache is already complete.")
        return

    device = torch.device(model_args.device)
    set_seed(args.seed, use_cuda=device.type == "cuda")
    model_args.phase = "build-jdv2-cache"
    model = GDTS(model_args, device).to(device).eval()
    state = torch.load(source_checkpoint, map_location=device)
    state_dict = state.get("model_state_dict", state)
    goal_state = {
        key[len("goal_module."):]: value
        for key, value in state_dict.items()
        if key.startswith("goal_module.")
    }
    if not goal_state:
        goal_state = state_dict
    model.goal_module.load_state_dict(goal_state, strict=True)
    for parameter in model.goal_module.parameters():
        parameter.requires_grad_(False)

    scene = model.dataset.scenes["eth"]
    if scene.semantic_map_pred is None:
        raise RuntimeError("ETH semantic map was not loaded for JMM cache build")
    tensor_image = create_tensor_image(
        scene.semantic_map_pred, down_factor=model_args.down_factor
    )
    record_dir = cache_dir / "records"
    record_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with torch.inference_mode():
        for scene_index, window in enumerate(windows):
            set_seed(args.seed + scene_index, use_cuda=device.type == "cuda")
            batch_data, batch_id = _build_collated_model_batch(
                window, model_args, scene, tensor_image
            )
            inputs, _ = model.prepare_inputs(batch_data, batch_id)
            num_agents = window.num_agents
            image = inputs["tensor_image"].unsqueeze(0).repeat(
                num_agents, 1, 1, 1
            )
            maps = inputs["input_traj_maps"][:, :model_args.obs_length]
            logits = model.goal_module(torch.cat((image, maps), dim=1))
            probability_map = torch.sigmoid(logits[:, -1:])
            goals_map, probability = generate_goal_candidates(
                probability_map,
                num_candidates=model_args.num_goal_candidates,
                device=device,
                use_ttst=model_args.use_ttst,
            )
            goals_world = model._map_goals_to_world(goals_map, scene)
            edge_index, edge_feat, edge_weight = build_interaction_graph(
                inputs["obs_traj_world"],
                inputs["scene_index"],
                graph_type=model_args.graph_type,
                radius=model_args.graph_radius,
                ttc_threshold=model_args.ttc_threshold,
                dt=model_args.trajectory_dt,
            )
            record = {
                "cache_id": f"jmm-eth-{scene_index:06d}",
                "window_directory": window.directory_name,
                "num_goal_candidates": int(model_args.num_goal_candidates),
                "scene_index": inputs["scene_index"].cpu(),
                "scene_ptr": inputs["scene_ptr"].cpu(),
                "frame_ids": inputs["frame_ids"].cpu(),
                "edge_index": edge_index.cpu(),
                "edge_feat": edge_feat.cpu(),
                "edge_weight": edge_weight.cpu(),
                "goal_candidates_map": goals_map.cpu(),
                "goal_candidates_world": goals_world.cpu(),
                "candidate_log_prior": probability.float().clamp_min(
                    1e-8
                ).log().cpu(),
                "contains_future_supervision": False,
            }
            _validate_jdv2_cache_record(record, window, scene_index)
            atomic_torch_save(
                record, str(_cache_record_path(cache_dir, scene_index))
            )
            completed = scene_index + 1
            if completed == 1 or completed % 25 == 0 or completed == len(windows):
                print(
                    f"Cached {completed}/{len(windows)} official scenes",
                    flush=True,
                )

    record_paths = sorted(record_dir.glob("*.pt"))
    manifest = dict(identity)
    manifest.update({
        **_source_provenance(),
        "status": "complete",
        "record_count": len(record_paths),
        "record_set_sha256": _record_set_hash(record_paths),
        "elapsed_seconds": time.time() - started,
    })
    atomic_json_save(manifest, str(manifest_path))
    _load_validated_jdv2_cache(cache_dir, windows, identity)
    _print_json(manifest)


def _manifest_identity(args, checkpoint: Path, data_file: Path) -> dict:
    return {
        **_source_provenance(),
        "protocol": "Joint Metrics Matter ETH official",
        "protocol_version": 1,
        "coordinate_system": "world_metres",
        "num_samples": JMM_NUM_SAMPLES,
        "num_scenes_expected": JMM_ETH_EXPECTED_WINDOWS,
        "num_agent_instances_expected": JMM_ETH_EXPECTED_AGENT_INSTANCES,
        "mean_pedestrians_expected": JMM_ETH_EXPECTED_MEAN_PEDS,
        "data_file": str(data_file.resolve()),
        "data_sha256": sha256_file(data_file),
        "official_data_sha256": JMM_ETH_SOURCE_SHA256,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "run_dir": str(Path(args.run_dir).resolve()),
        "seed": int(args.seed),
    }


def command_infer(args: argparse.Namespace) -> None:
    import torch

    from src.models.model import GDTS
    from src.models.model_utils.cnn_big_images_utils import create_tensor_image
    from src.utils import set_seed

    run_dir = Path(args.run_dir).expanduser().resolve()
    data_file = Path(args.data_file).expanduser().resolve()
    checkpoint = _resolve_checkpoint(run_dir, args.checkpoint)
    windows = build_jmm_eth_windows(data_file, strict_official=True)
    model_args = _load_run_arguments(run_dir, args.device)

    cache_dir = None
    cache_manifest = None
    if model_args.jdv2_active:
        if args.jdv2_cache_dir is None:
            raise RuntimeError(
                "Active JDV2 official evaluation requires "
                "--jdv2-cache-dir built by build-jdv2-cache"
            )
        cache_dir = Path(args.jdv2_cache_dir).expanduser().resolve()
        source_checkpoint = _resolve_source_checkpoint(model_args)
        cache_identity = _jdv2_cache_identity(
            model_args, data_file, source_checkpoint, args.jdv2_cache_seed
        )
        cache_manifest = _load_validated_jdv2_cache(
            cache_dir, windows, cache_identity
        )

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "jmm_official" / checkpoint.stem
    )
    trajectory_root = output_dir / "trajectories" / "gdts"
    manifest_path = output_dir / "manifest.json"
    identity = _manifest_identity(args, checkpoint, data_file)
    model_identity = {
        "goal_model_type": model_args.goal_model_type,
        "sample_coupling": (
            "independent_per_agent_aligned_by_sample_index"
            if model_args.goal_model_type == "independent"
            else "socially_coupled_scene_sample"
        ),
    }
    if cache_manifest is not None:
        cache_manifest_path = cache_dir / JMM_JDV2_CACHE_MANIFEST
        model_identity.update({
            "jdv2_cache_dir": str(cache_dir),
            "jdv2_cache_manifest_sha256": sha256_file(cache_manifest_path),
            "jdv2_cache_record_set_sha256": cache_manifest["record_set_sha256"],
        })
    if manifest_path.is_file():
        with open(manifest_path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        mismatches = {
            key: (existing.get(key), value)
            for key, value in identity.items()
            if existing.get(key) != value
        }
        # Older completed manifests predate these audit fields.  Accept their
        # absence, but never resume into a directory that explicitly records a
        # different model family or sample-coupling contract.
        mismatches.update({
            key: (existing.get(key), value)
            for key, value in model_identity.items()
            if existing.get(key) not in (None, value)
        })
        if mismatches:
            raise ValueError(
                "Output manifest belongs to a different evaluation run; "
                f"choose another --output-dir. Mismatches: {mismatches}"
            )

    identity.update(model_identity)

    running_manifest = dict(identity)
    running_manifest.update({
        "status": "running",
        "trajectory_root": str(trajectory_root),
        "completed_scenes": sum(
            standardized_window_is_complete(
                trajectory_root / window.sequence_name / window.directory_name
            )
            for window in windows
        ),
    })
    atomic_write_json(manifest_path, running_manifest)

    device = torch.device(model_args.device)
    set_seed(args.seed, use_cuda=device.type == "cuda")
    model = GDTS(model_args, device).to(device)
    state = torch.load(checkpoint, map_location=device)
    state_dict = state.get("model_state_dict", state)
    model.load_state_dict(state_dict, strict=True)
    if cache_manifest is not None and isinstance(state, dict):
        expected_source_hash = cache_manifest["goal_checkpoint_sha256"]
        checkpoint_source_hash = state.get("source_checkpoint_hash")
        if checkpoint_source_hash not in (None, expected_source_hash):
            raise RuntimeError(
                "JDV2 checkpoint/cache source Goal U-Net hash mismatch: "
                f"checkpoint={checkpoint_source_hash}, "
                f"cache={expected_source_hash}"
            )
    model.eval()
    scene = model.dataset.scenes["eth"]
    if scene.semantic_map_pred is None:
        raise RuntimeError("ETH semantic map was not loaded for JMM inference")
    tensor_image = create_tensor_image(
        scene.semantic_map_pred, down_factor=model_args.down_factor
    )

    started = time.time()
    completed = 0
    with torch.inference_mode():
        for scene_index, window in enumerate(windows):
            scene_dir = (
                trajectory_root / window.sequence_name / window.directory_name
            )
            if not args.overwrite and standardized_window_is_complete(scene_dir):
                completed += 1
                continue
            cache_record = None
            if cache_dir is not None:
                cache_record = load_cache_record(
                    str(_cache_record_path(cache_dir, scene_index)),
                    allow_future_supervision=False,
                )
                _validate_jdv2_cache_record(
                    cache_record, window, scene_index
                )
                if cache_record["num_goal_candidates"] != \
                        model_args.num_goal_candidates:
                    raise RuntimeError("JMM JDV2 cache/model candidate mismatch")
            # Per-scene seeding makes an interrupted/resumed export identical.
            set_seed(args.seed + scene_index, use_cuda=device.type == "cuda")
            if model_args.jdv2_active and model_args.jdv2_refinement_policy in {
                "structured_gumbel_assignment",
                "exact_lexicographic_persistent_tie",
            }:
                model.jdv2_sampler.set_sampling_context(args.seed, scene_index)
            predictions = _predict_window(
                model, model_args, scene, tensor_image, window, cache_record
            )
            write_standardized_window(trajectory_root, window, predictions)
            completed += 1
            if completed == 1 or completed % 10 == 0 or completed == len(windows):
                elapsed = time.time() - started
                print(
                    f"Exported {completed}/{len(windows)} scenes "
                    f"({elapsed:.1f}s elapsed)",
                    flush=True,
                )

    completed_manifest = dict(identity)
    completed_manifest.update({
        "status": "complete",
        "trajectory_root": str(trajectory_root),
        "completed_scenes": completed,
        "checkpoint_epoch": state.get("epoch") if isinstance(state, dict) else None,
        "elapsed_seconds": time.time() - started,
        "metrics_computed": False,
    })
    atomic_write_json(manifest_path, completed_manifest)
    print(
        "Inference export complete. Metrics were NOT computed.\n"
        f"Trajectory root: {trajectory_root}\n"
        f"Manifest: {manifest_path}"
    )


def command_score(args: argparse.Namespace) -> None:
    trajectory_root = Path(args.trajectory_root).expanduser().resolve()
    metrics = score_standardized_trajectories(
        trajectory_root, strict_official=not args.allow_incomplete
    )
    destination = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json is not None
        else trajectory_root.parent.parent / "metrics.json"
    )
    payload = {
        "protocol": "Joint Metrics Matter ETH official",
        "trajectory_root": str(trajectory_root),
        "scorer_provenance": _source_provenance(),
        **metrics,
    }
    atomic_write_json(destination, payload)
    manifest_path = trajectory_root.parent.parent / "manifest.json"
    if manifest_path.is_file():
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest.update({
            "metrics_computed": True,
            "metrics_file": str(destination),
            "metrics_protocol": payload["protocol"],
            "metrics_scorer_provenance": payload["scorer_provenance"],
            "scored_at_unix": time.time(),
        })
        atomic_write_json(manifest_path, manifest)
    _print_json(payload)
    print(f"Saved metrics: {destination}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export and score GDTS using the official Joint Metrics Matter "
            "ETH (mean 1.4 pedestrians) protocol."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-data", help="download and verify the pinned official ETH data"
    )
    prepare.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    prepare.set_defaults(handler=command_prepare_data)

    inspect = subparsers.add_parser(
        "inspect", help="report protocol window counts without running a model"
    )
    inspect.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    inspect.add_argument(
        "--allow-nonstandard-data",
        action="store_true",
        help="inspect arbitrary compatible data without enforcing 253/364",
    )
    inspect.set_defaults(handler=command_inspect)

    cache = subparsers.add_parser(
        "build-jdv2-cache",
        help=(
            "build a future-free frozen JDV2 deployment cache for the "
            "official ETH windows"
        ),
    )
    cache.add_argument("--run-dir", type=Path, required=True)
    cache.add_argument("--cache-dir", type=Path, required=True)
    cache.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    cache.add_argument("--device", default="cuda:0")
    cache.add_argument("--seed", type=int, default=2035)
    cache.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "rebuild records and replace the complete manifest; existing "
            "records are never trusted without manifest/hash validation"
        ),
    )
    cache.set_defaults(handler=command_build_jdv2_cache)

    infer = subparsers.add_parser(
        "infer",
        help="run K=20 inference and export files; does not compute metrics",
    )
    infer.add_argument("--run-dir", type=Path, required=True)
    infer.add_argument(
        "--checkpoint",
        required=True,
        help="best, an epoch number, filename, or checkpoint path",
    )
    infer.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    infer.add_argument("--output-dir", type=Path)
    infer.add_argument("--device", default="cuda:0")
    infer.add_argument("--seed", type=int, default=2025)
    infer.add_argument(
        "--jdv2-cache-dir",
        type=Path,
        help="required future-free official-window cache for active JDV2",
    )
    infer.add_argument(
        "--jdv2-cache-seed", type=int, default=2035,
        help="candidate-cache seed recorded by build-jdv2-cache",
    )
    infer.add_argument(
        "--overwrite",
        action="store_true",
        help="recompute already complete scene directories",
    )
    infer.set_defaults(handler=command_infer)

    score = subparsers.add_parser(
        "score", help="explicitly compute metrics from a completed export"
    )
    score.add_argument("--trajectory-root", type=Path, required=True)
    score.add_argument("--output-json", type=Path)
    score.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="score a non-official subset (never use for paper comparison)",
    )
    score.set_defaults(handler=command_score)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

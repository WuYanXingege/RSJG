"""Official Joint Metrics Matter (JMM) ETH evaluation protocol helpers.

The ICCV 2023 JMM benchmark does not evaluate arbitrary trajectory fragments.
It uses the S-GAN/AgentFormer ETH test sequence, 8 observed plus 12 future
steps, a one-timestep sliding stride, and only agents present in all 20 steps.
This module keeps that protocol independent from GDTS's training caches.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence
from urllib.request import urlopen

import numpy as np


JMM_ETH_SEQUENCE = "biwi_eth"
JMM_ETH_FRAME_SCALE = 10
JMM_OBS_LEN = 8
JMM_PRED_LEN = 12
JMM_NUM_SAMPLES = 20
JMM_ETH_EXPECTED_WINDOWS = 253
JMM_ETH_EXPECTED_AGENT_INSTANCES = 364
JMM_ETH_EXPECTED_MEAN_PEDS = (
    JMM_ETH_EXPECTED_AGENT_INSTANCES / JMM_ETH_EXPECTED_WINDOWS
)

# Pin the exact file used by the authors' Joint AgentFormer submodule.  Its
# frame/agent/world-coordinate columns generate the 253 ETH windows reported
# by the official code and the table label ETH (1.4).
JMM_AGENTFORMER_COMMIT = "f4c29aa587026a4bce20c631494bf27eac09c93f"
JMM_ETH_SOURCE_URL = (
    "https://raw.githubusercontent.com/ericaweng/Joint_AgentFormer/"
    f"{JMM_AGENTFORMER_COMMIT}/datasets/eth_ucy/eth/biwi_eth.txt"
)
JMM_ETH_SOURCE_SHA256 = (
    "d7cdcedd6472ebaa794bc56e037542e350dfe2ee992f818ab8c417e9fc2a17ad"
)


@dataclass(frozen=True)
class JMMWindow:
    """One synchronized official evaluation scene.

    ``frame_ids`` use the standardized output scale (e.g. 870 rather than
    AgentFormer's internal frame 87), and ``world_coordinates`` has shape
    ``[20, num_agents, 2]`` in metres.
    """

    sequence_name: str
    frame_ids: np.ndarray
    agent_ids: np.ndarray
    world_coordinates: np.ndarray

    def __post_init__(self) -> None:
        frame_ids = np.asarray(self.frame_ids)
        agent_ids = np.asarray(self.agent_ids)
        coordinates = np.asarray(self.world_coordinates)
        expected_shape = (
            JMM_OBS_LEN + JMM_PRED_LEN,
            agent_ids.shape[0],
            2,
        )
        if frame_ids.shape != (JMM_OBS_LEN + JMM_PRED_LEN,):
            raise ValueError("frame_ids must contain exactly 20 entries")
        if agent_ids.ndim != 1 or agent_ids.size == 0:
            raise ValueError("agent_ids must be a non-empty 1-D array")
        if coordinates.shape != expected_shape:
            raise ValueError(
                "world_coordinates must have shape "
                f"{expected_shape}, got {coordinates.shape}"
            )
        if not np.isfinite(coordinates).all():
            raise ValueError("world_coordinates contain NaN or Inf")
        if len(np.unique(frame_ids)) != frame_ids.size:
            raise ValueError("frame_ids must be unique")
        if len(np.unique(agent_ids)) != agent_ids.size:
            raise ValueError("agent_ids must be unique")

    @property
    def last_observation_frame(self) -> int:
        return int(self.frame_ids[JMM_OBS_LEN - 1])

    @property
    def directory_name(self) -> str:
        return f"frame_{self.last_observation_frame:06d}"

    @property
    def num_agents(self) -> int:
        return int(self.agent_ids.size)


def sha256_file(path: os.PathLike | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer_column(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or Inf")
    rounded = np.rint(values)
    if not np.allclose(values, rounded, atol=1e-6, rtol=0.0):
        raise ValueError(f"{name} must contain integer-valued entries")
    return rounded.astype(np.int64)


def load_jmm_eth_table(path: os.PathLike | str) -> np.ndarray:
    """Load either the authors' 17-column file or a 4-column export.

    Returns a float64 table with columns ``frame_id, agent_id, x, y``.  The
    17-column AgentFormer representation stores world x/y at columns 13/15.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JMM ETH source file not found: {path}")
    raw = np.genfromtxt(path, dtype=str)
    raw = np.atleast_2d(raw)
    if raw.shape[0] == 0:
        raise ValueError(f"JMM ETH source file is empty: {path}")
    if raw.shape[1] == 17:
        selected = raw[:, [0, 1, 13, 15]]
    elif raw.shape[1] == 4:
        selected = raw
    else:
        raise ValueError(
            "Expected the official 17-column AgentFormer file or a "
            f"4-column frame/agent/x/y file, got {raw.shape[1]} columns"
        )
    try:
        table = selected.astype(np.float64)
    except ValueError as exc:
        raise ValueError(f"Non-numeric JMM trajectory field in {path}") from exc
    if not np.isfinite(table).all():
        raise ValueError(f"JMM ETH source contains NaN or Inf: {path}")
    table[:, 0] = _integer_column(table[:, 0], "frame_id")
    table[:, 1] = _integer_column(table[:, 1], "agent_id")
    keys = table[:, :2].astype(np.int64)
    if np.unique(keys, axis=0).shape[0] != keys.shape[0]:
        raise ValueError("Duplicate frame_id/agent_id entries in JMM source")
    return table


def build_jmm_eth_windows(
    path: os.PathLike | str,
    *,
    strict_official: bool = True,
) -> list[JMMWindow]:
    """Reproduce the official AgentFormer/JMM ETH window construction.

    The last observation frame advances by one internal timestep.  Agents are
    taken from that frame and retained only when all eight past and twelve
    future positions exist at consecutive internal frame IDs.
    """

    table = load_jmm_eth_table(path)
    frames = _integer_column(table[:, 0], "frame_id")
    agent_ids = _integer_column(table[:, 1], "agent_id")
    coordinates = table[:, 2:4]
    first_frame = int(frames.min())
    last_frame = int(frames.max())

    positions = {
        (int(frame), int(agent)): coordinate
        for frame, agent, coordinate in zip(frames, agent_ids, coordinates)
    }
    agents_at_frame: dict[int, list[int]] = {}
    for frame, agent in zip(frames, agent_ids):
        agents_at_frame.setdefault(int(frame), []).append(int(agent))

    windows: list[JMMWindow] = []
    first_last_observation = first_frame + JMM_OBS_LEN - 1
    final_last_observation = last_frame - JMM_PRED_LEN
    for last_observation in range(
        first_last_observation, final_last_observation + 1
    ):
        internal_frames = np.arange(
            last_observation - JMM_OBS_LEN + 1,
            last_observation + JMM_PRED_LEN + 1,
            dtype=np.int64,
        )
        valid_agents = [
            agent
            for agent in agents_at_frame.get(last_observation, [])
            if all((int(frame), agent) in positions for frame in internal_frames)
        ]
        if not valid_agents:
            continue
        world = np.stack(
            [
                np.stack(
                    [positions[(int(frame), agent)] for agent in valid_agents],
                    axis=0,
                )
                for frame in internal_frames
            ],
            axis=0,
        )
        windows.append(
            JMMWindow(
                sequence_name=JMM_ETH_SEQUENCE,
                frame_ids=internal_frames * JMM_ETH_FRAME_SCALE,
                agent_ids=np.asarray(valid_agents, dtype=np.int64),
                world_coordinates=world,
            )
        )

    summary = summarize_windows(windows)
    if strict_official:
        expected = (
            JMM_ETH_EXPECTED_WINDOWS,
            JMM_ETH_EXPECTED_AGENT_INSTANCES,
        )
        actual = (summary["num_scenes"], summary["num_agent_instances"])
        if actual != expected:
            raise ValueError(
                "This is not the official JMM ETH (1.4) test protocol: "
                f"expected {expected[0]} windows/{expected[1]} agent "
                f"instances, got {actual[0]}/{actual[1]}. Use the pinned "
                "AgentFormer source file; do not subsample agents to force "
                "the mean."
            )
    return windows


def summarize_windows(windows: Sequence[JMMWindow]) -> dict[str, object]:
    counts = [window.num_agents for window in windows]
    distribution = {
        str(count): counts.count(count) for count in sorted(set(counts))
    }
    return {
        "num_scenes": len(windows),
        "num_agent_instances": int(sum(counts)),
        "mean_pedestrians_per_scene": (
            float(np.mean(counts)) if counts else float("nan")
        ),
        "pedestrian_count_distribution": distribution,
    }


def download_official_jmm_eth(
    destination: os.PathLike | str,
    *,
    url: str = JMM_ETH_SOURCE_URL,
    expected_sha256: str = JMM_ETH_SOURCE_SHA256,
) -> Path:
    """Download the pinned official source atomically and verify its hash."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        actual_hash = sha256_file(destination)
        if actual_hash != expected_sha256:
            raise ValueError(
                f"Existing file has unexpected SHA-256: {destination}\n"
                f"expected={expected_sha256}\nactual={actual_hash}"
            )
        build_jmm_eth_windows(destination, strict_official=True)
        return destination

    temporary_path = None
    try:
        with urlopen(url, timeout=60) as response, tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                temporary.write(block)
        actual_hash = sha256_file(temporary_path)
        if actual_hash != expected_sha256:
            raise ValueError(
                "Downloaded JMM source failed SHA-256 verification: "
                f"expected={expected_sha256}, actual={actual_hash}"
            )
        build_jmm_eth_windows(temporary_path, strict_official=True)
        os.replace(temporary_path, destination)
        temporary_path = None
        return destination
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _standard_table(
    frame_ids: np.ndarray,
    agent_ids: np.ndarray,
    coordinates: np.ndarray,
) -> np.ndarray:
    rows = []
    # Match the authors' exporter: one complete trajectory per agent.
    for agent_index, agent_id in enumerate(agent_ids):
        for time_index, frame_id in enumerate(frame_ids):
            x, y = coordinates[time_index, agent_index]
            rows.append((frame_id, agent_id, x, y))
    return np.asarray(rows, dtype=np.float64)


def _atomic_savetxt(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savetxt(temporary, values, fmt="%.3f")
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_standardized_window(
    trajectory_root: os.PathLike | str,
    window: JMMWindow,
    predictions_world: np.ndarray,
) -> Path:
    """Write one scene in the standardized official JMM directory format.

    ``trajectory_root`` is the method root; files are written below
    ``biwi_eth/frame_xxxxxx/{obs,gt,sample_000,...}.txt``.
    """

    predictions = np.asarray(predictions_world, dtype=np.float64)
    expected = (JMM_NUM_SAMPLES, JMM_PRED_LEN, window.num_agents, 2)
    if predictions.shape != expected:
        raise ValueError(
            f"predictions_world must have shape {expected}, got "
            f"{predictions.shape}"
        )
    if not np.isfinite(predictions).all():
        raise ValueError("predictions_world contain NaN or Inf")

    scene_dir = (
        Path(trajectory_root) / window.sequence_name / window.directory_name
    )
    obs = _standard_table(
        window.frame_ids[:JMM_OBS_LEN],
        window.agent_ids,
        window.world_coordinates[:JMM_OBS_LEN],
    )
    gt = _standard_table(
        window.frame_ids[JMM_OBS_LEN:],
        window.agent_ids,
        window.world_coordinates[JMM_OBS_LEN:],
    )
    _atomic_savetxt(scene_dir / "obs.txt", obs)
    _atomic_savetxt(scene_dir / "gt.txt", gt)
    for sample_index, sample in enumerate(predictions):
        prediction = _standard_table(
            window.frame_ids[JMM_OBS_LEN:], window.agent_ids, sample
        )
        _atomic_savetxt(
            scene_dir / f"sample_{sample_index:03d}.txt", prediction
        )
    return scene_dir


def standardized_window_is_complete(scene_dir: os.PathLike | str) -> bool:
    scene_dir = Path(scene_dir)
    required = [scene_dir / "obs.txt", scene_dir / "gt.txt"] + [
        scene_dir / f"sample_{sample_index:03d}.txt"
        for sample_index in range(JMM_NUM_SAMPLES)
    ]
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def _load_standard_table(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing standardized trajectory file: {path}")
    values = np.loadtxt(path, dtype=np.float64)
    values = np.atleast_2d(values)
    if values.shape[1] != 4 or not np.isfinite(values).all():
        raise ValueError(f"Invalid standardized trajectory file: {path}")
    _integer_column(values[:, 0], f"frame_id in {path}")
    _integer_column(values[:, 1], f"agent_id in {path}")
    return values


def _table_to_tensor(
    table: np.ndarray,
    frame_ids: np.ndarray,
    agent_ids: np.ndarray,
    *,
    source: Path,
) -> np.ndarray:
    coordinate_by_key: dict[tuple[int, int], np.ndarray] = {}
    for row in table:
        key = (int(round(row[0])), int(round(row[1])))
        if key in coordinate_by_key:
            raise ValueError(f"Duplicate frame/agent row in {source}: {key}")
        coordinate_by_key[key] = row[2:4]
    expected_keys = {
        (int(frame), int(agent)) for frame in frame_ids for agent in agent_ids
    }
    actual_keys = set(coordinate_by_key)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)[:3]
        extra = sorted(actual_keys - expected_keys)[:3]
        raise ValueError(
            f"Frame/agent grid mismatch in {source}; "
            f"missing={missing}, extra={extra}"
        )
    return np.stack(
        [
            np.stack(
                [coordinate_by_key[(int(frame), int(agent))] for agent in agent_ids]
            )
            for frame in frame_ids
        ]
    )


def _ids_in_order(table: np.ndarray) -> np.ndarray:
    result = []
    seen = set()
    for value in _integer_column(table[:, 1], "agent_id"):
        if int(value) not in seen:
            seen.add(int(value))
            result.append(int(value))
    return np.asarray(result, dtype=np.int64)


def score_standardized_trajectories(
    trajectory_root: os.PathLike | str,
    *,
    strict_official: bool = True,
) -> dict[str, float | int]:
    """Compute the JMM table metrics from standardized trajectory files."""

    sequence_root = Path(trajectory_root) / JMM_ETH_SEQUENCE
    if not sequence_root.is_dir():
        raise FileNotFoundError(
            f"Missing standardized sequence directory: {sequence_root}"
        )
    scene_dirs = sorted(
        path for path in sequence_root.iterdir()
        if path.is_dir() and path.name.startswith("frame_")
    )
    if strict_official and len(scene_dirs) != JMM_ETH_EXPECTED_WINDOWS:
        raise ValueError(
            f"Expected {JMM_ETH_EXPECTED_WINDOWS} official ETH scenes, "
            f"found {len(scene_dirs)}"
        )

    joint_ade = []
    joint_fde = []
    marginal_ade_sum = 0.0
    marginal_fde_sum = 0.0
    total_agents = 0
    for scene_dir in scene_dirs:
        obs_table = _load_standard_table(scene_dir / "obs.txt")
        gt_table = _load_standard_table(scene_dir / "gt.txt")
        agent_ids = _ids_in_order(gt_table)
        if not np.array_equal(np.sort(agent_ids), np.sort(_ids_in_order(obs_table))):
            raise ValueError(f"obs/gt agent IDs differ in {scene_dir}")
        obs_frames = np.unique(_integer_column(obs_table[:, 0], "obs frame_id"))
        gt_frames = np.unique(_integer_column(gt_table[:, 0], "gt frame_id"))
        if obs_frames.size != JMM_OBS_LEN or gt_frames.size != JMM_PRED_LEN:
            raise ValueError(
                f"Expected 8 observation/12 future frames in {scene_dir}, "
                f"got {obs_frames.size}/{gt_frames.size}"
            )
        _table_to_tensor(obs_table, obs_frames, agent_ids, source=scene_dir / "obs.txt")
        gt = _table_to_tensor(
            gt_table, gt_frames, agent_ids, source=scene_dir / "gt.txt"
        )
        samples = []
        for sample_index in range(JMM_NUM_SAMPLES):
            sample_path = scene_dir / f"sample_{sample_index:03d}.txt"
            sample_table = _load_standard_table(sample_path)
            samples.append(
                _table_to_tensor(
                    sample_table, gt_frames, agent_ids, source=sample_path
                )
            )
        predictions = np.stack(samples, axis=0)  # [K,T,N,2]
        distances = np.linalg.norm(predictions - gt[None], axis=-1)
        per_sample_agent_ade = distances.mean(axis=1)  # [K,N]
        per_sample_agent_fde = distances[:, -1]       # [K,N]
        joint_ade.append(float(per_sample_agent_ade.mean(axis=1).min()))
        joint_fde.append(float(per_sample_agent_fde.mean(axis=1).min()))
        marginal_ade_sum += float(per_sample_agent_ade.min(axis=0).sum())
        marginal_fde_sum += float(per_sample_agent_fde.min(axis=0).sum())
        total_agents += int(agent_ids.size)

    if not scene_dirs or total_agents == 0:
        raise ValueError("No complete standardized scenes to score")
    if strict_official and total_agents != JMM_ETH_EXPECTED_AGENT_INSTANCES:
        raise ValueError(
            f"Expected {JMM_ETH_EXPECTED_AGENT_INSTANCES} agent instances, "
            f"found {total_agents}"
        )
    return {
        "minJADE@20": float(np.mean(joint_ade)),
        "minJFDE@20": float(np.mean(joint_fde)),
        "minADE@20": marginal_ade_sum / total_agents,
        "minFDE@20": marginal_fde_sum / total_agents,
        "num_scenes": len(scene_dirs),
        "num_agent_instances": total_agents,
        "mean_pedestrians_per_scene": total_agents / len(scene_dirs),
    }


def atomic_write_json(path: os.PathLike | str, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, indent=2, sort_keys=True)
            temporary.write("\n")
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

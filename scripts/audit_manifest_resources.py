"""Measure bounded sequential loading of complete-scene shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
from pathlib import Path

from ebjd.data import SceneManifestDataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_rss_bytes() -> int:
    status = Path("/proc/self/status").read_text(encoding="utf-8")
    line = next(value for value in status.splitlines() if value.startswith("VmRSS:"))
    return int(line.split()[1]) * 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    before = current_rss_bytes()
    dataset = SceneManifestDataset(manifest_path)
    maximum = before
    agents = 0
    maximum_n = 0
    for index in range(len(dataset)):
        item = dataset[index]
        count = int(item["observed"].shape[0])
        agents += count
        maximum_n = max(maximum_n, count)
        maximum = max(maximum, current_rss_bytes())
    result = {
        "schema": "ebjd-manifest-load-resources-v1",
        "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
        "scenes": len(dataset), "agents": agents, "maximum_N": maximum_n,
        "shard_bytes": sum(path.stat().st_size for path in dataset.shards),
        "rss_before_bytes": before, "maximum_observed_rss_bytes": maximum,
        "maximum_observed_rss_increase_bytes": maximum - before,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "complete_scene_streaming": True,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

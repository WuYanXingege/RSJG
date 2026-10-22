from copy import deepcopy
from pathlib import Path

import torch
import yaml

from ebjd.data import SyntheticSceneDataset, collate_scenes


def load_paper_config(fold="hotel"):
    root = Path(__file__).resolve().parents[1]

    def read(path):
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def merge(left, right):
        result = deepcopy(left)
        for key, value in right.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result

    return merge(
        merge(read(root / "configs" / "architecture.yaml"),
              read(root / "configs" / "training.yaml")),
        read(root / "configs" / "folds" / f"{fold}.yaml"),
    )


def make_batch(agent_counts=(2,), pixels=16):
    dataset = SyntheticSceneDataset(len(agent_counts), max(agent_counts) + 1, pixels, seed=7)
    items = []
    for index, count in enumerate(agent_counts):
        item = dataset[index]
        item = {key: value for key, value in item.items()}
        item["observed"] = item["observed"][:count]
        item["future"] = item["future"][:count]
        item["semantic_maps"] = item["semantic_maps"][:count]
        items.append(item)
    return collate_scenes(items)


def assert_finite_valid(tensor, valid):
    mask = valid[:, None, :, None, None].expand_as(tensor)
    assert torch.isfinite(tensor[mask]).all()

import torch

from ebjd.data import SyntheticSceneDataset, collate_scenes


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

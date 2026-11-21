"""Independent fixed synthetic observations/maps/candidates, never a real bank."""
import math
import torch
from ebjd.data import collate_scenes
from bcep_joint.common import local_seed, save
from .common import OUT, read, write, sha


def prepare():
    folder = OUT / 'synthetic_inputs'
    assert not folder.exists(), 'never regenerate an existing audit dataset'
    folder.mkdir(parents=True)
    # Keep the native 168-update epoch length. Only the first 64 shuffled IDs
    # are materialized/used; every logical batch includes 1,2,9,57 people.
    order = torch.randperm(2681, generator=torch.Generator().manual_seed(3201)).tolist()
    pattern = [1, 2, 9, 57] + [2] * 12
    rows = {}
    for position, index in enumerate(order[:64]):
        n = pattern[position % 16]
        gen = torch.Generator().manual_seed(local_seed('SSEC-INDEPENDENT-SYNTHETIC', index, 2035))
        side = math.ceil(n ** .5)
        centers = torch.stack((torch.arange(n) % side, torch.arange(n) // side), -1).float() * .55
        centers += torch.randn(n, 2, generator=gen) * .04
        velocity = torch.randn(n, 1, 2, generator=gen) * .18
        times = torch.arange(-7, 13).reshape(1, 20, 1) * .4
        path = centers[:, None] + velocity * times + torch.randn(n, 20, 2, generator=gen) * .025
        maps = torch.zeros(n, 6, 256, 256, dtype=torch.uint8)
        maps[:, 1] = 1
        item = dict(observed=path[:, :8].contiguous(), future=path[:, 8:].contiguous(), semantic_maps=maps,
                    scene_id=f'SSEC_SYNTHETIC_{index}', agent_ids=[f'SYNTH_AGENT_{j}' for j in range(n)],
                    frame_ids=list(range(20)), timestamps=[.4 * j for j in range(20)], source_sequence='SYNTHETIC_ONLY',
                    metadata=dict(synthetic=True, unit='m', dt_seconds=.4, crop_pixels=256, semantic_channels=6))
        # Candidate construction uses observed motion plus independent offsets;
        # it never reads future targets or any Provider/bank file.
        goals = item['observed'][None, :, -1] + 4.8 * velocity[:, 0][None]
        goals = (goals + torch.randn(20, n, 2, generator=gen) * 1.2).contiguous()
        name = f'{index:04d}.pt'
        save(dict(item=item, goals=goals, original_slots=torch.arange(20).expand(n, 20).clone()), folder / name)
        rows[str(index)] = dict(file=name, sha256=sha(folder / name), agents=n, shuffled_position=position)
    manifest = dict(synthetic_only=True, train_ids=list(range(2681)), materialized_rows=rows,
                    first_epoch_order=order, logical_pattern=pattern, map_shape=[6, 256, 256],
                    construction_seed_namespace='SSEC-INDEPENDENT-SYNTHETIC / index / 2035',
                    optimization_seed=3201, real_input_files=0)
    write(folder / 'MANIFEST.json', manifest)
    return manifest


class SyntheticData:
    TEST_ONLY = True

    def __init__(self):
        self.folder = OUT / 'synthetic_inputs'
        self.manifest = read(self.folder / 'MANIFEST.json')
        assert self.manifest['synthetic_only'] and self.manifest['real_input_files'] == 0
        self.accesses = []
        self.cache = {}

    def row(self, index):
        index = int(index)
        if index not in self.cache:
            r = self.manifest['materialized_rows'][str(index)]
            p = self.folder / r['file']
            assert p.resolve().is_relative_to(self.folder.resolve()) and sha(p) == r['sha256']
            x = torch.load(p, weights_only=False, map_location='cpu')
            assert x['item']['metadata']['synthetic'] and x['item']['source_sequence'] == 'SYNTHETIC_ONLY'
            assert x['item']['observed'].shape[0] == r['agents']
            assert not x['goals'].requires_grad
            self.cache[index] = x
        return self.cache[index]

    def batch(self, ids, purpose):
        assert purpose == 'train', 'only synthetic audit samples; no dev/test input API'
        self.accesses.append(list(ids))
        return collate_scenes([self.row(i)['item'] for i in ids])


class SyntheticBank:
    TEST_ONLY = True

    def __init__(self, data):
        assert type(data) is SyntheticData
        self.data = data

    def batch(self, ids, batch):
        goals = torch.zeros(len(ids), 20, batch.valid.shape[1], 2)
        for b, i in enumerate(ids):
            x = self.data.row(i)
            n = len(x['item']['agent_ids'])
            assert batch.scene_ids[b] == x['item']['scene_id']
            assert batch.agent_ids[b] == x['item']['agent_ids']
            assert torch.equal(x['original_slots'], torch.arange(20).expand(n, 20))
            goals[b, :, :n] = x['goals']
        return goals

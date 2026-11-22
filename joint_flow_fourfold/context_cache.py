"""Fold-local frozen context/latent targets; no heldout cache or access exists."""
import time
from pathlib import Path

import torch

from bcep.contracts import canonical_hash, digest
from bcep.motion_frame import frame, to_local
from bcep_joint.common import read, save, write
from joint_flow_dependency.inference import encode_observed
from joint_flow_dependency.model import undirected_edges


@torch.no_grad()
def build_cache(inputs, provider, device):
    root = Path(inputs.output)/'latent_cache'
    root.mkdir(parents=True, exist_ok=True)
    identity = dict(inputs.identity, schema='frozen-flow-synchronous-endpoint-latents-v1', precision='fp32')
    key = canonical_hash(identity)
    if (root/'IDENTITY.json').exists():
        assert read(root/'IDENTITY.json') == identity
    else:
        write(root/'IDENTITY.json', identity)
    result = {}
    for purpose in ('train', 'selection_dev'):
        folder = root/purpose
        folder.mkdir(exist_ok=True)
        manifest_path = root/f'{purpose}_MANIFEST.json'
        previous = read(manifest_path) if manifest_path.exists() else None
        indices = inputs.split[purpose+'_indices']
        if previous is not None:
            assert len(previous) == len(indices)
        result[purpose], manifest = [], []
        start = time.monotonic()
        for position, index in enumerate(indices):
            path = folder/f'{position:04d}.pt'
            if previous is not None:
                entry = previous[position]
                assert entry['position'] == position and entry['file'] == str(path.relative_to(root))
                assert digest(path) == entry['sha256']
            if path.exists():
                record = torch.load(path, map_location='cpu', weights_only=False)
                assert record['identity'] == key and record['index'] == index
            else:
                batch = inputs.data.batch([index], purpose, truth=False).to(device)
                context, _ = encode_observed(provider, batch)
                observed = batch.observed[0]
                truth = inputs.data.batch([index], purpose, truth=True).future[0, :, -1].to(device)
                origin, rotation, scale = frame(observed, 12, provider.base.dt, provider.base.scales[-1])
                local = to_local(truth, origin, rotation, scale)
                latent, _ = provider.base.flow.transform(local, context, 12*provider.base.dt, inverse=True)
                assert torch.isfinite(latent).all()
                record = dict(index=index, identity=key,
                    inputs=dict(observed=observed.cpu(), context=context.cpu(), edges=undirected_edges(observed).cpu()),
                    latent_truth=latent.cpu())
                save(record, path)
            n = len(inputs.split['members'][index]['agent_ids'])
            assert record['inputs']['observed'].shape == (n, 8, 2)
            assert record['inputs']['context'].shape == (n, 128)
            assert record['latent_truth'].shape == (n, 2)
            assert all(torch.isfinite(value).all() for value in record['inputs'].values())
            assert torch.isfinite(record['latent_truth']).all()
            result[purpose].append(record)
            manifest.append(dict(position=position, file=str(path.relative_to(root)), sha256=digest(path)))
            if (position+1) % 100 == 0 or position+1 == len(indices):
                progress = dict(phase='FROZEN_LATENT_CACHE', fold=inputs.fold, split=purpose,
                    windows=position+1, total=len(indices), seconds=time.monotonic()-start)
                write(Path(inputs.output)/'PROGRESS.json', progress)
                print(progress, flush=True)
        write(manifest_path, manifest)
    return result

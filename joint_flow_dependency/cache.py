"""Small frozen context/ground-truth latent cache, with explicit observed/target separation."""
import time
import torch
from bcep.motion_frame import frame, to_local
from bcep.contracts import canonical_hash, digest
from bcep_joint.common import read, write, save
from .inputs import RUN
from .model import undirected_edges
from .inference import encode_observed, flow_world


@torch.no_grad()
def build_cache(inputs, provider, device):
    root = RUN/'latent_cache'
    root.mkdir(parents=True, exist_ok=True)
    identity = dict(inputs.identity, schema='frozen-flow-synchronous-endpoint-latents-v1', precision='fp32')
    key = canonical_hash(identity)
    if (root/'IDENTITY.json').exists():assert read(root/'IDENTITY.json') == identity
    else:write(root/'IDENTITY.json', identity)
    result = {}
    for purpose in ('train', 'selection_dev'):
        folder = root/purpose; folder.mkdir(exist_ok=True)
        result[purpose] = []; manifest = []
        start = time.monotonic()
        for pos, index in enumerate(inputs.split[purpose+'_indices']):
            path = folder/f'{pos:04d}.pt'
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
                z, _ = provider.base.flow.transform(local, context, 12*provider.base.dt, inverse=True)
                assert torch.isfinite(z).all()
                record = dict(index=index, identity=key,
                              inputs=dict(observed=observed.cpu(), context=context.cpu(), edges=undirected_edges(observed).cpu()),
                              latent_truth=z.cpu())
                save(record, path)
            result[purpose].append(record)
            manifest.append(dict(position=pos, file=str(path.relative_to(root)), sha256=digest(path)))
            if (pos+1) % 100 == 0 or pos+1 == len(inputs.split[purpose+'_indices']):
                progress = dict(phase='FROZEN_LATENT_CACHE', split=purpose, windows=pos+1,
                                total=len(inputs.split[purpose+'_indices']), seconds=time.monotonic()-start)
                write(RUN/'PROGRESS.json', progress); print(progress, flush=True)
        write(root/f'{purpose}_MANIFEST.json', manifest)
    return result

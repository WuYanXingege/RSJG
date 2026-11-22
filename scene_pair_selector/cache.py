"""Resumable, identity-bound offline two-branch labels and train-only scaling."""
from dataclasses import fields
import time
import torch
from bcep.contracts import canonical_hash, digest
from bcep_joint.common import read, save, write
from bcep_joint.inference import generate as native_generate, keyed_noise
from ebjd.unified import gather_candidates
from ebjd.metrics import BatchMetricStatistics, hard_metric_statistics, MetricAccumulator
from .inference import paired_context, decode, generate
from .model import raw_features
from .inputs import RUN


def cpu_statistics(stats):
    return {f.name: getattr(stats, f.name).detach().cpu() for f in fields(stats)}


def accumulate(records, switches=None):
    groups = {name: MetricAccumulator() for name in ('all', 'N_ge_9')}
    for pos, record in enumerate(records):
        branch = 'I' if switches is not None and bool(switches[pos]) else 'H0'
        stats = BatchMetricStatistics(**record['statistics'][branch])
        groups['all'].update_statistics(stats)
        if int(stats.agent_valid.sum()) >= 9:
            groups['N_ge_9'].update_statistics(stats)
    return {name: acc.as_dict() for name, acc in groups.items()}


@torch.no_grad()
def preflight(inputs, model, device):
    ids = inputs.split['selection_dev_indices']
    pick = [next(i for i in ids if test(len(inputs.split['members'][i]['agent_ids'])))
            for test in (lambda n: n == 1, lambda n: 2 <= n <= 4, lambda n: n >= 9)]
    results = []
    for idx in pick:
        batch, goals = inputs.batch('selection_dev', idx, device, truth=False)
        _, native, pi, _ = native_generate(model, batch.observed, batch.semantic_maps, batch.valid,
            batch.scene_ids, batch.agent_ids, goals, 2035, mode='H')
        path, detail = generate(model, None, batch.observed, batch.semantic_maps, batch.valid,
            batch.scene_ids, batch.agent_ids, goals, force='H0')
        assert torch.equal(detail['selected_pi'], pi)
        assert torch.equal(path, native), 'forced H0 must exactly replay native forward'
        assert detail['ddim_calls'] == 1
        results.append(dict(agents=int(batch.valid.sum()), exact_native_pi=True,
                            path_max_abs=float((path-native).abs().max()), exact_path=True))
    receipt = dict(passed=True, windows=results, whole_noise_vector_inverse_gather=True,
                   candidate_bijection=True, future_loaded=False, strict_checkpoint_loading=True)
    write(RUN/'PREFLIGHT.json', receipt)
    return receipt


@torch.no_grad()
def build_cache(inputs, model, device):
    identity = dict(inputs.identity, schema='scene-pair-two-branch-cache-v1', noise_seed=2035,
                    K=20, DDIM=20, precision='fp32', device=str(device))
    root = RUN/'cache'
    root.mkdir(parents=True, exist_ok=True)
    if (root/'IDENTITY.json').exists():
        assert read(root/'IDENTITY.json') == identity
    else:
        write(root/'IDENTITY.json', identity)
    identity_hash = canonical_hash(identity)
    records = {}
    for purpose in ('train', 'selection_dev'):
        directory = root/purpose
        directory.mkdir(exist_ok=True)
        records[purpose] = []
        manifest = []
        start = time.monotonic()
        for pos, idx in enumerate(inputs.split[purpose+'_indices']):
            path = directory/f'{pos:04d}.pt'
            if path.exists():
                record = torch.load(path, map_location='cpu', weights_only=False)
                assert record['identity'] == identity_hash and record['index'] == idx
            else:
                # Observed-only data path ends before future loading.
                batch, goals = inputs.batch(purpose, idx, device, truth=False)
                context, ip, hp = paired_context(model, batch.observed, batch.semantic_maps, batch.valid, goals)
                noise = keyed_noise(batch.scene_ids, batch.agent_ids, batch.valid, 2035).to(device)
                paths = {name: decode(model, context, goals, noise, pi) for name, pi in [('I', ip), ('H0', hp)]}
                truth = inputs.data.batch([idx], purpose, truth=True).future.to(device)
                statistics = {name: cpu_statistics(hard_metric_statistics(p, truth, batch.valid)) for name, p in paths.items()}
                # Explicit input whitelist: labels and IDs cannot enter scorer calls.
                features = dict(observed=batch.observed.cpu(), social=context.social_h.cpu(), valid=batch.valid.cpu(),
                                initial_goals=gather_candidates(goals, ip).cpu(),
                                feedback_goals=gather_candidates(goals, hp).cpu())
                target = torch.stack([statistics['I'][key]-statistics['H0'][key]
                                      for key in ('scene_jade', 'scene_jfde')], -1)
                record = dict(identity=identity_hash, index=idx, features=features,
                              initial_pi=ip.cpu(), feedback_pi=hp.cpu(), target=target, statistics=statistics)
                save(record, path)
            records[purpose].append(record)
            manifest.append(dict(position=pos, file=str(path.relative_to(root)), sha256=digest(path)))
            if (pos+1) % 50 == 0 or pos+1 == len(inputs.split[purpose+'_indices']):
                progress = dict(phase='CACHE', purpose=purpose, windows=pos+1,
                                total=len(inputs.split[purpose+'_indices']), seconds=time.monotonic()-start)
                write(RUN/'PROGRESS.json', progress)
                print(progress, flush=True)
        write(root/f'{purpose}_MANIFEST.json', manifest)
    return records


@torch.no_grad()
def fit_normalization(train, model):
    """Population moments over valid training nodes/edges of I and H0 only."""
    sums = [torch.zeros(212, dtype=torch.float64), torch.zeros(6, dtype=torch.float64)]
    squares = [x.clone() for x in sums]
    counts = [0, 0]
    for record in train:
        f = record['features']
        for name in ('initial_goals', 'feedback_goals'):
            node, edge, _ = raw_features(f['observed'], f['social'], f[name], f['valid'])
            node = node.permute(0, 2, 1, 3)[f['valid']].reshape(-1, 212)
            for k, x in enumerate((node, edge.reshape(-1, 6))):
                x = x.double()
                sums[k] += x.sum(0)
                squares[k] += x.square().sum(0)
                counts[k] += len(x)
    for k, prefix in enumerate(('node', 'edge')):
        mean = sums[k]/counts[k]
        std = (squares[k]/counts[k]-mean.square()).clamp_min(0).sqrt().clamp_min(1e-3)
        getattr(model, prefix+'_mean').copy_(mean.float())
        getattr(model, prefix+'_std').copy_(std.float())
    return dict(source='train only', windows=len(train), valid_node_observations=counts[0],
                valid_edge_observations=counts[1], minimum_std=1e-3)


def load_cache():
    records = {}
    for purpose in ('train', 'selection_dev'):
        manifest = read(RUN/'cache'/f'{purpose}_MANIFEST.json')
        records[purpose] = []
        for row in manifest:
            path = RUN/'cache'/row['file']
            assert digest(path) == row['sha256']
            records[purpose].append(torch.load(path, map_location='cpu', weights_only=False))
    return records

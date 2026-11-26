"""Pin the five official teachers and build train-only endpoint caches."""
from dataclasses import dataclass
from pathlib import Path
import math
import time

import numpy as np
import torch

from bcep.contracts import canonical_hash, digest, require
from bcep_joint.common import read, write
from bcep_joint.inference import check_permutation
from ebjd.metrics import MetricAccumulator, hard_metric_statistics
from ebjd.unified import gather_candidates
from joint_flow_dependency.inference import decode, encode_observed, sample_endpoints
from joint_flow_dependency.model import EdgeDependence
from joint_flow_fourfold.common import immutable
from joint_flow_fourfold.inputs import Inputs, state_hash
from joint_flow_frozen_test.inputs import Inputs as HistoricalZara2

SOURCE = Path(__file__).resolve().parents[1]
ASSETS = SOURCE.parent / 'RSJG_EBJD_assets/experiments'
ARCHIVE = SOURCE / 'docs/research/continuous_joint_flow/fourfold_confirmation_20261122'
FOUR_FOLD = ASSETS / 'joint_flow_fourfold_confirmation_20261122'
BASE_COMMIT = '3296fc5968463ed14bf43265c064b3856d55ed30'
TEACHER_SEED = 2035
METRICS = ('JADE', 'JFDE', 'minADE', 'minFDE')
FOLDS = ('zara1', 'hotel', 'univ', 'eth', 'zara2')
PURPOSE = 'heldout_confirmation'


@dataclass
class FoldAssets:
    """Frozen teachers plus access to observed inputs and train/dev targets."""

    fold: str
    split: dict
    data: object
    train_bank: object
    model: torch.nn.Module
    provider: torch.nn.Module
    dependence: torch.nn.Module
    identity: dict
    dev_B: dict
    historical: Path
    device: torch.device
    geometry: dict

    def batch(self, index, purpose, truth=False, device=None):
        require(purpose in ('train', 'selection_dev'), 'training asset reader disallows heldout')
        batch = self.data.batch([index], purpose, truth=truth)
        return batch.to(self.device if device is None else device)

    def frozen_states(self):
        modules = {'bridge_and_A': self.model, 'provider': self.provider, 'B': self.dependence}
        for network in modules.values():
            require(not any(m.training for m in network.modules()), 'frozen teacher entered training mode')
            require(not any(p.requires_grad or p.grad is not None for p in network.parameters()),
                    'frozen teacher has gradient state')
        return {name: state_hash(network) for name, network in modules.items()}


def _historical_reference(fold, split, identity):
    """Reconcile official receipt and anonymous metric tables without loading truth."""
    archived = ARCHIVE / fold / 'test'
    historical = (ASSETS / 'zara2_joint_flow_frozen_test_20261122' if fold == 'zara2'
                  else FOUR_FOLD / fold / 'test')
    receipt = read(historical / 'SCORE_RECEIPT.json')
    require(receipt == read(archived / 'SCORE_RECEIPT.json') and receipt['status'] == 'COMPLETE',
            'historical scoring receipt differs from the fixed report')
    seal = read(historical / 'PREDICTIONS_FROZEN.json')
    require(receipt['prediction_manifest_sha256'] == digest(historical / 'PREDICTIONS_FROZEN.json'),
            'official prediction seal hash differs')
    seeds = list(range(2035, 2040))
    heldout = split[PURPOSE + '_indices']
    require(receipt['windows'] == len(heldout) and receipt['seeds'] == seeds and
            receipt['K_per_seed'] == 20 and receipt['prediction_frames'] == 12 and
            receipt['all_predictions_verified_before_any_truth'] and receipt['parameter_updates'] == 0,
            'historical scoring population or protocol differs')
    source = seal['identity']
    members = canonical_hash([split['members'][index] for index in heldout])
    if fold == 'zara2':
        require(source['checkpoint_sha256'] == identity['A_sha256'] and
                source['B_checkpoint_sha256'] == identity['B_sha256'] and
                source['B_completed_epoch'] == identity['B_epoch'] and source['heldout_member_hash'] == members,
                'historical ZARA2 A/B and heldout identities differ')
    else:
        require(source['upstream']['checkpoint_sha256'] == identity['A_sha256'] and
                source['B_sha256'] == identity['B_sha256'] and
                source['B_epoch'] == identity['B_epoch'] and source['heldout_members'] == members,
                'official A/B and heldout identities differ')
    hashes = {}
    for name in ('PER_SEED.csv', 'PER_WINDOW.csv'):
        actual = digest(historical / name)
        require(actual == digest(archived / name) == receipt['output_sha256'][name],
                f'official {fold} {name} hash differs')
        hashes[name] = actual
    return historical, dict(
        score_receipt_sha256=digest(historical / 'SCORE_RECEIPT.json'),
        prediction_seal_sha256=receipt['prediction_manifest_sha256'],
        heldout_members_sha256=members,
        tables=hashes,
    )


def _regular(fold, output, device):
    root = FOUR_FOLD / fold
    manifest_path = root / 'UPSTREAM_FROZEN.json'
    selected_path = root / 'B_FROZEN.json'
    pinned = read(ARCHIVE / 'FROZEN_MODELS.json')['folds'][fold]
    require(digest(manifest_path) == pinned['upstream_manifest_sha256'] and
            digest(selected_path) == pinned['B_manifest_sha256'], 'official teacher manifest changed')
    selected = read(selected_path)
    native = Inputs(manifest_path, output, device)
    require(native.stages['joint']['sha256'] == pinned['A_sha256'] and
            native.stages['joint']['frozen']['epoch'] == pinned['A_epoch'] and
            native.stages['density']['sha256'] == pinned['density_sha256'] and
            native.stages['head']['sha256'] == pinned['head_sha256'],
            'official upstream teacher assets differ')
    checkpoint = Path(selected['rank1_checkpoint'])
    require(selected['fold'] == fold and selected['rank1_epoch'] == pinned['B_epoch'] and
            digest(checkpoint) == selected['rank1_sha256'] == pinned['B_sha256'] and
            selected['test_access'] is False, 'official B checkpoint differs')
    rank = selected['rank1_metrics']
    require(rank['seed'] == TEACHER_SEED and rank['windows'] == len(native.split['selection_dev_indices']) and
            rank['all'] == read(ARCHIVE / fold / 'B_FROZEN.json')['rank1_metrics']['all'],
            'official dev-B scores differ')
    model, provider = native.load_frozen(device)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    require(saved['completed_epoch'] == pinned['B_epoch'] and saved['identity'] == selected['identity'],
            'loaded B training identity differs')
    dependence = EdgeDependence()
    dependence.load_state_dict(saved['model'], strict=True)
    identity = dict(
        A_epoch=pinned['A_epoch'], A_sha256=pinned['A_sha256'],
        B_epoch=pinned['B_epoch'], B_sha256=pinned['B_sha256'],
        provider=native.provider_identity,
        upstream_manifest_sha256=pinned['upstream_manifest_sha256'],
        B_manifest_sha256=pinned['B_manifest_sha256'],
        train_bank_sha256=native.banks['train'].hash,
    )
    return native, model, provider, dependence, identity, rank['all'], native.geometry


def _zara2(device):
    # Native historical loader already validates A5, B35, Provider epochs,
    # train/dev bank identities, split, and B's training contract.
    historic = HistoricalZara2()
    native = historic._base
    model, provider, dependence = historic.load_frozen(device)
    rank = historic.archive_training['rank1_metrics']
    require(rank['windows'] == len(native.split['selection_dev_indices']) and
            rank['actual_ddim_calls'] == rank['windows'] and rank['ddim_steps'] == 20,
            'historical ZARA2 B dev scores are incomplete')
    identity = dict(
        A_epoch=5, A_sha256=historic.identity['checkpoint_sha256'],
        B_epoch=35, B_sha256=historic.identity['B_checkpoint_sha256'],
        provider=historic.provider_identity,
        B_history_commit=historic.archive_config.get('runtime_source_commit', historic.identity['frozen_dev_commit']),
        B_training_receipt_sha256=canonical_hash(historic.archive_training),
        train_bank_sha256=native.banks['train'].hash,
    )
    return native, model, provider, dependence, identity, rank['all'], historic.geometry


def load_fold(fold, output, device='cuda'):
    """Load official FP32 teacher modules strictly on the requested device."""
    require(fold in FOLDS, f'unsupported fold {fold}')
    device = torch.device(device)
    native, model, provider, dependence, identity, native_metrics, geometry = (
        _zara2(device) if fold == 'zara2' else _regular(fold, Path(output), device)
    )
    for module in (model, provider, dependence):
        module.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    baseline = {metric: float(native_metrics[metric]) for metric in METRICS}
    require(all(math.isfinite(v) and v > 0 for v in baseline.values()), 'invalid dev B normalization')
    historical, test_identity = _historical_reference(fold, native.split, identity)
    identity.update(fold=fold, source_report_commit=BASE_COMMIT, historical=test_identity,
                    split_sha256=canonical_hash(native.split), dev_B_seed=TEACHER_SEED,
                    dev_B=baseline)
    assets = FoldAssets(fold, native.split, native.data, native.banks['train'], model, provider,
                        dependence, identity, baseline, historical, device, geometry)
    assets.frozen_states()
    return assets


def scale_indices(split):
    """Rank only train windows by a stable hash of the original member."""
    indices = split['train_indices']
    require(bool(indices) and len(set(indices)) == len(indices), 'invalid train population')
    sorted_indices = sorted(indices, key=lambda i: (
        canonical_hash(['direct-k20-b-train-scale-v1', split['fold'], split['members'][i]]), i
    ))
    return tuple(sorted_indices[:256])


def _arrays(path, n, with_path):
    with np.load(path, allow_pickle=False) as archive:
        shapes = {'A': (20, n, 2), 'B': (20, n, 2)}
        if with_path:
            shapes['B_path'] = (20, n, 12, 2)
        require(set(archive.files) == set(shapes), 'train cache has unexpected fields')
        values = {key: archive[key] for key in shapes}
    for key, array in values.items():
        require(array.shape == shapes[key] and array.dtype == np.float32 and np.isfinite(array).all(),
                f'{path.name}: invalid teacher {key}')
    if with_path:
        np.testing.assert_allclose(values['B_path'][..., -1, :], values['B'], rtol=1e-6, atol=1e-6)
    return values


@torch.no_grad()
def _teacher_arrays(assets, index, with_path):
    # Preserve the native Bank.batch identity checks before transferring to GPU.
    host = assets.data.batch([index], 'train', truth=False)
    require(host.valid.all() and not host.future.count_nonzero(), 'teacher cache accepts observations only')
    goals = assets.train_bank.batch([index], host).to(assets.device)
    batch = host.to(assets.device)
    model, provider = assets.model, assets.provider
    bridge_context = model.encode_context(batch.observed, batch.semantic_maps, batch.valid)
    a, permutation, _ = model.coupling.assign(
        bridge_context, goals, model.bridge.representation.goal_scale, straight_through=False
    )
    check_permutation(permutation, batch.valid)
    require(torch.equal(a, gather_candidates(goals, permutation)), 'A H0 gather differs from native')
    context, features = encode_observed(provider, batch)
    b = sample_endpoints(provider, assets.dependence, batch, context, features, seed=TEACHER_SEED)
    require(b['correction_calls'] == 1 and b['raw_indices'].unique().numel() == 20,
            'B native endpoint cache violated K20 contract')
    result = dict(A=a[0].cpu().numpy(), B=b['corrected'][0].cpu().numpy())
    if with_path:
        # One real 20-step bridge call on exactly this B sample for normalization.
        path = decode(model, bridge_context, b['corrected'], b['noise'])[0]
        result['B_path'] = path.cpu().numpy()
    return result


class TeacherCache:
    """Verified immutable train-only A/B goals; hold no feature maps or labels."""

    def __init__(self, folder, manifest, scale_receipt):
        self.folder = folder
        self.rows = {row['index']: row for row in manifest['rows']}
        self.indices = tuple(self.rows)
        self.scales = scale_receipt['scales']
        self.manifest_sha256 = digest(folder / 'MANIFEST.json')
        self.identity = dict(
            teacher_identity=manifest['identity']['teachers'],
            teachers_manifest_sha256=self.manifest_sha256,
            scale_receipt_sha256=digest(folder / 'TRAIN_B_SCALES.json'),
            train_member_count=len(self.indices),
            scale_member_count=scale_receipt['scored_windows'],
            scale_members_sha256=scale_receipt['members_sha256'],
            scales=self.scales,
        )
        self.receipt = dict(
            teacher_cache_wall_seconds=manifest['cache_wall_seconds'],
            scale_score_wall_seconds=scale_receipt['score_seconds'],
            teacher_cache_ddim_calls=manifest['scale_B_ddim_calls'],
            teacher_A_ddim_calls=0,
            teacher_B_endpoints_only_windows=len(self.indices)-manifest['scale_B_ddim_calls'],
        )

    def load(self, index, device='cpu'):
        row = self.rows[index]
        path = self.folder / row['file']
        require(digest(path) == row['sha256'], 'teacher goals changed after cache seal')
        arrays = _arrays(path, row['n'], row['contains_B_path'])
        return {branch: torch.from_numpy(arrays[branch]).to(device) for branch in ('A', 'B')}


@torch.no_grad()
def prepare_training(assets, output):
    """Atomically seal all teacher records before scoring the B scale subset."""
    output = Path(output)
    folder = output / 'teacher_cache'
    folder.mkdir(parents=True, exist_ok=True)
    indices = tuple(assets.split['train_indices'])
    chosen = scale_indices(assets.split)
    chosen_set = set(chosen)
    members = assets.split['members']
    train_members_sha = canonical_hash([members[i] for i in indices])
    scale_members_sha = canonical_hash([members[i] for i in chosen])
    identity = dict(schema='direct-k20-train-teachers-v1', fold=assets.fold,
                    teachers=assets.identity, purpose='train', seed=TEACHER_SEED,
                    M_teacher_B=256, K=20, unit='m', dtype='float32',
                    train_members_sha256=train_members_sha, scale_members_sha256=scale_members_sha,
                    scale_protocol='stable SHA rank of domain, fold and original ordered observation member',
                    source_sha256={rel: digest(SOURCE / rel) for rel in (
                        'direct_k20/assets.py', 'bcep_joint/inference.py', 'joint_flow_dependency/inference.py',
                        'joint_flow_dependency/sampling.py', 'bcep_v2/joint.py')})
    immutable(folder / 'IDENTITY.json', identity)
    identity_sha = canonical_hash(identity)
    previous = folder / 'MANIFEST.json'
    sealed = read(previous) if previous.exists() else None
    if sealed is not None:
        require(sealed['identity'] == identity and len(sealed['rows']) == len(indices),
                'sealed teacher manifest changed')
    states = assets.frozen_states()
    rows = []
    start = time.monotonic()
    prior_generation_seconds = 0.
    for ordinal, index in enumerate(indices, 1):
        member = members[index]
        n = len(member['agent_ids'])
        path = folder / f'{index:06d}.npz'
        sidecar = path.with_suffix('.json')
        row = dict(index=index, n=n, file=path.name, member_sha256=canonical_hash(member),
                   record_sha256=assets.split['records'][index]['sha256'],
                   identity_sha256=identity_sha, contains_B_path=index in chosen_set)
        if sidecar.exists():
            expected = read(sidecar)
            require(expected == (row | dict(sha256=digest(path), generation_seconds=expected['generation_seconds'])),
                    'teacher cache receipt or payload changed')
            row = expected
            prior_generation_seconds += row['generation_seconds']
        else:
            require(sealed is None, 'sealed teacher record is incomplete')
            frame_start = time.monotonic()
            values = _teacher_arrays(assets, index, index in chosen_set)
            if path.exists():
                orphan = _arrays(path, n, index in chosen_set)
                require(all(np.array_equal(orphan[k], v) for k, v in values.items()),
                        'orphaned teacher record differs from replay')
            else:
                temporary = path.with_suffix('.tmp.npz')
                np.savez(temporary, **values)
                temporary.replace(path)
            _arrays(path, n, index in chosen_set)
            row.update(sha256=digest(path), generation_seconds=time.monotonic() - frame_start)
            write(sidecar, row)
        if sealed is not None:
            require(sealed['rows'][ordinal-1] == row, 'teacher record differs from global seal')
        rows.append(row)
        if ordinal % 100 == 0 or ordinal == len(indices):
            status = dict(phase='TRAIN_TEACHER_CACHE', fold=assets.fold, windows=ordinal,
                          total=len(indices), seconds=time.monotonic() - start)
            write(folder / 'PROGRESS.json', status)
            print(status, flush=True)
    require(assets.frozen_states() == states, 'teacher state changed during cache export')
    if sealed is None:
        sealed = dict(identity=identity, rows=rows, seal_unix=time.time(), target_reads_during_export=0,
                      A_ddim_calls=0, scale_B_ddim_calls=len(chosen),
                      scale_indices=list(chosen),
                      cache_wall_seconds=prior_generation_seconds + time.monotonic() - start)
        immutable(previous, sealed)
    require(sealed['scale_indices'] == list(chosen), 'train scale sample differs from frozen plan')
    scale_file = folder / 'TRAIN_B_SCALES.json'
    teacher_sha = digest(previous)
    if scale_file.exists():
        receipt = read(scale_file)
        require(receipt['teachers_manifest_sha256'] == teacher_sha and
                receipt['members_sha256'] == scale_members_sha and receipt['scored_windows'] == len(chosen),
                'normalization receipt differs from the sealed teacher cache')
    else:
        score_start = time.monotonic()
        first_target_read = time.time()
        require(first_target_read >= sealed['seal_unix'], 'normalization opened targets before teacher freeze')
        by_index = {row['index']: row for row in rows}
        metric = MetricAccumulator()
        for index in chosen:
            entry = by_index[index]
            target = assets.batch(index, 'train', truth=True, device='cpu')
            require(target.valid.all() and int(target.valid.sum()) == entry['n'], 'train scale target misaligned')
            path = folder / entry['file']
            require(digest(path) == entry['sha256'], 'train-scale prediction changed')
            prediction = torch.from_numpy(_arrays(path, entry['n'], True)['B_path'])[None]
            metric.update_statistics(hard_metric_statistics(prediction, target.future, target.valid))
        stats = metric.as_dict()
        raw = {name: float(stats[name]) for name in METRICS}
        require(stats['scene_count'] == len(chosen) and all(math.isfinite(v) for v in raw.values()),
                'train B scale score incomplete')
        receipt = dict(seed=TEACHER_SEED, purpose='train', members_sha256=scale_members_sha,
                       scored_windows=len(chosen), scored_agents=stats['agent_count'],
                       teachers_manifest_sha256=teacher_sha, normalizer_floor=1e-6, metrics_m=raw,
                       scales={name: max(value, 1e-6) for name, value in raw.items()},
                       first_target_read_unix=first_target_read, score_seconds=time.monotonic()-score_start)
        immutable(scale_file, receipt)
    return TeacherCache(folder, sealed, receipt)

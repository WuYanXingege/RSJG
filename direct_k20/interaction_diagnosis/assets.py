"""Read-only access to the original fold, sealed trajectories and frozen weights."""

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch

from bcep.contracts import require
from bcep.motion_frame import frame
from bcep_v2.provider import GEOMETRY
from direct_k20.common import ATOL, RTOL, digest, immutable, read
from direct_k20.endpoint_interaction import EndpointInteraction
from direct_k20.evaluate import _read_prediction
from direct_k20.interaction_eval import DEV, TEST, _fold_protocol, _gate, _original_s0, _output, _row
from direct_k20.interaction_run import DOC as ARCHIVE, RUN as ORIGINAL, _load, _split
from direct_k20.model import decode_student, encode_observed, flow_world, student_noise


FOLDS = ('hotel', 'univ')
PURPOSES = (TEST, DEV)
MODELS = ('S0', 'R12')
EXPECTED_WINDOWS = {TEST: {'hotel': 445, 'univ': 947}, DEV: {'hotel': 688, 'univ': 622}}


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def read_arrays(path, shapes):
    with np.load(path, allow_pickle=False) as saved:
        require(set(saved.files) == set(shapes), 'unexpected prediction tensor names')
        arrays = {name: saved[name] for name in shapes}
    require(all(array.shape == shapes[name] and array.dtype == np.float32
                and np.isfinite(array).all() for name, array in arrays.items()),
            'invalid shape, dtype or coordinates in prediction')
    return arrays


@dataclass
class SealedFold:
    fold: str
    purpose: str
    split: dict
    identity: dict
    indices: list
    seeds: list
    rows: dict
    folder: Path
    cache_folder: Path | None
    cache: dict
    source_hashes: dict
    recovery_folder: Path

    def pair(self, index, seed, reproduce=None):
        row = self.rows[index, seed]
        n = row['n']
        source = self.folder / row['file']
        shape = (20, n, 12, 2)
        predictions = {}
        if source.is_file():
            require(digest(source) == row['sha256'], 'paired trajectory shard changed')
            members = ('R12',) if row['S0_reused'] else MODELS
            predictions.update(read_arrays(source, dict.fromkeys(members, shape)))
        if row['S0_reused']:
            historic = self.cache[index, seed]
            prior = self.cache_folder / historic['file']
            require(historic['sha256'] == row['S0_sha256'], 'S0 outside linked original seal')
            if prior.is_file():
                require(digest(prior) == row['S0_sha256'], 'historical S0 trajectory changed')
                predictions['S0'] = _read_prediction(prior, n)
        recoveries = []
        for model in MODELS:
            if model in predictions:
                continue
            path = self.recovery_folder / f'seed_{seed}' / f'{row["window"]}_{model}.npz'
            side = path.with_suffix('.json')
            expected = {key: row[key] for key in ('window', 'seed', 'member_sha256', 'protocol_sha256')}
            expected.update(model=model, original_pair_sha256=row['sha256'],
                            original_cache_sha256=row['S0_sha256'])
            if side.is_file():
                saved = read(side)
                require(saved == expected | dict(sha256=digest(path), seconds=saved['seconds']),
                        'recovered trajectory identity changed')
            else:
                require(reproduce is not None, 'missing pose shard must be recovered before scoring')
                started = time.monotonic()
                value = reproduce(model, seed, row['member_sha256'])
                if path.is_file():
                    require(np.array_equal(read_arrays(path, {model: shape})[model], value),
                            'interrupted recovery differs on replay')
                else:
                    save_npz(path, **{model: value})
                saved = expected | dict(sha256=digest(path), seconds=time.monotonic()-started)
                immutable(side, saved)
            predictions.update(read_arrays(path, {model: shape}))
            recoveries.append(saved)
        return predictions, recoveries


def reference_fold(fold, purpose, run):
    identity_path = ORIGINAL / fold / 'R12_IDENTITY.json'
    identity = read(identity_path)
    archived = read(ARCHIVE / fold / 'R12_IDENTITY.json')
    require(all(identity[key] == archived[key] for key in
                ('S0_sha256', 'S0_epoch', 'R12_sha256', 'dev_manifest_sha256', 'split_sha256')),
            'frozen checkpoint or split differs from fixed archive')
    split, manifest = _split(fold)
    require(manifest == identity['upstream_sha256'], 'upstream manifest changed')
    barrier = _gate(ORIGINAL, fold, identity, purpose)
    indices, protocol = _fold_protocol(split, identity, purpose, barrier)
    require(len(indices) == EXPECTED_WINDOWS[purpose][fold], 'unexpected locked fold population')
    cache_folder, cache, cache_sha = (
        _original_s0(split, indices, identity, protocol) if purpose == TEST else (None, {}, None))
    protocol['historical_S0_seal_sha256'] = cache_sha
    output = _output(ORIGINAL, purpose)
    archived_dir = ARCHIVE / ('test' if purpose == TEST else 'dev')
    for filename in ('SCORE_RECEIPT.json', 'PER_WINDOW.csv', 'PER_SEED.csv', 'ALL_PREDICTIONS_SEALED.json'):
        require(digest(output / filename) == digest(archived_dir / filename),
                f'archived {purpose} {filename} changed')
    scoring = read(output / 'SCORE_RECEIPT.json')
    global_seal = output / 'ALL_PREDICTIONS_SEALED.json'
    require(scoring['prediction_seal_sha256'] == digest(global_seal)
            and scoring['status'] == 'COMPLETE'
            and scoring['prediction_seal_unix'] <= scoring['first_truth_read_unix'],
            'original scoring provenance differs')
    folder = output / fold
    local = folder / 'PREDICTIONS_FROZEN.json'
    require(digest(local) == read(global_seal)['folds'][fold]['seal_sha256'],
            'individual original prediction seal changed')
    seal = read(local)
    require(seal['protocol'] == protocol == read(folder / 'EXPORT_CONFIG.json')
            and seal['truth_reads'] == 0 and seal['windows'] == len(indices)
            and seal['prediction_pairs'] == len(seal['rows']) == len(indices) * len(protocol['seeds']),
            'source seal membership, seed or geometry changed')
    rows = {}
    ordered = iter(seal['rows'])
    for ordinal, index in enumerate(indices, 1):
        for seed in protocol['seeds']:
            expected = _row(split, protocol, ordinal, index, seed)
            actual = next(ordered)
            require(all(actual[key] == value for key, value in expected.items()), 'original window order or ID changed')
            if actual['S0_reused']:
                require(cache[index, seed]['sha256'] == actual['S0_sha256'], 'historical S0 row is not aligned')
            rows[index, seed] = actual
    source_hashes = dict(pair_seal_sha256=digest(local), score_receipt_sha256=digest(output/'SCORE_RECEIPT.json'),
                         global_pair_seal_sha256=digest(global_seal), S0_cache_seal_sha256=cache_sha)
    return SealedFold(fold, purpose, split, identity, indices, protocol['seeds'], rows,
                      folder, cache_folder, cache, source_hashes, run / 'recovered' / purpose / fold)


def frozen_assets(reference, device):
    assets, student = _load(reference.fold, reference.identity, device)
    ident = reference.identity
    require(digest(ident['R12_checkpoint']) == ident['R12_sha256'], 'R12 checkpoint differs')
    saved = torch.load(ident['R12_checkpoint'], map_location='cpu', weights_only=False)
    require(saved['completed_epoch'] == 12, 'R12 completed epoch differs')
    refiner = EndpointInteraction()
    refiner.load_state_dict(saved['interaction'], strict=True)
    refiner.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    return assets, student, refiner


@torch.no_grad()
def native_endpoints(assets, student, refiner, batch):
    observed, maps, valid = batch.observed[0], batch.semantic_maps[0], batch.valid[0]
    require(bool(valid.all()) and not bool(batch.future.count_nonzero()),
            'endpoint export received anything beyond packed observations')
    context, features = encode_observed(assets.provider, batch)
    raw = flow_world(assets.provider, observed, context, student(observed, context))
    g0, _ = assets.provider.head(assets.provider.base, observed, maps, valid, raw,
                                 context, features, dict(layout=GEOMETRY))
    dt = float(assets.split['dt'])
    g1, unary, social = refiner(g0, observed, context, valid, dt=dt, return_components=True)
    _, rotation, _ = frame(observed, 12, dt, observed.new_ones(()))
    unary_world = (rotation @ unary.unsqueeze(-1)).squeeze(-1)
    social_world = (rotation @ social.unsqueeze(-1)).squeeze(-1)
    if len(observed) == 1:
        require(not bool(social_world.count_nonzero()), 'singleton social term is not zero')
    actual = g1.double()-g0.double()
    residual = (actual-unary_world.double()-social_world.double()).abs()
    # The world-coordinate add can lose a few low bits of the displacement.
    tolerance = torch.finfo(torch.float32).eps * (
        g0.double().abs()+g1.double().abs()+unary_world.double().abs()+social_world.double().abs()) + 1e-7
    require(bool((residual <= tolerance).all()), 'unary and social do not compose the native correction')
    arrays = {name: value.cpu().numpy() for name, value in
              zip(('raw', 'g0', 'g1', 'unary_world', 'social_world'),
                  (raw, g0, g1, unary_world, social_world))}
    return arrays, float(residual.max())


def replay_missing(assets, batch, endpoints, model, seed, window_key):
    """Use unchanged native observed context and the original keyed noise."""
    with torch.no_grad():
        bridge = assets.model.encode_context(batch.observed, batch.semantic_maps, batch.valid)
        goals = torch.from_numpy(endpoints['g0' if model == 'S0' else 'g1']).to(assets.device)
        noise = student_noise(window_key, batch.agent_ids[0], seed, goals.device)
        return decode_student(assets.model, bridge, goals, noise)[0].cpu().numpy()

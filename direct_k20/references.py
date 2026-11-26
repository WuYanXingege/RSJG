"""Bind the five frozen A/B prediction exports to their published receipts.

Discovery reads only manifests, checkpoint hashes, baseline CSVs and file
metadata. Baseline scores are bound to the official source seals and stable members.
No future trajectories are loaded here.
"""
import csv
from pathlib import Path

from bcep.contracts import canonical_hash, digest, require
from bcep_joint.common import read

SOURCE = Path(__file__).resolve().parents[1]
ASSETS = SOURCE.parent / 'RSJG_EBJD_assets/experiments'
REFERENCE_COMMIT = '3296fc5968463ed14bf43265c064b3856d55ed30'
FOLDS = ('zara1', 'hotel', 'univ', 'eth', 'zara2')
SEEDS = tuple(range(2035, 2040))
PURPOSE = 'heldout_confirmation'
ARCHIVES = SOURCE / 'docs/research/continuous_joint_flow'
FORMAL = ASSETS / 'joint_flow_fourfold_confirmation_20261122'
ARCHIVE = ARCHIVES / 'fourfold_confirmation_20261122'


def _locked_split(path, file_sha, canonical_sha):
    require(digest(path) == file_sha, 'split lock file changed')
    split = read(path)
    require(canonical_hash(split) == canonical_sha, 'split content changed')
    return split


def _new_fold(fold, identity, barrier):
    directory = FORMAL / fold
    selected_path = directory / 'B_FROZEN.json'
    upstream_path = directory / 'UPSTREAM_FROZEN.json'
    selected = read(selected_path)
    upstream = read(upstream_path)
    indexed = barrier['folds'][fold]
    expected = read(ARCHIVE / 'FROZEN_MODELS.json')['folds'][fold]
    require(identity['fold'] == selected['fold'] == upstream['fold'] == fold,
            f'{fold} model identity mismatch')
    require(identity['freeze_barrier'] == canonical_hash(barrier), f'{fold} freeze barrier changed')
    require(digest(selected_path) == indexed['B_manifest_sha256'] == expected['B_manifest_sha256'],
            f'{fold} selected B record changed')
    require(digest(upstream_path) == indexed['upstream_manifest_sha256'] == expected['upstream_manifest_sha256'],
            f'{fold} upstream record changed')
    a, b = upstream['joint'], selected
    a_sha = identity['upstream']['checkpoint_sha256']
    b_sha = identity['B_sha256']
    require(a_sha == a['sha256'] == indexed['A_sha256'] == expected['A_sha256'] == digest(a['checkpoint']),
            f'{fold} A checkpoint differs')
    require(b_sha == b['rank1_sha256'] == indexed['B_sha256'] == expected['B_sha256'] == digest(b['rank1_checkpoint']),
            f'{fold} B checkpoint differs')
    a_epoch = a['frozen']['epoch']
    b_epoch = b['rank1_epoch']
    require((a_epoch, b_epoch) == (expected['A_epoch'], expected['B_epoch']) ==
            (indexed['A_epoch'], indexed['B_epoch']) == (a_epoch, identity['B_epoch']),
            f'{fold} selected epoch differs')
    split = _locked_split(upstream['split_path'], upstream['split_sha256'], identity['upstream']['split_sha256'])
    require(upstream['split_sha256'] == identity['upstream']['split_file_sha256'], f'{fold} split identity changed')
    return split, (a_epoch, a_sha, b_epoch, b_sha), identity['heldout_members']


def _historical_zara2(identity):
    formal = ASSETS / 'bcep_ebjd_v2_integrated'
    original = read(formal / 'FORMAL_RUN.json')
    plan = read(original['input_plan'])
    split_descriptor = plan['folds']['zara2']['split']
    split = _locked_split(split_descriptor['path'], split_descriptor['sha256'], identity['split_sha256'])
    archive = ARCHIVE / 'zara2/test'
    historical = read(ARCHIVE / 'ZARA2_REUSE.json')
    expected = read(archive / 'RUN_CONFIG.json')['asset_identity']
    require(expected == identity and original['commit'] == identity['base'], 'historical ZARA2 identity changed')
    a_epoch, b_epoch = historical['A_epoch'], historical['B_epoch']
    require(a_epoch == 5 and b_epoch == 35 == identity['B_completed_epoch'], 'unexpected historical ZARA2 epochs')
    a_sha = identity['checkpoint_sha256']
    b_sha = identity['B_checkpoint_sha256']
    a_path = formal / f'zara2/joint/epoch_{a_epoch:03d}.pt'
    b_path = ASSETS / f'zara2_joint_flow_dependency_20261122/dependence/epoch_{b_epoch:03d}.pt'
    require(digest(a_path) == a_sha and digest(b_path) == b_sha, 'historical ZARA2 checkpoint differs')
    return split, (a_epoch, a_sha, b_epoch, b_sha), identity['heldout_member_hash']


def _reference(directory, archive, receipt):
    for filename in ('PER_SEED.csv', 'PER_WINDOW.csv'):
        require(digest(directory / filename) == receipt['output_sha256'][filename] == digest(archive / filename),
                f'{archive.name} {filename} differs from the published scores')
    with (archive / 'PER_SEED.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    baseline = {(int(row['seed']), row['group']): row for row in rows}
    require(len(rows) == len(baseline) == 10 and
            set(baseline) == {(s, g) for s in SEEDS for g in ('all', 'N_ge_9')},
            'archived baseline seed/group count differs')
    return baseline


def _discover_fold(fold, barrier):
    directory = (ASSETS / 'zara2_joint_flow_frozen_test_20261122' if fold == 'zara2'
                 else FORMAL / fold / 'test')
    archive = ARCHIVE / fold / 'test'
    seal_path = directory / 'PREDICTIONS_FROZEN.json'
    receipt_path = directory / 'SCORE_RECEIPT.json'
    config_path = directory / 'RUN_CONFIG.json'
    seal = read(seal_path)
    receipt = read(receipt_path)
    config = read(config_path)
    identity = seal['identity']
    require(digest(receipt_path) == digest(archive / 'SCORE_RECEIPT.json'),
            f'{fold} receipt hash differs from {REFERENCE_COMMIT}')
    require(receipt['status'] == 'COMPLETE' and receipt['all_predictions_verified_before_any_truth'],
            f'{fold} source scoring was not sealed')
    require(receipt['prediction_manifest_sha256'] == digest(seal_path) and
            receipt['prediction_manifest_canonical_sha256'] == canonical_hash(seal),
            f'{fold} source prediction seal changed')
    require(receipt['source_config_sha256'] == seal['source_config_sha256'] == digest(config_path),
            f'{fold} source config changed')
    require(identity == config.get('identity', config.get('asset_identity')) and
            receipt['seal_unix'] == seal['seal_unix'] <
            receipt['verification_complete_unix'] < receipt['first_truth_read_unix'],
            f'{fold} source seal is not bound to its config and scoring timeline')
    require(seal['seeds'] == receipt['seeds'] == list(SEEDS) and
            (seal['K'], seal['prediction_frames'], seal['prediction_dtype']) == (20, 12, 'float32') and
            seal['heldout_future_reads'] == 0, f'{fold} source prediction contract changed')
    if fold == 'zara2':
        split, clocks, member_sha = _historical_zara2(identity)
        public_seal = read(ARCHIVES / 'zara2_frozen_test_20261122/PREDICTIONS_FROZEN_PUBLIC.json')
    else:
        split, clocks, member_sha = _new_fold(fold, identity, barrier)
        public_seal = read(archive / 'PREDICTIONS_FROZEN_PUBLIC.json')
    require(public_seal['original_manifest_sha256'] == receipt['prediction_manifest_sha256'],
            f'{fold} private source seal differs from the public one')
    require(split['fold'] == fold, f'{fold} source split changed')
    indices = tuple(split[PURPOSE + '_indices'])
    ordered_members = [split['members'][index] for index in indices]
    require(canonical_hash(ordered_members) == member_sha, f'{fold} ordered window/agent/frame membership changed')
    counts = {}
    for member in ordered_members:
        agent_ids = member['agent_ids']
        require(len(agent_ids) == len(set(agent_ids)) and len(agent_ids) > 0, f'{fold} invalid source agent list')
        require(member['frame_ids'] == sorted(set(member['frame_ids'])), f'{fold} invalid source frame order')
        counts[len(agent_ids)] = counts.get(len(agent_ids), 0) + 1
    n_windows = len(indices)
    keys = {(seed, index) for seed in SEEDS for index in indices}
    require(n_windows == len(set(indices)) == seal['windows'] == receipt['windows'] and
            len(keys) == len(seal['rows']) == len(public_seal['rows']) == receipt['prediction_files_verified'],
            f'{fold} source predictions are incomplete')
    public_rows = {(row['seed'], row['window']): row for row in public_seal['rows']}
    require(len(public_rows) == len(keys), f'{fold} public manifest duplicates prediction')
    position = {index: pos for pos, index in enumerate(indices, 1)}
    case = (canonical_hash(dict(identity=identity, source_config_sha256=seal['source_config_sha256']))
            if fold == 'zara2' else canonical_hash(identity))
    rows = {}
    missing = []
    for row in seal['rows']:
        seed, index = row['seed'], row['index']
        key = (seed, index)
        require(key in keys and key not in rows, f'{fold} unexpected or duplicate source shard')
        window = f'test_{position[index]:04d}'
        n = len(split['members'][index]['agent_ids'])
        path = f'predictions/seed_{seed}/{index}.npz'
        require((row['window'], row['n'], row['file'], row['identity']) == (window, n, path, case),
                f'{fold}/{seed}/{window} source shard is misaligned')
        require(public_rows[seed, window] ==
                dict(seed=seed, window=window, n=n, prediction_sha256=row['sha256']),
                f'{fold}/{seed}/{window} source hash differs from public archive')
        if not (directory / row['file']).is_file():
            missing.append(key)
        rows[key] = row
    baseline = _reference(directory, archive, receipt)
    totals = {
        'all': (n_windows, sum(n * c for n, c in counts.items())),
        'N_ge_9': (sum(c for n, c in counts.items() if n >= 9),
                   sum(n * c for n, c in counts.items() if n >= 9)),
    }
    for (seed, group), row in baseline.items():
        require((int(row['scene_count']), int(row['agent_count'])) == totals[group],
                f'{fold}/{seed}/{group} baseline support differs')
    a_epoch, a_sha, b_epoch, b_sha = clocks
    metadata = dict(A_epoch=a_epoch, A_sha256=a_sha, B_epoch=b_epoch, B_sha256=b_sha,
                    windows=n_windows, agents=totals['all'][1], source_shards=len(rows),
                    missing_shards=len(missing), ordered_members_sha256=member_sha,
                    source_config_sha256=seal['source_config_sha256'],
                    source_seal_sha256=receipt['prediction_manifest_sha256'],
                    source_receipt_sha256=digest(receipt_path),
                    baseline_per_seed_sha256=receipt['output_sha256']['PER_SEED.csv'],
                    baseline_per_window_sha256=receipt['output_sha256']['PER_WINDOW.csv'])
    return dict(split=split, indices=indices, rows=rows, directory=directory, baseline=baseline,
                metadata=metadata, missing=missing)


def discover():
    """Return verified assets per fold, including any absent (seed, index) shard keys."""
    delivery = read(FORMAL / 'DELIVERY.json')
    require(delivery['verified_remote_sha'] == REFERENCE_COMMIT, 'formal evaluation commit differs')
    barrier = read(FORMAL / 'FROZEN_MODELS.json')
    require(barrier['status'] == 'ALL_AVAILABLE_MODELS_FROZEN_BEFORE_TEST', 'source evaluation did not freeze all models')
    return {fold: _discover_fold(fold, barrier) for fold in FOLDS}

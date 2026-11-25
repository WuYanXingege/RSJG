"""Replay the native HOTEL bridge for fixed checkpoints and sealed K20 goals."""

import math
from pathlib import Path
import time

import torch

from bcep.contracts import canonical_hash, digest, require
from bcep_joint.common import read, write
from joint_flow_fourfold.common import immutable
from joint_flow_fourfold.inputs import state_hash
from joint_flow_fourfold.scoring import csvwrite
from k20_dual_finetune.evaluate import (
    NATIVE_METRICS,
    PURPOSE,
    SEED,
    SCOPE as ORIGINAL_SCOPE,
    _csv,
    _export,
    _groups,
    _history,
    _score,
    _summary_csv,
    _verify_seal,
)


SCOPE = '看到 HOTEL 第 8 轮终点双降后追加的单种子探索性开发评估'
BASE_ARCHIVE = (
    Path(__file__).resolve().parents[1]
    / 'docs/research/continuous_joint_flow/k20_dual_finetune_20261125/hotel'
)
EPOCH8_SHA256 = '863d169a44122fbe40668d2299202b39d605dbae708a62f2da8e2c20eb6797cc'
GROUP_COUNTS = {'all': (445, 1197), 'single': (144, 144), 'multi': (301, 1053)}


def _verified_epoch8_endpoints(inputs, original, identity):
    """Pin every reused endpoint to the original evaluation and archive."""
    require(identity['checkpoint_sha256'] == EPOCH8_SHA256, 'unexpected epoch-eight checkpoint')
    directory = original / 'endpoints'
    summary = read(original / 'SUMMARY.json')
    gate = read(original / 'ENDPOINT_GATE.json')
    receipt_path = directory / 'SCORE_RECEIPT.json'
    receipt = read(receipt_path)
    for document, archived_name in (
        (summary, 'SUMMARY.json'),
        (gate, 'ENDPOINT_GATE.json'),
        (receipt, 'ENDPOINT_SCORE_RECEIPT.json'),
    ):
        require(document == read(BASE_ARCHIVE / archived_name),
                f'original HOTEL {archived_name} differs from its committed archive')
    old_identity = dict(identity, scope=ORIGINAL_SCOPE)
    require(summary['identity'] == old_identity and summary['evaluated_epoch'] == 8,
            'loaded model or ordered evaluation membership differs from the endpoint run')
    require(summary['status'] == 'COMPLETE' and summary['ddim_calls'] == 0 and
            summary['endpoint_gate_passed'] is False and gate['endpoint_gate_passed'] is False and
            gate['pass_minFDE'] is False and gate['pass_JFDE'] is True and
            summary['endpoint_groups'] == gate['groups'],
            'original endpoint decision differs from the recorded gate')
    require(digest(receipt_path) == gate['endpoint_score_receipt_sha256'] and
            receipt == summary['endpoint_receipt'], 'original endpoint score receipt changed')
    seal = read(directory / 'PREDICTIONS_FROZEN.json')
    verified = _verify_seal(inputs.split, seal, directory, old_identity, 'endpoint')
    require(verified['manifest_sha256'] == receipt['manifest_sha256'] and
            receipt['seal_unix'] == seal['seal_unix'] and
            receipt['seal_unix'] <= receipt['completed_unix'] <= receipt['first_truth_read_unix'] and
            receipt['status'] == 'COMPLETE' and receipt['phase'] == 'endpoint' and receipt['seed'] == SEED and
            receipt['files'] == receipt['truth_reads'] == GROUP_COUNTS['all'][0],
            'original endpoint seal or scoring chronology does not match its receipt')
    table = directory / 'PER_WINDOW.csv'
    require(digest(table) == receipt['table_sha256'] ==
            summary['output_sha256']['PER_WINDOW.csv'] == digest(BASE_ARCHIVE / 'PER_WINDOW.csv'),
            'original endpoint window scores differ from the archived receipt')
    return directory, seal, dict(verified, original_receipt_sha256=digest(receipt_path),
                                 original_table_sha256=digest(table),
                                 original_gate_sha256=digest(original / 'ENDPOINT_GATE.json'),
                                 original_endpoint_groups=summary['endpoint_groups'])


def _check_endpoint_replay(rows, archived_rows, groups, previous_groups):
    """Reconcile final-frame metrics against the archived endpoint-only scores."""
    require(len(rows) == len(archived_rows) == GROUP_COUNTS['all'][0],
            'original and replayed endpoint window counts differ')
    max_drift = {metric: 0. for metric in ('JFDE', 'minFDE')}
    for current, prior in zip(rows, archived_rows):
        require(current['window'] == prior['window'] and current['n'] == int(prior['n']) and
                current['seed'] == int(prior['seed']),
                'decoded HOTEL window order differs from the archive')
        for metric in max_drift:
            value = current['new_' + metric]
            expected = float(prior['new_' + metric])
            require(math.isclose(value, expected, rel_tol=1e-6, abs_tol=1e-6),
                    f'replayed {metric} differs from the original window score')
            max_drift[metric] = max(max_drift[metric], abs(value - expected))
    for group in GROUP_COUNTS:
        for metric in max_drift:
            require(math.isclose(groups[group]['new'][metric], previous_groups[group]['new'][metric],
                                 rel_tol=1e-6, abs_tol=1e-6),
                    f'replayed group {metric} differs from the original endpoint summary')
    return max_drift


@torch.no_grad()
def full_evaluation(inputs, model, provider, dependence, output, checkpoint_sha256, epoch,
                    historical, *, old_evaluation=None, device='cuda'):
    """Score full trajectories at epoch 8 or 16 without the endpoint gate."""
    require(inputs.fold == 'hotel' and epoch in (8, 16), 'invalid follow-up fold or epoch')
    require((epoch == 8) == (old_evaluation is not None),
            'only epoch eight reuses the original sealed endpoints')
    start = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    member_ids = inputs.split[PURPOSE + '_indices']
    members = [inputs.split['members'][index] for index in member_ids]
    require((len(members), sum(len(member['agent_ids']) for member in members)) == GROUP_COUNTS['all'],
            'HOTEL follow-up population differs')
    members_sha256 = canonical_hash(members)
    reference, historical_windows, historical_identity = _history(inputs, Path(historical), members_sha256)
    modules = dict(model=model, provider=provider, dependence=dependence)
    for module in modules.values():
        module.eval()
    hashes = {name: state_hash(module) for name, module in modules.items()}
    identity = dict(fold='hotel', seed=SEED, epoch=epoch, checkpoint_sha256=checkpoint_sha256,
                    purpose=PURPOSE, scope=SCOPE, split_members_sha256=members_sha256,
                    output_state_sha256=hashes, **historical_identity)
    immutable(output / 'EVALUATION_CONFIG.json', dict(
        identity=identity, M=256, K=20, reuse_original_endpoint_seal=epoch == 8,
        complete_trajectory_scoring=True, trajectory_training_updates=0,
        bridge=dict(ddim_steps=20, eta=0., noise='observed scene, ordered agents, original raw IDs and seed'),
        marginal_aggregation='agent-weighted', joint_aggregation='window-weighted',
    ))
    summary_path = output / 'SUMMARY.json'
    trajectory_dir = output / 'trajectories'
    if summary_path.exists():
        summary = read(summary_path)
        require(summary['status'] == 'COMPLETE' and summary['identity'] == identity and
                summary['ddim_calls'] == len(members), 'completed evaluation identity changed')
        require(summary['output_sha256'] == {name: digest(output / name)
                for name in ('PER_WINDOW.csv', 'SUMMARY.csv')}, 'completed evaluation table changed')
        receipt = read(trajectory_dir / 'SCORE_RECEIPT.json')
        require(receipt == summary['trajectory_receipt'] and
                receipt['manifest_sha256'] == digest(trajectory_dir / 'PREDICTIONS_FROZEN.json') and
                receipt['table_sha256'] == digest(trajectory_dir / 'PER_WINDOW.csv'),
                'completed trajectory scoring receipt changed')
        return summary
    if epoch == 8:
        endpoint_dir, endpoints, verified_endpoints = _verified_epoch8_endpoints(
            inputs, Path(old_evaluation), identity)
    else:
        endpoint_dir = output / 'endpoints'
        endpoints = _export(inputs, model, provider, dependence, endpoint_dir, identity, device=device)
        verified_endpoints = _verify_seal(inputs.split, endpoints, endpoint_dir, identity, 'endpoint')
    write(output / 'ENDPOINT_VERIFICATION.json', verified_endpoints)
    decoder_identity = dict(identity, endpoint_seal_sha256=verified_endpoints['manifest_sha256'])
    trajectories = _export(inputs, model, provider, dependence, trajectory_dir, decoder_identity,
                           endpoint_directory=endpoint_dir, device=device)
    require(trajectories['ddim_calls'] == len(members) and trajectories['exporter_truth_reads'] == 0,
            'incomplete observed-only HOTEL trajectory seal')
    rows, receipt = _score(inputs, trajectory_dir, trajectories, decoder_identity,
                           historical_windows, endpoints=endpoint_dir)
    groups = _groups('hotel', rows, reference)
    for group, counts in GROUP_COUNTS.items():
        require((groups[group]['scene_count'], groups[group]['agent_count']) == counts,
                f'HOTEL {group} population differs')
    if epoch == 8:
        replay_drift = _check_endpoint_replay(
            rows, _csv(endpoint_dir / 'PER_WINDOW.csv'), groups,
            verified_endpoints['original_endpoint_groups'],
        )
    else:
        replay_drift = None
    require(hashes == {name: state_hash(module) for name, module in modules.items()},
            'trajectory evaluation changed model parameters')
    csvwrite(output / 'PER_WINDOW.csv', rows)
    _summary_csv(output / 'SUMMARY.csv', 'hotel', groups)
    summary = dict(
        status='COMPLETE', scope=SCOPE, fold='hotel', evaluated_epoch=epoch, seed=SEED,
        identity=identity, groups=groups,
        all_four_improved=all(groups['all']['delta'][metric] < 0. for metric in NATIVE_METRICS),
        evaluation_training_updates=0, ddim_calls=trajectories['ddim_calls'],
        endpoint_seal_reused=epoch == 8, endpoint_verification=verified_endpoints,
        endpoint_score_max_abs_replay_drift_m=replay_drift,
        endpoint_export_wall_seconds=0. if epoch == 8 else endpoints['wall_seconds'],
        trajectory_export_wall_seconds=trajectories['wall_seconds'],
        trajectory_receipt=receipt, states_unchanged=True,
        wall_seconds_this_attempt=time.monotonic() - start,
        output_sha256={name: digest(output / name) for name in ('PER_WINDOW.csv', 'SUMMARY.csv')},
    )
    write(output / 'SUMMARY.json', summary)
    return summary

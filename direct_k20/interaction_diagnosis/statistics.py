"""Window-paired diagnostics for frozen K20 endpoints and sealed trajectories."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import math

import numpy as np


STAGES = ('raw', 'g0', 'g1')
ENDPOINT_METRICS = ('J', 'L', 'G', 'minFDE')
PATH_METRICS = ('JADE', 'JFDE', 'minADE', 'minFDE')
SEEDS = (2035, 2036, 2037, 2038, 2039)
FRAME_NUMBERS = (3, 6, 9, 12)
PAIR_DISTANCE_M = 2.
RANDOM_PARTITIONS = 20
PARTITION_SEED = 3201
AXIS_SPEED_MPS = .001  # Native motion_frame heading fallback.
LOW_MOTION_MPS = .2  # Original HOTEL endpoint diagnostic.
BOUND_ATOL_M = 1e-12
MOTION_BUCKETS = ('stay_slow', 'start', 'stop', 'keep_moving')
BASE_GROUPS = ('all', 'N1', 'N2_8', 'N9plus')
TOP_GROUPS = ('top10', 'remaining90', 'top10_N1', 'top10_N2_8', 'top10_N9plus')
MULTI_GROUPS = ('paired', 'partition_informative', 'partition_uninformative')


@dataclass
class WindowDiagnosis:
    """Anonymous scalar row, private pooled samples, and private case winners."""

    row: dict
    samples: dict
    winners: dict


def _array(value, shape, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f'invalid {name}, expected {shape}, got {value.shape}')
    return value


def group_for_agents(n):
    return 'N1' if n == 1 else 'N2_8' if n <= 8 else 'N9plus'


def demography(split):
    """Return exact N histograms without accessing any trajectory or identifier."""
    rows = []
    for purpose in ('train', 'selection_dev', 'heldout_confirmation'):
        counts = Counter(len(split['members'][i]['agent_ids']) for i in split[purpose + '_indices'])
        for n, windows in sorted(counts.items()):
            rows.append(dict(fold=split['fold'], split=purpose, n=n, group=group_for_agents(n),
                             windows=windows, agents=n * windows))
    return rows


def endpoint_scores(goals, truth):
    """Hard means and winners over one meter-space [K,N,2] endpoint array."""
    errors = np.linalg.norm(goals - truth[None], axis=-1)
    costs = errors.mean(axis=1)
    individual_slots = np.argmin(errors, axis=0)
    marginal = errors[individual_slots, np.arange(len(truth))]
    joint_slot = int(np.argmin(costs))
    joint, separate = float(costs[joint_slot]), float(marginal.mean())
    return dict(J=joint, L=separate, G=joint-separate, minFDE=separate), errors, costs, joint_slot, individual_slots


def greedy_observed_pairs(observed):
    """Select disjoint pairs ordered by their mean distance over the final three observations."""
    n = len(observed)
    lhs, rhs = np.triu_indices(n, 1)
    distance = np.linalg.norm(observed[lhs, -3:] - observed[rhs, -3:], axis=-1).mean(axis=1)
    order = np.lexsort((rhs, lhs, distance))
    taken = np.zeros(n, dtype=bool)
    pairs = []
    for edge in order:
        if distance[edge] > PAIR_DISTANCE_M:
            break
        i, j = int(lhs[edge]), int(rhs[edge])
        if not taken[i] and not taken[j]:
            pairs.append((i, j))
            taken[[i, j]] = True
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def partition_jfde(errors, pairs):
    """Allow each observed pair or singleton one independent shared-slot selection."""
    n = errors.shape[1]
    singleton = np.ones(n, dtype=bool)
    singleton[pairs.ravel()] = False
    pair_minima = (errors[:, pairs[:, 0]] + errors[:, pairs[:, 1]]).min(axis=0)
    return float((pair_minima.sum() + errors[:, singleton].min(axis=0).sum()) / n)


def _partition_rows(row, observed, errors, endpoints):
    n = len(observed)
    pairs = greedy_observed_pairs(observed)
    pair_count = len(pairs)
    informative = pair_count > 0 and n - pair_count > 1
    row.update(pairs=pair_count, paired_agents=2*pair_count, pair_coverage=2*pair_count/n,
               partition_has_pair=bool(pair_count), partition_informative=informative)
    nonce = '|'.join(str(row[key]) for key in ('fold', 'split', 'window')).encode()
    entropy = int.from_bytes(hashlib.sha256(nonce).digest()[:8], 'little')
    rng = np.random.default_rng([PARTITION_SEED, entropy])
    draws = [rng.permutation(n)[:2*pair_count].reshape(-1, 2) for _ in range(RANDOM_PARTITIONS)]
    for stage in STAGES:
        ep = endpoints[stage]
        partition = partition_jfde(errors[stage], pairs)
        random = np.array([partition_jfde(errors[stage], draw) for draw in draws])
        candidates = np.append(random, partition)
        if candidates.min() < ep['L'] - BOUND_ATOL_M or candidates.max() > ep['J'] + BOUND_ATOL_M:
            raise ArithmeticError('partition minima outside the marginal and scene bounds')
        row.update({
            f'{stage}_JP_m': partition,
            f'{stage}_J_minus_JP_m': ep['J'] - partition,
            f'{stage}_JP_minus_L_m': partition - ep['L'],
            f'{stage}_random_JP_m': float(random.mean()),
            f'{stage}_random_JP_sd_m': float(random.std(ddof=1)),
            f'{stage}_random_J_minus_JP_m': ep['J'] - float(random.mean()),
            f'{stage}_random_JP_minus_L_m': float(random.mean()) - ep['L'],
            f'{stage}_JP_minus_random_JP_m': partition - float(random.mean()),
        })


def _optional_mean(values):
    return float(values.mean()) if values.size else None


def _hotel_motion(row, observed, future, g0, joint_slot, individual_slots, dt):
    n = len(observed)
    past_delta = np.diff(observed, axis=1)
    past_speed = np.linalg.norm(past_delta, axis=-1).sum(axis=1) / (7 * dt)
    next_delta = np.diff(np.concatenate((observed[:, -1:], future), axis=1), axis=1)
    next_speed = np.linalg.norm(next_delta, axis=-1).sum(axis=1) / (12 * dt)
    prior_low, future_low = past_speed <= LOW_MOTION_MPS, next_speed <= LOW_MOTION_MPS
    row['low_speed_agents'] = int(prior_low.sum())
    row['future_low_speed_agents'] = int(future_low.sum())
    row['observed_speed_mean_mps'] = float(past_speed.mean())
    row['future_speed_mean_mps'] = float(next_speed.mean())
    recent_velocity = past_delta[:, -3:].mean(axis=1) / dt
    recent_speed = np.linalg.norm(recent_velocity, axis=-1)
    axis_valid = ~prior_low & (recent_speed > AXIS_SPEED_MPS)
    row['axis_agents'] = int(axis_valid.sum())
    row['no_axis_agents'] = n - row['axis_agents']
    longitudinal = recent_velocity[axis_valid] / recent_speed[axis_valid, None]
    lateral = np.stack((-longitudinal[:, 1], longitudinal[:, 0]), axis=1)
    cv = observed[:, -1] + 12 * dt * recent_velocity
    forecasts = dict(cv=cv, g0_joint=g0[joint_slot],
                     g0_person=g0[individual_slots, np.arange(n)])
    euclidean = {}
    for name, predicted in forecasts.items():
        difference = predicted - future[:, -1]
        euclidean[name] = np.linalg.norm(difference, axis=-1)
        row[f'{name}_no_axis_FDE_m'] = _optional_mean(euclidean[name][~axis_valid])
        forward = (difference[axis_valid] * longitudinal).sum(axis=1)
        sideways = (difference[axis_valid] * lateral).sum(axis=1)
        row.update({f'{name}_long_signed_m': _optional_mean(forward),
                    f'{name}_long_abs_m': _optional_mean(np.abs(forward)),
                    f'{name}_lat_signed_m': _optional_mean(sideways),
                    f'{name}_lat_abs_m': _optional_mean(np.abs(sideways))})
    truth_displacement = future[:, -1] - observed[:, -1]
    truth_displacement = truth_displacement[axis_valid]
    turning_valid = np.linalg.norm(truth_displacement, axis=1) > AXIS_SPEED_MPS * (12 * dt)
    change_forward = (truth_displacement * longitudinal).sum(axis=1)
    change_lateral = (truth_displacement * lateral).sum(axis=1)
    angle = np.degrees(np.abs(np.arctan2(change_lateral, change_forward)))[turning_valid]
    row['truth_angle_agents'] = int(turning_valid.sum())
    row['truth_turn_abs_deg'] = _optional_mean(angle)
    row['cv_FDE_m'] = float(euclidean['cv'].mean())
    row['cv_beats_g0_J_rate'] = float(row['cv_FDE_m'] < row['g0_J_m'])
    row['cv_beats_g0_L_rate'] = float(row['cv_FDE_m'] < row['g0_L_m'])
    row['g0_people_cv_better_rate'] = float((euclidean['cv'] < euclidean['g0_person']).mean())
    motion_types = (prior_low & future_low, prior_low & ~future_low,
                    ~prior_low & future_low, ~prior_low & ~future_low)
    for bucket, selected in zip(MOTION_BUCKETS, motion_types):
        stem = f'motion_{bucket}'
        row[f'{stem}_agents'] = int(selected.sum())
        for name, values in euclidean.items():
            row[f'{stem}_{name}_FDE_m'] = _optional_mean(values[selected])


def _path_by_seed(paths, n, name):
    if isinstance(paths, dict):
        mapping = paths
    else:
        paths = np.asarray(paths)
        if paths.ndim == 4:
            paths = paths[None]
        if len(paths) not in (1, len(SEEDS)):
            raise ValueError('expected seed 2035 or all five sealed seeds')
        mapping = dict(zip(SEEDS, paths))
    if set(mapping) not in ({SEEDS[0]}, set(SEEDS)):
        raise ValueError(f'invalid {name} seeds')
    return {seed: _array(array, (20, n, 12, 2), name) for seed, array in sorted(mapping.items())}


def _path_row(path, truth):
    residual = path - truth[None]
    errors = np.linalg.norm(residual, axis=-1)
    ade, fde = errors.mean(axis=-1), errors[..., -1]
    scene_ade, scene_fde = ade.mean(axis=1), fde.mean(axis=1)
    winners = dict(JADE=int(scene_ade.argmin()), JFDE=int(scene_fde.argmin()),
                   person_minFDE=fde.argmin(axis=0))
    values = dict(JADE_m=float(scene_ade.min()), JFDE_m=float(scene_fde.min()),
                  minADE_m=float(ade.min(axis=0).mean()), minFDE_m=float(fde.min(axis=0).mean()))
    for name in ('JADE', 'JFDE'):
        k = winners[name]
        for t in FRAME_NUMBERS:
            frame = t - 1
            q = residual[k, :, frame] - (t / 12) * residual[k, :, -1]
            stem = f'{name}_winner_t{t}'
            values[f'{stem}_error_m'] = float(errors[k, :, frame].mean())
            values[f'{stem}_q_m'] = float(np.linalg.norm(q, axis=-1).mean())
    return values, winners


def _paths(row, future, path0, path1):
    winners = {}
    paths = {'S0': _path_by_seed(path0, len(future), 'S0'),
             'R12': _path_by_seed(path1, len(future), 'R12')}
    if paths['S0'].keys() != paths['R12'].keys():
        raise ValueError('model seeds must pair')
    row['path_seed_count'] = len(paths['S0'])
    averages = {}
    for model, seed_paths in paths.items():
        values = []
        winners[model] = {}
        for seed, path in seed_paths.items():
            scores, winners[model][seed] = _path_row(path, future)
            values.append(scores)
        averages[model] = {key: math.fsum(item[key] for item in values)/len(values) for key in values[0]}
        row.update({f'path_{model}_{key}': value for key, value in averages[model].items()})
    row.update({f'path_delta_{key[:-2]}_cm': 100*(averages['R12'][key]-averages['S0'][key])
                for key in averages['S0']})
    return winners


def diagnose_window(*, fold, split, window, observed, future, raw, g0, g1,
                    unary_world, social_world, path0, path1, dt=.4):
    """Return one anonymous row from the fully paired valid agent population."""
    n = len(observed)
    if not n or dt <= 0:
        raise ValueError('empty or invalid window')
    observed = _array(observed, (n, 8, 2), 'observations')
    future = _array(future, (n, 12, 2), 'future')
    goals = {name: _array(value, (20, n, 2), name)
             for name, value in zip(STAGES, (raw, g0, g1))}
    g0, g1 = goals['g0'], goals['g1']
    unary_world = _array(unary_world, (20, n, 2), 'unary displacement')
    social_world = _array(social_world, (20, n, 2), 'social displacement')
    row = dict(fold=fold, split=split, window=window, n=n, group=group_for_agents(n))
    endpoints, errors, costs, joint_slots, individual_slots = {}, {}, {}, {}, {}
    for name, values in goals.items():
        endpoints[name], errors[name], costs[name], joint_slots[name], individual_slots[name] = endpoint_scores(
            values, future[:, -1])
        row.update({f'{name}_{metric}_m': score for metric, score in endpoints[name].items()})
    for stage, previous in (('g0', 'raw'), ('g1', 'g0'), ('g1', 'raw')):
        for metric in ENDPOINT_METRICS:
            row[f'delta_{stage}_{previous}_{metric}_cm'] = 100 * (endpoints[stage][metric] - endpoints[previous][metric])

    k0, k1 = joint_slots['g0'], joint_slots['g1']
    shift = g1 - g0
    samples = {name: np.linalg.norm(value, axis=-1)
               for name, value in (('unary', unary_world), ('social', social_world), ('combined', shift))}
    component_sum = samples['unary'] + samples['social']
    combined = samples['combined']
    per_slot = combined.mean(axis=1)
    h = float(per_slot.max())
    delta = endpoints['g1']['J'] - endpoints['g0']['J']
    fixed_delta = float(costs['g1'][k0] - costs['g0'][k0])
    switch_delta = float(costs['g1'][k1] - costs['g1'][k0])
    if abs(delta - fixed_delta - switch_delta) > BOUND_ATOL_M:
        raise ArithmeticError('winner decomposition did not reconstruct the observed JFDE change')
    toward = future[:, -1] - g0[k0]
    toward_norm = np.linalg.norm(toward, axis=1)
    has_direction = toward_norm > BOUND_ATOL_M
    projection = (shift[k0, has_direction] * toward[has_direction]).sum(axis=-1) / toward_norm[has_direction]
    where_both = (samples['unary'] > BOUND_ATOL_M) & (samples['social'] > BOUND_ATOL_M)
    active_dots = (unary_world * social_world).sum(axis=-1)[where_both]
    # Use the component vector sum for cancellation, avoiding float32 coordinate rounding.
    vector_sum_norm = float(np.linalg.norm(unary_world + social_world, axis=-1).sum())
    component_norm = float(component_sum.sum())
    row.update(
        H_m=h, abs_delta_J_m=abs(delta), H_bound_slack_m=h-abs(delta),
        H_bound_violation=abs(delta) > h + BOUND_ATOL_M,
        abs_delta_J_ge_1cm=abs(delta) >= .01, H_ge_1cm=h >= .01,
        winner_S0_slot=k0, winner_R12_slot=k1, winner_switch_rate=float(k0 != k1),
        winner_delta_fixed_cm=100*fixed_delta, winner_delta_switch_cm=100*switch_delta,
        winner_shift_m=float(per_slot[k0]), nonwinner_shift_m=float(np.delete(per_slot, k0).mean()),
        winner_direction_agents=int(has_direction.sum()),
        winner_toward_projection_mean_m=_optional_mean(projection),
        winner_toward_positive_rate=_optional_mean(projection > 0),
        branch_opposition_pairs=int(active_dots.size),
        branch_opposition_rate=_optional_mean(active_dots < 0),
        branch_norm_sum_m=component_norm, branch_combined_norm_sum_m=vector_sum_norm,
        cancel_ratio=1-vector_sum_norm/component_norm if component_norm else 0.,
        branch_reconstruction_max_abs_m=float(np.abs(unary_world + social_world - shift).max()),
    )
    for name, sample in samples.items():
        row[f'{name}_mean_m'] = float(sample.mean())
        row[f'{name}_median_m'], row[f'{name}_p90_m'] = [float(value) for value in np.quantile(sample, (.5, .9))]
    _partition_rows(row, observed, errors, endpoints)
    if fold == 'hotel':
        _hotel_motion(row, observed, future, g0, k0, individual_slots['g0'], dt)
    winners = _paths(row, future, path0, path1)
    return WindowDiagnosis(row, samples, winners)


def _mean(rows, key, weight=None):
    valid = [row for row in rows if row.get(key) is not None]
    if weight is None:
        return math.fsum(row[key] for row in valid)/len(valid) if valid else None
    count = sum(row[weight] for row in valid)
    return math.fsum(row[key]*row[weight] for row in valid)/count if count else None


def _maximum(rows, key):
    return max((row[key] for row in rows), default=None)


def _summarize(group, fold, split, examples, whole_j):
    rows = [example.row for example in examples]
    scope = dict(fold=fold, split=split, group=group, windows=len(rows),
                 agents=sum(row['n'] for row in rows))
    for stage in STAGES:
        for metric in ENDPOINT_METRICS:
            key = f'{stage}_{metric}_m'
            scope[key] = _mean(rows, key, 'n' if metric == 'minFDE' else None)
    for stage, previous in (('g0', 'raw'), ('g1', 'g0'), ('g1', 'raw')):
        for metric in ENDPOINT_METRICS:
            key = f'delta_{stage}_{previous}_{metric}_cm'
            scope[key] = _mean(rows, key, 'n' if metric == 'minFDE' else None)
    scope['g0_J_error_share'] = math.fsum(row['g0_J_m'] for row in rows) / whole_j if whole_j else None
    scope['top10_windows'] = sum(row['is_top10'] for row in rows)
    path_keys = sorted({key for row in rows for key in row if key.startswith('path_')
                        and key.endswith(('_m', '_cm'))})
    for key in path_keys:
        weight = 'n' if any(metric in key for metric in ('minADE', 'minFDE')) else None
        scope[key] = _mean(rows, key, weight)

    scope['H_max_m'] = _maximum(rows, 'H_m')
    scope['abs_delta_J_max_m'] = _maximum(rows, 'abs_delta_J_m')
    scope['H_bound_min_slack_m'] = min((row['H_bound_slack_m'] for row in rows), default=None)
    scope['H_bound_violation_windows'] = sum(row['H_bound_violation'] for row in rows)
    scope['abs_delta_J_ge_1cm_windows'] = sum(row['abs_delta_J_ge_1cm'] for row in rows)
    scope['H_ge_1cm_windows'] = sum(row['H_ge_1cm'] for row in rows)
    scope['branch_reconstruction_max_abs_m'] = _maximum(rows, 'branch_reconstruction_max_abs_m')
    scope['H_median_m'], scope['H_p90_m'] = ([float(value) for value in np.quantile([row['H_m'] for row in rows], (.5, .9))]
                                            if rows else [None, None])
    for name in ('unary', 'social', 'combined'):
        samples = np.concatenate([example.samples[name].ravel() for example in examples]) if examples else None
        quantiles = [float(value) for value in np.quantile(samples, (.5, .9))] if samples is not None else [None, None]
        scope[f'{name}_median_m'], scope[f'{name}_p90_m'] = quantiles
    scope['winner_toward_projection_mean_m'] = _mean(rows, 'winner_toward_projection_mean_m', 'winner_direction_agents')
    scope['winner_toward_positive_rate'] = _mean(rows, 'winner_toward_positive_rate', 'winner_direction_agents')
    scope['branch_opposition_rate'] = _mean(rows, 'branch_opposition_rate', 'branch_opposition_pairs')
    branch_norm = math.fsum(row['branch_norm_sum_m'] for row in rows)
    combined_norm = math.fsum(row['branch_combined_norm_sum_m'] for row in rows)
    scope['cancel_ratio'] = 1 - combined_norm/branch_norm if branch_norm else (0. if rows else None)
    for key in ('winner_shift_m', 'nonwinner_shift_m', 'winner_switch_rate',
                'winner_delta_fixed_cm', 'winner_delta_switch_cm'):
        scope[key] = _mean(rows, key)

    multi = [row for row in rows if row['n'] > 1]
    informative = [row for row in multi if row['partition_informative']]
    multi_agents = sum(row['n'] for row in multi)
    paired = sum(row['paired_agents'] for row in multi)
    scope.update(paired_agents=paired, pair_coverage=paired/multi_agents if multi_agents else None,
                 multi_windows=len(multi), multi_agents=multi_agents,
                 partition_pair_windows=sum(row['partition_has_pair'] for row in multi),
                 partition_informative_windows=len(informative),
                 partition_uninformative_windows=len(multi)-len(informative))
    for stage in STAGES:
        for term in ('JP', 'J_minus_JP', 'JP_minus_L', 'random_JP', 'random_JP_sd',
                     'random_J_minus_JP', 'random_JP_minus_L', 'JP_minus_random_JP'):
            key = f'{stage}_{term}_m'
            scope[key] = _mean(informative, key)
            scope[f'{key}_all_multi'] = _mean(multi, key)

    if fold == 'hotel':
        for key in ('low_speed_agents', 'future_low_speed_agents', 'axis_agents', 'no_axis_agents',
                    'truth_angle_agents'):
            scope[key] = sum(row[key] for row in rows)
        for key in ('cv_FDE_m', 'cv_beats_g0_J_rate', 'cv_beats_g0_L_rate'):
            scope[key] = _mean(rows, key)
        for key in ('observed_speed_mean_mps', 'future_speed_mean_mps', 'g0_people_cv_better_rate'):
            scope[key] = _mean(rows, key, 'n')
        for model in ('cv', 'g0_joint', 'g0_person'):
            key = f'{model}_no_axis_FDE_m'
            scope[key] = _mean(rows, key, 'no_axis_agents')
            for term in ('long_signed', 'long_abs', 'lat_signed', 'lat_abs'):
                key = f'{model}_{term}_m'
                scope[key] = _mean(rows, key, 'axis_agents')
        scope['truth_turn_abs_deg'] = _mean(rows, 'truth_turn_abs_deg', 'truth_angle_agents')
        for bucket in MOTION_BUCKETS:
            weight = f'motion_{bucket}_agents'
            scope[weight] = sum(row[weight] for row in rows)
            for model in ('cv', 'g0_joint', 'g0_person'):
                key = f'motion_{bucket}_{model}_FDE_m'
                scope[key] = _mean(rows, key, weight)
    return scope


def summarize_windows(diagnoses):
    """Group windows after each window already averages the available inference seeds."""
    fold_splits = defaultdict(list)
    for example in diagnoses:
        fold_splits[example.row['fold'], example.row['split']].append(example)
    summary = []
    for (fold, split), examples in fold_splits.items():
        ranked = sorted(examples, key=lambda example: (-example.row['g0_J_m'], example.row['window']))
        top = ranked[:math.ceil(len(examples)/10)]
        top_ids = {example.row['window'] for example in top}
        if len({example.row['window'] for example in examples}) != len(examples):
            raise ValueError('duplicate anonymous window')
        full_j = math.fsum(example.row['g0_J_m'] for example in examples)
        for example in examples:
            example.row['is_top10'] = example.row['window'] in top_ids
            example.row['g0_J_error_share'] = example.row['g0_J_m'] / full_j if full_j else None
        groups = {'all': examples, 'top10': top, 'remaining90': ranked[len(top):]}
        groups.update({key: [example for example in examples if example.row['group'] == key]
                       for key in BASE_GROUPS[1:]})
        groups.update({f'top10_{key}': [example for example in top if example.row['group'] == key]
                       for key in BASE_GROUPS[1:]})
        groups.update(
            paired=[example for example in examples if example.row['partition_has_pair']],
            partition_informative=[example for example in examples if example.row['partition_informative']],
            partition_uninformative=[example for example in examples
                                     if example.row['n'] > 1 and not example.row['partition_informative']],
        )
        for group in (*BASE_GROUPS, *TOP_GROUPS, *MULTI_GROUPS):
            summary.append(_summarize(group, fold, split, groups[group], full_j))
    return summary

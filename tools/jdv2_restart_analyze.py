#!/usr/bin/env python3
"""CPU-only comparisons of bounded restart artifacts. No sampling or GPU import-use."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from tools.jdv2_restart_observer import compare_tensor
from tools.jdv2_restart_probe import ROOT
from tools.jdv2_stage_a_bank import REPO, write_json, sha256


def read(path):
    return json.loads(path.read_text())


def compare_maps(a, b):
    common = [k for k in a if k in b]
    all_rows = {k: compare_tensor(a[k], b[k]) for k in common}
    return {'keys_equal': set(a) == set(b), 'tensor_count': len(common),
            'all_bitwise_equal': set(a) == set(b) and all(r['bitwise_equal'] for r in all_rows.values()),
            'differences': {k: row for k, row in all_rows.items() if not row['bitwise_equal']},
            'equal_tensor_count': sum(row['bitwise_equal'] for row in all_rows.values())}


def exact_replay(process):
    from src.models.joint_dependency_v2.exact_lexicographic_assignment import solve_exact_persistent_tie
    values = read(process / 'TRACE_METADATA.json')['values']
    tensors = torch.load(process / 'TRACE.pt', weights_only=True, map_location='cpu')
    prefixes = [k.removesuffix('/input/score') for k in tensors if k.startswith('solve_exact_persistent_tie/') and k.endswith('/input/score')]
    rows = []
    for p in prefixes:
        priorities = []
        for i in range(tensors[p + '/input/score'].shape[0]):
            priorities.append(tuple(map(int, values[p + f'/output/priorities/{i}'])))
        kwargs = {'score': tensors[p + '/input/score'], 'mask': tensors[p + '/input/mask'],
                  'previous_ids': tensors[p + '/input/previous_ids'],
                  'goal_candidates_world': tensors[p + '/input/goal_candidates_world'],
                  'priorities': priorities}
        first = solve_exact_persistent_tie(**kwargs)
        second = solve_exact_persistent_tie(**kwargs)
        actual_assignment = tuple(map(int, values[p + '/output/assignment']))
        actual_objective = tuple(map(int, values[p + '/output/objective']))
        actual_encoded = int(values[p + '/output/encoded_objective'])
        rows.append({'key': p, 'same_input_two_CPU_solves_equal': first == second,
            'matches_actual_assignment': first.assignment == actual_assignment,
            'matches_actual_objective': first.objective == actual_objective,
            'matches_actual_encoded_objective': first.encoded_objective == actual_encoded,
            'assignment': list(first.assignment), 'objective_decimal': list(map(str, first.objective)),
            'encoded_objective_decimal': str(first.encoded_objective),
            'actual_priorities_used': True})
    return {'problems': len(rows), 'CPU_solver_calls': 2 * len(rows), 'rows': rows,
            'all_pass': bool(rows) and all(all(r[k] for k in ('same_input_two_CPU_solves_equal', 'matches_actual_assignment', 'matches_actual_objective', 'matches_actual_encoded_objective')) for r in rows),
            'region_replay_budget_consumed': 0, 'note': 'CPU rechecking captured exact solver payloads, not encode/sampler or GPU region replay'}


def analyze(output):
    processes = [ROOT / p for p in ('P0', 'P1', 'P2', 'P3') if (ROOT / p).exists()]
    completed = [p for p in processes if (p / 'COMPLETE.json').exists()]
    budget = {}
    for p in processes:
        end = p / ('COMPLETE.json' if (p / 'COMPLETE.json').exists() else 'FAILED.json')
        if end.exists():
            budget[p.name] = read(end)
        else:
            attempts = sorted((p / 'budget').glob('*_attempt.json'))
            completions = sorted((p / 'budget').glob('*_complete.json'))
            budget[p.name] = {'status': 'IN_FLIGHT_OR_ABRUPT_DEATH_UNKNOWN',
                'full_forwards_attempted': len(attempts), 'full_forwards_completed': len(completions),
                'in_flight_unknown': len(attempts) - len(completions)}
    comparisons = {}
    for ia, pa in enumerate(completed):
        for pb in completed[ia + 1:]:
            key = pa.name + '_' + pb.name
            windows = {}
            for i in range(14):
                with np.load(pa / f'windows/{i:02d}.npz', allow_pickle=False) as a, np.load(pb / f'windows/{i:02d}.npz', allow_pickle=False) as b:
                    windows[str(i)] = compare_maps({k: torch.from_numpy(a[k]) for k in a.files}, {k: torch.from_numpy(b[k]) for k in b.files})
            targets = compare_maps(torch.load(pa / 'TARGET.pt', weights_only=True, map_location='cpu'), torch.load(pb / 'TARGET.pt', weights_only=True, map_location='cpu'))
            from tools.jdv2_stage_b_identity_contract_audit import _rng_equal
            ra = torch.load(pa / 'RNG.pt', weights_only=False, map_location='cpu')
            rb = torch.load(pb / 'RNG.pt', weights_only=False, map_location='cpu')
            sa, sb = read(pa / 'STATE.json'), read(pb / 'STATE.json')
            comparisons[key] = {'windows': windows, 'target': targets,
                'global_rng_equal': {k: _rng_equal(ra[k], rb[k]) for k in ra},
                'settings_equal': sa['settings'] == sb['settings'],
                'state_before_equal': sa['before'] == sb['before'],
                'state_after_equal': sa['after'] == sb['after']}
    trace = None
    exact = {}
    for p in completed:
        if (p / 'TRACE.pt').exists():
            exact[p.name] = exact_replay(p)
    if all((ROOT / p / 'TRACE.pt').exists() for p in ('P2', 'P3')):
        trace = compare_maps(*(torch.load(ROOT / p / 'TRACE.pt', weights_only=True, map_location='cpu') for p in ('P2', 'P3')))
        ma, mb = (read(ROOT / p / 'TRACE_METADATA.json') for p in ('P2', 'P3'))
        trace['integer_scalar_payloads_equal'] = ma['values'] == mb['values']
        trace['different_scalar_keys'] = [k for k in ma['values'] if ma['values'].get(k) != mb['values'].get(k)]
        trace['event_order_equal'] = ma['events'] == mb['events']
        trace['invalid_snapshots'] = {'P2': ma['invalid_snapshots'], 'P3': mb['invalid_snapshots']}
        trace['first_captured_tensor_difference'] = next(iter(trace['differences']), None)
        trace['round_ID_differences'] = {str(i): trace['differences'].get(f'ParallelConditionalSampler._emit_diagnostic/{i:03d}/input/selected') for i in range(3)}
        trace['different_scalar_payloads'] = {k: {'P2': ma['values'].get(k), 'P3': mb['values'].get(k)} for k in trace['different_scalar_keys']}
    untraced = comparisons.get('P0_P1')
    reproduced = bool(untraced and any(not w['all_bitwise_equal'] for w in untraced['windows'].values()))
    neutrality = 'TRACE_NEUTRALITY_UNCERTIFIED'
    if len(completed) == 4 and not reproduced and all(
            c['target']['all_bitwise_equal'] and all(w['all_bitwise_equal'] for w in c['windows'].values()) and
            all(c['global_rng_equal'].values()) and c['settings_equal'] and c['state_before_equal'] and c['state_after_equal']
            for c in comparisons.values()) and trace and not any(trace['invalid_snapshots'].values()):
        neutrality = 'BOUNDED_OUTPUT_AND_RNG_EQUIVALENCE_OBSERVED_NOT_UNIVERSAL_PROOF'
    if len(completed) != 4:
        status = 'BLOCKED_OR_INCOMPLETE_SEE_BUDGET'
    elif reproduced and neutrality == 'TRACE_NEUTRALITY_UNCERTIFIED':
        status = 'REPRODUCED_OUTPUT_DIVERGENCE_TRACE_INCONCLUSIVE'
    elif all(c['target']['all_bitwise_equal'] for c in comparisons.values()):
        status = 'NOT_REPRODUCED_WITHIN_BUDGET'
    else:
        status = 'TRACE_NEUTRALITY_UNCERTIFIED'
    result = {'task_status': 'COMPLETE_BOUNDED_DIAGNOSTIC' if len(completed) == 4 else 'INCOMPLETE',
        'status': status, 'probe_source_commit': read(completed[0] / 'PROVENANCE.json')['probe_source_commit'] if completed else None,
        'historical_root_cause': 'OPEN', 'untraced_output_divergence_reproduced': reproduced,
        'trace_neutrality': neutrality, 'budget': budget,
        'totals': {k: sum(v.get(k, 0) for v in budget.values()) for k in ('processes_started', 'initialization_attempted', 'initialization_completed', 'full_forwards_attempted', 'full_forwards_completed', 'full_forwards_failed', 'in_flight_unknown', 'region_replays')},
        'comparisons': comparisons, 'trace_P2_P3': trace, 'CPU_exact_solver': exact,
        'operator_root_cause': None, 'lower_kernel_cause': None,
        'scope': 'seed2036 fresh protocol prefix0-13; not historical seed2035 allocator/warmup restoration',
        'historical_pairing_metrics_unchanged': True, 'production_modified': False,
        'training_started': False, 'SDD_resumed': False}
    write_json(output / 'RESULTS.json', result)
    manifest = {'root': str(ROOT.relative_to(REPO)), 'files': {p.relative_to(ROOT).as_posix(): {'sha256': sha256(p), 'bytes': p.stat().st_size}
                for p in sorted(ROOT.rglob('*')) if p.is_file()},
                'large_native_tensor_files_not_committed': True}
    write_json(output / 'TRACE_MANIFEST.json', manifest)
    print(json.dumps({k: result[k] for k in ('task_status', 'status', 'trace_neutrality', 'totals')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    analyze(parser.parse_args().output)

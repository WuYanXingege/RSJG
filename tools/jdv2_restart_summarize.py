#!/usr/bin/env python3
"""Publish compact contracts from already finished probe artifacts; no Torch/GPU."""
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.jdv2_stage_a_bank import REPO, write_json, sha256


def read(path):
    return json.loads(path.read_text())


def main():
    root = REPO / 'outputs/joint_dependency_v2/eth/joint_dependency_v2/audits/restart_trace_ea846fc'
    out = REPO / 'docs/joint_dependency_v2/reviews/2026-10-04_80ee0e2_restart_trace'
    results = read(out / 'RESULTS.json')
    trace = results['trace_P2_P3']
    contracts, execution = {}, {}
    for name in ('P0', 'P1', 'P2', 'P3'):
        p = root / name
        state = read(p / 'STATE.json')
        resources = [read(p / 'RESOURCE_INITIAL.json')] + [read(f) for f in sorted((p / 'resources').glob('*.json'))]
        started = read(p / 'STARTED.json')
        owners = [owner for r in resources for owner in r['compute_processes'] if owner[1] != str(started['pid'])]
        assert not owners
        execution[name] = {'started': started, 'completed': read(p / 'COMPLETE.json'),
            'resource_gate_count': len(resources), 'other_compute_owners': owners,
            'first_resource': resources[0], 'last_resource': resources[-1],
            'strict_load': state['strict_load'], 'production_cache_validation': state['cache_validation'],
            'all_modules_eval': state['all_modules_eval'], 'parameters_and_buffers_unchanged': state['unchanged'],
            'settings': state['settings'], 'sampler_context_before': state['sampler_context_before'],
            'sampler_context_after': state['sampler_context_after'],
            'state_fingerprint_sha256': hashlib.sha256(json.dumps(state['before'], sort_keys=True).encode()).hexdigest()}
        contracts[name] = {'target': read(p / 'TARGET_METADATA.json')}
        if (p / 'TRACE_METADATA.json').exists():
            metadata = read(p / 'TRACE_METADATA.json')
            contracts[name]['trace'] = {k: v for k, v in metadata.items() if k != 'values'}
    key = trace['first_captured_tensor_difference']
    inputs = [k for k in contracts['P2']['trace']['tensors'] if k.startswith('SocialMotionEncoder.forward/000/input/')]
    rows = {}
    for k in inputs:
        a, b = (contracts[p]['trace']['tensors'][k] for p in ('P2', 'P3'))
        rows[k] = {'P2': a, 'P3': b, 'dtype_shape_raw_equal': all(a[f] == b[f] for f in ('dtype', 'shape', 'raw_sha256'))}
    assert all(r['dtype_shape_raw_equal'] for r in rows.values())
    draws = [k for k in contracts['P2']['trace']['tensors'] if k.startswith('diffusion_draw/')]
    assert len(draws) == 121 and not any(k in trace['differences'] for k in draws)
    generators = {}
    for p in ('P2', 'P3'):
        tm = contracts[p]['trace']['tensors']
        generators[p] = {g: {'before_sha256': tm['explicit_generators/before/' + g]['raw_sha256'],
                             'after_sha256': tm['explicit_generators/after/' + g]['raw_sha256'],
                             'consumed_state_changed': tm['explicit_generators/before/' + g]['raw_sha256'] != tm['explicit_generators/after/' + g]['raw_sha256']}
                         for g in ('initial', 'round_1', 'round_2')}
    summary = {'status': results['status'], 'trace_neutrality': results['trace_neutrality'],
        'historical_root_cause': 'OPEN',
        'untraced_P0_P1': {'window13': results['comparisons']['P0_P1']['windows']['13'],
            'all_three_global_rng_boundaries_equal': all(results['comparisons']['P0_P1']['global_rng_equal'].values()),
            'Round0_IDs_bitwise_equal': 'auxiliary/sampled/initial_candidate_index' not in results['comparisons']['P0_P1']['target']['differences'],
            'first_ID_divergent_round': 'UNKNOWN_ROUND1_OR_ROUND2; no internal untraced round1 payload',
            'agent_feat_difference': results['comparisons']['P0_P1']['target']['differences']['auxiliary/agent_feat']},
        'traced_P2_P3': {'earliest_captured_boundary': key, 'difference': trace['differences'][key],
            'boundary_inputs': rows, 'qualifier': 'diagnostic clue only; observer neutrality uncertified; not an operator or historical root-cause certification',
            'first_ID_divergent_round': None, 'Round0_Round1_Round2_IDs_all_equal': all(v is None for v in trace['round_ID_differences'].values()),
            'Y_equal': 'Y' not in results['comparisons']['P2_P3']['windows']['13']['differences'],
            'actual_diffusion_draws': len(draws), 'actual_diffusion_draws_all_bitwise_equal': True,
            'actual_round0_Gumbel_and_perturbed_equal': not any(('gumbel_noise' in k or '/perturbed' in k) for k in trace['differences']),
            'explicit_generator_states': generators,
            'exact_actual_priorities_equal': not any('/priorities/' in k for k in trace['different_scalar_keys']),
            'conditional_scores': {str(i): trace['differences'].get(f'ParallelConditionalSampler._emit_diagnostic/{i:03d}/input/score') for i in (1, 2)},
            'actual_solver_problem_count': sum(v['problems'] for v in results['CPU_exact_solver'].values()),
            'formal_CPU_solver_calls': sum(v['CPU_solver_calls'] for v in results['CPU_exact_solver'].values()),
            'additional_P2_precheck_CPU_solver_calls': 36,
            'all_actual_CPU_objective_rechecks_pass': all(v['all_pass'] for v in results['CPU_exact_solver'].values())},
        'first_divergent_operator': None, 'lower_kernel_cause': None,
        'region_replays': 0, 'stop_reason': '56 complete forwards reached; trace neutrality remains uncertified. No operator/region follow-up.',
        'only_next_recommendation': 'Separately authorize a minimal SocialMotionEncoder boundary/observer-neutrality diagnosis using captured identical inputs before any operator attribution or energy intervention.'}
    write_json(out / 'FIRST_DIVERGENCE.json', summary)
    write_json(out / 'TRACE_CONTRACTS.json', contracts)
    write_json(out / 'EXECUTION.json', {'processes': execution, 'totals': results['totals'],
        'probe_provenance': read(root / 'P0/PROVENANCE.json'),
        'analysis_sources_sha256': {p: sha256(REPO / p) for p in ('tools/jdv2_restart_analyze.py', 'tools/jdv2_restart_summarize.py')},
        'SDD_resumed': False, 'skills_modified': False, 'production_code_modified': False,
        'final_GPU_readonly_observation': {'time_utc': '2026-10-04T06:29:53.388756+00:00', 'compute_processes': [], 'scope': 'separate read-only nvidia-smi query after all four processes exited'}})
    print(json.dumps({'contracts_processes': len(contracts), 'first_boundary': key,
                      'boundary_inputs': len(rows), 'exact_problems': summary['traced_P2_P3']['actual_solver_problem_count']}))


if __name__ == '__main__':
    main()

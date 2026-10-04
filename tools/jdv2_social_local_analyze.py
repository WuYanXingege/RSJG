#!/usr/bin/env python3
"""CPU-only all-output comparisons; no local forward, no GPU execution."""
import argparse
import itertools
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tools.jdv2_social_local import ROOT, OLD, PRIOR, read
from tools.jdv2_restart_observer import compare_tensor
from tools.jdv2_stage_a_bank import write_json, sha256, REPO


def analyze(output):
    processes = [p for p in ('U0','U1','T0','T1') if (ROOT/p).exists()]
    complete = [p for p in processes if (ROOT/p/'COMPLETE.json').exists()]
    outputs, metadata, execution = {}, {}, {}
    for process in processes:
        root = ROOT / process
        terminal = root / ('COMPLETE.json' if process in complete else 'FAILED.json')
        budget = read(terminal) if terminal.exists() else {'status': 'INFLIGHT_UNKNOWN',
            'encoder_attempted': len(list((root/'budget').glob('*_attempt.json'))),
            'encoder_completed': len(list((root/'budget').glob('*_complete.json'))),
            'unknown_inflight': len(list((root/'budget').glob('*_attempt.json'))) - len(list((root/'budget').glob('*_complete.json')))}
        execution[process] = {'started': read(root/'STARTED.json'), 'budget': budget,
            'initialization': read(root/'INITIALIZED.json') if (root/'INITIALIZED.json').exists() else None,
            'validation': read(root/'VALIDATION.json') if (root/'VALIDATION.json').exists() else None,
            'resources': {p.name:read(p) for p in sorted(root.glob('RESOURCE*.json'))}}
        if (root/'OUTPUTS.pt').exists():
            saved = torch.load(root/'OUTPUTS.pt', weights_only=True, map_location='cpu')
            outputs.update({process+'/'+k: v for k,v in saved.items()})
            metadata[process] = {'output':read(root/'OUTPUT_CONTRACTS.json')}
            if (root/'TRACE_CONTRACTS.json').exists():
                metadata[process]['trace'] = read(root/'TRACE_CONTRACTS.json')
    pairs = []
    for a,b in itertools.combinations(outputs,2):
        pa,ca,_ = a.split('/');pb,cb,_ = b.split('/')
        pairs.append({'a':a,'b':b,'same_process':pa==pb,'same_call_position':ca==cb,
            'a_has_internal_observer':pa.startswith('T') and ca=='call4',
            'b_has_internal_observer':pb.startswith('T') and cb=='call4',
            **compare_tensor(outputs[a],outputs[b])})
    unobserved = [p for p in pairs if not p['a_has_internal_observer'] and not p['b_has_internal_observer']]
    varies = any(not p['bitwise_equal'] for p in unobserved) if unobserved else None
    trace = None
    if all((ROOT/p/'TRACE.pt').exists() for p in ('T0','T1')):
        a,b=(torch.load(ROOT/p/'TRACE.pt',weights_only=True,map_location='cpu') for p in ('T0','T1'))
        comparisons={k:compare_tensor(a[k],b[k]) for k in a}
        valid=all(not metadata[p]['trace']['invalid_snapshots'] for p in ('T0','T1'))
        first=next((k for k,v in comparisons.items() if not v['bitwise_equal']),None)
        trace={'snapshots_valid':valid,'keys_equal':set(a)==set(b),'comparisons':comparisons,
            'first_different_boundary':first if valid else None,
            'boundary_status':('FIRST_DIFFERENT_LOCAL_MODULE_BOUNDARY' if first else 'LOCAL_OUTPUT_CONSISTENCY_OBSERVED') if valid else 'INVALID_SNAPSHOT',
            'scope':'local module boundary only; no operator/kernel/historical attribution'}
    old={}
    manifest=read(PRIOR/'TRACE_MANIFEST.json')['files']
    for p in ('P0','P1','P2','P3'):
        f=OLD/p/'TARGET.pt'
        assert sha256(f)==manifest[p+'/TARGET.pt']['sha256']
        old[p]=torch.load(f,weights_only=True,map_location='cpu')['auxiliary/agent_feat']
    historical={k:{p:compare_tensor(v,h) for p,h in old.items()} for k,v in outputs.items()}
    budget_keys=('processes_started','initialization_attempted','initialization_completed','encoder_attempted','encoder_completed','encoder_failed','unknown_inflight','full_A_forwards','net_encode_calls','sampler_calls','diffusion_calls','independent_submodule_replays')
    totals={k:sum(e['budget'].get(k,0) for e in execution.values()) for k in budget_keys}
    contracts_pass=bool(complete) and all(all(execution[p]['validation'][k] for k in ('input_unchanged','model_unchanged','global_RNG_unchanged','hooks_removed','snapshots_valid')) for p in complete)
    status=('LOCAL_UNOBSERVED_OUTPUT_VARIATION_REPRODUCED' if varies else 'NOT_REPRODUCED_IN_STANDALONE_CONTEXT_WITHIN_BUDGET') if len(complete)==4 else 'BLOCKED_OR_INCOMPLETE_SEE_EXECUTION'
    result={'task_status':'COMPLETE_BOUNDED_DIAGNOSTIC' if len(complete)==4 else 'BLOCKED_OR_INCOMPLETE',
        'status':status,'totals':totals,'unobserved_output_variation':varies,
        'probe_source_commit':execution[complete[0]]['initialization']['source_commit'] if complete else None,
        'prior_archive_commit':'634a467868c3d6a488837be4edc18ccff52db864',
        'source_contract':'INPUT_MANIFEST.json; unique frozen P2 inputs, P3 authentication only',
        'per_call':[{'process':p,'call':int(k.split('/')[0][4:]),
                     'internal_observer':p.startswith('T') and k.startswith('call4/'),
                     'completed':True,'h':row}
                    for p,m in metadata.items() for k,row in m['output']['tensors'].items()],
        'settings_by_process':{p:execution[p]['initialization']['settings'] for p in complete},
        'batch_validation':{p:{k:execution[p]['validation'][k] for k in
                            ('input_unchanged','model_unchanged','global_RNG_unchanged','hooks_removed','snapshots_valid')}
                            for p in complete},
        'unobserved_calls':sum(not (k.startswith('T') and '/call4/' in k) for k in outputs),
        'observed_calls':sum(k.startswith('T') and '/call4/' in k for k in outputs),
        'observer_semantic_snapshot_contract':'PASS_CHECKED_PROPERTIES_ONLY' if contracts_pass else 'NOT_ESTABLISHED',
        'canonical_GPU_trace_neutrality':'TRACE_NEUTRALITY_UNCERTIFIED','historical_root_cause':'OPEN',
        'causal_link_to_previous29_ID_changes':'NOT_ESTABLISHED_NO_DOWNSTREAM_EXECUTION',
        'precision':'original canonical encoder FP32 island; no BF16/FP32 intervention',
        'all_output_pairs':pairs,'same_process_pairs':sum(p['same_process'] for p in pairs),
        'cross_process_same_position_pairs':sum(not p['same_process'] and p['same_call_position'] for p in pairs),
        'trace_T0_T1':trace,'historical_h_comparisons':historical,
        'formal_bank_metrics_unchanged':{'delta_JADE':0.03391811925732022,'delta_JFDE':0.0774227560742364},
        'SDD_resumed':False,'production_fixed':False,'skills_modified':False,
        'limitations':['16 local calls max; common h retention and between-call CPU ledger costs',
            'restored before13 RNG is not historical encoder-entry snapshot', 'CUDA addresses/allocator/workspace/scheduling not restored',
            'two observed calls do not establish universal or canonical observer neutrality']}
    write_json(output/'RESULTS.json',result)
    write_json(output/'EXECUTION.json',execution)
    write_json(output/'TRACE_CONTRACTS.json',metadata)
    first=next((ROOT/p/'INPUT_MANIFEST.json' for p in processes if (ROOT/p/'INPUT_MANIFEST.json').exists()),None)
    write_json(output/'INPUT_MANIFEST.json',read(first) if first else {'status':'NOT_AVAILABLE','reason':'authentication not completed'})
    write_json(output/'ARTIFACT_MANIFEST.json',{'root':str(ROOT.relative_to(REPO)), 'files':{str(p.relative_to(ROOT)):{'sha256':sha256(p),'bytes':p.stat().st_size} for p in sorted(ROOT.rglob('*')) if p.is_file()}})
    print(json.dumps({'status':status,'budget':totals,'unobserved_varies':varies,'first_boundary':trace['first_different_boundary'] if trace else None}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    analyze(parser.parse_args().output)

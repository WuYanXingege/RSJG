#!/usr/bin/env python3
"""CPU-only aggregation of all bounded message diagnostic artifacts."""
import argparse
import itertools
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tools.jdv2_message_diagnostic import ROOT,BASE,read
from tools.jdv2_message_observer import compare_traces
from tools.jdv2_restart_observer import compare_tensor
from tools.jdv2_stage_a_bank import write_json,sha256


def analyze(destination):
    processes={};outputs={};contracts={};traces={};trace_contracts={}
    for name in ('U0','U1','T0','T1'):
        folder=ROOT/name
        terminal=folder/('COMPLETE.json' if (folder/'COMPLETE.json').exists() else 'FAILED.json')
        processes[name]={'terminal':read(terminal),'initialization':read(folder/'INITIALIZED.json') if (folder/'INITIALIZED.json').exists() else None,
            'validation':read(folder/'VALIDATION.json') if (folder/'VALIDATION.json').exists() else None,
            'resources':{p.name:read(p) for p in sorted((folder/'resources').glob('*.json'))}}
        if (folder/'OUTPUTS.pt').exists():
            outputs.update({name+'/'+k:v for k,v in torch.load(folder/'OUTPUTS.pt',weights_only=True,map_location='cpu').items()})
            contracts[name]=read(folder/'OUTPUT_CONTRACTS.json')
        if (folder/'TRACE.pt').exists():
            traces[name]=torch.load(folder/'TRACE.pt',weights_only=True,map_location='cpu')
            trace_contracts[name]=read(folder/'TRACE_CONTRACTS.json')
    def observed(k):return k.startswith('T') and '/call2/' in k
    pairs=[]
    for a,b in itertools.combinations(outputs,2):
        pairs.append({'a':a,'b':b,'both_unobserved':not observed(a) and not observed(b),'both_traced':observed(a) and observed(b),
            'same_process':a[:2]==b[:2],'cross_process_same_position':a[:2]!=b[:2] and a[3:]==b[3:],**compare_tensor(outputs[a],outputs[b])})
    def summary(rows):return {'pairs':len(rows),'different_pairs':sum(not r['bitwise_equal'] for r in rows),
        'max_abs_diff':max((r.get('max_abs',0) or 0 for r in rows),default=0)}
    counters={k:sum(p['terminal'].get(k,0) for p in processes.values()) for k in (
        'processes_started','initialization_attempted','initialization_completed','initialization_failed','initialization_unknown_inflight','message_attempted','message_completed','message_failed','message_unknown_inflight',
        'extra_attempted','extra_completed','extra_failed','extra_unknown_inflight','full_A_forwards','net_encode_calls','encoder_calls','GRU_replays','sampler_calls','diffusion_calls')}
    trace_comparison=compare_traces(traces['T0'],traces['T1'],trace_contracts['T0'],trace_contracts['T1']) if len(traces)==2 else None
    extra=read(ROOT/'T1/EXTRA_STATUS.json') if (ROOT/'T1/EXTRA_STATUS.json').exists() else {'status':'UNAVAILABLE'}
    result={'base_commit':BASE,'source_commit':processes['U0']['terminal'].get('source_commit'),
        'status':'LOCAL_MESSAGE_VARIATION_REPRODUCED' if any(not r['bitwise_equal'] for r in pairs) else 'NOT_REPRODUCED_WITHIN_BUDGET',
        'counts':counters,'output_count':len(outputs),'unobserved_outputs':sum(not observed(k) for k in outputs),
        'traced_outputs':sum(observed(k) for k in outputs),'all_pairs':summary(pairs),
        'unobserved_pairs':summary([r for r in pairs if r['both_unobserved']]),'traced_pairs':summary([r for r in pairs if r['both_traced']]),
        'within_process':summary([r for r in pairs if r['same_process']]),
        'cross_process_same_position':summary([r for r in pairs if r['cross_process_same_position']]),
        'pairs':pairs,'trace_comparison':trace_comparison,'extra':extra,
        'extra_gate':read(ROOT/'T1/EXTRA_GATE.json') if (ROOT/'T1/EXTRA_GATE.json').exists() else None,
        'extra_validation':read(ROOT/'T1/EXTRA_VALIDATION.json') if (ROOT/'T1/EXTRA_VALIDATION.json').exists() else None,
        'per_call_contracts':contracts,'canonical_GPU_trace_neutrality':'TRACE_NEUTRALITY_UNCERTIFIED','historical_root_cause':'OPEN','causal_link_to_previous29_ID_changes':'NOT_ESTABLISHED',
        'limits':['Independent message input from prior GRU output, no GRU replay or full A forward.',
            'Python hooks/frame observer is not certified neutral on GPU.',
            'Controlled primitive variation does not identify kernel reduction schedule or prove historical 29 ID differences.',
            'No first-add original snapshot; stage A zero and stage B diagnostic state are explicitly constructed.',
            'No historic bank/metric modification; no training or production fix.']}
    manifest={str(p.relative_to(ROOT)):{'sha256':sha256(p),'bytes':p.stat().st_size} for p in sorted(ROOT.rglob('*')) if p.is_file()}
    write_json(destination/'RESULTS.json',result)
    write_json(destination/'EXECUTION.json',{'counts':counters,'processes':processes})
    write_json(destination/'TRACE_CONTRACTS.json',{'outputs':contracts,'traces':trace_contracts})
    write_json(destination/'INPUT_MANIFEST.json',read(ROOT/'U0/INPUT_MANIFEST.json'))
    write_json(destination/'CPU_CHECKS.json',read(ROOT/'CPU_CHECKS.json'))
    write_json(destination/'ARTIFACT_MANIFEST.json',{'local_root':str(ROOT),'files':manifest})
    print(__import__('json').dumps({k:result[k] for k in ('status','counts','all_pairs','unobserved_pairs','traced_pairs','within_process','cross_process_same_position')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path)
    analyze(parser.parse_args().output)

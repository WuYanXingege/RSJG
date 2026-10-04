#!/usr/bin/env python3
"""Post-run CPU read-only cross-process contracts and explicit round-score summaries."""
import itertools
import json
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.jdv2_h_sampler_contract import ROOT,OLD,load,read,compare_tensor,H_HASH
from tools.jdv2_stage_b_identity_contract_audit import _rng_equal
from tools.jdv2_restart_observer import state_fingerprint


def check():
    calls={};versions={};reference=load(OLD/'TRACE.pt')
    states=[read(ROOT/p/'STATE.json') for p in ('P0','P1')]
    assert states[0]==states[1]
    settings=[read(ROOT/p/'INITIALIZED.json')['settings'] for p in ('P0','P1')]
    assert settings[0]==settings[1]
    first_rng=None
    for process in ('P0','P1'):
        storage=set();versions[process]=[]
        for folder in sorted((ROOT/process).glob('call*')):
            name=process+'/'+folder.name;t=load(folder/'TRACE.pt');m=read(folder/'TRACE_METADATA.json');v=read(folder/'VALIDATION.json');rng=__import__('torch').load(folder/'RNG.pt',weights_only=False,map_location='cpu')
            assert all(v['checks'].values()) and not m['invalid_snapshots']
            assert m['native_dtype_roundtrip'] and m['tensors']['injected_h']['raw_sha256']==H_HASH['AB'.index(folder.name[-1])]
            storage.add(v['h_storage_id_within_process']);versions[process].append(v['h_version'])
            if first_rng is None:first_rng=rng['before']
            assert _rng_equal(first_rng,rng['before']) and _rng_equal(first_rng,rng['after'])
            for k in ('initial','round_1','round_2'):
                for when in ('before','after'):
                    field='explicit_generators/'+when+'/'+k
                    assert compare_tensor(t[field],reference[field])['bitwise_equal']
                unchanged=compare_tensor(t['explicit_generators/before/'+k],t['explicit_generators/after/'+k])['bitwise_equal']
                assert unchanged==(k!='initial')
            calls[name]=t
        assert len(storage)==1 and versions[process]==[1,2,3,4]
    non_h_keys=[k for k in next(iter(calls.values())) if k.startswith('GDTS._jdv2_goal_outputs/000/input/inputs/')]
    summary={};pair_equal=[]
    for a,b in itertools.combinations(calls,2):
        ta,tb=calls[a],calls[b];group='same_'+a[-1] if a[-1]==b[-1] else 'cross_AB'
        assert all(compare_tensor(ta[k],tb[k])['bitwise_equal'] for k in non_h_keys)
        for r in (0,1,2):
            assert compare_tensor(ta[f'callback/round{r}/mask'],tb[f'callback/round{r}/mask'])['bitwise_equal']
            k=f'callback/round{r}/score';v=compare_tensor(ta[k],tb[k]);row=summary.setdefault(str(r),{}).setdefault(group,{'pairs':0,'different_pairs':0,'max_abs':0,'max_different_elements':0})
            row['pairs']+=1;row['different_pairs']+=not v['bitwise_equal'];row['max_abs']=max(row['max_abs'],v.get('max_abs',0) or 0);row['max_different_elements']=max(row['max_different_elements'],v.get('different_elements',0))
        equal=compare_tensor(ta['ParallelConditionalSampler.forward/000/output/goals'],tb['ParallelConditionalSampler.forward/000/output/goals'])['bitwise_equal'];assert equal
        pair_equal.append({'a':a,'b':b,'final_goals_equal':equal})
    return {'status':'PASS','call_count':len(calls),'pair_count':len(pair_equal),'same_non_h_inputs_all_pairs':True,'same_masks_all_rounds':True,
        'model_parameters_buffers_inputs_identical_across_processes':True,'settings_identical_across_processes':True,'global_RNG_all_calls_equal_and_unchanged':True,
        'actual_generators_match_P2':True,'initial_generator_consumed_round_generators_unchanged':True,
        'h_versions_by_process':versions,'one_shared_h_storage_per_process':True,'round_scores':summary,
        'all_final_goals_equal':True,'actual_CPU_exact_replays':0,'extra_GPU_calls':0,
        'note':'CPU post-run comparison of existing payload only, added after GPU completion; not part of GPU source or pre-run CPU semantic tests'}


if __name__=='__main__':print(json.dumps(check(),indent=2))

#!/usr/bin/env python3
"""CPU-only all-call comparisons; gated once-per-problem original exact audit."""
import argparse
import itertools
import json
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from tools.jdv2_h_sampler_contract import ROOT,BASE,OLD,load,read,sha256,compare_tensor
from tools.jdv2_stage_a_bank import write_json
from src.models.joint_dependency_v2 import exact_lexicographic_assignment as exact


def key_round(r):return f'callback/round{r}/candidate_index' if r!='final' else 'ParallelConditionalSampler.forward/000/output/candidate_index'

def id_compare(a,b):
    agents=[]
    for i in range(a.shape[0]):
        va,vb=a[i].tolist(),b[i].tolist();same=sorted(va)==sorted(vb)
        agents.append({'agent':i,'different_slots':sum(x!=y for x,y in zip(va,vb)),'candidate_multiset_equal':same,
            'removed':sorted(set(va)-set(vb)),'added':sorted(set(vb)-set(va))})
    return {'bitwise_equal':torch.equal(a,b),'different_slots':int((a!=b).sum()),'per_agent':agents}


def exact_audit(calls,round_index):
    result={'status':'COMPLETE','problems':[],'attempted':0,'completed':0,'failed':0,'unknown_inflight':0,
        'representatives':['P0/call1_A','P0/call2_B'],'round':round_index}
    tapes={label:calls[name] for label,name in zip('AB',result['representatives'])}
    start=(round_index-1)*9
    for agent in range(9):
        prefix=f'solve_exact_persistent_tie/{start+agent:03d}/'
        assignments={label:tuple(map(int,v[1]['values'][prefix+'output/assignment'])) for label,v in tapes.items()}
        common={}
        for field in ('mask','previous_ids','goal_candidates_world'):
            common[field]=compare_tensor(tapes['A'][0][prefix+'input/'+field],tapes['B'][0][prefix+'input/'+field])['bitwise_equal']
        priority_map={label:tuple(tuple(map(int,v[1]['values'][prefix+f'output/priorities/{i}'])) for i in range(20)) for label,v in tapes.items()}
        common['priorities']=priority_map['A']==priority_map['B']
        for label,(tensor,meta) in tapes.items():
            dest=ROOT/'cpu_exact'/f'round{round_index}_agent{agent}_{label}.json'
            if dest.exists():
                row=read(dest);result['problems'].append(row);result['attempted']+=1;result['completed']+=1;continue
            attempt=dest.with_suffix('.attempt.json')
            if attempt.exists():raise RuntimeError('Do not retry previously attempted CPU problem')
            score,mask,previous,goals=[tensor[prefix+'input/'+k] for k in ('score','mask','previous_ids','goal_candidates_world')]
            result['attempted']+=1;write_json(attempt,{'agent':agent,'h':label,'round':round_index,'attempted':1,'unknown_inflight':1})
            solved=exact.solve_exact_persistent_tie(score,mask,previous,goals,priorities=priority_map[label])
            expected=meta['values'];matches=(solved.assignment==assignments[label] and tuple(map(str,solved.objective))==tuple(expected[prefix+'output/objective']) and str(solved.encoded_objective)==expected[prefix+'output/encoded_objective'])
            row={'agent':agent,'h':label,'round':round_index,'matches_actual_assignment_and_objectives':matches,'common_AB_problem_fields':common,
                'actual_assignment':list(assignments[label]),'objective':[str(x) for x in solved.objective],'encoded_objective':str(solved.encoded_objective)}
            if assignments['A']!=assignments['B']:
                lift=exact.lift_fp32_to_integers(score);geometry,geometry_unit=exact.exact_negative_squared_geometry(goals,previous)
                stay=tuple(tuple(int(k==int(previous[p])) for k in range(score.shape[1])) for p in range(score.shape[0]))
                values={}
                for candidate_label,assignment in assignments.items():
                    feasible=len(set(assignment))==len(assignment) and all(0<=k<score.shape[1] and bool(mask[p,k]) for p,k in enumerate(assignment))
                    values[candidate_label]={'feasible':feasible}
                    if feasible:
                        objective=exact._exact_objective(assignment,lift.values,stay,geometry,priority_map[label])
                        values[candidate_label]['objective']=[str(x) for x in objective]
                        values[candidate_label]['J_exact']=str(objective[0]*lift.unit)
                row.update(cross_assignment_evaluation=values,primary_unit=str(lift.unit),geometry_unit=str(geometry_unit))
                other='B' if label=='A' else 'A'
                if values['A']['feasible'] and values['B']['feasible']:
                    own=tuple(map(int,values[label]['objective']));alternative=tuple(map(int,values[other]['objective']))
                    row['own_minus_other_J_exact']=str((own[0]-alternative[0])*lift.unit)
                    row['first_objective_level_separating']=next((name for name,x,y in zip(('J','C_stay','C_geom','R'),own,alternative) if x!=y),None)
            write_json(dest,row);result['problems'].append(row);result['completed']+=1
            if not matches:result['status']='CPU_REPLAY_MISMATCH';return result
    return result


def analyze(destination):
    processes={};calls={};validation={};contracts={};summary_calls={}
    for p in ('P0','P1'):
        folder=ROOT/p
        if not folder.exists():continue
        terminal=next((folder/n for n in ('COMPLETE.json','FAILED.json') if (folder/n).exists()),None)
        processes[p]={'terminal':read(terminal) if terminal else {'status':'UNKNOWN_IN_FLIGHT'},
            'initialization':read(folder/'INITIALIZED.json') if (folder/'INITIALIZED.json').exists() else None,
            'resources':{f.name:read(f) for f in sorted((folder/'resources').glob('*.json'))}}
        for call in sorted(folder.glob('call*')):
            if not (call/'TRACE.pt').exists():continue
            name=p+'/'+call.name;tensor=load(call/'TRACE.pt');meta=read(call/'TRACE_METADATA.json');calls[name]=(tensor,meta)
            validation[name]=read(call/'VALIDATION.json') if (call/'VALIDATION.json').exists() else {'checks':{'missing_validation':False}}
            contracts[name]={k:v for k,v in meta.items() if k!='values'}
            contracts[name]['exact_values_local_file']=str(call/'TRACE_METADATA.json')
            summary_calls[name]={'h':call.name[-1],'validation':validation[name],
                'ids':{str(r):tensor[key_round(r)].tolist() for r in (0,1,2,'final')},
                'key_dtypes':{k:str(v.dtype) for k,v in tensor.items() if k in ('injected_h','UnaryGoalResidual.forward/000/output/score','UnaryGoalResidual.forward/000/output/residual','RelationInference.forward/000/output/0','RelationInference.forward/000/output/1')},
                'actual_exact_objectives':{k:v for k,v in meta['values'].items() if k.startswith('solve_exact_persistent_tie/') and any(k.endswith('/'+f) for f in ('assignment','objective','encoded_objective'))}}
    counts={k:sum(v['terminal'].get(k,0) for v in processes.values()) for k in set().union(*(set(v['terminal']) for v in processes.values())) if isinstance(next((v['terminal'][k] for v in processes.values() if k in v['terminal']),None),int)}
    ready=len(calls)==8 and len(processes)==2 and all(v['terminal']['status']=='COMPLETE' for v in processes.values())
    gates=ready and all(all(x['checks'].values()) for x in validation.values())
    rounds=(0,1,2,'final');id_pairs=[];continuous={};same_stable={'A':True,'B':True};cross_equal={str(r):True for r in rounds}
    for a,b in itertools.combinations(calls,2):
        ta,ma=calls[a];tb,mb=calls[b];group='same_'+a[-1] if a[-1]==b[-1] else 'cross_AB'
        row={'a':a,'b':b,'group':group,'rounds':{str(r):id_compare(ta[key_round(r)],tb[key_round(r)]) for r in rounds}}
        if group.startswith('same_'):same_stable[a[-1]] &= all(x['bitwise_equal'] for x in row['rounds'].values())
        else:
            for r in rounds:cross_equal[str(r)] &= row['rounds'][str(r)]['bitwise_equal']
        id_pairs.append(row)
        for k in ta:
            if not ta[k].is_floating_point() or k.startswith('solve_exact_persistent_tie/') or k.startswith('GDTS._jdv2_goal_outputs/') or '/input/' in k or k.startswith('callback/'):continue
            comparison=compare_tensor(ta[k],tb[k]);bucket=continuous.setdefault(k,{}).setdefault(group,{'pairs':0,'different_pairs':0,'max_different_elements':0,'max_abs':0})
            bucket['pairs']+=1;bucket['different_pairs']+=not comparison['bitwise_equal'];bucket['max_different_elements']=max(bucket['max_different_elements'],comparison.get('different_elements',0));bucket['max_abs']=max(bucket['max_abs'],comparison.get('max_abs',0) or 0)
    actual_random_equal=gates and all(v['checks']['actual_random_payload_equal_P2'] for v in validation.values())
    if gates:
        first=next((r for r in (0,1,2) if not cross_equal[str(r)]),None)
        if not all(same_stable.values()):status='INCONCLUSIVE_DOWNSTREAM_ID_VARIATION'
        elif first is None:status='NO_ID_CHANGE_OBSERVED_FOR_PREDECLARED_H_PAIR'
        elif not cross_equal['final']:status='REPEATABLE_CONTROLLED_H_REPLACEMENT_ID_CHANGE'
        else:status='CONTROLLED_TRANSIENT_ID_CHANGE_FINAL_EQUAL'
    else:status='CONTRACT_FAILED_OR_INPUT_BLOCKED';first=None
    cpu={'status':'NOT_RUN','attempted':0,'completed':0,'failed':0,'unknown_inflight':0,'reason':'Requires stable same-h round chains and first discrete A/B difference in exact round1/2'}
    if gates and all(same_stable.values()) and first in (1,2):
        cpu=exact_audit(calls,first)
        if cpu['status']!='COMPLETE':status='CPU_REPLAY_MISMATCH_NO_CONCLUSION_UPGRADE'
    results={'task_status':'COMPLETE_BOUNDED_DIAGNOSTIC' if gates else 'STOPPED_CONTRACT_OR_INPUT',
        'base_commit':BASE,'source_commit':next((v['terminal'].get('source_commit') for v in processes.values() if v['terminal'].get('source_commit')),None),
        'controlled_propagation_result':status,'same_h_ID_stability':same_stable if ready else None,
        'actual_random_payload_equal':actual_random_equal if ready else None,'observer_semantic_snapshot_contract':'PASS_CHECKED_PROPERTIES_ONLY' if gates else 'FAILED_OR_UNAVAILABLE',
        'canonical_GPU_trace_neutrality':'TRACE_NEUTRALITY_UNCERTIFIED','historical_root_cause':'OPEN',
        'causal_link_to_previous29_ID_changes':'NOT_ESTABLISHED','pair_energy_causal_contribution':'NOT_ESTABLISHED',
        'counts':counts,'calls':summary_calls,'ID_pairs':id_pairs,'continuous_comparisons':continuous,
        'first_discrete_AB_difference_round':first if all(same_stable.values()) else None,
        'cross_AB_round_equality':cross_equal if ready else None,'cpu_exact_review':cpu,
        'unrun':{'Y_JADE_JFDE':'not authorized; no trajectory generation','energy_ablation':'not authorized','kernel_diagnostic':'not run','alternate_h_seed_window':'not authorized'},
        'limits':['Only predeclared h A/B and fixed actual P2 inputs','Same observer on all arms; not historical allocator/stream restoration','Candidate multiset equality does not establish trajectory multiset equality','Stable IDs do not imply continuous payload bitwise determinism']}
    write_json(destination/'RESULTS.json',results);write_json(destination/'EXECUTION.json',{'counts':counts,'processes':processes})
    write_json(destination/'TRACE_CONTRACTS.json',contracts)
    write_json(destination/'INPUT_MANIFEST.json',read(ROOT/'P0/INPUT_MANIFEST.json') if (ROOT/'P0/INPUT_MANIFEST.json').exists() else read(ROOT/'CPU_CHECKS.json')['input_authentication'])
    write_json(destination/'CPU_CHECKS.json',read(ROOT/'CPU_CHECKS.json'))
    write_json(destination/'ARTIFACT_MANIFEST.json',{'local_root':str(ROOT),'files':{str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in sorted(ROOT.rglob('*')) if p.is_file()}})
    print(json.dumps({k:results[k] for k in ('task_status','controlled_propagation_result','counts','same_h_ID_stability','first_discrete_AB_difference_round','cross_AB_round_equality')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path);analyze(parser.parse_args().output)

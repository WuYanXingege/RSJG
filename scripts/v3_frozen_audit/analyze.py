"""Three independent native Hungarian solves on one captured trained forward."""
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from replay import SOURCE,OUT,GUARD,install_no_training_guard,endpoint_error,write_csv
from matching import value,solve,second_best_gap,center,describe,assignment_change,gap_bin,people_bin,proximity_bin
from bcep_joint.common import read,write
from bcep.contracts import digest,canonical_hash
from bcep_v2.runtime import configure_cuda
from bcep_v2.joint import assignment,neighbors
from ebjd.unified import gather_candidates

DOC = SOURCE/'docs/research/bcep_v3/frozen_feedback_audit_20261121'
NEAR_TIE = 1e-6  # Raw assignment-score sum units; fixed before looking at results.


def csvg(path):
    with path.open() as f:return list(csv.DictReader(f))


def subsets(rows):
    for label,pred in [('all',lambda r:True),('multi',lambda r:r['n_agents']>1),('N_ge_9',lambda r:r['n_agents']>=9)]:
        yield label,[r for r in rows if pred(r)]


def error_summary(rows,group):
    result=dict(group=group,windows=len(rows))
    for mode in 'ABC':result[mode+'_JFDE']=float(np.mean([r[mode+'_JFDE'] for r in rows]))
    for trans in ['B_minus_A','C_minus_B','C_minus_A']:
        d=np.array([r[trans] for r in rows])
        result.update({trans+'_mean':float(d.mean()),trans+'_median':float(np.median(d)),
            trans+'_improved':int((d<0).sum()),trans+'_worse':int((d>0).sum()),trans+'_equal':int((d==0).sum()),
            trans+'_abs_le_1e-6':int((np.abs(d)<=1e-6).sum()),
            trans+'_improved_gt_1cm':int((d<-.01).sum()),trans+'_worse_gt_1cm':int((d>.01).sum())})
    total=sum(r['BC_total_choices'] for r in rows);changed=sum(r['BC_changed_choices'] for r in rows)
    result.update(BC_changed_choices=changed,BC_total_choices=total,BC_changed_choice_ratio=changed/total,
        BC_windows_with_choice_changes=sum(r['BC_changed_choices']>0 for r in rows),
        BC_changed_agents=sum(r['BC_changed_agents'] for r in rows),total_agents=sum(r['n_agents'] for r in rows),
        BC_mean_window_changed_ratio=float(np.mean([r['BC_changed_choices']/r['BC_total_choices'] for r in rows])),
        BC_windows_with_cross_person_change=sum(r['BC_relative_changed_choices']>0 for r in rows),
        BC_uniform_world_relabel_only_windows=sum(r['BC_uniform_world_relabel_only'] for r in rows))
    return result


@torch.no_grad()
def main():
    install_no_training_guard();configure_cuda();torch.set_num_threads(1)
    proof=read(OUT/'REPLAY_PROOF.json');assert proof['archive_per_window_JFDE_exact']
    assert proof['guard']==dict(optimizer_constructions=0,optimizer_step_calls=0,backward_calls=0)
    assert not DOC.exists(), 'Refuse to overwrite an audit publication'
    files=read(OUT/'WINDOW_FILES.json');assert len(files)==590
    replay={int(r['index']):r for r in csvg(OUT/'REPLAY_WINDOWS.csv')}
    archived={int(r['index']):r for r in csvg(SOURCE/'docs/research/bcep_v3/zara2_trial_20261120/RANK1_WINDOWS.csv')}
    # Gate all 590 windows BEFORE any comparative attribution or metric aggregation.
    records=[];start=time.time()
    for file in files:
        path=OUT/file['file'];assert digest(path)==file['sha256']
        with np.load(path,allow_pickle=False) as z:d={k:z[k] for k in z.files}
        valid=torch.from_numpy(d['valid']).cuda()
        pi,_,_=assignment(torch.from_numpy(d['C_scores']).cuda(),valid,False)
        assert np.array_equal(pi.cpu().numpy(),d['C_pi'])
        goals=torch.from_numpy(d['goals']).cuda()
        truth=torch.from_numpy(d['truth_endpoint']).cuda()[:,:,None,:]
        c=float(endpoint_error(gather_candidates(goals,pi),truth,valid))
        assert c==float(replay[file['index']]['C_JFDE'])==float(d['native_H_JFDE'])
        d['index']=file['index'];records.append(d)
    gate=dict(windows=590,C_permutation_exact_vs_native_replay=True,C_JFDE_exact_vs_archive=True,
        history_permutation_archive_exists=False,user_approved_native_replay_reference=True,
        near_tie_score_sum_threshold=NEAR_TIE,optimizer_updates=0)
    write(OUT/'C_GATE.json',gate)
    print('C gate PASS: 590 exact native-replay permutations and archive JFDE',flush=True)
    rows=[];agents=[];choices=[];worlds=[];relabels=[];all_scores=[]
    for pos,d in enumerate(records):
        idx=d['index'];n=int(d['valid'].sum());group=people_bin(n)
        valid=torch.from_numpy(d['valid']).cuda();goals=torch.from_numpy(d['goals']).cuda()
        truth=torch.from_numpy(d['truth_endpoint']).cuda()[:,:,None,:]
        pis={};metrics={};world_errors={}
        for mode in 'ABC':
            # Three separate calls to the unchanged original Hungarian path.
            pi,_,_=assignment(torch.from_numpy(d[mode+'_scores']).cuda(),valid,False)
            pis[mode]=pi[0].cpu().numpy()
            inference_sort=np.sort(pis[mode],axis=0)
            assert np.array_equal(inference_sort,np.broadcast_to(np.arange(20)[:,None],(20,n)))
            points=gather_candidates(goals,pi)
            error=(points-truth[:,None,:,-1]).norm(dim=-1)
            we=((error*valid[:,None]).sum(-1)/valid.sum(1)[:,None])[0]
            metrics[mode]=float(we.min());world_errors[mode]=we.cpu().numpy()
        assert np.array_equal(pis['C'],d['C_pi'][0]) and np.array_equal(pis['A'],d['A_native_pi'][0])
        assert metrics['C']==float(replay[idx]['C_JFDE'])
        assert metrics['A']==float(archived[idx]['I_JFDE'])
        if n==1:assert metrics['A']==metrics['B']==metrics['C']
        row=dict(index=idx,n_agents=n,people_bin=group,**{m+'_JFDE':metrics[m] for m in 'ABC'})
        for after,before in [('B','A'),('C','B'),('C','A')]:
            row[after+'_minus_'+before]=metrics[after]-metrics[before]
            row.update({before+after+'_'+k:v for k,v in assignment_change(pis[before],pis[after]).items()})
        for mode in 'ABC':row[mode+'_best_world']=int(world_errors[mode].argmin())
        rows.append(row)
        s={m:d[m+'_scores'][0] for m in 'ABC'}
        terms=dict(S0=s['A'],old_feedback=d['old_feedback'][0],HCRA_delta=d['HCRA_delta'][0],
            actual_HCRA_rounded=s['C'].astype(np.float64)-s['B'].astype(np.float64))
        all_scores.append(dict(n_agents=n,terms=terms))
        for agent in range(n):
            a=dict(index=idx,person=agent,n_agents=n,people_bin=group,
                AB_changed_choices=int(np.count_nonzero(pis['A'][:,agent]!=pis['B'][:,agent])),
                BC_changed_choices=int(np.count_nonzero(pis['B'][:,agent]!=pis['C'][:,agent])))
            for mode in 'ABC':
                if n>1:
                    gap,runner=second_best_gap(s[mode][agent],pis[mode][:,agent])
                else:gap,runner=None,None
                a[mode+'_gap']=gap;a[mode+'_gap_class']=gap_bin(gap)
                if n>1 and gap<=NEAR_TIE:
                    matrix=s[mode][agent]
                    # Same entries, simultaneous deterministic row/column reindexing.
                    rp=np.arange(20)[::-1];cp=np.roll(np.arange(20),1)
                    perm=solve(matrix[rp][:,cp]);back=np.empty(20,np.int64);back[rp]=cp[perm]
                    changed=int(np.count_nonzero(back!=pis[mode][:,agent]))
                    alternate=pis[mode].copy();alternate[:,agent]=back
                    alternate_pi=torch.from_numpy(alternate[None]).cuda()
                    alternate_jfde=float(endpoint_error(gather_candidates(goals,alternate_pi),truth,valid))
                    relabels.append(dict(index=idx,person=agent,n_agents=n,mode=mode,gap=gap,
                        gap_class=gap_bin(gap),changed_after_row_column_relabel=changed,
                        original_minus_relabel_objective=value(matrix,pis[mode][:,agent])-value(matrix,back),
                        original_JFDE=metrics[mode],one_person_relabel_JFDE=alternate_jfde,
                        one_person_relabel_delta_JFDE=alternate_jfde-metrics[mode]))
            for name,t in terms.items():
                a[name+'_mean']=float(t[agent].mean());a[name+'_std']=float(t[agent].std())
                a[name+'_max_abs']=float(np.abs(t[agent]).max())
                projected=center(t[agent]);a[name+'_centered_rms']=float(np.sqrt(np.mean(projected**2)))
                a[name+'_centered_max_abs']=float(np.abs(projected).max())
            bpi,cpi=pis['B'][:,agent],pis['C'][:,agent]
            a['B_score_loss_of_C']=value(s['B'][agent],bpi)-value(s['B'][agent],cpi)
            delta=terms['actual_HCRA_rounded'][agent]
            a['S0_advantage_C_vs_B']=value(s['A'][agent],cpi)-value(s['A'][agent],bpi)
            old_actual=s['B'][agent].astype(np.float64)-s['A'][agent].astype(np.float64)
            a['actual_old_feedback_advantage_C_vs_B']=value(old_actual,cpi)-value(old_actual,bpi)
            a['raw_old_feedback_advantage_C_vs_B']=value(terms['old_feedback'][agent],cpi)-value(terms['old_feedback'][agent],bpi)
            old_center=center(terms['old_feedback'][agent]);hcra_center=center(terms['HCRA_delta'][agent])
            den=float(np.linalg.norm(old_center)*np.linalg.norm(hcra_center))
            a['old_HCRA_centered_score_cosine']=float((old_center*hcra_center).sum()/den) if den>0 else None
            a['actual_HCRA_advantage_of_C']=value(delta,cpi)-value(delta,bpi)
            a['raw_HCRA_advantage_of_C']=value(terms['HCRA_delta'][agent],cpi)-value(terms['HCRA_delta'][agent],bpi)
            a['C_objective_gain_vs_B']=value(s['C'][agent],cpi)-value(s['C'][agent],bpi)
            if n>1:
                assert a['B_score_loss_of_C']>=0 and a['C_objective_gain_vs_B']>=0
                assert abs(a['C_objective_gain_vs_B']-(a['actual_HCRA_advantage_of_C']-a['B_score_loss_of_C']))<1e-10
                assert abs(a['S0_advantage_C_vs_B']+a['actual_old_feedback_advantage_C_vs_B']+a['B_score_loss_of_C'])<1e-10
            bound=40*a['actual_HCRA_rounded_centered_max_abs']
            a['HCRA_assignment_difference_upper_bound']=bound
            a['strict_stability_certificate']=bool(n>1 and a['B_gap']>bound)
            if a['strict_stability_certificate']:assert a['BC_changed_choices']==0
            agents.append(a)
        errs=(goals-truth[:,None,:,-1]).norm(dim=-1)[0].cpu().numpy()
        b,receiver,sender=neighbors(torch.from_numpy(d['observed']).cuda(),valid)
        receiver=receiver.cpu().numpy();sender=sender.cpu().numpy()
        base=d['baseline'][0].astype(np.float64);g=d['goals'][0].transpose(1,0,2).astype(np.float64)
        # Descriptive physical reference paths; no geometry is fed back into the model.
        ref=base[:,None]+np.arange(1,13)[None,None,:,None]/12*(g-base[:,-1,None])[:,:,None]
        initial_paths=np.stack([ref[i,pis['A'][:,i]] for i in range(n)],axis=1)
        last=d['observed'][0,:,-1].astype(np.float64)
        for person in range(n):
            nb=sender[receiver==person]
            for world in range(20):
                worlds.append(dict(index=idx,person=person,world=world,n_agents=n,
                    A=int(pis['A'][world,person]),B=int(pis['B'][world,person]),C=int(pis['C'][world,person])))
            for before,after in [('A','B'),('B','C')]:
                changed=np.flatnonzero(pis[before][:,person]!=pis[after][:,person])
                for world in changed:
                    old=int(pis[before][world,person]);new=int(pis[after][world,person])
                    if len(nb):
                        near_old=float(np.linalg.norm(ref[person,old][None]-initial_paths[world,nb],axis=-1).min())
                        near_new=float(np.linalg.norm(ref[person,new][None]-initial_paths[world,nb],axis=-1).min())
                        obs_near=float(np.linalg.norm(last[nb]-last[person],axis=-1).min())
                    else:near_old=near_new=obs_near=float('nan')
                    choices.append(dict(index=idx,person=person,world=int(world),n_agents=n,people_bin=group,transition=before+after,
                        old_candidate=old,new_candidate=new,old_FDE=float(errs[old,person]),new_FDE=float(errs[new,person]),
                        new_minus_old_FDE=float(errs[new,person])-float(errs[old,person]),
                        endpoint_shift_m=float(np.linalg.norm(g[person,new]-g[person,old])),
                        neighbors=len(nb),observed_nearest_neighbor_m=obs_near,
                        initial_A_neighbor_reference_min_before_m=near_old,initial_A_neighbor_reference_min_after_m=near_new,
                        proximity_before=proximity_bin(near_old),
                        is_old_best_world=bool(world==row[before+'_best_world']),is_new_best_world=bool(world==row[after+'_best_world'])))
        if (pos+1)%100==0:print(f'analyzed {pos+1}/590; elapsed {time.time()-start:.1f}s',flush=True)
    group_rows=[error_summary(rr,label) for label,rr in subsets(rows)]
    for label in ['1','2-4','5-8','9-12','13-20','21+']:
        rr=[r for r in rows if r['people_bin']==label]
        if rr:group_rows.append(error_summary(rr,'N_'+label))
    for n in sorted({r['n_agents'] for r in rows}):
        group_rows.append(error_summary([r for r in rows if r['n_agents']==n],'N_exact_'+str(n)))
    score_rows=[]
    for group,rr in subsets(all_scores):
        for term in rr[0]['terms']:
            for projection in ['raw','row_column_centered']:
                values=np.concatenate([(r['terms'][term] if projection=='raw' else center(r['terms'][term])).ravel() for r in rr])
                score_rows.append(dict(group=group,term=term,projection=projection,**describe(values)))
    gap_rows=[]
    for group,rr in subsets(agents):
        for label in dict.fromkeys(r['B_gap_class'] for r in rr):
            aa=[r for r in rr if r['B_gap_class']==label]
            gap_rows.append(dict(group=group,B_gap_class=label,agents=len(aa),
                changed_agents=sum(r['BC_changed_choices']>0 for r in aa),
                changed_agent_ratio=sum(r['BC_changed_choices']>0 for r in aa)/len(aa),
                changed_choices=sum(r['BC_changed_choices'] for r in aa),choices=len(aa)*20,
                mean_HCRA_centered_rms=float(np.mean([r['HCRA_delta_centered_rms'] for r in aa])),
                stability_certified_agents=sum(r['strict_stability_certificate'] for r in aa)))
    relation_rows=[]
    for trans in ['AB','BC']:
        rr=[r for r in choices if r['transition']==trans]
        for field in ['people_bin','proximity_before']:
            for label in dict.fromkeys(r[field] for r in rr):
                aa=[r for r in rr if r[field]==label];delta=np.array([r['new_minus_old_FDE'] for r in aa])
                relation_rows.append(dict(transition=trans,stratify=field,stratum=label,changed_choices=len(aa),
                    distinct_windows=len({r['index'] for r in aa}),mean_candidate_FDE_delta_m=float(delta.mean()),
                    median_candidate_FDE_delta_m=float(np.median(delta)),better_candidates=int((delta<0).sum()),
                    worse_candidates=int((delta>0).sum()),equal_candidates=int((delta==0).sum()),
                    mean_endpoint_shift_m=float(np.mean([r['endpoint_shift_m'] for r in aa]))))
    cooperation=[]
    for group,rr in subsets(agents):
        switched=[r for r in rr if r['BC_changed_choices']>0]
        cooperation.append(dict(group=group,switched_agents=len(switched),
            old_feedback_supports_C=sum(r['actual_old_feedback_advantage_C_vs_B']>0 for r in switched),
            old_feedback_opposes_C=sum(r['actual_old_feedback_advantage_C_vs_B']<0 for r in switched),
            old_feedback_neutral_C=sum(r['actual_old_feedback_advantage_C_vs_B']==0 for r in switched),
            HCRA_strictly_prefers_C=sum(r['actual_HCRA_advantage_of_C']>0 for r in switched),
            HCRA_neutral_C=sum(r['actual_HCRA_advantage_of_C']==0 for r in switched)))
    bridge=[]
    replay_rows=[dict(index=int(r['index']),n_agents=int(r['n_agents']),on=float(r['GRG_on_H_JADE']),off=float(r['GRG_off_H_JADE'])) for r in replay.values()]
    for group,rr in subsets(replay_rows):
        delta=np.array([r['on']-r['off'] for r in rr])
        bridge.append(dict(group=group,windows=len(rr),on_H_JADE=float(np.mean([r['on'] for r in rr])),
            off_H_JADE=float(np.mean([r['off'] for r in rr])),on_minus_off_mean_m=float(delta.mean()),
            on_minus_off_median_m=float(np.median(delta)),on_better_windows=int((delta<0).sum()),
            on_worse_windows=int((delta>0).sum()),equal_windows=int((delta==0).sum())))
    examples=[]
    for trans in ['B_minus_A','C_minus_B','C_minus_A']:
        for sign,label in [(1,'largest_harm'),(-1,'largest_benefit')]:
            selected=sorted((r for r in rows if r[trans]*sign>0),key=lambda r:r[trans]*sign,reverse=True)[:5]
            examples.extend(dict(reason=trans+'_'+label,**r) for r in selected)
    DOC.mkdir(parents=True)
    write_csv(DOC/'THREE_WAY_WINDOWS.csv',rows);write_csv(DOC/'GROUPS.csv',group_rows)
    write_csv(DOC/'AGENT_MATCHING.csv',agents);write_csv(DOC/'WORLD_CANDIDATES.csv',worlds)
    write_csv(DOC/'CHANGED_CHOICES.csv',choices);write_csv(DOC/'RELATION_GROUPS.csv',relation_rows)
    write_csv(DOC/'COOPERATION.csv',cooperation)
    write_csv(DOC/'SCORE_DISTRIBUTIONS.csv',score_rows);write_csv(DOC/'GAP_GROUPS.csv',gap_rows)
    if relabels:write_csv(DOC/'TIE_RELABEL_CHECKS.csv',relabels)
    else:(DOC/'TIE_RELABEL_CHECKS.csv').write_text('index,person,n_agents,mode,gap,gap_class,changed_after_row_column_relabel,original_minus_relabel_objective,original_JFDE,one_person_relabel_JFDE,one_person_relabel_delta_JFDE\n')
    write_csv(DOC/'KEY_EXAMPLES.csv',examples);write_csv(DOC/'GRG_GROUPS.csv',bridge)
    write_csv(DOC/'REPLAY_WINDOWS.csv',list(replay.values()))
    write(DOC/'REPLAY_PROOF.json',proof);write(DOC/'C_GATE.json',gate)
    summary=dict(groups=group_rows[:3],GRG=bridge,agent_matrices=len(agents),choice_slots=len(worlds),
        near_tie_threshold=NEAR_TIE,strict_ties={m:sum(r[m+'_gap']==0 for r in agents) for m in 'ABC'},
        near_unique={m:sum(r[m+'_gap'] is not None and 0<r[m+'_gap']<=NEAR_TIE for r in agents) for m in 'ABC'},
        relabel_checks=len(relabels),relabel_changed=sum(r['changed_after_row_column_relabel']>0 for r in relabels),
        optimizer_guard=GUARD,model_constructions_in_analysis=0,real_optimizer_updates=0,synthetic_optimizer_updates=0,
        elapsed_seconds=time.time()-start)
    write(DOC/'SUMMARY.json',summary)
    (DOC/'.gitattributes').write_text('*.csv -text whitespace=cr-at-eol\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()

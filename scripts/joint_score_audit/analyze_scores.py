"""Spectra, complete-assignment gaps and native Sinkhorn on frozen score caches."""
import math,time
import numpy as np
import torch
from common import *
setup()
sys.path.insert(0,str(SOURCE/'scripts/v3_frozen_audit'))
from matching import second_best_gap,solve,value
from ebjd.unified import log_sinkhorn




@torch.no_grad()
def main():
    from bcep_v2.runtime import configure_cuda
    configure_cuda();DOC.mkdir(parents=True,exist_ok=True)
    v3agents={(int(r['index']),int(r['person'])):r for r in csvread(SOURCE/'docs/research/bcep_v3/frozen_feedback_audit_20261121/AGENT_MATCHING.csv')}
    matrix=[];spectra=[];softrows=[];features=[];relabels=[];start=time.time()
    ids=[r['index'] for r in read(REPLAY/'WINDOW_FILES.json')]
    for label in MODELS:
        assert read(OUT/label/'PROOF.json')['weights_unchanged']
        with np.load(OUT/label/'STATIC_FEATURES.npz') as z:q=feature_stats(z['queries'])
        for index in ids:
            rec=score_record(label,index);n=len(rec['final'])
            with np.load(OUT/label/'windows'/f'{index}.npz') as z:
                fw=feature_stats(z['world']);fc=feature_stats(z['candidate'])
            features.append(dict(model=label,index=index,n_agents=n,people_bin=people(n),
                **{'query_'+k:v for k,v in q.items()},**{'world_'+k:v for k,v in fw.items()},**{'candidate_'+k:v for k,v in fc.items()}))
            for stage,pi_name,old in [('initial','pi0','A'),('final','pi','C')]:
                s=rec['S0' if stage=='initial' else 'final'];pi=rec[pi_name]
                tensor=torch.from_numpy(s).cuda()
                p=log_sinkhorn(tensor,20)
                if n==1:p=torch.eye(20,device='cuda')[None]
                p=array(p).astype(np.float64)
                eff=center(s);sv=effective_singular_values(s);raw_sv=np.linalg.svd(s.astype(np.float64),compute_uv=False)
                for person in range(n):
                    if n==1:gap=None
                    elif label=='v3_e12':gap=float(v3agents[index,person][old+'_gap'])
                    else:gap,_=second_best_gap(s[person],pi[:,person])
                    gap_class='singleton' if n==1 else 'strict' if gap==0 else 'near' if gap<=1e-6 else 'separated'
                    vals=sv[person];power=vals**2;energy=power.sum();raw_energy=np.square(s[person].astype(np.float64)).sum()
                    prob=power/energy if energy else np.zeros_like(power)
                    rank64=int((vals>vals[0]*20*np.finfo(np.float64).eps).sum()) if vals[0]>0 else 0
                    rank32=int((vals>vals[0]*20*np.finfo(np.float32).eps).sum()) if vals[0]>0 else 0
                    floor=20*np.finfo(np.float32).eps*float(np.max(np.abs(s[person])))
                    row=dict(model=label,stage=stage,index=index,person=person,n_agents=n,people_bin=people(n),
                        raw_rms=float(np.sqrt(raw_energy/400)),effective_rms=float(np.sqrt(energy/400)),
                        effective_energy_fraction=float(energy/raw_energy) if raw_energy else 0.,
                        raw_rank64=int((raw_sv[person]>raw_sv[person,0]*20*np.finfo(np.float64).eps).sum()),
                        effective_rank64=rank64,effective_rank32_relative=rank32,
                        singular_values_above_input_roundoff_bound=int((vals>floor).sum()),input_roundoff_bound=floor,
                        stable_rank=float(energy/vals[0]**2) if vals[0] else 0.,
                        entropy_rank=float(np.exp(-np.sum(prob[prob>0]*np.log(prob[prob>0])))) if energy else 0.,
                        energy_top1=float(prob[:1].sum()),energy_top2=float(prob[:2].sum()),energy_top5=float(prob[:5].sum()),
                        energy95_rank=int(np.searchsorted(np.cumsum(prob),.95)+1) if energy else 0,
                        gap=gap,gap_class=gap_class,gap_reused=label=='v3_e12')
                    matrix.append(row)
                    spectra.append(dict(model=label,stage=stage,index=index,person=person,**{'sigma_'+str(j+1):float(x) for j,x in enumerate(vals)}))
                    soft=p[person];row_sums=soft.sum(1);col_sums=soft.sum(0)
                    pp=soft/row_sums[:,None];logp=np.zeros_like(pp);np.log(pp,out=logp,where=pp>0)
                    entropy=-np.sum(pp*logp,axis=1)
                    hard=np.eye(20)[pi[:,person]]
                    softrows.append(dict(model=label,stage=stage,index=index,person=person,n_agents=n,people_bin=people(n),gap_class=gap_class,
                        uniform_fro=float(np.linalg.norm(soft-.05)),uniform_max_abs=float(np.abs(soft-.05).max()),
                        uniform_row_TV_mean=float(np.abs(soft-.05).sum(1).mean()/2),
                        entropy_mean=float(entropy.mean()),entropy_fraction_of_logK=float(entropy.mean()/np.log(20)),
                        maximum_element=float(soft.max()),minimum_element=float(soft.min()),
                        row_sum_max_error=float(np.abs(row_sums-1).max()),column_sum_max_error=float(np.abs(col_sums-1).max()),
                        hard_soft_row_TV=float(np.abs(soft-hard).sum()/40),hard_soft_fro=float(np.linalg.norm(soft-hard)),
                        soft_mass_on_hungarian=float(soft[np.arange(20),pi[:,person]].mean()),
                        row_argmax_agrees_with_hard=float((soft.argmax(1)==pi[:,person]).mean())))
                    if label.startswith('v2') and n>1:
                        rp=np.arange(20)[::-1];cp=np.roll(np.arange(20),1)
                        qpi=solve(s[person][rp][:,cp]);back=np.empty(20,np.int64);back[rp]=cp[qpi]
                        relabels.append(dict(model=label,stage=stage,index=index,person=person,n_agents=n,gap_class=gap_class,
                            changed_choices=int(np.count_nonzero(back!=pi[:,person])),
                            original_minus_remapped_score=value(s[person],pi[:,person])-value(s[person],back)))
        print('analyzed scores',label,'seconds',round(time.time()-start,2),flush=True)
    for r in csvread(SOURCE/'docs/research/bcep_v3/frozen_feedback_audit_20261121/TIE_RELABEL_CHECKS.csv'):
        if r['mode'] not in ['A','C']:continue
        relabels.append(dict(model='v3_e12',stage='initial' if r['mode']=='A' else 'final',index=int(r['index']),person=int(r['person']),n_agents=int(r['n_agents']),
            gap_class='strict' if float(r['gap'])==0 else 'near',changed_choices=int(r['changed_after_row_column_relabel']),
            original_minus_remapped_score=float(r['original_minus_relabel_objective'])))
    csvwrite(DOC/'SCORE_MATRICES.csv',matrix);csvwrite(DOC/'SINGULAR_VALUES.csv',spectra)
    csvwrite(DOC/'SOFT_HARD.csv',softrows);csvwrite(DOC/'FEATURES.csv',features);csvwrite(DOC/'RELABEL.csv',relabels)
    summaries=[]
    for label in MODELS:
        for stage in ['initial','final']:
            for group in ['all','single','multi','N_ge_9','N_2-4','N_5-8','N_9-12','N_13-20','N_21+']:
                rr=[r for r in matrix if r['model']==label and r['stage']==stage and group in groups(r['n_agents'])]
                ss=[r for r in softrows if r['model']==label and r['stage']==stage and group in groups(r['n_agents'])]
                if not rr:continue
                windows=sorted({r['index'] for r in rr})
                out=dict(model=label,stage=stage,group=group,windows=len(windows),matrices=len(rr),
                    strict_ties=sum(r['gap_class']=='strict' for r in rr),near_positive=sum(r['gap_class']=='near' for r in rr))
                for source,keys in [(rr,['raw_rms','effective_rms','effective_energy_fraction','stable_rank','entropy_rank','effective_rank64','effective_rank32_relative','singular_values_above_input_roundoff_bound','energy_top1','energy_top5']),
                                    (ss,['uniform_fro','uniform_max_abs','entropy_fraction_of_logK','maximum_element','hard_soft_row_TV','soft_mass_on_hungarian'])]:
                    for key in keys:
                        # Explicit window macro average; each scene counts once.
                        by_window={}
                        for r in source:by_window.setdefault(r['index'],[]).append(r[key])
                        out[key+'_scene_macro']=float(np.mean([np.mean(v) for v in by_window.values()]))
                gaps=[r['gap'] for r in rr if r['gap'] is not None]
                out['gap_agent_median']=float(np.median(gaps)) if gaps else None
                out['gap_agent_max']=max(gaps) if gaps else None
                summaries.append(out)
    csvwrite(DOC/'SCORE_SUMMARY.csv',summaries)
    selected=[]
    ff=[r for r in features if r['model']=='v3_e12']
    for group in ['1','2-4','5-8','9-12','13-20','21+']:
        inds=[r['index'] for r in ff if r['people_bin']==group]
        selected.extend(inds[j] for j in np.linspace(0,len(inds)-1,6,dtype=int))
    write(OUT/'GRADIENT_WINDOWS.json',sorted(selected))
    write(DOC/'REUSE_LEDGER.json',dict(v3_score_matrices_regenerated=0,v3_score_windows_reused=590,
        v3_initial_final_gap_matrices_reused=8346,v3_relabel_matrix_checks_reused=8346,
        v2_score_windows_new=1180,v2_historical_pi_windows_reused=1180,new_feature_windows=1770,DDIM_calls=0,
        gradient_windows=sorted(selected),optimizer_steps=0))
    write(DOC/'SCORE_ANALYSIS_DONE.json',dict(seconds=time.time()-start,optimizer_guard=GUARD))
    print('SCORE ANALYSIS COMPLETE',flush=True)

if __name__=='__main__':main()

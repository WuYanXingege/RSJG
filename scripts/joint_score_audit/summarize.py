"""Derive compact tables ONLY from audit CSVs; no model forward."""
from collections import defaultdict
from common import *


def stats(values):
    a=np.asarray(values,float)
    return dict(mean=float(a.mean()),minimum=float(a.min()),p05=float(np.quantile(a,.05)),
        median=float(np.median(a)),p95=float(np.quantile(a,.95)),maximum=float(a.max()))


def main():
    matrices=csvread(DOC/'SCORE_MATRICES.csv');soft=csvread(DOC/'SOFT_HARD.csv')
    relabel=csvread(DOC/'RELABEL.csv');features=csvread(DOC/'FEATURES.csv')
    tie=[];dist=[]
    for model in MODELS:
        for stage in ['initial','final']:
            for group in ['all','single','multi','N_ge_9','N_2-4','N_5-8','N_9-12','N_13-20','N_21+']:
                rr=[r for r in matrices if r['model']==model and r['stage']==stage and group in groups(int(r['n_agents']))]
                ss=[r for r in soft if r['model']==model and r['stage']==stage and group in groups(int(r['n_agents']))]
                ll=[r for r in relabel if r['model']==model and r['stage']==stage and group in groups(int(r['n_agents']))]
                for cls in ['singleton','strict','near','separated']:
                    aa=[r for r in ss if r['gap_class']==cls];bb=[r for r in ll if r['gap_class']==cls]
                    if not aa:continue
                    row=dict(model=model,stage=stage,group=group,tie_class=cls,matrices=len(aa),windows=len({r['index'] for r in aa}),
                        relabel_changed_matrices=sum(int(r['changed_choices'])>0 for r in bb),
                        relabel_changed_choices=sum(int(r['changed_choices']) for r in bb),
                        relabel_score_loss_max=max([abs(float(r['original_minus_remapped_score'])) for r in bb],default=0))
                    for key in ['uniform_fro','entropy_fraction_of_logK','maximum_element','row_sum_max_error','column_sum_max_error','hard_soft_row_TV','row_argmax_agrees_with_hard']:
                        row[key+'_agent_mean']=float(np.mean([float(r[key]) for r in aa]))
                        row[key+'_agent_max']=max(float(r[key]) for r in aa)
                    tie.append(row)
                for key in ['effective_rank64','effective_rank32_relative','singular_values_above_input_roundoff_bound','stable_rank','entropy_rank','energy_top1','energy_top2','energy_top5','energy95_rank']:
                    dist.append(dict(model=model,stage=stage,group=group,unit='agent_matrix',metric=key,**stats([r[key] for r in rr])))
        for group in ['all','multi','N_ge_9']:
            rr=[r for r in features if r['model']==model and group in groups(int(r['n_agents']))]
            for key in [k for k in rr[0] if k.startswith(('world_','candidate_','query_'))]:
                dist.append(dict(model=model,stage='features',group=group,unit='window',metric=key,**stats([r[key] for r in rr])))
    csvwrite(DOC/'TIES_SOFT_HARD_GROUPS.csv',tie);csvwrite(DOC/'SPECTRUM_FEATURE_DISTRIBUTIONS.csv',dist)
    gradients=[];parameters=[];selected=[]
    for model in MODELS:
        rr=csvread(DOC/f'GRADIENTS_{model}.csv');pr=csvread(DOC/f'PARAMETER_GRADIENTS_{model}.csv')
        for r in rr:
            mm=[m for m in matrices if m['model']==model and m['stage']=='final' and m['index']==r['index']]
            selected.append(dict(model=model,index=r['index'],n_agents=r['n_agents'],
                final_strict_matrices=sum(m['gap_class']=='strict' for m in mm),final_near_matrices=sum(m['gap_class']=='near' for m in mm),
                final_score_gradient_norm=r['g_final_score_norm'],feedback_to_direct_ratio=r['feedback_to_direct_ratio'],
                native_initial_pi_changes=r['initial_pi_changed_vs_inference'],native_final_pi_changes=r['final_pi_changed_vs_inference']))
        for group in ['all','single','multi','N_ge_9','N_2-4','N_5-8','N_9-12','N_13-20','N_21+']:
            aa=[r for r in rr if group in groups(int(r['n_agents']))]
            for key in ['g_soft_norm','g_soft_effective_norm','g_final_score_norm','native_sinkhorn_gain','uniform_jacobian_relative_residual',
                        'g_initial_st_adjoint_norm','g_initial_feedback_only_norm','feedback_to_direct_ratio',
                        'initial_chain_residual_relative_to_direct','initial_chain_residual_relative_to_indirect','g_world_features_norm','g_candidate_features_norm']:
                values=[r[key] for r in aa if r[key]!='']
                if values:gradients.append(dict(model=model,group=group,windows=len(values),metric=key,**stats(values)))
            for module in sorted({r['module'] for r in pr}):
                pp=[r for r in pr if r['module']==module and group in groups(int(r['n_agents']))]
                parameters.append(dict(model=model,group=group,module=module,windows=len(pp),
                    nonzero_windows=sum(int(r['nonzero_tensors'])>0 for r in pp),**stats([r['gradient_norm'] for r in pp])))
    csvwrite(DOC/'GRADIENT_SUMMARY.csv',gradients);csvwrite(DOC/'PARAMETER_GRADIENT_SUMMARY.csv',parameters)
    csvwrite(DOC/'GRADIENT_TIE_CONTEXT.csv',selected)
    print('CSV-only summaries complete')


if __name__=='__main__':main()

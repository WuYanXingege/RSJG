"""Pure read-only selection and paired statistics; never choose by JFDE."""
import statistics
from common import LIMITS


def select_rank(history):
    assert history
    return min(history,key=lambda r:(r['validation'],r['epoch']))


def decision(values):
    return dict(old_epoch5=LIMITS,new_rank1=values,new_minus_old=[n-o for n,o in zip(values,LIMITS)],
                passes=[values[0]<=LIMITS[0],values[1]<LIMITS[1],values[2]<LIMITS[2]])


def paired(new,old):
    a={int(r['index']):r for r in new};b={int(r['index']):r for r in old}
    assert len(a)==len(b)==590 and a.keys()==b.keys()
    result=[]
    for ordinal,idx in enumerate(sorted(a),1):
        n=int(a[idx]['n_agents']);assert n==int(b[idx]['n_agents'])
        row=dict(window=f'w{ordinal:04d}',n_agents=n)
        for k in ['H_JADE','I_JFDE','H_JFDE','feedback_gain']:
            row['new_'+k]=float(a[idx][k]);row['old_'+k]=float(b[idx][k]);row['delta_'+k]=row['new_'+k]-row['old_'+k]
        row['new_G_JFDE']=float(a[idx]['G_JFDE'])
        for nk,ok in [('minFDE_agent_micro','minFDE'),('minFDE_scene','oracle')]:
            row['delta_'+nk]=float(a[idx][nk])-float(b[idx][ok]);assert abs(row['delta_'+nk])<1e-6
        assert abs(row['delta_H_JFDE']-(row['delta_I_JFDE']-row['delta_feedback_gain']))<1e-6
        result.append(row)
    assert sum(r['n_agents']>=9 for r in result)==127
    return result


def summarize(rr):
    result=[]
    for group,predicate in [('all',lambda r:True),('multi',lambda r:r['n_agents']>1),('N_ge_9',lambda r:r['n_agents']>=9)]:
        rows=[r for r in rr if predicate(r)]
        for k in ['H_JADE','I_JFDE','H_JFDE','feedback_gain']:
            dd=[r['delta_'+k] for r in rows];benefit=1 if k=='feedback_gain' else -1
            result.append(dict(group=group,metric=k,windows=len(rows),
                old_mean=statistics.mean(r['old_'+k] for r in rows),new_mean=statistics.mean(r['new_'+k] for r in rows),
                mean_delta=statistics.mean(dd),median_delta=statistics.median(dd),
                improved=sum(x*benefit>0 for x in dd),regressed=sum(x*benefit<0 for x in dd),equal=sum(x==0 for x in dd),
                abs_delta_le_1e_6=sum(abs(x)<=1e-6 for x in dd)))
    return result

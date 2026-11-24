"""Paired, independent-K20 seed statistics; folds are averaged within seed first."""
import math
import statistics
from .common import SEEDS,FOLDS,GROUPS,METRICS


def paired_summary(rows):
    assert len(rows)==5 and {r['seed'] for r in rows}==set(SEEDS)
    first=rows[0]
    out={k:first[k] for k in ('fold','group','fold_count','windows','agents','metric')}
    out['seed_count']=5;out['ddof']=1
    if first['B_m'] is None:
        assert all(r['B_m'] is None for r in rows)
        out.update({k:None for k in ('B_mean_m','B_std_m','Bstar_mean_m','Bstar_std_m','delta_mean_cm','delta_std_cm',
                                      'improved','tied','worsened','four_metrics_improved_seeds')})
        return out
    for model in ('B','Bstar'):
        values=[r[model+'_m'] for r in rows]
        out[model+'_mean_m']=math.fsum(values)/5;out[model+'_std_m']=statistics.stdev(values)
    d=[r['delta_cm'] for r in rows]
    out.update(delta_mean_cm=math.fsum(d)/5,delta_std_cm=statistics.stdev(d),
               improved=sum(x<0 for x in d),tied=sum(x==0 for x in d),worsened=sum(x>0 for x in d),
               four_metrics_improved_seeds=sum(r['four_metrics_improved'] for r in rows))
    return out


def tables(native):
    per_seed=[]
    for fold in FOLDS:
        for seed in SEEDS:
            for group in GROUPS:
                b=native[fold][str(seed)][group]['B'];s=native[fold][str(seed)][group]['Bstar']
                for q in METRICS:
                    per_seed.append(dict(fold=fold,seed=seed,group=group,fold_count=1 if b else 0,
                        windows=b['scene_count'] if b else 0,agents=b['agent_count'] if b else 0,metric=q,
                        B_m=b[q] if b else None,Bstar_m=s[q] if b else None,delta_cm=100*(s[q]-b[q]) if b else None,
                        four_metrics_improved=all(s[k]<b[k] for k in METRICS) if b else None))
    for seed in SEEDS:
        for group in ('all','N_ge_9'):
            folds=[f for f in FOLDS if native[f][str(seed)][group]['B']]
            assert len(folds)==(3 if group=='all' else 2)
            b={q:math.fsum(native[f][str(seed)][group]['B'][q] for f in folds)/len(folds) for q in METRICS}
            s={q:math.fsum(native[f][str(seed)][group]['Bstar'][q] for f in folds)/len(folds) for q in METRICS}
            for q in METRICS:
                per_seed.append(dict(fold='equal_fold_mean',seed=seed,group=group,fold_count=len(folds),
                    windows=sum(native[f][str(seed)][group]['B']['scene_count'] for f in folds),
                    agents=sum(native[f][str(seed)][group]['B']['agent_count'] for f in folds),metric=q,
                    B_m=b[q],Bstar_m=s[q],delta_cm=100*(s[q]-b[q]),four_metrics_improved=all(s[k]<b[k] for k in METRICS)))
    keys=list(dict.fromkeys((r['fold'],r['group'],r['metric']) for r in per_seed))
    summary=[paired_summary([r for r in per_seed if (r['fold'],r['group'],r['metric'])==k]) for k in keys]
    return per_seed,summary

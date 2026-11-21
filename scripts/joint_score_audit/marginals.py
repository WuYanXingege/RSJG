"""Missing row/column distributions from cached scores; no scoring model replay."""
from common import *


@torch.no_grad()
def main():
    setup()
    from ebjd.unified import log_sinkhorn
    from bcep_v2.runtime import configure_cuda
    configure_cuda()
    original={(r['model'],r['stage'],int(r['index']),int(r['person'])):r for r in csvread(DOC/'SOFT_HARD.csv')}
    rows=[]
    for model in MODELS:
        for record in read(OUT/model/'FILES.json'):
            index=record['index'];rec=score_record(model,index);n=len(rec['final'])
            for stage,key in [('initial','S0'),('final','final')]:
                p=log_sinkhorn(torch.from_numpy(rec[key]).cuda(),20)
                if n==1:p=torch.eye(20,device='cuda')[None]
                for person,a in enumerate(array(p).astype(np.float64)):
                    previous=original[model,stage,index,person]
                    assert float(a.max())==float(previous['maximum_element'])
                    assert float(np.abs(a.sum(1)-1).max())==float(previous['row_sum_max_error'])
                    assert float(np.abs(a.sum(0)-1).max())==float(previous['column_sum_max_error'])
                    row=dict(model=model,stage=stage,index=index,person=person,n_agents=n,people_bin=people(n),gap_class=previous['gap_class'])
                    for axis,label in [(1,'row'),(0,'column')]:
                        sums=a.sum(axis);den=sums[:,None] if axis==1 else sums[None,:]
                        pp=a/den;lp=np.zeros_like(pp);np.log(pp,out=lp,where=pp>0)
                        for metric,values in [('sum',sums),('entropy',-(pp*lp).sum(axis)),('maximum',a.max(axis))]:
                            for q,name in [(0,'min'),(.01,'p01'),(.5,'median'),(.99,'p99'),(1,'max')]:
                                row[label+'_'+metric+'_'+name]=float(np.quantile(values,q))
                    rows.append(row)
        print('row/column distributions',model,flush=True)
    csvwrite(DOC/'SOFT_ROW_COLUMN_DISTRIBUTIONS.csv',rows)
    write(DOC/'MARGINALS_PROOF.json',dict(matrices=len(rows),cached_scores_only=True,scoring_model_forwards=0,
        extrema_exactly_match_previous_native_soft_statistics=True,optimizer_guard=GUARD))


if __name__=='__main__':main()

"""Original H-JADE validation, with endpoint diagnostics from the same forward."""
import time
import torch
from bcep_joint.inference import generate
from bcep_joint.common import write
from bcep_v2.formal_manifest import write_csv
from ebjd.metrics import MetricAccumulator,hard_metric_statistics
from ebjd.unified import gather_candidates


@torch.no_grad()
def window(model,batch,goals,device='cuda'):
    x=batch.to(device);g=goals.to(device);initial=[];original=model.coupling.assign
    def capture(*a,**kw):
        r=original(*a,**kw,return_details=True);initial.append(r[3]['initial_pi']);return r[:3]
    model.coupling.assign=capture
    try:_,h,pi,_=generate(model,x.observed,x.semantic_maps,x.valid,x.scene_ids,x.agent_ids,g,2035,mode='H')
    finally:model.coupling.assign=original
    if len(initial)!=1:raise RuntimeError('unexpected pairing count')
    stats=hard_metric_statistics(h,x.future,x.valid);acc=MetricAccumulator();acc.update_statistics(stats);metrics=acc.as_dict()
    out={}
    for mode,points in [('G',g),('I',gather_candidates(g,initial[0])),('H',gather_candidates(g,pi))]:
        error=(points-x.future[:,None,:,-1]).norm(dim=-1)
        out[mode+'_JFDE']=float(((error*x.valid[:,None]).sum(-1)/x.valid.sum(1)[:,None]).min(1).values.mean())
        marginal=error.min(1).values[x.valid]
        if mode=='G':raw=marginal
        else:torch.testing.assert_close(marginal,raw,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(raw,stats.agent_minfde[x.valid],atol=1e-6,rtol=1e-6)
    return dict(n_agents=int(x.valid.sum()),H_JADE=metrics['JADE'],minFDE=metrics['minFDE'],
                oracle=metrics['scene_weighted_minFDE'],feedback_gain=out['I_JFDE']-out['H_JFDE'],**out),stats


def validation(model,data,bank,out,start_epoch):
    epoch=start_epoch
    def validate():
        nonlocal epoch
        model.eval();rows=[];acc={'all':MetricAccumulator(),'N_ge_9':MetricAccumulator()};start=time.monotonic()
        for pos,i in enumerate(data.split['selection_dev_indices']):
            b=data.batch([i],'selection_dev');row,s=window(model,b,bank.batch([i],b));rows.append(dict(epoch=epoch,index=i,**row))
            acc['all'].update_statistics(s)
            if row['n_agents']>=9:acc['N_ge_9'].update_statistics(s)
            if (pos+1)%100==0:write(out/'VALIDATION_PROGRESS.json',dict(epoch=epoch,completed_windows=pos+1,total_windows=len(data.split['selection_dev_indices'])))
        groups=[]
        for group,a in acc.items():
            rr=[r for r in rows if group=='all' or r['n_agents']>=9]
            groups.append(dict(epoch=epoch,group=group,**a.as_dict(),**{k:sum(r[k] for r in rr)/len(rr) for k in ['G_JFDE','I_JFDE','H_JFDE','feedback_gain']}))
        if len(rows)!=590 or len([r for r in rows if r['n_agents']>=9])!=127:raise ValueError('validation population changed')
        validate.metrics=acc['all'].as_dict()
        if abs(validate.metrics['minFDE']-.3520422351797007)>1e-6 or abs(validate.metrics['scene_weighted_minFDE']-.33942601943420153)>1e-6:raise ValueError('marginal changed')
        write_csv(out/'dev'/f'epoch_{epoch:03d}_windows.csv',rows);write(out/'dev'/f'epoch_{epoch:03d}_groups.json',groups)
        write(out/'VALIDATION_PROGRESS.json',dict(epoch=epoch,completed_windows=590,total_windows=590,seconds=time.monotonic()-start))
        epoch+=1
        return validate.metrics['JADE']
    return validate

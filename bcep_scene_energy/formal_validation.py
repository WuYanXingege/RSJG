"""Same full H-JADE selector and DDIM; detached SSEC diagnostics from that forward."""
import csv,time
from pathlib import Path
import torch
from bcep_joint.common import write
from bcep_joint.inference import keyed_noise
from ebjd.unified import gather_candidates
from ebjd.metrics import MetricAccumulator,hard_metric_statistics
from .identity import bank_occurrence_ids
from .inference import generate
from .formal_callbacks import SlotModelView
from .formal_monitor import population

COUNT_KEYS=['attempted','accepted','rejected','strict_ties','reliable_decreases','roundoff_uncertain','native_not_exact_optimal']
MEAN_KEYS=['I_JFDE','H_JFDE','feedback_gain','G_JFDE','minFDE_scene','final_proposal_effective_score_RMS',
    'proposal_matching_gap_mean','sinkhorn_row_entropy','sinkhorn_normalized_entropy']


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    with temp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    temp.replace(path)


@torch.no_grad()
def window(model,batch,goals,ids,device='cuda'):
    x=batch.to(device);g=goals.to(device);ids=ids.to(device);view=SlotModelView(model)
    if str(device).startswith('cuda'):torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    _,h,pi,checks=generate(view,x.observed,x.semantic_maps,x.valid,x.scene_ids,x.agent_ids,g,2035,candidate_ids=ids,mode='H')
    if str(device).startswith('cuda'):torch.cuda.synchronize()
    elapsed=time.perf_counter()-start;captured=view.coupling.last
    if captured is None or len(captured['observations'])!=1:raise RuntimeError('one validation window required')
    initial=captured['initial_pi'];stats=hard_metric_statistics(h,x.future,x.valid)
    acc=MetricAccumulator();acc.update_statistics(stats);metrics=acc.as_dict();out={}
    for mode,points in [('G',g),('I',gather_candidates(g,initial)),('H',gather_candidates(g,pi))]:
        err=(points-x.future[:,None,:,-1]).norm(dim=-1)
        out[mode+'_JFDE']=float(((err*x.valid[:,None]).sum(-1)/x.valid.sum(1)[:,None]).min(1).values.mean())
        marginal=err.min(1).values[x.valid]
        if mode=='G':raw=marginal
        else:assert torch.equal(raw,marginal)
    torch.testing.assert_close(raw,stats.agent_minfde[x.valid],atol=1e-6,rtol=1e-6)
    assert torch.equal(h[...,-1,:],gather_candidates(g,pi))
    # The identity-aware adapter and native generate preserve original slot keys;
    # audit code separately captures actual DDIM noise to verify this contract.
    row=dict(n_agents=int(x.valid.sum()),population=population(int(x.valid.sum())),H_JADE=metrics['JADE'],
        minFDE_agent_micro=metrics['minFDE'],minFDE_scene=metrics['scene_weighted_minFDE'],
        feedback_gain=out['I_JFDE']-out['H_JFDE'],**out,**{k:v for k,v in captured['observations'][0].items() if k not in ['n_agents','population','soft_kind']},
        validation_seconds=elapsed,coupling_forward_seconds=captured['coupling_forward_seconds'],
        allocated_peak_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0,
        exact_endpoint=True,exact_noise_identity=checks['occurrence_noise_keys_unchanged'])
    return row,stats


def validation(model,inputs,out,start_epoch,observer=None):
    out=Path(out);epoch=start_epoch
    def validate():
        nonlocal epoch
        model.eval();rows=[];acc={k:MetricAccumulator() for k in ['all','N_ge_9']};start=time.perf_counter()
        for pos,index in enumerate(inputs.split['selection_dev_indices']):
            batch=inputs.data.batch([index],'selection_dev');g,ids,_=bank_occurrence_ids(inputs.banks[1],[index],batch)
            row,stats=window(model,batch,g,ids);rows.append(dict(epoch=epoch,index=index,**row))
            acc['all'].update_statistics(stats)
            if row['n_agents']>=9:acc['N_ge_9'].update_statistics(stats)
            if (pos+1)%50==0:
                write(out/'VALIDATION_PROGRESS.json',dict(epoch=epoch,completed_windows=pos+1,total_windows=590,seconds=time.perf_counter()-start))
                print('SSEC dev',epoch,pos+1,'/590',flush=True)
        if len(rows)!=590 or sum(r['n_agents']>=9 for r in rows)!=127:raise ValueError('official validation population changed')
        groups=[]
        for name,a in acc.items():
            rr=[r for r in rows if name=='all' or r['n_agents']>=9]
            groups.append(dict(epoch=epoch,group=name,**a.as_dict(),**{k:sum(r[k] for r in rr)/len(rr) for k in MEAN_KEYS},
                **{k:sum(r[k] for r in rr) for k in COUNT_KEYS},accepted_energy_delta_sum=sum(r['accepted_energy_delta'] for r in rr)))
        costs=[]
        for name in ['1','2-8','9-19','20-41','42-57']:
            rr=[r for r in rows if r['population']==name]
            if rr:costs.append(dict(epoch=epoch,population=name,windows=len(rr),
                full_validation_seconds=sum(r['validation_seconds'] for r in rr),mean_window_seconds=sum(r['validation_seconds'] for r in rr)/len(rr),
                coupling_forward_seconds=sum(r['coupling_forward_seconds'] for r in rr),peak_bytes=max(r['allocated_peak_bytes'] for r in rr)))
        validate.metrics=acc['all'].as_dict()
        if abs(validate.metrics['minFDE']-.3520422351797007)>1e-6 or abs(validate.metrics['scene_weighted_minFDE']-.33942601943420153)>1e-6:
            raise ValueError('frozen candidate marginal changed')
        write_csv(out/'dev'/f'epoch_{epoch:03d}_windows.csv',rows)
        write(out/'dev'/f'epoch_{epoch:03d}_groups.json',groups);write_csv(out/'dev'/f'epoch_{epoch:03d}_costs.csv',costs)
        if observer is not None and epoch>0:write(out/'dev'/f'epoch_{epoch:03d}_training.json',observer.epoch_summary(epoch))
        write(out/'VALIDATION_PROGRESS.json',dict(epoch=epoch,completed_windows=590,total_windows=590,seconds=time.perf_counter()-start))
        validate.last_groups=groups;epoch+=1
        return validate.metrics['JADE']
    return validate

"""Score the globally sealed HOTEL single-swap endpoints, never proposals."""
import math
import time
import numpy as np
import torch
from hotel_endpoint_diagnostic.math import endpoint_errors,finite_error_statistics,motion_groups
from joint_marginal_residual.assets import load_inputs
from .common import RUN,CACHE,REFERENCE,read,write,digest,canonical_hash,csvwrite
from .math import directed_distance

GROUPS=('all','all_low_speed','mixed','all_moving','mixed_N2_8','all_moving_N2_8')
OUTPUTS=('SUMMARY.csv','PER_WINDOW.csv','METRICS.json','BOUND_CHECK.json','NATIVE_REPLAY.json','DECISION.json')
ABS_TOL=1e-5
BUDGET=.005


def load_shard(path,n):
    with np.load(path,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    for key,shape in [('old_end',(20,n,2)),('new_end',(20,n,2)),('observed',(n,8,2))]:
        assert a[key].shape==shape and a[key].dtype==np.float32 and np.isfinite(a[key]).all(),key
    assert a['valid'].shape==(n,) and a['valid'].dtype==np.bool_ and a['valid'].all()
    for key in ('native_ids','chosen_ids'):
        ids=a[key];assert ids.shape==(20,) and len(np.unique(ids))==20 and (ids>=0).all() and (ids<256).all()
    assert np.count_nonzero(a['native_ids']!=a['chosen_ids'])<=1
    return a


def aggregate(records,key):
    if not records:return None
    agents=sum(r['n'] for r in records)
    return dict(windows=len(records),agents=agents,
                minFDE=math.fsum(r[key]['agent_min_sum'] for r in records)/agents,
                JFDE=math.fsum(r[key]['J'] for r in records)/len(records))


def jfde64(goals,truth,valid):
    return float(np.linalg.norm(np.asarray(goals,dtype=np.float64)[:,valid]-np.asarray(truth,dtype=np.float64)[None,valid],axis=-1).mean(1).min())


@torch.no_grad()
def score():
    torch.set_num_threads(2)
    config=read(RUN/'RUN_CONFIG.json');seal_path=RUN/'PREDICTIONS_FROZEN.json';seal=read(seal_path)
    assert seal['future_reads']==0 and seal['config_sha256']==digest(RUN/'RUN_CONFIG.json')
    rows=seal['rows'];assert len(rows)==688 and len({r['window'] for r in rows})==688
    inputs,_,identity=load_inputs('hotel')
    assert identity=={k:v for k,v in config['identity'].items() if k!='states'}
    cache_seal=read(CACHE/'PREDICTIONS_FROZEN.json');cache_rows=cache_seal['rows']
    indices=inputs.split['selection_dev_indices'];assert len(indices)==688
    for pos,(index,row,cached) in enumerate(zip(indices,rows,cache_rows),1):
        assert row['index']==index and row['window']==f'dev_{pos:04d}' and row['window']==cached['window']
        assert row['n']==len(inputs.split['members'][index]['agent_ids'])==cached['n']
        assert row['member_sha256']==canonical_hash(inputs.split['members'][index])==cached['member_sha256']
        assert digest(RUN/row['file'])==row['sha256'] and digest(CACHE/cached['file'])==cached['sha256']
        a=load_shard(RUN/row['file'],row['n'])
        with np.load(CACHE/cached['file'],allow_pickle=False) as z:
            assert np.array_equal(a['old_end'],z['native_post'])
            assert np.array_equal(a['native_ids'],z['native_ids'])
            assert np.array_equal(a['observed'],z['observed']) and np.array_equal(a['valid'],z['valid'])
        h=directed_distance(a['old_end'],a['new_end'],a['valid'])
        assert h<=BUDGET and abs(h-row['stats']['h_m'])<=1e-12
        changed=bool(np.any(a['chosen_ids']!=a['native_ids']))
        assert changed==bool(row['stats']['replaced'])
        if not changed:assert np.array_equal(a['old_end'],a['new_end']) and h==0.
    assert sum(r['n'] for r in rows)==5203
    receipt_path=RUN/'SCORE_RECEIPT.json'
    if receipt_path.exists():
        receipt=read(receipt_path)
        assert receipt['status']=='COMPLETE' and receipt['prediction_manifest_sha256']==digest(seal_path)
        assert receipt['outputs']=={name:digest(RUN/name) for name in OUTPUTS}
        return receipt
    verification=dict(seal_unix=seal['seal_unix'],verification_unix=time.time(),verified_windows=688,
                      verified_agents=5203,truth_reads=0,prediction_manifest_sha256=digest(seal_path))
    assert verification['verification_unix']>seal['seal_unix'];write(RUN/'PRE_SCORE_VERIFICATION.json',verification)
    records=[];window_rows=[];start=time.monotonic();first_read=None
    for pos,row in enumerate(rows,1):
        a=load_shard(RUN/row['file'],row['n']);s=row['stats']
        if first_read is None:
            first_read=time.time();assert first_read>verification['verification_unix']>seal['seal_unix']
        batch=inputs.data.batch([row['index']],'selection_dev',truth=True)
        assert np.array_equal(batch.observed[0].numpy(),a['observed']) and np.array_equal(batch.valid[0].numpy(),a['valid'])
        truth=batch.future[0,:,-1];valid=batch.valid[0]
        old=finite_error_statistics(endpoint_errors(a['old_end'],truth),valid)
        new=finite_error_statistics(endpoint_errors(a['new_end'],truth),valid)
        group=motion_groups(a['observed'],a['valid']);h=directed_distance(a['old_end'],a['new_end'],a['valid'])
        delta=new['J']-old['J'];delta64=jfde64(a['new_end'],truth.numpy(),a['valid'])-jfde64(a['old_end'],truth.numpy(),a['valid'])
        record=dict(n=row['n'],groups=group['groups'],old=old,new=new);records.append(record)
        out=dict(window=row['window'],n=row['n'],low_speed_agents=group['low_count'],
                 **{g:g in group['groups'] for g in GROUPS},
                 B_minFDE_m=old['agent_min_sum']/row['n'],new_minFDE_m=new['agent_min_sum']/row['n'],
                 delta_minFDE_cm=100*(new['agent_min_sum']-old['agent_min_sum'])/row['n'],
                 B_agent_min_sum_m=old['agent_min_sum'],new_agent_min_sum_m=new['agent_min_sum'],
                 B_JFDE_m=old['J'],new_JFDE_m=new['J'],delta_JFDE_cm=100*delta,
                 h_m=h,delta_JFDE_fp64_m=delta64,delta_minus_h_m=delta-h,
                 exceeds_h_strict=delta>h,exceeds_h_tolerance=delta>h+ABS_TOL,
                 exceeds_budget_strict=delta>BUDGET,exceeds_budget_tolerance=delta>BUDGET+ABS_TOL,
                 replaced=bool(s['replaced']),fallback_reason=s['fallback_reason'] or '',
                 p0_m=s['p0'],pnew_m=s['pnew'],gain_m=s['gain'],
                 positive_swaps=s['positive_swaps'],head_evaluated=s['head_evaluated'],accepted_rank=s['accepted_rank'],
                 selection_seconds=s['seconds'])
        window_rows.append(out)
        if pos%100==0 or pos==688:write(RUN/'SCORE_PROGRESS.json',dict(windows=pos,total=688,truth_reads=pos,seconds=time.monotonic()-start))
    metrics={};summary=[]
    for group in GROUPS:
        subset=[r for r in records if group in r['groups']]
        old=aggregate(subset,'old');new=aggregate(subset,'new')
        subset_rows=[r for r in window_rows if r[group]]
        counts=dict(windows=len(subset),agents=sum(r['n'] for r in subset),replaced_windows=sum(r['replaced'] for r in subset_rows),
                    no_positive_gain_windows=sum(r['fallback_reason']=='no_positive_gain' for r in subset_rows),
                    no_feasible_windows=sum(r['fallback_reason']=='no_feasible' for r in subset_rows))
        metrics[group]=dict(**counts,B=old,new=new)
        for q in ('minFDE','JFDE'):
            summary.append(dict(group=group,metric=q,**counts,B_m=old[q] if old else None,new_m=new[q] if new else None,
                                delta_cm=100*(new[q]-old[q]) if old else None))
    references=REFERENCE
    checks=[]
    for q,reference in references.items():
        actual=metrics['all']['B'][q];allowed=1e-6+1e-6*abs(reference)
        checks.append(dict(metric=q,reference_m=reference,actual_m=actual,difference_m=actual-reference,allowed_m=allowed,passed=abs(actual-reference)<=allowed))
    replay=dict(passed=all(x['passed'] for x in checks),atol=1e-6,rtol=1e-6,checks=checks)
    bounds=dict(budget_m=BUDGET,fp32_absolute_tolerance_m=ABS_TOL,windows=688,
                max_h_m=max(r['h_m'] for r in window_rows),
                max_delta_JFDE_m=max(r['delta_JFDE_cm']/100 for r in window_rows),
                max_actual_deterioration_m=max(0.,max(r['delta_JFDE_cm']/100 for r in window_rows)),
                max_delta_minus_h_m=max(r['delta_minus_h_m'] for r in window_rows),
                max_delta_fp64_minus_h_m=max(r['delta_JFDE_fp64_m']-r['h_m'] for r in window_rows),
                h_exceeds_budget_windows=sum(r['h_m']>BUDGET for r in window_rows),
                delta_exceeds_h_strict_windows=sum(r['exceeds_h_strict'] for r in window_rows),
                delta_exceeds_h_tolerance_windows=sum(r['exceeds_h_tolerance'] for r in window_rows),
                delta_exceeds_budget_strict_windows=sum(r['exceeds_budget_strict'] for r in window_rows),
                delta_exceeds_budget_tolerance_windows=sum(r['exceeds_budget_tolerance'] for r in window_rows))
    bounds['passed']=(bounds['h_exceeds_budget_windows']==0 and bounds['delta_exceeds_h_tolerance_windows']==0 and
                      bounds['delta_exceeds_budget_tolerance_windows']==0 and bounds['max_delta_fp64_minus_h_m']<=1e-12)
    improvement=metrics['all']['B']['minFDE']-metrics['all']['new']['minFDE']
    passed=improvement>=.020 and bounds['passed']
    decision=dict(endpoint_gate_passed=passed,all_bounds_passed=bounds['passed'],improvement_m=improvement,
                  minFDE_required_improvement_m=.020,improvement_minus_requirement_m=improvement-.020,
                  stop_stage='endpoint_gate_failed_stop_route' if not passed else 'endpoint_gate_passed_requires_frozen_DDIM',
                  training_updates=0,test_access=0,DDIM_calls=0,
                  replaced_windows=metrics['all']['replaced_windows'],
                  no_positive_gain_windows=metrics['all']['no_positive_gain_windows'],
                  no_feasible_windows=metrics['all']['no_feasible_windows'])
    assert decision['replaced_windows']+decision['no_positive_gain_windows']+decision['no_feasible_windows']==688
    csvwrite(RUN/'SUMMARY.csv',summary);csvwrite(RUN/'PER_WINDOW.csv',window_rows)
    write(RUN/'METRICS.json',metrics);write(RUN/'NATIVE_REPLAY.json',replay);write(RUN/'BOUND_CHECK.json',bounds);write(RUN/'DECISION.json',decision)
    assert replay['passed'],'B replay mismatch: fix same identity/config before interpreting result'
    assert bounds['passed'],'JFDE protection failure beyond tolerance: fix implementation before deciding'
    receipt=dict(status='COMPLETE',windows=688,agents=5203,training_updates=0,test_access=0,DDIM_calls=0,
        prediction_manifest_sha256=digest(seal_path),config_sha256=digest(RUN/'RUN_CONFIG.json'),
        global_seal_unix=seal['seal_unix'],verification_unix=verification['verification_unix'],first_truth_read_unix=first_read,
        truth_reads=688,score_wall_seconds=time.monotonic()-start,completed_unix=time.time(),
        outputs={name:digest(RUN/name) for name in OUTPUTS})
    write(receipt_path,receipt);return receipt

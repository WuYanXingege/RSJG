"""Endpoint-only native and equal-budget random controls after global sealing."""
import math as pymath
import statistics
import time
from pathlib import Path
import numpy as np
import torch
from joint_marginal_residual.assets import load_inputs
from .common import RUN,read,write,digest,canonical_hash,csvwrite
from .math import endpoint_errors,finite_error_statistics,expected_uniform_metrics,motion_groups

GROUPS=('all','all_low_speed','mixed','all_moving','mixed_N2_8','all_moving_N2_8')
METRICS=('minFDE','L','G','J')
OUTPUTS=('SUMMARY.csv','RAW256.csv','PER_WINDOW.csv','PER_REPETITION.csv','REPETITION_GROUPS.csv',
         'METRICS.json','NATIVE_REPLAY.json','DECOMPOSITION_CHECK.json')


def load_shard(path,n):
    with np.load(path,allow_pickle=False) as z:
        assert set(z.files)=={'raw','native_ids','native_pre','native_post','random_ids','random_post','observed','valid'}
        values={k:z[k].copy() for k in z.files}
    for key,shape in [('raw',(256,n,2)),('native_pre',(20,n,2)),('native_post',(20,n,2)),
                      ('random_post',(16,20,n,2)),('observed',(n,8,2))]:
        assert values[key].shape==shape and values[key].dtype==np.float32 and np.isfinite(values[key]).all()
    assert values['valid'].shape==(n,) and values['valid'].dtype==np.bool_ and values['valid'].all()
    for ids in [values['native_ids'],*values['random_ids']]:
        assert ids.shape==(20,) and np.issubdtype(ids.dtype,np.integer)
        assert len(np.unique(ids))==20 and (ids>=0).all() and (ids<256).all()
    assert values['random_ids'].shape==(16,20)
    assert np.array_equal(values['raw'][values['native_ids']],values['native_pre'])
    return values


def compact(stats):
    return dict(L=float(stats['L']),J=float(stats['J']),G=float(stats['G']),
                agent_min_sum=float(stats['agent_min_sum']),n=int(stats['n']))


def aggregate(records):
    if not records:return None
    n=sum(r['n'] for r in records)
    return dict(windows=len(records),agents=n,minFDE=pymath.fsum(r['agent_min_sum'] for r in records)/n,
                **{k:pymath.fsum(r[k] for r in records)/len(records) for k in ('L','G','J')})


def _flat(out,prefix,record):
    out.update({prefix+k:record[k] for k in ('L','G','J','agent_min_sum')})
    out[prefix+'minFDE']=record['agent_min_sum']/record['n']


def _identity_drift(native,uniform):
    return abs((native['J']-uniform['J'])-((native['L']-uniform['L'])+(native['G']-uniform['G'])))


@torch.no_grad()
def score():
    torch.set_num_threads(2)
    config=read(RUN/'RUN_CONFIG.json');seal_path=RUN/'PREDICTIONS_FROZEN.json';seal=read(seal_path)
    assert seal['config_sha256']==digest(RUN/'RUN_CONFIG.json') and seal['future_reads']==0
    assert seal['random_ids_sha256']==digest(RUN/'RANDOM_IDS.json')
    rows=seal['rows'];assert len(rows)==688 and len({r['window'] for r in rows})==688
    inputs,_,identity=load_inputs('hotel');assert identity=={k:v for k,v in config['identity'].items() if k!='states'}
    split=inputs.split;indices=split['selection_dev_indices']
    assert len(indices)==688
    for pos,(index,row) in enumerate(zip(indices,rows),1):
        assert index==row['index'] and row['n']==len(split['members'][index]['agent_ids'])
        assert row['window']==f'dev_{pos:04d}' and digest(RUN/row['file'])==row['sha256']
        load_shard(RUN/row['file'],row['n'])
    assert sum(r['n'] for r in rows)==5203
    previous=RUN/'SCORE_RECEIPT.json'
    if previous.exists():
        receipt=read(previous)
        assert receipt['status']=='COMPLETE' and receipt['prediction_manifest_sha256']==digest(seal_path)
        assert receipt['outputs']=={name:digest(RUN/name) for name in OUTPUTS}
        return receipt
    verification=dict(seal_unix=seal['seal_unix'],verification_unix=time.time(),verified_windows=688,
                      verified_agents=5203,truth_reads=0,prediction_manifest_sha256=digest(seal_path))
    assert verification['verification_unix']>seal['seal_unix'];write(RUN/'PRE_SCORE_VERIFICATION.json',verification)
    all_records=[];window_rows=[];rep_rows=[];first_read=None;start=time.monotonic();drifts=[]
    for pos,row in enumerate(rows,1):
        path=RUN/row['file'];assert digest(path)==row['sha256'];a=load_shard(path,row['n'])
        if first_read is None:
            first_read=time.time();assert first_read>verification['verification_unix']>seal['seal_unix']
        batch=inputs.data.batch([row['index']],'selection_dev',truth=True)
        assert np.array_equal(batch.observed[0].numpy(),a['observed']) and np.array_equal(batch.valid[0].numpy(),a['valid'])
        truth=batch.future[0,:,-1];valid=batch.valid[0]
        grouping=motion_groups(a['observed'],a['valid'])
        raw_errors=endpoint_errors(a['raw'],truth)
        raw=compact(finite_error_statistics(raw_errors,valid))
        pre=compact(finite_error_statistics(endpoint_errors(a['native_pre'],truth),valid))
        post=compact(finite_error_statistics(endpoint_errors(a['native_post'],truth),valid))
        uniform_pre=compact(expected_uniform_metrics(raw_errors,k=20,valid=valid))
        uniform_post=[compact(finite_error_statistics(endpoint_errors(x,truth),valid)) for x in a['random_post']]
        uniform_mean=dict(n=row['n'],**{k:pymath.fsum(x[k] for x in uniform_post)/16 for k in ('L','G','J','agent_min_sum')})
        record=dict(window=row['window'],n=row['n'],groups=list(grouping['groups']),raw=raw,pre=pre,post=post,
                    uniform_pre=uniform_pre,uniform_post=uniform_post,uniform_mean=uniform_mean)
        all_records.append(record)
        out=dict(window=row['window'],n=row['n'],low_speed_agents=int(grouping['low_count']))
        out.update({g:g in grouping['groups'] for g in GROUPS})
        for prefix,stat in [('raw256_',raw),('native_pre_',pre),('uniform_pre_exact_',uniform_pre),
                            ('native_post_',post),('uniform_post_mean_',uniform_mean)]:_flat(out,prefix,stat)
        for stage,native,uniform in [('pre',pre,uniform_pre),('post',post,uniform_mean)]:
            for q in METRICS:
                nv=native[q] if q!='minFDE' else native['agent_min_sum']/row['n']
                uv=uniform[q] if q!='minFDE' else uniform['agent_min_sum']/row['n']
                out[f'{stage}_delta_{q}_cm']=100*(nv-uv)
            drift=_identity_drift(native,uniform);drifts.append(drift);out[stage+'_decomposition_error_m']=drift
        for q in METRICS:
            values=[x[q] if q!='minFDE' else x['agent_min_sum']/row['n'] for x in uniform_post]
            out[f'uniform_post_{q}_sd_m']=statistics.stdev(values)
        window_rows.append(out)
        for repetition,stat in enumerate(uniform_post):
            rr=dict(window=row['window'],n=row['n'],repetition=repetition)
            rr.update({g:g in grouping['groups'] for g in GROUPS});_flat(rr,'uniform_post_',stat)
            for q in METRICS:
                nv=post[q] if q!='minFDE' else post['agent_min_sum']/row['n']
                uv=stat[q] if q!='minFDE' else stat['agent_min_sum']/row['n']
                rr['delta_'+q+'_cm']=100*(nv-uv)
            drifts.append(_identity_drift(post,stat));rep_rows.append(rr)
        if pos%100==0 or pos==688:
            write(RUN/'SCORE_PROGRESS.json',dict(windows=pos,total=688,truth_reads=pos,seconds=time.monotonic()-start))
    summary=[];rawrows=[];rep_group_rows=[];metrics={}
    for group in GROUPS:
        subset=[r for r in all_records if group in r['groups']];count=len(subset);agents=sum(r['n'] for r in subset)
        values={key:aggregate([r[key] for r in subset]) for key in ('raw','pre','post','uniform_pre','uniform_mean')}
        repetitions=[aggregate([r['uniform_post'][rep] for r in subset]) for rep in range(16)]
        metrics[group]=dict(windows=count,agents=agents,**values,repetitions=repetitions)
        for q in METRICS:
            rawrows.append(dict(group=group,windows=count,agents=agents,metric=q,value_m=values['raw'][q] if count else None))
        for stage,nk,uk in [('pre','pre','uniform_pre'),('post','post','uniform_mean')]:
            if count:drifts.append(_identity_drift(values[nk],values[uk]))
            for q in METRICS:
                native=values[nk][q] if count else None;uniform=values[uk][q] if count else None
                sd=statistics.stdev(x[q] for x in repetitions) if count and stage=='post' else None
                summary.append(dict(group=group,stage=stage,windows=count,agents=agents,metric=q,native_m=native,
                    uniform_m=uniform,delta_cm=100*(native-uniform) if count else None,
                    uniform_sd_m=sd,delta_sd_cm=100*sd if sd is not None else None,
                    repetitions=16 if stage=='post' else None,uniform_kind='16 fixed uniform K20 draws' if stage=='post' else 'exact uniform K20 expectation'))
        for repetition,random in enumerate(repetitions):
            out=dict(group=group,repetition=repetition,windows=count,agents=agents)
            for q in METRICS:
                out['native_'+q+'_m']=values['post'][q] if count else None
                out['uniform_'+q+'_m']=random[q] if count else None
                out['delta_'+q+'_cm']=100*(values['post'][q]-random[q]) if count else None
            if count:drifts.append(_identity_drift(values['post'],random))
            rep_group_rows.append(out)
    checks=[];actual=dict(raw256_minFDE=metrics['all']['raw']['minFDE'],selected20_minFDE=metrics['all']['pre']['minFDE'],
                         corrected20_minFDE=metrics['all']['post']['minFDE'],corrected20_JFDE=metrics['all']['post']['J'])
    for k,value in actual.items():
        reference=float(config['reference'][k]);allowed=1e-6+1e-6*abs(reference)
        checks.append(dict(metric=k,reference_m=reference,actual_m=value,difference_m=value-reference,allowed_m=allowed,passed=abs(value-reference)<=allowed))
    replay=dict(passed=all(r['passed'] for r in checks),atol=1e-6,rtol=1e-6,checks=checks)
    decomposition=dict(passed=max(drifts)<=1e-12,maximum_absolute_error_m=max(drifts),checked_window_and_group_comparisons=len(drifts),tolerance_m=1e-12)
    csvwrite(RUN/'SUMMARY.csv',summary);csvwrite(RUN/'RAW256.csv',rawrows);csvwrite(RUN/'PER_WINDOW.csv',window_rows)
    csvwrite(RUN/'PER_REPETITION.csv',rep_rows);csvwrite(RUN/'REPETITION_GROUPS.csv',rep_group_rows)
    write(RUN/'METRICS.json',metrics);write(RUN/'NATIVE_REPLAY.json',replay);write(RUN/'DECOMPOSITION_CHECK.json',decomposition)
    assert replay['passed'],'native endpoint replay mismatch; resolve fixed identity/precision before interpretation'
    assert decomposition['passed'],'Delta J != Delta L + Delta G'
    receipt=dict(status='COMPLETE',windows=688,agents=5203,training_updates=0,test_access=0,DDIM_calls=0,
        prediction_manifest_sha256=digest(seal_path),config_sha256=digest(RUN/'RUN_CONFIG.json'),
        global_seal_unix=seal['seal_unix'],verification_unix=verification['verification_unix'],first_truth_read_unix=first_read,
        truth_reads=688,score_wall_seconds=time.monotonic()-start,completed_unix=time.time(),
        outputs={name:digest(RUN/name) for name in OUTPUTS})
    write(previous,receipt);return receipt

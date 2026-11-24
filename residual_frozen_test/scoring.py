"""Native paired metrics; reads truth only after the global three-fold seal audit."""
import csv
import math
import time
import torch
from bcep_formal.data import FormalData
from ebjd.metrics import MetricAccumulator, hard_metric_statistics
from joint_marginal_residual.assets import load_inputs
from .common import *

OUTPUTS=('PER_WINDOW.csv','SUMMARY.csv','SUBGROUPS.csv','NATIVE_METRICS.json','BASELINE_REPLAY.json')


@torch.no_grad()
def score(config):
    from .run import verify_all
    seal_path=RUN/'PREDICTIONS_FROZEN.json';seal=read(seal_path)
    assert seal['rows']==verify_all(config)
    assert seal['config_sha256']==digest(RUN/'RUN_CONFIG.json') and seal['heldout_future_reads']==0
    verification=dict(status='ALL_THREE_FOLDS_VERIFIED_BEFORE_TRUTH',files_verified=2148,
        prediction_manifest_sha256=digest(seal_path),verification_unix=time.time(),seal_unix=seal['seal_unix'],heldout_future_reads=0)
    assert verification['verification_unix']>seal['seal_unix']
    receipt_path=RUN/'SCORE_RECEIPT.json'
    if receipt_path.exists():
        receipt=read(receipt_path)
        assert receipt['prediction_manifest_sha256']==digest(seal_path)
        assert receipt['outputs']=={p:digest(RUN/p) for p in OUTPUTS}
        assert read(RUN/'BASELINE_REPLAY.json')['passed']
        return receipt
    write(RUN/'PRE_SCORE_VERIFICATION.json',verification)
    native={};windows=[];summaries=[];checks=[];truth_reads=0;first_read=None;tick=time.monotonic()
    for fold in FOLDS:
        inputs,_,_=load_inputs(fold);split=inputs.split
        scoring=FormalData(split,phase='scoring',verify=False)
        acc={g:{b:MetricAccumulator() for b in ('B','Bstar')} for g in GROUPS}
        rows=[r for r in seal['rows'] if r['fold']==fold]
        for row in rows:
            path=RUN/row['file'];assert digest(path)==row['sha256'];values=arrays(path,row['n'])
            if first_read is None:
                first_read=time.time();assert first_read>verification['verification_unix']>seal['seal_unix']
            batch=scoring.batch([row['index']],PURPOSE,truth=True);truth_reads+=1
            assert batch.valid.all() and int(batch.valid.sum())==row['n']
            assert canonical_hash(split['members'][row['index']])==row['member_sha256']
            assert torch.equal(batch.valid[0],torch.from_numpy(values['valid']))
            out=dict(fold=fold,window=row['window'],seed=SEED,n=row['n'])
            for b in ('B','Bstar'):
                stats=hard_metric_statistics(torch.from_numpy(values[b])[None],batch.future,batch.valid)
                for group in groups_for(row['n']):acc[group][b].update_statistics(stats)
                out.update({b+'_JADE':float(stats.scene_jade[0]),b+'_JFDE':float(stats.scene_jfde[0])})
                for q,v in [('minADE',stats.agent_minade),('minFDE',stats.agent_minfde)]:
                    total=float(v[stats.agent_valid].double().sum());out[b+'_'+q+'_sum']=total;out[b+'_'+q]=total/row['n']
            out.update({q+'_delta_cm':100*(out['Bstar_'+q]-out['B_'+q]) for q in METRICS});windows.append(out)
        native[fold]={g:{b:a.as_dict() if a.scene_count else None for b,a in branches.items()} for g,branches in acc.items()}
        for g in GROUPS:
            b=native[fold][g]['B'];s=native[fold][g]['Bstar']
            for q in METRICS:
                summaries.append(dict(fold=fold,group=g,seed=SEED,fold_count=1 if b else 0,
                    windows=b['scene_count'] if b else 0,agents=b['agent_count'] if b else 0,metric=q,
                    B_m=b[q] if b else None,Bstar_m=s[q] if s else None,delta_cm=100*(s[q]-b[q]) if b else None))
        historical_rows=list(csv.DictReader(historical(fold).open()))
        assert digest(historical(fold))==config['historical_B'][fold]['sha256']
        for old in historical_rows:
            if int(old['seed'])!=SEED:continue
            b=native[fold][old['group']]['B']
            assert (b['scene_count'] if b else 0)==int(old['scene_count'])
            assert (b['agent_count'] if b else 0)==int(old['agent_count'])
            if b is None:continue
            for q in METRICS:
                ref=float(old['B_'+q]);difference=b[q]-ref;t=config['replay_tolerance']
                checks.append(dict(fold=fold,group=old['group'],metric=q,historical_B_m=ref,current_B_m=b[q],
                    difference_m=difference,allowed_m=t['atol']+t['rtol']*abs(ref),
                    passed=abs(difference)<=t['atol']+t['rtol']*abs(ref)))
        write(RUN/'SCORE_PROGRESS.json',dict(fold=fold,truth_reads=truth_reads,total=2148,unix=time.time()))
    assert truth_reads==2148
    for g in ('all','N_ge_9'):
        available=[f for f in FOLDS if native[f][g]['B']]
        assert len(available)==(3 if g=='all' else 2)
        for q in METRICS:
            b=math.fsum(native[f][g]['B'][q] for f in available)/len(available)
            s=math.fsum(native[f][g]['Bstar'][q] for f in available)/len(available)
            summaries.append(dict(fold='equal_fold_mean',group=g,seed=SEED,fold_count=len(available),
                windows=sum(native[f][g]['B']['scene_count'] for f in available),
                agents=sum(native[f][g]['B']['agent_count'] for f in available),metric=q,B_m=b,Bstar_m=s,delta_cm=100*(s-b)))
    csvwrite(RUN/'PER_WINDOW.csv',windows)
    csvwrite(RUN/'SUMMARY.csv',[r for r in summaries if r['group'] in ('all','N_ge_9')])
    csvwrite(RUN/'SUBGROUPS.csv',[r for r in summaries if r['group']!='all' and r['fold']!='equal_fold_mean'])
    write(RUN/'NATIVE_METRICS.json',native)
    replay=dict(passed=bool(checks) and all(c['passed'] for c in checks),checks=checks,
                maximum_absolute_difference_m=max(abs(c['difference_m']) for c in checks),tolerance=config['replay_tolerance'])
    write(RUN/'BASELINE_REPLAY.json',replay)
    assert replay['passed'],'historical seed2035 B replay exceeded fixed FP32 tolerance; investigate before reporting gains'
    receipt=dict(status='COMPLETE',seed=SEED,training_updates=0,truth_reads=truth_reads,
        first_truth_read_unix=first_read,global_seal_unix=seal['seal_unix'],global_verification_unix=verification['verification_unix'],
        prediction_manifest_sha256=digest(seal_path),source_config_sha256=digest(RUN/'RUN_CONFIG.json'),
        scoring_wall_seconds=time.monotonic()-tick,completed_unix=time.time(),historical_B_replay_passed=True,
        outputs={p:digest(RUN/p) for p in OUTPUTS})
    immutable(receipt_path,receipt);return receipt

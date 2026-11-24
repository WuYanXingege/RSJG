"""Score only four new seeds after all 8592 pairs are sealed; reuse seed2035 scores."""
import csv
import time
import torch
from bcep_formal.data import FormalData
from ebjd.metrics import MetricAccumulator,hard_metric_statistics
from joint_marginal_residual.assets import load_inputs
from .common import *
from .statistics import tables

OUTPUTS=('PER_WINDOW.csv','PER_SEED.csv','SUMMARY.csv','SUBGROUPS.csv','NATIVE_METRICS.json','BASELINE_REPLAY.json')


@torch.no_grad()
def score(config):
    from .run import verify_all,verify_old
    seal_path=RUN/'PREDICTIONS_FROZEN.json';seal=read(seal_path)
    assert seal['rows']==verify_all(config)
    assert seal['config_sha256']==digest(RUN/'RUN_CONFIG.json') and seal['heldout_future_reads']==0
    verify_old()
    verification=dict(status='ALL_8592_NEW_PAIRS_VERIFIED_BEFORE_TRUTH',files_verified=8592,
        prediction_manifest_sha256=digest(seal_path),verification_unix=time.time(),seal_unix=seal['seal_unix'],new_truth_reads=0)
    assert verification['verification_unix']>seal['seal_unix']
    receipt_path=RUN/'SCORE_RECEIPT.json'
    if receipt_path.exists():
        receipt=read(receipt_path)
        assert receipt['prediction_manifest_sha256']==digest(seal_path)
        assert receipt['outputs']=={p:digest(RUN/p) for p in OUTPUTS}
        assert read(RUN/'BASELINE_REPLAY.json')['passed'];return receipt
    write(RUN/'PRE_SCORE_VERIFICATION.json',verification)
    original=read(OLD_RUN/'NATIVE_METRICS.json')
    native={f:{'2035':original[f]} for f in FOLDS}
    windows=list(csv.DictReader((OLD_RUN/'PER_WINDOW.csv').open()))
    assert len(windows)==2148 and all(int(r['seed'])==2035 for r in windows)
    checks=[];truth_reads=0;new_pairs=0;first_read=None;tick=time.monotonic()
    mapping={(r['fold'],r['seed'],r['index']):r for r in seal['rows']}
    for fold in FOLDS:
        inputs,_,_=load_inputs(fold);split=inputs.split;scoring=FormalData(split,phase='scoring',verify=False)
        acc={seed:{g:{b:MetricAccumulator() for b in ('B','Bstar')} for g in GROUPS} for seed in NEW_SEEDS}
        for index in split[PURPOSE+'_indices']:
            values={}
            for seed in NEW_SEEDS:
                row=mapping[fold,seed,index];path=RUN/row['file'];assert digest(path)==row['sha256']
                values[seed]=arrays(path,row['n'])
            if first_read is None:
                first_read=time.time();assert first_read>verification['verification_unix']>seal['seal_unix']
            batch=scoring.batch([index],PURPOSE,truth=True);truth_reads+=1
            assert batch.valid.all() and int(batch.valid.sum())==row['n']
            assert canonical_hash(split['members'][index])==row['member_sha256']
            for seed in NEW_SEEDS:
                row=mapping[fold,seed,index];v=values[seed];assert torch.equal(batch.valid[0],torch.from_numpy(v['valid']))
                out=dict(fold=fold,window=row['window'],seed=seed,n=row['n'])
                for b in ('B','Bstar'):
                    stats=hard_metric_statistics(torch.from_numpy(v[b])[None],batch.future,batch.valid)
                    for group in groups_for(row['n']):acc[seed][group][b].update_statistics(stats)
                    out.update({b+'_JADE':float(stats.scene_jade[0]),b+'_JFDE':float(stats.scene_jfde[0])})
                    for q,t in [('minADE',stats.agent_minade),('minFDE',stats.agent_minfde)]:
                        total=float(t[stats.agent_valid].double().sum());out[b+'_'+q+'_sum']=total;out[b+'_'+q]=total/row['n']
                out.update({q+'_delta_cm':100*(out['Bstar_'+q]-out['B_'+q]) for q in METRICS});windows.append(out);new_pairs+=1
        for seed in NEW_SEEDS:
            native[fold][str(seed)]={g:{b:a.as_dict() if a.scene_count else None for b,a in branches.items()} for g,branches in acc[seed].items()}
        assert digest(historical(fold))==config['historical_B'][fold]['sha256']
        for old in csv.DictReader(historical(fold).open()):
            seed=int(old['seed']);assert seed in SEEDS
            b=native[fold][str(seed)][old['group']]['B']
            assert (b['scene_count'] if b else 0)==int(old['scene_count'])
            assert (b['agent_count'] if b else 0)==int(old['agent_count'])
            if b is None:continue
            for q in METRICS:
                ref=float(old['B_'+q]);difference=b[q]-ref;t=config['replay_tolerance']
                checks.append(dict(fold=fold,seed=seed,group=old['group'],metric=q,historical_B_m=ref,current_B_m=b[q],
                    difference_m=difference,allowed_m=t['atol']+t['rtol']*abs(ref),
                    passed=abs(difference)<=t['atol']+t['rtol']*abs(ref)))
        write(RUN/'SCORE_PROGRESS.json',dict(fold=fold,truth_reads=truth_reads,new_pairs=new_pairs,total_new_pairs=8592,unix=time.time()))
    assert truth_reads==2148 and new_pairs==8592 and len(windows)==10740
    assert len({(r['fold'],int(r['seed']),r['window']) for r in windows})==10740
    per_seed,summary=tables(native)
    csvwrite(RUN/'PER_WINDOW.csv',windows);csvwrite(RUN/'PER_SEED.csv',per_seed)
    csvwrite(RUN/'SUMMARY.csv',[r for r in summary if r['group'] in ('all','N_ge_9')])
    csvwrite(RUN/'SUBGROUPS.csv',[r for r in summary if r['group']!='all' and r['fold']!='equal_fold_mean'])
    write(RUN/'NATIVE_METRICS.json',native)
    replay=dict(passed=len(checks)==100 and all(c['passed'] for c in checks),checks=checks,
                maximum_absolute_difference_m=max(abs(c['difference_m']) for c in checks),tolerance=config['replay_tolerance'])
    write(RUN/'BASELINE_REPLAY.json',replay)
    assert replay['passed'],'same-seed historical B exceeded frozen FP32 tolerance; investigate before interpreting Bstar'
    receipt=dict(status='COMPLETE',seeds=list(SEEDS),new_seeds=list(NEW_SEEDS),training_updates=0,
        new_pairs_scored=8592,reused2035_pairs=2148,total_pairs=10740,unique_test_windows=2148,
        new_truth_reads=truth_reads,truth_reads_for_2035=0,first_truth_read_unix=first_read,
        global_seal_unix=seal['seal_unix'],global_verification_unix=verification['verification_unix'],
        prediction_manifest_sha256=digest(seal_path),source_config_sha256=digest(RUN/'RUN_CONFIG.json'),
        reused2035_receipt_sha256=config['prior2035']['receipt_sha256'],
        scoring_wall_seconds=time.monotonic()-tick,completed_unix=time.time(),historical_B_replay_passed=True,
        outputs={p:digest(RUN/p) for p in OUTPUTS})
    immutable(receipt_path,receipt);return receipt

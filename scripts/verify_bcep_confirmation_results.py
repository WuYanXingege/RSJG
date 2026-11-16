"""Independent CPU audit of frozen files and raw-CSV aggregation; no inference."""
import argparse,csv,math,statistics,sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bcep.contracts import digest,canonical_hash
from bcep_confirmation.common import read,write,budget,source_identity

def main(path):
    plan=read(path);root=Path(plan['assets']);docs=Path(plan['docs']);metrics=['minFDE','recall_0.25','recall_0.5','recall_1.0','P50','P90','P95']
    with budget(root,'READONLY_CPU',7200):
        state=read(root/'PIPELINE_STATE.json');assert state['status']=='COMPLETE' and not state['active']
        split=read(plan['split_lock']);cp=read(root/'reports/CHECKPOINTS_FROZEN.json');out=read(root/'reports/OUTPUTS_FROZEN.json')
        assert digest(plan['split_lock'])==plan['split_sha256'] and digest(plan['contract'])==plan['contract_sha256']
        assert cp['split_hash']==out['split_hash']==canonical_hash(split)
        assert cp['created_unix']<out['created_unix']<state['finished_unix']
        assert len(cp['runs'])==4 and len(out['outputs'])==20
        for run in cp['runs']:
            assert run['identity']['source']==source_identity()
            for kind in ['best','last']:assert digest(run[kind]['path'])==run[kind]['sha256']
            assert run['completed_epoch']==6 and run['last_update']==6*len(split['train_indices'])
            assert run['cumulative_wall_seconds']<3600
        assert len({(x['run'],x['seed']) for x in out['outputs']})==20
        for x in out['outputs']:
            assert digest(x['path'])==x['sha256']
            assert Path(x['path']).stat().st_mtime<=out['created_unix']
        groups=defaultdict(list)
        rows=list(csv.DictReader((docs/'PER_RUN_PER_SEED.csv').open()));assert len(rows)==120
        for r in rows:
            assert all(math.isfinite(float(r[k])) for k in metrics)
            groups[(r['model_kind'],r['method'],r['group'],int(r['training_seed']))].append(r)
        counts=[len(split['members'][i]['agent_ids']) for i in split['heldout_confirmation_indices']]
        expected={'all':(len(counts),sum(counts)),'N1':(sum(n==1 for n in counts),sum(n for n in counts if n==1)),'Nge2':(sum(n>=2 for n in counts),sum(n for n in counts if n>=2))}
        for r in rows:assert (int(r['windows']),int(r['agents']))==expected[r['group']]
        agg=read(docs/'AGGREGATES.json');means={}
        def equal(a,b):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(a,b)
        for a in agg['per_training_seed']:
            key=(a['model_kind'],a['method'],a['group'],a['training_seed']);rs=groups[key]
            assert {int(r['inference_seed']) for r in rs}==set(range(2035,2040)) and len(rs)==5
            means[key]={}
            for m in metrics:
                vals=[float(r[m]) for r in rs];means[key][m]=statistics.mean(vals)
                equal(statistics.mean(vals),a['metrics'][m]['mean']);equal(statistics.stdev(vals),a['metrics'][m]['inference_sd'])
        for a in agg['across_training_seeds']:
            for m in metrics:
                vals=[means[(a['model_kind'],a['method'],a['group'],seed)][m] for seed in [3102,3103]]
                equal(statistics.mean(vals),a['metrics'][m]['mean']);equal(statistics.stdev(vals),a['metrics'][m]['training_seed_sd'])
        pairs=list(csv.DictReader((docs/'PAIRED_DELTAS.csv').open()));assert len(pairs)==6
        for p in pairs:
            seed=int(p['training_seed']);label=p['contrast']
            left=('bcep_flow','KMeans20') if label.startswith('Flow') else ('mdn8','KMeans20')
            right=('mdn8','KMeans20') if label.startswith('Flow−MDN8') else (left[0],'IID20')
            for m in metrics:
                factor=100 if m.startswith('recall') else 1;field=m+'_pp' if factor==100 else m
                equal(float(p[field]),factor*(means[(*left,'all',seed)][m]-means[(*right,'all',seed)][m]))
        protected=read(root/'reports/PROTECTION_RECEIPT.json');assert protected['passed']
        for x in protected['files']:assert digest(x['path'])==x['sha256']==x['after_sha256']
        receipt=dict(status='PASS',raw_csv_rows=len(rows),training_repeats=2,inference_seeds=5,outputs_verified=20,checkpoints_verified=8,protected_files_verified=len(protected['files']),groups=expected,aggregation='independent statistics.mean/stdev, ddof=1',recall_deltas_unit='percentage_points',optimizer_updates=0,inference_calls=0)
        write(root/'reports/RESULT_AUDIT.json',receipt);write(docs/'RESULT_AUDIT.json',receipt);print(receipt)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);main(p.parse_args().plan)

#!/usr/bin/env python3
"""One CPU-only, budgeted, two-phase frozen selector control. No torch import."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from selector_control.core import METHODS,select_one,local_seed,identity_index,probe_correspondence,risks
from selector_control.checks import run_checks
from bcep.evaluation import endpoint_metrics


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)


def read(path):return json.loads(Path(path).read_text())
def git(*args,cwd=ROOT):return subprocess.check_output(['git',*args],cwd=cwd,text=True).strip()
def records(value):
    if isinstance(value,dict):
        if 'path' in value and 'sha256' in value:yield {'path':value['path'],'sha256':value['sha256']}
        for item in value.values():yield from records(item)
    elif isinstance(value,list):
        for item in value:yield from records(item)


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    tmp=Path(str(path)+'.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    tmp.replace(path)


def numeric_summary(values):
    a=np.asarray(values,dtype=float)
    return {'mean':float(a.mean()),'sd':float(a.std(ddof=1))}


def schema_header(path,key):
    # Header only: no truth payload is loaded before OUTPUTS_FROZEN.json.
    with zipfile.ZipFile(path) as z, z.open(key+'.npy') as f:
        version=np.lib.format.read_magic(f)
        shape,fortran,dtype=np.lib.format._read_array_header(f,version)
    return shape,str(dtype)


def run(plan_path,resume=False,precheck_fix=False):
    plan=read(plan_path);assets=Path(plan['asset_root']);docs=Path(plan['documentation_root'])
    state_path=assets/'RUN_RECEIPT.json';claim=assets/'RUN_CLAIM.json'
    config=read(plan['config']);start=time.monotonic();started=time.time();used=0.0
    source_files=[Path(__file__),*sorted((ROOT/'selector_control').glob('*.py')),ROOT/'bcep/coverage_selector.py',ROOT/'bcep/evaluation.py']
    identity={'source_commit':git('rev-parse','HEAD'),'files':{str(p.relative_to(ROOT)):sha(p) for p in source_files},'plan_sha256':sha(plan_path),'config_sha256':sha(plan['config'])}
    if claim.exists():
        if not resume:raise RuntimeError('run already claimed; explicit --resume required')
        old=read(state_path)
        if old['status']=='COMPLETE':raise RuntimeError('complete run cannot restart')
        previous_claim=read(claim)
        if previous_claim['identity']!=identity:
            prior=previous_claim['identity']
            # A failed synthetic test may be repaired before any real selection.
            # Algorithms, config, plan and budget cannot change via this route.
            allowed={'selector_control/checks.py','scripts/run_frozen_selector_control.py'}
            if not (precheck_fix and old['phase']=='PRECHECK' and not list((assets/'selection').glob('*.npz'))
                    and prior['plan_sha256']==identity['plan_sha256'] and prior['config_sha256']==identity['config_sha256']
                    and all(s==identity['files'].get(p) for p,s in prior['files'].items() if p not in allowed)):
                raise RuntimeError('resume code/config/plan differs')
            history=previous_claim.get('precheck_repair_history',[])+[{'identity':prior,'receipt':old}]
            atomic_json(claim,{'identity':identity,'created_unix':previous_claim['created_unix'],'precheck_repair_history':history})
        try:os.kill(old['pid'],0)
        except ProcessLookupError:pass
        else:raise RuntimeError('previous run PID still exists')
        used=float(old.get('cpu_wall_seconds',0))
        if old.get('active'):
            used=max(used,float(old.get('previous_seconds',0))+max(0,started-old['invocation_started_unix']))
    else:
        with claim.open('x') as f:json.dump({'identity':identity,'created_unix':started},f,indent=2)
    if used>=plan['budget_seconds']:raise RuntimeError('cumulative CPU budget exhausted')
    state={'status':'RUNNING','active':True,'pid':os.getpid(),'previous_seconds':used,'invocation_started_unix':started,
           'cpu_wall_seconds':used,'cpu_budget_seconds':plan['budget_seconds'],'optimizer_updates':0,'gpu_tasks':0,'gpu_seconds':0,
           'models_loaded':0,'optimizers_created':0,'identity':identity,'phase':'PRECHECK','complete_banks':[]}
    def save_state():
        state['cpu_wall_seconds']=used+time.monotonic()-start
        atomic_json(state_path,state)
    def emit(name,data):
        atomic_json(assets/'reports'/name,data);atomic_json(docs/name,data)
    def check():
        save_state()
        if state['cpu_wall_seconds']>=plan['budget_seconds']:raise TimeoutError('CPU wall budget exhausted')
    def timeout(*_):raise TimeoutError('CPU wall budget exhausted')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,plan['budget_seconds']-used)
    protection=None;paper=ROOT.parent/'RSJG_EBJD_Paper'
    try:
        save_state();emit('SOURCE_IDENTITY.json',identity)
        if (assets/'reports/PRECHECK.json').exists():tests=read(assets/'reports/PRECHECK.json')
        else:tests=run_checks();emit('PRECHECK.json',tests)
        print('Synthetic checks PASS',flush=True)
        # All historical sources are byte-checked; no checkpoint deserialization.
        expected={x['path']:x['sha256'] for x in records(read(plan['prior_protection_index']))}
        for r in records(read(plan['source_artifacts'])):expected[r['path']]=r['sha256']
        for sub in ('bcep','bcep_control'):
            for p in (ROOT/sub).rglob('*.py'):expected[str(p)]=hashlib.sha256(subprocess.check_output(['git','show',plan['evidence_sha']+':'+str(p.relative_to(ROOT))],cwd=ROOT)).hexdigest()
        for name in ('REPORT_ZH.md','ARTIFACTS.json','PLAN_AND_AUTHORIZATION.json'):
            p=Path(plan['source_artifacts']).with_name(name)
            expected[str(p)]=hashlib.sha256(subprocess.check_output(['git','show',plan['evidence_sha']+':'+str(p.relative_to(ROOT))],cwd=ROOT)).hexdigest()
        initial={'head':git('rev-parse','HEAD',cwd=paper),'status':git('status','--porcelain=v1',cwd=paper)}
        protection={'paper_before':initial,'files':[]}
        for p,s in expected.items():
            actual=sha(p)
            if actual!=s:raise ValueError('historical hash mismatch: '+p)
            protection['files'].append({'path':p,'before_sha256':actual})
        atomic_json(assets/'identity/PROTECTION_BEFORE.json',protection)
        datasets={};inputs=[];common_members=None
        for entry in plan['fixed_input_files']:
            p=entry['path'];assert sha(p)==entry['sha256']
            with np.load(p,allow_pickle=False) as z:
                # Deliberately exclude truth from the generation input projection.
                d={k:z[k] for k in ('atoms','iid20','coverage20','selected_indices','valid')}
                meta=json.loads(str(z['metadata_json']))
            assert d['atoms'].shape==(320,256,5,2) and str(d['atoms'].dtype)=='float32'
            assert d['iid20'].shape==d['coverage20'].shape==(320,20,5,2)
            assert d['iid20'].dtype==d['coverage20'].dtype==np.float32
            assert d['valid'].shape==(320,5) and d['valid'].dtype==bool and int(d['valid'].sum())==621
            assert d['selected_indices'].shape==(320,20,5) and d['selected_indices'].dtype==np.int64
            assert schema_header(p,'truth')==((320,5,2),'float32')
            for field in ('atoms','iid20','coverage20'):assert np.isfinite(d[field]).all()
            assert meta['M']==256 and meta['K']==20 and meta['unit']=='m' and meta['complete']
            assert meta['dt']==.4 and meta['pred_length']==12 and meta['export_optimizer_updates']==0
            assert meta['member_hash']=='3f978031af7003722efd0f04ed2cc4cd9b2f8babfdd9ddcb945c40329eb8a599'
            assert meta['split_hash']=='ebe0e117ee8e1a36e72028cfb5e6b3759df7b96821b94ea8a9a478fcbf149188'
            assert meta['source_manifest_hash']=='8824bbad3b1b6c964bd828b1bee3ea5870e614b300b6d14090644a72818c54de'
            kind=meta['model_kind'];density='flow' if kind=='bcep_flow' else 'mdn8'
            assert kind in ('bcep_flow','mdn8')
            checkpoint=plan['checkpoint_sources']['flow'] if density=='flow' else plan['checkpoint_sources']['mdn8']['best.pt']
            assert meta['checkpoint_sha256']==checkpoint['sha256']
            assert meta['seed'] in config['inference_seeds']
            index=identity_index(meta,d['valid']);assert len(index)==621
            if common_members is None:common_members=meta['members']
            else:assert common_members==meta['members']
            # Validate identity hash using the historical canonical JSON convention.
            canonical=json.dumps(meta['members'],sort_keys=True,separators=(',',':')).encode()
            assert hashlib.sha256(canonical).hexdigest()==meta['member_hash']
            for row,agent in index.values():
                ids=d['selected_indices'][row,:,agent];assert len(set(ids))==20 and np.all((ids>=0)&(ids<256))
            key=(density,meta['seed']);assert key not in datasets
            d.update(metadata=meta,path=p,input_sha256=entry['sha256'],index=index)
            datasets[key]=d
            inputs.append({**entry,'model_kind':kind,'density':density,'seed':meta['seed'],'checkpoint_sha256':meta['checkpoint_sha256'],
                           'member_hash':meta['member_hash'],'split_hash':meta['split_hash'],'scenes':320,'agent_occurrences':621,
                           'single_scenes':int((d['valid'].sum(1)==1).sum()),'unit':'m','schema_checked':True})
        assert len(datasets)==10
        emit('INPUT_IDENTITY.json',{'status':'PASS','files':inputs,'members':common_members,'truth_payload_read_during_generation':False})
        check();state['phase']='SELECTION_NO_TRUTH';save_state()
        frozen=[]
        for (density,seed),d in datasets.items():
            destination=assets/'selection'/f'{density}_seed{seed}.npz';receipt=destination.with_suffix('.json')
            if receipt.exists():
                saved=read(receipt);assert sha(destination)==saved['sha256'];frozen.append(saved);continue
            output={m:np.zeros((320,20,5,2),dtype=np.float64) for m in METHODS}
            ids_out=np.full((320,20,5),-1,dtype=np.int64)
            times=np.zeros((320,5,4));swap_time=np.zeros((320,5));swap_accepted=np.zeros((320,5),bool)
            km_seed=np.zeros((320,5),np.uint64);km_restart=np.full((320,5),-1);km_iterations=np.zeros((320,5),int)
            swaps=np.full((320,5,2),-1,int);objectives=np.zeros((320,5,2))
            for row,member in enumerate(d['metadata']['members']):
                for agent,agent_id in enumerate(member['agent_ids']):
                    s=local_seed(d['metadata']['model_kind'],member['scene_id'],agent_id,seed)
                    r=select_one(d['atoms'][row,:,agent],d['iid20'][row,:,agent],d['coverage20'][row,:,agent],d['selected_indices'][row,:,agent],s)
                    for j,m in enumerate(METHODS):output[m][row,:,agent]=r['outputs'][m];times[row,agent,j]=r['seconds'][m]
                    ids_out[row,:,agent]=r['swap_indices'];swap_time[row,agent]=r['swap_additional_seconds']
                    swap_accepted[row,agent]=r['swap_accepted'];objectives[row,agent]=[r['swap_before'],r['swap_after']]
                    if r['swap_accepted']:swaps[row,agent]=[r['removed_position'],r['added_index']]
                    km_seed[row,agent]=s;km_restart[row,agent]=r['kmeans_restart'];km_iterations[row,agent]=r['kmeans_iterations']
                if row%10==0:check()
            meta={**d['metadata'],'input_sha256':d['input_sha256'],'source_identity':identity,'contains_truth':False}
            temporary=destination.with_suffix('.tmp.npz')
            np.savez_compressed(temporary,**output,swap_indices=ids_out,valid=d['valid'],seconds=times,swap_additional_seconds=swap_time,
                                swap_accepted=swap_accepted,swap_moves=swaps,swap_objectives=objectives,kmeans_seed=km_seed,
                                kmeans_restart=km_restart,kmeans_iterations=km_iterations,metadata_json=json.dumps(meta))
            temporary.replace(destination)
            saved={'density':density,'seed':seed,'path':str(destination),'sha256':sha(destination),'contains_truth':False}
            atomic_json(receipt,saved);frozen.append(saved);state['complete_banks']=frozen;check()
            print(f'Selected {density}/{seed}: 621 occurrences; cumulative {state["cpu_wall_seconds"]:.1f}s; no truth read',flush=True)
        assert len(frozen)==10
        # Phase boundary: all algorithms and all outputs are frozen before any scoring.
        freeze={'status':'ALL_SELECTIONS_FROZEN','created_unix':time.time(),'code':identity,'banks':frozen,'truth_read':False}
        emit('OUTPUTS_FROZEN.json',freeze)
        state['phase']='SCORING';save_state()
        rows=[];risk_rows=[];partitions=[];timings={};historical=[];occurrence_rows=[]
        for f in frozen:
            density,seed=f['density'],f['seed'];d=datasets[(density,seed)]
            assert sha(f['path'])==f['sha256']
            with np.load(f['path'],allow_pickle=False) as z:out={k:z[k] for k in z.files if k!='metadata_json'}
            with np.load(d['path'],allow_pickle=False) as z:truth=z['truth'].astype(np.float64)
            assert truth.shape==(320,5,2) and np.isfinite(truth[d['valid']]).all()
            next_seed=config['probe_cycle'][str(seed)];probe=datasets[(density,next_seed)]
            mapping=probe_correspondence(d['metadata'],d['valid'],probe['metadata'],probe['valid'])
            sizes=d['valid'].sum(1);groups={'all':d['valid'],'N1':d['valid']&(sizes[:,None]==1),'Nge2':d['valid']&(sizes[:,None]>=2)}
            for mi,method in enumerate(METHODS):
                pred=out[method];support=np.zeros((320,5));squared=support.copy();prisk=support.copy();dup=support.copy()
                for key,(r,a),(r2,a2) in mapping:
                    s=pred[r,:,a];support[r,a],squared[r,a]=risks(d['atoms'][r,:,a],s)
                    prisk[r,a],_=risks(probe['atoms'][r2,:,a2],s)
                    dup[r,a]=(20-len(np.unique(s,axis=0)))/20
                    occurrence_rows.append(dict(density=density,seed=seed,method=method,scene_id=key[0],agent_id=key[1],N=int(sizes[r]),
                                                R_support=support[r,a],R_probe=prisk[r,a],squared_distortion=squared[r,a],duplicate_rate=dup[r,a]))
                for group,mask in groups.items():
                    score=endpoint_metrics(pred,truth,mask)
                    rows.append(dict(density=density,method=method,seed=seed,group=group,scenes=int(mask.any(1).sum()),**score,
                                     duplicate_rate=float(dup[mask].mean()),R_support=float(support[mask].mean()),R_probe=float(prisk[mask].mean()),squared_distortion=float(squared[mask].mean())))
                    risk_rows.append({k:rows[-1][k] for k in ('density','method','seed','group','agents','R_support','R_probe','squared_distortion','duplicate_rate')})
                tk=density+'/'+method;timings.setdefault(tk,[]).extend(out['seconds'][:,:,mi][d['valid']].tolist())
                if method in ('Greedy20','OneSwap20'):
                    e=np.linalg.norm(pred-truth[:,None],axis=-1).min(1)[d['valid']]
                    pool=np.linalg.norm(d['atoms'].astype(float)-truth[:,None],axis=-1).min(1)[d['valid']]
                    assert np.all(e>=pool-1e-12)
                    partitions.append(dict(density=density,seed=seed,method=method,total=len(e),hit=int((e<=.5).sum()),
                                           selection_miss=int(((pool<=.5)&(e>.5)).sum()),pool_miss=int((pool>.5).sum())))
                if method in ('IID20','Greedy20'):
                    old=d['iid20'] if method=='IID20' else d['coverage20']
                    # Historical float32-only recomputation is labelled, not mixed into main table.
                    historical.append(dict(density=density,seed=seed,method=method,**endpoint_metrics(old,truth.astype('float32'),d['valid'])))
            timings.setdefault(density+'/OneSwap_additional',[]).extend(out['swap_additional_seconds'][d['valid']].tolist())
            assert sha(f['path'])==f['sha256']
            check()
        metrics=['minFDE','recall_0.25','recall_0.5','recall_1.0','P50','P90','P95','duplicate_rate','R_support','R_probe','squared_distortion']
        summary=[]
        for density in ('flow','mdn8'):
            for method in METHODS:
                for group in ('all','N1','Nge2'):
                    subset=[r for r in rows if (r['density'],r['method'],r['group'])==(density,method,group)];assert len(subset)==5
                    summary.append(dict(density=density,method=method,group=group,scenes_per_seed=subset[0]['scenes'],agent_occurrences_per_seed=subset[0]['agents'],
                                        metrics={m:numeric_summary([r[m] for r in subset]) for m in metrics}))
        emit('EIGHT_CELL_METRICS.json',dict(status='COMPLETE',unit='m',ddof=1,training_seeds=1,inference_seeds=config['inference_seeds'],
             evidence_scope='Repeatedly observed internal dev; 621 occurrences per seed, not independent people or training repeats',per_seed=rows,summary=summary))
        paired=[]
        for density in ('flow','mdn8'):
            for group in ('all','N1','Nge2'):
                for left,right in [('Greedy20','IID20'),('KMeans20','IID20'),('Greedy20','KMeans20'),('OneSwap20','Greedy20')]:
                    for metric in metrics:
                        values=[]
                        for seed in config['inference_seeds']:
                            def find(method):return next(r[metric] for r in rows if (r['density'],r['group'],r['seed'],r['method'])==(density,group,seed,method))
                            delta=find(left)-find(right);values.append(delta)
                            paired.append(dict(density=density,group=group,contrast=left+'-'+right,metric=metric,seed=seed,delta=delta))
                        s=numeric_summary(values)
                        paired.append(dict(density=density,group=group,contrast=left+'-'+right,metric=metric,seed='summary',delta=s['mean'],sd=s['sd'],negative_seeds=sum(x<0 for x in values),zero_seeds=sum(x==0 for x in values),positive_seeds=sum(x>0 for x in values)))
        for name,data in [('EIGHT_CELL_METRICS.csv',rows),('PAIRED_DELTAS.csv',paired),('SUPPORT_AND_PROBE_RISK.csv',risk_rows)]:
            write_csv(assets/'reports'/name,data);write_csv(docs/name,data)
        write_csv(assets/'reports/PER_OCCURRENCE_RISKS.csv',occurrence_rows)
        emit('SUPPORT_SUBSET_PARTITION.json',{'threshold_m':.5,'applies_only_to_support_subsets':True,'rows':partitions})
        emit('HISTORICAL_FLOAT32_REPLAY.json',{'description':'Exact old output arrays; historical float32 scoring only. Main comparison uniformly float64.','rows':historical})
        timing_summary={k:dict(occurrences=len(v),mean_s=float(np.mean(v)),P50_s=float(np.quantile(v,.5)),P90_s=float(np.quantile(v,.9)),total_s=float(sum(v))) for k,v in timings.items()}
        cpu_model=next((l.split(':',1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),'unknown')
        emit('TIMING.json',dict(machine=platform.node(),cpu=cpu_model,numpy=np.__version__,threads=1,methods=timing_summary,
                               scope='Selector CPU only; OneSwap20 includes replayed Greedy + additional swap. No density forward or end-to-end inference timing.'))
        tests.update(input_banks=10,old_iid_greedy_exact_occurrence_regressions=6210,real_subset_lower_bound=True,all_selections_frozen_before_real_truth=True)
        emit('PRECHECK.json',tests)
        state['status']='COMPLETE'
    except BaseException as exc:
        state.update(status='INCOMPLETE',failure=repr(exc),traceback=traceback.format_exc())
        print(traceback.format_exc(),flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if protection is not None:
            try:
                for item in protection['files']:
                    item['after_sha256']=sha(item['path']);item['unchanged']=item['after_sha256']==item['before_sha256']
                protection['paper_after']={'head':git('rev-parse','HEAD',cwd=paper),'status':git('status','--porcelain=v1',cwd=paper)}
                protection['passed']=all(x['unchanged'] for x in protection['files']) and protection['paper_before']==protection['paper_after']
                emit('PROTECTION_BEFORE_AFTER.json',protection)
                if not protection['passed']:state.update(status='INCOMPLETE',failure='protected source/assets changed')
            except Exception as exc:state.update(status='INCOMPLETE',protection_failure=repr(exc))
        state.update(active=False,finished_unix=time.time(),exit_code=0 if state['status']=='COMPLETE' else 1)
        save_state();atomic_json(docs/'RUN_RECEIPT.json',state)
        print(json.dumps({k:state[k] for k in ('status','phase','cpu_wall_seconds','optimizer_updates','gpu_tasks','exit_code')}),flush=True)
    return state['exit_code']


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--resume',action='store_true');parser.add_argument('--precheck-fix',action='store_true')
    args=parser.parse_args();sys.exit(run(args.plan,args.resume,args.precheck_fix))

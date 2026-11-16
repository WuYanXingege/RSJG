"""All checkpoint/support freezes precede access to confirmation future labels."""
import csv,json,time
from pathlib import Path
import numpy as np
import torch
from bcep.provider import BCEP
from bcep_control.mdn import MDNProvider
from bcep.contracts import digest,canonical_hash,require
from bcep.evaluation import endpoint_metrics
from selector_control.algorithms import kmeans_centers
from selector_control.core import local_seed
from .protocol import PurposeDataset
from .common import read,write,budget,state_digest

def evaluate(plan,split,frozen):
    root=Path(plan['assets']);docs=Path(plan['docs']);identity=canonical_hash(split)
    with budget(root,'READONLY_CPU',7200):
        assert frozen['status']=='ALL_RUNS_TERMINAL' and len(frozen['runs'])==4
        indices=split['heldout_confirmation_indices'];members=[split['members'][i] for i in indices]
        nmax=max(len(m['agent_ids']) for m in members);S=len(indices)
        dataset=PurposeDataset(split,'export');outputs=[];timings=[]
        for entry in frozen['runs']:
            if entry.get('best') is None:continue
            path=Path(entry['best']['path']);assert digest(path)==entry['best']['sha256']
            state=torch.load(path,map_location='cpu',weights_only=False)
            require(state['schema']=='bcep-confirmation-pilot-v1','checkpoint schema mismatch')
            require(state['identity']==entry['identity'] and state['config_hash']==entry['config_hash'],'frozen checkpoint identity differs')
            require(state['split_hash']==identity and state['completed_epoch']>=1,'selected checkpoint lacks complete validation')
            require(state['selection_value']==state['best'] and state['selected_epoch']==state['completed_epoch'],'not selected best')
            factory=BCEP if entry['model_kind']=='bcep_flow' else MDNProvider
            model=factory(split['pred_length'],split['dt']);model.load_state_dict(state['model'],strict=True);model.eval()
            np.testing.assert_array_equal(model.scales.numpy(),np.asarray(split['scales'],dtype='float32'));assert bool(model.scales_fitted)
            before=state_digest(model.state_dict())
            for seed in range(2035,2040):
                start=time.monotonic();read_s=support_s=transfer_s=select_s=0.
                atoms=np.zeros((S,256,nmax,2),np.float32);km=np.zeros((S,20,nmax,2),np.float64);valid=np.zeros((S,nmax),bool)
                for row,index in enumerate(indices):
                    t=time.monotonic();item=dataset.get(index,'heldout_confirmation',False);read_s+=time.monotonic()-t
                    assert 'future' not in item;n=len(item['observed']);v=torch.ones(n,dtype=torch.bool)
                    t=time.monotonic()
                    sample,_=model.sample_support(item['observed'],item['semantic_maps'],v,item['scene_id'],item['agent_ids'],seed,256)
                    support_s+=time.monotonic()-t
                    t=time.monotonic();x=sample.cpu().numpy();atoms[row,:,:n]=x;valid[row,:n]=True;transfer_s+=time.monotonic()-t
                    t=time.monotonic()
                    for agent,agent_id in enumerate(item['agent_ids']):
                        r=kmeans_centers(x[:,agent],20,local_seed(entry['model_kind'],item['scene_id'],agent_id,seed),n_init=5,max_iter=100,tolerance_m=1e-6)
                        km[row,:,agent]=r['centers']
                    select_s+=time.monotonic()-t
                    if row%100==0:
                        write(root/'EVALUATION_PROGRESS.json',dict(phase='OBSERVED_ONLY_GENERATION',run=entry['name'],seed=seed,scenes_done=row+1,scenes_total=S,complete_banks=len(outputs),truth_read=False))
                meta=dict(run=entry['name'],model_kind=entry['model_kind'],training_seed=entry['seed'],inference_seed=seed,checkpoint_sha256=digest(path),split_hash=identity,
                          members=members,member_hash=canonical_hash(members),unit='m',M=256,K=20,dt=.4,pred_length=12,obs_length=8,contains_truth=False,
                          purpose='heldout_confirmation',dtype='FP32 CPU model, float64 KMeans/scoring',source_identity=entry['identity'])
                output=root/'exports'/f"{entry['name']}_seed{seed}.npz";tmp=output.with_suffix('.tmp.npz')
                np.savez_compressed(tmp,atoms=atoms,iid20=atoms[:,:20],kmeans20=km,valid=valid,metadata_json=json.dumps(meta));tmp.replace(output)
                outputs.append(dict(run=entry['name'],seed=seed,path=str(output),sha256=digest(output),checkpoint_sha256=digest(path)))
                timings.append(dict(run=entry['name'],seed=seed,data_seconds=read_s,model_encoder_and_support_seconds=support_s,cpu_transfer_copy_seconds=transfer_s,kmeans_seconds=select_s,complete_candidate_seconds=time.monotonic()-start,windows=S,agents=int(valid.sum()),density_forward_separately_instrumented=False))
                write(root/'reports/OUTPUT_PROGRESS.json',outputs);write(root/'reports/EVALUATION_TIMING.json',timings)
                print(f"Export {entry['name']} seed{seed}: {S} scenes, {int(valid.sum())} occurrences, {timings[-1]['complete_candidate_seconds']:.1f}s; truth not read",flush=True)
            assert before==state_digest(model.state_dict());del model,state
        if len(outputs)!=20:
            write(root/'reports/OUTPUTS_FROZEN.json',dict(status='EVALUATION_INCOMPLETE',outputs=outputs,reason='fewer than four fully selected runs'));return
        freeze=dict(status='ALL_OUTPUTS_FROZEN',created_unix=time.time(),split_hash=identity,outputs=outputs,truth_read=False)
        write(root/'reports/OUTPUTS_FROZEN.json',freeze);write(docs/'OUTPUTS_FROZEN.json',freeze)
        # First permitted future materialization for this recording in this task.
        scoring=PurposeDataset(split,'scoring',root/'reports/OUTPUTS_FROZEN.json');truth=np.zeros((S,nmax,2),np.float64)
        for row,index in enumerate(indices):
            item=scoring.get(index,'heldout_confirmation');truth[row,:len(item['future'])]=item['future'][:,-1].numpy()
        all_rows=[];nll=[]
        for entry in frozen['runs']:
            factory=BCEP if entry['model_kind']=='bcep_flow' else MDNProvider
            state=torch.load(entry['best']['path'],map_location='cpu',weights_only=False);model=factory(split['pred_length'],split['dt']);model.load_state_dict(state['model'],strict=True);model.eval()
            nll_sum=0.;count=0
            with torch.no_grad():
                for index in indices:
                    item=scoring.get(index,'heldout_confirmation');v=torch.ones(len(item['observed']),dtype=torch.bool)
                    c=model.loss_components(item['observed'],item['semantic_maps'],v,item['future'],.25)
                    nll_sum+=float(c['nll_final_sum']);count+=c['valid_agent_count']
            nll.append(dict(run=entry['name'],training_seed=entry['seed'],model_kind=entry['model_kind'],final_world_nll_agent_micro=nll_sum/count,agent_occurrences=count,scope='original continuous density, not representatives'))
            for bank in [x for x in outputs if x['run']==entry['name']]:
                assert digest(bank['path'])==bank['sha256']
                with np.load(bank['path'],allow_pickle=False) as z:
                    meta=json.loads(str(z['metadata_json']));assert meta['members']==members and meta['checkpoint_sha256']==entry['best']['sha256']
                    valid=z['valid'];counts=valid.sum(1)
                    for name,field in [('IID20','iid20'),('KMeans20','kmeans20')]:
                        prediction=z[field].astype(np.float64)
                        for group,mask in [('all',valid),('N1',valid&(counts[:,None]==1)),('Nge2',valid&(counts[:,None]>=2))]:
                            if mask.sum():all_rows.append(dict(run=entry['name'],model_kind=entry['model_kind'],training_seed=entry['seed'],inference_seed=bank['seed'],method=name,group=group,windows=int(mask.any(1).sum()),**endpoint_metrics(prediction,truth,mask)))
            del model,state
        write(root/'reports/CONFIRMATION_NLL.json',nll);write(docs/'CONFIRMATION_NLL.json',nll)
        for output_dir in (root/'reports',docs):
            with (output_dir/'PER_RUN_PER_SEED.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(all_rows[0]),lineterminator='\n');w.writeheader();w.writerows(all_rows)
        summary=[];per_training=[]
        metrics=['minFDE','recall_0.25','recall_0.5','recall_1.0','P50','P90','P95']
        for kind in ['bcep_flow','mdn8']:
            for method in ['IID20','KMeans20']:
                for group in ['all','N1','Nge2']:
                    means=[]
                    for seed in [3102,3103]:
                        rows=[r for r in all_rows if (r['model_kind'],r['method'],r['group'],r['training_seed'])==(kind,method,group,seed)];assert len(rows)==5
                        stats={k:dict(mean=float(np.mean([r[k] for r in rows])),inference_sd=float(np.std([r[k] for r in rows],ddof=1))) for k in metrics}
                        per_training.append(dict(model_kind=kind,method=method,group=group,training_seed=seed,metrics=stats));means.append(stats)
                    summary.append(dict(model_kind=kind,method=method,group=group,training_repeats=2,metrics={k:dict(mean=float(np.mean([v[k]['mean'] for v in means])),training_seed_sd=float(np.std([v[k]['mean'] for v in means],ddof=1)),individual_training_means=[v[k]['mean'] for v in means]) for k in metrics}))
        agg=dict(status='CONFIRMATION_COMPLETE',heldout_recording=split['chosen_recording'],counts=read(plan['contract'])['counts'],per_training_seed=per_training,across_training_seeds=summary,
                 statistical_scope='n=2 independently initialized training seeds; five inference seeds within each. One held-out recording, not independent windows or outer test.',provider_replacement='NOT_ESTABLISHED',full_ebjd_performance='NOT_EVALUATED')
        write(root/'reports/AGGREGATES.json',agg);write(docs/'AGGREGATES.json',agg);write(docs/'EVALUATION_TIMING.json',timings)
        write(root/'EVALUATION_PROGRESS.json',dict(phase='COMPLETE',complete_banks=20,truth_read=True))

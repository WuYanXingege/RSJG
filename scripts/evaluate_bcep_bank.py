"""Identity-checked internal export: one support draw, paired IID20/coverage20."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ebjd.data import SceneManifestDataset
from bcep.provider import BCEP
from bcep.evaluation import endpoint_metrics,energy_score_iid
from bcep.supervision import LikelihoodAccumulator
from bcep.contracts import (load_locked_split,manifest_members,check_item,validate_checkpoint,
    digest,canonical_hash,code_identity,require,PURPOSES)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['manifest','split-lock','contract','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--random-init-smoke',action='store_true')
    p.add_argument('--subset',choices=list(PURPOSES),required=True);p.add_argument('--purpose',choices=list(PURPOSES.values()),required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(2035,2040)));p.add_argument('--limit',type=int);p.add_argument('--require-complete',action='store_true')
    a=p.parse_args(argv)
    require(bool(a.checkpoint)!=a.random_init_smoke,'specify exactly one weight source')
    require(a.purpose==PURPOSES[a.subset],'purpose does not match subset')
    require(a.limit is None or a.limit>0,'limit must be positive')
    require(len(a.seeds)==len(set(a.seeds)) and set(a.seeds)<=set(range(2035,2040)),'inference seeds must be unique members of 2035..2039')
    split,contract,members=load_locked_split(a.split_lock,a.contract)
    manifest_sha=digest(a.manifest)
    if a.subset=='historical_reference':
        require(manifest_sha in contract['historical_manifest_hashes'],'historical manifest not explicitly bound')
        _,members=manifest_members(a.manifest);indices=list(range(len(members)))
    else:
        require(manifest_sha==split['source_sha256'],'internal source manifest mismatch')
        indices=list(split['dev_indices'] if a.subset=='internal_dev' else split['train_indices'])
        if a.subset=='interface_smoke':indices=[split['train_indices'][0],split['dev_indices'][0]]
    full_indices=list(indices)
    if a.limit is not None:indices=indices[:a.limit]
    if a.random_init_smoke:require(a.subset=='interface_smoke' and len(indices)<=4,'random initialization only allowed for interface smoke')
    # Validate provenance before loading data or generating any proposals.
    checkpoint=None
    if a.checkpoint:
        checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        validate_checkpoint(checkpoint,a.contract,a.split_lock)
        require(checkpoint.get('protocol')==dict(dt=contract['dt'],pred_length=contract['pred_length'],unit=contract['unit']),'checkpoint protocol mismatch')
    nonformal_reasons=[name for name,active in [
        ('random_initialization',a.random_init_smoke),('limit_requested',a.limit is not None),
        ('interface_smoke',a.subset=='interface_smoke'),('untrained_checkpoint',checkpoint is not None and checkpoint['update']==0),
        ('synthetic_data',contract.get('data_kind')=='synthetic')] if active]
    nonformal=bool(nonformal_reasons)
    if a.require_complete:require(not nonformal and len(indices)==len(full_indices),'cannot treat export as complete: '+','.join(nonformal_reasons))
    torch.set_num_threads(2);torch.manual_seed(3101)
    ds=SceneManifestDataset(a.manifest)
    require(ds.metadata['unit']==contract['unit'] and ds.metadata['dt_seconds']==contract['dt'] and ds.metadata['pred_length']==contract['pred_length'],'dataset protocol mismatch')
    model=BCEP(contract['pred_length'],contract['dt'])
    if checkpoint:model.load_state_dict(checkpoint['model'],strict=True)
    else:model.scales.copy_(torch.tensor(split['scales']));model.scales_fitted.fill_(True)
    model.eval();a.output.mkdir(parents=True,exist_ok=True);results=[]
    for seed in a.seeds:
        output_json=a.output/f'result_seed{seed}.json'
        require(not output_json.exists(),'refusing to overwrite existing export')
        rows=[];acc=LikelihoodAccumulator()
        with torch.no_grad():
            for index in indices:
                item=ds[index];check_item(item,members[index])
                observed=item['observed'];valid=torch.ones(len(observed),dtype=torch.bool)
                atoms,_=model.sample_support(observed,item['semantic_maps'],valid,item['scene_id'],item['agent_ids'],seed,count=256)
                iid=atoms[:20].clone();coverage,meta=model.select_support(atoms,valid,item['agent_ids'],20)
                require(torch.isfinite(atoms).all().item(),'nonfinite generated candidates')
                # Only after both candidate sets are fixed can truth enter scoring.
                truth=item['future'][:,-1];components=model.loss_components(observed,item['semantic_maps'],valid,item['future'])
                acc.update(components)
                rows.append(dict(iid20=iid.numpy(),coverage20=coverage.numpy(),truth=truth.numpy(),identity=members[index],n=len(observed),energy=energy_score_iid(atoms.numpy(),truth.numpy()),selector_metadata=meta))
        maximum=max(r['n'] for r in rows);truth=np.zeros((len(rows),maximum,2),np.float32);valid=np.zeros((len(rows),maximum),bool)
        ids=np.full((len(rows),maximum),'',dtype=f'<U{max(len(x) for r in rows for x in r["identity"]["agent_ids"])}')
        for i,r in enumerate(rows):truth[i,:r['n']]=r['truth'];valid[i,:r['n']]=True;ids[i,:r['n']]=r['identity']['agent_ids']
        metadata=dict(subset=a.subset,purpose=a.purpose,nonformal=nonformal,nonformal_reasons=nonformal_reasons,complete_members=indices==full_indices,independent_final_test=False,
            checkpoint_sha256=digest(a.checkpoint) if checkpoint else None,checkpoint_epoch=checkpoint['completed_epoch'] if checkpoint else None,
            checkpoint_training_updates=checkpoint['update'] if checkpoint else 0,export_optimizer_updates=0,weights_origin='random_initialization',
            checkpoint_selection_metric=checkpoint.get('selection_metric_name') if checkpoint else None,checkpoint_selection_value=checkpoint.get('selection_value') if checkpoint else None,
            member_hash=canonical_hash([r['identity'] for r in rows]),members=[r['identity'] for r in rows],split_hash=digest(a.split_lock),
            source_manifest_hash=manifest_sha,training_source_manifest_hash=split['source_sha256'],code_identity=code_identity(),seed=seed,K=20,M=256,coordinate_unit='m',dt=contract['dt'],pred_length=contract['pred_length'],data_kind=contract.get('data_kind','real'))
        metrics={}
        for method in ['iid20','coverage20']:
            pred=np.zeros((len(rows),20,maximum,2),np.float32)
            for i,r in enumerate(rows):pred[i,:,:r['n']]=r[method]
            count=valid.sum(1);groups={}
            for group,mask in [('all',np.ones(len(rows),bool)),('N_eq_1',count==1),('N_ge_2',count>=2)]:
                groups[group]=dict(endpoint_metrics(pred[mask],truth[mask],valid[mask]),windows=int(mask.sum())) if mask.any() else None
            metrics[method]=groups
            path=a.output/f'{method}_seed{seed}.npz';require(not path.exists(),'bank already exists')
            np.savez_compressed(path,endpoints=pred,truth=truth,valid=valid,scene_ids=np.array([r['identity']['scene_id'] for r in rows]),agent_ids=ids,metadata_json=json.dumps(dict(metadata,selector=method)))
        result=dict(metadata=metadata,metrics=metrics,likelihood=acc.metrics('eval'),iid256_energy_agent_micro=float(np.concatenate([r['energy'] for r in rows]).mean()),selector_indices=[r['selector_metadata'] for r in rows])
        output_json.write_text(json.dumps(result,indent=2));results.append(result)
    summary={'status':'NONFORMAL' if nonformal else 'COMPLETE_INTERNAL_OR_HISTORICAL','subset':a.subset,'purpose':a.purpose,'seeds':a.seeds,'five_inference_not_training_seeds':len(a.seeds)==5,'export_optimizer_updates':0,'per_seed_metrics':[r['metrics'] for r in results]}
    # Paired inference seeds are not independent training runs.
    aggregate={}
    for method in ['iid20','coverage20']:
        aggregate[method]={}
        for group in ['all','N_eq_1','N_ge_2']:
            values=[r['metrics'][method][group] for r in results]
            if values[0] is None:aggregate[method][group]=None;continue
            aggregate[method][group]={key:dict(mean=float(np.mean([v[key] for v in values])),sample_sd=float(np.std([v[key] for v in values],ddof=1)) if len(values)>1 else None) for key in values[0]}
    summary['inference_seed_statistics']=aggregate
    summary['paired_coverage_minus_iid']={key:[r['metrics']['coverage20']['all'][key]-r['metrics']['iid20']['all'][key] for r in results] for key in results[0]['metrics']['iid20']['all']}
    eligible_for_screen=not nonformal and a.subset=='internal_dev' and sorted(a.seeds)==list(range(2035,2040))
    summary['pilot_screening']={'status':'NOT_EVALUATED_NONFORMAL_OR_INCOMPLETE', 'requires_separate_training_authorization':True}
    if eligible_for_screen:
        iid=aggregate['iid20']['all'];cov=aggregate['coverage20']['all']
        checks={'minFDE_improvement_at_least_0.01m':iid['minFDE']['mean']-cov['minFDE']['mean']>=.01,
                'Recall_0.5_not_lower':cov['recall_0.5']['mean']>=iid['recall_0.5']['mean'],
                'P90_not_higher':cov['P90']['mean']<=iid['P90']['mean']}
        summary['pilot_screening']={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'scope':'same-checkpoint selector screen only; not provider replacement evidence'}
    (a.output/'SUMMARY.json').write_text(json.dumps(summary,indent=2));return summary

if __name__=='__main__':print(json.dumps(main()))

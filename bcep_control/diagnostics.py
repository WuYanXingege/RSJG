"""Frozen, observed-only generation followed by GT scoring; CPU-only analysis."""
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from bcep.contracts import canonical_hash, check_item, digest
from bcep.evaluation import endpoint_metrics, energy_score_iid
from bcep.supervision import LikelihoodAccumulator
from bcep.pilot_runner import atomic_json
from .contracts import state_digest
from .support_gap import support_gap_diagnostics


def grouped_metrics(prediction, truth, valid):
    count = valid.sum(1)
    groups = {'all':np.ones(len(count),bool), 'N_eq_1':count==1, 'N_ge_2':count>=2}
    return {g:dict(endpoint_metrics(prediction[mask],truth[mask],valid[mask]),windows=int(mask.sum()))
            for g,mask in groups.items() if mask.any()}


def aggregate_results(results):
    stats = {}
    for method in ['iid20','coverage20']:
        stats[method] = {}
        for group in ['all','N_eq_1','N_ge_2']:
            stats[method][group] = {}
            for key in results[0]['metrics'][method][group]:
                values = [r['metrics'][method][group][key] for r in results]
                stats[method][group][key] = dict(mean=float(np.mean(values)), sample_sd=float(np.std(values,ddof=1)))
    return stats


def gap_summary(rows):
    result = {}
    for group,subset in [('all',rows), ('N_eq_1',[r for r in rows if r['N']==1]), ('N_ge_2',[r for r in rows if r['N']>=2])]:
        block = dict(occurrences=len(subset),errors={},partition={})
        for name in ['pool_error','chosen_error','compression_gap']:
            a = np.array([r[name] for r in subset])
            block['errors'][name] = dict(mean=float(a.mean()),P50=float(np.quantile(a,.5)),P90=float(np.quantile(a,.9)),P95=float(np.quantile(a,.95)))
        for radius in [.25,.5,1.]:
            pool = np.array([r['pool_error'] for r in subset]); selected = np.array([r['chosen_error'] for r in subset])
            counts = [int((selected<=radius).sum()),int(((pool<=radius)&(selected>radius)).sum()),int((pool>radius).sum())]
            assert sum(counts)==len(subset)
            block['partition'][str(radius)] = {k:dict(count=c,rate=c/len(subset)) for k,c in zip(['selected_hit','pool_hit_selection_miss','pool_miss'],counts)}
        result[group] = block
    return result


def export_frozen(model, dataset, split, contract, checkpoint, output, kind, old_banks=None):
    output.mkdir(parents=True,exist_ok=True)
    model.eval(); before = state_digest(model.state_dict())
    indices = split['dev_indices']; members = [split['members'][i] for i in indices]
    assert len(indices)==320 and sum(len(m['agent_ids']) for m in members)==621
    assert canonical_hash(members)==contract['dev_member_hash']
    maximum=max(len(m['agent_ids']) for m in members)
    results=[]; gap_rows=[]; examples=[]; regression=[]
    started=time.monotonic()
    with torch.no_grad():
        for seed in range(2035,2040):
            atoms_all=np.zeros((len(indices),256,maximum,2),np.float32)
            iid=np.zeros((len(indices),20,maximum,2),np.float32);cov=np.zeros_like(iid)
            truth=np.zeros((len(indices),maximum,2),np.float32);valid_all=np.zeros((len(indices),maximum),bool)
            selection=np.full((len(indices),20,maximum),-1,np.int64)
            acc=LikelihoodAccumulator(); energies=[]; duplicates={m:[] for m in ['iid20','coverage20']};seed_rows=[]
            original=None
            if old_banks:
                old_iid=np.load(old_banks/f'iid20_seed{seed}.npz',allow_pickle=False)
                old_cov=np.load(old_banks/f'coverage20_seed{seed}.npz',allow_pickle=False)
                original=json.loads((old_banks/f'result_seed{seed}.json').read_text())
                assert original['metadata']['member_hash']==contract['dev_member_hash']
                assert original['metadata']['checkpoint_sha256']==digest(checkpoint)
                assert original['metadata']['seed']==seed and original['metadata']['M']==256 and original['metadata']['K']==20
                assert original['metadata']['members']==members
            exact=True;max_absolute=0.; max_distance=0.
            for row,index in enumerate(indices):
                item=dataset[index];check_item(item,split['members'][index]);n=len(item['observed'])
                valid=torch.ones(n,dtype=torch.bool)
                # Neither sampling nor selection accepts a future target.
                atoms,_=model.sample_support(item['observed'],item['semantic_maps'],valid,item['scene_id'],item['agent_ids'],seed,256)
                chosen,meta=model.select_support(atoms,valid,item['agent_ids'],20)
                selected=np.array([a['indices'] for a in meta['agents']],np.int64).T
                np.testing.assert_array_equal(chosen.numpy(),np.take_along_axis(atoms.numpy(),selected[...,None],axis=0))
                if original:
                    assert meta==original['selector_indices'][row]
                    for a,b in [(atoms[:20].numpy(),old_iid['endpoints'][row,:,:n]),(chosen.numpy(),old_cov['endpoints'][row,:,:n])]:
                        exact=exact and np.array_equal(a,b);max_absolute=max(max_absolute,float(np.abs(a-b).max()))
                        max_distance=max(max_distance,float(np.linalg.norm(a-b,axis=-1).max()))
                        np.testing.assert_allclose(a,b,atol=1e-7,rtol=1e-6)
                    assert list(old_iid['scene_ids'])[row]==item['scene_id']
                    assert list(old_iid['agent_ids'][row,:n])==item['agent_ids']
                    assert old_iid['valid'][row].sum()==n and old_cov['valid'][row].sum()==n
                # GT enters only after the complete support and both outputs are fixed.
                target=item['future'][:,-1].numpy()
                if original:np.testing.assert_array_equal(target,old_iid['truth'][row,:n])
                diag=support_gap_diagnostics(atoms.numpy(),selected,target)
                atoms_all[row,:,:n]=atoms.numpy();iid[row,:,:n]=atoms[:20].numpy();cov[row,:,:n]=chosen.numpy()
                selection[row,:,:n]=selected;truth[row,:n]=target;valid_all[row,:n]=True
                energies.extend(energy_score_iid(atoms.numpy(),target).tolist())
                acc.update(model.loss_components(item['observed'],item['semantic_maps'],valid,item['future'],.25))
                for agent in range(n):
                    record=dict(seed=seed,index=index,scene_id=item['scene_id'],agent_id=item['agent_ids'][agent],N=n,
                                pool_error=float(diag['pool_error'][agent]),chosen_error=float(diag['chosen_error'][agent]),compression_gap=float(diag['compression_gap'][agent]))
                    seed_rows.append(record)
                    for method,pred in [('iid20',iid[row,:,agent]),('coverage20',cov[row,:,agent])]:
                        duplicates[method].append(1-len(np.unique(pred,axis=0))/20)
                    if seed==2035:
                        category='selected_hit' if record['chosen_error']<=.5 else 'pool_hit_selection_miss' if record['pool_error']<=.5 else 'pool_miss'
                        examples.append(dict(**record,category=category,truth=target[agent].tolist(),support=atoms[:,agent].tolist(),selected_indices=selected[:,agent].tolist()))
            metadata=dict(model_kind=kind,checkpoint_sha256=digest(checkpoint),member_hash=canonical_hash(members),members=members,
                          seed=seed,M=256,K=20,unit='m',export_optimizer_updates=0,complete=True,
                          split_hash=contract['split_hash'],source_manifest_hash=contract['source_manifest_hash'],dt=contract['dt'],
                          pred_length=contract['pred_length'],scene_ids=[m['scene_id'] for m in members],agent_ids=[m['agent_ids'] for m in members])
            np.savez_compressed(output/f'support_seed{seed}.npz',atoms=atoms_all,iid20=iid,coverage20=cov,truth=truth,
                                valid=valid_all,selected_indices=selection,metadata_json=json.dumps(metadata))
            result=dict(seed=seed,metadata=metadata,metrics={m:grouped_metrics(p,truth,valid_all) for m,p in [('iid20',iid),('coverage20',cov)]},
                        likelihood=acc.metrics('eval'),energy_iid256_agent_micro=float(np.mean(energies)),
                        duplicate_fraction={m:float(np.mean(v)) for m,v in duplicates.items()},support_gap=gap_summary(seed_rows))
            atomic_json(result,output/f'result_seed{seed}.json');results.append(result);gap_rows.extend(seed_rows)
            if original:
                regression.append(dict(seed=seed,exactly_equal=exact,max_absolute=max_absolute,max_euclidean=max_distance,
                                       selector_metadata_exact=True,identities_exact=True,atol=1e-7,rtol=1e-6))
                old_iid.close();old_cov.close()
            print(f'{kind} frozen export seed={seed} scenes=320 occurrences=621 complete',flush=True)
    after=state_digest(model.state_dict());assert before==after
    per_seed_gap={str(s):gap_summary([r for r in gap_rows if r['seed']==s]) for s in range(2035,2040)}
    examples=sorted(examples,key=lambda r:(r['scene_id'],r['agent_id']))
    examples=[r for cat in ['selected_hit','pool_hit_selection_miss','pool_miss'] for r in [x for x in examples if x['category']==cat][:2]]
    atomic_json(dict(label='post-hoc failure visualization; seed2035, radius0.5m, identity-sorted first2 per category; no selection feedback',examples=examples),output/'EXAMPLES.json')
    with (output/'SUPPORT_GAP_DIAGNOSTICS.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(gap_rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(gap_rows)
    gap_report=dict(per_seed=per_seed_gap,pooled_description=gap_summary(gap_rows),
                    pooled_scope='3105 occurrence-seed pairs, not independent people or training replicates',
                    bound_scope='M256 nearest-point diagnostic, NOT a K20 deployable predictor; quantiles not additively decomposed')
    atomic_json(gap_report,output/'SUPPORT_GAP_DIAGNOSTICS.json')
    summary=dict(status='COMPLETE',model_kind=kind,checkpoint_sha256=digest(checkpoint),parameter_count=sum(p.numel() for p in model.parameters()),
                 state_hash_before=before,state_hash_after=after,optimizer_updates=0,seeds=list(range(2035,2040)),windows_per_seed=320,
                 occurrences_per_seed=621,statistics=aggregate_results(results),regression=regression,
                 final_nll_agent_micro=results[0]['likelihood']['eval_nll_final_agent_micro'],
                 energy_iid256=dict(mean=float(np.mean([r['energy_iid256_agent_micro'] for r in results])),sample_sd=float(np.std([r['energy_iid256_agent_micro'] for r in results],ddof=1))),
                 duplicate_fraction={m:float(np.mean([r['duplicate_fraction'][m] for r in results])) for m in ['iid20','coverage20']},wall_seconds=time.monotonic()-started)
    atomic_json(summary,output/'SUMMARY.json')
    return summary

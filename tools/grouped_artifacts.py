#!/usr/bin/env python3
"""plan/canary/export/verify genuine grouped targets and observation reuse."""
import argparse,copy,json,sys,time
from pathlib import Path
from collections import Counter,defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.p2_grouped_artifacts import *
from src.p2_data import combine_train,base_packing_plan

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['plan','canary','export','verify'])
    ap.add_argument('--archive',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.archive=a.archive.resolve();a.output=a.output.resolve();torch.set_num_threads(4);torch.set_num_interop_threads(1)
    mp=a.archive/'PREPARATION_MANIFEST.json';m=json.loads(mp.read_text());reg=Registry(m,m['manifest_hash'])
    if 'records' in m:
        atomic_json(m,a.output/'original_inline_preparation_manifest.json')
        m=write_manifest(m,reg.records,mp);reg=Registry(m,m['manifest_hash'])
    grant=PreparationGrant(reg,make_grant(reg));atomic_json(grant.value,a.archive/'PURPOSE_GRANT.json')
    canary=[];seen=set()
    for r in reg.records:
        if r['role']!='outer' and r['source']['source_sha256'] not in seen:
            canary.append(r['window_id']);seen.add(r['source']['source_sha256'])
    plan=dict(canary=canary,rule='first immutable registered window per non-outer source; before loss',
        counts=COUNTS,target_counts={'train':2537,'inner_valid':1267,'outer':0},grant_hash=grant.value['manifest_hash'])
    if a.mode=='plan':atomic_json(plan,a.archive/'ARTIFACT_PLAN.json');print(json.dumps(plan));return
    if json.loads((a.archive/'ARTIFACT_PLAN.json').read_text())!=plan:raise RuntimeError('plan changed')
    if a.mode=='export' and not (a.archive/'CANARY_RESULT.json').exists():raise RuntimeError('canary required')
    previous=ROOT/'docs/joint_dependency_v2/reviews/2026-10-05_bda1ae1_p2_observation_provenance_cpu/P2_RUNTIME_MANIFEST.json'
    oldm=json.loads(previous.read_text());old=Registry(oldm,oldm['manifest_hash'])
    rows=copy.deepcopy(reg.records);ledger=Counter();catalog=[];by_source=defaultdict(list);roundtrip_max=0.
    chosen=[r for r in rows if a.mode!='canary' or r['window_id'] in canary]
    for r in chosen:by_source[r['source']['source_sha256']].append(r)
    start=time.time()
    for source,records in by_source.items():
        scene=records[0]['source']['scene_family'];assets=load_assets(reg.manifest['assets'][scene])
        futures=scan_targets(records,grant,ledger) if records[0]['role']!='outer' and a.mode!='verify' else {}
        for r in records:
            obs,chain=migrate_observation(old.by_id[r['window_id']],r,grant)
            if old.by_id[r['window_id']]['source']==r['source']:
                item=dict(window_id=r['window_id'],kind='observation',role=r['role'],artifact=chain['old_artifact'],
                    content_sha256=chain['old_content_sha256'],reuse_chain=chain,grant_hash=grant.value['manifest_hash'],
                    source_identity=digest(r['source']),parent_manifest_hash=m['manifest_hash'])
            else:item=save_payload(obs,r,'observation',a.output,grant,chain)
            r['artifacts']['observation']=item['artifact'];catalog.append(item);ledger['observation_verified']+=1
            if r['role']=='outer':
                if r['artifacts'].get('target'):raise RuntimeError('outer target present')
                continue
            if a.mode=='verify':
                side=a.output/'target'/(r['window_id']+'.json');meta=json.loads(side.read_text())
                r['artifacts']['target']=dict(path=str(side.with_suffix('.pt')),sha256=meta['sha256'])
                value=FileProviders().read(r,'target');validate_target(value,r)
            else:value=build_target(futures[r['window_id']],r,assets)
            target=save_payload(value,r,'target',a.output,grant)
            r['artifacts']['target']=target['artifact'];catalog.append(target)
            full=combine_train(obs,value)
            if full['input_traj_maps'].shape[1]!=20 or not torch.equal(full['abs_pixel_coord'][:8],obs['abs_pixel_coord']):raise RuntimeError('full20/obs8')
            if a.mode=='verify':
                from src.p2_grouped_training import GeometryScene
                from src.p2_data import observation_inputs
                geom=GeometryScene(scene,reg.manifest['assets'][scene]['H.txt'])
                inputs=observation_inputs(obs,geom,'cpu')
                assert inputs['x_augmented'].shape==(8,len(r['source']['agent_ids']),8)
                assert torch.isfinite(inputs['x_augmented']).all()
                # Independent homogeneous roundtrip, only authorized target slots.
                from src.data_src.data_utils import pixel2world_numpy
                pixel=value['abs_pixel_coord'].numpy()
                if scene in ('eth','hotel'):pixel=pixel[...,[1,0]]
                world=pixel2world_numpy(pixel.reshape(-1,2),assets['H'])
                back=world2pixel_numpy(world,assets['H_inv'])
                err=float(np.max(np.abs(back-pixel.reshape(-1,2))));roundtrip_max=max(roundtrip_max,err)
                if err>1e-7:raise RuntimeError('geometry roundtrip')
                ledger['observation_only_input_adapter_verified']+=1
            ledger[r['role']+'_targets_verified']+=1
            if ledger['observation_verified']%100==0:print(json.dumps(dict(progress=ledger,seconds=time.time()-start)),flush=True)
    result=dict(mode=a.mode,status='ARTIFACT_VALIDATION_ONLY_NOT_PRODUCTION',counts=dict(ledger),seconds=time.time()-start,roundtrip_max_abs=roundtrip_max,
        outer_targets=0,inner_outer_gradient=0,teacher=0,production_loader='BLOCKED_PENDING_ACTUAL_QUALIFICATION')
    if a.mode=='canary':atomic_json(result,a.archive/'CANARY_RESULT.json')
    else:
        assert ledger['observation_verified']==4249 and ledger['train_targets_verified']==2537 and ledger['inner_valid_targets_verified']==1267
        runtime=write_manifest(dict(m,preparation_parent_hash=m['manifest_hash']),rows,a.archive/'GROUPED_RUNTIME_MANIFEST.json')
        cat=write_shards(catalog,a.archive/('catalog_'+a.mode),'items')
        result.update(catalog_shards=cat,runtime_manifest_hash=runtime['manifest_hash'],packing={
            role:dict(packs=len(base_packing_plan(reg,role)),exposures=sum(map(len,base_packing_plan(reg,role))),sha256=digest(base_packing_plan(reg,role))) for role in ('train','inner_valid')})
        atomic_json(result,a.archive/('ARTIFACT_'+a.mode.upper()+'_RESULT.json'))
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()

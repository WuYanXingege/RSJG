#!/usr/bin/env python3
"""Explicit CPU-only plan/canary/export/verify. Default performs no work."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
from collections import Counter,defaultdict

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.p2_observation_guard import install


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['plan','canary','export','verify'])
    p.add_argument('--enable-observation-preparation',action='store_true')
    p.add_argument('--device',choices=['cpu'],required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--manifest-hash',required=True)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not a.enable_observation_preparation:p.error('preparation is disabled by default')
    permit,guard_report=install()
    from src.p2_observation import (registry_from_file,make_grant,PreparationGate,inspect_assets,
        load_assets,scan_observations,build_payload,save_artifact,verify_artifact,estimate_record,
        Budget,atomic_json,producer_hash)
    from src.p2_protocol import digest,file_hash,sealed,ROOT,Registry
    import torch
    import numpy as np
    import PIL
    a.archive=a.archive.resolve();a.output=a.output.resolve()
    if not a.output.is_relative_to(ROOT/'outputs') or 'p2_observation' not in a.output.name:
        raise RuntimeError('new dedicated P2 observation outputs root required')
    a.archive.mkdir(parents=True,exist_ok=True)
    attempt=time.time_ns();report=dict(phase=a.phase,pid=os.getpid(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        counters={},errors=[],status='RUNNING')
    ledger=Counter({k:0 for k in ['raw_hash_passes','raw_hash_bytes','raw_scan_bytes','id_rows_parsed',
        'unique_raw_obs_xy_decodes','window_local_observation_slots','observation_tensor_windows',
        'real_future_queries','real_target_tensors','real_teacher_writes','full_source_pickle_loads',
        'static_asset_conversions','completed','reused']})
    budget=None
    try:
        reg=registry_from_file(a.manifest,a.manifest_hash)
        groups=defaultdict(list)
        for r in reg.records:groups[r['source']['source_sha256']].append(r)
        budget=Budget(a.output)
        # Conservative cumulative task wall budget includes earlier investigation/tests.
        if a.phase=='plan':
            from datetime import datetime,timezone
            original_start=datetime(2026,10,5,7,46,57,tzinfo=timezone.utc).timestamp()
            if original_start<budget.state['start']:
                budget.state['start']=original_start
                atomic_json(budget.state,a.output/'budget.json')
        budget.check()
        gp=a.archive/'PREPARATION_GRANT.json'
        if a.phase=='plan':
            grant=make_grant(reg)
            if gp.exists() and json.loads(gp.read_text())!=grant:raise RuntimeError('existing grant differs')
            atomic_json(grant,gp)
        else:grant=json.loads(gp.read_text())
        gate=PreparationGate(reg,grant,make_grant(reg)['manifest_hash'])
        for r in reg.records:gate.authorize(r,'artifact_preparation')
        if a.phase=='plan':
            bindings={s:inspect_assets(s) for s in sorted({r['source']['scene_family'] for r in reg.records})}
            atomic_json(bindings,a.archive/'MAP_ASSET_PROVENANCE.json')
            source_rows=[]
            for sha,rs in sorted(groups.items()):
                s=rs[0]['source'];raw=ROOT/s['original_source_path']
                if file_hash(raw)!=sha:raise RuntimeError('plan source SHA')
                ledger['raw_hash_passes']+=1;ledger['raw_hash_bytes']+=raw.stat().st_size
                aliases=[]
                for alias in rs[0]['aliases']:
                    ap=ROOT/alias;actual=file_hash(ap)
                    if actual!=sha:raise RuntimeError('source alias hash mismatch')
                    aliases.append(dict(path=alias,sha256=actual,bytes=ap.stat().st_size))
                    ledger['raw_hash_passes']+=1;ledger['raw_hash_bytes']+=ap.stat().st_size
                source_rows.append(dict(local_source_path=s['original_source_path'],local_source_sha256=sha,
                    aliases=aliases,scene=s['scene_family'],logical_role=rs[0]['role'],windows=len(rs),
                    agent_window_exposures=sum(len(r['source']['agent_ids']) for r in rs),
                    release_owner='UNKNOWN (GDTS README attributes package to Goal-SAR; no archive/member binding)',
                    release_version_or_commit='UNKNOWN',archive_sha256=None,archive_member=None,member_sha256=None,
                    converter_commit_and_path='UNKNOWN',converter_parameters={},converter_execution_log_sha256=None,
                    recording_id='UNKNOWN',camera_session_clip='UNKNOWN',frame_offset_resampling_rule='UNKNOWN',
                    agent_id_namespace_remap_rule='UNKNOWN',evidence_paths_and_sha256=[],status='PARTIAL_BYTE_BINDING_ONLY'))
            atomic_json(source_rows,a.archive/'RECORDING_RELEASE_BINDING.json')
            selected=[min(rs,key=lambda r:r['window_id'])['window_id'] for _,rs in sorted(groups.items())]
            atomic_json(dict(rule='first lexicographic window_id per source SHA',window_ids=selected,
                window_ids_hash=digest(selected),source_count=8,parent_hash=a.manifest_hash),a.archive/'CANARY_PLAN.json')
            total=sum(estimate_record(r,bindings[r['source']['scene_family']]) for r in reg.records)
            free=__import__('shutil').disk_usage(a.output).free
            plan=dict(expected=4249,estimated_payload_with_overhead_bytes=total,free_bytes=free,
                max_record_estimate=max(estimate_record(r,bindings[r['source']['scene_family']]) for r in reg.records),
                storage_cap_bytes=32*1024**3,estimated_peak_rss_bound_bytes=2*1024**3,
                workers=1,compute_threads=1,roles=reg.counts,source_metadata=source_rows,
                grant_hash=grant['manifest_hash'],producer_hash=producer_hash(),
                environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,PIL=PIL.__version__),
                cohort='registered offline complete-trajectory cohort; not online agent-selection certification')
            atomic_json(plan,a.archive/'RESOURCE_PLAN.json')
            if total>32*1024**3 or free<total+1024**3:raise RuntimeError('BUDGET_PLAN_STORAGE')
            report['status']='PLANNED'
        else:
            bindings=json.loads((a.archive/'MAP_ASSET_PROVENANCE.json').read_text())
            canary=json.loads((a.archive/'CANARY_PLAN.json').read_text())
            if canary['window_ids_hash']!=digest(canary['window_ids']) or canary['parent_hash']!=a.manifest_hash:
                raise RuntimeError('canary plan binding')
            actual=[min(rs,key=lambda r:r['window_id'])['window_id'] for _,rs in sorted(groups.items())]
            if canary['window_ids']!=actual:raise RuntimeError('canary selection changed')
            if a.phase in ('export','verify'):
                cert=json.loads((a.archive/'CANARY_ACCEPTANCE.json').read_text())
                if cert['status']!='PASSED' or cert['window_ids_hash']!=canary['window_ids_hash'] or cert['producer_hash']!=producer_hash():
                    raise RuntimeError('canary not accepted for this producer')
            wanted=set(canary['window_ids']) if a.phase=='canary' else set(reg.by_id)
            catalog=[]
            for sha,rs_all in sorted(groups.items()):
                rs=[r for r in rs_all if r['window_id'] in wanted]
                binding=bindings[rs[0]['source']['scene_family']]
                remaining=[]
                for r in rs:
                    budget.check()
                    if (a.output/(r['window_id']+'.pt')).exists() or (a.output/(r['window_id']+'.json')).exists() or a.phase=='verify':
                        catalog.append(verify_artifact(r,a.output,gate,binding,permit));ledger['reused']+=1
                    else:remaining.append(r)
                if remaining:
                    coords=scan_observations(remaining,gate,ledger)
                    assets=load_assets(binding);ledger['static_asset_conversions']+=1
                    for r in remaining:
                        budget.check(estimate_record(r,binding))
                        payload,outside=build_payload(coords[r['window_id']],r,assets)
                        ledger['observation_tensor_windows']+=1
                        row=save_artifact(payload,r,a.output,gate,binding,permit,outside)
                        budget.bytes+=row['bytes']+row['sidecar_bytes'];catalog.append(row)
                        ledger['completed']+=1
                        if ledger['completed']%100==0:
                            print(json.dumps(dict(completed=ledger['completed'],phase=a.phase,elapsed=budget.report()['wall_seconds'])),flush=True)
                        del payload
                    del coords,assets
            if {r['window_id'] for r in catalog}!=wanted or len(catalog)!=len(wanted):raise RuntimeError('coverage set mismatch')
            catalog.sort(key=lambda r:r['window_id'])
            if a.phase=='canary':
                atomic_json(dict(status='PASSED',window_ids_hash=canary['window_ids_hash'],
                    producer_hash=producer_hash(),catalog=catalog,provenance='PENDING'),
                    a.archive/'CANARY_ACCEPTANCE.json')
            else:
                shards=[]
                for i in range(0,len(catalog),250):
                    target=a.archive/'catalog'/('part_%02d.json'%(i//250))
                    atomic_json(catalog[i:i+250],target)
                    shards.append(dict(path=str(target.relative_to(ROOT)),sha256=file_hash(target),count=min(250,len(catalog)-i)))
                atomic_json(dict(status='COMPLETE',expected=4249,completed=len(catalog),
                    roles=dict(Counter(r['role'] for r in catalog)),record_files=shards,
                    total_artifact_bytes=sum(r['bytes']+r['sidecar_bytes'] for r in catalog),
                    max_artifact_bytes=max(r['bytes'] for r in catalog),readback_all=True,
                    grant_hash=grant['manifest_hash'],parent_manifest_hash=a.manifest_hash,
                    content_hash_rule='sorted key/shape/dtype/contiguous tensor bytes'),
                    a.archive/'OBSERVATION_ARTIFACT_CATALOG.json')
                refs={r['window_id']:r['observation'] for r in catalog}
                records=copy_records(reg.records,refs)
                m=dict(reg.manifest);m.pop('record_files');m.pop('manifest_hash')
                parts=[]
                for i in range(0,len(records),250):
                    target=a.archive/'runtime_records'/('part_%02d.json'%(i//250))
                    # Registry runtime shards are JSONL; use one compact record per line.
                    target=target.with_suffix('.jsonl');target.parent.mkdir(exist_ok=True)
                    target.write_text(''.join(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n' for r in records[i:i+250]))
                    parts.append(dict(path=str(target.relative_to(ROOT)),sha256=file_hash(target),count=min(250,len(records)-i)))
                m['record_files']=parts;m['preparation_grant_hash']=grant['manifest_hash']
                m['observation_status']='OBSERVATION_ARTIFACT_PREPARED_PROVENANCE_PENDING'
                m=sealed(m);Registry(m,m['manifest_hash'])
                atomic_json(m,a.archive/'P2_RUNTIME_MANIFEST.json')
            report.update(status='PASSED',completed=len(catalog),roles=dict(Counter(r['role'] for r in catalog)),
                out_of_raster_obs_slots=sum(r['out_of_raster_obs_slots'] for r in catalog),
                missing=[],failed=[],hash_mismatch=0)
    except Exception as exc:
        report.update(status='FAILED',errors=[str(exc)],traceback=traceback.format_exc())
    finally:
        report['counters']=dict(ledger);report['guard']=guard_report()
        report['budget']=budget.report() if budget else None
        atomic_json(report,a.archive/('RUN_'+a.phase+'_'+str(attempt)+'.json'))
        print(json.dumps({k:report[k] for k in ('phase','status','errors','counters')}),flush=True)
    return 0 if report['status'] in ('PASSED','PLANNED') else 2


def copy_records(records,refs):
    import copy
    result=copy.deepcopy(records)
    for r in result:r['artifacts']['observation']=refs[r['window_id']]
    return result


if __name__=='__main__':raise SystemExit(main())

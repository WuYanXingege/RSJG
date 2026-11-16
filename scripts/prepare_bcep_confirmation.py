"""Metadata-only recording choice, then train-only normalization; no optimizer."""
import sys,json,time,subprocess,collections,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from bcep.contracts import digest,canonical_hash,manifest_members
from bcep.motion_frame import frame,to_local
from bcep_confirmation.common import ROOT,read,write,budget
from bcep_confirmation.protocol import PurposeDataset,validate_split

def main(paths_file):
    p=read(paths_file);a=Path(p['assets']);docs=Path(p['docs']);project=ROOT.parent
    with budget(a,'CPU_PREPARATION',1800):
        old=read(project/'RSJG_EBJD_assets/bcep/eth_internal_split.json')
        source=Path(old['source_manifest']);assert digest(source)==old['source_sha256']
        manifest,rows=manifest_members(source);assert rows==old['members']
        assert manifest['metadata']['split']=='train' and manifest['metadata']['future_used_for_crop'] is False
        mapping={'biwi_hotel':'eth/hotel','crowds_zara01':'ucy/zara01','crowds_zara02':'ucy/zara02','crowds_zara03':'ucy/zara03','students001':'ucy/students01','students003':'ucy/students03','uni_examples':'ucy/uni_examples'}
        seqs={r['sequence'] for r in rows};assert seqs==set(mapping),'BLOCKED_SPLIT_IDENTITY: unresolved sequence'
        sources=[]
        for r in manifest['source_files']:
            assert digest(r['path'])==r['sha256']
            seq=Path(r['path']).stem.replace('_train','').replace('_val','');assert seq in mapping
            sources.append({**r,'source_sequence':seq,'recording_id':mapping[seq]})
        assert len(sources)==len(seqs) and len({r['sha256'] for r in sources})==len(sources)
        # The exporter consumes these canonical source tables directly. No
        # per-window rotation, resampling, mirroring or alias expansion is used.
        counts=collections.Counter(mapping[r['sequence']] for r in rows)
        dev=old['dev_indices'];dev_recs={mapping[rows[i]['sequence']] for i in dev}
        excluded=[i for i,r in enumerate(rows) if mapping[r['sequence']] in dev_recs and i not in set(dev)]
        candidates=[]
        for rec,n in counts.items():
            remaining=len(rows)-len(dev)-len(excluded)-n
            eligible=rec not in dev_recs and n>=100 and remaining>=1895
            candidates.append(dict(recording_id=rec,windows=n,remaining_train=remaining,eligible=eligible))
        eligible=sorted([x for x in candidates if x['eligible']],key=lambda x:(-x['windows'],x['recording_id']))
        assert eligible,'BLOCKED_NO_ELIGIBLE_GROUP'
        chosen=eligible[0]['recording_id'];held=[i for i,r in enumerate(rows) if mapping[r['sequence']]==chosen]
        train=sorted(set(range(len(rows)))-set(dev)-set(held)-set(excluded))
        provenance=dict(source_files=sources,exporter=dict(path=str(ROOT/'ebjd/export_ethucy.py'),sha256=digest(ROOT/'ebjd/export_ethucy.py')),
            recording_ids=mapping,recording_identity_basis='Canonical original annotation-sequence IDs, not map names. Direct source table -> synchronized windows; no offline augmentation or renamed derivative among the seven source entries.',
            alias_rules=['strip _train/_val for source labels as actual exporter does','students001 -> students01, students003 -> students03','crowds_zara0N -> zara0N','same source SHA must not occur in separate purposes'],
            public_sources=['https://github.com/crowdbotp/OpenTraj/blob/master/datasets/UCY/README.md','https://github.com/vita-epfl/trajnetplusplusdataset/blob/master/trajnetdataset/convert.py'],
            map_groups={'univ':['students001','students003','uni_examples'],'zara_shared_map':['crowds_zara01','crowds_zara02','crowds_zara03'],'hotel':['biwi_hotel']},
            scope='Recording-held-out, not unseen-map or unseen-location; original videos were not independently downloaded. Same map does not imply same annotated recording.',
            static_maps=manifest['map_sources'],future_derived_training_inputs=False,
            history='Confirmation may have been trained on by old models or observed in prior project work. Only these four new models exclude it from fitting and selection.')
        for maps in manifest['map_sources'].values():
            for field in ('semantic','homography'):assert digest(maps[field+'_path'])==maps[field+'_sha256']
        split=dict(schema='bcep-recording-confirmation-v1',source_manifest=str(source),source_sha256=digest(source),unit='m',dt=.4,pred_length=12,obs_length=8,
            members=rows,recording_ids=mapping,train_indices=train,selection_dev_indices=dev,heldout_confirmation_indices=held,excluded_indices=excluded,
            excluded_reasons={str(i):'alias of selection recording outside original selected members' for i in excluded},
            scale_source='train_only',scales=None,member_hashes={k:canonical_hash([rows[i] for i in v]) for k,v in [('train',train),('selection_dev',dev),('heldout_confirmation',held),('excluded',excluded)]},
            chosen_recording=chosen,candidate_groups=candidates,choice_rule='eligible recording windows descending, canonical recording_id lexicographic',provenance=provenance)
        validate_split(split)
        # Choice is atomically frozen BEFORE reading ANY training future for scales.
        write(a/'identity/METADATA_CHOICE_LOCK.json',split)
        dataset=PurposeDataset(split,'scale');total=torch.zeros(12,2,dtype=torch.float64);agents=0
        for i in train:
            item=dataset.get(i,'train');obs=item['observed'].double();future=item['future'].double()
            for h in range(1,13):
                b,r,s=frame(obs,h,.4,torch.ones(2,dtype=torch.float64));total[h-1]+=to_local(future[:,h-1],b,r,s).square().sum(0)
            agents+=len(obs)
        split['scales']=(total/agents).sqrt().clamp_min(.1).tolist();split['scale_agents']=agents
        split['scale_access_indices_hash']=canonical_hash([x[0] for x in dataset.access]);assert [x[0] for x in dataset.access]==train
        write(a/'identity/SPLIT_LOCK.json',split)
        small={k:v for k,v in split.items() if k not in ['members','train_indices','selection_dev_indices','heldout_confirmation_indices','excluded_indices']}
        small.update(full_lock_path=str(a/'identity/SPLIT_LOCK.json'),full_lock_sha256=digest(a/'identity/SPLIT_LOCK.json'),canonical_split_hash=canonical_hash(split),
                     counts={purpose:dict(windows=len(ids),agent_occurrences=sum(len(rows[i]['agent_ids']) for i in ids)) for purpose,ids in [('train',train),('selection_dev',dev),('heldout_confirmation',held),('excluded',excluded)]})
        write(docs/'SPLIT_LOCK.json',small)
        contract=dict(schema='bcep-confirmation-contract-v1',split_sha256=digest(a/'identity/SPLIT_LOCK.json'),canonical_split_hash=canonical_hash(split),source_manifest_sha256=digest(source),
            selection='selection_dev final world NLL agent micro; exact ties earlier',train_weighting='one scene/update; original total composite NLL',
            lambda_aux=.25,lr=1e-4,lr_gamma=1.0,FP32=True,patience=3,max_epochs=6,M=256,K=20,inference_seeds=list(range(2035,2040)),
            default_allow_real_training=False,confirmation_access='only after all four runs terminal; all outputs frozen before future reads',counts=small['counts'])
        write(docs/'DATA_CONTRACT.json',contract);write(a/'identity/DATA_CONTRACT.json',contract)
        matrix=[dict(name=f'{kind}_seed{seed}',model_kind=kind,seed=seed,max_epochs=6,max_updates=6*len(train),budget_seconds=3600) for kind,seed in [('bcep_flow',3102),('mdn8',3102),('mdn8',3103),('bcep_flow',3103)]]
        plan={**p,'evidence_sha':'074d54b04be6a717b2c248a1fe3c2596a4fd9be8','authorization':'User explicitly forwarded BCEP_NewSequence_IndependentSeeds_Package.zip with 执行 and 上传github; four bounded new provider-only runs authorized',
            'default_allow_real_training':False,'authorized_run_matrix':matrix,'split_lock':str(a/'identity/SPLIT_LOCK.json'),'split_sha256':digest(a/'identity/SPLIT_LOCK.json'),
            'contract':str(a/'identity/DATA_CONTRACT.json'),'contract_sha256':digest(a/'identity/DATA_CONTRACT.json'),'budgets':dict(cpu_preflight_seconds=1800,gpu_preflight_seconds=600,training_per_run_seconds=3600,training_total_seconds=14400,readonly_cpu_seconds=7200),
            'selector':read(ROOT/'docs/research/bcep_v1/bcep_frozen_selector_control_20261116_215702/SELECTOR_CONFIG.json')['kmeans'],
            'run_command':f'{project}/.conda/rsjg/bin/python -B -u scripts/run_bcep_confirmation.py --plan {docs}/PLAN_AND_AUTHORIZATION.json --execute-authorized-four-runs',
            'no_bridge_or_coupling_training':True,'no_automatic_followup':True}
        write(docs/'PLAN_AND_AUTHORIZATION.json',plan);write(a/'identity/PLAN_AND_AUTHORIZATION.json',plan);write(docs/'RUN_MATRIX.json',matrix)
        print(json.dumps(small['counts']),chosen,flush=True)
if __name__=='__main__':main(sys.argv[1])

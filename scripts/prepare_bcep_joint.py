import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,sys,time,json,subprocess,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from bcep.provider import BCEP
from bcep_confirmation.protocol import PurposeDataset
from bcep.contracts import canonical_hash,digest
from selector_control.algorithms import kmeans_centers
from selector_control.core import local_seed
from ebjd.representation import EndpointBridgeDecomposition,cv_baseline
from bcep_joint.common import read,write,budget,ROOT,state_digest

def main(paths):
    root=Path(paths['assets']);docs=Path(paths['docs']);old=ROOT.parent/'RSJG_BCEP_Confirmation_20261116_223645/docs/research/bcep_v1/bcep_confirmation_20261116_223645'
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    with budget(root,'PREPARATION_CPU',7200):
        prior=read(old/'PLAN_AND_AUTHORIZATION.json');split=read(prior['split_lock']);assert digest(prior['split_lock'])==prior['split_sha256']
        cps=read(old/'CHECKPOINTS_FROZEN.json');outputs=read(old/'OUTPUTS_FROZEN.json');assert len(cps['runs'])==4 and len(outputs['outputs'])==20
        frozen={x['name']:x for x in cps['runs']}
        protected={x['path']:x['sha256'] for x in read(old/'PROTECTION_RECEIPT.json')['files']}
        for run in cps['runs']:
            for key in ['best','last']:protected[run[key]['path']]=run[key]['sha256']
        for row in outputs['outputs']:protected[row['path']]=row['sha256']
        for p in old.iterdir():
            if p.is_file():protected[str(p)]=digest(p)
        lock=read(ROOT/'docs/restoration/20261116_sixlayer_provider/RESTORED_BASELINE_LOCK.json')
        source=lock['folds']['eth']['source']['resolved_config'];assert digest(source['path'])==source['sha256'];c=read(source['path'])
        protected[source['path']]=source['sha256']
        for p,h in protected.items():assert digest(p)==h,p
        trees={name:dict(head=subprocess.check_output(['git','-C',str(ROOT.parent/name),'rev-parse','HEAD'],text=True).strip(),status=subprocess.check_output(['git','-C',str(ROOT.parent/name),'status','--porcelain'],text=True).strip()) for name in ['RSJG_EBJD_Paper','RSJG_EBJD_BCEP','RSJG_BCEP_Confirmation_20261116_223645']}
        write(root/'identity/PROTECTED_BEFORE.json',dict(files=[dict(path=p,sha256=h) for p,h in protected.items()],worktrees=trees))
        assert (len(split['train_indices']),len(split['selection_dev_indices']),len(split['heldout_confirmation_indices']))==(2792,320,998)
        # Exactly the restored FP64 train-only RMS rule; not Provider scales.
        data=PurposeDataset(split,'scale');rep=EndpointBridgeDecomposition();gs=bs=0.;gn=bn=0
        for i in split['train_indices']:
            x=data.get(i,'train');base,_=cv_baseline(x['observed'][None]);z=rep._encode_unscaled(x['future'][None].double(),base.double())
            gs+=float(z[...,11:,:].square().sum());gn+=z[...,11:,:].numel();bs+=float(z[...,:11,:].square().sum());bn+=z[...,:11,:].numel()
        scales=dict(goal=max((gs/gn)**.5,.5),bridge=max((bs/bn)**.5,.1),source='2792_train_only_restored_RMS',provider_scales_unchanged=True)
        t=c['training'];cfg=dict(schema='bcep-sixlayer-random-downstream-v1',initialization_role='random_all_downstream',provider_origin='self_trained_flow',teacher_used=False,hidden_dim=192,heads=4,ff_dim=768,fjab_blocks=6,history_social_blocks=c['history_encoder']['social_blocks'],crop_width_m=c['map_encoder']['crop_width_m'],map_agent_chunk=c['map_encoder']['agent_chunk'],scales=scales,precision='fp32',proposal_eta=0,rollout=False,projection=False,batch_scenes=t['batch_scenes'],gradient_accumulation=t['gradient_accumulation'],weight_decay=t['weight_decay'],betas=t['optimizer_betas'],eps=t['optimizer_eps'],clip_norm=5.,base_lr=1e-4,warmup_epochs=5,scheduler_horizon=30,max_epochs=20,budget_seconds=7200,downstream_seed=3201,steps=20,worlds=20,allow_real_training=False,source_resolved=source)
        assert cfg['batch_scenes']==4 and cfg['gradient_accumulation']==4
        cfg['optimizer_updates_per_epoch']=int(np.ceil(np.ceil(2792/4)/4));cfg['max_updates']=20*cfg['optimizer_updates_per_epoch']
        write(root/'identity/RESOLVED_CONFIG.json',cfg);write(docs/'RESOLVED_CONFIG.json',cfg)
        plan=dict(schema='bcep-sixlayer-joint-plan-v1',assets=str(root),docs=str(docs),split_path=prior['split_lock'],split_file_sha256=prior['split_sha256'],split_hash=canonical_hash(split),source_manifest_sha256=split['source_sha256'],config_path=str(root/'identity/RESOLVED_CONFIG.json'),config_sha256=digest(root/'identity/RESOLVED_CONFIG.json'),allow_real_training=False,authorization='User explicitly requested executing BCEP_FrozenProvider_SixLayer_JointPilot_Package.zip and uploading report.',package_sha256=paths['package_sha256'],runs=[],budgets=dict(preparation_cpu=7200,gpu_precheck=600,training_per_run=7200,training_total=14400,final_eval=14400),history_exposure='ZARA02 already publicly observed in previous confirmation; mechanism pilot, not blind/unseen-location/external ETH test')
        data=PurposeDataset(split,'training');bank_receipts=[]
        for seed,epoch,update in [(3102,3,8376),(3103,5,13960)]:
            origin=frozen[f'bcep_flow_seed{seed}'];best=origin['best'];assert (best['epoch'],best['update'])==(epoch,update) and digest(best['path'])==best['sha256']
            state=torch.load(best['path'],map_location='cpu',weights_only=False);model=BCEP();model.load_state_dict(state['model'],strict=True);model.eval();model.requires_grad_(False);before=state_digest(model.state_dict())
            row=dict(name=f'flow{seed}_joint3201',provider_seed=seed,provider=best,downstream_seed=3201,banks={})
            for purpose in ['train','selection_dev','heldout_confirmation']:
                ids=split[purpose+'_indices'];members=[split['members'][i] for i in ids];nmax=max(len(m['agent_ids']) for m in members)
                for inference in (range(2035,2040) if purpose=='heldout_confirmation' else [2035]):
                    started=time.monotonic();arr=np.zeros((len(ids),20,nmax,2),np.float32);mask=np.zeros((len(ids),nmax),bool);conversion=0.;source_bank=None
                    if purpose=='heldout_confirmation':
                        oldbank=next(x for x in outputs['outputs'] if x['run']==origin['name'] and x['seed']==inference);assert digest(oldbank['path'])==oldbank['sha256']
                        with np.load(oldbank['path'],allow_pickle=False) as z:
                            meta=json.loads(str(z['metadata_json']));assert meta['members']==members and meta['checkpoint_sha256']==best['sha256']
                            original=z['kmeans20'];arr=original.astype(np.float32);mask=z['valid'].copy();conversion=float(np.max(np.abs(original-arr.astype(np.float64))));assert np.array_equal(arr,original.astype(np.float32))
                        source_bank=oldbank
                    else:
                        for j,i in enumerate(ids):
                            x=data.get(i,purpose,False);n=len(x['agent_ids']);valid=torch.ones(n,dtype=torch.bool)
                            atoms,_=model.sample_support(x['observed'],x['semantic_maps'],valid,x['scene_id'],x['agent_ids'],inference,256)
                            for ai,agent in enumerate(x['agent_ids']):
                                centers=kmeans_centers(atoms[:,ai].numpy(),20,local_seed('bcep_flow',x['scene_id'],agent,inference),n_init=5,max_iter=100,tolerance_m=1e-6)['centers'];conversion=max(conversion,float(np.max(np.abs(centers-centers.astype(np.float32).astype(np.float64)))));arr[j,:,ai]=centers
                            mask[j,:n]=True
                            if j%100==0:write(root/'PREPARATION_PROGRESS.json',dict(provider_seed=seed,purpose=purpose,window=j+1,total=len(ids),truth_used=False))
                    meta=dict(schema='bcep-joint-candidates-v1',provider_kind='bcep_flow',provider_training_seed=seed,provider_sha256=best['sha256'],split_hash=canonical_hash(split),purpose=purpose,members=members,member_hash=canonical_hash(members),seed=inference,M=256,K=20,unit='m',dtype='float32',conversion='float64 KMeans centers to float32 exactly once; order unchanged',max_conversion_abs_m=conversion,selector=dict(n_init=5,max_iter=100,tolerance_m=1e-6,source_sha256=digest(ROOT/'selector_control/algorithms.py'),local_seed_source_sha256=digest(ROOT/'selector_control/core.py')),contains_truth=False,source_bank=source_bank,training_predictions_in_sample=purpose=='train')
                    path=root/'banks'/f'flow{seed}_{purpose}_{inference}.npz';tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,candidates=arr,valid=mask,metadata_json=json.dumps(meta));tmp.replace(path)
                    receipt=dict(path=str(path),sha256=digest(path),purpose=purpose,seed=inference,provider_seed=seed,windows=len(ids),agents=int(mask.sum()),seconds=time.monotonic()-started,max_conversion_abs_m=conversion,source_bank=source_bank)
                    row['banks'][f'{purpose}_{inference}']=receipt;bank_receipts.append(receipt);write(root/'reports/BANK_ADAPTER_RECEIPT.json',bank_receipts)
                    print(f'BANK flow{seed} {purpose} seed{inference}: {len(ids)} windows / {mask.sum()} agents, {receipt["seconds"]:.1f}s',flush=True)
            assert state_digest(model.state_dict())==before;plan['runs'].append(row);del model,state
        write(root/'RUN_PLAN.json',plan);write(docs/'RUN_PLAN.json',plan);write(docs/'BANK_ADAPTER_RECEIPT.json',bank_receipts)
        write(root/'PREPARATION_PROGRESS.json',dict(status='COMPLETE',formal_optimizer_updates=0))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--paths',required=True);main(read(p.parse_args().paths))

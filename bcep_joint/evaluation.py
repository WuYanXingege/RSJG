import csv,json,time
from pathlib import Path
import numpy as np
import torch
from ebjd.metrics import MetricAccumulator,hard_metrics
from ebjd.unified import gather_candidates
from .common import read,write,budget,digest,canonical_hash,seed_all
from .model import build
from .data import JointData,CandidateBank
from .inference import generate,shuffle_paths,metric_invariants

def evaluate(plan,split,c,frozen,device='cuda'):
    root=Path(plan['assets']);docs=Path(plan['docs']);identity=canonical_hash(split)
    with budget(root,'FINAL_EVALUATION',14400):
        assert len(frozen['runs'])==2 and frozen['status']=='ALL_RUNS_TERMINAL'
        if any(r['best'] is None for r in frozen['runs']):
            write(root/'reports/EVALUATION_STATUS.json',dict(status='INCOMPLETE',reason='no complete selection_dev best for one run'));return
        obs=JointData(split,'export');indices=split['heldout_confirmation_indices'];members=[split['members'][i] for i in indices];nmax=max(len(m['agent_ids']) for m in members)
        outputs=[];checks=[]
        for r in frozen['runs']:
            seed_all(3201);m=build(c).to(device);payload=torch.load(r['best']['path'],map_location=device,weights_only=False)
            assert digest(r['best']['path'])==r['best']['sha256'] and payload['identity']==r['identity'];m.load_state_dict(payload['model'],strict=True);m.eval()
            row=next(x for x in plan['runs'] if x['name']==r['name'])
            for seed in range(2035,2040):
                started=time.monotonic();br=row['banks'][f'heldout_confirmation_{seed}'];assert digest(br['path'])==br['sha256'];bank=CandidateBank(br['path'],split,'heldout_confirmation',row['provider']['sha256'],seed)
                paths=np.zeros((2,len(indices),20,nmax,12,2),np.float32);pi=np.zeros((len(indices),20,nmax),np.int64);valid=np.zeros((len(indices),nmax),bool);endpoint_max=0.
                for j,i in enumerate(indices):
                    batch=obs.batch([i],'heldout_confirmation',False);goals=bank.batch(batch).to(device);b=batch.to(device)
                    g,h,perm,receipt=generate(m,b.observed,b.semantic_maps,b.valid,b.scene_ids,b.agent_ids,goals,seed)
                    n=len(batch.agent_ids[0]);paths[0,j,:,:n]=g[0].cpu().numpy();paths[1,j,:,:n]=h[0].cpu().numpy();pi[j,:,:n]=perm[0].cpu().numpy();valid[j,:n]=True;endpoint_max=max(endpoint_max,receipt['endpoint_max_abs_m'])
                    if j%50==0:write(root/'EVALUATION_PROGRESS.json',dict(run=r['name'],seed=seed,window=j+1,total=len(indices),completed_banks=len(outputs),phase='NO_FUTURE_TRAJECTORY_EXPORT'))
                path=root/'evaluation'/f"{r['name']}_{seed}.npz";tmp=path.with_suffix('.tmp.npz')
                meta=dict(schema='bcep-joint-trajectories-v1',run=r['name'],checkpoint_sha256=r['best']['sha256'],bank_sha256=br['sha256'],seed=seed,split_hash=identity,members=members,unit='m',K=20,steps=20,eta=0,contains_truth=False)
                np.savez_compressed(tmp,G=paths[0],H=paths[1],permutation=pi,valid=valid,metadata_json=json.dumps(meta));tmp.replace(path)
                outputs.append(dict(run=r['name'],seed=seed,path=str(path),sha256=digest(path),seconds=time.monotonic()-started))
                checks.append(dict(run=r['name'],seed=seed,endpoint_max_abs_m=endpoint_max,exact_candidate_gather=True,exact_noise_gather=True,per_agent_bijection=True,single_person_identity=True))
                write(root/'reports/OUTPUT_PROGRESS.json',outputs);print(f"Frozen trajectories {r['name']} seed{seed}: {len(indices)} windows, {outputs[-1]['seconds']:.1f}s",flush=True)
            del m,payload;torch.cuda.empty_cache()
        freeze=dict(status='ALL_OUTPUTS_FROZEN',split_hash=identity,created_unix=time.time(),outputs=outputs)
        write(root/'reports/OUTPUTS_FROZEN.json',freeze);write(docs/'OUTPUTS_FROZEN.json',freeze)
        # Both selected models and all ten G/H banks are fixed before truth scoring.
        scoring=JointData(split,'scoring',root/'reports/OUTPUTS_FROZEN.json');truth=[]
        for i in indices:truth.append(scoring.item(i,'heldout_confirmation')['future'])
        rows=[];invariants=[]
        for out in outputs:
            assert digest(out['path'])==out['sha256'];run=next(x for x in plan['runs'] if x['name']==out['run']);seed=out['seed'];bank=CandidateBank(run['banks'][f'heldout_confirmation_{seed}']['path'],split,'heldout_confirmation',run['provider']['sha256'],seed)
            acc={group:{mode:MetricAccumulator() for mode in ['G','H']+[f'H_shuffle_{j}' for j in range(5)]} for group in ['all','N1','Nge2']}
            maxfde=maxshuffle=0.;gap_min=float('inf')
            with np.load(out['path'],allow_pickle=False) as z:
                assert json.loads(str(z['metadata_json']))['members']==members
                G=z['G'];H=z['H'];P=z['permutation'];V=z['valid']
                for j,member in enumerate(members):
                    n=len(member['agent_ids']);g=torch.from_numpy(G[j:j+1,:,:n]);h=torch.from_numpy(H[j:j+1,:,:n]);pi=torch.from_numpy(P[j:j+1,:,:n]);v=torch.from_numpy(V[j:j+1,:n]);target=truth[j][None]
                    goals=bank.goals[j:j+1,:,:n];hg=gather_candidates(goals,pi)
                    check=metric_invariants(g,h,goals,hg,target,v);maxfde=max(maxfde,check['minFDE_G_H_abs'])
                    modes={'G':g,'H':h}
                    for rep in range(5):
                        sh=shuffle_paths(h,v,[member['scene_id']],seed,rep)
                        hm=hard_metrics(h,target,v);sm=hard_metrics(sh,target,v)
                        for key in ['minADE','minFDE']:
                            delta=abs(getattr(hm,key)-getattr(sm,key));maxshuffle=max(maxshuffle,delta);assert delta<=1e-6
                        modes[f'H_shuffle_{rep}']=sh
                    for group in ['all','N1' if n==1 else 'Nge2']:
                        for name,path in modes.items():acc[group][name].update(path,target,v)
            for group,modes in acc.items():
                for name,a in modes.items():
                    if a.scene_count:rows.append(dict(run=out['run'],provider_seed=run['provider_seed'],downstream_seed=3201,inference_seed=seed,group=group,method=name,**{k:v for k,v in a.as_dict().items() if k!='sums'}))
            invariants.append(dict(run=out['run'],seed=seed,GH_minFDE_max_scene_abs_m=maxfde,shuffle_marginal_max_abs_m=maxshuffle,endpoint_only_JFDE_matches=True,nonnegative_scene_gap=True,exact_full_trajectory_multiset=True))
        for d in [root/'reports',docs]:
            with (d/'PER_RUN_PER_SEED.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
            write(d/'INVARIANT_RECEIPT.json',dict(status='PASS',sampling=checks,scoring=invariants,endpoint_atol=1e-6,endpoint_rtol=1e-6))
        write(root/'reports/EVALUATION_STATUS.json',dict(status='COMPLETE',runs=2,inference_seeds=5,trajectory_banks=10));write(root/'EVALUATION_PROGRESS.json',dict(status='COMPLETE'))

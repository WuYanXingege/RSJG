"""CPU-only P4 numerical counterexamples and P5 read-only noise cacheability."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
SOURCE=Path(__file__).resolve().parents[2];sys.path.insert(0,str(SOURCE))
OUT=SOURCE.parent/'RSJG_perf_outputs/bcep_v3_speed_audit_20261120'
import torch
from bcep_v3.coupling import HCRACoupling
from bcep_v3.geometry import reference_paths,pair_features
from bcep_v2.joint import neighbors,pair_loss
from ebjd.representation import cv_baseline
from ebjd.unified import hard_assignment_indices,gather_candidates
from bcep_joint.inference import keyed_noise


def write(name,value):(OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def bytehash(t):return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def simplified(context,goals):
    ratio=torch.arange(1,13,device=goals.device,dtype=torch.float32)/12
    last=context.observed[:,:,-1].float()
    return last[:,:,None,None]+ratio[None,None,None,:,None]*(goals.transpose(1,2).float()-last[:,:,None])[:,:,:,None]


def geometry_audit():
    rows=[]
    for n in [1,2,9,57]:
        torch.manual_seed(101+n)
        obs=torch.randn(1,n,8,2)*.2+torch.arange(n)[None,:,None,None]*.4
        base,velocity=cv_baseline(obs)
        c=SimpleNamespace(observed=obs,baseline=base,mean_velocity=velocity,valid=torch.ones(1,n,dtype=torch.bool),social_h=torch.randn(1,n,192))
        g=torch.randn(1,20,n,2);truth=torch.randn(1,n,2)
        model=HCRACoupling();model.enable_v3()
        with torch.no_grad():
            model.compatibility[-1].weight.normal_(std=.01);model.csee.out.weight.normal_(std=.01)
            model.hcra.delta[-1].weight.normal_(std=.02);model.world_queries.mul_(10)
        scale=torch.tensor(1.);pa=reference_paths(c,g);pb=simplified(c,g)
        b,i,j=neighbors(obs,c.valid);feature_errors=torch.zeros(12);time_changed=0
        for start in range(0,len(b),64):
            sl=slice(start,start+64)
            fa=pair_features(pa,obs,b[sl],i[sl],j[sl],scale);fb=pair_features(pb,obs,b[sl],i[sl],j[sl],scale)
            feature_errors=torch.maximum(feature_errors,(fa-fb).abs().flatten(0,2).amax(0))
            time_changed+=int((fa[...,1]!=fb[...,1]).sum())
        a=model.assign(c,g,scale,straight_through=True,return_details=True)
        with patch('bcep_v3.coupling.reference_paths',simplified):
            z=model.assign(c,g,scale,straight_through=True,return_details=True)
        parameters=list(model.parameters())
        la=pair_loss(a[0],truth,c.valid)[0]+.001*a[2].square().mean()
        lb=pair_loss(z[0],truth,c.valid)[0]+.001*z[2].square().mean()
        ga=torch.autograd.grad(la,parameters,allow_unused=True)
        gb=torch.autograd.grad(lb,parameters,allow_unused=True)
        abs_error=rel_error=0.;grad_exact=True
        for x,y in zip(ga,gb):
            if x is None or y is None:
                assert x is None and y is None;continue
            grad_exact &= torch.equal(x,y)
            abs_error=max(abs_error,float((x-y).abs().max()))
            rel_error=max(rel_error,float(((x-y).abs()/torch.maximum(x.abs(),y.abs()).clamp_min(1e-8)).max()))
        for out in [a,z]:
            assert torch.equal(out[0],gather_candidates(g,out[1]))
            assert torch.equal(out[1].sort(1).values,torch.arange(20)[None,:,None].expand_as(out[1]))
        ea=(a[0]-truth[:,None]).norm(dim=-1).min(1).values
        eb=(z[0]-truth[:,None]).norm(dim=-1).min(1).values
        assert torch.equal(ea,eb)
        noise=keyed_noise(['SYNTHETIC'],[[str(k) for k in range(n)]],c.valid,2035)
        assert torch.equal(gather_candidates(gather_candidates(noise,a[1]),a[1].argsort(1)),noise)
        rows.append(dict(n_agents=n,path_max_abs=float((pa-pb).abs().max()),
            relation_feature_max_abs=feature_errors.tolist(),argmin_time_changed_elements=time_changed,
            scores_max_abs=float((a[2]-z[2]).detach().abs().max()),
            initial_permutation_equal=torch.equal(a[3]['initial_pi'],z[3]['initial_pi']),
            final_permutation_equal=torch.equal(a[1],z[1]),final_changed_elements=int((a[1]!=z[1]).sum()),
            gradient_max_abs=abs_error,gradient_max_relative_floor1e_minus8=rel_error,gradient_bitwise=grad_exact,
            hard_endpoint_gather_exact=True,keyed_noise_inverse_gather_exact=True,marginal_minFDE_equal=True))
    # Negative control: a tiny score perturbation can change a legal hard optimum.
    score=torch.zeros(1,2,20,20);pert=score.clone();pert[:,:,0,1]=1e-8;pert[:,:,1,0]=1e-8
    valid=torch.ones(1,2,dtype=torch.bool);p=hard_assignment_indices(score,valid);q=hard_assignment_indices(pert,valid)
    assert torch.allclose(score,pert,atol=1e-6,rtol=1e-6) and not torch.equal(p,q)
    write('GEOMETRY.json',dict(status='NON_BITWISE_OPTIONAL',production_implementation_changed=False,
        geometry_algebra_only_in_diagnostic=True,optimizer_updates=0,rows=rows,
        near_tie_negative_control=dict(score_max_abs=1e-8,allclose_at_1e_minus6=True,permutation_changed=True,
                                       note='Constructed score control, not a claim that this same perturbation occurs in every geometry input.')))


def noise_audit():
    root=SOURCE.parent/'RSJG_EBJD_assets/experiments/bcep_ebjd_v2_integrated'
    run=json.loads((root/'FORMAL_RUN.json').read_text());plan=json.loads(Path(run['input_plan']).read_text())
    desc=plan['folds']['zara2']['split'];raw=Path(desc['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==desc['sha256'];split=json.loads(raw)
    members=[split['members'][i] for i in split['selection_dev_indices']]
    assert len(members)==590 and sum(len(m['agent_ids'])>=9 for m in members)==127
    def key(m):
        return ('joint-pilot-bridge-noise',torch.__version__,str(m['scene_id']),tuple(m['agent_ids']),tuple(range(20)),2035,
                tuple(True for _ in m['agent_ids']),tuple(m['frame_ids']),'world_xy_m_dt0.4_obs8_pred12')
    def native(m):return keyed_noise([m['scene_id']],[m['agent_ids']],torch.ones(1,len(m['agent_ids']),dtype=torch.bool),2035)
    cache={};rng=torch.get_rng_state();elapsed=[];hashes=[]
    for repeat in range(3):
        start=time.perf_counter();outputs=[native(m) for m in members];elapsed.append(time.perf_counter()-start)
        digest=hashlib.sha256()
        for m,z in zip(members,outputs):
            digest.update(z.numpy().tobytes())
            if repeat==0:cache[key(m)]=z
            else:assert z.numpy().tobytes()==cache[key(m)].numpy().tobytes()
        hashes.append(digest.hexdigest())
    cached=[]
    for _ in range(3):
        start=time.perf_counter();copies=[cache[key(m)].clone() for m in members];cached.append(time.perf_counter()-start)
        assert all(x.numpy().tobytes()==cache[key(m)].numpy().tobytes() for m,x in zip(members,copies))
    assert torch.equal(rng,torch.get_rng_state()) and len(set(hashes))==1
    # Frame/coordinate identity must be part of an eventual key even though the
    # underlying native generator hashes scene/agent/world/seed only.
    m=dict(members[0]);old=key(m);m['frame_ids']=list(m['frame_ids']);m['frame_ids'][0]-=1;assert key(m)!=old
    write('CPU_NOISE.json',dict(status='READ_ONLY_CPU_MEASURED',windows=590,agents=sum(len(m['agent_ids']) for m in members),
        K=20,seed=2035,canonical_split_file_sha256=desc['sha256'],native_seconds=elapsed,cached_clone_seconds=cached,
        repetitions=3,exploratory=True,bytes_equal=True,global_cpu_rng_unchanged=True,tensor_cache_bytes=sum(z.numel()*z.element_size() for z in cache.values()),
        concatenated_noise_sha256=hashes[0],raw_member_ids_exported=False,optimizer_updates=0,
        cache_key_fields=['algorithm','torch_version','scene_id','ordered_agent_ids','world_0_to_19','seed','valid_mask_and_order','frame_ids','coordinate_identity'],
        production_cache_enabled=False,note='Read-only dev identity metadata and CPU random draws only; no model, trajectories, labels, or test/heldout reads. Three CPU timings are exploratory, not a GPU or full-validation speedup.'))


if __name__=='__main__':
    torch.set_num_threads(1);assert not torch.cuda.is_available() and not torch.cuda.is_initialized()
    OUT.mkdir(parents=True,exist_ok=True);geometry_audit();noise_audit()
    print('P4 geometry and P5 CPU noise audits complete; no optimizer steps or CUDA context.')

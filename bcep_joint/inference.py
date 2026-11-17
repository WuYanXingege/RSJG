"""Observed-only keyed G/H inference and exact candidate/noise identities."""
import torch
from ebjd.unified import gather_candidates
from ebjd.sampling import differentiable_sample
from ebjd.metrics import hard_metrics,hard_metric_statistics
from .common import local_seed

def keyed_noise(scene_ids,agent_ids,valid,seed):
    z=torch.zeros(len(scene_ids),20,valid.shape[1],12,2)
    for b,scene in enumerate(scene_ids):
        pos=valid[b].cpu().nonzero().flatten().tolist()
        assert len(pos)==len(agent_ids[b])
        for i,agent in zip(pos,agent_ids[b]):
            for k in range(20):
                gen=torch.Generator().manual_seed(local_seed('joint-pilot-bridge-noise',scene,agent,k,seed))
                z[b,k,i]=torch.randn(12,2,generator=gen)
    return z

def check_permutation(pi,valid):
    assert pi.shape==(len(valid),20,valid.shape[1])
    assert torch.equal(pi.sort(1).values,torch.arange(20,device=pi.device)[None,:,None].expand_as(pi))
    for b in range(len(valid)):
        if valid[b].sum()==1:assert torch.equal(pi[b,:,valid[b]],torch.arange(20,device=pi.device)[:,None])

@torch.no_grad()
def generate(model,observed,maps,valid,scene_ids,agent_ids,goals,seed):
    model.eval();device=observed.device
    assert goals.shape==(len(observed),20,observed.shape[1],2)
    context=model.encode_context(observed,maps,valid)
    h,pi,scores=model.coupling.assign(context,goals,model.bridge.representation.goal_scale,straight_through=False)
    check_permutation(pi,valid);assert torch.equal(h,gather_candidates(goals,pi))
    z=keyed_noise(scene_ids,agent_ids,valid,seed).to(device);zh=gather_candidates(z,pi)
    assert torch.equal(zh,gather_candidates(z,pi))
    g,ge,_=differentiable_sample(model.bridge,context,z,20,proposal_goals=goals,proposal_eta=0.)
    hpath,he,_=differentiable_sample(model.bridge,context,zh,20,proposal_goals=h,proposal_eta=0.)
    assert torch.isfinite(g).all() and torch.isfinite(hpath).all()
    endpoint_max=max(float((g[...,-1,:]-goals).abs().max()),float((hpath[...,-1,:]-h).abs().max()))
    torch.testing.assert_close(g[...,-1,:],goals,atol=1e-6,rtol=1e-6);torch.testing.assert_close(hpath[...,-1,:],h,atol=1e-6,rtol=1e-6)
    for b in range(len(valid)):
        if valid[b].sum()==1:torch.testing.assert_close(g[b,:,valid[b]],hpath[b,:,valid[b]],rtol=1e-6,atol=1e-6)
    return g,hpath,pi,dict(endpoint_max_abs_m=endpoint_max,exact_candidate_gather=True,exact_noise_gather=True,per_agent_bijection=True)

def shuffle_paths(paths,valid,scene_ids,seed,repetition):
    pi=torch.arange(20,device=paths.device)[None,:,None].expand(paths.shape[:3]).clone()
    for b,scene in enumerate(scene_ids):
        pos=valid[b].nonzero().flatten().tolist()
        for i in pos[1:]:
            gen=torch.Generator().manual_seed(local_seed('offline-joint-destruction',scene,seed,repetition,i))
            pi[b,:,i]=torch.randperm(20,generator=gen).to(paths.device)
    check_permutation(pi,valid);out=gather_candidates(paths,pi)
    # Explicit inverse proves preservation of full trajectory vectors, not coordinate sorting.
    assert torch.equal(gather_candidates(out,pi.argsort(1)),paths)
    return out

def metric_invariants(g,h,goals,hgoals,truth,valid):
    gm=hard_metrics(g,truth,valid);hm=hard_metrics(h,truth,valid)
    err=torch.linalg.vector_norm(goals.float()-truth[:,None,:,-1,:].float(),dim=-1)
    ferr=torch.linalg.vector_norm(hgoals.float()-truth[:,None,:,-1,:].float(),dim=-1)
    torch.testing.assert_close(err.min(1).values[valid],ferr.min(1).values[valid],atol=1e-6,rtol=1e-6)
    endpoint_values=[]
    for path,goal,m in [(g,goals,gm),(h,hgoals,hm)]:
        s=hard_metric_statistics(path,truth,valid)
        e=torch.linalg.vector_norm(goal.float()-truth[:,None,:,-1,:].float(),dim=-1)
        joint=((e*valid[:,None]).sum(-1)/valid.sum(-1)[:,None]).min(-1).values
        torch.testing.assert_close(joint,s.scene_jfde,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(e.min(1).values[valid],s.agent_minfde[valid],atol=1e-6,rtol=1e-6)
        assert ((joint-s.scene_marginal_fde)>=-1e-6).all()
        endpoint_values.append(float(joint.double().mean()))
    return dict(endpoint_only_G_JFDE=endpoint_values[0],endpoint_only_H_JFDE=endpoint_values[1],minFDE_G_H_abs=abs(gm.minFDE-hm.minFDE),endpoint_only_JFDE_matches=True,scene_gap_nonnegative=True)

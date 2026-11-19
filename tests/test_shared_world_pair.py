"""Shared-world objective: CPU engineering fixtures, no real training data."""
import math
import subprocess
import types
import torch
import pytest
from bcep_joint.common import seed_all, read
from bcep_joint.checks import synthetic_batch, equal
from bcep_joint.inference import generate, keyed_noise
from bcep_v2.joint import build_joint, pair_loss, shared_world_terms, endpoint_world_error, FeedbackCoupling
from bcep_v2.objective import bridge_and_pair_loss
from bcep_v2.state import run_stage
from ebjd.unified import gather_candidates
from tests.test_bcep_integrated_v2 import config

BASE = '468c5eddde48bdeec47c4bb3c39418352be16bb9'

def original(name):
    m = types.ModuleType('bcep_v2._reference_' + name)
    m.__package__ = 'bcep_v2'
    exec(compile(subprocess.check_output(['git', 'show', BASE + ':bcep_v2/' + name + '.py']), '<original>', 'exec'), m.__dict__)
    return m

def test_analytic_max_envelope_boundaries_and_shared_worlds():
    I = torch.tensor([[.2,.4,.6,.7,.1]], dtype=torch.double, requires_grad=True)
    H = torch.tensor([[.4,.4,.55,.2,.6]], dtype=torch.double, requires_grad=True)
    C,a = shared_world_terms(I,H)
    torch.testing.assert_close(a, torch.tensor([[0,0,.25,.5,0]], dtype=torch.double))
    grid = torch.linspace(0,.5,10001,dtype=torch.double)
    direct = ((1-grid)*H[...,None]+grid*I[...,None]-.1*grid.square()).max(-1).values
    torch.testing.assert_close(C,direct,atol=1e-9,rtol=0)
    q=(-C/.1).softmax(1); loss=-.1*(torch.logsumexp(-C/.1,1)-math.log(5)).mean()
    di,dh=torch.autograd.grad(loss,(I,H));torch.testing.assert_close(di,q*a);torch.testing.assert_close(dh,q*(1-a))
    for which in range(2):
        for k in range(5):
            vals=[]
            for sign in [-1,1]:
                x,y=I.detach().clone(),H.detach().clone();[x,y][which][0,k]+=sign*1e-6
                c,_=shared_world_terms(x,y);vals.append(-.1*(torch.logsumexp(-c/.1,1)-math.log(5)))
            assert abs(float((vals[1]-vals[0])/2e-6)-float([di,dh][which][0,k]))<3e-6
    p=torch.tensor([4,0,2,1,3]);cp,_=shared_world_terms(I[:,p],H[:,p]);torch.testing.assert_close(cp,C[:,p])
    # Different winning worlds cannot be minimized independently before coupling.
    i=torch.tensor([[0.,1.]]);h=torch.tensor([[1.,0.]])
    c,_=shared_world_terms(i,h); assert float(c.min()) > 0 and float(i.min())==float(h.min())==0


def test_same_initialization_legacy_forward_gradients_and_inference():
    seed_all(3201);old=original('joint').build_joint(config())
    seed_all(3201);new=build_joint(dict(config(),pair_objective='shared_world_bounded'))
    new.load_state_dict(old.state_dict(),strict=True)
    equal(old.state_dict(),new.state_dict()); assert len(list(old.parameters()))==len(list(new.parameters()))
    x=synthetic_batch(2,pixels=32);g=torch.randn(1,20,2,2);noise=torch.randn(1,1,2,12,2);times=torch.tensor([.4])
    before=original('objective').bridge_and_pair_loss(old,x,g,time_draw=times,noise=noise)
    after=bridge_and_pair_loss(new,x,g,time_draw=times,noise=noise)
    for a,b in zip(before,after):torch.testing.assert_close(a,b,atol=0,rtol=0)
    before[0].backward();after[0].backward()
    for (n,p),(nn,q) in zip(old.named_parameters(),new.named_parameters()):
        assert n==nn
        if p.grad is None:assert q.grad is None
        else:torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-5)
    assert new.last_pair_metrics['pair_a_mean']==0
    with torch.no_grad():
        oldout=generate(old,x.observed,x.semantic_maps,x.valid,x.scene_ids,x.agent_ids,g,2035)
        x.future.fill_(float('nan'))
        newout=generate(new,x.observed,x.semantic_maps,x.valid,x.scene_ids,x.agent_ids,g,2035)
        for a,b in zip(oldout[:3],newout[:3]):torch.testing.assert_close(a,b,atol=0,rtol=0)
    # Absent option defaults to old loss and strict loads old checkpoints.
    default=build_joint(config());default.load_state_dict(old.state_dict(),strict=True)
    assert default.pair_objective=='final_coordinate'
    with pytest.raises(ValueError):build_joint(dict(config(),pair_objective='unknown'))


def test_branch_paths_padding_singleton_and_candidate_noise_identity():
    seed_all(18);m=FeedbackCoupling();m.compatibility[-1].weight.data.normal_(0,.01);m.world_queries.data.normal_(0,1.)
    valid=torch.tensor([[1,1,0],[1,0,0]],dtype=torch.bool)
    ctx=types.SimpleNamespace(valid=valid,observed=torch.randn(2,3,8,2),social_h=torch.randn(2,3,192,requires_grad=True))
    g=torch.randn(2,20,3,2,requires_grad=True);g.data.masked_fill_(~valid[:,None,:,None],float('nan'))
    truth=torch.randn(2,3,2);truth.masked_fill_(~valid[...,None],float('nan'))
    saved={};msg=m.messages
    def tap(*args):
        saved['initial_matrix']=args[-1];return msg(*args)
    m.messages=tap
    H,pi,_,I,p0=m.assign(ctx,g,torch.tensor(1.),straight_through=True,return_initial=True)
    ie=endpoint_world_error(I,truth,valid);he=endpoint_world_error(H,truth,valid)
    C,a=shared_world_terms(ie,he);q=(-C.detach()/.1).softmax(1)
    gi=torch.autograd.grad((q*a*ie).sum(),[m.score[-1].weight,m.compatibility[-1].weight,ctx.social_h],retain_graph=True,allow_unused=True)
    assert gi[1] is None and gi[0].abs().sum()>0 and gi[2].abs().sum()>0
    gh=torch.autograd.grad((q*(1-a)*he).sum(),[m.score[-1].weight,m.compatibility[-1].weight,saved['initial_matrix']],retain_graph=True)
    assert all(z.isfinite().all() and z.abs().sum()>0 for z in gh)
    loss,_=pair_loss(H,truth,valid,initial=I);loss.backward();assert g.grad is None
    assert a[1].count_nonzero()==0
    assert torch.equal(I[1,:,0],H[1,:,0]);assert all(p.grad is None or p.grad.isfinite().all() for p in m.parameters())
    # Candidate relabeling moves candidate-keyed noise with its candidate identity.
    ctx2=types.SimpleNamespace(valid=valid[:1,:2],observed=ctx.observed[:1,:2],social_h=ctx.social_h[:1,:2].detach())
    raw=g.detach()[:1,:,:2];relabel=torch.stack([torch.randperm(20),torch.randperm(20)],-1)[None]
    shuffled=gather_candidates(raw,relabel)
    with torch.no_grad():
        h,p,_,i,p0=m.assign(ctx2,raw,torch.tensor(1.),straight_through=False,return_initial=True)
        hh,pp,_,ii,pp0=m.assign(ctx2,shuffled,torch.tensor(1.),straight_through=False,return_initial=True)
    assert torch.equal(h,hh) and torch.equal(i,ii)
    z=keyed_noise(['scene'],[['a','b']],valid[:1,:2],2035)
    assert torch.equal(gather_candidates(z,p),gather_candidates(gather_candidates(z,relabel),pp))
    assert torch.equal(gather_candidates(z,p0),gather_candidates(gather_candidates(z,relabel),pp0))
    # Constant/tied scores stay deterministic; padding and singletons use identity.
    from bcep_v2.joint import assignment
    ties=torch.zeros(2,3,20,20)
    for _ in range(2):assert torch.equal(assignment(ties,valid,True)[0],torch.arange(20)[None,:,None].expand(2,20,3))


def test_grouped_backward_and_accumulation_denominators(monkeypatch):
    from ebjd.data import SyntheticSceneDataset,collate_scenes
    from bcep_v2 import stages
    seed_all(92);dataset=SyntheticSceneDataset(scenes=4,agents=3,pixels=32)
    class Data:
        def batch(self,ids,purpose):return collate_scenes([dataset[i] for i in ids])
    class Bank:
        def batch(self,ids,b):return b.observed[:,None,:,-1]+torch.linspace(-1,1,20)[None,:,None,None]
    # Keep normal grouping/accumulation/backward; isolate random augmentation in this fixture.
    monkeypatch.setattr(stages,'augment',lambda x,g,r:(x,g,None))
    c=dict(config(),pair_objective='shared_world_bounded',batch_scenes=2,gradient_accumulation=2)
    seed_all(22);model=build_joint(c);model.coupling.compatibility[-1].weight.data.normal_(0,.01)
    collected=[];direct_grads=[];checking=False;actual=stages.bridge_and_pair_loss
    def capture(*args,**kwargs):
        v=actual(*args,**kwargs)
        n=args[1].valid.sum().item();sc=len(args[1].valid)
        if checking:
            j=len(collected);micro_start=(j//2)*2
            count=sum(len(dataset[k]['agent_ids']) for k in range(micro_start,micro_start+2))
            reference=(v[3]*n/count+v[4]*sc/2)/2
            direct_grads.append(torch.autograd.grad(reference,model.coupling.score[-1].weight,retain_graph=True)[0])
        collected.append((v,n,sc,dict(model.last_pair_metrics)));return v
    monkeypatch.setattr(stages,'bridge_and_pair_loss',capture)
    seed_all(88)
    backward,_=stages.joint_callbacks(model,Data(),'cpu',dict(c,group_max_slots=100),Bank(),Bank())
    broad=backward(list(range(4)),torch.Generator(),torch.Generator());grad={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    reference=sum(float(v[0].detach())/2 for v,_,_,_ in collected)
    assert broad['loss']==pytest.approx(reference,abs=1e-6)
    qa=sum(float(d['pair_q_a'])*n for _,_,n,d in collected)
    assert broad['pair_q_a_scene_sum']==pytest.approx(qa,abs=1e-7)
    assert broad['pair_q_a']==pytest.approx(qa/4,abs=1e-7)
    model.zero_grad(set_to_none=True);collected.clear();seed_all(88);checking=True
    backward,_=stages.joint_callbacks(model,Data(),'cpu',dict(c,group_max_slots=1),Bank(),Bank())
    narrow=backward(list(range(4)),torch.Generator(),torch.Generator())
    # Compare with a direct scalar baseline on the identical physical forwards.
    # Arbitrary regrouping can perturb Hungarian tie decisions in FP32; it is not a loss identity.
    reference = 0.
    raw_qa = 0.
    physical_rows = list(collected)
    for micro_start in range(0,4,2):
        count = sum(physical_rows[j][1] for j in range(micro_start,micro_start+2))
        for j in range(micro_start,micro_start+2):
            v,n,sc,d=physical_rows[j]
            reference += float((v[3]*n/count+v[4]*sc/2).detach())/2
            raw_qa += float(d['pair_q_a'])*sc
    assert narrow['loss']==pytest.approx(reference,abs=1e-6)
    assert narrow['pair_q_a_scene_sum']==pytest.approx(raw_qa,abs=1e-7)
    assert narrow['pair_q_a']==pytest.approx(raw_qa/4,abs=1e-7)
    assert narrow['valid_scene_count']==4
    torch.testing.assert_close(model.coupling.score[-1].weight.grad,sum(direct_grads),atol=1e-7,rtol=1e-6)
    assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())


def test_resume_top5_and_new_raw_statistics(tmp_path):
    # Exactly 24 synthetic optimizer updates (12 continuous + 5 interrupted + 7 resumed).
    rec=dict(seed=7,lr=.001,weight_decay=.0001,clip=5.,logical_scenes=1,max_epochs=6,minimum=6,patience=20,scheduler_horizon=6)
    ident=dict(stage='joint',pair_objective='shared_world_bounded',synthetic=True)
    def run(folder,limit,resume=False,identity=ident):
        seed_all(7);m=torch.nn.Linear(2,40)
        def backward(ids,order,views):
            z=m(torch.randn(1,2)).reshape(1,20,2);i=z[...,0].square();h=z[...,1].square();c,a=shared_world_terms(i,h)
            q=(-c.detach()/.1).softmax(1);loss=-.1*(torch.logsumexp(-c/.1,1)-math.log(20)).mean();loss.backward()
            return dict(loss=float(loss.detach()),valid_scene_count=1,pair_q_a_scene_sum=float((q*a).sum()))
        count=read(folder/'EPOCH_METRICS.json')[-1]['epoch'] if resume else 0
        def validate():
            nonlocal count
            v=[6,2,2,5,3,4][count];count+=1;return v
        return run_stage(folder,identity,rec,m,[0,1],backward,validate,limit,resume=resume)
    a=run(tmp_path/'a',12);p=run(tmp_path/'b',5);assert p['completed_epoch']==2 and p['cursor']==1
    b=run(tmp_path/'b',12,True)
    for key in ['model','optimizer','scheduler','rng','history','update','cursor','completed_epoch','epoch_raw_sums','epoch_counts']:equal(a[key],b[key])
    top=read(tmp_path/'b/TOP5.json');assert [r['epoch'] for r in top]==[2,3,5,6,4]
    from bcep.contracts import digest
    for r in top:
        path=tmp_path/'b'/r['file'];assert digest(path)==r['sha256'];s=torch.load(path,weights_only=False)
        assert all(k in s for k in ['optimizer','scheduler','rng','order','cursor','stale'])
    assert all('train_pair_q_a_scene_macro' in r for r in b['history'])
    with pytest.raises(ValueError,match='identity'):run(tmp_path/'b',12,True,dict(ident,pair_objective='final_coordinate'))

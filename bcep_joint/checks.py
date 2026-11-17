import copy,inspect,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from ebjd.data import SceneBatch,augment_batch_with_points
from ebjd.unified import gather_candidates,hard_assignment_indices
from ebjd.representation import EndpointBridgeDecomposition,cv_baseline
from .common import read,write,budget,seed_all,state_digest,forbid_weight_reads,source_identity,canonical_hash
from .model import build,optimizer_for
from .data import CandidateBank,JointData
from .inference import generate,shuffle_paths,metric_invariants,keyed_noise
from . import runner

def equal(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b,(a,b)

def synthetic_batch(n=3,pixels=256):
    gen=torch.Generator().manual_seed(8401);obs=torch.randn(1,n,8,2,generator=gen)*.3;obs+=torch.arange(n)[None,:,None,None]*1.2
    future=obs[:,:,-1:, :]+torch.randn(1,n,12,2,generator=gen)*.2
    maps=torch.zeros(1,n,6,pixels,pixels);maps[:,:,1]=1
    return SceneBatch(obs,future,maps,torch.ones(1,n,dtype=torch.bool),['synthetic'],[[str(i) for i in range(n)]],[list(range(20))],[[i*.4 for i in range(20)]],['synthetic'],{})

def recovery_test(root):
    class Tiny(nn.Module):
        def __init__(self,c):
            super().__init__();self.bridge=nn.Module();self.bridge.encoder=nn.Module();self.bridge.encoder.map_unet=nn.Linear(2,4);self.bridge.denoiser=nn.Linear(4,2);self.coupling=nn.Linear(4,2)
    class Data:
        def batch(self,ids,purpose):
            b=synthetic_batch(2,8)
            from ebjd.data import collate_scenes
            return collate_scenes([dict(observed=b.observed[0]+i*.01,future=b.future[0]+i*.01,semantic_maps=b.semantic_maps[0],metadata={},scene_id=str(i),agent_ids=['0','1'],frame_ids=list(range(20)),timestamps=[j*.4 for j in range(20)],source_sequence='synthetic') for i in ids])
    class Bank:
        def batch(self,b):return b.observed[:,None,:,-1,:].expand(-1,20,-1,-1).clone()+torch.arange(20)[None,:,None,None]*.01
    def logical(m,opt,b,g,gen,device,acc):
        b,g=augment_batch_with_points(b,g,gen);h=m.bridge.encoder.map_unet(b.observed[:,:,-1]);pred=m.bridge.denoiser(h)+m.coupling(h)
        loss=(pred-b.future[:,:,-1]+torch.randn_like(pred)*.01).square().mean();(loss/acc).backward();return [float(loss.detach())]*3
    def fake_generate(m,obs,maps,valid,sids,aids,goals,seed):
        h=m.bridge.encoder.map_unet(obs[:,:,-1]);delta=(m.bridge.denoiser(h)+m.coupling(h))[:,None,:,None,:]
        p=goals[:,:,:,None,:].expand(-1,-1,-1,12,-1)+delta
        return p.detach(),p.detach(),None,{}
    config=dict(downstream_seed=3201,budget_seconds=1000,batch_scenes=2,gradient_accumulation=4,max_epochs=2,weight_decay=.0001,betas=[.9,.999],eps=1e-8)
    split=dict(train_indices=list(range(7)),selection_dev_indices=[7,8]);old=runner.train_logical,runner.generate
    try:
        runner.train_logical=logical;runner.generate=fake_generate
        a=runner.run(config,split,Data(),Bank(),Bank(),{'test':1},root/'continuous',device='cpu',model_factory=Tiny)
        partial=runner.run(config,split,Data(),Bank(),Bank(),{'test':1},root/'resumed',device='cpu',model_factory=Tiny,stop_after_microbatches=3)
        assert partial['pending']==3 and partial['update']==0 and partial['pending_gradients']
        b=runner.run(config,split,Data(),Bank(),Bank(),{'test':1},root/'resumed',device='cpu',model_factory=Tiny,resume=True)
        for key in ['model','optimizer','scheduler','rng','pending_gradients','update','microbatch','epoch','completed_epoch','cursor','pending','order']:equal(a[key],b[key])
        equal(a['selection']['best'],b['selection']['best'])
        for x,y in zip(a['selection']['history'],b['selection']['history']):
            equal({k:v for k,v in x.items() if k!='elapsed_seconds'},{k:v for k,v in y.items() if k!='elapsed_seconds'})
        return dict(status='PASS',interrupt_pending_microbatches=3,resume_model_optimizer_rng_exact=True,complete_updates_per_arm=a['update'],synthetic_only=True)
    finally:runner.train_logical,runner.generate=old

def cpu_checks(plan):
    root=Path(plan['assets']);docs=Path(plan['docs']);split=read(plan['split_path']);c=read(plan['config_path'])
    with budget(root,'PREPARATION_CPU',7200):
        data=JointData(split);banks=[]
        for row in plan['runs']:
            for key,r in row['banks'].items():banks.append(CandidateBank(r['path'],split,r['purpose'],row['provider']['sha256'],r['seed']))
        seed_all(3201);m=build(c);first=state_digest(m.state_dict());opt=optimizer_for(m,c);initial=copy.deepcopy(opt.state_dict())
        seed_all(3201);other=build(c);assert state_digest(other.state_dict())==first;equal(optimizer_for(other,c).state_dict(),initial)
        assert all(g['lr']==1e-4 for g in opt.param_groups) and not opt.state
        bank=banks[0];sample=data.batch(split['train_indices'][:2],'train');goals=bank.batch(sample)
        damaged=copy.deepcopy(sample);damaged.agent_ids[0]=list(reversed(damaged.agent_ids[0])) if len(damaged.agent_ids[0])>1 else ['wrong']
        try:bank.batch(damaged)
        except ValueError:pass
        else:raise AssertionError('bad identity accepted')
        try:CandidateBank(bank.path,split,'selection_dev',plan['runs'][0]['provider']['sha256'],2035)
        except ValueError:pass
        else:raise AssertionError('wrong purpose accepted')
        # The original rigid transform must carry endpoint targets and candidates identically.
        target=sample.future[:,:,-1].unsqueeze(1);a,pts=augment_batch_with_points(sample,target,torch.Generator().manual_seed(3201));torch.testing.assert_close(pts[:,0],a.future[:,:,-1],atol=1e-6,rtol=1e-6)
        baseline,_=cv_baseline(a.observed);rep=EndpointBridgeDecomposition(c['scales']['goal'],c['scales']['bridge']);latent=rep.encode_target_conditioned(a.future,baseline,a.future[:,:,-1],eta=0.)
        restored,_=rep.decode_conditioned(latent,baseline,a.future[:,:,-1],eta=0.);torch.testing.assert_close(restored,a.future,atol=2e-5,rtol=2e-5)
        assert 'future' not in inspect.signature(generate).parameters
        try:
            with forbid_weight_reads():torch.load('/not/an/authorized/pretrained/file')
        except RuntimeError:pass
        else:raise AssertionError('pretrained load guard failed')
        recovery=recovery_test(root/'receipts/synthetic_recovery')
        initial_tensor_manifest=[dict(name=n,shape=list(t.shape),sha256=state_digest({n:t}),origin='random_initialization_or_fixed_math_buffer') for n,t in m.state_dict().items()]
        write(root/'identity/DOWNSTREAM_INITIAL_TENSORS.json',initial_tensor_manifest)
        result=dict(status='PASS',downstream_initial_hash=first,optimizer_initial_state_equal=True,all_parameters_lr=1e-4,all_banks_checked=len(banks),identity_negative=True,purpose_negative=True,augmentation_endpoint_alignment=True,no_future_signature=True,forbidden_pretrained_read=True,recovery=recovery,formal_updates=0)
        write(root/'reports/CPU_PRECHECK.json',result);write(docs/'CPU_PRECHECK.json',result)
        return result

def gpu_checks(plan):
    root=Path(plan['assets']);docs=Path(plan['docs']);c=read(plan['config_path'])
    with budget(root,'GPU_PRECHECK',600):
        seed_all(3201);m=build(c).cuda();opt=optimizer_for(m,c);b=synthetic_batch();goals=b.observed[:,None,:,-1].expand(-1,20,-1,-1).clone()+torch.randn(1,20,3,2)*.5
        before=state_digest(m.state_dict());m.train();gen=torch.Generator().manual_seed(3201)
        runner.train_logical(m,opt,b,goals,gen,'cuda',1)
        norms={}
        for name,module in [('shared_encoder',m.bridge.encoder),('map_unet',m.bridge.encoder.map_unet),('denoiser',m.bridge.denoiser),('coupling',m.coupling)]:
            grads=[p.grad for p in module.parameters() if p.grad is not None];assert grads and all(torch.isfinite(g).all() for g in grads);norms[name]=sum(float(g.abs().sum()) for g in grads);assert norms[name]>0
        runner.commit_step(m,opt);assert state_digest(m.state_dict())!=before
        b=b.to('cuda');goals=goals.cuda();out=generate(m,b.observed,b.semantic_maps,b.valid,b.scene_ids,b.agent_ids,goals,2035)
        # Truth substitutions are not accepted by the inference API and cannot change outputs.
        b.future=torch.randn_like(b.future)*100
        repeat=generate(m,b.observed,b.semantic_maps,b.valid,b.scene_ids,b.agent_ids,goals,2035)
        for x,y in zip(out[:3],repeat[:3]):torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-6)
        g,h,pi,receipt=out;hg=gather_candidates(goals,pi);inv=metric_invariants(g,h,goals,hg,b.future,b.valid)
        for rep in range(5):shuffle_paths(h,b.valid,b.scene_ids,2035,rep)
        single=synthetic_batch(1).to('cuda');sg=goals[:,:,:1]
        x,y,_,_=generate(m,single.observed,single.semantic_maps,single.valid,single.scene_ids,single.agent_ids,sg,2035);torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-6)
        context=m.encode_context(b.observed,b.semantic_maps,b.valid);perm=torch.randperm(20,device='cuda');s1=m.coupling(context,goals,m.bridge.representation.goal_scale);s2=m.coupling(context,goals[:,perm],m.bridge.representation.goal_scale)
        torch.testing.assert_close(s2,s1[...,perm],atol=2e-5,rtol=2e-5)
        # Assignment tie ambiguity is reported, not hidden or used to alter coupling.
        ch,p1,_=m.coupling.assign(context,goals,m.bridge.representation.goal_scale,straight_through=False);ch2,p2,_=m.coupling.assign(context,goals[:,perm],m.bridge.representation.goal_scale,straight_through=False)
        mapped=perm[p2];score_error=float((s2-s1[...,perm]).detach().abs().max())
        def objective(scores,indices):return scores.double().gather(-1,indices.permute(0,2,1)[...,None]).squeeze(-1).sum(-1)
        cost_gap=float((objective(s1,p1)-objective(s1,mapped)).detach().abs().max())
        assert cost_gap<=40*score_error+1e-7,'reordered assignment changes objective beyond floating point bound'
        exact_reordered=perm[hard_assignment_indices(s1[...,perm],b.valid)]
        exact_tie_gap=float((objective(s1,p1)-objective(s1,exact_reordered)).detach().abs().max())
        assert exact_tie_gap<=1e-7
        tie_record=dict(original_indices=p1.cpu().tolist(),reordered_indices=p2.cpu().tolist(),mapped_to_original_indices=mapped.cpu().tolist(),candidate_input_permutation=perm.cpu().tolist(),objective_max_abs_diff=cost_gap,mathematical_perturbation_bound=40*score_error+1e-7,exact_score_permutation_objective_abs_diff=exact_tie_gap,exact_score_permutation_assignment_equal=bool(torch.equal(p1,exact_reordered)))
        r=dict(status='PASS',source_hash=canonical_hash(source_identity()),gradient_abs_sums=norms,synthetic_optimizer_updates=1,formal_updates=0,no_future_output_close=True,single_person_identity=True,endpoint_invariants=inv,assignment_index_audit=tie_record,score_column_equivariance_max_abs=score_error,hard_assignment_after_reorder_exact=bool(torch.equal(ch,ch2)),near_tie_score_pairs=int((s1.sort(-1).values.diff(dim=-1).abs()<2e-5).sum()),assignment_note='float32 attention reductions may perturb nearly tied Hungarian optima; score equivariance checked, exact candidate bijections always enforced',sampling=receipt,peak_allocated_bytes=torch.cuda.max_memory_allocated())
        write(root/'reports/GPU_PRECHECK.json',r);write(docs/'GPU_PRECHECK.json',r)
        del m,opt;torch.cuda.empty_cache();return r

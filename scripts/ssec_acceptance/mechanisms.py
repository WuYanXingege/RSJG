"""Neural score/gradient/permutation audit and exact uniqueness timing, CPU only."""
from common import *
sys.path.insert(0,str(SOURCE/'tests'))
from test_bcep_scene_energy import fixture,coupling
from bcep_scene_energy.matching import solve,assignment_certificate,reference_gap
from bcep_v2.joint import pair_loss


def relative_changes(a,b):
    a=a[0].numpy();b=b[0].numpy()
    aa=a[np.argsort(a[:,0])];bb=b[np.argsort(b[:,0])]
    return int(np.count_nonzero(a!=b)),int(np.count_nonzero(aa!=bb))


def main():
    setup();rows=[];permutations=[];traces=[];starvation=[];root_causes=[]
    for n in [1,2,9,20,42,57]:
        m=coupling();before=state_hash(m);ctx,g=fixture(n);g.requires_grad_()
        captures=[]
        hook=m.coordinator.register_forward_pre_hook(lambda module,args:captures.append(args[2]))
        h,pi,score,details=m.assign(ctx,g,torch.tensor(1.),straight_through=True,return_details=True)
        loss,_=pair_loss(h,ctx.baseline[:,:,-1]+.7,ctx.valid)
        params=list(m.energy.phi.parameters())+list(m.energy.rho.parameters())
        actual_scores=details['scenes'][0]['proposal_scores']
        gg=torch.autograd.grad(loss,params+[captures[0]]+actual_scores+[g],allow_unused=True)
        assert gg[-1] is None
        eff=score-score.mean(-1,keepdim=True)-score.mean(-2,keepdim=True)+score.mean((-1,-2),keepdim=True)
        soft=details['final_soft'].detach().double();pp=soft/soft.sum(-1,keepdim=True);logp=torch.where(pp>0,pp.log(),0.)
        rows.append(dict(N=n,phi_gradient_norm=gradnorm(gg[:4]),rho_gradient_norm=gradnorm(gg[4:8]),
            incumbent_ST_gradient_norm=gradnorm(gg[8:9]),proposal_score_gradient_norm=gradnorm(gg[9:-1]),candidate_gradient_is_none=True,
            raw_score_RMS=float(score.detach().double().square().mean().sqrt()),effective_score_RMS=float(eff.detach().double().square().mean().sqrt()),
            effective_energy_fraction=float(eff.detach().double().square().sum()/score.detach().double().square().sum()) if bool(score.detach().any()) else None,
            soft_uniform_Fro=float((soft-.05).square().sum((-1,-2)).sqrt().mean()),normalized_row_entropy=float((-(pp*logp).sum(-1)/np.log(20)).mean()),
            max_soft_entry=float(soft.max()),accepted=sum(t['accepted'] for t in details['scenes'][0]['trace']),
            rejected=sum(not t['accepted'] for t in details['scenes'][0]['trace'])))
        if n>1:assert rows[-1]['phi_gradient_norm']>0 and rows[-1]['rho_gradient_norm']>0 and rows[-1]['incumbent_ST_gradient_norm']>0
        traces.extend(dict(N=n,**t) for t in details['scenes'][0]['trace']);hook.remove()
        del loss,gg,h,score,details,captures
        with torch.no_grad():
            incumbent=torch.arange(20)[None,:,None].expand(1,20,n)
            a=m.assign(ctx,g,torch.tensor(1.),straight_through=False,incumbent=incumbent)
            order=torch.arange(19,-1,-1)
            b=m.assign(ctx,g,torch.tensor(1.),straight_through=False,incumbent=incumbent[:,order]) if n>1 else a
            if n>1:
                raw,relative=relative_changes(a[1][:,order],b[1]);permutations.append(dict(N=n,operation='uniform_world_relabel',raw_changed=raw,relative_changed=relative))
                gen=torch.Generator().manual_seed(2035)
                candidate_order=torch.stack([torch.randperm(20,generator=gen) for _ in range(n)])
                rg=g.transpose(1,2).gather(2,candidate_order[None,:,:,None].expand(1,n,20,2)).transpose(1,2)
                cp=m.assign(ctx,rg,torch.tensor(1.),straight_through=False,incumbent=candidate_order.argsort(1).T[None])
                back=candidate_order.gather(1,cp[1][0].T).T[None]
                raw,relative=relative_changes(a[1],back);permutations.append(dict(N=n,operation='independent_candidate_relabel_carried_incumbent',raw_changed=raw,relative_changed=relative))
                control_relative=relative
                fresh_a=m.assign(ctx,g,torch.tensor(1.),straight_through=False,return_details=True)
                fresh_b=m.assign(ctx,rg,torch.tensor(1.),straight_through=False,return_details=True)
                back=candidate_order.gather(1,fresh_b[1][0].T).T[None]
                raw,relative=relative_changes(fresh_a[1],back);permutations.append(dict(N=n,operation='candidate_relabel_native_S0_recomputed',raw_changed=raw,relative_changed=relative))
                initial_back=candidate_order.gather(1,fresh_b[3]['initial_pi'][0].T).T[None]
                score_back=fresh_b[3]['initial_scores'].gather(3,candidate_order.argsort(1)[None,:,None].expand(1,n,20,20))
                gaps=[assignment_certificate(fresh_a[3]['initial_scores'][0,i].numpy(),fresh_a[3]['initial_pi'][0,:,i].numpy())['gap'] for i in range(n)]
                root_causes.append(dict(N=n,initial_changed_slots=int(torch.count_nonzero(initial_back!=fresh_a[3]['initial_pi'])),
                    initial_score_max_abs_change_after_inverse_relabel=float((score_back-fresh_a[3]['initial_scores']).abs().max()),
                    initial_strict_tied_people=sum(v==0 for v in gaps),initial_gap_min=min(gaps),initial_gap_max=max(gaps),
                    final_relative_changed_slots=relative,carried_incumbent_control_changed=control_relative))
                ix=torch.arange(n-1,-1,-1)
                from types import SimpleNamespace
                cc=SimpleNamespace(**{k:v[:,ix] for k,v in vars(ctx).items()})
                mapped=m.assign(cc,g[:,:,ix],torch.tensor(1.),straight_through=False,incumbent=incumbent[:,:,ix],visit_orders=[list(range(n-1,-1,-1))])
                raw,relative=relative_changes(a[1][:,:,ix],mapped[1]);permutations.append(dict(N=n,operation='people_relabel_same_physical_visit_order',raw_changed=raw,relative_changed=relative))
                changed=m.assign(cc,g[:,:,ix],torch.tensor(1.),straight_through=False,incumbent=incumbent[:,:,ix])
                raw,relative=relative_changes(a[1][:,:,ix],changed[1]);permutations.append(dict(N=n,operation='people_relabel_changed_physical_visit_order',raw_changed=raw,relative_changed=relative))
        assert state_hash(m)==before
    # An actual neural N=2 coordinate optimum makes every hard proposal reject.
    m=coupling();ctx,g=fixture(2)
    with torch.no_grad():fixed=m.assign(ctx,g,torch.tensor(1.),straight_through=False)[1]
    for policy in ['accepted_only','proposal_st']:
        m.coordinator.rejection_gradient=policy
        h,pi,s,d=m.assign(ctx,g,torch.tensor(1.),straight_through=True,incumbent=fixed,return_details=True)
        loss,_=pair_loss(h,ctx.baseline[:,:,-1]+.7,ctx.valid)
        pp=list(m.energy.phi.parameters())+list(m.energy.rho.parameters());grads=torch.autograd.grad(loss,pp,allow_unused=True)
        starvation.append(dict(policy=policy,accepted=sum(t['accepted'] for t in d['scenes'][0]['trace']),
            phi_gradient_norm=gradnorm(grads[:4]),rho_gradient_norm=gradnorm(grads[4:]),hard_incumbent_unchanged=bool(torch.equal(pi,fixed))))
    assert starvation[0]['phi_gradient_norm']==starvation[0]['rho_gradient_norm']==0
    assert starvation[1]['phi_gradient_norm']>0 and starvation[1]['rho_gradient_norm']>0
    assert all(r['accepted']==0 and r['hard_incumbent_unchanged'] for r in starvation)
    csvwrite(DOC/'NEURAL_MECHANISMS.csv',rows);csvwrite(DOC/'PERMUTATION_AUDIT.csv',permutations)
    csvwrite(DOC/'INITIAL_RELABEL_DIAGNOSIS.csv',root_causes)
    csvwrite(DOC/'CPU_ENERGY_TRACE.csv',traces);csvwrite(DOC/'REJECTION_GRADIENT.csv',starvation)
    rng=np.random.default_rng(2035);timings=[]
    for kind in ['random','strict_tie','near_tie']:
        for rep in range(100):
            if kind=='random':s=rng.normal(size=(20,20)).astype('f')
            elif kind=='strict_tie':s=rng.integers(-2,3,(20,20)).astype('f')
            else:s=np.eye(20,dtype='f')*np.float32(1e-8*(rep+1))
            pi,_=solve(s)
            start=time.perf_counter();certificate=assignment_certificate(s,pi);new=time.perf_counter()-start
            start=time.perf_counter();old=reference_gap(s,pi);oldtime=time.perf_counter()-start
            assert certificate['gap']==old
            timings.append(dict(kind=kind,repetition=rep,new_seconds=new,original_seconds=oldtime,gap=old,unique=certificate['unique'],
                new_additional_Hungarian=0,original_additional_Hungarian=20))
    csvwrite(DOC/'UNIQUENESS_BENCHMARK.csv',timings)
    write(DOC/'MECHANISM_RECEIPT.json',dict(passed=True,optimizer_guard=GUARD,unique_vs_original_cases=300,
        scores_from_actual_neural_modules=True,training_updates=0,claim_unconditional_people_equivariance=False))
    print('Neural mechanisms, rejected-gradient controls, and exact uniqueness timing complete',flush=True)


if __name__=='__main__':main()

"""Layerwise default-entry, gradient and visitation-order identity checks."""
from common import *
from test_bcep_scene_energy import fixture,coupling
from bcep_v2.joint import pair_loss
from bcep_scene_energy.matching import assignment_certificate
from localize import capture


def run(device):
    rows=[];permutations=[];traces=[];initial=[]
    for n in [2,9,20,42,57]:
        m=coupling().to(device);before=state_hash(m);ctx,g=fixture(n)
        ctx=types.SimpleNamespace(**{k:v.to(device) for k,v in vars(ctx).items()});g=g.to(device)
        order=order_for(n,device);rg=reordered(g,order)
        a,pi,da=capture(m,ctx,g);b,pj,db=capture(m,ctx,rg)
        ident=da['candidate_identity'];other=db['candidate_identity']
        assert torch.equal(ident.gather(g),other.gather(rg))
        input_map=order.gather(2,other.canonical_to_input)
        assert torch.equal(input_map,ident.canonical_to_input)
        row=dict(device=device,N=n,canonical_candidates_exact=True,identity_roundtrip_exact=True,
            endpoint_max_abs=float((a['endpoint']-b['endpoint']).abs().max()),
            candidate_feature_max_abs=float((a['candidate']-b['candidate']).abs().max()),
            world_attention_max_abs=float((a['world']-b['world']).abs().max()),
            S0_max_abs=float((da['canonical_initial_scores']-db['canonical_initial_scores']).abs().max()),
            initial_changed=int((da['initial_pi']!=map_pi(db['initial_pi'],order)).sum()),
            final_changed=int((pi!=map_pi(pj,order)).sum()),final_relative_changed=relative_changes(pi,map_pi(pj,order)))
        assert all(row[k]==0 for k in ['endpoint_max_abs','candidate_feature_max_abs','world_attention_max_abs','S0_max_abs','initial_changed','final_changed','final_relative_changed'])
        for i in range(n):
            cert=assignment_certificate(da['canonical_initial_scores'][0,i].cpu().numpy(),da['canonical_initial_pi'][0,:,i].cpu().numpy())
            initial.append(dict(device=device,N=n,person=i,**cert))
        traces.extend(dict(device=device,N=n,**t) for t in da['scenes'][0]['trace'])
        # Canonical ST reduction must be identical too, not merely hard output.
        params=list(m.energy.phi.parameters())+list(m.energy.rho.parameters());gradients=[];losses=[]
        for goals in [g,rg]:
            goals=goals.detach().requires_grad_()
            h,p,s,d=m.assign(ctx,goals,torch.tensor(1.,device=device),straight_through=True,return_details=True)
            loss,_=pair_loss(h,ctx.baseline[:,:,-1]+.7,ctx.valid)
            actual=d['scenes'][0]['proposal_scores']
            gg=torch.autograd.grad(loss,params+actual+[goals],allow_unused=True)
            assert gg[-1] is None and all(v is None or torch.isfinite(v).all() for v in gg)
            gradients.append(gg[:8]);losses.append(float(loss.detach()))
            row.update(phi_gradient_norm=norm(gg[:4]),rho_gradient_norm=norm(gg[4:8]),score_gradient_norm=norm(gg[8:-1]),frozen_candidate_gradient_none=gg[-1] is None)
        row['phi_rho_gradients_exact']=all(torch.equal(x,y) for x,y in zip(*gradients));row['pair_loss_abs_delta']=abs(losses[0]-losses[1])
        assert row['phi_rho_gradients_exact'] and row['pair_loss_abs_delta']==0 and row['phi_gradient_norm']>0 and row['rho_gradient_norm']>0
        rows.append(row)
        with torch.no_grad():
            for mode in ['default','carried']:
                incumbent=None if mode=='default' else da['initial_pi']
                ref=m.assign(ctx,g,torch.tensor(1.,device=device),straight_through=False,incumbent=incumbent)
                reverse=torch.arange(n-1,-1,-1,device=device)
                rc=types.SimpleNamespace(**{k:v[:,reverse] for k,v in vars(ctx).items()})
                for operation in ['people_relabel_same_physical_visit','people_relabel_reversed_physical_visit','reverse_physical_visit_only']:
                    if operation=='reverse_physical_visit_only':
                        out=m.assign(ctx,g,torch.tensor(1.,device=device),straight_through=False,incumbent=incumbent,visit_orders=[list(range(n-1,-1,-1))])
                        mapped=out[1]
                    else:
                        out=m.assign(rc,g[:,:,reverse],torch.tensor(1.,device=device),straight_through=False,
                            incumbent=None if incumbent is None else incumbent[:,:,reverse],
                            visit_orders=[list(range(n-1,-1,-1))] if operation=='people_relabel_same_physical_visit' else None)
                        mapped=out[1][:,:,reverse]
                    permutations.append(dict(device=device,N=n,entry=mode,operation=operation,raw_changed=int((ref[1]!=mapped).sum()),relative_changed=relative_changes(ref[1],mapped)))
            # Uniform world rename is tested by transporting the same incumbent;
            # merely renumbering output worlds is always a label transformation.
            w=torch.arange(19,-1,-1,device=device)
            ref=m.assign(ctx,g,torch.tensor(1.,device=device),straight_through=False,incumbent=da['initial_pi'])
            out=m.assign(ctx,g,torch.tensor(1.,device=device),straight_through=False,incumbent=da['initial_pi'][:,w])
            permutations.append(dict(device=device,N=n,entry='carried',operation='uniform_world_relabel',raw_changed=int((ref[1][:,w]!=out[1]).sum()),relative_changed=relative_changes(ref[1][:,w],out[1])))
        assert state_hash(m)==before;print(device,n,'default relabel PASS; gradient exact',flush=True)
    return rows,permutations,traces,initial


def main():
    setup(cuda=True);combined=[[],[],[],[]]
    for device in ['cpu','cuda']:
        for dst,src in zip(combined,run(device)):dst.extend(src)
    for name,rows in zip(['LAYER_AND_GRADIENT_COMPARISON','VISIT_ORDER_AUDIT','ENERGY_TRACE','INITIAL_MATCH_CERTIFICATES'],combined):csvwrite(DOC/(name+'.csv'),rows)
    write(DOC/'MECHANISM_RECEIPT.json',dict(passed=True,default_entry_recomputed=True,CPU_GPU_cases=10,optimizer_guard=GUARD,
        states_unchanged=True,candidate_gradients_none=True,phi_rho_exact_under_relabel=True))
if __name__=='__main__':main()

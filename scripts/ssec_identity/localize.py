"""Reproduce original failures without updating parameters or regenerating assets."""
from common import *
from test_bcep_scene_energy import fixture
from bcep_scene_energy.matching import assignment_certificate


def capture(m,ctx,g):
    captured={}
    hooks=[m.endpoint.register_forward_hook(lambda mod,args,out:captured.update(endpoint=out.detach())),
           m.candidate_norm.register_forward_hook(lambda mod,args,out:captured.update(candidate=out.detach())),
           m.scene_cross_attention.register_forward_hook(lambda mod,args,out:captured.update(world=out[0].detach()))]
    try:
        with torch.no_grad():h,pi,s,d=m.assign(ctx,g,torch.tensor(1.,device=g.device),straight_through=False,return_details=True)
    finally:
        for h in hooks:h.remove()
    return captured,pi,d


def main():
    setup();rows=[];certs=[];expected={2:0,9:42,20:18,42:119,57:101}
    for n in expected:
        torch.manual_seed(3201);m=old_class()();m.enable_energy();before=state_hash(m)
        ctx,g=fixture(n);order=order_for(n);rg=reordered(g,order);inverse=order.argsort(-1)
        a,pa,da=capture(m,ctx,g);b,pb,db=capture(m,ctx,rg)
        def inv(v):return v.gather(2,inverse[...,None].expand_as(v))
        endpoint_b=inv(b['endpoint'].transpose(1,2)).transpose(1,2)
        score_b=db['initial_scores'].gather(3,inverse[:,:,None].expand_as(db['initial_scores']))
        ai=da['initial_pi'];bi=map_pi(db['initial_pi'],order)
        gaps=[]
        for i in range(n):
            c=assignment_certificate(da['initial_scores'][0,i].numpy(),ai[0,:,i].numpy());gaps.append(c['gap'])
            certs.append(dict(N=n,person=i,changed_slots=int((ai[:,:,i]!=bi[:,:,i]).sum()),**c))
        with torch.no_grad():
            # Freeze the SAME physical initial choice; use reordered coordinates.
            transported=inverse.gather(2,ai.transpose(1,2)).transpose(1,2)
            carried=m.assign(ctx,rg,torch.tensor(1.),straight_through=False,incumbent=transported)
        row=dict(N=n,input_inverse_exact=bool(torch.equal(g,reordered(rg,inverse))),
            endpoint_max_abs=float((a['endpoint']-endpoint_b).abs().max()),
            candidate_feature_max_abs=float((a['candidate']-inv(b['candidate'])).abs().max()),
            world_attention_max_abs=float((a['world']-b['world']).abs().max()),
            score_max_abs=float((da['initial_scores']-score_b).abs().max()),
            strict_tied_people=sum(v==0 for v in gaps),initial_changed=int((ai!=bi).sum()),
            initial_relative_changed=relative_changes(ai,bi),final_relative_changed=relative_changes(pa,map_pi(pb,order)),
            same_initial_final_relative_changed=relative_changes(pa,map_pi(carried[1],order)))
        assert row['final_relative_changed']==expected[n] and state_hash(m)==before
        rows.append(row);print(row,flush=True)
    csvwrite(DOC/'BASE_LAYER_LOCALIZATION.csv',rows);csvwrite(DOC/'BASE_INITIAL_MATCHES.csv',certs)
    write(DOC/'BASE_REPRODUCTION.json',dict(passed=True,expected_relative_changes=expected,optimizer_guard=GUARD))
if __name__=='__main__':main()

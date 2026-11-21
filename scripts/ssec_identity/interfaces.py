"""Reuse unchanged native interface checks and verify real bank identity transport."""
from common import *
import importlib.util
from bcep_scene_energy.model import build
from bcep_scene_energy.identity import bank_occurrence_ids,canonical_order
from bcep_scene_energy.inference import IdentityBoundModel
from bcep_v2.objective import bridge_and_pair_loss


def main():
    setup(cuda=True)
    # The previous zero-update interface suite is unchanged. Its imports resolve
    # this audit's isolated common.OUT/DOC, never the historical acceptance root.
    spec=importlib.util.spec_from_file_location('_native_interface_regression',SOURCE/'scripts/ssec_acceptance/interfaces.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.main()
    from bcep_v3.formal_inputs import FormalInputs
    inputs=FormalInputs();seed();m,_=build(inputs.config);m.cuda().eval();before=state_hash(m);rows=[]
    for n in [1,2,9,42]:
        index=next(i for i in inputs.split['selection_dev_indices'] if len(inputs.split['members'][i]['agent_ids'])==n)
        batch=inputs.data.batch([index],'selection_dev');g,ids,namespace=bank_occurrence_ids(inputs.banks[1],[index],batch)
        order=order_for(n);rg=reordered(g,order);x=batch.to('cuda');g=g.cuda();rg=rg.cuda();ids=ids.cuda();order=order.cuda()
        with torch.no_grad():
            ctx=m.encode_context(x.observed,x.semantic_maps,x.valid)
            a=m.coupling.assign(ctx,g,m.bridge.representation.goal_scale,straight_through=False,return_details=True,candidate_ids=ids)
            b=m.coupling.assign(ctx,rg,m.bridge.representation.goal_scale,straight_through=False,return_details=True,candidate_ids=order)
        inverse=order.argsort(-1)
        hback=reordered(b[0],inverse) if n==1 else b[0]
        initial_back=map_pi(b[3]['initial_pi'],order)
        final_back=map_pi(b[1],order)
        if n==1:
            initial_back=initial_back.gather(1,inverse.transpose(1,2));final_back=final_back.gather(1,inverse.transpose(1,2))
        assert torch.equal(a[0],hback) and torch.equal(a[1],final_back) and torch.equal(a[3]['initial_pi'],initial_back)
        truth=x.future[:,:,-1]
        fde=(g-truth[:,None]).norm(dim=-1).min(1).values
        assert torch.equal(fde,(a[0]-truth[:,None]).norm(dim=-1).min(1).values)
        rows.append(dict(N=n,index=index,bank_hash=namespace[0]['bank_hash'],row_sha256=namespace[0]['row_sha256'],
            initial_changed=int((a[3]['initial_pi']!=initial_back).sum()),final_changed=int((a[1]!=final_back).sum()),
            endpoint_max_abs=float((a[0]-hback).abs().max()),minFDE_abs_difference=0.,noise_identity_origin='verified bank row + agent + original slot',
            purpose='selection_dev forward only'))
    # Explicit IDs also work through the UNCHANGED training objective; only
    # synthetic backward is allowed. Coincident occurrences never use array-ID guesses.
    from bcep_v3.fixtures import SyntheticData,SyntheticBank
    d=SyntheticData((2,),pixels=256);batch=d.batch([0],'train');g=SyntheticBank().batch([0],batch).cuda();g[:,1]=g[:,0]
    x=batch.to('cuda');ids=torch.arange(20,device='cuda').expand(1,2,20);order=order_for(2,'cuda')
    noise=torch.randn((1,1,2,12,2),generator=torch.Generator().manual_seed(3201)).cuda();t=torch.tensor([.37],device='cuda')
    values=[]
    for goals,identity in [(g,ids),(reordered(g,order),order)]:
        loss=bridge_and_pair_loss(IdentityBoundModel(m,identity),x,goals,time_draw=t,noise=noise)[0]
        gg=torch.autograd.grad(loss,list(m.energy.parameters()) if hasattr(m,'energy') else list(m.coupling.energy.parameters()),allow_unused=True)
        assert all(v is not None and torch.isfinite(v).all() for v in gg)
        values.append((float(loss.detach()),gg))
    assert values[0][0]==values[1][0] and all(torch.equal(a,b) for a,b in zip(values[0][1],values[1][1]))
    assert state_hash(m)==before
    csvwrite(DOC/'OFFICIAL_BANK_RELABEL.csv',rows)
    write(DOC/'IDENTITY_INTERFACE_RECEIPT.json',dict(passed=True,optimizer_guard=GUARD,official_forward_cases=4,
        official_backward_calls=0,duplicate_occurrence_native_objective_gradients_exact=True,states_unchanged=True))
if __name__=='__main__':main()

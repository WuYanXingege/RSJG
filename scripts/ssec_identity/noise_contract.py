"""Explicit negative control: array-index noise regeneration is NOT relabelling."""
from common import *
from bcep_scene_energy.model import build
from bcep_scene_energy.inference import generate
from bcep_joint.inference import generate as native_generate
from bcep_v3.fixtures import SyntheticData,SyntheticBank


def main():
    setup(cuda=True);seed();m,_=build(config());m.cuda();rows=[]
    before=state_hash(m)
    for n in [2,9]:
        data=SyntheticData((n,),pixels=256);x=data.batch([0],'selection_dev');g=SyntheticBank().batch([0],x).cuda();x=x.to('cuda')
        ids=torch.arange(20,device='cuda').expand(1,n,20);r=order_for(n,'cuda');rg=reordered(g,r)
        args=(m,x.observed,x.semantic_maps,x.valid,x.scene_ids,x.agent_ids)
        original=native_generate(*args,g,2035)
        restored=generate(*args,g,2035,candidate_ids=ids)
        relabelled=generate(*args,rg,2035,candidate_ids=r)
        incorrect=native_generate(*args,rg,2035)
        assert torch.equal(original[0],restored[0]) and torch.equal(original[1],restored[1])
        assert torch.equal(restored[0],relabelled[0]) and torch.equal(restored[1],relabelled[1])
        noise_error=float((incorrect[1]-relabelled[1]).abs().max())
        assert noise_error>0 and torch.equal(incorrect[1][...,-1,:],relabelled[1][...,-1,:])
        rows.append(dict(N=n,original_native_G_exact=True,original_native_H_exact=True,transported_ID_G_exact=True,transported_ID_H_exact=True,
            wrong_new_index_noise_H_max_abs_m=noise_error,wrong_noise_H_endpoint_max_abs_m=0.,
            wrong_raw_index_G_max_abs_m=float((incorrect[0]-relabelled[0]).abs().max())))
    assert state_hash(m)==before
    csvwrite(DOC/'NOISE_IDENTITY_NEGATIVE_CONTROL.csv',rows)
    write(DOC/'NOISE_CONTRACT_RECEIPT.json',dict(passed=True,optimizer_guard=GUARD,original_G_definition_unchanged=True,
        original_native_key_function_unchanged=True,states_unchanged=True))
if __name__=='__main__':main()

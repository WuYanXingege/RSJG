"""End-to-end K20 / 20-step DDIM capacity, no optimizer, synthetic inputs only."""
from common import *
from bcep_scene_energy.model import build
from bcep_v3.fixtures import SyntheticData,SyntheticBank
import bcep_joint.inference as inference


def main():
    setup(cuda=True);seed();m,_=build(config());m.cuda().eval();before=state_hash(m);rows=[]
    native=m.coupling.assign;count=[0]
    def assigned(*a,**kw):count[0]+=1;return native(*a,**kw)
    m.coupling.assign=assigned
    for n in [1,2,9,20,42,57]:
        data=SyntheticData((n,),pixels=256);x=data.batch([0],'selection_dev')
        goals=SyntheticBank().batch([0],x).cuda();x=x.to('cuda');count[0]=0
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        _,path,pi,checks=inference.generate(m,x.observed,x.semantic_maps,x.valid,x.scene_ids,x.agent_ids,goals,2035,mode='H')
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        assert count[0]==1 and checks['endpoint_max_abs_m']<=1e-6 and bool(torch.isfinite(path).all())
        rows.append(dict(N=n,K=20,DDIM_steps=20,coupling_calls=count[0],seconds=elapsed,
            allocated_peak_MiB=torch.cuda.max_memory_allocated()/2**20,**checks))
        print('DDIM capacity',n,round(elapsed,3),'seconds',round(rows[-1]['allocated_peak_MiB'],1),'MiB',flush=True)
        del x,goals,path,pi
    m.coupling.assign=native
    assert state_hash(m)==before and GUARD['optimizer_step_calls']==0
    csvwrite(DOC/'DDIM_CAPACITY.csv',rows)
    write(DOC/'DDIM_RECEIPT.json',dict(passed=True,states_unchanged=True,optimizer_guard=GUARD,
        purpose='synthetic capacity only; single measurements, not an A/B latency benchmark'))


if __name__=='__main__':main()

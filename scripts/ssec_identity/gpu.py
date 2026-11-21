"""Same-weights A/B cost, default full-objective gradients, native G/H DDIM."""
from common import *
from bcep_scene_energy.model import build
from bcep_scene_energy.inference import generate,IdentityBoundModel
from bcep_scene_energy.identity import canonical_order
from bcep_v3.fixtures import SyntheticData,SyntheticBank
from bcep_v2.objective import bridge_and_pair_loss
from ebjd import unified
from bcep_joint.inference import keyed_noise


def main():
    setup(cuda=True);seed();model,_=build(config());model.cuda();before=state_hash(model)
    fixed_class=type(model.coupling);legacy_class=old_class();costs=[];endtoend=[];maps=[];grads=[]
    native=unified.linear_sum_assignment;count=[0]
    def counted(*a,**kw):count[0]+=1;return native(*a,**kw)
    unified.linear_sum_assignment=counted
    for n in [1,2,9,20,42,57]:
        data=SyntheticData((n,),pixels=256);batch=data.batch([0],'train');g=SyntheticBank().batch([0],batch).cuda();x=batch.to('cuda')
        order=order_for(n,'cuda');rg=reordered(g,order);ids=torch.arange(20,device='cuda').expand(1,n,20)
        noise=torch.randn((1,1,n,12,2),generator=torch.Generator().manual_seed(3201)).cuda();t=torch.tensor([.37],device='cuda')
        with torch.no_grad():ctx=model.encode_context(x.observed,x.semantic_maps,x.valid)
        # Before/after use the SAME module state and input, in this process.
        for label,cls in [('base',legacy_class),('fixed',fixed_class)]:
            model.coupling.__class__=cls;model.train()
            for scope in ['assignment_forward','full_native_objective_forward_backward']:
                for rep in range(-1,3):
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();count[0]=0;start=time.perf_counter()
                    if scope=='assignment_forward':
                        with torch.no_grad():out=model.coupling.assign(ctx,g,model.bridge.representation.goal_scale,straight_through=False)
                    else:
                        loss=bridge_and_pair_loss(model,x,g,time_draw=t,noise=noise)[0]
                        gg=torch.autograd.grad(loss,list(model.parameters()),allow_unused=True)
                        assert all(v is None or torch.isfinite(v).all() for v in gg)
                    torch.cuda.synchronize();seconds=time.perf_counter()-start
                    if rep>=0:costs.append(dict(N=n,model=label,scope=scope,repetition=rep,seconds=seconds,allocated_peak_MiB=torch.cuda.max_memory_allocated()/2**20,Hungarian_calls=count[0]))
                    if scope=='assignment_forward':del out
                    else:del loss,gg
        model.coupling.__class__=fixed_class
        # Real native objective on synthetic inputs, default entry, exact gradients.
        pp=list(model.parameters());allgrads=[];losses=[]
        for goals in [g,rg]:
            goals=goals.detach().requires_grad_()
            loss=bridge_and_pair_loss(model,x,goals,time_draw=t,noise=noise)[0]
            gg=torch.autograd.grad(loss,pp+[goals],allow_unused=True)
            assert gg[-1] is None;allgrads.append(gg[:-1]);losses.append(float(loss.detach()))
        differences=[float((a-b).abs().max()) for a,b in zip(*allgrads) if a is not None and b is not None]
        same=all((a is None and b is None) or (a is not None and b is not None and torch.equal(a,b)) for a,b in zip(*allgrads))
        # N1 uses native identity worlds; pair softmin is permutation invariant
        # mathematically, but a changed sum order need not be bitwise identical.
        grads.append(dict(N=n,loss_abs_delta=abs(losses[0]-losses[1]),all_model_gradients_exact=same,
                          gradient_max_abs=max(differences,default=0),candidate_gradient_none=True))
        if n>1:assert same and losses[0]==losses[1]
        del allgrads,gg,loss
        model.eval()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        first=generate(model,x.observed,x.semantic_maps,x.valid,x.scene_ids,x.agent_ids,g,2035,candidate_ids=ids)
        second=generate(model,x.observed,x.semantic_maps,x.valid,x.scene_ids,x.agent_ids,rg,2035,candidate_ids=order)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        a,b=first,second
        assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]) and torch.equal(a[2],map_pi(b[2],order))
        z=keyed_noise(x.scene_ids,x.agent_ids,x.valid,2035).cuda();rz=reordered(z,order)
        ha=unified.gather_candidates(z,a[2]);hb=unified.gather_candidates(rz,b[2])
        assert torch.equal(ha,hb)
        endtoend.append(dict(case='original',N=n,G_max_abs=float((a[0]-b[0]).abs().max()),H_max_abs=float((a[1]-b[1]).abs().max()),
            H_endpoint_max_abs=float((a[1][...,-1,:]-b[1][...,-1,:]).abs().max()),final_relative_changed=relative_changes(a[2],map_pi(b[2],order)),
            keyed_noise_exact=True,two_GH_calls_seconds=elapsed,allocated_peak_MiB=torch.cuda.max_memory_allocated()/2**20,ddim_steps=20))
        cm=canonical_order(g,x.valid,ids);rm=canonical_order(rg,x.valid,order)
        for i in range(n):
            for k in range(20):
                j=int(cm.canonical_to_input[0,i,k]);rj=int(rm.canonical_to_input[0,i,k])
                maps.append(dict(N=n,person=i,canonical_slot=k,original_input_slot=j,relabelled_input_slot=rj,transported_original_id=int(order[0,i,rj]),
                    coordinates_bitwise_equal=bool(torch.equal(g[0,j,i].view(torch.int32),rg[0,rj,i].view(torch.int32)))))
        if n==2:
            dup=g.clone();dup[:,1]=dup[:,0];rdup=reordered(dup,order)
            aa=generate(model,x.observed,x.semantic_maps,x.valid,x.scene_ids,x.agent_ids,dup,2035,candidate_ids=ids)
            bb=generate(model,x.observed,x.semantic_maps,x.valid,x.scene_ids,x.agent_ids,rdup,2035,candidate_ids=order)
            assert torch.equal(aa[0],bb[0]) and torch.equal(aa[1],bb[1])
            endtoend.append(dict(case='duplicate_with_occurrence_ids',N=n,G_max_abs=0.,H_max_abs=0.,H_endpoint_max_abs=0.,
                final_relative_changed=relative_changes(aa[2],map_pi(bb[2],order)),keyed_noise_exact=True,two_GH_calls_seconds=None,allocated_peak_MiB=None,ddim_steps=20))
        if n==9:
            from bcep_v2.stages import augment
            g_cpu=g.cpu();r_cpu=order.cpu()
            ax,ag,_=augment(batch,g_cpu,torch.Generator().manual_seed(3301))
            bx,bg,_=augment(batch,reordered(g_cpu,r_cpu),torch.Generator().manual_seed(3301))
            assert torch.equal(ax.observed,bx.observed) and torch.equal(ax.semantic_maps,bx.semantic_maps)
            ax=ax.to('cuda');bx=bx.to('cuda')
            aa=generate(model,ax.observed,ax.semantic_maps,ax.valid,ax.scene_ids,ax.agent_ids,ag.cuda(),2035,candidate_ids=ids)
            bb=generate(model,bx.observed,bx.semantic_maps,bx.valid,bx.scene_ids,bx.agent_ids,bg.cuda(),2035,candidate_ids=order)
            assert torch.equal(aa[0],bb[0]) and torch.equal(aa[1],bb[1])
            endtoend.append(dict(case='native_synchronized_augmentation',N=n,G_max_abs=0.,H_max_abs=0.,H_endpoint_max_abs=0.,
                final_relative_changed=relative_changes(aa[2],map_pi(bb[2],order)),keyed_noise_exact=True,two_GH_calls_seconds=None,allocated_peak_MiB=None,ddim_steps=20))
        assert state_hash(model)==before
        csvwrite(DOC/'GPU_COST_RAW.csv',costs);csvwrite(DOC/'END_TO_END_DDIM.csv',endtoend)
        csvwrite(DOC/'CANDIDATE_MAPPING_PROOF.csv',maps);csvwrite(DOC/'FULL_OBJECTIVE_GRADIENTS.csv',grads)
        print('N',n,'same-weight cost and exact G/H identity PASS',flush=True)
    unified.linear_sum_assignment=native
    summary=[]
    for n in [1,2,9,20,42,57]:
        for scope in ['assignment_forward','full_native_objective_forward_backward']:
            old=[r for r in costs if r['N']==n and r['scope']==scope and r['model']=='base']
            new=[r for r in costs if r['N']==n and r['scope']==scope and r['model']=='fixed']
            a=float(np.median([r['seconds'] for r in old]));b=float(np.median([r['seconds'] for r in new]))
            summary.append(dict(N=n,scope=scope,base_median_seconds=a,fixed_median_seconds=b,ratio=b/a,
                fixed_peak_MiB=max(r['allocated_peak_MiB'] for r in new),Hungarian_calls=new[0]['Hungarian_calls']))
    csvwrite(DOC/'GPU_COST_SUMMARY.csv',summary)
    write(DOC/'GPU_RECEIPT.json',dict(passed=True,optimizer_guard=GUARD,device=torch.cuda.get_device_name(),torch=torch.__version__,
        FP32=True,parameters=sum(p.numel() for p in model.parameters()),same_weights_same_inputs=True,states_unchanged=True,state_sha256=before,
        network_sha256={str(p.relative_to(SOURCE)):sha(p) for p in (SOURCE/'bcep_scene_energy').glob('*.py')}))
if __name__=='__main__':main()

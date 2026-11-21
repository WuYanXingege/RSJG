"""Same-session/input FP32 v2/v3/SSEC A/B; synthetic forward+VJP, zero optimizer steps."""
from common import *
from bcep_v2.joint import build_joint
from bcep_v3.model import build as build_v3
from bcep_scene_energy.model import build as build_energy
from bcep_v3.fixtures import SyntheticData,SyntheticBank
from bcep_v2.objective import bridge_and_pair_loss
from ebjd import unified


def tensors_hash(*values):
    h=hashlib.sha256()
    for v in values:h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    setup(cuda=True);c=config();models={};origins={}
    for label in ['v2','v3','energy_only','energy_csee']:
        seed()
        if label=='v2':m=build_joint(c)
        elif label=='v3':m,_=build_v3(c)
        else:m,origins[label]=build_energy(c,variant=label)
        models[label]=m
    before={k:state_hash(m) for k,m in models.items()}
    write(DOC/'INITIALIZATION_ORIGIN.json',origins)
    parameters=[dict(model=k,total_parameters=sum(p.numel() for p in m.parameters()),
        trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),
        parameter_tensors=len(list(m.parameters()))) for k,m in models.items()]
    csvwrite(DOC/'PARAMETERS.csv',parameters)
    rows=[];traces=[];gradrows=[]
    native=unified.linear_sum_assignment;counter=[0]
    def counted(*a,**kw):counter[0]+=1;return native(*a,**kw)
    unified.linear_sum_assignment=counted
    cases=[(n,False) for n in [1,2,9,20,42,57]]+[(57,True)]
    for case_number,(n,dense) in enumerate(cases):
        data=SyntheticData((n,),pixels=256);batch=data.batch([0],'train')
        if dense:batch.observed.zero_()
        goals=SyntheticBank().batch([0],batch)
        x=batch.to('cuda');g=goals.cuda();t=torch.tensor([.37],device='cuda')
        noise=torch.randn((1,1,n,12,2),generator=torch.Generator().manual_seed(3201)).cuda()
        identity=tensors_hash(x.observed,x.future,x.semantic_maps,x.valid,g,t,noise)
        labels=list(models);labels=labels[case_number%4:]+labels[:case_number%4]
        for label in labels:
            m=models[label].cuda().train();params=[(name,p) for name,p in m.named_parameters() if p.requires_grad]
            for kind in ['coupling_pair','full_native_objective']:
                for repetition in range(-1,3):
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();baseline=torch.cuda.memory_allocated()
                    counter[0]=0;start=time.perf_counter()
                    if kind=='coupling_pair':
                        context=m.encode_context(x.observed,x.semantic_maps,x.valid)
                        h,pi,score=m.coupling.assign(context,g,m.bridge.representation.goal_scale,straight_through=True)
                        from bcep_v2.joint import pair_loss
                        loss,_=pair_loss(h,x.future[:,:,-1],x.valid)
                    else:loss=bridge_and_pair_loss(m,x,g,time_draw=t,noise=noise)[0]
                    torch.cuda.synchronize();middle=time.perf_counter()
                    gg=torch.autograd.grad(loss,[p for _,p in params],allow_unused=True)
                    torch.cuda.synchronize();end=time.perf_counter()
                    assert all(v is None or bool(torch.isfinite(v).all()) for v in gg)
                    if repetition>=0:
                        row=dict(model=label,N=n,dense_all_tied_neighbors=dense,kind=kind,repetition=repetition,
                            input_sha256=identity,forward_seconds=middle-start,backward_seconds=end-middle,total_seconds=end-start,
                            allocated_peak_MB=torch.cuda.max_memory_allocated()/2**20,allocated_baseline_MB=baseline/2**20,
                            allocated_delta_MB=(torch.cuda.max_memory_allocated()-baseline)/2**20,hungarian_calls=counter[0],
                            synthetic_loss=float(loss.detach()),optimizer_updates=0)
                        if label.startswith('energy'):
                            trace=m.coupling.last_diagnostics[0]['trace']
                            row.update(accepted=sum(v['accepted'] for v in trace),rejected=sum(not v['accepted'] for v in trace),edges=m.coupling.last_diagnostics[0]['edges'])
                            if repetition==0:
                                traces.extend(dict(model=label,N=n,dense_all_tied_neighbors=dense,kind=kind,**v) for v in trace)
                        else:row.update(accepted=None,rejected=None,edges=m.coupling.last_edge_count)
                        rows.append(row)
                        for group,prefix in [('phi','coupling.energy.phi.'),('rho','coupling.energy.rho.'),('csee','coupling.csee.'),('shared_encoder','bridge.encoder.')]:
                            members=[v for (name,p),v in zip(params,gg) if name.startswith(prefix)]
                            if members:
                                norm=gradnorm(members)
                                if label.startswith('energy') and n>1 and group in ['phi','rho']:assert norm>0
                                gradrows.append(dict(model=label,N=n,dense_all_tied_neighbors=dense,kind=kind,repetition=repetition,module=group,
                                    norm=norm,used_tensors=sum(v is not None for v in members),nonzero_tensors=sum(v is not None and bool(v.any()) for v in members)))
                    del loss,gg
                    if kind=='coupling_pair':del context,h,pi,score
                print(label,n,'dense',dense,kind,round(rows[-1]['total_seconds'],4),'s',round(rows[-1]['allocated_peak_MB'],1),'MB',flush=True)
                csvwrite(DOC/'GPU_BENCHMARK.csv',rows);csvwrite(DOC/'GPU_GRADIENTS.csv',gradrows)
                if traces:csvwrite(DOC/'GPU_ENERGY_TRACE.csv',traces)
            m.cpu();torch.cuda.empty_cache()
        del x,g,t,noise,batch,goals
    unified.linear_sum_assignment=native
    after={k:state_hash(m) for k,m in models.items()};assert before==after and GUARD['optimizer_step_calls']==0
    write(DOC/'GPU_RECEIPT.json',dict(passed=True,device=torch.cuda.get_device_name(),total_device_memory=torch.cuda.get_device_properties(0).total_memory,
        torch=torch.__version__,input_maps=[6,256,256],FP32=True,autocast=False,TF32=False,
        warmup_per_case=1,repetitions=3,same_session_same_inputs=True,model_state_sha256=before,states_unchanged=True,
        optimizer_guard=GUARD,real_updates=0,synthetic_updates=0,benchmark_weights='fresh seed3201 structural comparison; no trained-weight warm start',
        network_source_sha256={str(p.relative_to(SOURCE)):sha(p) for p in (SOURCE/'bcep_scene_energy').glob('*.py')}))


if __name__=='__main__':main()

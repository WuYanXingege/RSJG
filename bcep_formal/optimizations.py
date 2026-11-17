"""Execution-only candidates; no change to checkpoint keys or loss weights."""
import math
import torch
from bcep.motion_frame import frame,to_local,to_world
from bcep_joint.common import local_seed

def finite_tensors(values, label):
    flags=[torch.isfinite(x).all() for x in values if x is not None]
    if flags and not bool(torch.stack(flags).all()):raise FloatingPointError('nonfinite '+label)

def provider_loss_batched(model, observed, maps, valid, future, aux_weight=.25):
    context=model.encoder(observed,maps,valid,model.dt);horizons=sorted(set([math.ceil(model.pred_length/3),math.ceil(2*model.pred_length/3),model.pred_length]));n=int(valid.sum())
    residual=[];jac=[]
    for h in horizons:
        b,r,s=frame(observed[valid],h,model.dt,model.scales[h-1]);residual.append(to_local(future[valid,h-1],b,r,s));jac.append(s.log().sum(-1))
    horizon=torch.tensor(horizons,device=observed.device,dtype=observed.dtype)[:,None,None].expand(-1,n,1).reshape(-1,1)*model.dt
    # The original scalar horizon is now a per-row constant. All flow layers,
    # transforms and density Jacobians are unchanged.
    logp=model.flow.log_prob(torch.cat(residual),context[valid].repeat(len(horizons),1),horizon).reshape(len(horizons),n)
    sums={h:-(logp[j]-jac[j]).sum() for j,h in enumerate(horizons)}
    aux={h:dict(nll_sum=v,valid_agent_count=n) for h,v in sums.items() if h!=model.pred_length}
    total=sums[model.pred_length]/n+aux_weight*torch.stack([v['nll_sum']/n for v in aux.values()]).mean()
    return dict(loss_total=total,nll_final_sum=sums[model.pred_length],valid_agent_count=n,auxiliary=aux,world_jacobian_included=True)

def train_logical_grouped(model,opt,cpu_batch,proposals,gen,device,accumulation,max_slots=32):
    from ebjd.data import augment_batch_with_points
    from bcep_joint.runner import bridge_and_pair_loss
    batch,proposals=augment_batch_with_points(cpu_batch,proposals,gen)
    bs=len(batch.observed);counts=batch.valid.sum(-1).tolist();agents=sum(counts)
    time_draw=torch.randint(1,101,(bs,),device=device).float()/100
    noise=torch.randn(bs,1,batch.valid.shape[1],12,2,device=device,dtype=torch.float32)
    totals=torch.zeros(3,device=device);start=0
    while start<bs:
        stop=start+1
        while stop<bs and max(counts[start:stop+1])*(stop+1-start)<=max_slots:stop+=1
        physical=batch.compact_scenes(start,stop).to(device);n=physical.valid.shape[1]
        goals=torch.zeros(stop-start,20,n,2,device=device);draw=torch.zeros(stop-start,1,n,12,2,device=device)
        for j,i in enumerate(range(start,stop)):
            pos=batch.valid[i].nonzero().flatten();goals[j,:,:len(pos)]=proposals[i,:,pos].to(device);draw[j,:,:len(pos)]=noise[i,:,pos.to(device)]
        _,bridge,pair,agent_part,scene_part=bridge_and_pair_loss(model,physical,goals,time_draw=time_draw[start:stop],noise=draw)
        total=sum(counts[start:stop])/agents*agent_part+(stop-start)/bs*scene_part
        finite_tensors([total],'joint loss');(total/accumulation).backward()
        totals+=torch.stack((total.detach(),pair.detach()*(stop-start)/bs,bridge.detach()*(stop-start)/bs))
        start=stop
    return totals.cpu().tolist()

def commit_step(model,opt):
    finite_tensors([p.grad for p in model.parameters()],'gradients')
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
    opt.step();opt.zero_grad(set_to_none=True)
    finite_tensors(model.parameters(),'parameters')
    return float(norm)

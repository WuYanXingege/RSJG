"""No GDTS imports, weights or teacher inputs. World-coordinate output [K,N,2]."""
import hashlib,json,math
import numpy as np
import torch
from torch import nn
from .context_encoder import ContextEncoder
from .conditional_flow import ConditionalFlow
from .motion_frame import frame,to_local,to_world
from .coverage_selector import select_goals

class BCEP(nn.Module):
    def __init__(self,pred_length=12,dt=.4):
        super().__init__();self.pred_length=pred_length;self.dt=dt
        self.encoder=ContextEncoder();self.flow=ConditionalFlow()
        self.register_buffer('scales',torch.ones(pred_length,2));self.register_buffer('scales_fitted',torch.tensor(False))
    def log_prob_world(self,world,observed,context,horizon):
        b,r,s=frame(observed,horizon,self.dt,self.scales[horizon-1]);local=to_local(world,b,r,s)
        return self.flow.log_prob(local,context,horizon*self.dt)-s.log().sum(-1)
    def loss(self,observed,semantic_maps,valid,future,aux_weight=.25):
        if future.shape!=(len(observed),self.pred_length,2):raise ValueError('future target shape differs')
        context=self.encoder(observed,semantic_maps,valid,self.dt)
        horizons=sorted(set([math.ceil(self.pred_length/3),math.ceil(2*self.pred_length/3),self.pred_length]))
        terms={h:-self.log_prob_world(future[valid,h-1],observed[valid],context[valid],h).mean() for h in horizons}
        auxiliary=[v for h,v in terms.items() if h!=self.pred_length]
        return terms[self.pred_length]+(aux_weight*torch.stack(auxiliary).mean() if auxiliary else 0)
    @torch.no_grad()
    def sample_support(self,observed,semantic_maps,valid,scene_id,agent_ids,seed,count=256):
        if type(count) is not int or count<1:raise ValueError('positive support count required')
        if len(agent_ids)!=len(observed):raise ValueError('identity count mismatch')
        active_ids=[agent_ids[i] for i in valid.nonzero().flatten().tolist()]
        if len(set(active_ids))!=len(active_ids):raise ValueError('valid agent identities must be unique')
        context=self.encoder(observed,semantic_maps,valid,self.dt);samples=torch.zeros(count,len(observed),2,device=observed.device,dtype=observed.dtype)
        b,r,s=frame(observed,self.pred_length,self.dt,self.scales[-1])
        for i in valid.nonzero().flatten().tolist():
            key=json.dumps([str(scene_id),str(agent_ids[i]),int(seed)],separators=(',',':')).encode()
            local_seed=int.from_bytes(hashlib.sha256(key).digest()[:8],'little')%(2**63-1)
            gen=torch.Generator(device='cpu').manual_seed(local_seed)
            z=torch.randn(count,2,generator=gen,dtype=observed.dtype).to(observed.device)
            residual,_=self.flow.transform(z,context[i].expand(count,-1),self.pred_length*self.dt)
            samples[:,i]=to_world(residual,b[i],r[i],s)
        return samples,torch.full((count,),1/count,device=observed.device,dtype=observed.dtype)
    def sample_iid(self,*args,k=20,**kwargs):return self.sample_support(*args,count=k,**kwargs)[0]
    @torch.no_grad()
    def predict(self,observed,semantic_maps,valid,scene_id,agent_ids,seed=2035,k=20,support=256):
        atoms,mass=self.sample_support(observed,semantic_maps,valid,scene_id,agent_ids,seed,count=support)
        output=torch.zeros(k,len(observed),2,device=observed.device,dtype=observed.dtype);metadata=[]
        for i in valid.nonzero().flatten().tolist():
            decision=select_goals(atoms[:,i].cpu().numpy(),k)
            output[:,i]=atoms[torch.as_tensor(decision['indices'],device=atoms.device),i]
            metadata.append({'agent_id':agent_ids[i],'indices':decision['indices'].tolist(),'risk':decision['distortion_by_prefix'].tolist(),'represented_mass':decision['represented_mass'].tolist()})
        return output,{'support':support,'k':k,'support_mass':'uniform','ordering':'greedy stable support index','agents':metadata,'unit':'m','scales_fitted':bool(self.scales_fitted)}

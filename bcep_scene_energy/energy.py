"""Neural potentials and exact candidate-pair expectations, never geometry of means."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from bcep_v2.joint import neighbors
from bcep_v3.geometry import reference_paths,pair_features


def motion_summary(observed,scale):
    o=observed.detach().float();s=scale.float().clamp_min(.5)
    return torch.cat((o[...,7,:]-o[...,4,:],o[...,7,:]-o[...,0,:],
                      o[...,7,:]-2*o[...,6,:]+o[...,5,:]),-1)/s


def undirected_edges(context):
    b,i,j=neighbors(context.observed,context.valid)
    n=context.valid.shape[1]
    keys=b*n*n+torch.minimum(i,j)*n+torch.maximum(i,j)
    keys=torch.unique(keys,sorted=True)
    return keys//(n*n),(keys//n)%n,keys%n


class CandidateLatentProjector(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(200,64),nn.SiLU(),nn.Linear(64,16))
    def forward(self,u,motion,goals,observed,scale):
        relative=(goals.detach().transpose(1,2)-observed.detach()[:,:,-1,None])/scale.float().clamp_min(.5)
        return self.net(torch.cat((u,motion[:,:,None].expand(-1,-1,20,-1),relative),-1))


class SharedSymmetricRelationPotential(nn.Module):
    def __init__(self,edge_chunk=64):
        super().__init__();self.net=nn.Sequential(nn.Linear(408,64),nn.SiLU(),nn.Linear(64,1))
        self.edge_chunk=edge_chunk;self.activation_checkpointing=True

    def chunk(self,u,motion,paths,observed,scale,b,i,j):
        ui=u[b,i,:,None].expand(-1,-1,20,-1);uj=u[b,j,None,:].expand(-1,20,-1,-1)
        mi=motion[b,i,None,None].expand(-1,20,20,-1);mj=motion[b,j,None,None].expand_as(mi)
        r=pair_features(paths,observed,b,i,j,scale.float().clamp_min(.5))
        reverse=pair_features(paths,observed,b,j,i,scale.float().clamp_min(.5)).transpose(1,2)
        a=self.net(torch.cat((ui,uj,r,mi,mj),-1)).squeeze(-1)
        c=self.net(torch.cat((uj,ui,reverse,mj,mi),-1)).squeeze(-1)
        return .5*(a+c)

    def forward(self,u,motion,context,goals,scale,edges):
        b,i,j=edges;pieces=[]
        # Geometry is built from detached observed history and real candidates.
        paths=reference_paths(context,goals.detach()).detach()
        for start in range(0,len(b),self.edge_chunk):
            sl=slice(start,start+self.edge_chunk)
            args=(u,motion,paths,context.observed.detach(),scale.detach(),b[sl],i[sl],j[sl])
            if self.activation_checkpointing and torch.is_grad_enabled():
                pieces.append(checkpoint(self.chunk,*args,use_reentrant=False,preserve_rng_state=False))
            else:pieces.append(self.chunk(*args))
        return torch.cat(pieces) if pieces else u.new_empty((0,20,20))


@dataclass
class EnergyState:
    z: torch.Tensor               # [N,K,16], one compact valid scene
    tables: torch.Tensor          # [E,K,K], undirected i<j
    edges: torch.Tensor           # [E,2]
    weights: torch.Tensor         # [E], symmetric 1/sqrt(deg_i deg_j)
    incidence: list


class SceneEnergy(nn.Module):
    def __init__(self):
        super().__init__();self.phi=CandidateLatentProjector();self.rho=SharedSymmetricRelationPotential()

    def prepare(self,u,context,goals,scale):
        motion=motion_summary(context.observed,scale)
        z=self.phi(u,motion,goals,context.observed,scale)
        edges=undirected_edges(context);tables=self.rho(u,motion,context,goals,scale,edges)
        # Static discrete incidence goes to CPU once. No GPU atomic reduction.
        edge_rows=torch.stack(edges,1).detach().cpu().tolist()
        states=[];positions=[]
        for b in range(len(context.valid)):
            active=context.valid[b].nonzero().flatten();ids=active.detach().cpu().tolist();lookup={v:k for k,v in enumerate(ids)}
            chosen=[e for e,(be,i,j) in enumerate(edge_rows) if be==b]
            pairs=[(lookup[edge_rows[e][1]],lookup[edge_rows[e][2]]) for e in chosen]
            ix=torch.tensor(pairs,device=u.device,dtype=torch.long).reshape(-1,2)
            degree=[0]*len(ids)
            for i,j in pairs:degree[i]+=1;degree[j]+=1
            weight=u.new_tensor([1/(degree[i]*degree[j])**.5 for i,j in pairs])
            incidence=[[(e,j,False) if i==a else (e,i,True) for e,(i,j) in enumerate(pairs) if a in (i,j)] for a in range(len(ids))]
            states.append(EnergyState(z[b,active],tables[chosen],ix,weight,incidence));positions.append(active)
        return states,positions

    @staticmethod
    def forward(state,p):
        """World energy for hard/ST/soft P [N,K,K]; includes soft second moments."""
        z=state.z;n,k,_=z.shape
        if n==1:return p.sum((0,2))*0
        mu=p@z;second=(p*z.square().sum(-1)[:,None]).sum(-1)
        global_energy=second.sum(0)-(mu.sum(0).square().sum(-1)-mu.square().sum(-1).sum(0))/(n-1)
        if len(state.edges):
            i,j=state.edges.unbind(1)
            local=torch.einsum('ewa,eac,ewc->ew',p[i],state.tables,p[j])
            global_energy=global_energy+(local*state.weights[:,None]).sum(0)
        return global_energy

    @staticmethod
    def complete_pair_formula(state,p):
        """Audit reference: all REAL candidate-pair squared distances before mixing."""
        z=state.z;n,k,_=z.shape;out=z.new_zeros(k)
        for i in range(n):
            for j in range(i+1,n):
                cost=(z[i,:,None]-z[j,None,:]).square().sum(-1)
                out=out+torch.einsum('wa,ac,wc->w',p[i],cost,p[j])/(n-1)
        for e,(i,j) in enumerate(state.edges.tolist()):
            out=out+state.weights[e]*torch.einsum('wa,ac,wc->w',p[i],state.tables[e],p[j])
        return out


class ConditionalReplacementScorer(nn.Module):
    def forward(self,state,p,i):
        z=state.z;n=len(z);mu=p@z
        if n==1:return p[i]*0
        other=mu.sum(0)-mu[i]
        # Terms constant in the candidate can be omitted from costs, then the
        # current expected row cost is subtracted. The result is exact delta E.
        cost=z[i].square().sum(-1)[None]-2/(n-1)*(other@z[i].T)
        if state.incidence[i]:
            entries=torch.tensor(state.incidence[i],device=z.device,dtype=torch.long)
            e,j,transpose=entries.unbind(1)
            tables=state.tables[e]
            oriented=torch.where(transpose[:,None,None].bool(),tables.transpose(1,2),tables)
            messages=torch.bmm(p[j],oriented.transpose(1,2))
            cost=cost+(state.weights[e,None,None]*messages).sum(0)
        return -(cost-(p[i]*cost).sum(-1,keepdim=True))

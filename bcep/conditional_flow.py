"""Four alternating conditional spline couplings. Generative direction z -> r."""
import math
import torch
from torch import nn
from .vendor.rational_quadratic import unconstrained_rational_quadratic_spline

class ConditionalFlow(nn.Module):
    def __init__(self,dim=128,bins=8):
        super().__init__();self.bins=bins
        self.nets=nn.ModuleList([nn.Sequential(nn.Linear(dim+2,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,3*bins-1)) for _ in range(4)])
        for net in self.nets:
            nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias)
            with torch.no_grad():net[-1].bias[2*bins:]=math.log(math.expm1(1-.001))
    def transform(self,x,context,horizon_seconds,inverse=False):
        logdet=torch.zeros_like(x[...,0]);h=(horizon_seconds.to(x).expand_as(x[...,:1]) if torch.is_tensor(horizon_seconds) else torch.full_like(x[...,:1],horizon_seconds))
        for i in (reversed(range(4)) if inverse else range(4)):
            fixed=i%2;changed=1-fixed
            params=self.nets[i](torch.cat((context,x[...,fixed:fixed+1],h),-1))
            y,ld=unconstrained_rational_quadratic_spline(x[...,changed],params[...,:self.bins],params[...,self.bins:2*self.bins],params[...,2*self.bins:],inverse=inverse,tails='linear',tail_bound=5.)
            x=torch.stack((x[...,0],y),-1) if changed==1 else torch.stack((y,x[...,1]),-1)
            logdet=logdet+ld
        return x,logdet
    def log_prob(self,residual,context,horizon_seconds):
        z,ld=self.transform(residual,context,horizon_seconds,inverse=True)
        return -.5*z.square().sum(-1)-math.log(2*math.pi)+ld

    def forward_inverse_consistency(self,x,context,horizon_seconds):
        y,ld=self.transform(x,context,horizon_seconds)
        z,il=self.transform(y,context,horizon_seconds,inverse=True)
        return {'max_inverse_error':(z-x).abs().max(),'max_logdet_error':(ld+il).abs().max()}

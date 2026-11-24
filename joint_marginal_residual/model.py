"""Small zero-initialized post-bridge correction; one shared 64-wide trunk."""
import torch
from torch import nn
from .features import BASE_DIM, DIFFERENCE_DIM


class ResidualHead(nn.Module):
    def __init__(self, base_dim=BASE_DIM, difference_dim=DIFFERENCE_DIM, width=64):
        super().__init__()
        if width != 64:raise ValueError('the fixed recipe requires width 64')
        self.shared_mlp = nn.Sequential(nn.Linear(base_dim,width), nn.SiLU(), nn.Linear(width,width), nn.SiLU())
        self.common_readout = nn.Sequential(nn.Linear(width,width), nn.SiLU(), nn.Linear(width,4))
        self.person_readout = nn.Sequential(nn.Linear(2*width+difference_dim,width), nn.SiLU(), nn.Linear(width,4,bias=False))
        nn.init.zeros_(self.common_readout[-1].weight)
        nn.init.zeros_(self.common_readout[-1].bias)
        nn.init.zeros_(self.person_readout[-1].weight)

    def forward(self, teacher, base_features, difference_features, valid, *, ablation=False, return_details=False):
        """One window: all person and scenario operations are permutation equivariant.

        P=J,a=0 is implemented only by zeroing the dedicated difference inputs;
        architecture, parameters and initialization are identical in the control.
        """
        teacher = teacher.detach()
        base_features = base_features.detach()
        difference_features = difference_features.detach()
        valid = valid.bool()
        if teacher.ndim != 4 or teacher.shape[-2:] != (12,2):raise ValueError('teacher must be [K,N,12,2]')
        k,n = teacher.shape[:2]
        if base_features.shape[:2] != (k,n) or difference_features.shape[:2] != (k,n) or valid.shape != (n,):
            raise ValueError('feature/mask axes do not match teacher')
        if not valid.any():raise ValueError('empty window')
        if teacher.dtype != torch.float32 or base_features.dtype != torch.float32 or difference_features.dtype != torch.float32:
            raise ValueError('the fixed head recipe is FP32')
        mask = valid[None,:,None]
        h = self.shared_mlp(torch.where(mask,base_features,0.))
        scene = (h*mask).sum(1)/valid.sum()
        common = self.common_readout(scene)
        dedicated = torch.zeros_like(difference_features) if ablation else difference_features
        v = self.person_readout(torch.cat((h,scene[:,None].expand(-1,n,-1),torch.where(mask,dedicated,0.)), -1))
        rel = torch.where(mask,v-(v*mask).sum(1,keepdim=True)/valid.sum(),0.)
        offsets = torch.where(mask,common[:,None]+rel,0.)
        dg,dm = offsets.split(2,-1)
        u = torch.arange(1,13,device=teacher.device,dtype=teacher.dtype)/12.
        correction = u[None,None,:,None]*dg[:,:,None] + (4*u*(1-u))[None,None,:,None]*dm[:,:,None]
        result = teacher + correction
        if return_details:
            return result,dict(common=common,relative=rel,dg=dg,dm=dm,correction=correction)
        return result

    @property
    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

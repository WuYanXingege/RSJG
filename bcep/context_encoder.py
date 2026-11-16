import torch
from torch import nn

class ContextEncoder(nn.Module):
    def __init__(self,dim=128):
        super().__init__()
        self.input=nn.Linear(5,dim)
        layer=nn.TransformerEncoderLayer(dim,4,256,dropout=0.,batch_first=True)
        self.temporal=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.social=nn.MultiheadAttention(dim,4,dropout=0.,batch_first=True)
        self.map=nn.Sequential(nn.Conv2d(6,32,3,2,1),nn.SiLU(),nn.Conv2d(32,64,3,2,1),nn.SiLU(),nn.Conv2d(64,128,3,2,1),nn.SiLU(),nn.AdaptiveAvgPool2d(1))
        self.missing_map=nn.Parameter(torch.zeros(dim))
        self.fuse=nn.Sequential(nn.Linear(3*dim,dim),nn.LayerNorm(dim),nn.SiLU())
    def forward(self,observed,semantic_maps,valid,dt):
        if observed.ndim!=3 or valid.shape!=observed.shape[:1] or not valid.any():raise ValueError('one nonempty scene required')
        if not torch.isfinite(observed[valid]).all():raise ValueError('nonfinite observation')
        x=torch.where(valid[:,None,None],observed,torch.zeros_like(observed))
        relative=x-x[:,-1:];velocity=torch.cat((torch.zeros_like(x[:,:1]),x[:,1:]-x[:,:-1]),1)/dt
        times=torch.arange(x.shape[1],device=x.device,dtype=x.dtype)[None,:,None].expand(len(x),-1,-1)*dt
        h=self.temporal(self.input(torch.cat((relative,velocity,times),-1)))[:,-1]
        social,_=self.social(h[None],h[None],h[None],key_padding_mask=~valid[None],need_weights=False)
        if semantic_maps is None:m=self.missing_map[None].expand_as(h)
        else:
            if semantic_maps.shape[:2]!=(len(x),6):raise ValueError('expected six semantic channels')
            if not torch.isfinite(semantic_maps[valid]).all():raise ValueError('nonfinite semantic input')
            maps=torch.where(valid[:,None,None,None],semantic_maps.to(x),torch.zeros_like(semantic_maps.to(x)))
            m=self.map(maps).flatten(1)
        return self.fuse(torch.cat((h,social[0],m),-1)).masked_fill(~valid[:,None],0)

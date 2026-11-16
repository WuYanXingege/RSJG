import torch
from torch import nn

class ContextEncoder(nn.Module):
    def __init__(self,dim=128):
        super().__init__()
        self.input=nn.Linear(5,dim)
        layer=nn.TransformerEncoderLayer(dim,4,256,dropout=0.,batch_first=True)
        self.temporal=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.social=nn.MultiheadAttention(dim,4,dropout=0.,batch_first=True)
        # Fixed SI normalization: position / 1m, velocity / (1m/s), distance^2 / 1m^2.
        self.edge=nn.Sequential(nn.Linear(5,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU())
        self.edge_bias=nn.Linear(32,4)
        self.edge_value=nn.Linear(32,dim)
        self.map=nn.Sequential(nn.Conv2d(6,32,3,2,1),nn.SiLU(),nn.Conv2d(32,64,3,2,1),nn.SiLU(),nn.Conv2d(64,128,3,2,1),nn.SiLU(),nn.AdaptiveAvgPool2d(1))
        self.missing_map=nn.Parameter(torch.zeros(dim))
        self.fuse=nn.Sequential(nn.Linear(3*dim,dim),nn.LayerNorm(dim),nn.SiLU())
    def social_aggregate(self,h,x,valid,dt):
        # x has already had padding/NaN removed. Keep self for singleton scenes.
        n,dim=h.shape;heads=4;depth=dim//heads
        velocity=(x[:,1:]-x[:,:-1])[:,-3:].mean(1)/dt
        dp=x[None,:,-1]-x[:,None,-1]
        dv=velocity[None]-velocity[:,None]
        edge=self.edge(torch.cat((dp,dv,dp.square().sum(-1,keepdim=True)),-1))
        q,k,v=torch.nn.functional.linear(h,self.social.in_proj_weight,self.social.in_proj_bias).chunk(3,-1)
        q,k,v=[t.reshape(n,heads,depth) for t in (q,k,v)]
        logits=torch.einsum('ihd,jhd->hij',q,k)/(depth**.5)+self.edge_bias(edge).permute(2,0,1)
        alpha=logits.masked_fill(~valid[None,None,:],float('-inf')).softmax(-1)
        message=torch.einsum('hij,jhd->ihd',alpha,v)+torch.einsum('hij,ijhd->ihd',alpha,self.edge_value(edge).reshape(n,n,heads,depth))
        return self.social.out_proj(message.reshape(n,dim)).masked_fill(~valid[:,None],0)
    def forward(self,observed,semantic_maps,valid,dt):
        if observed.ndim!=3 or valid.shape!=observed.shape[:1] or not valid.any():raise ValueError('one nonempty scene required')
        if not torch.isfinite(observed[valid]).all():raise ValueError('nonfinite observation')
        x=torch.where(valid[:,None,None],observed,torch.zeros_like(observed))
        relative=x-x[:,-1:];velocity=torch.cat((torch.zeros_like(x[:,:1]),x[:,1:]-x[:,:-1]),1)/dt
        times=torch.arange(x.shape[1],device=x.device,dtype=x.dtype)[None,:,None].expand(len(x),-1,-1)*dt
        h=self.temporal(self.input(torch.cat((relative,velocity,times),-1)))[:,-1]
        social=self.social_aggregate(h,x,valid,dt)
        if semantic_maps is None:m=self.missing_map[None].expand_as(h)
        else:
            if semantic_maps.shape[:2]!=(len(x),6):raise ValueError('expected six semantic channels')
            if not torch.isfinite(semantic_maps[valid]).all():raise ValueError('nonfinite semantic input')
            maps=torch.where(valid[:,None,None,None],semantic_maps.to(x),torch.zeros_like(semantic_maps.to(x)))
            m=self.map(maps).flatten(1)
        return self.fuse(torch.cat((h,social,m),-1)).masked_fill(~valid[:,None],0)

"""Observed-only invertible motion frame, with world density Jacobian."""
import torch

def frame(observed, horizon, dt, scale):
    if observed.shape[-2]<2 or dt<=0 or horizon<1:raise ValueError('invalid temporal protocol')
    velocity=(observed[...,1:,:]-observed[...,:-1,:])[...,-3:,:].mean(-2)/dt
    speed=velocity.norm(dim=-1,keepdim=True)
    unit=velocity/speed.clamp_min(1e-8)
    axis=torch.zeros_like(unit);axis[...,0]=1
    unit=torch.where(speed>1e-3,unit,axis)
    rotation=torch.stack((unit,torch.stack((-unit[...,1],unit[...,0]),-1)),-1)
    baseline=observed[...,-1,:]+horizon*dt*velocity
    if not torch.isfinite(scale).all() or (scale<=0).any():raise ValueError('scale must be finite positive')
    return baseline,rotation,scale

def to_world(local, baseline, rotation, scale):
    return baseline+torch.matmul(rotation,(local*scale).unsqueeze(-1)).squeeze(-1)

def to_local(world, baseline, rotation, scale):
    return torch.matmul(rotation.transpose(-1,-2),(world-baseline).unsqueeze(-1)).squeeze(-1)/scale

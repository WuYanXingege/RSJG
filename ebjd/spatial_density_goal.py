"""Full-support spatial endpoint density with explicit in-/out-crop domains.

Inside: learned masses over map-aligned, rectangle-truncated Gaussian kernels.
Outside: four non-overlapping square-polar sectors, exponential radial excess,
and uniform tangential coordinate. Their Jacobian is r, hence density 1/(2r).
A learned gate assigns total mass to each domain. The outside branch cannot
explain in-crop labels. No labels/candidates are clipped, discarded or GT-picked.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .unified import UnifiedCouplingBridge

SPATIAL_SCHEMA = "ebjd-spatial-density-bridge-v1"
SPATIAL_OBJECTIVE = "density_nll_and_detached_sample_pairing"
SPATIAL_GOAL_OPTIONS = dict(components=20, grid_size=64, kernel_sigma_cells=0.5,
                            tail_min_scale=0.05, initial_tail_mass=0.1)


@dataclass
class SpatialDensity:
    spatial_logits: torch.Tensor
    tail_logit: torch.Tensor
    side_logits: torch.Tensor
    tail_scale: torch.Tensor


class SpatialGoalDensity(nn.Module):
    noise_dimension = 3
    proposal_source = "online_domain_separated_spatial_density_iid20"

    def __init__(self, dim=192, grid_size=64, kernel_sigma_cells=0.5,
                 tail_min_scale=0.05, initial_tail_mass=0.1, components=20):
        super().__init__()
        if (grid_size < 2 or kernel_sigma_cells <= 0 or tail_min_scale <= 0
                or not 0 < initial_tail_mass < 1 or components != 20):
            raise ValueError("invalid spatial density configuration")
        self.grid_size = grid_size
        self.kernel_sigma_cells = float(kernel_sigma_cells)
        self.tail_min_scale = float(tail_min_scale)
        self.components = components
        self.spatial_features = nn.Conv2d(34, 32, 3, padding=1)
        self.social_film = nn.Linear(dim, 64)
        self.spatial_readout = nn.Conv2d(32, 1, 1)
        self.tail = nn.Sequential(nn.LayerNorm(dim + 32), nn.Linear(dim + 32, dim),
                                  nn.SiLU(), nn.Linear(dim, 9))
        nn.init.normal_(self.tail[-1].weight, std=0.001)
        nn.init.zeros_(self.tail[-1].bias)
        with torch.no_grad():
            self.tail[-1].bias[0] = math.log(initial_tail_mass / (1 - initial_tail_mass))
            self.tail[-1].bias[5:] = math.log(math.expm1(1.0))
        axis = torch.linspace(-1, 1, grid_size)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("grid", torch.stack((xx, yy), -1).flatten(0, 1), persistent=False)

    def forward(self, context):
        with torch.autocast(device_type=context.social_h.device.type, enabled=False):
            b, n = context.valid.shape
            features = context.map_features.float().reshape(b * n, 32, *context.map_features.shape[-2:])
            features = F.interpolate(features, size=(self.grid_size, self.grid_size),
                                     mode="bilinear", align_corners=True)
            coordinates = self.grid.float().T.reshape(1, 2, self.grid_size, self.grid_size).expand(b*n, -1, -1, -1)
            spatial = self.spatial_features(torch.cat((features, coordinates), 1))
            scale, shift = self.social_film(context.social_h.float()).reshape(b*n, 64).chunk(2, -1)
            spatial = F.silu(spatial * (1 + scale[..., None, None]) + shift[..., None, None])
            logits = self.spatial_readout(spatial).reshape(b, n, -1)
            pooled = features.mean((-2, -1)).reshape(b, n, 32)
            tail = self.tail(torch.cat((context.social_h.float(), pooled), -1))
            return SpatialDensity(logits, tail[..., 0], tail[..., 1:5],
                                  F.softplus(tail[..., 5:]) + self.tail_min_scale)

    @staticmethod
    def log_masses(distribution):
        local = F.logsigmoid(-distribution.tail_logit)[..., None] + F.log_softmax(distribution.spatial_logits, -1)
        exterior = F.logsigmoid(distribution.tail_logit)[..., None] + F.log_softmax(distribution.side_logits, -1)
        return torch.cat((local, exterior), -1)

    def world_centres(self, context):
        return context.map_center.float()[..., None, :] + self.grid.float() * (context.crop_width_m / 2)

    def kernel_sigma_m(self, context):
        return context.crop_width_m / (self.grid_size - 1) * self.kernel_sigma_cells

    def kernel_cdf_bounds(self, context):
        h = context.crop_width_m / 2
        means = self.grid.float() * h
        sigma = self.kernel_sigma_m(context)
        return torch.special.ndtr((-h-means)/sigma), torch.special.ndtr((h-means)/sigma)

    @staticmethod
    def exterior_side(relative):
        # Disjoint sectors: +x,-x,+y,-y. Ties assigned to x (measure-zero edge).
        x, y = relative.unbind(-1)
        return torch.where(x.abs() >= y.abs(), (x < 0).long(), 2+(y < 0).long())

    def log_prob(self, context, truth, goal_scale, *, distribution=None):
        """Normalized continuous density; returned in train-scale coordinates.

        Density in physical coordinates is multiplied by goal_scale**2. That
        Jacobian matches the previous normalized NLL convention. Sampling and
        coverage still use world metres. There is no nearest-cell GT clipping.
        """
        d = self(context) if distribution is None else distribution
        with torch.autocast(device_type=truth.device.type, enabled=False):
            truth = torch.where(context.valid[..., None], truth.float(), context.map_center.float())
            relative = truth-context.map_center.float()
            radius = relative.abs().amax(-1)
            half = context.crop_width_m/2
            inside = radius <= half
            sigma = torch.as_tensor(self.kernel_sigma_m(context), device=truth.device)
            lo, hi = self.kernel_cdf_bounds(context)
            log_z = (hi-lo).log().sum(-1)
            kernels = (-0.5*((truth[..., None, :]-self.world_centres(context))/sigma).square()
                       - sigma.log()-0.5*math.log(2*math.pi)).sum(-1)-log_z
            local = torch.logsumexp(F.log_softmax(d.spatial_logits, -1)+kernels, -1)
            side = self.exterior_side(relative)
            scale = (d.tail_scale*goal_scale.float()).gather(-1,side[..., None]).squeeze(-1)
            side_logmass = F.log_softmax(d.side_logits,-1).gather(-1,side[...,None]).squeeze(-1)
            # Safe dummy values for the inactive branch; no modification of GT.
            r = radius.clamp_min(half)
            exterior = side_logmass-(r-half)/scale-scale.log()-torch.log(2*r)
            log_world = torch.where(inside, F.logsigmoid(-d.tail_logit)+local,
                                     F.logsigmoid(d.tail_logit)+exterior)
            return log_world+2*goal_scale.float().log()

    def nll(self, context, truth, goal_scale, *, distribution=None):
        return -self.log_prob(context, truth, goal_scale, distribution=distribution)[context.valid].mean()

    @torch.no_grad()
    def diagnostics(self, context, truth, goal_scale, *, distribution=None):
        d = self(context) if distribution is None else distribution
        with torch.autocast(device_type=truth.device.type, enabled=False):
            logp = F.log_softmax(d.spatial_logits,-1)
            entropy = -(logp.exp()*logp).sum(-1)/math.log(self.grid_size**2)
            outside = ((truth.float()-context.map_center.float()).abs()>context.crop_width_m/2).any(-1)
            valid=context.valid
            return dict(spatial_tail_mass=d.tail_logit.sigmoid()[valid].mean(),
                spatial_tail_responsibility=outside[valid].float().mean(),
                spatial_entropy_fraction=entropy[valid].mean(),
                spatial_outside_crop_fraction=outside[valid].float().mean())

    @torch.no_grad()
    def sample(self, context, goal_scale, *, generator=None, noise=None, distribution=None):
        """20 IID samples; normals transformed into three independent uniforms.

        Inverse-CDF draws implement the SAME normalized density as log_prob.
        Local normal kernels are truncated at the map border, not clamped after
        sampling. Outside draws use the square-polar Jacobian described above.
        Sampling never sees future coordinates or differentiates through argmin.
        """
        d = self(context) if distribution is None else distribution
        expected = (*context.valid.shape,self.components,self.noise_dimension)
        if noise is None:
            noise=torch.randn(expected,device=context.social_h.device,dtype=torch.float32,generator=generator)
        if tuple(noise.shape)!=expected:raise ValueError(f"endpoint noise must be {expected}")
        epsilon=torch.finfo(torch.float32).eps
        uniforms=torch.special.ndtr(noise.float()).clamp(epsilon,1-epsilon)
        cdf=self.log_masses(d).exp().cumsum(-1);cdf[..., -1]=1.
        cells=self.grid_size**2
        index=torch.searchsorted(cdf.contiguous(),uniforms[...,0].contiguous()).clamp_max(cells+3)
        local_index=index.clamp_max(cells-1)
        centres=self.world_centres(context).gather(-2,local_index[...,None].expand(-1,-1,-1,2))
        lo,hi=self.kernel_cdf_bounds(context)
        probabilities=lo[local_index]+(hi-lo)[local_index]*uniforms[...,1:]
        local=centres+self.kernel_sigma_m(context)*torch.special.ndtri(probabilities.clamp(epsilon,1-epsilon))
        side=(index-cells).clamp_min(0)
        scale=(d.tail_scale*goal_scale.float()).gather(-1,side)
        excess=-scale*torch.log1p(-uniforms[...,1])
        radius=context.crop_width_m/2+excess
        tangent=2*uniforms[...,2]-1
        sign=torch.where(side.remainder(2)==0,1.,-1.)
        x=torch.where(side<2,sign*radius,tangent*radius)
        y=torch.where(side<2,tangent*radius,sign*radius)
        exterior=context.map_center.float()[...,None,:]+torch.stack((x,y),-1)
        goals=torch.where((index>=cells)[...,None],exterior,local)
        return goals.permute(0,2,1,3).masked_fill(~context.valid[:,None,:,None],0)


class SpatialDensityCouplingBridge(UnifiedCouplingBridge):
    def __init__(self, bridge, **goal_options):
        super().__init__(bridge,worlds=20,singleton_backward="identity")
        self.goal=SpatialGoalDensity(bridge.encoder.history_stem[0].out_features,**goal_options)

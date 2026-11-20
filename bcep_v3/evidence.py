"""Exact K20/observed evidence only. No inferred cluster probabilities."""
import torch
from torch import nn


def require_tensor(condition, message):
    if condition.device.type == 'cuda':
        torch._assert_async(condition, message)
    elif not condition:
        raise ValueError(message)


def validate_inputs(context, goals):
    b, n = context.valid.shape
    if goals.shape != (b, 20, n, 2) or context.observed.shape != (b, n, 8, 2):
        raise ValueError('expected world-metre K20 / 8 observed frames')
    if context.valid.dtype != torch.bool:
        raise ValueError('nonempty boolean valid mask required')
    require_tensor(context.valid.any(1).all(), 'nonempty boolean valid mask required')
    require_tensor((torch.isfinite(goals) | ~context.valid[:, None, :, None]).all(), 'nonfinite valid candidate')
    require_tensor((torch.isfinite(context.observed) | ~context.valid[..., None, None]).all(), 'nonfinite observed history')


class CandidateSetEvidence(nn.Module):
    """One relative-geometry-biased attention, candidate permutation equivariant."""
    def __init__(self, dim=192, heads=4):
        super().__init__()
        self.heads = heads
        # displacement from observed / baseline, mean and RMS pair distance;
        # three absent statistics and three explicit presence masks.
        self.evidence = nn.Linear(12, dim)
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.bias = nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, heads))
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, old, context, goals, scale, optional_evidence=None):
        if optional_evidence is not None:
            raise ValueError('support-pool provenance unavailable: nonempty sidecar disabled')
        b, n, k, d = old.shape
        scale = scale.float().clamp_min(0.5)
        g = goals.transpose(1, 2).float()
        rel = (g[:, :, :, None] - g[:, :, None, :]) / scale
        dist = rel.square().sum(-1).add(1e-8).sqrt()
        obs = (g - context.observed[:, :, -1, None]) / scale
        base = (g - context.baseline[:, :, -1, None]) / scale
        derived = torch.cat((obs, base, dist.mean(-1, keepdim=True),
                             dist.square().mean(-1, keepdim=True).sqrt()), -1)
        missing = torch.zeros_like(derived)  # values=0, presence=0; never confidence=1
        x = self.norm(old + self.evidence(torch.cat((derived, missing), -1)))
        q, key, value = self.qkv(x).reshape(b, n, k, 3, self.heads, d // self.heads).unbind(3)
        q, key, value = [v.transpose(2, 3) for v in (q, key, value)]
        bias = self.bias(torch.cat((rel, dist[..., None]), -1)).permute(0, 1, 4, 2, 3)
        weight = ((q @ key.transpose(-1, -2)) / (d // self.heads)**0.5 + bias).softmax(-1)
        out = (weight @ value).transpose(2, 3).reshape(b, n, k, d)
        return old + self.out(out) * context.valid[..., None, None]

"""Standard full-covariance MDN control, not a new proposed model component."""
import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from bcep.context_encoder import ContextEncoder
from bcep.motion_frame import frame, to_local, to_world
from bcep.provider import BCEP


class MixtureHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(129, 128), nn.SiLU(),
                                 nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 48))
        # Default Linear random weights AND biases break component symmetry.
        # No data-dependent initialization, extra regularizer, or scale tuning.

    def parameters_at(self, context, horizon_seconds):
        horizon = torch.full_like(context[..., :1], horizon_seconds)
        raw = self.net(torch.cat((context, horizon), -1)).reshape(*context.shape[:-1], 8, 6)
        logits, mean = raw[..., 0], raw[..., 1:3]
        diagonal = F.softplus(torch.stack((raw[..., 3], raw[..., 5]), -1)) + .001
        zero = torch.zeros_like(raw[..., 4])
        lower = torch.stack((diagonal[..., 0], zero, raw[..., 4], diagonal[..., 1]), -1)
        return logits, mean, lower.reshape(*mean.shape[:-1], 2, 2)

    def log_prob(self, residual, context, horizon_seconds):
        logits, mean, lower = self.parameters_at(context, horizon_seconds)
        delta = (residual[..., None, :] - mean).unsqueeze(-1)
        white = torch.linalg.solve_triangular(lower, delta, upper=False).squeeze(-1)
        component = (-.5 * white.square().sum(-1) - math.log(2 * math.pi)
                     - lower.diagonal(dim1=-2, dim2=-1).log().sum(-1))
        return torch.logsumexp(F.log_softmax(logits, -1) + component, -1)


class MDNProvider(BCEP):
    """Reuse observed encoder, frame, supervision and selector, no flow instance."""
    def __init__(self, pred_length=12, dt=.4):
        nn.Module.__init__(self)
        self.pred_length, self.dt = pred_length, dt
        # First constructor is identical to BCEP: same CPU seed -> same encoder.
        self.encoder = ContextEncoder()
        self.mdn = MixtureHead()
        self.register_buffer('scales', torch.ones(pred_length, 2))
        self.register_buffer('scales_fitted', torch.tensor(False))

    def log_prob_world(self, world, observed, context, horizon):
        baseline, rotation, scale = frame(observed, horizon, self.dt, self.scales[horizon-1])
        residual = to_local(world, baseline, rotation, scale)
        return self.mdn.log_prob(residual, context, horizon*self.dt) - scale.log().sum(-1)

    @torch.no_grad()
    def sample_support(self, observed, semantic_maps, valid, scene_id, agent_ids, seed, count=256):
        if type(count) is not int or count < 1 or len(agent_ids) != len(observed):
            raise ValueError('invalid support count or identity count')
        active = valid.nonzero().flatten().tolist()
        if len({agent_ids[i] for i in active}) != len(active):
            raise ValueError('duplicate valid agent identity')
        context = self.encoder(observed, semantic_maps, valid, self.dt)
        logits, means, lower = self.mdn.parameters_at(context, self.pred_length*self.dt)
        baseline, rotation, scale = frame(observed, self.pred_length, self.dt, self.scales[-1])
        samples = torch.zeros(count, len(observed), 2, device=observed.device, dtype=observed.dtype)
        for i in active:
            key = json.dumps([str(scene_id), str(agent_ids[i]), int(seed)], separators=(',', ':')).encode()
            local_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % (2**63-1)
            generator = torch.Generator(device='cpu').manual_seed(local_seed)
            component = torch.multinomial(logits[i].softmax(-1).cpu(), count, replacement=True,
                                          generator=generator).to(observed.device)
            noise = torch.randn(count, 2, generator=generator, dtype=observed.dtype).to(observed.device)
            residual = means[i, component] + (lower[i, component] @ noise.unsqueeze(-1)).squeeze(-1)
            samples[:, i] = to_world(residual, baseline[i], rotation[i], scale)
        if not torch.isfinite(samples[:, valid]).all():
            raise ValueError('nonfinite MDN support')
        return samples, torch.full((count,), 1/count, device=observed.device, dtype=observed.dtype)

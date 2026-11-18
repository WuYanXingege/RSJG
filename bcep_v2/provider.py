"""Observed-only M256/KMeans20 candidates and bounded spatial corrections."""

import torch
from torch import nn
from torch.nn import functional as F

from bcep.motion_frame import frame, to_local, to_world
from bcep.provider import BCEP
from selector_control.algorithms import kmeans_centers
from selector_control.core import local_seed

GEOMETRY = dict(
    schema="bcep-v2-geometry-v1",
    width_m=32.0,
    pixels=256,
    center="last_observed",
    axes="world_x_columns_y_rows",
    crop_align_corners=True,
    stride=8,
    offset=0,
    feature_align_corners=False,
    outside="six_zero_channels",
)


def spatial_encoding(base, observed, maps, valid):
    """Read the actual pool input of the original encoder within this call."""
    features = []

    def capture(pool, args):
        features.append(args[0])

    hook = base.encoder.map[-1].register_forward_pre_hook(capture)
    try:
        context = base.encoder(observed, maps, valid, base.dt)
    finally:
        hook.remove()
    return context, features[0] if maps is not None else None


def verify_cnn_geometry(encoder):
    stride, offset = 1, 0.0
    for layer in encoder.map:
        if isinstance(layer, nn.Conv2d):
            assert all(
                param[0] == param[1]
                for param in (layer.kernel_size, layer.stride, layer.padding, layer.dilation)
            )
            offset += stride * ((layer.kernel_size[0] - 1) * layer.dilation[0] / 2 - layer.padding[0])
            stride *= layer.stride[0]
    if (stride, offset) != (8, 0):
        raise ValueError("unsupported feature sampling geometry")


def query_map(features, maps, goals, centers, geometry):
    if geometry is None or geometry.get("layout") != GEOMETRY:
        raise ValueError("map crop requires verified geometry")
    if maps.shape[1:] != (6, 256, 256) or features.shape[1:] != (128, 32, 32):
        raise ValueError("crop/feature geometry mismatch")
    pixel = (goals.float() - centers[:, None].float() + 16.0) * (255.0 / 32.0)
    feature = pixel / 8.0
    grid = (2 * (feature + 0.5) / 32.0 - 1).unsqueeze(2)
    sampled = F.grid_sample(features.float(), grid, align_corners=False).squeeze(-1).transpose(1, 2)
    coverage = F.grid_sample(
        maps.float().sum(1, keepdim=True),
        (2 * pixel / 255.0 - 1).unsqueeze(2),
        mode="nearest",
        align_corners=True,
    )[:, 0, :, 0]
    inside = ((feature >= 0) & (feature <= 31)).all(-1) & (coverage > 0)
    return sampled, inside


class CandidateHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.missing = nn.Parameter(torch.zeros(128))
        self.embedding = nn.Sequential(nn.Linear(259, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU())
        self.readout = nn.Sequential(nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 2))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, base, observed, maps, valid, goals, context, features, geometry):
        if goals.shape != (20, len(observed), 2) or not valid.any():
            raise ValueError("expected [20,N,2] and a nonempty scene")
        goals = goals.detach().transpose(0, 1)
        safe = torch.where(valid[:, None, None], goals, 0.0)
        origin, rotation, scale = frame(
            torch.where(valid[:, None, None], observed, 0.0), 12, base.dt, base.scales[-1]
        )
        local = to_local(safe.float(), origin[:, None], rotation[:, None], scale)
        if maps is None:
            sampled = self.missing.expand(len(valid), 20, -1)
            inside = torch.zeros_like(local[..., 0], dtype=torch.bool)
        else:
            sampled, inside = query_map(features, maps, safe, observed[:, -1], geometry)
        embedding = self.embedding(
            torch.cat(
                (
                    context[:, None].expand(-1, 20, -1),
                    torch.where(inside[..., None], sampled, self.missing),
                    local,
                    inside[..., None].float(),
                ),
                -1,
            )
        )
        raw = self.readout(
            torch.cat((embedding, embedding.mean(1, keepdim=True).expand_as(embedding)), -1)
        )
        shift = 0.5 * raw / (1 + raw.norm(dim=-1, keepdim=True))
        world_shift = (rotation[:, None] @ shift.unsqueeze(-1)).squeeze(-1)
        world_shift = torch.where(valid[:, None, None], world_shift, 0.0)
        return (safe + world_shift.to(safe)).transpose(0, 1), world_shift


def head_loss_components(goals, shift, truth, valid):
    if truth.shape != (len(valid), 2) or goals.shape != (20, len(valid), 2):
        raise ValueError("endpoint layout mismatch")
    distance = (goals[:, valid].float() - truth[valid].detach()[None]).norm(dim=-1).amin(0)
    movement = shift[valid].square().sum(-1).mean(1)
    return (distance + 0.1 * movement).sum(), distance.sum(), movement.detach().sum()


def head_sums(goals, shift, truth, valid):
    return head_loss_components(goals, shift, truth, valid)[:2]


class Provider(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = BCEP()
        verify_cnn_geometry(self.base.encoder)
        self.head = CandidateHead()

    @torch.no_grad()
    def candidates(self, observed, maps, valid, scene_id, agent_ids, seed, geometry, *, refined=True):
        import hashlib, json

        context, features = spatial_encoding(self.base, observed, maps, valid)
        origin, rot, scale = frame(observed, 12, self.base.dt, self.base.scales[-1])
        goals = observed.new_zeros(20, len(valid), 2)
        active = valid.nonzero().flatten().tolist()
        if len(agent_ids) != len(valid) or len({agent_ids[i] for i in active}) != len(active):
            raise ValueError("invalid agent identity")
        for i in active:
            key = json.dumps(
                [str(scene_id), str(agent_ids[i]), int(seed)], separators=(",", ":")
            ).encode()
            rng = torch.Generator().manual_seed(
                int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**63 - 1)
            )
            noise = torch.randn(256, 2, generator=rng, dtype=observed.dtype).to(observed.device)
            residual, _ = self.base.flow.transform(noise, context[i].expand(256, -1), 12 * self.base.dt)
            atoms = to_world(residual, origin[i], rot[i], scale)
            centers = kmeans_centers(
                atoms.cpu().numpy(),
                20,
                local_seed("bcep_flow", scene_id, agent_ids[i], seed),
                n_init=5,
                max_iter=100,
                tolerance_m=1e-6,
            )["centers"]
            goals[:, i] = torch.as_tensor(centers).to(goals)
        if not refined:
            return goals
        return self.head(self.base, observed, maps, valid, goals, context, features, geometry)[0]

"""Observed-only shared graph scorer; all pooling respects valid participants."""
import torch
from torch import nn
from bcep_v2.joint import neighbors


def raw_features(observed, social, goals, valid):
    """B,K,N nodes and B,K,E edges, no identity, label or future input."""
    last = observed[:, :, -1]
    velocity = (last - observed[:, :, -4]) / 1.2
    history = (observed - last[:, :, None]).flatten(-2)
    common = torch.cat((history, velocity, social), -1)
    node = torch.cat((common[:, None].expand(-1, 20, -1, -1), goals-last[:, None]), -1)
    b, i, j = neighbors(observed, valid)
    relative = torch.cat((last[b, j]-last[b, i], velocity[b, j]-velocity[b, i]), -1)
    edge = torch.cat((relative[:, None].expand(-1, 20, -1), goals[b, :, j]-goals[b, :, i]), -1)
    return node, edge, (b, i, j)


def masked_pool(x, mask, dim):
    mask = mask[..., None]
    mean = x.masked_fill(~mask, 0).sum(dim) / mask.sum(dim).clamp_min(1)
    maximum = x.masked_fill(~mask, -torch.inf).amax(dim)
    maximum = torch.where(mask.any(dim), maximum, 0)
    return torch.cat((mean, maximum), -1)


class SceneScorer(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.register_buffer('node_mean', torch.zeros(212))
        self.register_buffer('node_std', torch.ones(212))
        self.register_buffer('edge_mean', torch.zeros(6))
        self.register_buffer('edge_std', torch.ones(6))
        self.node = nn.Sequential(nn.Linear(212, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.edge = nn.Sequential(nn.Linear(2*width+6, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.agent = nn.Sequential(nn.Linear(3*width, width), nn.SiLU())
        self.world = nn.Sequential(nn.Linear(2*width+1, width), nn.SiLU())
        self.readout = nn.Sequential(nn.Linear(2*width, width), nn.SiLU(), nn.Linear(width, 2))

    def forward(self, observed, social, goals, valid):
        node, edge, (b, i, j) = raw_features(observed, social, goals, valid)
        node = self.node((node-self.node_mean)/self.node_std)
        batch, worlds, agents, width = node.shape
        messages = self.edge(torch.cat((node[b, :, i], node[b, :, j], (edge-self.edge_mean)/self.edge_std), -1))
        # Dense placement makes neighbour max/mean differentiable and deterministic;
        # no atomic scatter reductions, and no invalid/singleton neighbours.
        dense = node.new_zeros(batch, worlds, agents, agents, width)
        edge_mask = torch.zeros(batch, worlds, agents, agents, dtype=torch.bool, device=node.device)
        dense[b, :, i, j] = messages
        edge_mask[b, :, i, j] = True
        pooled = masked_pool(dense, edge_mask, 3)
        agent = self.agent(torch.cat((node, pooled), -1))
        scene = masked_pool(agent, valid[:, None].expand(-1, worlds, -1), 2)
        count = valid.sum(1).float().log1p()[:, None, None].expand(-1, worlds, -1)
        world = self.world(torch.cat((scene, count), -1))
        return self.readout(torch.cat((world.mean(1), world.amax(1)), -1))

    def difference(self, observed, social, initial_goals, feedback_goals, valid):
        return self(observed, social, initial_goals, valid)-self(observed, social, feedback_goals, valid)


def choose_initial(delta, initial_pi, feedback_pi, valid):
    same = (initial_pi == feedback_pi).all(1)
    same = (same | ~valid).all(1)
    return (delta[:, 0] <= 0) & (delta[:, 1] < 0) & (valid.sum(1) > 1) & ~same

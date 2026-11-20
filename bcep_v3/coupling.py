"""One additional world-conditioned residual; exactly two old Hungarian solves."""
import torch
from torch import nn
from torch.nn import functional as F
from ebjd.unified import gather_candidates
from bcep_v2.joint import FeedbackCoupling, assignment, neighbors
from .evidence import CandidateSetEvidence, validate_inputs
from .geometry import reference_paths, pair_features


class RelationalAssignment(nn.Module):
    def __init__(self, dim=192, heads=4, relation_dim=16, edge_chunk=64):
        super().__init__()
        self.heads, self.relation_dim, self.edge_chunk = heads, relation_dim, edge_chunk
        self.pair_encoder = nn.Sequential(nn.Linear(12, 32), nn.SiLU(), nn.Linear(32, relation_dim))
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.bias = nn.Linear(relation_dim, heads)
        self.value = nn.Linear(relation_dim, dim)
        self.social_out = nn.Linear(dim, dim)
        self.delta = nn.Sequential(nn.Linear(2 * dim + relation_dim, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.delta[-1].weight); nn.init.zeros_(self.delta[-1].bias)

    def relations(self, context, goals, initial, scale, edge_indices=None):
        bsz, n, k, _ = initial.shape
        b, i, j = neighbors(context.observed, context.valid) if edge_indices is None else edge_indices
        paths = reference_paths(context, goals)
        world_rel, query_rel = [], []
        for start in range(0, b.numel(), self.edge_chunk):
            sl = slice(start, start + self.edge_chunk)
            be, ie, je = b[sl], i[sl], j[sl]
            raw = pair_features(paths, context.observed, be, ie, je, scale.float().clamp_min(0.5))
            pair = self.pair_encoder(raw)  # E,A,C,R; geometry evaluated BEFORE ST mixing
            conditioned = torch.einsum('ewc,eacr->ewar', initial[be, je].float(), pair)
            query_rel.append(conditioned)
            world_rel.append(torch.einsum('ewa,ewar->ewr', initial[be, ie].float(), conditioned))
        dense = goals.new_zeros(bsz * n * n, k, self.relation_dim, dtype=torch.float32)
        query = goals.new_zeros(bsz * n, k, k, self.relation_dim, dtype=torch.float32)
        legal = torch.zeros(bsz * n * n, dtype=torch.bool, device=goals.device)
        if world_rel:
            # Each directed edge key occurs exactly once. Deterministic index_copy,
            # never floating atomic index_add. Receiver reduction uses dense matmul.
            key = b * n * n + i * n + j
            dense = dense.index_copy(0, key, torch.cat(world_rel))
            legal = legal.index_fill(0, key, True)
            incidence = F.one_hot(b * n + i, bsz * n).T.float()
            q = torch.cat(query_rel)
            query = (incidence @ q.flatten(1)).reshape_as(query)
            query = query / incidence.sum(-1).clamp_min(1)[:, None, None, None]
        return dense.reshape(bsz, n, n, k, -1).permute(0, 3, 1, 2, 4), query.reshape(bsz, n, k, k, -1), legal.reshape(bsz, n, n)

    def forward(self, u, context, goals, initial, scale, edge_indices=None):
        b, n, k, d = u.shape
        rel, query_rel, legal = self.relations(context, goals, initial, scale, edge_indices)
        v = torch.einsum('bnwa,bnad->bwnd', initial.float(), u)
        q, key, value = self.qkv(self.norm(v)).reshape(b, k, n, 3, self.heads, d // self.heads).unbind(3)
        q, key, value = [t.transpose(2, 3) for t in (q, key, value)]
        logits = q @ key.transpose(-1, -2) / (d // self.heads)**0.5
        logits = logits + self.bias(rel).permute(0, 1, 4, 2, 3)
        # No-neighbour rows get a safe zero output, including singleton/padding.
        mask = legal[:, None, None]
        weight = logits.masked_fill(~mask, -1e4).softmax(-1) * mask
        weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-12)
        rv = self.value(rel).reshape(b, k, n, n, self.heads, d // self.heads).permute(0, 1, 4, 2, 3, 5)
        social = weight @ value + (weight[..., None] * rv).sum(-2)
        social = social.transpose(2, 3).reshape(b, k, n, d)
        v_rel = v + self.social_out(social) * legal.any(-1)[:, None, :, None]
        world = v_rel.permute(0, 2, 1, 3)[:, :, :, None].expand(b, n, k, k, d)
        candidate = u[:, :, None].expand(b, n, k, k, d)
        return self.delta(torch.cat((candidate, world, query_rel), -1)).squeeze(-1) * context.valid[..., None, None]


class HCRACoupling(FeedbackCoupling):
    def enable_v3(self):
        self.csee = CandidateSetEvidence()
        self.hcra = RelationalAssignment()

    def assign(self, context, proposals, goal_scale, *, straight_through, return_details=False, optional_evidence=None):
        validate_inputs(context, proposals)
        valid = context.valid
        proposals = proposals.detach()
        safe = torch.where(valid[:, None, :, None], proposals, 0.0)
        scores0, old = self.forward_features(context, safe, goal_scale)
        pi0, initial, soft0 = assignment(scores0, valid, straight_through)
        edges = neighbors(context.observed, valid)
        old_final = scores0 + self.messages(old, context.observed, safe, valid, initial, edge_indices=edges)
        u = self.csee(old, context, safe, goal_scale, optional_evidence)
        delta = self.hcra(u, context, safe, initial, goal_scale, edge_indices=edges)
        scores = old_final + delta
        indices, _, soft = assignment(scores, valid, straight_through)
        hard = gather_candidates(proposals, indices)
        if straight_through:
            surrogate = torch.einsum('bnwa,bnad->bwnd', soft.float(), safe.transpose(1, 2).float())
            hard = hard + (surrogate - surrogate.detach()).to(hard)
        result = (hard, indices, scores)
        if return_details:
            return (*result, dict(initial_pi=pi0, initial_st=initial, initial_soft=soft0,
                                  old_final=old_final, score_delta=delta, features=u))
        return result

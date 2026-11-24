"""CUDA launch amortization for independent native CandidateHead evaluations.

Every slot uses exactly the original per-set tensor shapes and operation order.
The observed-only motion frame is evaluated with the original function once per
window, then copied unchanged into every native set evaluation. There is no
embedding cache, cross-set GEMM batching, algebraic factorization,
mixed precision, compilation or approximate acceptance screen. The original
forward's Python ``not valid.any()`` check is performed before capture/update;
its remaining tensor body is copied below because scalar reads are not capturable.
Final accepted endpoints are also replayed through the original forward.
"""
import torch

from bcep.motion_frame import frame, to_local
from bcep_v2.provider import GEOMETRY, query_map


def _native_tensor_body(head, observed, maps, valid, goals, context, features, motion_frame):
    # bcep_v2.provider.CandidateHead.forward, after the validated input guard.
    goals = goals.detach().transpose(0, 1)
    safe = torch.where(valid[:, None, None], goals, 0.0)
    origin, rotation, scale = motion_frame
    local = to_local(safe.float(), origin[:, None], rotation[:, None], scale)
    if maps is None:
        sampled = head.missing.expand(len(valid), 20, -1)
        inside = torch.zeros_like(local[..., 0], dtype=torch.bool)
    else:
        sampled, inside = query_map(features, maps, safe, observed[:, -1], dict(layout=GEOMETRY))
    embedding = head.embedding(
        torch.cat(
            (
                context[:, None].expand(-1, 20, -1),
                torch.where(inside[..., None], sampled, head.missing),
                local,
                inside[..., None].float(),
            ),
            -1,
        )
    )
    raw = head.readout(
        torch.cat((embedding, embedding.mean(1, keepdim=True).expand_as(embedding)), -1)
    )
    shift = 0.5 * raw / (1 + raw.norm(dim=-1, keepdim=True))
    world_shift = (rotation[:, None] @ shift.unsqueeze(-1)).squeeze(-1)
    world_shift = torch.where(valid[:, None, None], world_shift, 0.0)
    return (safe + world_shift.to(safe)).transpose(0, 1), world_shift


class NativeHeadGraph:
    """One reusable native head graph for a fixed N, evaluated <=chunk sets/time.

    ``update`` copies a new window's observations/maps/context into fixed inputs.
    ``evaluate(raw, ids_batch)`` accepts [C,20], 1<=C<=chunk, returning [C,20,N,2].
    A short last chunk pads by repeating its final set; padding is never selected.
    ``native(raw, ids)`` uses the unmodified original forward, for exact replay.
    CPU execution falls back to separate original native calls.
    """

    @torch.no_grad()
    def __init__(self, provider, batch, context, features, chunk=16):
        assert isinstance(chunk, int) and chunk > 0
        assert not provider.training and not provider.head.training
        self.provider = provider
        self.chunk = chunk
        self.device = batch.observed.device
        self.n = batch.observed.shape[1]
        self.observed = batch.observed[0].clone()
        self.maps = None if batch.semantic_maps is None else batch.semantic_maps[0].clone()
        self.valid = batch.valid[0].clone()
        self.context = context.clone()
        self.features = None if features is None else features.clone()
        self.motion_frame = tuple(x.clone() for x in frame(
            torch.where(self.valid[:, None, None], self.observed, 0.0),
            12, provider.base.dt, provider.base.scales[-1],
        ))
        self.goals = self.observed.new_zeros(chunk, 20, self.n, 2)
        self.graph = None
        self.outputs = None
        self.evaluated_sets = 0
        self.native_calls = 0
        self.graph_replays = 0
        self.padded_calls = 0
        self.update(batch, context, features)
        if self.device.type == 'cuda':
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._calls()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream):
                self.outputs = torch.stack(self._calls())

    def _calls(self):
        return [
            _native_tensor_body(
                self.provider.head, self.observed, self.maps, self.valid,
                self.goals[slot], self.context, self.features, self.motion_frame,
            )[0]
            for slot in range(self.chunk)
        ]

    @torch.no_grad()
    def update(self, batch, context, features):
        assert len(batch.observed) == 1 and batch.observed.shape[1] == self.n
        assert batch.observed.dtype == torch.float32 and context.dtype == torch.float32
        assert batch.valid[0].any() and torch.isfinite(batch.observed).all()
        assert batch.observed.device == self.device
        assert (self.maps is None) == (batch.semantic_maps is None)
        assert (self.features is None) == (features is None)
        self.observed.copy_(batch.observed[0])
        self.valid.copy_(batch.valid[0])
        self.context.copy_(context)
        if self.maps is not None:
            self.maps.copy_(batch.semantic_maps[0])
        if self.features is not None:
            self.features.copy_(features)
        current_frame = frame(
            torch.where(self.valid[:, None, None], self.observed, 0.0),
            12, self.provider.base.dt, self.provider.base.scales[-1],
        )
        for target, source in zip(self.motion_frame, current_frame):
            target.copy_(source)

    def _ids(self, raw, ids, batched):
        assert raw.ndim == 3 and raw.shape[1:] == (self.n, 2)
        assert raw.dtype == torch.float32 and raw.device == self.device
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        assert ids.ndim == (2 if batched else 1) and ids.shape[-1] == 20
        assert ids.numel() > 0 and ids.min() >= 0 and ids.max() < len(raw)
        ordered = ids.sort(dim=-1).values
        assert (ordered[..., 1:] != ordered[..., :-1]).all()
        if batched:
            assert len(ids) <= self.chunk
        return ids

    @torch.no_grad()
    def evaluate(self, raw, ids_batch):
        ids = self._ids(raw, ids_batch, True)
        count = len(ids)
        if self.graph is None:
            result = torch.stack([self.native(raw, row) for row in ids])
        else:
            self.goals[:count].copy_(raw[ids])
            if count < self.chunk:
                self.goals[count:].copy_(self.goals[count-1].expand(self.chunk-count, -1, -1, -1))
            self.graph.replay()
            result = self.outputs[:count].clone()
            self.graph_replays += 1
            self.padded_calls += self.chunk - count
        self.evaluated_sets += count
        return result

    @torch.no_grad()
    def native(self, raw, ids):
        ids = self._ids(raw, ids, False)
        result, _ = self.provider.head(
            self.provider.base, self.observed, self.maps, self.valid, raw[ids],
            self.context, self.features, dict(layout=GEOMETRY),
        )
        self.native_calls += 1
        return result

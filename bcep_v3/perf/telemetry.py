"""P0: detached FP64 summaries, two host transfers per logical update."""
import json
import time

import torch

from bcep_v3.formal_telemetry import ModuleTelemetry
from bcep_v3.model import NEW_PREFIXES


class BatchedModuleTelemetry(ModuleTelemetry):
    """Same update IDs, parameter references, definitions and JSONL fields.

    FP64 cross-tensor reduction may round differently from Python's serial sum;
    this affects only reported statistics, never the objective or gradient.
    In-memory history is bounded; the complete per-step journal is still written.
    A partial trailing line is rejected, not silently joined/truncated on resume.
    A closed append is not an fsync/power-loss durability guarantee.
    """
    def __init__(self, *args, max_history_rows=64, **kwargs):
        super().__init__(*args, **kwargs)
        if max_history_rows is not None and max_history_rows < 0:
            raise ValueError('max_history_rows must be nonnegative or None')
        self.max_history_rows = max_history_rows
        self.host_transfers = 0
        self.present = {}
        if self.path and self.path.exists() and self.path.stat().st_size:
            with self.path.open('rb') as f:
                f.seek(-1, 2)
                if f.read(1) != b'\n':
                    raise ValueError('incomplete telemetry journal tail; requires explicit offline repair')

    def _host(self, values):
        self.host_transfers += 1
        return torch.stack(values).detach().cpu().tolist()

    @torch.no_grad()
    def capture_backward(self):
        self.before = {n:p.detach().clone() for n,p in self.named.items()}
        zero = next(iter(self.named.values())).new_zeros((), dtype=torch.float64)
        packed = []
        for prefix in NEW_PREFIXES:
            tensors = [p.grad for n,p in self.named.items() if n.startswith(prefix) and p.grad is not None]
            self.present[prefix] = len(tensors)
            finite = torch.stack([torch.isfinite(g).all() for g in tensors]).all() if tensors else zero == 0
            square_sum = torch.stack([g.double().square().sum() for g in tensors]).sum() if tensors else zero
            packed.extend([finite.to(torch.float64), square_sum])
        values = self._host(packed)
        self.grad = {}
        for index,prefix in enumerate(NEW_PREFIXES):
            finite,square_sum = values[index*2:index*2+2]
            if not finite:
                self.before = {}
                raise FloatingPointError(prefix+' nonfinite gradient')
            self.grad[prefix] = dict(gradient_l2=square_sum**.5, present_tensors=self.present[prefix])
        return self.grad

    @torch.no_grad()
    def step_done(self, *args):
        self.update += 1
        row = dict(update=self.update, domain=self.domain, unix=time.time(), modules={})
        packed = []
        for prefix in NEW_PREFIXES:
            values = [(p.detach()-self.before[n]).double() for n,p in self.named.items() if n.startswith(prefix)]
            packed.extend([torch.stack([x.square().sum() for x in values]).sum(),
                           torch.stack([x.abs().max() for x in values]).max()])
        values = self._host(packed)
        for index,prefix in enumerate(NEW_PREFIXES):
            square_sum,maximum = values[index*2:index*2+2]
            row['modules'][prefix] = dict(self.grad[prefix], parameter_delta_l2=square_sum**.5,
                                         parameter_delta_max=maximum)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a') as f:
                f.write(json.dumps(row)+'\n')
        self.rows.append(row)
        if self.max_history_rows is not None:
            del self.rows[:max(0, len(self.rows)-self.max_history_rows)]
        self.before = {}

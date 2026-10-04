#!/usr/bin/env python3
"""CPU observer semantics and real saved-input authentication; zero encoder forwards."""
import gc
import json
from pathlib import Path
import sys
import weakref
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tools.jdv2_social_local import ModuleObserver, authenticate
from tools.jdv2_restart_observer import Recorder, state_fingerprint, compare_tensor


class TemporalStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.count = 0
        self.fail = False
    def forward(self, x):
        self.count += 1
        if self.fail:
            raise ValueError('original exception')
        unused = x.clone()
        self.unused = weakref.ref(unused)
        self.final = x.unsqueeze(0)
        return unused, self.final


class MessageStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.count = 0
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.register_buffer('buffer', torch.ones(1))
    def forward(self, x, edges, features, weights):
        self.count += 1
        self.original_return = x
        return x


class FakeEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal_encoder = TemporalStub()
        self.message_layers = torch.nn.ModuleList([MessageStub()])
    def forward(self, x, edges, features, weights):
        _, h = self.temporal_encoder(x)
        return self.message_layers[0](h[-1], edges, features, weights)


def checks():
    results = []
    model = FakeEncoder().eval()
    x = torch.ones(2, 3)
    args = (x, torch.empty(2, 0, dtype=torch.long), torch.empty(0, 14), torch.empty(0))
    before = state_fingerprint(model)
    rng = torch.get_rng_state().clone()
    with ModuleObserver(model) as observer:
        result = model(*args)
        assert result is model.message_layers[0].original_return
        assert observer.recorder.tensors['temporal_encoder/final_hidden'] is model.temporal_encoder.final
    gc.collect()
    assert model.temporal_encoder.unused() is None
    results.append('original return objects preserved; unused GRU sequence not retained')
    assert model.temporal_encoder.count == model.message_layers[0].count == 1
    assert x.equal(torch.ones_like(x)) and state_fingerprint(model) == before
    assert rng.equal(torch.get_rng_state())
    results.append('no extra calls/draws or input/parameter/buffer mutation')
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    _, meta = observer.recorder.finish()
    assert not meta['invalid_snapshots']
    results.append('scope removes hooks; empty graph fields and deferred serialization valid')
    model.temporal_encoder.fail = True
    try:
        with ModuleObserver(model):
            model(*args)
    except ValueError as error:
        assert str(error) == 'original exception'
    else:
        raise AssertionError('exception swallowed')
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    results.append('original exception propagates and hooks removed on failure')
    recorder = Recorder()
    recorder.take('original', x)
    recorder.take('alias', x[0])
    x.add_(1)
    _, meta = recorder.finish()
    assert set(meta['invalid_snapshots']) == {'original', 'alias'}
    results.append('later in-place update invalidates all aliased snapshots')
    assert not compare_tensor(torch.tensor([1.]), torch.tensor([1.], dtype=torch.bfloat16))['bitwise_equal']
    assert not compare_tensor(torch.tensor([0.]), torch.tensor([-0.]))['bitwise_equal']
    results.append('raw-bit comparator distinguishes dtype and signed zero')
    inputs, encoder, source = authenticate()
    assert len(inputs) == 5 and len(encoder.state_dict()) == 16
    results.append('real P2/P3 input hashes/layouts authenticated; strict extracted16-key encoder matches both state fingerprints')
    return {'status': 'PASS', 'checks': results, 'count': len(results), 'device': 'CPU_ONLY',
            'real_encoder_forward_calls': 0, 'canonical_GPU_neutrality': 'NOT_CERTIFIED_BY_CPU_TESTS',
            'input_authentication': source}


if __name__ == '__main__':
    print(json.dumps(checks(), indent=2))

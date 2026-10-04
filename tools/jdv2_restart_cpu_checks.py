#!/usr/bin/env python3
"""Small CPU-only observer tests; no historical permutation audit or model run."""
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tools.jdv2_restart_observer import BoundaryObserver, Recorder, compare_tensor, state_fingerprint
from src.models.joint_dependency_v2.joint_sampler import weighted_gumbel_top_p
from src.models.joint_dependency_v2.exact_lexicographic_assignment import solve_exact_persistent_tie


def identity(x):
    return x


def check():
    checks = []
    x = torch.tensor([1., -0., 3.])
    rng = torch.get_rng_state().clone()
    with BoundaryObserver(enabled=False, extra_functions=[identity]) as disabled:
        assert identity(x) is x
    assert not disabled.recorder.tensors
    checks.append('disabled returns original object, no trace')
    with BoundaryObserver(enabled=True, extra_functions=[identity]) as enabled:
        assert identity(x) is x
    data, meta = enabled.recorder.finish()
    assert all(t.equal(x) for t in data.values()) and not meta['invalid_snapshots']
    assert torch.equal(rng, torch.get_rng_state())
    checks.append('enabled preserves return identity, values and CPU RNG')
    r = Recorder()
    r.take('alias_a', x)
    r.take('alias_b', x[1:])
    r.take('empty', torch.empty(0, 2))
    r.take('bf16', torch.tensor([1., -0., 0.125], dtype=torch.bfloat16))
    _, m = r.finish()
    assert m['tensors']['alias_a']['alias_group'] == m['tensors']['alias_b']['alias_group']
    assert m['native_dtype_roundtrip'] and not m['invalid_snapshots']
    checks.append('empty, alias, signed zero and native BF16 serialization')
    x.add_(1)
    _, m = r.finish()
    assert set(m['invalid_snapshots']) == {'alias_a', 'alias_b'}
    checks.append('later in-place update invalidates both alias snapshots')
    assert not compare_tensor(torch.tensor([1.]), torch.tensor([1.], dtype=torch.bfloat16))['bitwise_equal']
    assert not compare_tensor(torch.tensor([0.]), torch.tensor([-0.]))['bitwise_equal']
    checks.append('dtype mismatch and signed-zero bits cannot masquerade as bitwise equal')
    module = torch.nn.BatchNorm1d(3).eval()
    with torch.no_grad():
        model_before = state_fingerprint(module)
        inp = torch.ones(2, 3)
        expected = module(inp)
        with BoundaryObserver(enabled=True, extra_functions=[module.forward]):
            actual = module(inp)
        assert expected.equal(actual) and inp.equal(torch.ones_like(inp))
        assert state_fingerprint(module) == model_before
    checks.append('module inputs, parameters, buffers and output unchanged')
    score = torch.tensor([[[.5, .2, .1]]]).expand(1, 2, 3)
    mask = torch.tensor([[[True, True, False]]]).expand_as(score)
    results = []
    generator_states = []
    for traced in (False, True):
        generator = torch.Generator().manual_seed(2036)
        before = torch.get_rng_state().clone()
        with BoundaryObserver(enabled=traced) as observer:
            chosen = weighted_gumbel_top_p(score, mask, 1., generator)
            solved = solve_exact_persistent_tie(score[0], mask[0], chosen[0], torch.tensor([[0., 0.], [1., 1.], [2., 2.]]),
                evaluation_seed=2036, window_index=13, agent_index=0)
        assert torch.equal(before, torch.get_rng_state())
        results.append((chosen, solved))
        generator_states.append(generator.get_state())
        if traced:
            d, m = observer.recorder.finish()
            assert not m['invalid_snapshots']
            assert any('gumbel_noise' in k for k in d)
            assert any('encoded_objective' in k for k in m['values'])
    assert results[0][0].equal(results[1][0]) and results[0][1] == results[1][1]
    assert generator_states[0].equal(generator_states[1])
    checks.append('actual masked Gumbel and exact solver, same explicit/global RNG, assignment and integer objective')
    torch.manual_seed(83)
    before = torch.get_rng_state().clone()
    plain = (torch.randn(2, 3), torch.randn_like(torch.ones(2, 3)))
    after = torch.get_rng_state().clone()
    torch.set_rng_state(before)
    with BoundaryObserver(enabled=True) as observed:
        traced = (torch.randn(2, 3), torch.randn_like(torch.ones(2, 3)))
    assert all(a.equal(b) for a, b in zip(plain, traced))
    assert torch.get_rng_state().equal(after)
    assert len(observed.recorder.tensors) == 2
    checks.append('actual randn/randn_like passthrough, unchanged draws and RNG post-state')
    return {'status': 'PASS', 'device': 'CPU_ONLY', 'checks': checks, 'count': len(checks),
            'GPU_neutrality': 'NOT_TESTED_BY_CPU_CHECKS'}


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))

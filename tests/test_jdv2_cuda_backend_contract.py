"""Backend metadata/gates on CPU; does not initialize CUDA."""
import copy
import subprocess
from types import SimpleNamespace
import pytest
import torch
from mc_wiring_fixture import args
from src.jdv2_objective_state import (ObjectiveRNG, semantic_fingerprint, validate_mode,
                                     CUDA_BACKEND, validate_request)


def cuda_args(tmp_path):
    a = args(tmp_path)
    a.device = 'cuda:0'; a.jdv2_mc_backend = CUDA_BACKEND
    return a


def test_cpu_v1_fingerprint_and_schema_unchanged(tmp_path):
    old = subprocess.check_output(['git', 'show', 'c9dbfec:src/jdv2_objective_state.py'], text=True)
    ns = {'__file__': __file__}; exec(compile(old, 'old_state', 'exec'), ns)
    a = args(tmp_path)
    assert semantic_fingerprint(a) == ns['semantic_fingerprint'](a)
    assert ObjectiveRNG(a).snapshot(9, 'resume_last').keys() == ns['ObjectiveRNG'](a).snapshot(9, 'resume_last').keys()


def test_backend_metadata_matrix_without_cuda(tmp_path):
    cpu = ObjectiveRNG(args(tmp_path)); cuda = ObjectiveRNG(cuda_args(tmp_path))
    cpu_state, state = cpu.snapshot(9, 'resume_last'), cuda.snapshot(9, 'resume_last')
    assert state['generator_state'].device.type == 'cpu'
    assert state['format_version'] == 2
    assert state['namespace'] == cpu_state['namespace']
    assert torch.equal(state['generator_state'], cpu_state['generator_state'])
    with pytest.raises(ValueError): cuda.restore(cpu_state, 9)
    with pytest.raises(ValueError): cpu.restore(state, 9)
    cuda.restore(state, 9)
    for key in ('loss_compute_backend', 'loss_effective_dtype', 'amp_policy'):
        bad = copy.deepcopy(state); bad[key] = 'bad'
        with pytest.raises(ValueError): cuda.restore(bad, 9)
    bf = cuda_args(tmp_path); bf.amp_enabled = True; bf.amp_dtype = 'bf16'
    with pytest.raises(ValueError): ObjectiveRNG(bf).restore(state, 9)


@pytest.mark.parametrize('field,value', [('amp_dtype','fp16'),('device','cpu'),
                                       ('jdv2_mc_backend','unknown')])
def test_backend_invalid_config(tmp_path, field, value):
    a = cuda_args(tmp_path); setattr(a, field, value)
    with pytest.raises(ValueError): validate_mode(a)


def test_cuda_owner_rejects_cpu_request_before_draw(tmp_path):
    owner = ObjectiveRNG(cuda_args(tmp_path))
    before = owner.generator.get_state().clone()
    u = torch.zeros(1,2); q = torch.tensor([[.5,.5]])
    with pytest.raises(ValueError):
        owner.draw(u,q,torch.tensor([0]),torch.empty((2,0),dtype=torch.long),torch.ones_like(q,dtype=torch.bool))
    assert owner.draw_calls == 0 and torch.equal(before,owner.generator.get_state())

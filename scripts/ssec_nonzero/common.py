"""Fixed paths and exact comparisons. No official input constructor is used."""
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[2]
BASE = 'ece95ce6f1f59d58a8119fef7cdd79bf9d04ee86'
OUT = SOURCE / 'runs/ssec_synthetic_nonzero_20261121'
PUBLIC = SOURCE / 'docs/research/bcep_scene_energy/synthetic_nonzero_recovery_20261121'
FROZEN = SOURCE / 'docs/research/bcep_scene_energy/stage2_ready_20261121/FROZEN_CONFIG.json'
FROZEN_SHA = '3ff36b653703307be4f38457d9cb9395c264ea7647c0df339913376f38b9d842'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_sha(t):
    t = t.detach().cpu().contiguous()
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def cpu_copy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().clone()
    if isinstance(x, dict):
        return {k: cpu_copy(v) for k, v in x.items()}
    if isinstance(x, list):
        return [cpu_copy(v) for v in x]
    if isinstance(x, tuple):
        return tuple(cpu_copy(v) for v in x)
    if isinstance(x, np.ndarray):
        return x.copy()
    return x


def exact(a, b, path='state'):
    """No tolerances, rounding or skipped leaves; also compare type/dtype/shape."""
    assert type(a) is type(b), (path, type(a), type(b))
    if torch.is_tensor(a):
        assert a.dtype == b.dtype and a.shape == b.shape, path
        assert torch.equal(a, b), path
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            exact(a[k], b[k], f'{path}.{k}')
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for k, (x, y) in enumerate(zip(a, b)):
            exact(x, y, f'{path}[{k}]')
    else:
        assert a == b, (path, a, b)


def rng_state(order=None, view=None):
    result = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                  python=random.getstate(), numpy=np.random.get_state())
    if order is not None:
        result.update(order=order.get_state(), view=view.get_state())
    return result


def frozen():
    assert sha(FROZEN) == FROZEN_SHA
    c = read(FROZEN)
    for name, value in c['implementation_files'].items():
        assert sha(SOURCE / name) == value, name
    return c


def worker_guard():
    """Deny access to ALL other RSJG worktrees/assets in optimizer workers.

    Library/OS reads are allowed. Files under the RSJG tree may be read only
    from this worktree or its Python environment. Protection hashing runs in
    the parent, never feeds data to an optimizer worker.
    """
    root = SOURCE.parent.resolve()
    allowed = [SOURCE.resolve(), (root / '.conda').resolve()]
    events = {'denied_cross_workspace_accesses': 0}

    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(root) and not any(path.is_relative_to(p) for p in allowed):
            events['denied_cross_workspace_accesses'] += 1
            raise PermissionError('synthetic worker cannot access another RSJG workspace or asset')

    sys.addaudithook(audit)
    return events


def setup_cuda():
    from bcep_v2.runtime import configure_cuda
    configure_cuda()
    torch.set_num_threads(1)
    assert torch.cuda.is_available() and not torch.is_autocast_enabled()
